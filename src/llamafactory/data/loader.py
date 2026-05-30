# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .megatron import (
    MegatronBlendedDataset,
    MegatronGPTDataset,
    MegatronGPTDatasetConfig,
    MegatronIndexedDataset,
    parse_blend_list,
)
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: Optional["PreTrainedTokenizer"] = None,
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    elif dataset_attr.load_from == "megatron":
        return _load_megatron_single_dataset(dataset_attr, data_args, training_args, tokenizer)
    elif dataset_attr.load_from == "megatron_list":
        return _load_megatron_list_dataset(dataset_attr, data_args, training_args, tokenizer)
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.14.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _resolve_bool(dataset_val: "Optional[bool]", global_val: bool) -> bool:
    r"""Resolve a boolean config value with dataset-level override.

    If ``dataset_attr`` explicitly sets the field (not ``None``), use it.
    Otherwise fall back to ``data_args`` global value.
    """
    return dataset_val if dataset_val is not None else global_val


def _load_megatron_single_dataset(
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: Optional["PreTrainedTokenizer"] = None,
) -> "torch.utils.data.Dataset":
    r"""Load a single Megatron GPT dataset."""
    seq_length = dataset_attr.megatron_seq_length or data_args.cutoff_len
    seed = dataset_attr.megatron_shuffle_seed or data_args.megatron_shuffle_seed or training_args.seed
    data_cache_path = dataset_attr.megatron_data_cache_path or data_args.megatron_data_cache_path
    num_samples = dataset_attr.num_samples if dataset_attr.num_samples is not None else dataset_attr.megatron_num_samples

    split_ratios = dataset_attr.megatron_split or data_args.megatron_split or "1,0,0"
    megatron_path = os.path.join(data_args.dataset_dir, dataset_attr.megatron_path)
    indexed_dataset = MegatronIndexedDataset(megatron_path)

    pad_token_id = 0
    eod_token_id = None
    if tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        eod_token_id = tokenizer.eos_token_id

    config = MegatronGPTDatasetConfig(
        path_prefix=megatron_path,
        seq_length=seq_length,
        seed=seed,
        num_samples=num_samples,
        data_cache_path=data_cache_path,
        reuse_megatron_cache=data_args.megatron_reuse_cache,
        split=dataset_attr.split,
        split_ratios=split_ratios,
        pad_token_id=pad_token_id,
        eod_token_id=eod_token_id,
        reset_attention_mask=_resolve_bool(dataset_attr.megatron_reset_attention_mask, data_args.megatron_reset_attention_mask),
        reset_position_ids=_resolve_bool(dataset_attr.megatron_reset_position_ids, data_args.megatron_reset_position_ids),
        eod_mask_loss=_resolve_bool(dataset_attr.megatron_eod_mask_loss, data_args.megatron_eod_mask_loss),
    )
    dataset = MegatronGPTDataset(config, indexed_dataset)

    if data_args.max_samples is not None:
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(max_samples))

    return dataset


def _load_megatron_list_dataset(
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: Optional["PreTrainedTokenizer"] = None,
) -> "torch.utils.data.Dataset":
    r"""Load a Megatron blended dataset from a .list file."""
    list_path = os.path.join(data_args.dataset_dir, dataset_attr.megatron_list_path)
    prefixes, weights = parse_blend_list(list_path)
    if not prefixes:
        raise ValueError(
            f"No valid dataset prefixes found in {dataset_attr.megatron_list_path}. "
            "Ensure the .list file is not empty and does not contain only comments."
        )

    split_ratios = dataset_attr.megatron_split or data_args.megatron_split or "1,0,0"

    pad_token_id = 0
    eod_token_id = None
    if tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        eod_token_id = tokenizer.eos_token_id

    reset_attention_mask = _resolve_bool(dataset_attr.megatron_reset_attention_mask, data_args.megatron_reset_attention_mask)
    reset_position_ids = _resolve_bool(dataset_attr.megatron_reset_position_ids, data_args.megatron_reset_position_ids)
    eod_mask_loss = _resolve_bool(dataset_attr.megatron_eod_mask_loss, data_args.megatron_eod_mask_loss)

    datasets = []
    for prefix in prefixes:
        indexed_dataset = MegatronIndexedDataset(prefix)
        config = MegatronGPTDatasetConfig(
            path_prefix=prefix,
            seq_length=dataset_attr.megatron_seq_length or data_args.cutoff_len,
            seed=dataset_attr.megatron_shuffle_seed or data_args.megatron_shuffle_seed or training_args.seed,
            num_samples=dataset_attr.num_samples if dataset_attr.num_samples is not None else dataset_attr.megatron_num_samples,
            data_cache_path=dataset_attr.megatron_data_cache_path or data_args.megatron_data_cache_path,
            reuse_megatron_cache=data_args.megatron_reuse_cache,
            split=dataset_attr.split,
            split_ratios=split_ratios,
            pad_token_id=pad_token_id,
            eod_token_id=eod_token_id,
            reset_attention_mask=reset_attention_mask,
            reset_position_ids=reset_position_ids,
            eod_mask_loss=eod_mask_loss,
        )
        datasets.append(MegatronGPTDataset(config, indexed_dataset))

    if weights is None:
        # Exhaustive blending: consume all samples from each dataset
        weights = [len(ds) for ds in datasets]
        size = None
    else:
        size = dataset_attr.num_samples if dataset_attr.num_samples is not None else dataset_attr.megatron_num_samples
        if size is None:
            # Compute maximum safe size to avoid oversampling any dataset
            weights_array = np.array(weights, dtype=np.float64)
            weights_array = weights_array / weights_array.sum()
            size = min(int(len(datasets[i]) / weights_array[i]) for i in range(len(datasets)))

    dataset = MegatronBlendedDataset(
        datasets=datasets,
        weights=weights,
        size=size,
        data_cache_path=dataset_attr.megatron_data_cache_path or data_args.megatron_data_cache_path,
        seed=dataset_attr.megatron_shuffle_seed or data_args.megatron_shuffle_seed or training_args.seed,
        split=dataset_attr.split,
    )

    if data_args.max_samples is not None:
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(max_samples))

    return dataset


def _get_merged_dataset(
    dataset_names: list[str] | None,
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
    dataset_attrs: list["DatasetAttr"] | None = None,
    tokenizer: Optional["PreTrainedTokenizer"] = None,
) -> Union["Dataset", "IterableDataset", dict[str, "Dataset"]] | None:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    if dataset_attrs is None:
        dataset_attrs = get_dataset_list(dataset_names, data_args.dataset_dir)

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, dataset_attrs):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args, tokenizer)

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Union["Dataset", "IterableDataset"] | None,
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Union["Dataset", "IterableDataset"] | None:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        train_dataset_attrs = get_dataset_list(data_args.dataset, data_args.dataset_dir) if data_args.dataset else []
        eval_dataset_attrs = get_dataset_list(data_args.eval_dataset, data_args.dataset_dir) if data_args.eval_dataset else []

        is_megatron = any(
            attr.load_from in ("megatron", "megatron_list") for attr in train_dataset_attrs + eval_dataset_attrs
        )

        dataset = _get_merged_dataset(
            data_args.dataset, model_args, data_args, training_args, stage, dataset_attrs=train_dataset_attrs, tokenizer=tokenizer
        )
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
            dataset_attrs=eval_dataset_attrs,
            tokenizer=tokenizer,
        )

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        # move front to make sure eval_dataset(if contain or split) can preprocessed appropriately
        train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)

        if is_megatron:
            # Megatron data is already tokenized; skip preprocessing entirely
            if data_args.packing:
                warnings.warn(
                    "packing is force-disabled for Megatron datasets because sample_index already handles packing semantics.",
                    UserWarning,
                )
                data_args.packing = False

            if data_args.streaming:
                raise ValueError(
                    "streaming is incompatible with Megatron datasets (bin/idx format requires random access via mmap)."
                )

            if data_args.neat_packing:
                warnings.warn(
                    "neat_packing is not supported for Megatron datasets and will be ignored.",
                    UserWarning,
                )
                data_args.neat_packing = False

            dataset_dict = DatasetDict({**train_dict, **eval_dict})
        else:
            if "train" in train_dict:
                train_dict["train"] = _get_preprocessed_dataset(
                    train_dict["train"], data_args, training_args, stage, template, tokenizer, processor, is_eval=False
                )

            for key in eval_dict:
                eval_dict[key] = _get_preprocessed_dataset(
                    eval_dict[key], data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )

            # Combine train and eval dictionaries
            dataset_dict = DatasetDict({**train_dict, **eval_dict})

        if data_args.tokenized_path is not None and not is_megatron:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        dataset_module = get_dataset_module(dataset_dict)
        if is_megatron:
            dataset_module["disable_shuffling"] = True

        return dataset_module
