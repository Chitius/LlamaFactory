# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import math
from typing import TYPE_CHECKING, Optional

import torch
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...data.megatron import MegatronBlendedDataset, MegatronGPTDataset
from ...extras.constants import AttentionFunction
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import CustomTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def _is_megatron_module(dataset_module: dict) -> bool:
    r"""Check whether the dataset module contains a Megatron dataset."""
    train_ds = dataset_module.get("train_dataset")
    if isinstance(train_ds, (MegatronGPTDataset, MegatronBlendedDataset)):
        return True
    if hasattr(train_ds, "dataset") and isinstance(train_ds.dataset, (MegatronGPTDataset, MegatronBlendedDataset)):
        return True
    return False


logger = get_logger(__name__)


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)
    if dataset_module.pop("disable_shuffling", False):
        finetuning_args.disable_shuffling = True

    reset_attn = data_args.megatron_reset_attention_mask

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if _is_megatron_module(dataset_module):
        reset_attn = data_args.megatron_reset_attention_mask
        reset_pos = data_args.megatron_reset_position_ids
        eod_mask = data_args.megatron_eod_mask_loss

        if reset_attn or reset_pos or eod_mask:
            from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling
            # Resolve actual attention implementation: when flash_attn="auto",
            # transformers may select a different implementation than "auto".
            # Use the model's actual _attn_implementation to ensure collator
            # generates the correct mask format (4D for eager/SDPA, cu_seq_lens for FA2/FA3).
            attn_impl = str(model_args.flash_attn)
            if attn_impl == "auto":
                actual_impl = getattr(model.config, "_attn_implementation", None)
                if actual_impl is not None:
                    attn_impl = actual_impl
                else:
                    attn_impl = "eager"

            # Defensive: if transformers introduces a new attn implementation that
            # we do not yet handle, fall back to eager with a loud warning.
            _KNOWN_ATTN_IMPLS = ("eager", "sdpa", "flash_attention_2", "fa2", "fa3", "disabled")
            if attn_impl not in _KNOWN_ATTN_IMPLS:
                logger.warning_rank0(
                    "Unknown attention implementation '%s' for document-boundary mask. "
                    "Falling back to 'eager'. Supported values: %s" % (attn_impl, _KNOWN_ATTN_IMPLS)
                )
                attn_impl = "eager"

            data_collator = MegatronDataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                block_diag_attn=reset_attn,
                attn_implementation=attn_impl,
                compute_dtype=model_args.compute_dtype or torch.float32,
            )
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")

        if isinstance(dataset_module.get("eval_dataset"), dict):
            for key in dataset_module["eval_dataset"].keys():
                try:
                    perplexity = math.exp(metrics[f"eval_{key}_loss"])
                except OverflowError:
                    perplexity = float("inf")

                metrics[f"eval_{key}_perplexity"] = perplexity
        else:
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            metrics["eval_perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
