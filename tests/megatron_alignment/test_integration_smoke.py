"""Integration smoke test for Megatron data adapter end-to-end via get_dataset()."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.data.megatron import MegatronGPTDataset, MegatronBlendedDataset
from llamafactory.hparams import DataArguments, ModelArguments


MEGATRON_PATH = os.path.join(os.path.dirname(__file__), "../../data/c4_demo_text_document")
CACHE_PATH = "/tmp/lf_smoke_cache"
OUTPUT_DIR = "/tmp/lf_smoke_out"


def setup_module():
    """Clean up cache before tests."""
    if os.path.isdir(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)


def teardown_module():
    """Clean up cache after tests."""
    if os.path.isdir(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory containing a dataset_info.json with a megatron entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_info = {
            "c4_demo_megatron": {
                "load_from": "megatron",
                "megatron_path": MEGATRON_PATH,
                "megatron_seq_length": 128,
                "megatron_shuffle_seed": 42,
                "megatron_data_cache_path": CACHE_PATH,
            }
        }
        info_path = Path(tmpdir) / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(dataset_info, f)
        yield tmpdir


@pytest.fixture
def tokenizer():
    """Load a small tokenizer for the test."""
    return AutoTokenizer.from_pretrained("gpt2")


def test_get_dataset_with_megatron(temp_dataset_dir, tokenizer):
    """Verify get_dataset() works end-to-end with a Megatron dataset entry."""
    # 1. Build minimal args
    data_args = DataArguments(
        dataset="c4_demo_megatron",
        dataset_dir=temp_dataset_dir,
        cutoff_len=128,
        megatron_data_cache_path=CACHE_PATH,
    )
    model_args = ModelArguments(model_name_or_path="gpt2")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=42,
    )

    # 2. Get template
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # 3. Call get_dataset
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="pt",
        tokenizer=tokenizer,
    )

    # 4. Assertions
    train_dataset = dataset_module.get("train_dataset")
    assert train_dataset is not None, "train_dataset is missing from dataset_module"
    assert isinstance(
        train_dataset, (MegatronGPTDataset, MegatronBlendedDataset)
    ), f"Expected MegatronGPTDataset or MegatronBlendedDataset, got {type(train_dataset)}"

    # data_args.packing should be forced to a falsy value for Megatron
    assert not data_args.packing, f"Expected packing to be falsy, got {data_args.packing}"

    # disable_shuffling flag should be set for Megatron datasets
    assert dataset_module.get("disable_shuffling") is True, (
        f"Expected disable_shuffling=True in dataset_module, got {dataset_module.get('disable_shuffling')}"
    )

    # 5. Iterate at least 2 batches via DataLoader
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        assert "input_ids" in batch, f"Batch missing 'input_ids'. Keys: {batch.keys()}"
        assert "labels" in batch, f"Batch missing 'labels'. Keys: {batch.keys()}"
        assert "attention_mask" in batch, f"Batch missing 'attention_mask'. Keys: {batch.keys()}"

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        assert input_ids.shape == torch.Size([4, 128]), (
            f"input_ids shape mismatch: expected [4, 128], got {list(input_ids.shape)}"
        )
        assert labels.shape == torch.Size([4, 128]), (
            f"labels shape mismatch: expected [4, 128], got {list(labels.shape)}"
        )

        if batch_count >= 2:
            break

    assert batch_count >= 2, f"Expected to iterate at least 2 batches, got {batch_count}"
