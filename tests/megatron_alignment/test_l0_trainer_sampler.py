"""L0 integration test: verify Megatron dataset forces SequentialSampler in CustomTrainer."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from llamafactory.hparams import TrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments
from llamafactory.train.pt.trainer import CustomTrainer


MEGATRON_PATH = os.path.join(os.path.dirname(__file__), "../../data/c4_demo_text_document")
CACHE_PATH = "/tmp/lf_test_l0_cache"
OUTPUT_DIR = "/tmp/lf_test_l0_out"


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
            "fineweb_megatron": {
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


def test_megatron_dataset_uses_sequential_sampler(temp_dataset_dir, tokenizer):
    """Verify that a Megatron dataset disables HF Trainer's RandomSampler via disable_shuffling."""
    # 1. Build minimal args
    data_args = DataArguments(
        dataset="fineweb_megatron",
        dataset_dir=temp_dataset_dir,
        cutoff_len=128,
        megatron_data_cache_path=CACHE_PATH,
    )
    model_args = ModelArguments(model_name_or_path="gpt2")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=42,
        per_device_train_batch_size=4,
        num_train_epochs=1,
    )
    # Explicitly do NOT set disable_shuffling=True; the workflow should auto-enable it.
    finetuning_args = FinetuningArguments()
    assert finetuning_args.disable_shuffling is False, "Test precondition failed: disable_shuffling should start as False"

    # 2. Get dataset
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="pt",
        tokenizer=tokenizer,
    )

    # 3. Simulate workflow behavior: pop the flag and set it on finetuning_args
    if dataset_module.pop("disable_shuffling", False):
        finetuning_args.disable_shuffling = True

    assert finetuning_args.disable_shuffling is True, (
        "Expected finetuning_args.disable_shuffling to be auto-enabled for Megatron dataset"
    )

    # 4. Build trainer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        processor=None,
        **dataset_module,
    )

    # 5. Verify sampler is SequentialSampler
    train_dataloader = trainer.get_train_dataloader()
    assert isinstance(train_dataloader.sampler, torch.utils.data.SequentialSampler), (
        f"Expected SequentialSampler for Megatron dataset, got {type(train_dataloader.sampler)}"
    )
