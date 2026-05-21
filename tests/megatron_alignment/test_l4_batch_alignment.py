"""Alignment test: MegatronGPTDataset vs Megatron-LM GPTDataset (L4 batch-level)."""

import os
import shutil
import sys
from datetime import datetime
from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from llamafactory.data.megatron.gpt_dataset import MegatronGPTDataset, MegatronGPTDatasetConfig
from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset
from utils import (
    build_megatron_stubs,
    check_megatron_source,
    find_helpers_cpp,
    load_megatron_module,
)

# ---------------------------------------------------------------------------
# Load Megatron-LM reference code directly (no full package install needed)
# ---------------------------------------------------------------------------

_megatron_dir = check_megatron_source(require_helpers_cpp=True)
build_megatron_stubs(include_tokenizer=True)

load_megatron_module(
    "megatron.core.datasets.utils",
    os.path.join(_megatron_dir, "megatron/core/datasets/utils.py"),
)
load_megatron_module(
    "megatron.core.datasets.blended_megatron_dataset_config",
    os.path.join(_megatron_dir, "megatron/core/datasets/blended_megatron_dataset_config.py"),
)
load_megatron_module(
    "megatron.core.datasets.indexed_dataset",
    os.path.join(_megatron_dir, "megatron/core/datasets/indexed_dataset.py"),
)
load_megatron_module(
    "megatron.core.datasets.helpers_cpp",
    find_helpers_cpp(_megatron_dir),
)
load_megatron_module(
    "megatron.core.datasets.helpers",
    os.path.join(_megatron_dir, "megatron/core/datasets/helpers.py"),
)
load_megatron_module(
    "megatron.core.datasets.megatron_dataset",
    os.path.join(_megatron_dir, "megatron/core/datasets/megatron_dataset.py"),
)
_megatron_mod = load_megatron_module(
    "megatron.core.datasets.gpt_dataset",
    os.path.join(_megatron_dir, "megatron/core/datasets/gpt_dataset.py"),
)
_GPTDataset = _megatron_mod.GPTDataset
_GPTDatasetConfig = _megatron_mod.GPTDatasetConfig
_Split = sys.modules["megatron.core.datasets.utils"].Split
_FakeTokenizer = sys.modules["megatron.core.tokenizers"].MegatronTokenizerBase

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
TEST_PREFIX = os.path.join(os.path.dirname(__file__), "../../data/c4_demo_text_document")
CACHE_PATH = "/tmp/lf_test_cache_l4"
SEQ_LENGTH = 128
SEED = 42
NUM_SAMPLES = 500
BATCH_SIZE = 4
NUM_BATCHES = 50
REPORT_PATH = "/tmp/lf_megatron_l4_report.md"


def _cleanup_cache():
    if os.path.isdir(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ours() -> MegatronGPTDataset:
    _cleanup_cache()
    os.makedirs(CACHE_PATH, exist_ok=True)

    indexed_ds = MegatronIndexedDataset(TEST_PREFIX, multimodal=False, mmap=True)
    config = MegatronGPTDatasetConfig(
        path_prefix=TEST_PREFIX,
        seq_length=SEQ_LENGTH,
        seed=SEED,
        num_samples=NUM_SAMPLES,
        data_cache_path=CACHE_PATH,
        add_extra_token=True,
        drop_last_partial_sequence=True,
        split="train",
        pad_token_id=-1,
    )
    dataset = MegatronGPTDataset(config, indexed_ds)
    yield dataset
    _cleanup_cache()


@pytest.fixture(scope="module")
def ref():
    _cleanup_cache()
    os.makedirs(CACHE_PATH, exist_ok=True)

    indexed_ds = sys.modules["megatron.core.datasets.indexed_dataset"].IndexedDataset(
        TEST_PREFIX, multimodal=False, mmap=True
    )
    config = _GPTDatasetConfig(
        random_seed=SEED,
        sequence_length=SEQ_LENGTH,
        blend=([TEST_PREFIX], None),
        split="1,0,0",
        path_to_cache=CACHE_PATH,
        tokenizer=_FakeTokenizer(),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=True,
        add_extra_token_to_sequence=True,
        drop_last_partial_validation_sequence=True,
    )

    indices = np.arange(len(indexed_ds), dtype=np.int32)
    dataset = _GPTDataset(
        indexed_dataset=indexed_ds,
        dataset_path=TEST_PREFIX,
        indexed_indices=indices,
        num_samples=NUM_SAMPLES,
        index_split=_Split.train,
        config=config,
    )
    yield dataset
    _cleanup_cache()


# ---------------------------------------------------------------------------
# L4: Batch-level alignment
# ---------------------------------------------------------------------------


def test_l4_batch_alignment(ours: MegatronGPTDataset, ref):
    """Compare first 100 batches from both datasets via DataLoader."""
    ours_loader = DataLoader(
        ours,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    ref_loader = DataLoader(
        ref,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    pad_id = ours.config.pad_token_id
    report_lines: List[str] = []
    report_lines.append("# L4 Batch-Level Alignment Report\n")
    report_lines.append(f"- **Date**: {datetime.now().isoformat()}\n")
    report_lines.append(f"- **Dataset**: `{TEST_PREFIX}`\n")
    report_lines.append(f"- **Config**: seq_length={SEQ_LENGTH}, seed={SEED}, num_samples={NUM_SAMPLES}\n")
    report_lines.append(f"- **Batch size**: {BATCH_SIZE}\n")
    report_lines.append(f"- **Batches compared**: {NUM_BATCHES}\n")
    report_lines.append("\n")

    all_pass = True
    mismatched_batches: List[int] = []

    for batch_idx, (ours_batch, ref_batch) in enumerate(zip(ours_loader, ref_loader)):
        if batch_idx >= NUM_BATCHES:
            break

        ours_input_ids = ours_batch["input_ids"].clone()
        ours_labels = ours_batch["labels"].clone()

        # Apply Megatron's pad-token remapping for a fair comparison
        ours_input_ids[ours_input_ids == pad_id] = 0
        ours_labels[ours_labels == pad_id] = 0

        ref_tokens = ref_batch["tokens"]

        match_input = torch.equal(ours_input_ids, ref_tokens)
        # Our labels are intentionally unshifted (same as input_ids) to match
        # HF AutoModelForCausalLM's internal shift. Megatron ref_labels are
        # shifted (text[1:]), so we compare ours to ref_tokens instead.
        match_label = torch.equal(ours_labels, ref_tokens)

        if match_input and match_label:
            report_lines.append(f"- Batch {batch_idx:3d}: **PASS**\n")
        else:
            all_pass = False
            mismatched_batches.append(batch_idx)
            report_lines.append(f"- Batch {batch_idx:3d}: **FAIL** (input_ids match={match_input}, labels match={match_label})\n")

        assert match_input, f"input_ids mismatch at batch {batch_idx}"
        assert match_label, f"labels mismatch at batch {batch_idx}"

    report_lines.append("\n")
    if all_pass:
        report_lines.append("## Result: **PASS** ✅\n")
    else:
        report_lines.append("## Result: **FAIL** ❌\n")
        report_lines.append(f"Mismatched batches: {mismatched_batches}\n")

    with open(REPORT_PATH, "w") as f:
        f.writelines(report_lines)
