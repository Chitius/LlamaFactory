"""Alignment test: MegatronGPTDataset vs Megatron-LM GPTDataset (L2 + L3)."""

import os
import shutil
import sys
from typing import Optional

import numpy as np
import pytest
import torch

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
CACHE_PATH = "/tmp/lf_test_cache_l2"
SEQ_LENGTH = 128
SEED = 42
NUM_SAMPLES = 500
NUM_RANDOM_SAMPLES_L3 = 100


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
# L2: Index construction alignment
# ---------------------------------------------------------------------------


def test_l2_document_index(ours: MegatronGPTDataset, ref):
    """Assert document_index arrays are equal."""
    assert np.array_equal(ours.document_index, ref.document_index), (
        f"document_index mismatch: shapes {ours.document_index.shape} vs {ref.document_index.shape}, "
        f"equal={np.array_equal(ours.document_index, ref.document_index)}"
    )


def test_l2_sample_index(ours: MegatronGPTDataset, ref):
    """Assert sample_index arrays are equal."""
    assert np.array_equal(ours.sample_index, ref.sample_index), (
        f"sample_index mismatch: shapes {ours.sample_index.shape} vs {ref.sample_index.shape}, "
        f"equal={np.array_equal(ours.sample_index, ref.sample_index)}"
    )


def test_l2_shuffle_index(ours: MegatronGPTDataset, ref):
    """Assert shuffle_index arrays are equal."""
    assert np.array_equal(ours.shuffle_index, ref.shuffle_index), (
        f"shuffle_index mismatch: shapes {ours.shuffle_index.shape} vs {ref.shuffle_index.shape}, "
        f"equal={np.array_equal(ours.shuffle_index, ref.shuffle_index)}"
    )


# ---------------------------------------------------------------------------
# L3: Sample output alignment
# ---------------------------------------------------------------------------


def test_l3_raw_text_random_subset(ours: MegatronGPTDataset, ref):
    """Compare raw token sequences from _get_text vs _query_document_sample_shuffle_indices."""
    rng = np.random.RandomState(42)
    total = min(len(ours), len(ref))
    indices = rng.choice(total, size=min(NUM_RANDOM_SAMPLES_L3, total), replace=False)

    for idx in indices:
        ours_text = ours._get_text(int(idx))
        ref_text, _ = ref._query_document_sample_shuffle_indices(int(idx))
        assert np.array_equal(ours_text, ref_text), f"Raw text mismatch at index {idx}"


def test_l3_input_ids_and_labels_random_subset(ours: MegatronGPTDataset, ref):
    """Compare input_ids and labels from __getitem__.

    Megatron remaps pad_token_id -> 0 in __getitem__; we apply the same remap to our outputs
    so the comparison is fair.
    """
    rng = np.random.RandomState(42)
    total = min(len(ours), len(ref))
    indices = rng.choice(total, size=min(NUM_RANDOM_SAMPLES_L3, total), replace=False)

    for idx in indices:
        ours_item = ours[int(idx)]
        ref_item = ref[int(idx)]

        ours_input_ids = ours_item["input_ids"].clone()
        ours_labels = ours_item["labels"].clone()

        # Apply Megatron's pad-token remapping for a fair comparison
        pad_id = ours.config.pad_token_id
        ours_input_ids[ours_input_ids == pad_id] = 0
        ours_labels[ours_labels == pad_id] = 0

        ref_input_ids = ref_item["tokens"]
        ref_labels = ref_item["labels"]

        assert torch.equal(ours_input_ids, ref_input_ids), (
            f"input_ids mismatch at index {idx}"
        )
        # Our labels are intentionally unshifted (same as input_ids) to match
        # HF AutoModelForCausalLM's internal shift in loss computation.
        # Megatron's ref_labels are shifted (text[1:]), so we compare ours to
        # ref_input_ids (text[:-1]) instead.
        assert torch.equal(ours_labels, ref_input_ids), (
            f"labels mismatch at index {idx}: our labels should match ref tokens (unshifted for HF)"
        )


def test_l3_length_equal(ours: MegatronGPTDataset, ref):
    assert len(ours) == len(ref)
