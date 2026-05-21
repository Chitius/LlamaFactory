"""Alignment test: MegatronIndexedDataset vs Megatron-LM IndexedDataset."""

import os
from typing import Tuple

import numpy as np
import pytest

from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset
from utils import build_megatron_stubs, check_megatron_source, load_megatron_module

# ---------------------------------------------------------------------------
# Load Megatron-LM reference code directly (no full package install needed)
# ---------------------------------------------------------------------------

_megatron_dir = check_megatron_source(require_helpers_cpp=False)
build_megatron_stubs(include_tokenizer=False)
_megatron_mod = load_megatron_module(
    "megatron.core.datasets.indexed_dataset",
    os.path.join(_megatron_dir, "megatron/core/datasets/indexed_dataset.py"),
)
MegatronReferenceDataset = _megatron_mod.IndexedDataset

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

TEST_PREFIX = os.path.join(os.path.dirname(__file__), "../../data/c4_demo_text_document")
NUM_RANDOM_SAMPLES = 100


@pytest.fixture(scope="module")
def ours() -> MegatronIndexedDataset:
    return MegatronIndexedDataset(TEST_PREFIX, multimodal=False, mmap=True)


@pytest.fixture(scope="module")
def ref() -> "MegatronReferenceDataset":
    return MegatronReferenceDataset(TEST_PREFIX, multimodal=False, mmap=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sequence_lengths_equal(ours: MegatronIndexedDataset, ref):
    """c. Assert sequence_lengths are array_equal."""
    assert np.array_equal(ours.sequence_lengths, ref.sequence_lengths)


def test_document_indices_equal(ours: MegatronIndexedDataset, ref):
    """d. Assert document_indices are array_equal."""
    assert np.array_equal(ours.document_indices, ref.document_indices)


def test_getitem_random_samples(ours: MegatronIndexedDataset, ref):
    """e. Randomly sample 1000 indices and assert __getitem__ outputs are array_equal."""
    rng = np.random.default_rng(seed=42)
    indices = rng.integers(0, len(ours), size=NUM_RANDOM_SAMPLES)

    for idx in indices:
        ours_item = ours[int(idx)]
        ref_item = ref[int(idx)]
        assert isinstance(ours_item, np.ndarray)
        assert isinstance(ref_item, np.ndarray)
        assert np.array_equal(ours_item, ref_item), f"Mismatch at index {idx}"


def test_get_method(ours: MegatronIndexedDataset, ref):
    """f. Test .get(idx, offset, length) on a few indices."""
    test_cases: Tuple[Tuple[int, int, int], ...] = (
        (0, 0, 10),
        (1, 5, 20),
        (10, 0, None),   # length=None -> till end
        (42, 3, 7),
        (100, 0, 1),
    )

    for idx, offset, length in test_cases:
        ours_item = ours.get(idx, offset=offset, length=length)
        ref_item = ref.get(idx, offset=offset, length=length)
        assert isinstance(ours_item, np.ndarray)
        assert isinstance(ref_item, np.ndarray)
        assert np.array_equal(ours_item, ref_item), (
            f"Mismatch at get(idx={idx}, offset={offset}, length={length})"
        )


def test_slice_consistency(ours: MegatronIndexedDataset, ref):
    """Bonus: verify slice behaviour matches."""
    # A few deterministic slices
    for start, stop in [(0, 5), (10, 15), (100, 110)]:
        ours_items = ours[start:stop]
        ref_items = ref[start:stop]
        assert len(ours_items) == len(ref_items)
        for o_item, r_item in zip(ours_items, ref_items):
            assert np.array_equal(o_item, r_item), f"Slice mismatch [{start}:{stop}]"


def test_length_equal(ours: MegatronIndexedDataset, ref):
    assert len(ours) == len(ref)
