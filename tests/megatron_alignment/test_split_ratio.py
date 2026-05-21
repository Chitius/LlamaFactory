"""Test split ratio slicing for MegatronGPTDataset."""

import os
import shutil

import numpy as np
import pytest

from llamafactory.data.megatron.gpt_dataset import MegatronGPTDataset, MegatronGPTDatasetConfig
from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset


TEST_PREFIX = os.path.join(os.path.dirname(__file__), "../../data/c4_demo_text_document")
CACHE_PATH = "/tmp/lf_test_cache_split"
SEQ_LENGTH = 128
SEED = 42


def _cleanup_cache():
    if os.path.isdir(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)


@pytest.fixture(scope="module")
def indexed_dataset():
    return MegatronIndexedDataset(TEST_PREFIX, multimodal=False, mmap=True)


@pytest.fixture(autouse=True)
def clean_cache():
    _cleanup_cache()
    os.makedirs(CACHE_PATH, exist_ok=True)
    yield
    _cleanup_cache()


def _make_dataset(indexed_dataset, split: str, split_ratios: str):
    config = MegatronGPTDatasetConfig(
        path_prefix=TEST_PREFIX,
        seq_length=SEQ_LENGTH,
        seed=SEED,
        num_samples=1000,
        data_cache_path=CACHE_PATH,
        add_extra_token=True,
        drop_last_partial_sequence=True,
        split=split,
        split_ratios=split_ratios,
        pad_token_id=-1,
    )
    return MegatronGPTDataset(config, indexed_dataset)


def test_split_1_0_0_train_contains_all(indexed_dataset):
    """With split '1,0,0', the train dataset should see all documents."""
    ds = _make_dataset(indexed_dataset, split="train", split_ratios="1,0,0")
    total_docs = len(indexed_dataset)
    # document_index contains repeated docs across epochs, so check unique set
    unique_docs = set(int(d) for d in ds.document_index)
    assert unique_docs == set(range(total_docs)), (
        f"train split with 1,0,0 should contain all {total_docs} documents"
    )


def test_split_50_50_0_disjoint_ranges(indexed_dataset):
    """With split '0.5,0.5,0', train and valid should be disjoint and correct."""
    total_docs = len(indexed_dataset)
    ds_train = _make_dataset(indexed_dataset, split="train", split_ratios="0.5,0.5,0")
    ds_valid = _make_dataset(indexed_dataset, split="valid", split_ratios="0.5,0.5,0")

    train_docs = set(int(d) for d in ds_train.document_index)
    valid_docs = set(int(d) for d in ds_valid.document_index)

    assert train_docs.isdisjoint(valid_docs), "train and valid document sets should be disjoint"

    expected_train_end = int(total_docs * 0.5)
    expected_valid_end = expected_train_end + int(total_docs * 0.5)

    assert train_docs == set(range(0, expected_train_end)), (
        f"train docs should be [0, {expected_train_end})"
    )
    assert valid_docs == set(range(expected_train_end, expected_valid_end)), (
        f"valid docs should be [{expected_train_end}, {expected_valid_end})"
    )

    # Sum of unique docs should approximately equal total docs (allowing rounding)
    assert len(train_docs) + len(valid_docs) <= total_docs
    assert abs((len(train_docs) + len(valid_docs)) - total_docs) <= 1


def test_split_50_50_0_test_raises_empty(indexed_dataset):
    """With split '0.5,0.5,0', test split has 0 documents and should raise ValueError."""
    with pytest.raises(ValueError, match="No documents available"):
        _make_dataset(indexed_dataset, split="test", split_ratios="0.5,0.5,0")


def test_split_cache_no_collision(indexed_dataset):
    """Cache files for different splits should not collide."""
    ds_train = _make_dataset(indexed_dataset, split="train", split_ratios="0.5,0.5,0")
    ds_valid = _make_dataset(indexed_dataset, split="valid", split_ratios="0.5,0.5,0")

    train_cache = ds_train._get_cache_base_name()
    valid_cache = ds_valid._get_cache_base_name()

    assert train_cache != valid_cache, (
        f"train cache '{train_cache}' should differ from valid cache '{valid_cache}'"
    )
