"""L6: Document-Boundary Attention Mask alignment tests.

These tests verify:
  - document_ids generation matches sample_index boundaries
  - prepare_4d_attention_mask correctly blocks cross-document attention
  - position_ids reset at document boundaries
  - labels correctly mask padding and EOD tokens
  - MegatronDataCollatorForLanguageModeling preserves dataset labels,
    pads document_ids/position_ids, and generates 4D masks.
"""

import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pytest
import torch

from llamafactory.data.collator import (
    apply_document_boundary_mask,
    prepare_4d_attention_mask,
    _compute_cu_seq_lens_for_document_boundary,
)
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
_get_ltor_masks_and_position_ids = _megatron_mod._get_ltor_masks_and_position_ids
_Split = sys.modules["megatron.core.datasets.utils"].Split

# Override the Megatron tokenizer stub with correct EOD for c4_demo (GPT-2)
class _MegatronFakeTokenizer:
    vocab_size = 50000
    eod = 50256
    eos = 50256
    pad = -1
    special_tokens_dict = {}
    unique_identifiers = {}

sys.modules["megatron.core.tokenizers"].MegatronTokenizerBase = _MegatronFakeTokenizer

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
TEST_PREFIX = os.path.join(os.path.dirname(__file__), "../../data/c4_demo_text_document")
CACHE_PATH = "/tmp/lf_test_cache_l6"
SEQ_LENGTH = 128
SEED = 42
NUM_SAMPLES = 500

CACHE_PATH_LF_L6A = "/tmp/lf_test_cache_l6a_lf"
CACHE_PATH_MG_L6A = "/tmp/lf_test_cache_l6a_mg"
REPORT_PATH_L6A = "/tmp/lf_megatron_l6a_report.md"


def _cleanup_cache():
    if os.path.isdir(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)


@pytest.fixture(scope="module")
def indexed_dataset():
    return MegatronIndexedDataset(TEST_PREFIX, multimodal=False, mmap=True)


@pytest.fixture(scope="module")
def dataset_reset_all(indexed_dataset):
    _cleanup_cache()
    os.makedirs(CACHE_PATH, exist_ok=True)

    config = MegatronGPTDatasetConfig(
        path_prefix=TEST_PREFIX,
        seq_length=SEQ_LENGTH,
        seed=SEED,
        num_samples=NUM_SAMPLES,
        data_cache_path=CACHE_PATH,
        add_extra_token=True,
        drop_last_partial_sequence=True,
        split="train",
        pad_token_id=0,
        eod_token_id=50256,
        reset_attention_mask=True,
        reset_position_ids=True,
        eod_mask_loss=True,
    )
    dataset = MegatronGPTDataset(config, indexed_dataset)
    yield dataset
    _cleanup_cache()


@pytest.fixture(scope="module")
def dataset_no_reset(indexed_dataset):
    _cleanup_cache()
    os.makedirs(CACHE_PATH, exist_ok=True)

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
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
    )
    dataset = MegatronGPTDataset(config, indexed_dataset)
    yield dataset
    _cleanup_cache()


# ---------------------------------------------------------------------------
# document_ids generation
# ---------------------------------------------------------------------------


def test_document_ids_generation(dataset_reset_all):
    """Verify _get_sample generates correct document_ids."""
    dataset = dataset_reset_all
    for idx in range(len(dataset)):
        text, document_ids, _ = dataset._get_sample(idx)
        shuffled_idx = int(dataset._shuffle_index[idx])
        doc_beg, off_beg = dataset._sample_index[shuffled_idx]
        doc_end, off_end = dataset._sample_index[shuffled_idx + 1]

        expected_text_len = dataset.config.seq_length + (1 if dataset.config.add_extra_token else 0)
        assert len(text) == expected_text_len, f"text length mismatch at {idx}"
        assert len(document_ids) == len(text), f"document_ids length mismatch at {idx}"

        # Trailing padding positions (if any) have document_id == 0
        # Note: pad_token_id may coincide with EOD tokens in the raw data,
        # so we only check the trailing padding added by _get_sample.
        pad_id = dataset.config.pad_token_id
        trailing_pad = 0
        for j in range(len(text) - 1, -1, -1):
            if text[j] == pad_id:
                trailing_pad += 1
            else:
                break
        if trailing_pad > 0:
            assert np.all(document_ids[-trailing_pad:] == 0), f"padding document_ids not zero at {idx}"

        # Non-padding positions have document_id >= 1
        non_pad_mask = document_ids != 0
        assert np.all(document_ids[non_pad_mask] >= 1), f"non-padding document_ids < 1 at {idx}"

        # Document_id changes exactly at document boundaries
        if doc_beg == doc_end:
            assert np.all(document_ids[non_pad_mask] == 1), f"single-doc sample has wrong ids at {idx}"
        else:
            unique_ids = np.unique(document_ids[non_pad_mask])
            expected_num_docs = doc_end - doc_beg + 1
            assert len(unique_ids) == expected_num_docs, (
                f"doc count mismatch at {idx}: expected {expected_num_docs}, got {len(unique_ids)}"
            )


# ---------------------------------------------------------------------------
# prepare_4d_attention_mask correctness
# ---------------------------------------------------------------------------


def test_prepare_4d_mask_blocks_cross_document_attention():
    """Verify 4D mask blocks cross-document and preserves causal."""
    document_ids = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
    mask_4d = prepare_4d_attention_mask(document_ids, torch.float32)

    assert mask_4d.shape == (1, 1, 5, 5)
    min_dtype = torch.finfo(torch.float32).min

    # Cross-doc should be masked
    assert mask_4d[0, 0, 0, 2] == min_dtype
    assert mask_4d[0, 0, 0, 3] == min_dtype
    assert mask_4d[0, 0, 2, 0] == min_dtype
    assert mask_4d[0, 0, 2, 1] == min_dtype

    # Causal violation should be masked
    assert mask_4d[0, 0, 0, 1] == min_dtype
    assert mask_4d[0, 0, 1, 2] == min_dtype

    # Same-doc lower triangular should be unmasked
    assert mask_4d[0, 0, 0, 0] == 0.0
    assert mask_4d[0, 0, 1, 0] == 0.0
    assert mask_4d[0, 0, 1, 1] == 0.0
    assert mask_4d[0, 0, 2, 2] == 0.0
    # query 2, key 3: same doc but future -> masked by causal constraint
    assert mask_4d[0, 0, 2, 3] == min_dtype
    assert mask_4d[0, 0, 3, 2] == 0.0  # query 3 can attend to key 2 (same doc, past)
    assert mask_4d[0, 0, 3, 3] == 0.0

    # Padding should be masked
    assert torch.all(mask_4d[0, 0, 4, :] == min_dtype)
    assert torch.all(mask_4d[0, 0, :, 4] == min_dtype)


# ---------------------------------------------------------------------------
# position_ids reset
# ---------------------------------------------------------------------------


def test_position_ids_reset(dataset_reset_all):
    """Verify position_ids reset at document boundaries."""
    dataset = dataset_reset_all
    for idx in range(min(100, len(dataset))):
        item = dataset[idx]
        position_ids = item["position_ids"].numpy()
        document_ids = item["document_ids"].numpy()

        non_pad = document_ids != 0
        if not np.any(non_pad):
            continue

        doc_ids_np = document_ids[non_pad]
        pos_ids_np = position_ids[non_pad]

        boundaries = np.where(np.diff(doc_ids_np, prepend=doc_ids_np[0]))[0]
        for b in boundaries:
            if b > 0 and b < len(pos_ids_np):
                assert pos_ids_np[b] == 0, f"position_ids should reset at boundary {b} in sample {idx}"

        # Within each segment, position_ids should be consecutive
        for doc_id in np.unique(doc_ids_np):
            segment_pos = pos_ids_np[doc_ids_np == doc_id]
            expected = np.arange(len(segment_pos))
            assert np.array_equal(segment_pos, expected), (
                f"position_ids not consecutive in doc {doc_id} of sample {idx}"
            )


# ---------------------------------------------------------------------------
# labels masking
# ---------------------------------------------------------------------------


def test_labels_masking_padding():
    """Verify padding positions are masked to -100 in labels.

    Uses a synthetic MockIndexedDataset with short documents to guarantee
    that at least one sample requires padding (c4_demo documents are too
    long to trigger padding at seq_length=128 with drop_last=True).
    """
    _cleanup_cache()
    os.makedirs(CACHE_PATH, exist_ok=True)

    class _MockIndexedDataset:
        def __init__(self, sequence_lengths):
            self.sequence_lengths = np.array(sequence_lengths, dtype=np.int32)

        def __len__(self):
            return len(self.sequence_lengths)

        def get(self, idx, offset=0, length=None):
            total = self.sequence_lengths[idx]
            if length is None:
                length = total - offset
            # Use idx*1000 offset so tokens from different docs are distinct
            return np.arange(offset, offset + length, dtype=np.int64) + idx * 1000

    # Two documents: 30 and 20 tokens. seq_length=32 => target=33.
    # Sample 0: doc0[0:30] + doc1[0:3]  = 33 (no pad)
    # Sample 1: doc1[3:20] = 17 < 33   => 16 pad tokens
    mock_ds = _MockIndexedDataset([30, 20])
    config = MegatronGPTDatasetConfig(
        path_prefix="/tmp/mock_prefix",
        seq_length=32,
        seed=SEED,
        num_samples=None,  # one epoch
        data_cache_path=CACHE_PATH,
        add_extra_token=True,
        drop_last_partial_sequence=False,
        split="train",
        pad_token_id=0,
        reset_attention_mask=True,
        reset_position_ids=False,
        eod_mask_loss=False,
    )
    dataset = MegatronGPTDataset(config, mock_ds)

    found_padding = False
    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"]
        labels = item["labels"]
        document_ids = item["document_ids"]

        # Identify padding via document_ids (padding positions are 0)
        pad_positions = document_ids == 0
        if torch.any(pad_positions):
            found_padding = True
            assert torch.all(labels[pad_positions] == -100), f"padding not masked at {idx}"
            # Padding input_ids should equal pad_token_id
            assert torch.all(input_ids[pad_positions] == config.pad_token_id), f"padding input_ids mismatch at {idx}"
            # Non-padding labels should NOT be -100
            non_pad = ~pad_positions
            if torch.any(non_pad):
                assert torch.all(labels[non_pad] != -100), f"non-padding incorrectly masked at {idx}"

    assert found_padding, "No padding sample found in synthetic dataset; test did not exercise padding logic"
    _cleanup_cache()


def test_labels_masking_eod():
    """Verify EOD positions are masked to -100 when eod_mask_loss=True."""
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
        pad_token_id=0,
        eod_token_id=50256,
        reset_attention_mask=True,
        reset_position_ids=False,
        eod_mask_loss=True,
    )
    dataset = MegatronGPTDataset(config, indexed_ds)

    eod_found = False
    for idx in range(min(200, len(dataset))):
        item = dataset[idx]
        input_ids = item["input_ids"]
        labels = item["labels"]
        eod_positions = input_ids == config.eod_token_id
        if torch.any(eod_positions):
            eod_found = True
            assert torch.all(labels[eod_positions] == -100), f"EOD not masked at {idx}"

    # c4_demo uses GPT-2 tokenizer (eod=50256). EOD may not appear in every sample,
    # but should appear somewhere in the first 200 samples.
    assert eod_found, "No EOD token found in first 200 samples"
    _cleanup_cache()


# ---------------------------------------------------------------------------
# Collator tests
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub for collator tests."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"

    def pad(self, encoded_inputs, return_tensors=None, **kwargs):
        # Simple manual pad for testing
        max_len = max(len(ex["input_ids"]) for ex in encoded_inputs)
        result = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            if key not in encoded_inputs[0]:
                continue
            padded = []
            for ex in encoded_inputs:
                val = ex[key]
                if isinstance(val, torch.Tensor):
                    val = val.tolist()
                pad_len = max_len - len(val)
                if key == "labels":
                    pad_val = -100
                elif key == "attention_mask":
                    pad_val = 0
                else:
                    pad_val = self.pad_token_id
                if self.padding_side == "right":
                    padded.append(val + [pad_val] * pad_len)
                else:
                    padded.append([pad_val] * pad_len + val)
            if return_tensors == "pt":
                result[key] = torch.tensor(padded, dtype=torch.long)
            else:
                result[key] = padded
        return result


def test_collator_preserves_dataset_labels_and_document_ids():
    """Verify collator preserves labels, pads document_ids, and keeps attention_mask 2D."""
    from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling

    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    examples = [
        {"input_ids": [1, 2, 3, 4], "labels": [1, -100, 3, 4], "attention_mask": [1, 1, 1, 1], "document_ids": [1, 1, 2, 2]},
        {"input_ids": [5, 6], "labels": [-100, 6], "attention_mask": [1, 1], "document_ids": [1, 1]},
    ]

    batch = collator(examples)

    # Labels preserved and padded
    assert batch["labels"][0, 0] == 1
    assert batch["labels"][0, 1] == -100
    assert batch["labels"][0, 2] == 3
    assert batch["labels"][1, 0] == -100
    assert batch["labels"][1, 1] == 6
    assert batch["labels"][1, 2] == -100  # padding

    # attention_mask should be 2D padding mask before apply_document_boundary_mask,
    # but collator calls apply_document_boundary_mask which converts it to 4D when block_diag_attn=True
    assert batch["attention_mask"].dim() == 4  # (bsz, 1, seq_len, seq_len)

    # document_ids should have been popped (apply_document_boundary_mask pops it)
    assert "document_ids" not in batch


def test_collator_generates_4d_mask_from_document_ids():
    """Verify collator produces correct 4D block-diagonal causal mask."""
    from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling

    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    examples = [
        {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "document_ids": [1, 1, 2, 2]},
    ]

    batch = collator(examples)

    assert batch["attention_mask"].dim() == 4  # (1, 1, 4, 4)

    min_dtype = torch.finfo(torch.float32).min
    # Cross-doc blocked
    assert batch["attention_mask"][0, 0, 0, 2] == min_dtype
    assert batch["attention_mask"][0, 0, 0, 3] == min_dtype
    assert batch["attention_mask"][0, 0, 2, 0] == min_dtype
    assert batch["attention_mask"][0, 0, 2, 1] == min_dtype

    # Same-doc causal unmasked
    assert batch["attention_mask"][0, 0, 0, 0] == 0.0
    assert batch["attention_mask"][0, 0, 1, 0] == 0.0
    assert batch["attention_mask"][0, 0, 1, 1] == 0.0
    assert batch["attention_mask"][0, 0, 2, 2] == 0.0
    assert batch["attention_mask"][0, 0, 3, 2] == 0.0
    assert batch["attention_mask"][0, 0, 3, 3] == 0.0


def test_collator_position_ids_padded():
    """Verify collator pads position_ids correctly."""
    from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling

    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        block_diag_attn=False,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    examples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1], "position_ids": [0, 1, 2]},
        {"input_ids": [4, 5], "labels": [4, 5], "attention_mask": [1, 1], "position_ids": [0, 1]},
    ]

    batch = collator(examples)

    assert "position_ids" in batch
    assert batch["position_ids"][0].tolist() == [0, 1, 2]
    assert batch["position_ids"][1].tolist() == [0, 1, 0]  # padded with 0


def test_collator_no_document_ids_when_disabled():
    """Verify collator does not generate 4D mask when block_diag_attn=False."""
    from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling

    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        block_diag_attn=False,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    examples = [
        {"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1, 1], "document_ids": [1, 1]},
    ]

    batch = collator(examples)

    # When block_diag_attn=False, attention_mask should stay 2D
    assert batch["attention_mask"].dim() == 2
    # document_ids should be popped regardless
    assert "document_ids" not in batch


# ---------------------------------------------------------------------------
# Backward compatibility: default (no reset) behavior unchanged
# ---------------------------------------------------------------------------


def test_default_behavior_unchanged(dataset_no_reset):
    """Verify default behavior (all resets disabled) is unchanged."""
    dataset = dataset_no_reset
    for idx in range(min(50, len(dataset))):
        item = dataset[idx]
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        # document_ids should NOT be present when reset_attention_mask=False
        assert "document_ids" not in item
        # position_ids should NOT be present when reset_position_ids=False
        assert "position_ids" not in item
        # labels should be unshifted (same as tokens)
        assert torch.equal(item["labels"], item["input_ids"])


def test_dataset_config_priority_resolution():
    """Verify _resolve_bool correctly handles dataset-level vs global config priority.

    Dataset-level explicit values (True or False) should always override
    data_args global defaults. Only when dataset-level is None (unset)
    should the global value be used.
    """
    from llamafactory.data.loader import _resolve_bool

    # Dataset unset -> use global
    assert _resolve_bool(None, False) is False
    assert _resolve_bool(None, True) is True

    # Dataset explicitly True -> override global False
    assert _resolve_bool(True, False) is True

    # Dataset explicitly False -> override global True
    assert _resolve_bool(False, True) is False

    # Both explicitly set -> use dataset (same value in these cases)
    assert _resolve_bool(True, True) is True
    assert _resolve_bool(False, False) is False


# ---------------------------------------------------------------------------
# L6a: Attention Mask structure alignment (LF 4D vs Megatron 2D)
# ---------------------------------------------------------------------------


def _cleanup_cache_l6a():
    for p in [CACHE_PATH_LF_L6A, CACHE_PATH_MG_L6A]:
        if os.path.isdir(p):
            shutil.rmtree(p)


@pytest.fixture(scope="module")
def dataset_l6a_lf() -> MegatronGPTDataset:
    _cleanup_cache_l6a()
    os.makedirs(CACHE_PATH_LF_L6A, exist_ok=True)

    indexed_ds = MegatronIndexedDataset(TEST_PREFIX, multimodal=False, mmap=True)
    config = MegatronGPTDatasetConfig(
        path_prefix=TEST_PREFIX,
        seq_length=SEQ_LENGTH,
        seed=SEED,
        num_samples=NUM_SAMPLES,
        data_cache_path=CACHE_PATH_LF_L6A,
        add_extra_token=True,
        drop_last_partial_sequence=True,
        split="train",
        pad_token_id=0,
        eod_token_id=50256,
        reset_attention_mask=True,
        reset_position_ids=True,
        eod_mask_loss=True,
    )
    dataset = MegatronGPTDataset(config, indexed_ds)
    yield dataset
    _cleanup_cache_l6a()


@pytest.fixture(scope="module")
def dataset_l6a_mg():
    _cleanup_cache_l6a()
    os.makedirs(CACHE_PATH_MG_L6A, exist_ok=True)

    indexed_ds = sys.modules["megatron.core.datasets.indexed_dataset"].IndexedDataset(
        TEST_PREFIX, multimodal=False, mmap=True
    )
    config = _GPTDatasetConfig(
        random_seed=SEED,
        sequence_length=SEQ_LENGTH,
        blend=([TEST_PREFIX], None),
        split="1,0,0",
        path_to_cache=CACHE_PATH_MG_L6A,
        tokenizer=_MegatronFakeTokenizer(),
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
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
    _cleanup_cache_l6a()


def test_l6a_attention_mask_alignment(dataset_l6a_lf: MegatronGPTDataset, dataset_l6a_mg):
    """Verify LF 4D attention_mask is mathematically equivalent to Megatron 2D mask.

    For each sample:
      1. LF generates document_ids from sample_index boundaries;
         prepare_4d_attention_mask produces a [1, 1, seq_len, seq_len] float mask.
      2. Megatron scans tokens for EOD (50256) and builds a [1, seq_len, seq_len]
         bool mask via _get_ltor_masks_and_position_ids.

    c4_demo guarantees EOD tokens appear only at document boundaries, so the two
    methods should produce identical masks on non-padding positions. Padding
    positions are excluded from comparison because Megatron does not modify the
    attention mask for padding (only the loss_mask), whereas LF masks padding.
    """
    min_dtype = torch.finfo(torch.float32).min
    total = min(len(dataset_l6a_lf), len(dataset_l6a_mg))
    num_compare = min(total, 500)  # compare up to 500 samples

    report_lines = [
        "# L6a Attention Mask Structure Alignment Report\n",
        f"- **Date**: {datetime.now().isoformat()}\n",
        f"- **Dataset**: `{TEST_PREFIX}`\n",
        f"- **Config**: seq_length={SEQ_LENGTH}, seed={SEED}, num_samples={NUM_SAMPLES}\n",
        f"- **Samples compared**: {num_compare}\n",
        "\n",
    ]

    all_pass = True
    mismatched_samples = []

    for idx in range(num_compare):
        # LF side
        item = dataset_l6a_lf[idx]
        document_ids = item["document_ids"]
        mask_4d = prepare_4d_attention_mask(document_ids.unsqueeze(0), torch.float32)

        # Megatron side
        item_mg = dataset_l6a_mg[idx]
        tokens = item_mg["tokens"]
        mask_2d_bool, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens, 50256, True, True, True, True
        )
        mask_mg_float = torch.where(mask_2d_bool.unsqueeze(0), min_dtype, 0.0)

        # Compare only non-padding positions
        non_pad_mask = document_ids != 0
        non_pad_indices = torch.where(non_pad_mask)[0]

        if len(non_pad_indices) == 0:
            # All padding – trivially equivalent
            report_lines.append(f"- Sample {idx:4d}: **PASS** (all padding)\n")
            continue

        mask_4d_sub = mask_4d[0, 0][non_pad_indices[:, None], non_pad_indices[None, :]]
        mask_mg_sub = mask_mg_float[0, 0][non_pad_indices[:, None], non_pad_indices[None, :]]

        match = torch.equal(mask_4d_sub, mask_mg_sub)

        if match:
            report_lines.append(f"- Sample {idx:4d}: **PASS**\n")
        else:
            all_pass = False
            mismatched_samples.append(idx)
            report_lines.append(f"- Sample {idx:4d}: **FAIL**\n")
            # Print debug info on first mismatch for easier diagnosis
            if len(mismatched_samples) == 1:
                diff = mask_4d_sub != mask_mg_sub
                diff_pos = torch.where(diff)
                report_lines.append(
                    f"  - First diff at (q={diff_pos[0][0].item()}, k={diff_pos[1][0].item()})\n"
                )

    report_lines.append("\n")
    if all_pass:
        report_lines.append("## Result: **PASS** ✅\n")
    else:
        report_lines.append("## Result: **FAIL** ❌\n")
        report_lines.append(f"Mismatched samples: {mismatched_samples}\n")

    with open(REPORT_PATH_L6A, "w") as f:
        f.writelines(report_lines)

    assert all_pass, (
        f"Attention mask mismatch in {len(mismatched_samples)} / {num_compare} samples. "
        f"Report written to {REPORT_PATH_L6A}."
    )


# ---------------------------------------------------------------------------
# L6b: Batch-Level Mask 验证
# ---------------------------------------------------------------------------


def test_l6b_batch_level_mask():
    """L6b: Verify 4D mask from collator correctly blocks cross-document attention.

    Constructs a synthetic dataset with two short documents (3 tokens each),
    draws one sample through MegatronGPTDataset, and validates the 4D
    attention_mask produced by MegatronDataCollatorForLanguageModeling.
    """
    cache_path = "/tmp/lf_test_cache_l6b"
    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    class _MockIndexedDataset:
        def __init__(self, sequence_lengths):
            self.sequence_lengths = np.array(sequence_lengths, dtype=np.int32)

        def __len__(self):
            return len(self.sequence_lengths)

        def get(self, idx, offset=0, length=None):
            total = self.sequence_lengths[idx]
            if length is None:
                length = total - offset
            # Tokens are distinct per document so boundaries are unambiguous
            return np.arange(offset, offset + length, dtype=np.int64) + idx * 3 + 1

    # Document A: [1,2,3], Document B: [4,5,6]
    mock_ds = _MockIndexedDataset([3, 3])
    config = MegatronGPTDatasetConfig(
        path_prefix="/tmp/mock_prefix_l6b",
        seq_length=4,
        seed=SEED,
        num_samples=None,  # one epoch
        data_cache_path=cache_path,
        add_extra_token=True,
        drop_last_partial_sequence=False,
        split="train",
        pad_token_id=0,
        eod_token_id=99,
        reset_attention_mask=True,
        reset_position_ids=False,
        eod_mask_loss=False,
    )
    dataset = MegatronGPTDataset(config, mock_ds)

    # --- Verify raw sample structure ---
    text, doc_ids_full, _ = dataset._get_sample(0)
    assert len(text) == 5, f"expected text length 5, got {len(text)}"
    assert len(doc_ids_full) == 5, f"expected document_ids length 5, got {len(doc_ids_full)}"
    expected_doc_ids_full = np.array([1, 1, 1, 2, 2], dtype=np.int64)
    assert np.array_equal(doc_ids_full, expected_doc_ids_full), (
        f"document_ids mismatch: {doc_ids_full} != {expected_doc_ids_full}"
    )

    # --- Verify __getitem__ ---
    item = dataset[0]
    tokens = item["input_ids"]
    doc_ids = item["document_ids"]
    assert len(tokens) == 4, f"expected tokens length 4, got {len(tokens)}"
    assert len(doc_ids) == 4, f"expected document_ids length 4, got {len(doc_ids)}"
    expected_doc_ids = torch.tensor([1, 1, 1, 2], dtype=torch.long)
    assert torch.equal(doc_ids, expected_doc_ids), (
        f"document_ids mismatch: {doc_ids} != {expected_doc_ids}"
    )

    # --- Collator produces 4D mask ---
    from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling

    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )
    example = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    batch = collator([example])
    mask = batch["attention_mask"]

    assert mask.dim() == 4, f"expected 4D mask, got {mask.dim()}D"
    assert mask.shape == (1, 1, 4, 4), f"expected shape (1,1,4,4), got {mask.shape}"

    min_dtype = torch.finfo(torch.float32).min

    # 1. Cross-document attention blocked: query 3 (doc2) cannot attend to keys 0,1,2 (doc1)
    assert mask[0, 0, 3, 0] == min_dtype
    assert mask[0, 0, 3, 1] == min_dtype
    assert mask[0, 0, 3, 2] == min_dtype

    # 2. Same-document (doc1) causal unmasked (lower triangular)
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 1, 0] == 0.0
    assert mask[0, 0, 1, 1] == 0.0
    assert mask[0, 0, 2, 0] == 0.0
    assert mask[0, 0, 2, 1] == 0.0
    assert mask[0, 0, 2, 2] == 0.0

    # 3. doc2 self-attend only
    assert mask[0, 0, 3, 3] == 0.0

    # 4. Causal / future blocked within doc1 and across docs
    assert mask[0, 0, 0, 1] == min_dtype
    assert mask[0, 0, 0, 2] == min_dtype
    assert mask[0, 0, 0, 3] == min_dtype
    assert mask[0, 0, 1, 2] == min_dtype
    assert mask[0, 0, 1, 3] == min_dtype
    assert mask[0, 0, 2, 3] == min_dtype

    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)

# ---------------------------------------------------------------------------
# cu_seq_lens computation tests
# ---------------------------------------------------------------------------


def test_compute_cu_seq_lens_basic():
    """Verify _compute_cu_seq_lens_for_document_boundary with mixed docs and padding."""
    document_ids = torch.tensor([[1, 1, 1, 2, 2, 0, 0], [1, 1, 1, 1, 2, 0, 0]])
    cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k = (
        _compute_cu_seq_lens_for_document_boundary(document_ids)
    )
    expected = torch.tensor([0, 3, 5, 7, 11, 12, 14], dtype=torch.int32)
    assert torch.equal(cu_seq_lens_q, expected), f"expected {expected}, got {cu_seq_lens_q}"
    assert max_length_q == 4, f"expected max_length=4, got {max_length_q}"
    assert cu_seq_lens_q.dtype == torch.int32, f"expected int32, got {cu_seq_lens_q.dtype}"
    assert torch.equal(cu_seq_lens_q, cu_seq_lens_k)
    assert max_length_q == max_length_k


def test_compute_cu_seq_lens_single_doc_no_padding():
    """Verify cu_seq_lens with single document and no padding."""
    document_ids = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
    cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k = (
        _compute_cu_seq_lens_for_document_boundary(document_ids)
    )
    expected = torch.tensor([0, 4, 8], dtype=torch.int32)
    assert torch.equal(cu_seq_lens_q, expected), f"expected {expected}, got {cu_seq_lens_q}"
    assert max_length_q == 4, f"expected max_length=4, got {max_length_q}"
    assert torch.equal(cu_seq_lens_q, cu_seq_lens_k)
    assert max_length_q == max_length_k


def test_compute_cu_seq_lens_multi_doc_no_padding():
    """Verify cu_seq_lens with multiple documents and no padding."""
    document_ids = torch.tensor([[1, 1, 2, 2], [1, 2, 2, 2]])
    cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k = (
        _compute_cu_seq_lens_for_document_boundary(document_ids)
    )
    expected = torch.tensor([0, 2, 4, 5, 8], dtype=torch.int32)
    assert torch.equal(cu_seq_lens_q, expected), f"expected {expected}, got {cu_seq_lens_q}"
    assert max_length_q == 3, f"expected max_length=3, got {max_length_q}"
    assert torch.equal(cu_seq_lens_q, cu_seq_lens_k)
    assert max_length_q == max_length_k


def test_compute_cu_seq_lens_all_padding():
    """Verify cu_seq_lens when all tokens are padding."""
    document_ids = torch.tensor([[0, 0, 0, 0]])
    cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k = (
        _compute_cu_seq_lens_for_document_boundary(document_ids)
    )
    expected = torch.tensor([0, 4], dtype=torch.int32)
    assert torch.equal(cu_seq_lens_q, expected), f"expected {expected}, got {cu_seq_lens_q}"
    assert max_length_q == 4, f"expected max_length=4, got {max_length_q}"
    assert torch.equal(cu_seq_lens_q, cu_seq_lens_k)
    assert max_length_q == max_length_k


# ---------------------------------------------------------------------------
# apply_document_boundary_mask FA2/FA3 tests
# ---------------------------------------------------------------------------


def test_apply_document_boundary_mask_fa2():
    """Verify apply_document_boundary_mask sets varlen keys for FA2."""
    features = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "document_ids": torch.tensor([[1, 1, 2, 2], [1, 1, 1, 1]]),
    }
    result = apply_document_boundary_mask(
        features,
        compute_dtype=torch.float32,
        attn_implementation="fa2",
        block_diag_attn=True,
    )
    assert "document_ids" not in result
    assert result["attention_mask"] is None
    assert "cu_seq_lens_q" in result
    assert "cu_seq_lens_k" in result
    assert "max_length_q" in result
    assert "max_length_k" in result
    assert result["cu_seq_lens_q"].dtype == torch.int32
    assert torch.equal(result["cu_seq_lens_q"], result["cu_seq_lens_k"])
    assert result["max_length_q"] == result["max_length_k"]


def test_apply_document_boundary_mask_fa3():
    """Verify apply_document_boundary_mask sets varlen keys for FA3."""
    features = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "document_ids": torch.tensor([[1, 1, 2, 2], [1, 1, 1, 1]]),
    }
    result = apply_document_boundary_mask(
        features,
        compute_dtype=torch.float32,
        attn_implementation="fa3",
        block_diag_attn=True,
    )
    assert "document_ids" not in result
    assert result["attention_mask"] is None
    assert "cu_seq_lens_q" in result
    assert "cu_seq_lens_k" in result
    assert "max_length_q" in result
    assert "max_length_k" in result
    assert result["cu_seq_lens_q"].dtype == torch.int32
    assert torch.equal(result["cu_seq_lens_q"], result["cu_seq_lens_k"])
    assert result["max_length_q"] == result["max_length_k"]


def test_apply_document_boundary_mask_fa2_disabled():
    """Verify apply_document_boundary_mask does not set varlen keys when block_diag_attn=False."""
    features = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "document_ids": torch.tensor([[1, 1, 2, 2], [1, 1, 1, 1]]),
    }
    result = apply_document_boundary_mask(
        features,
        compute_dtype=torch.float32,
        attn_implementation="fa2",
        block_diag_attn=False,
    )
    assert "document_ids" not in result
    assert "attention_mask" not in result or result["attention_mask"] is not None
    assert "cu_seq_lens_q" not in result
    assert "cu_seq_lens_k" not in result
    assert "max_length_q" not in result
    assert "max_length_k" not in result


def test_collator_fa2_outputs_varlen_keys():
    """Verify MegatronDataCollatorForLanguageModeling outputs varlen keys for FA2."""
    from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling

    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        block_diag_attn=True,
        attn_implementation="fa2",
        compute_dtype=torch.float32,
    )

    examples = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "document_ids": [1, 1, 2, 2],
            "position_ids": [0, 1, 0, 1],
        },
        {
            "input_ids": [5, 6],
            "labels": [5, 6],
            "attention_mask": [1, 1],
            "document_ids": [1, 1],
            "position_ids": [0, 1],
        },
    ]

    batch = collator(examples)

    assert "cu_seq_lens_q" in batch
    assert "cu_seq_lens_k" in batch
    assert "max_length_q" in batch
    assert "max_length_k" in batch
    assert batch["attention_mask"] is None
    assert batch["cu_seq_lens_q"].dtype == torch.int32
    assert torch.equal(batch["cu_seq_lens_q"], batch["cu_seq_lens_k"])
    assert batch["max_length_q"] == batch["max_length_k"]



# ---------------------------------------------------------------------------
# L6d: Padding 场景验证
# ---------------------------------------------------------------------------


def test_l6d_padding_scenario():
    """L6d: Verify padding sample mask, labels, document_ids, and position_ids.

    Constructs a synthetic dataset with two short documents (50 tokens each),
    draws samples through MegatronGPTDataset, and validates:
      - padding positions are correctly masked in labels and document_ids
      - 4D attention_mask from collator blocks padding rows/columns
      - position_ids reset at document boundaries
    """
    cache_path = "/tmp/lf_test_cache_l6d"
    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    class _MockIndexedDataset:
        def __init__(self, sequence_lengths):
            self.sequence_lengths = np.array(sequence_lengths, dtype=np.int32)

        def __len__(self):
            return len(self.sequence_lengths)

        def get(self, idx, offset=0, length=None):
            total = self.sequence_lengths[idx]
            if length is None:
                length = total - offset
            # Distinct tokens per document: doc0=1..50, doc1=101..150
            return np.arange(offset, offset + length, dtype=np.int64) + idx * 100 + 1

    mock_ds = _MockIndexedDataset([50, 50])
    config = MegatronGPTDatasetConfig(
        path_prefix="/tmp/mock_prefix_l6d",
        seq_length=64,
        seed=1,
        num_samples=None,  # one epoch
        data_cache_path=cache_path,
        add_extra_token=True,
        drop_last_partial_sequence=False,
        split="train",
        pad_token_id=0,
        eod_token_id=99,
        reset_attention_mask=True,
        reset_position_ids=True,
        eod_mask_loss=False,
    )
    dataset = MegatronGPTDataset(config, mock_ds)

    # Dataset should have exactly 2 samples
    assert len(dataset) == 2

    # --- Sample 0: no padding (doc0[0:50] + doc1[0:14] = 64 tokens) ---
    item0 = dataset[0]
    pad_mask0 = item0["document_ids"] == 0
    assert not torch.any(pad_mask0), "Sample 0 should have no padding"

    # Verify position_ids reset at document boundary in sample 0
    pos_ids0 = item0["position_ids"].numpy()
    doc_ids0 = item0["document_ids"].numpy()
    assert pos_ids0[0] == 0
    assert pos_ids0[49] == 49
    assert pos_ids0[50] == 0, "position_ids should reset at document boundary"
    assert pos_ids0[63] == 13
    assert np.array_equal(pos_ids0[:50], np.arange(50))
    assert np.array_equal(pos_ids0[50:64], np.arange(14))

    # --- Sample 1: has padding (doc1[14:50] + extra = 36 tokens, pad = 28) ---
    item1 = dataset[1]
    pad_mask1 = item1["document_ids"] == 0
    assert torch.any(pad_mask1), "Sample 1 should have padding"
    num_pad = int(torch.sum(pad_mask1).item())
    assert num_pad == 28, f"Expected 28 padding tokens, got {num_pad}"

    # 1. padding labels == -100
    assert torch.all(item1["labels"][pad_mask1] == -100)

    # 2. padding input_ids == pad_token_id
    assert torch.all(item1["input_ids"][pad_mask1] == config.pad_token_id)

    # 3. non-padding labels != -100
    non_pad1 = ~pad_mask1
    assert torch.all(item1["labels"][non_pad1] != -100)

    # 4. padding document_ids == 0
    assert torch.all(item1["document_ids"][pad_mask1] == 0)

    # 5. non-padding document_ids == 1 (single document segment)
    assert torch.all(item1["document_ids"][non_pad1] == 1)

    # 6. position_ids within non-padding are consecutive
    pos_ids1 = item1["position_ids"].numpy()
    non_pad_pos = pos_ids1[non_pad1.numpy()]
    assert np.array_equal(non_pad_pos, np.arange(36))

    # --- Collator 4D mask verification for sample 1 ---
    from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling

    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )
    example1 = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in item1.items()}
    batch = collator([example1])
    mask = batch["attention_mask"]

    assert mask.dim() == 4, f"expected 4D mask, got {mask.dim()}D"
    assert mask.shape == (1, 1, 64, 64), f"expected shape (1,1,64,64), got {mask.shape}"

    min_dtype = torch.finfo(torch.float32).min

    # 7. padding query rows all min_dtype
    assert torch.all(mask[0, 0, 36:, :] == min_dtype), "padding query rows not fully masked"

    # 8. padding key columns all min_dtype
    assert torch.all(mask[0, 0, :, 36:] == min_dtype), "padding key columns not fully masked"

    # 9. Non-padding self-attend is unmasked
    assert mask[0, 0, 35, 35] == 0.0
    assert mask[0, 0, 35, 0] == 0.0

    # 10. Causal: future positions are masked for non-padding
    assert mask[0, 0, 0, 35] == min_dtype

    # --- Megatron alignment (optional): sample 1 non-padding position_ids ---
    text1_np, doc_ids1_np, pos_ids1_raw = dataset._get_sample(1)
    non_pad_mask_np = doc_ids1_np != 0
    non_pad_text = torch.from_numpy(text1_np[non_pad_mask_np]).long()
    _, loss_mask_mg, pos_ids_mg = _get_ltor_masks_and_position_ids(
        non_pad_text, 99, True, True, False, True
    )
    assert torch.equal(
        torch.from_numpy(pos_ids1_raw[non_pad_mask_np]).long(),
        pos_ids_mg,
    ), "non-padding position_ids mismatch with Megatron"

    # loss_mask consistency: Megatron loss_mask=1 ↔ LF labels!=-100
    lf_loss_mask = (item1["labels"] != -100).float()
    assert torch.all(lf_loss_mask[non_pad1] == 1.0)

    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)


# ---------------------------------------------------------------------------
# MegatronDataCollatorForSeq2Seq tests
# ---------------------------------------------------------------------------


def test_megatron_data_collator_for_seq2seq_basic():
    """Verify MegatronDataCollatorForSeq2Seq pads document_ids/position_ids and generates 4D mask."""
    from llamafactory.data.megatron.collator import MegatronDataCollatorForSeq2Seq

    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    examples = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "document_ids": [1, 1, 2, 2],
            "position_ids": [0, 1, 0, 1],
        },
        {
            "input_ids": [5, 6],
            "labels": [5, 6],
            "attention_mask": [1, 1],
            "document_ids": [1, 1],
            "position_ids": [0, 1],
        },
    ]

    batch = collator(examples)

    # attention_mask should be 4D
    assert batch["attention_mask"].dim() == 4
    assert batch["attention_mask"].shape == (2, 1, 4, 4)

    min_dtype = torch.finfo(torch.float32).min
    # Cross-document attention blocked
    assert batch["attention_mask"][0, 0, 0, 2] == min_dtype
    assert batch["attention_mask"][0, 0, 0, 3] == min_dtype
    assert batch["attention_mask"][0, 0, 2, 0] == min_dtype
    assert batch["attention_mask"][0, 0, 2, 1] == min_dtype

    # document_ids should be popped
    assert "document_ids" not in batch

    # position_ids should be present and correctly padded
    assert "position_ids" in batch
    assert batch["position_ids"][0].tolist() == [0, 1, 0, 1]
    assert batch["position_ids"][1].tolist() == [0, 1, 0, 0]  # padded with 0


def test_megatron_data_collator_for_seq2seq_tensor_input():
    """Verify collator handles tensor inputs (as returned by MegatronGPTDataset)."""
    from llamafactory.data.megatron.collator import MegatronDataCollatorForSeq2Seq

    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "document_ids": torch.tensor([1, 1, 2, 2]),
            "position_ids": torch.tensor([0, 1, 0, 1]),
        },
        {
            "input_ids": torch.tensor([5, 6]),
            "labels": torch.tensor([5, 6]),
            "attention_mask": torch.tensor([1, 1]),
            "document_ids": torch.tensor([1, 1]),
            "position_ids": torch.tensor([0, 1]),
        },
    ]

    batch = collator(examples)

    assert batch["attention_mask"].dim() == 4
    assert batch["attention_mask"].shape == (2, 1, 4, 4)
    assert "document_ids" not in batch
    assert "position_ids" in batch
    assert batch["position_ids"][0].tolist() == [0, 1, 0, 1]
    assert batch["position_ids"][1].tolist() == [0, 1, 0, 0]


# ---------------------------------------------------------------------------
# CustomMcaTrainer pad tests
# ---------------------------------------------------------------------------


def test_mca_trainer_pad_batched_inputs_with_document_ids():
    """Verify CustomMcaTrainer._pad_batched_inputs pads document_ids and position_ids."""
    from unittest.mock import MagicMock

    from transformers import BatchEncoding

    from llamafactory.train.mca.trainer import CustomMcaTrainer

    trainer = CustomMcaTrainer.__new__(CustomMcaTrainer)
    trainer._language_input_names = ["input_ids", "attention_mask", "labels"]

    tokenizer = _FakeTokenizer()

    def mock_pad(encoded_inputs, return_tensors=None, **kwargs):
        max_len = kwargs.get("max_length")
        if max_len is None:
            max_len = max(len(ex["input_ids"]) for ex in encoded_inputs)
        result = {}
        for key in encoded_inputs[0]:
            padded = []
            for ex in encoded_inputs:
                val = ex[key]
                pad_len = max_len - len(val)
                if key == "labels":
                    pad_val = -100
                elif key == "attention_mask":
                    pad_val = 0
                else:
                    pad_val = tokenizer.pad_token_id
                padded.append(val + [pad_val] * pad_len)
            if return_tensors == "pt":
                result[key] = torch.tensor(padded, dtype=torch.long)
            else:
                result[key] = padded
        return BatchEncoding(result)

    mock_tokenizer = MagicMock()
    mock_tokenizer.padding_side = "right"
    mock_tokenizer.pad = mock_pad
    trainer.processing_class = mock_tokenizer

    mock_args = MagicMock()
    mock_args.device = "cpu"
    trainer.args = mock_args

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
        "labels": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "document_ids": torch.tensor([[1, 1, 2, 2], [1, 1, 1, 1]]),
        "position_ids": torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]]),
    }

    result = trainer._pad_batched_inputs(inputs, seq_length=10)

    assert result["document_ids"].shape == (2, 10)
    assert result["position_ids"].shape == (2, 10)
    assert torch.all(result["document_ids"][:, 4:] == 0)
    assert torch.all(result["position_ids"][:, 4:] == 0)
    assert result["document_ids"].device.type == "cpu"
    assert result["position_ids"].device.type == "cpu"


# ---------------------------------------------------------------------------
# SFTDataCollatorWith4DAttentionMask tests
# ---------------------------------------------------------------------------


def test_sft_collator_with_document_ids():
    """Verify SFTDataCollatorWith4DAttentionMask handles document_ids and position_ids correctly."""
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    from llamafactory.data.mm_plugin import get_mm_plugin

    tokenizer = _FakeTokenizer()
    fake_template = type("_FakeTemplate", (), {"mm_plugin": get_mm_plugin("base")})()

    collator = SFTDataCollatorWith4DAttentionMask(
        template=fake_template,
        tokenizer=tokenizer,
        model=None,
        processor=None,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    examples = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "document_ids": [1, 1, 2, 2],
            "position_ids": [0, 1, 0, 1],
        },
        {
            "input_ids": [5, 6],
            "labels": [5, 6],
            "attention_mask": [1, 1],
            "document_ids": [1, 1],
            "position_ids": [0, 1],
        },
    ]

    batch = collator(examples)

    # attention_mask should be 4D
    assert batch["attention_mask"].dim() == 4
    assert batch["attention_mask"].shape == (2, 1, 4, 4)

    min_dtype = torch.finfo(torch.float32).min
    # Cross-document attention blocked
    assert batch["attention_mask"][0, 0, 0, 2] == min_dtype
    assert batch["attention_mask"][0, 0, 0, 3] == min_dtype
    assert batch["attention_mask"][0, 0, 2, 0] == min_dtype
    assert batch["attention_mask"][0, 0, 2, 1] == min_dtype

    # document_ids should be popped
    assert "document_ids" not in batch

    # position_ids should be present and correctly padded
    assert "position_ids" in batch
    assert batch["position_ids"][0].tolist() == [0, 1, 0, 1]
    assert batch["position_ids"][1].tolist() == [0, 1, 0, 0]


def test_sft_collator_with_document_ids_tensor_input():
    """Verify SFTDataCollatorWith4DAttentionMask handles tensor document_ids/position_ids."""
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    from llamafactory.data.mm_plugin import get_mm_plugin

    tokenizer = _FakeTokenizer()
    fake_template = type("_FakeTemplate", (), {"mm_plugin": get_mm_plugin("base")})()

    collator = SFTDataCollatorWith4DAttentionMask(
        template=fake_template,
        tokenizer=tokenizer,
        model=None,
        processor=None,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "document_ids": torch.tensor([1, 1, 2, 2]),
            "position_ids": torch.tensor([0, 1, 0, 1]),
        },
        {
            "input_ids": torch.tensor([5, 6]),
            "labels": torch.tensor([5, 6]),
            "attention_mask": torch.tensor([1, 1]),
            "document_ids": torch.tensor([1, 1]),
            "position_ids": torch.tensor([0, 1]),
        },
    ]

    batch = collator(examples)

    assert batch["attention_mask"].dim() == 4
    assert batch["attention_mask"].shape == (2, 1, 4, 4)
    assert "document_ids" not in batch
    assert "position_ids" in batch
    assert batch["position_ids"][0].tolist() == [0, 1, 0, 1]
    assert batch["position_ids"][1].tolist() == [0, 1, 0, 0]


def test_sft_collator_hf_path_unchanged():
    """Verify SFTDataCollatorWith4DAttentionMask behavior unchanged for standard HF path."""
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    from llamafactory.data.mm_plugin import get_mm_plugin

    tokenizer = _FakeTokenizer()
    fake_template = type("_FakeTemplate", (), {"mm_plugin": get_mm_plugin("base")})()

    collator = SFTDataCollatorWith4DAttentionMask(
        template=fake_template,
        tokenizer=tokenizer,
        model=None,
        processor=None,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    examples = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
        },
        {
            "input_ids": [5, 6],
            "labels": [5, 6],
            "attention_mask": [1, 1],
        },
    ]

    batch = collator(examples)

    # attention_mask should be 4D (standard HF path with block_diag_attn)
    assert batch["attention_mask"].dim() == 4
    assert batch["attention_mask"].shape == (2, 1, 4, 4)

    # No document_ids in input -> no document_ids in output
    assert "document_ids" not in batch

    # No position_ids in input, model=None -> no position_ids in output
    assert "position_ids" not in batch


# ---------------------------------------------------------------------------
# shift_labels tests
# ---------------------------------------------------------------------------


def test_shift_labels_true():
    """Verify shift_labels=True produces text[1:] labels with text[:-1] input_ids."""
    cache_path = "/tmp/lf_test_cache_shift_true"
    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    class _MockIndexedDataset:
        def __init__(self, sequence_lengths):
            self.sequence_lengths = np.array(sequence_lengths, dtype=np.int32)

        def __len__(self):
            return len(self.sequence_lengths)

        def get(self, idx, offset=0, length=None):
            total = self.sequence_lengths[idx]
            if length is None:
                length = total - offset
            return np.arange(offset, offset + length, dtype=np.int64) + idx * 1000 + 1

    # Single document of 66 tokens, seq_length=64 => add_extra_token=True => text length 65
    mock_ds = _MockIndexedDataset([66])
    config = MegatronGPTDatasetConfig(
        path_prefix="/tmp/mock_prefix_shift",
        seq_length=64,
        seed=SEED,
        num_samples=None,
        data_cache_path=cache_path,
        add_extra_token=True,
        drop_last_partial_sequence=False,
        split="train",
        pad_token_id=0,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        shift_labels=True,
    )
    dataset = MegatronGPTDataset(config, mock_ds)

    checked = False
    for idx in range(len(dataset)):
        item = dataset[idx]
        text, _, _ = dataset._get_sample(idx)
        text_tensor = torch.from_numpy(text).long()

        # input_ids should be text[:-1]
        assert torch.equal(item["input_ids"], text_tensor[:-1]), f"input_ids mismatch at {idx}"
        # labels should be text[1:] (before padding mask)
        raw_labels = text_tensor[1:]
        # Account for padding mask: padding positions in shifted labels should be -100
        # This aligns with Megatron loss_mask semantics: mask the prediction of a pad token.
        expected_labels = raw_labels.clone()
        expected_labels[expected_labels == config.pad_token_id] = -100
        assert torch.equal(item["labels"], expected_labels), f"labels mismatch at {idx}"
        # For non-padding positions, labels should differ from input_ids (shifted)
        non_pad = item["labels"] != -100
        if torch.any(non_pad):
            checked = True
            assert not torch.equal(item["labels"][non_pad], item["input_ids"][non_pad]), f"labels should be shifted at {idx}"

    assert checked, "No non-padding sample found to verify shift semantics"

    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)


def test_shift_labels_false():
    """Verify shift_labels=False (default) preserves backward compatibility: labels == input_ids (modulo padding mask)."""
    cache_path = "/tmp/lf_test_cache_shift_false"
    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    class _MockIndexedDataset:
        def __init__(self, sequence_lengths):
            self.sequence_lengths = np.array(sequence_lengths, dtype=np.int32)

        def __len__(self):
            return len(self.sequence_lengths)

        def get(self, idx, offset=0, length=None):
            total = self.sequence_lengths[idx]
            if length is None:
                length = total - offset
            return np.arange(offset, offset + length, dtype=np.int64) + idx * 1000 + 1

    mock_ds = _MockIndexedDataset([66])
    config = MegatronGPTDatasetConfig(
        path_prefix="/tmp/mock_prefix_shift",
        seq_length=64,
        seed=SEED,
        num_samples=None,
        data_cache_path=cache_path,
        add_extra_token=True,
        drop_last_partial_sequence=False,
        split="train",
        pad_token_id=0,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        shift_labels=False,
    )
    dataset = MegatronGPTDataset(config, mock_ds)

    for idx in range(len(dataset)):
        item = dataset[idx]
        # labels should equal input_ids everywhere except padding positions (which are -100)
        pad_mask = item["input_ids"] == config.pad_token_id
        non_pad = ~pad_mask
        if torch.any(non_pad):
            assert torch.equal(item["labels"][non_pad], item["input_ids"][non_pad]), f"labels should equal input_ids at {idx}"
        if torch.any(pad_mask):
            assert torch.all(item["labels"][pad_mask] == -100), f"padding should be masked at {idx}"

    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)


def test_mca_collator_with_shifted_labels():
    """Verify MegatronDataCollatorForSeq2Seq preserves shifted labels semantics."""
    from llamafactory.data.megatron.collator import MegatronDataCollatorForSeq2Seq

    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        block_diag_attn=False,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    # Simulate features where labels are already shifted (Megatron convention)
    examples = [
        {
            "input_ids": [1, 2, 3, 4],      # predicts ->
            "labels": [2, 3, 4, 5],          # <- next tokens
            "attention_mask": [1, 1, 1, 1],
        },
        {
            "input_ids": [10, 11],
            "labels": [11, 12],
            "attention_mask": [1, 1],
        },
    ]

    batch = collator(examples)

    # Labels should preserve shifted semantics (not be further shifted by collator)
    assert batch["labels"][0].tolist() == [2, 3, 4, 5]
    assert batch["labels"][1].tolist() == [11, 12, -100, -100]

    # input_ids should remain unchanged
    assert batch["input_ids"][0].tolist() == [1, 2, 3, 4]
    assert batch["input_ids"][1].tolist() == [10, 11, 0, 0]
