"""Tests for MCA path PackedSeqParams / varlen FlashAttention support.

These tests verify:
  - _compute_packed_seq_params_for_document_boundary correctness
  - MegatronDataCollatorForSeq2Seq generates PackedSeqParams for fa2/fa3
  - MegatronDataCollatorForSeq2Seq generates 4D mask for eager/sdpa
  - CustomMcaTrainer._pad_batched_inputs handles packed_seq_params
  - _resolve_attn_implementation_for_mca mapping logic

TODO: True end-to-end GPU validation with mcore_adapter is not possible in this
environment because mcore_adapter is a dummy package and Megatron compiled modules
are built for a different Python version.
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch
from transformers import PreTrainedTokenizerBase

# Ensure llamafactory is importable
sys.path.insert(0, "/home/public/liuyichuan/playground/LlamaFactory/src")

# Mock mcore_adapter before any MCA imports trigger the availability check
import importlib.util

_mock_mcore_adapter = MagicMock()
_mock_mcore_adapter.__spec__ = importlib.util.spec_from_loader("mcore_adapter", loader=None)
sys.modules["mcore_adapter"] = _mock_mcore_adapter
sys.modules["mcore_adapter.models"] = _mock_mcore_adapter.models
sys.modules["mcore_adapter.trainer"] = _mock_mcore_adapter.trainer
sys.modules["mcore_adapter.trainer.dpo_config"] = _mock_mcore_adapter.trainer.dpo_config
# NOTE: trainer.utils must be mocked before CustomMcaTrainer imports it.
_mock_trainer_utils = MagicMock()
sys.modules["mcore_adapter.trainer.utils"] = _mock_trainer_utils
_mock_mcore_adapter.trainer.McaTrainer = object

from llamafactory.data.megatron.collator import (
    MegatronDataCollatorForSeq2Seq,
    _compute_packed_seq_params_for_document_boundary,
)
from llamafactory.extras.constants import AttentionFunction


# ---------------------------------------------------------------------------
# Fake tokenizer stub (same style as test_l6_document_boundary_mask.py)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub for collator tests."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"

    def pad(self, encoded_inputs, return_tensors=None, **kwargs):
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


# ---------------------------------------------------------------------------
# _compute_packed_seq_params_for_document_boundary tests
# ---------------------------------------------------------------------------


def test_compute_packed_seq_params_basic():
    """Mixed docs and padding; trailing padding included as last segment."""
    document_ids = torch.tensor([[1, 1, 1, 2, 2, 0, 0], [1, 1, 1, 1, 2, 0, 0]])
    psp = _compute_packed_seq_params_for_document_boundary(document_ids)

    assert psp.qkv_format == "thd"
    # Row 0: segments [3, 2, 2] (valid [1,1,1,2,2] + padding [0,0])
    # Row 1: segments [4, 1, 2] (valid [1,1,1,1,2] + padding [0,0])
    # cu = [0, 3, 5, 7, 11, 12, 14]
    expected_cu = torch.tensor([0, 3, 5, 7, 11, 12, 14], dtype=torch.int32)
    assert torch.equal(psp.cu_seqlens_q, expected_cu)
    assert torch.equal(psp.cu_seqlens_kv, expected_cu)
    assert psp.max_seqlen_q == 4
    assert psp.max_seqlen_kv == 4
    assert psp.cu_seqlens_q.dtype == torch.int32


def test_compute_packed_seq_params_single_doc_no_padding():
    document_ids = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
    psp = _compute_packed_seq_params_for_document_boundary(document_ids)
    expected_cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    assert torch.equal(psp.cu_seqlens_q, expected_cu)
    assert psp.max_seqlen_q == 4


def test_compute_packed_seq_params_multi_doc_no_padding():
    document_ids = torch.tensor([[1, 1, 2, 2], [1, 2, 2, 2]])
    psp = _compute_packed_seq_params_for_document_boundary(document_ids)
    expected_cu = torch.tensor([0, 2, 4, 5, 8], dtype=torch.int32)
    assert torch.equal(psp.cu_seqlens_q, expected_cu)
    assert psp.max_seqlen_q == 3


def test_compute_packed_seq_params_all_padding():
    """All padding should produce [0, 4] cu_seqlens and max_seqlen=4."""
    document_ids = torch.tensor([[0, 0, 0, 0]])
    psp = _compute_packed_seq_params_for_document_boundary(document_ids)
    expected_cu = torch.tensor([0, 4], dtype=torch.int32)
    assert torch.equal(psp.cu_seqlens_q, expected_cu)
    assert psp.max_seqlen_q == 4


def test_compute_packed_seq_params_left_padding():
    """Left padding (leading zeros) treated as doc-id 0 segment; trailing padding appended."""
    document_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    psp = _compute_packed_seq_params_for_document_boundary(document_ids)
    # valid_len=6, padding_len=0, valid_row=[0,0,1,1,2,2]
    # segments [2, 2, 2], cu = [0, 2, 4, 6]
    expected_cu = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    assert torch.equal(psp.cu_seqlens_q, expected_cu)
    assert psp.max_seqlen_q == 2


# ---------------------------------------------------------------------------
# MegatronDataCollatorForSeq2Seq tests
# ---------------------------------------------------------------------------


def test_seq2seq_collator_fa2_outputs_packed_seq_params():
    """Verify FA2 mode produces PackedSeqParams and None attention_mask."""
    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
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

    assert "packed_seq_params" in batch
    assert batch["attention_mask"] is None
    assert "document_ids" not in batch
    psp = batch["packed_seq_params"]
    assert psp.qkv_format == "thd"
    assert psp.cu_seqlens_q is not None
    assert psp.cu_seqlens_kv is not None
    assert psp.max_seqlen_q == 2
    assert psp.max_seqlen_kv == 2


def test_seq2seq_collator_fa3_outputs_packed_seq_params():
    """Verify FA3 mode also produces PackedSeqParams."""
    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        block_diag_attn=True,
        attn_implementation="fa3",
        compute_dtype=torch.float32,
    )

    examples = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "document_ids": [1, 1, 2, 2],
        },
    ]

    batch = collator(examples)
    assert "packed_seq_params" in batch
    assert batch["attention_mask"] is None


def test_seq2seq_collator_eager_outputs_4d_mask():
    """Verify eager mode produces 4D attention_mask."""
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
        },
    ]

    batch = collator(examples)

    assert batch["attention_mask"].dim() == 4
    assert batch["attention_mask"].shape == (1, 1, 4, 4)
    assert "packed_seq_params" not in batch
    assert "document_ids" not in batch

    min_dtype = torch.finfo(torch.float32).min
    assert batch["attention_mask"][0, 0, 0, 2] == min_dtype
    assert batch["attention_mask"][0, 0, 0, 0] == 0.0


def test_seq2seq_collator_sdpa_outputs_4d_mask():
    """Verify SDPA mode produces 4D attention_mask."""
    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        block_diag_attn=True,
        attn_implementation="sdpa",
        compute_dtype=torch.float32,
    )

    examples = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "document_ids": [1, 1, 2, 2],
        },
    ]

    batch = collator(examples)
    assert batch["attention_mask"].dim() == 4
    assert "packed_seq_params" not in batch


def test_seq2seq_collator_no_document_ids_passthrough():
    """Verify collator works normally when document_ids are absent."""
    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        block_diag_attn=True,
        attn_implementation="fa2",
        compute_dtype=torch.float32,
    )

    examples = [
        {"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1, 1]},
    ]

    batch = collator(examples)
    # No document_ids -> no packed_seq_params, attention_mask stays 2D
    assert "packed_seq_params" not in batch
    assert batch["attention_mask"].dim() == 2


def test_seq2seq_collator_block_diag_disabled():
    """Verify block_diag_attn=False does not generate PackedSeqParams even with document_ids."""
    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        block_diag_attn=False,
        attn_implementation="fa2",
        compute_dtype=torch.float32,
    )

    examples = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "document_ids": [1, 1, 2, 2],
        },
    ]

    batch = collator(examples)
    assert "packed_seq_params" not in batch
    assert "document_ids" not in batch
    # attention_mask should remain 2D because block_diag_attn=False
    assert batch["attention_mask"].dim() == 2


# ---------------------------------------------------------------------------
# _resolve_attn_implementation_for_mca tests
# ---------------------------------------------------------------------------


def test_resolve_attn_implementation_for_mca_mapping():
    # workflow.py already loaded due to earlier mock, but re-import to be safe
    from llamafactory.train.mca.workflow import _resolve_attn_implementation_for_mca

    assert _resolve_attn_implementation_for_mca(AttentionFunction.FA2) == "flash_attention_2"
    assert _resolve_attn_implementation_for_mca(AttentionFunction.FA3) == "fa3"
    assert _resolve_attn_implementation_for_mca(AttentionFunction.SDPA) == "sdpa"
    assert _resolve_attn_implementation_for_mca(AttentionFunction.DISABLED) == "eager"
    assert _resolve_attn_implementation_for_mca(AttentionFunction.AUTO) == "eager"


# ---------------------------------------------------------------------------
# CustomMcaTrainer._pad_batched_inputs tests
# ---------------------------------------------------------------------------


def test_mca_trainer_pad_batched_inputs_with_packed_seq_params():
    """Verify _pad_batched_inputs moves PackedSeqParams tensors to the correct device."""
    from llamafactory.data.megatron.collator import PackedSeqParams
    from llamafactory.train.mca.trainer import CustomMcaTrainer

    trainer = CustomMcaTrainer.__new__(CustomMcaTrainer)
    trainer._language_input_names = ["input_ids", "attention_mask", "labels"]

    tokenizer = _FakeTokenizer()

    class _MockBatchEncoding(dict):
        def to(self, device):
            return self

    def mock_pad(encoded_inputs, return_tensors=None, **kwargs):
        if isinstance(encoded_inputs, dict):
            keys = list(encoded_inputs.keys())
            num_samples = len(encoded_inputs[keys[0]])
            items = [{k: encoded_inputs[k][i] for k in keys} for i in range(num_samples)]
        else:
            items = encoded_inputs
        max_len = kwargs.get("max_length")
        if max_len is None:
            max_len = max(len(ex["input_ids"]) for ex in items)
        result = {}
        for key in items[0]:
            padded = []
            for ex in items:
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
        return _MockBatchEncoding(result)

    from transformers import PreTrainedTokenizerBase
    mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.padding_side = "right"
    mock_tokenizer.pad = mock_pad
    trainer.processing_class = mock_tokenizer

    mock_args = MagicMock()
    mock_args.device = "cpu"
    trainer.args = mock_args

    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu.clone(),
        max_seqlen_q=2,
        max_seqlen_kv=2,
    )

    inputs = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        "labels": torch.tensor([[1, 2], [3, 4]]),
        "packed_seq_params": psp,
    }

    result = trainer._pad_batched_inputs(inputs, seq_length=6)

    assert "packed_seq_params" in result
    out_psp = result["packed_seq_params"]
    assert out_psp.qkv_format == "thd"
    assert torch.equal(out_psp.cu_seqlens_q, cu)
    assert out_psp.cu_seqlens_q.device.type == "cpu"
    assert out_psp.cu_seqlens_kv.device.type == "cpu"


def test_mca_trainer_pad_batched_inputs_missing_packed_seq_params():
    """Verify _pad_batched_inputs works normally when packed_seq_params is absent."""
    from llamafactory.train.mca.trainer import CustomMcaTrainer

    trainer = CustomMcaTrainer.__new__(CustomMcaTrainer)
    trainer._language_input_names = ["input_ids", "attention_mask", "labels"]

    tokenizer = _FakeTokenizer()

    class _MockBatchEncoding(dict):
        def to(self, device):
            return self

    def mock_pad(encoded_inputs, return_tensors=None, **kwargs):
        if isinstance(encoded_inputs, dict):
            keys = list(encoded_inputs.keys())
            num_samples = len(encoded_inputs[keys[0]])
            items = [{k: encoded_inputs[k][i] for k in keys} for i in range(num_samples)]
        else:
            items = encoded_inputs
        max_len = kwargs.get("max_length", max(len(ex["input_ids"]) for ex in items))
        result = {}
        for key in items[0]:
            padded = []
            for ex in items:
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
        return _MockBatchEncoding(result)

    mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.padding_side = "right"
    mock_tokenizer.pad = mock_pad
    trainer.processing_class = mock_tokenizer

    mock_args = MagicMock()
    mock_args.device = "cpu"
    trainer.args = mock_args

    inputs = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        "labels": torch.tensor([[1, 2], [3, 4]]),
    }

    result = trainer._pad_batched_inputs(inputs, seq_length=6)
    assert "packed_seq_params" not in result
    assert result["input_ids"].shape == (2, 6)
