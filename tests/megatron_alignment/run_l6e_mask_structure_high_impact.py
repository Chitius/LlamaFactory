#!/usr/bin/env python3
"""L6e: Mask Structure Direct Validation on High-Impact Synthetic Short Documents.

This script performs a direct mathematical verification of attention mask structures
between Llama-Factory (LF) and Megatron-LM on the synthetic_short_doc dataset,
without relying on loss comparison or model training.

Verification targets:
  A. LF 4D mask vs Megatron 2D mask element-wise equivalence (non-padding)
  B. Mask semantic correctness (cross-document blocked, causal preserved)
  C. Position IDs reset at document boundaries
  D. Labels correctly mask padding and EOD tokens

The synthetic_short_doc dataset guarantees:
  - 200 documents, each 20 tokens + EOD(50256)
  - 100% samples span >= 4 documents
  - ~45-50% tokens sit near document boundaries
"""

import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC_ROOT = os.path.join(PROJECT_ROOT, "..")
DATA_PREFIX = os.path.join(SRC_ROOT, "data", "synthetic_short_doc_text_document")
CACHE_PATH_LF = "/tmp/lf_test_cache_l6e_lf"
CACHE_PATH_MG = "/tmp/lf_test_cache_l6e_mg"
REPORT_PATH = "/tmp/lf_megatron_l6e_report.md"

# =============================================================================
# Constants
# =============================================================================

SEQ_LENGTH = 64
SEED = 42
EOD_TOKEN = 50256
PAD_TOKEN_ID = 0

# =============================================================================
# Load LF modules
# =============================================================================

sys.path.insert(0, os.path.join(SRC_ROOT, "src"))

from llamafactory.data.collator import prepare_4d_attention_mask
from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling
from llamafactory.data.megatron.gpt_dataset import MegatronGPTDataset, MegatronGPTDatasetConfig
from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset

# =============================================================================
# Load Megatron-LM reference code via stubs
# =============================================================================

from utils import (
    build_megatron_stubs,
    check_megatron_source,
    find_helpers_cpp,
    load_megatron_module,
)

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
_megatron_gpt_mod = load_megatron_module(
    "megatron.core.datasets.gpt_dataset",
    os.path.join(_megatron_dir, "megatron/core/datasets/gpt_dataset.py"),
)
_GPTDataset = _megatron_gpt_mod.GPTDataset
_GPTDatasetConfig = _megatron_gpt_mod.GPTDatasetConfig
_get_ltor_masks_and_position_ids = _megatron_gpt_mod._get_ltor_masks_and_position_ids
_Split = sys.modules["megatron.core.datasets.utils"].Split

# Override Megatron tokenizer stub with correct EOD for synthetic data
class _MegatronFakeTokenizer:
    vocab_size = 50000
    eod = EOD_TOKEN
    eos = EOD_TOKEN
    pad = -1
    special_tokens_dict = {}
    unique_identifiers = {}

sys.modules["megatron.core.tokenizers"].MegatronTokenizerBase = _MegatronFakeTokenizer


# =============================================================================
# Fake tokenizer for collator
# =============================================================================

class _FakeTokenizer:
    """Minimal tokenizer stub for collator tests."""

    def __init__(self):
        self.pad_token_id = PAD_TOKEN_ID
        self.eos_token_id = EOD_TOKEN
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


# =============================================================================
# Dataset builders
# =============================================================================

def _cleanup_caches() -> None:
    for p in [CACHE_PATH_LF, CACHE_PATH_MG]:
        if os.path.isdir(p):
            shutil.rmtree(p)


def build_lf_dataset() -> MegatronGPTDataset:
    _cleanup_caches()
    os.makedirs(CACHE_PATH_LF, exist_ok=True)

    indexed_ds = MegatronIndexedDataset(DATA_PREFIX, multimodal=False, mmap=True)
    config = MegatronGPTDatasetConfig(
        path_prefix=DATA_PREFIX,
        seq_length=SEQ_LENGTH,
        seed=SEED,
        num_samples=None,
        data_cache_path=CACHE_PATH_LF,
        add_extra_token=True,
        drop_last_partial_sequence=True,
        split="train",
        pad_token_id=PAD_TOKEN_ID,
        eod_token_id=EOD_TOKEN,
        reset_attention_mask=True,
        reset_position_ids=True,
        eod_mask_loss=True,
    )
    return MegatronGPTDataset(config, indexed_ds)


def build_megatron_dataset() -> Any:
    _cleanup_caches()
    os.makedirs(CACHE_PATH_MG, exist_ok=True)

    indexed_ds = sys.modules["megatron.core.datasets.indexed_dataset"].IndexedDataset(
        DATA_PREFIX, multimodal=False, mmap=True
    )
    config = _GPTDatasetConfig(
        random_seed=SEED,
        sequence_length=SEQ_LENGTH,
        blend=([DATA_PREFIX], None),
        split="1,0,0",
        path_to_cache=CACHE_PATH_MG,
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
        dataset_path=DATA_PREFIX,
        indexed_indices=indices,
        num_samples=None,
        index_split=_Split.train,
        config=config,
    )
    return dataset


# =============================================================================
# Validation helpers
# =============================================================================

def validate_mask_semantics(
    mask_4d: torch.Tensor,
    document_ids: torch.Tensor,
    sample_idx: int,
) -> Tuple[bool, List[str], Dict[str, int]]:
    """Validate mask semantics for a single sample.

    Returns:
        (pass, error_messages, counters)
    """
    min_dtype = torch.finfo(mask_4d.dtype).min
    seq_len = document_ids.size(0)
    errors: List[str] = []
    counters = {
        "cross_doc_blocked": 0,
        "same_doc_causal": 0,
        "same_doc_future": 0,
        "padding_blocked": 0,
    }

    for i in range(seq_len):
        for j in range(seq_len):
            is_pad_i = document_ids[i] == 0
            is_pad_j = document_ids[j] == 0
            masked = mask_4d[0, 0, i, j] == min_dtype

            if is_pad_j:
                # Padding key must be masked for all queries
                counters["padding_blocked"] += 1
                if not masked:
                    errors.append(
                        f"Sample {sample_idx}: padding key j={j} not masked for query i={i}"
                    )
                continue

            if is_pad_i:
                # Padding query should also be masked (typically all padding rows are masked)
                if not masked:
                    errors.append(
                        f"Sample {sample_idx}: padding query i={i} not masked for key j={j}"
                    )
                continue

            same_doc = document_ids[i] == document_ids[j]
            if not same_doc:
                counters["cross_doc_blocked"] += 1
                if not masked:
                    errors.append(
                        f"Sample {sample_idx}: cross-doc attend allowed: i={i}(doc={document_ids[i]}) -> j={j}(doc={document_ids[j]})"
                    )
            else:
                if i >= j:
                    counters["same_doc_causal"] += 1
                    if masked:
                        errors.append(
                            f"Sample {sample_idx}: same-doc causal blocked: i={i} -> j={j}"
                        )
                else:
                    counters["same_doc_future"] += 1
                    if not masked:
                        errors.append(
                            f"Sample {sample_idx}: same-doc future allowed: i={i} -> j={j}"
                        )

    return len(errors) == 0, errors, counters


def validate_position_ids(
    position_ids: torch.Tensor,
    document_ids: torch.Tensor,
    sample_idx: int,
) -> Tuple[bool, List[str]]:
    """Validate position_ids reset at document boundaries."""
    errors: List[str] = []
    non_pad = document_ids != 0
    if not torch.any(non_pad):
        return True, errors

    doc_ids_np = document_ids[non_pad].numpy()
    pos_ids_np = position_ids[non_pad].numpy()

    boundaries = np.where(np.diff(doc_ids_np, prepend=doc_ids_np[0]))[0]
    for b in boundaries:
        if b > 0 and b < len(pos_ids_np):
            if pos_ids_np[b] != 0:
                errors.append(
                    f"Sample {sample_idx}: position_ids not reset at boundary {b}: got {pos_ids_np[b]}"
                )

    for doc_id in np.unique(doc_ids_np):
        segment_pos = pos_ids_np[doc_ids_np == doc_id]
        expected = np.arange(len(segment_pos))
        if not np.array_equal(segment_pos, expected):
            errors.append(
                f"Sample {sample_idx}: position_ids not consecutive in doc {doc_id}: "
                f"got {segment_pos.tolist()}, expected {expected.tolist()}"
            )

    return len(errors) == 0, errors


def validate_labels(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    document_ids: torch.Tensor,
    sample_idx: int,
) -> Tuple[bool, List[str]]:
    """Validate labels mask padding and EOD correctly."""
    errors: List[str] = []
    pad_mask = document_ids == 0
    if torch.any(pad_mask):
        if not torch.all(labels[pad_mask] == -100):
            errors.append(f"Sample {sample_idx}: padding labels not all -100")

    eod_mask = input_ids == EOD_TOKEN
    if torch.any(eod_mask):
        if not torch.all(labels[eod_mask] == -100):
            errors.append(
                f"Sample {sample_idx}: EOD labels not all -100 (found {labels[eod_mask].tolist()})"
            )

    non_pad = ~pad_mask
    if torch.any(non_pad):
        # Non-padding labels should NOT be -100 unless they are EOD
        non_pad_non_eod = non_pad & (input_ids != EOD_TOKEN)
        if torch.any(non_pad_non_eod):
            if not torch.all(labels[non_pad_non_eod] != -100):
                errors.append(
                    f"Sample {sample_idx}: non-padding non-EOD labels incorrectly masked to -100"
                )

    return len(errors) == 0, errors


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("L6e Mask Structure High-Impact Validation")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    print("[1/4] Building LF dataset ...")
    dataset_lf = build_lf_dataset()
    print(f"      LF samples: {len(dataset_lf)}")

    print("[2/4] Building Megatron dataset ...")
    dataset_mg = build_megatron_dataset()
    print(f"      MG samples: {len(dataset_mg)}")

    num_samples = min(len(dataset_lf), len(dataset_mg))
    print(f"      Samples to validate: {num_samples}")

    # ------------------------------------------------------------------
    # Build collator
    # ------------------------------------------------------------------
    print("[3/4] Initializing collator ...")
    tokenizer = _FakeTokenizer()
    collator = MegatronDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------
    print("[4/4] Running validation ...")
    min_dtype = torch.finfo(torch.float32).min

    all_pass = True
    mask_mismatch_samples: List[int] = []
    semantic_fail_samples: List[int] = []
    pos_id_fail_samples: List[int] = []
    label_fail_samples: List[int] = []

    total_cross_doc_blocked = 0
    total_same_doc_causal = 0
    total_same_doc_future = 0
    total_padding_blocked = 0
    total_boundaries = 0

    report_lines: List[str] = [
        "# L6e Mask Structure High-Impact Validation Report\n",
        f"- **Date**: {datetime.now().isoformat()}\n",
        f"- **Dataset**: `{DATA_PREFIX}`\n",
        f"- **Config**: seq_length={SEQ_LENGTH}, seed={SEED}, eod_token={EOD_TOKEN}\n",
        f"- **Samples validated**: {num_samples}\n",
        "\n",
    ]

    for idx in range(num_samples):
        # LF side ------------------------------------------------------
        item_lf = dataset_lf[idx]
        document_ids = item_lf["document_ids"]
        position_ids = item_lf["position_ids"]
        input_ids = item_lf["input_ids"]
        labels = item_lf["labels"]

        # LF collator -> 4D mask
        example = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in item_lf.items()}
        batch = collator([example])
        mask_lf = batch["attention_mask"]  # (1, 1, seq_len, seq_len)

        # Megatron side ------------------------------------------------
        item_mg = dataset_mg[idx]
        tokens = item_mg["tokens"]

        # Compute reference mask via Megatron native function
        mask_mg_bool, loss_mask_mg, position_ids_mg = _get_ltor_masks_and_position_ids(
            tokens, EOD_TOKEN, True, True, True, True
        )
        mask_mg = torch.where(mask_mg_bool.unsqueeze(0), min_dtype, 0.0)

        # A. Mask element-wise equivalence -------------------------------
        non_pad_mask = document_ids != 0
        non_pad_indices = torch.where(non_pad_mask)[0]

        mask_match = True
        if len(non_pad_indices) > 0:
            mask_lf_sub = mask_lf[0, 0][
                non_pad_indices[:, None], non_pad_indices[None, :]
            ]
            mask_mg_sub = mask_mg[0, 0][
                non_pad_indices[:, None], non_pad_indices[None, :]
            ]
            mask_match = torch.equal(mask_lf_sub, mask_mg_sub)

        if not mask_match:
            all_pass = False
            mask_mismatch_samples.append(idx)

        # B. Mask semantics validation ----------------------------------
        semantic_pass, semantic_errors, counters = validate_mask_semantics(
            mask_lf, document_ids, idx
        )
        total_cross_doc_blocked += counters["cross_doc_blocked"]
        total_same_doc_causal += counters["same_doc_causal"]
        total_same_doc_future += counters["same_doc_future"]
        total_padding_blocked += counters["padding_blocked"]

        # Count document boundaries
        doc_ids_np = document_ids.numpy()
        non_pad_doc_ids = doc_ids_np[doc_ids_np != 0]
        if len(non_pad_doc_ids) > 0:
            total_boundaries += len(np.where(np.diff(non_pad_doc_ids))[0])

        if not semantic_pass:
            all_pass = False
            semantic_fail_samples.append(idx)
            for err in semantic_errors[:3]:  # cap detailed output
                report_lines.append(f"- {err}\n")

        # C. Position IDs validation ------------------------------------
        pos_pass, pos_errors = validate_position_ids(position_ids, document_ids, idx)
        if not pos_pass:
            all_pass = False
            pos_id_fail_samples.append(idx)
            for err in pos_errors[:3]:
                report_lines.append(f"- {err}\n")

        # D. Labels validation ------------------------------------------
        label_pass, label_errors = validate_labels(labels, input_ids, document_ids, idx)
        if not label_pass:
            all_pass = False
            label_fail_samples.append(idx)
            for err in label_errors[:3]:
                report_lines.append(f"- {err}\n")

        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == num_samples - 1:
            status = "PASS" if (mask_match and semantic_pass and pos_pass and label_pass) else "FAIL"
            print(f"      Sample {idx + 1:3d}/{num_samples:3d}: {status}")

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    report_lines.extend(
        [
            "\n",
            "## Statistics\n",
            f"- **Samples validated**: {num_samples}\n",
            f"- **Document boundaries crossed**: {total_boundaries}\n",
            f"- **Cross-document (i,j) pairs blocked**: {total_cross_doc_blocked}\n",
            f"- **Same-document causal (i,j) pairs kept**: {total_same_doc_causal}\n",
            f"- **Same-document future (i,j) pairs blocked**: {total_same_doc_future}\n",
            f"- **Padding (i,j) pairs blocked**: {total_padding_blocked}\n",
            "\n",
            "## Results\n",
            f"- **Mask equivalence failures**: {len(mask_mismatch_samples)}\n",
            f"- **Mask semantic failures**: {len(semantic_fail_samples)}\n",
            f"- **Position ID failures**: {len(pos_id_fail_samples)}\n",
            f"- **Label failures**: {len(label_fail_samples)}\n",
            "\n",
        ]
    )

    if mask_mismatch_samples:
        report_lines.append(f"- **Mask mismatch samples**: {mask_mismatch_samples}\n")
    if semantic_fail_samples:
        report_lines.append(f"- **Semantic fail samples**: {semantic_fail_samples}\n")
    if pos_id_fail_samples:
        report_lines.append(f"- **Position ID fail samples**: {pos_id_fail_samples}\n")
    if label_fail_samples:
        report_lines.append(f"- **Label fail samples**: {label_fail_samples}\n")

    report_lines.append("\n")
    if all_pass:
        report_lines.append("## Overall Result: **PASS** ✅\n")
    else:
        report_lines.append("## Overall Result: **FAIL** ❌\n")

    with open(REPORT_PATH, "w") as f:
        f.writelines(report_lines)

    print(f"\nReport written to {REPORT_PATH}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Samples validated:          {num_samples}")
    print(f"Document boundaries:        {total_boundaries}")
    print(f"Cross-doc blocked pairs:    {total_cross_doc_blocked}")
    print(f"Same-doc causal pairs:      {total_same_doc_causal}")
    print(f"Same-doc future pairs:      {total_same_doc_future}")
    print(f"Padding blocked pairs:      {total_padding_blocked}")
    print(f"Mask mismatches:            {len(mask_mismatch_samples)}")
    print(f"Semantic failures:          {len(semantic_fail_samples)}")
    print(f"Position ID failures:       {len(pos_id_fail_samples)}")
    print(f"Label failures:             {len(label_fail_samples)}")
    print(f"Status:                     {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert len(mask_mismatch_samples) == 0, (
        f"Mask mismatch in {len(mask_mismatch_samples)} / {num_samples} samples: "
        f"{mask_mismatch_samples}"
    )
    assert len(semantic_fail_samples) == 0, (
        f"Mask semantic failures in {len(semantic_fail_samples)} / {num_samples} samples: "
        f"{semantic_fail_samples}"
    )
    assert len(pos_id_fail_samples) == 0, (
        f"Position ID failures in {len(pos_id_fail_samples)} / {num_samples} samples: "
        f"{pos_id_fail_samples}"
    )
    assert len(label_fail_samples) == 0, (
        f"Label failures in {len(label_fail_samples)} / {num_samples} samples: "
        f"{label_fail_samples}"
    )
    assert num_samples >= 20, f"Only {num_samples} samples available, expected at least 20"
    assert total_cross_doc_blocked > 0, "No cross-document pairs were validated"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
