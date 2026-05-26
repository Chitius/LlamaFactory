#!/usr/bin/env python
"""Verify labels semantics for MCA PT with document-boundary enabled."""

import sys
sys.path.insert(0, "/home/public/liuyichuan/playground/LlamaFactory/src")

import torch
from llamafactory.data.megatron.gpt_dataset import MegatronGPTDataset, MegatronGPTDatasetConfig
from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset
from transformers import AutoTokenizer

# Load tokenizer to get pad/eos token ids
tokenizer = AutoTokenizer.from_pretrained("/tmp/test_llama_mca", trust_remote_code=False)
print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
print(f"Tokenizer bos_token_id: {tokenizer.bos_token_id}")

# Build dataset with document-boundary enabled
config = MegatronGPTDatasetConfig(
    path_prefix="/home/public/liuyichuan/playground/LlamaFactory/data/c4_demo_text_document",
    seq_length=128,
    seed=42,
    pad_token_id=tokenizer.pad_token_id,
    reset_attention_mask=True,
    reset_position_ids=True,
    eod_mask_loss=True,
    eod_token_id=tokenizer.eos_token_id,
    shift_labels=True,
    add_extra_token=True,
)

indexed_dataset = MegatronIndexedDataset("/home/public/liuyichuan/playground/LlamaFactory/data/c4_demo_text_document")
dataset = MegatronGPTDataset(config, indexed_dataset)

print(f"Dataset length: {len(dataset)}")
print()

# Examine first 3 samples
for idx in range(min(3, len(dataset))):
    sample = dataset[idx]
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    document_ids = sample["document_ids"]
    position_ids = sample.get("position_ids")

    print(f"===== Sample {idx} =====")
    print(f"input_ids shape: {input_ids.shape}, labels shape: {labels.shape}")
    print(f"document_ids unique: {torch.unique(document_ids).tolist()}")
    if position_ids is not None:
        print(f"position_ids first 10: {position_ids[:10].tolist()}")

    # Verify shift: labels[i] should == input_ids[i+1]
    # But only where both are valid (not -100, not pad)
    shifted_match = []
    mismatch_positions = []
    for i in range(len(input_ids) - 1):
        if labels[i] != -100 and input_ids[i + 1] != tokenizer.pad_token_id:
            match = (labels[i] == input_ids[i + 1]).item()
            shifted_match.append(match)
            if not match:
                mismatch_positions.append(i)

    if shifted_match:
        shift_acc = sum(shifted_match) / len(shifted_match)
        print(f"Shift correctness: {shift_acc:.4f} ({sum(shifted_match)}/{len(shifted_match)})")
        if mismatch_positions:
            print(f"  Mismatch positions (first 5): {mismatch_positions[:5]}")
    else:
        print("Shift correctness: N/A (no valid positions)")

    # Verify padding positions are -100
    pad_positions = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
    if len(pad_positions) > 0:
        pad_labels = labels[pad_positions]
        pad_masked = (pad_labels == -100).all().item()
        print(f"Padding mask: {pad_masked} ({len(pad_positions)} pad positions)")
        if not pad_masked:
            bad = pad_positions[pad_labels != -100][:5].tolist()
            print(f"  Bad padding positions: {bad}")
    else:
        print("Padding mask: N/A (no padding in this sample)")

    # Verify EOD positions are -100
    eod_positions = (labels == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eod_positions) > 0:
        eod_masked = False  # labels should be -100 at EOD, so we shouldn't find eos_token_id in labels
        print(f"EOD mask: FAIL (found {len(eod_positions)} labels == eos_token_id)")
        print(f"  EOD positions: {eod_positions[:10].tolist()}")
    else:
        print("EOD mask: PASS (no labels == eos_token_id, i.e., all masked to -100)")

    # Print first 10 tokens for inspection
    print(f"input_ids  first 20: {input_ids[:20].tolist()}")
    print(f"labels     first 20: {labels[:20].tolist()}")
    print(f"document_ids first 20: {document_ids[:20].tolist()}")
    print()

print("Labels semantics verification complete.")
