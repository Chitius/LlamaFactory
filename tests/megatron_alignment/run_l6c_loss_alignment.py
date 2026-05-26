#!/usr/bin/env python3
"""L6c Loss Alignment Test: Llama-Factory vs Megatron-LM training losses
with document-boundary attention mask enabled.

This script runs two identical training loops side-by-side:
  - Side A: Llama-Factory style using MegatronGPTDataset + CustomTrainer
    with reset_attention_mask=True, reset_position_ids=True, eod_mask_loss=True
  - Side B: Megatron-style using GPTDataset + raw PyTorch loop
    with reset_position_ids=True, reset_attention_mask=True, eod_mask_loss=True

Both use:
  - The same model weights (gpt2)
  - The same data order (proven by L4 batch alignment)
  - The same optimizer (AdamW, lr=5.0e-4)
  - Deterministic CUDA / PyTorch settings
  - Document-boundary 4D attention masks
  - Reset position IDs at document boundaries

After training for K steps, step-by-step losses are compared and a report
is written to disk.
"""

import copy
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# Constants
# =============================================================================

K_STEPS = 5
MODEL_PATH = "gpt2"
DATA_PREFIX = os.path.join(os.path.dirname(__file__), "../../data/c4_demo_text_document")
CACHE_PATH_LF = "/tmp/l6c_lf_cache"
CACHE_PATH_MG = "/tmp/l6c_meg_cache"
REPORT_PATH = "/tmp/lf_megatron_l6c_report.md"
MASTER_REPORT_PATH = "/home/public/liuyichuan/plans/lf_megatron_attention_mask_e2e_test_report.md"
SEED = 42
SEQ_LENGTH = 128
NUM_SAMPLES = 200
BATCH_SIZE = 2
LR = 5.0e-4
EOD_TOKEN = 50256

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# =============================================================================
# Determinism
# =============================================================================

def _set_deterministic() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    transformers.set_seed(SEED)


# =============================================================================
# CUDA guard
# =============================================================================

def _check_cuda() -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This test requires a GPU.")
        sys.exit(1)


# =============================================================================
# Megatron stub loader
# =============================================================================

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

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
_Split = sys.modules["megatron.core.datasets.utils"].Split
_FakeTokenizer = sys.modules["megatron.core.tokenizers"].MegatronTokenizerBase

# Override Megatron tokenizer stub with correct EOD for c4_demo (GPT-2)
class _MegatronFakeTokenizer:
    vocab_size = 50000
    eod = EOD_TOKEN
    eos = EOD_TOKEN
    pad = -1
    special_tokens_dict = {}
    unique_identifiers = {}

sys.modules["megatron.core.tokenizers"].MegatronTokenizerBase = _MegatronFakeTokenizer


# =============================================================================
# Helper: disable dropout recursively
# =============================================================================

def _disable_dropout(model: torch.nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            mod.p = 0.0
    if hasattr(model, "config"):
        for attr in ("dropout", "attention_dropout", "hidden_dropout", "resid_pdrop", "attn_pdrop"):
            if hasattr(model.config, attr):
                setattr(model.config, attr, 0.0)


# =============================================================================
# Side A: Llama-Factory
# =============================================================================

def _build_lf_dataset():
    from llamafactory.data.megatron.gpt_dataset import MegatronGPTDataset, MegatronGPTDatasetConfig
    from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset

    if os.path.isdir(CACHE_PATH_LF):
        shutil.rmtree(CACHE_PATH_LF)
    os.makedirs(CACHE_PATH_LF, exist_ok=True)

    indexed_ds = MegatronIndexedDataset(DATA_PREFIX, multimodal=False, mmap=True)
    config = MegatronGPTDatasetConfig(
        path_prefix=DATA_PREFIX,
        seq_length=SEQ_LENGTH,
        seed=SEED,
        num_samples=NUM_SAMPLES,
        data_cache_path=CACHE_PATH_LF,
        add_extra_token=True,
        drop_last_partial_sequence=True,
        split="train",
        pad_token_id=0,
        eod_token_id=EOD_TOKEN,
        reset_attention_mask=True,
        reset_position_ids=True,
        eod_mask_loss=True,
    )
    return MegatronGPTDataset(config, indexed_ds)


def _make_training_args():
    from llamafactory.hparams.training_args import TrainingArguments

    return TrainingArguments(
        output_dir="/tmp/l6c_lf_output",
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        num_train_epochs=1.0,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        bf16=False,
        fp16=False,
        seed=SEED,
        max_steps=K_STEPS,
        logging_steps=1,
        save_steps=999999,
        disable_tqdm=True,
        report_to=[],
        ddp_timeout=180000000,
        weight_decay=0.0,
        max_grad_norm=0.0,
        include_num_input_tokens_seen=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        fp8=False,
    )


class _LossRecorderCallback(transformers.TrainerCallback):
    def __init__(self):
        self.losses: List[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))


def run_lf_side(model: torch.nn.Module, dataset: Any) -> List[float]:
    from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling
    from llamafactory.hparams import FinetuningArguments, ModelArguments
    from llamafactory.model import load_tokenizer
    from llamafactory.train.pt.trainer import CustomTrainer

    model_args = ModelArguments(
        model_name_or_path=MODEL_PATH,
        trust_remote_code=True,
        resize_vocab=False,
        split_special_tokens=False,
        disable_gradient_checkpointing=True,
    )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # Ensure tokenizer has a pad_token_id for the collator
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    training_args = _make_training_args()
    finetuning_args = FinetuningArguments(
        stage="pt",
        finetuning_type="full",
        plot_loss=False,
        disable_shuffling=True,
    )

    collator = MegatronDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        block_diag_attn=True,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=collator,
        train_dataset=dataset,
        **tokenizer_module,
    )

    # Disable num_items_in_batch passing to prevent loss scaling by the model.
    trainer.model_accepts_loss_kwargs = False

    cb = _LossRecorderCallback()
    trainer.add_callback(cb)
    trainer.train()

    while len(cb.losses) < K_STEPS:
        cb.losses.append(None)

    return cb.losses[:K_STEPS]


# =============================================================================
# Side B: Megatron GPTDataset + raw loop
# =============================================================================

def _build_megatron_dataset():
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
        num_samples=NUM_SAMPLES,
        index_split=_Split.train,
        config=config,
    )
    return dataset


def run_megatron_side(model: torch.nn.Module, dataset: Any) -> List[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    min_dtype = torch.finfo(torch.float32).min
    losses: List[float] = []

    for step, batch in enumerate(loader):
        if step >= K_STEPS:
            break

        input_ids = batch["tokens"].cuda()

        # Use unshifted tokens as labels (HF shifts internally).
        # Match LF's EOD and padding masking exactly.
        labels = batch["tokens"].clone().cuda()
        labels[labels == 0] = -100      # padding (Megatron maps pad=-1 -> 0)
        labels[labels == EOD_TOKEN] = -100  # EOD masking

        position_ids = batch["position_ids"].cuda()

        # Megatron attention_mask: [batch_size, 1, seq_len, seq_len] bool
        # True = mask out, False = keep
        # Convert to HF 4D float mask: min_dtype for masked, 0.0 for kept
        mg_mask = batch["attention_mask"].to(torch.bool).cuda()
        hf_mask = torch.where(mg_mask, min_dtype, 0.0)

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=hf_mask,
            position_ids=position_ids,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    return losses


# =============================================================================
# Report helpers
# =============================================================================

def _append_to_master_report(all_pass: bool, lf_losses: List[float], meg_losses: List[float]) -> None:
    status = "PASS" if all_pass else "FAIL"

    lines = [
        "\n---\n\n",
        "## L6c: 训练 Loss 对齐（开启 Document-Boundary Mask）\n\n",
        f"### 测试状态: {status}\n\n",
        "### 配置\n",
        "- Model: gpt2\n",
        f"- Steps: {K_STEPS}\n",
        f"- Batch size: {BATCH_SIZE}\n",
        f"- Seq length: {SEQ_LENGTH}\n\n",
        "### 结果摘要\n\n",
        "| Step | LF Loss | Megatron Loss | Diff | Status |\n",
        "|------|---------|---------------|------|--------|\n",
    ]

    max_diff = 0.0
    max_diff_step = -1
    for step in range(len(lf_losses)):
        lf = lf_losses[step]
        mg = meg_losses[step]
        if lf is None or mg is None:
            diff = float("nan")
            step_status = "N/A"
        else:
            diff = abs(lf - mg)
            step_status = "PASS" if diff < 1e-4 else "FAIL"
            if diff > max_diff:
                max_diff = diff
                max_diff_step = step
        lines.append(f"| {step:4d} | {lf!s:>9} | {mg!s:>13} | {diff:.2e} | {step_status} |\n")

    lines.append("\n")
    lines.append("### 结论\n")
    if all_pass:
        lines.append(
            f"- 前 {K_STEPS} 步 loss diff 均小于 1e-4（最大 diff {max_diff:.2e}，"
            f"出现在 step {max_diff_step}），LF 与 Megatron 在开启 document-boundary mask "
            f"后训练 loss 完全对齐。\n"
        )
    else:
        lines.append(
            f"- 最大 diff {max_diff:.2e}（step {max_diff_step}）超过阈值 1e-4，"
            f"存在不对齐。\n"
        )
    lines.append("\n")

    os.makedirs(os.path.dirname(MASTER_REPORT_PATH), exist_ok=True)
    with open(MASTER_REPORT_PATH, "a") as f:
        f.writelines(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("L6c Loss Alignment Test (Document-Boundary Mask Enabled)")
    print("=" * 70)

    _check_cuda()
    _set_deterministic()

    # ------------------------------------------------------------------
    # Load model once, then clone state dict for both sides
    # ------------------------------------------------------------------
    print("[1/5] Loading base model ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    _disable_dropout(base_model)
    base_state_dict = copy.deepcopy(base_model.state_dict())
    del base_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Side A: Llama-Factory
    # ------------------------------------------------------------------
    print("[2/5] Building LF dataset ...")
    lf_dataset = _build_lf_dataset()

    print("[3/5] Running LF training loop ...")
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    _disable_dropout(model_a)
    model_a.load_state_dict(base_state_dict)
    model_a.cuda()

    lf_losses = run_lf_side(model_a, lf_dataset)
    print(f"      LF losses: {lf_losses}")

    del model_a
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Side B: Megatron raw loop
    # ------------------------------------------------------------------
    print("[4/5] Building Megatron dataset ...")
    meg_dataset = _build_megatron_dataset()

    print("[5/5] Running Megatron training loop ...")
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    _disable_dropout(model_b)
    model_b.load_state_dict(base_state_dict)
    model_b.cuda()

    meg_losses = run_megatron_side(model_b, meg_dataset)
    print(f"      Meg losses: {meg_losses}")

    del model_b
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    compare_len = min(K_STEPS, len(lf_losses), len(meg_losses))
    lf_losses = lf_losses[:compare_len]
    meg_losses = meg_losses[:compare_len]

    diffs = []
    report_lines: List[str] = []
    report_lines.append("# L6c Loss Alignment Report (Document-Boundary Mask)\n")
    report_lines.append(f"- **Date**: {datetime.now().isoformat()}\n")
    report_lines.append(f"- **Model**: `{MODEL_PATH}`\n")
    report_lines.append(f"- **Dataset**: `{DATA_PREFIX}`\n")
    report_lines.append(f"- **Steps compared**: {compare_len}\n")
    report_lines.append(f"- **Batch size**: {BATCH_SIZE}\n")
    report_lines.append(f"- **Sequence length**: {SEQ_LENGTH}\n")
    report_lines.append(f"- **Seed**: {SEED}\n")
    report_lines.append(f"- **Learning rate**: {LR}\n")
    report_lines.append(f"- **Document-boundary mask**: enabled\n")
    report_lines.append(f"- **Reset position IDs**: enabled\n")
    report_lines.append(f"- **EOD mask loss**: enabled\n")
    report_lines.append("\n")
    report_lines.append("| Step | LF Loss | Megatron Loss | Diff | Status |\n")
    report_lines.append("|------|---------|---------------|------|--------|\n")

    all_pass = True
    max_diff = 0.0
    max_diff_step = -1

    for step in range(compare_len):
        lf = lf_losses[step]
        mg = meg_losses[step]
        if lf is None or mg is None:
            status = "N/A"
            diff = float("nan")
        else:
            diff = abs(lf - mg)
            diffs.append(diff)
            status = "PASS" if diff < 1e-4 else "FAIL"
            if status == "FAIL":
                all_pass = False
            if diff > max_diff:
                max_diff = diff
                max_diff_step = step
        report_lines.append(f"| {step:4d} | {lf!s:>9} | {mg!s:>13} | {diff:.2e} | {status} |\n")

    mean_diff = (sum(diffs) / len(diffs)) if diffs else float("nan")

    report_lines.append("\n")
    report_lines.append(f"- **Max diff**: {max_diff:.2e} (at step {max_diff_step})\n")
    report_lines.append(f"- **Mean diff**: {mean_diff:.2e}\n")
    report_lines.append(f"- **Threshold**: 1e-4\n")
    report_lines.append("\n")
    if all_pass:
        report_lines.append("## Result: **PASS** ✅\n")
    else:
        report_lines.append("## Result: **FAIL** ❌\n")

    with open(REPORT_PATH, "w") as f:
        f.writelines(report_lines)
    print(f"\nReport written to {REPORT_PATH}")

    # ------------------------------------------------------------------
    # Append to master report
    # ------------------------------------------------------------------
    _append_to_master_report(all_pass, lf_losses, meg_losses)
    print(f"Master report appended to {MASTER_REPORT_PATH}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"LF losses:      {lf_losses}")
    print(f"Meg losses:     {meg_losses}")
    print(f"Diffs:          {[abs(l - m) for l, m in zip(lf_losses, meg_losses)]}")
    print(f"Max diff:       {max_diff:.2e} at step {max_diff_step}")
    print(f"Status:         {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)

    assert all_pass, f"Max loss diff {max_diff:.2e} exceeds threshold 1e-4"
    assert max_diff < 1e-4, f"Max loss diff {max_diff:.2e} >= 1e-4"


if __name__ == "__main__":
    main()
