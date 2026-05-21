"""L5 Loss Alignment Test: Llama-Factory vs Megatron-LM training losses.

This is the final alignment level (L5). It validates that training losses computed
by Llama-Factory (LF) and Megatron-LM are numerically aligned step-by-step for the
first K training steps on a small model.

================================================================================
MANUAL RUN INSTRUCTIONS
================================================================================

Because L5 requires:
  - A real model (e.g. DeepSeek V3) loaded into GPU memory
  - Megatron-LM training framework fully set up
  - Deterministic CUDA kernels (which may have a performance penalty)

this test is SKIPPED by default in pytest.  To run it manually:

  1. Ensure Megatron-LM is available at the expected path (see MEGATRON_PATH below).

  2. (Optional) Run only the Llama-Factory side to collect LF losses:

       python -m pytest tests/megatron_alignment/test_l5_loss_alignment.py::test_l5_loss_alignment -v --run-l5

     or directly:

       python tests/megatron_alignment/test_l5_loss_alignment.py --run-l5

  3. Run the Megatron side separately (see comments in `run_megatron_training_loop()`)
     and save losses to /tmp/l5_megatron_losses.json.

  4. Re-run this test with both loss files present to get the comparison report.

Environment variables that control the test:
  --run-l5            (pytest custom option) Actually execute the training loops.
  L5_K_STEPS          Number of steps to compare (default: 10).
  L5_MODEL_PATH       Path to the small causal-LM checkpoint (default: deepseek_v3_500m).
  L5_DATA_PREFIX      Path prefix of the Megatron indexed dataset (default: fineweb_edu_50k_text_document).
  L5_OUTPUT_DIR       Where to write the markdown report (default: /tmp/l5_loss_alignment_report.md).
================================================================================
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import transformers

# ---------------------------------------------------------------------------
# Constants & environment overrides (read at runtime, not import time)
# ---------------------------------------------------------------------------


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


K_STEPS = _env_int("L5_K_STEPS", 5)
MODEL_PATH = _env_str("L5_MODEL_PATH", "gpt2")
DATA_PREFIX = _env_str("L5_DATA_PREFIX", os.path.join(os.path.dirname(__file__), "../../data/c4_demo_text_document"))
CACHE_PATH = "/tmp/lf_test_cache_l5"
REPORT_PATH = _env_str("L5_OUTPUT_DIR", "/tmp/l5_loss_alignment_report.md")
LF_LOSS_PATH = "/tmp/l5_lf_losses.json"
MEGATRON_LOSS_PATH = "/tmp/l5_megatron_losses.json"

SEED = 42
SEQ_LENGTH = 128
NUM_SAMPLES = 200
BATCH_SIZE = 2
LR = 5.0e-4

# ---------------------------------------------------------------------------
# 1. Load our implementation
# ---------------------------------------------------------------------------
from llamafactory.data.megatron.gpt_dataset import MegatronGPTDataset, MegatronGPTDatasetConfig
from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.pt.trainer import CustomTrainer
from transformers import DataCollatorForSeq2Seq

# ---------------------------------------------------------------------------
# 2. pytest custom option --run-l5
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--run-l5",
        action="store_true",
        default=False,
        help="Run the L5 loss alignment test (requires GPU + Megatron).",
    )


@pytest.fixture(scope="session")
def run_l5(request):
    return request.config.getoption("--run-l5")


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------


def _cleanup():
    if os.path.isdir(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)
    os.makedirs(CACHE_PATH, exist_ok=True)


def _make_training_args():
    """Build TrainingArguments directly to bypass distributed-check validation."""
    from llamafactory.hparams.training_args import TrainingArguments

    kwargs = {
        "output_dir": "/tmp/l5_lf_output",
        "overwrite_output_dir": True,
        "do_train": True,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "learning_rate": LR,
        "num_train_epochs": 1.0,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "bf16": False,
        "fp16": False,
        "seed": SEED,
        "max_steps": K_STEPS,
        "logging_steps": 1,
        "save_steps": 999999,
        "disable_tqdm": True,
        "report_to": [],
        "ddp_timeout": 180000000,
        "weight_decay": 0.0,
        "max_grad_norm": 0.0,
        "include_num_input_tokens_seen": False,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }
    return TrainingArguments(**kwargs)


def _set_deterministic():
    """Enable deterministic mode for reproducible loss values."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    transformers.set_seed(SEED)


def _build_lf_dataset_and_trainer():
    """Build the LF dataset, model, and trainer for K steps of PT.

    Returns:
        trainer: CustomTrainer instance ready to train.
    """
    from llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments

    model_args = ModelArguments(
        model_name_or_path=MODEL_PATH,
        trust_remote_code=True,
        resize_vocab=False,
        split_special_tokens=False,
    )
    data_args = DataArguments(
        template="deepseek3",
        dataset=None,
        cutoff_len=SEQ_LENGTH,
        max_samples=None,
        overwrite_cache=True,
        preprocessing_num_workers=0,
        dataloader_num_workers=0,
        packing=False,
    )
    training_args = _make_training_args()
    finetuning_args = FinetuningArguments(
        stage="pt",
        finetuning_type="full",
        plot_loss=False,
    )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Build the Megatron dataset directly so we do not need a dataset_info entry.
    indexed_ds = MegatronIndexedDataset(DATA_PREFIX, multimodal=False, mmap=True)
    config = MegatronGPTDatasetConfig(
        path_prefix=DATA_PREFIX,
        seq_length=SEQ_LENGTH,
        seed=SEED,
        num_samples=NUM_SAMPLES,
        data_cache_path=CACHE_PATH,
        add_extra_token=True,
        drop_last_partial_sequence=True,
        split="train",
        pad_token_id=-1,
    )
    train_dataset = MegatronGPTDataset(config, indexed_ds)

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # Disable dropout for determinism
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            mod.p = 0.0
    if hasattr(model, "config"):
        for attr in ("dropout", "attention_dropout", "hidden_dropout", "resid_pdrop", "attn_pdrop"):
            if hasattr(model.config, attr):
                setattr(model.config, attr, 0.0)
    model.train()

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        **tokenizer_module,
    )
    return trainer


# ---------------------------------------------------------------------------
# 4. Megatron side (stub / instructions)
# ---------------------------------------------------------------------------


def run_megatron_training_loop() -> List[float]:
    """Run Megatron-LM training and return per-step losses.

    IMPORTANT
    ---------
    This function is intentionally a **stub**.  Full Megatron-LM engine
    initialization requires:
      - torch.distributed + process group setup
      - megatron.core.initialize.initialize_megatron()
      - model / optimizer / scheduler construction via Megatron APIs
      - A dataset built with megatron.core.datasets.gpt_dataset.GPTDataset

    Because that initialization is highly environment-specific, we provide two
    recommended workflows below.

    ---------------------------------------------------------------------------
    Option A: Minimal standalone Megatron loop (no Megatron engine)
    ---------------------------------------------------------------------------
    If you want to align *only* the data pipeline + PyTorch model forward,
    you can load the SAME model weights in a plain nn.Module, build
    Megatron's GPTDataset, and run a raw PyTorch training loop:

        from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        # Build Megatron GPTDataset (see test_l2_l3_gpt_dataset.py for how to
        # load the reference GPTDataset without installing the megatron package).
        ref_dataset = ...
        loader = DataLoader(ref_dataset, batch_size=BATCH_SIZE, shuffle=False)

        losses = []
        for step, batch in enumerate(loader):
            if step >= K_STEPS:
                break
            outputs = model(input_ids=batch["tokens"], labels=batch["labels"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss))

        with open(MEGATRON_LOSS_PATH, "w") as f:
            json.dump(losses, f)

    ---------------------------------------------------------------------------
    Option B: Run real Megatron-LM pre-training
    ---------------------------------------------------------------------------
    Use the standard Megatron-LM pretrain_gpt.py script with a tiny model
    config that matches the HF checkpoint (DeepSeek V3 500M).  Ensure:
      --seed 42
      --lr 5.0e-4
      --min-lr 5.0e-4
      --lr-decay-style constant
      --disable-bias-linear
      --no-bias-gelu-fusion
      --no-bias-dropout-fusion
      --no-masked-softmax-fusion
      --no-gradient-accumulation-fusion
      --sequence-parallel-size 1
      --tensor-model-parallel-size 1
      --pipeline-model-parallel-size 1

    After training, extract per-step losses from the Megatron logs and write
    them to:

        /tmp/l5_megatron_losses.json

    as a JSON list of floats, e.g. [4.1234, 3.9876, ...].

    Returns:
        List of loss values for steps 0..K_STEPS-1, read from disk if available.
    """
    if os.path.exists(MEGATRON_LOSS_PATH):
        with open(MEGATRON_LOSS_PATH) as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# 5. Loss recording callback
# ---------------------------------------------------------------------------


class _LossRecorderCallback(transformers.TrainerCallback):
    """Simple callback that records the training loss after each logging step."""

    def __init__(self):
        self.losses: List[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))


# ---------------------------------------------------------------------------
# 6. Main test
# ---------------------------------------------------------------------------


def test_l5_loss_alignment(run_l5: bool):
    """Compare LF and Megatron step-by-step training losses for K steps.

    Skipped unless ``--run-l5`` is passed to pytest.
    """
    if not run_l5 and not os.environ.get("L5_FORCE_RUN"):
        pytest.skip(
            "L5 requires GPU and full Megatron training setup. "
            "Pass --run-l5 to pytest (or set L5_FORCE_RUN=1) to execute."
        )

    # ------------------------------------------------------------------
    # Guard: we need at least one of the loss files, and ideally the model
    # ------------------------------------------------------------------
    if not os.path.exists(LF_LOSS_PATH) and not torch.cuda.is_available():
        pytest.skip("L5: No cached LF losses and CUDA unavailable. Cannot generate LF losses on this node.")

    # ==================================================================
    # LF SIDE
    # ==================================================================
    lf_losses: List[float] = []
    if os.path.exists(LF_LOSS_PATH):
        print(f"[L5] Loading cached LF losses from {LF_LOSS_PATH}")
        with open(LF_LOSS_PATH) as f:
            lf_losses = json.load(f)
    else:
        print("[L5] Running Llama-Factory training loop ...")
        _cleanup()
        _set_deterministic()

        trainer = _build_lf_dataset_and_trainer()
        cb = _LossRecorderCallback()
        trainer.add_callback(cb)

        # Train for exactly K_STEPS (max_steps is already set in args)
        trainer.train()
        lf_losses = cb.losses

        # Pad if logging gave fewer entries (e.g. first log may be delayed)
        while len(lf_losses) < K_STEPS:
            lf_losses.append(None)

        with open(LF_LOSS_PATH, "w") as f:
            json.dump(lf_losses, f)
        print(f"[L5] LF losses saved to {LF_LOSS_PATH}")

    # ==================================================================
    # MEGATRON SIDE
    # ==================================================================
    megatron_losses: List[float] = run_megatron_training_loop()

    if not megatron_losses:
        pytest.skip(
            "L5: Megatron losses not available. "
            f"Please produce {MEGATRON_LOSS_PATH} and re-run."
        )

    # Ensure same length
    compare_len = min(K_STEPS, len(lf_losses), len(megatron_losses))
    lf_losses = lf_losses[:compare_len]
    megatron_losses = megatron_losses[:compare_len]

    # ==================================================================
    # COMPARISON & REPORT
    # ==================================================================
    diffs = []
    report_lines: List[str] = []
    report_lines.append("# L5 Loss Alignment Report\n")
    report_lines.append(f"- **Date**: {datetime.now().isoformat()}\n")
    report_lines.append(f"- **Model**: `{MODEL_PATH}`\n")
    report_lines.append(f"- **Dataset**: `{DATA_PREFIX}`\n")
    report_lines.append(f"- **Steps compared**: {compare_len}\n")
    report_lines.append(f"- **Batch size**: {BATCH_SIZE}\n")
    report_lines.append(f"- **Sequence length**: {SEQ_LENGTH}\n")
    report_lines.append(f"- **Seed**: {SEED}\n")
    report_lines.append(f"- **Learning rate**: {LR}\n")
    report_lines.append("\n")
    report_lines.append("| Step | LF Loss | Megatron Loss | Diff | Status |\n")
    report_lines.append("|------|---------|---------------|------|--------|\n")

    all_pass = True
    for step in range(compare_len):
        lf = lf_losses[step]
        mg = megatron_losses[step]
        if lf is None or mg is None:
            status = "N/A"
            diff = float("nan")
        else:
            diff = abs(lf - mg)
            diffs.append(diff)
            status = "PASS" if diff < 1e-4 else "FAIL"
            if status == "FAIL":
                all_pass = False
        report_lines.append(f"| {step:4d} | {lf!s:>9} | {mg!s:>13} | {diff:.2e} | {status} |\n")

    max_diff = max(diffs) if diffs else float("nan")
    mean_diff = (sum(diffs) / len(diffs)) if diffs else float("nan")

    report_lines.append("\n")
    report_lines.append(f"- **Max diff**: {max_diff:.2e}\n")
    report_lines.append(f"- **Mean diff**: {mean_diff:.2e}\n")
    report_lines.append(f"- **Threshold**: 1e-4\n")
    report_lines.append("\n")
    if all_pass:
        report_lines.append("## Result: **PASS** ✅\n")
    else:
        report_lines.append("## Result: **FAIL** ❌\n")

    with open(REPORT_PATH, "w") as f:
        f.writelines(report_lines)
    print(f"[L5] Report written to {REPORT_PATH}")

    # Final assertion
    assert all_pass, f"Max loss diff {max_diff:.2e} exceeds threshold 1e-4"
    assert max_diff < 1e-4, f"Max loss diff {max_diff:.2e} >= 1e-4"


# ---------------------------------------------------------------------------
# 7. CLI entrypoint for manual execution without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L5 Loss Alignment Test (manual runner)")
    parser.add_argument("--run-l5", action="store_true", help="Actually execute training loops.")
    parser.add_argument("--k-steps", type=int, default=K_STEPS, help="Number of steps to compare.")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to HF model checkpoint.")
    parser.add_argument("--data-prefix", type=str, default=DATA_PREFIX, help="Path prefix of Megatron indexed dataset.")
    args = parser.parse_args()

    if args.run_l5:
        os.environ["L5_FORCE_RUN"] = "1"
    if args.k_steps != K_STEPS:
        os.environ["L5_K_STEPS"] = str(args.k_steps)
    if args.model_path != MODEL_PATH:
        os.environ["L5_MODEL_PATH"] = args.model_path
    if args.data_prefix != DATA_PREFIX:
        os.environ["L5_DATA_PREFIX"] = args.data_prefix

    test_l5_loss_alignment(run_l5=args.run_l5)
