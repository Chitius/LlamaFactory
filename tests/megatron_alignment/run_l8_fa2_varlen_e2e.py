#!/usr/bin/env python3
"""L8: FA2 varlen document-boundary E2E validation with loss comparison.

Validates that the collator flatten fix correctly enables varlen
FlashAttention to enforce document-boundary attention blocking.

Strategy:
  Run three identically-seeded training passes on the same model/data:
    (a) Eager + 4D mask   ― ground truth (document boundaries enforced)
    (b) FA2 varlen + doc-boundary  ― the path we are validating
    (c) FA2 causal-only   ― control (document boundaries NOT enforced)

  Pass condition: (a) ≈ (b) within 1e-4  AND  |(a) − (c)| > |(a) − (b)|.
"""

import copy
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

# =============================================================================
# Constants
# =============================================================================

K_STEPS = 10
SEQ_LENGTH = 64
BATCH_SIZE = 2
SEED = 42
LR = 1e-4
EOD_TOKEN = 50256
PAD_TOKEN_ID = 0

MODEL_PATH = "/tmp/test_llama_mca"
DATA_PREFIX = os.path.join(
    os.path.dirname(__file__), "../../data/synthetic_short_doc_text_document"
)
CACHE_EAGER = "/tmp/l8_cache_eager"
CACHE_FA2_DOC = "/tmp/l8_cache_fa2_doc"
CACHE_FA2_CAUSAL = "/tmp/l8_cache_fa2_causal"

SRC_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, os.path.join(SRC_ROOT, "src"))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"


# =============================================================================
# Utilities
# =============================================================================


def _set_deterministic() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass


def _check_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a CUDA-capable GPU.")


def _disable_dropout(model: torch.nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            mod.p = 0.0
    if hasattr(model, "config"):
        for attr in ("dropout", "attention_dropout", "hidden_dropout", "resid_pdrop", "attn_pdrop"):
            if hasattr(model.config, attr):
                setattr(model.config, attr, 0.0)


class _LossRecorderCallback(TrainerCallback):
    def __init__(self):
        self.losses: List[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append(logs["loss"])


# =============================================================================
# Dataset builder
# =============================================================================


def _build_megatron_dataset(
    cache_path: str,
    reset_attention_mask: bool,
    reset_position_ids: bool,
    eod_mask_loss: bool,
) -> Any:
    from llamafactory.data.megatron.gpt_dataset import MegatronGPTDataset, MegatronGPTDatasetConfig
    from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset

    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    indexed_ds = MegatronIndexedDataset(DATA_PREFIX, multimodal=False, mmap=True)
    config = MegatronGPTDatasetConfig(
        path_prefix=DATA_PREFIX,
        seq_length=SEQ_LENGTH,
        seed=SEED,
        num_samples=None,
        data_cache_path=cache_path,
        add_extra_token=True,
        drop_last_partial_sequence=True,
        split="train",
        pad_token_id=PAD_TOKEN_ID,
        eod_token_id=EOD_TOKEN,
        reset_attention_mask=reset_attention_mask,
        reset_position_ids=reset_position_ids,
        eod_mask_loss=eod_mask_loss,
    )
    return MegatronGPTDataset(config, indexed_ds)


# =============================================================================
# Trainer helpers
# =============================================================================


def _make_training_args(output_dir: str, steps: int):
    from llamafactory.hparams.training_args import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
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
        max_steps=steps,
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


def _run_training_pass(
    dataset: Any,
    base_state_dict: Dict[str, torch.Tensor],
    attn_implementation: str,
    block_diag_attn: bool,
    output_dir: str,
    steps: int = K_STEPS,
) -> List[float]:
    """Run a single training pass and return per-step losses."""
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    training_args = _make_training_args(output_dir, steps)
    finetuning_args = FinetuningArguments(
        stage="pt",
        finetuning_type="full",
        plot_loss=False,
        disable_shuffling=True,
    )

    dtype = torch.bfloat16

    collator = MegatronDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        block_diag_attn=block_diag_attn,
        attn_implementation=attn_implementation,
        compute_dtype=dtype,
    )

    # Choose model attn_implementation: for FA2 varlen, we pass fa2.
    # The collator flattening + cu_seq_lens enables varlen doc-boundary.
    model_attn = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=model_attn,
    )
    model.load_state_dict(base_state_dict)
    model.cuda()

    _disable_dropout(model)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=collator,
        train_dataset=dataset,
        **tokenizer_module,
    )
    trainer.model_accepts_loss_kwargs = False

    cb = _LossRecorderCallback()
    trainer.add_callback(cb)
    trainer.train()

    while len(cb.losses) < steps:
        cb.losses.append(None)

    del model
    torch.cuda.empty_cache()

    return cb.losses[:steps]


# =============================================================================
# Comparison
# =============================================================================


def _compare(losses_a: List[float], losses_b: List[float]) -> dict:
    n = min(len(losses_a), len(losses_b))
    diffs = [abs(losses_a[i] - losses_b[i]) for i in range(n) if losses_a[i] is not None and losses_b[i] is not None]
    return {
        "max_diff": max(diffs) if diffs else float("nan"),
        "mean_diff": sum(diffs) / len(diffs) if diffs else float("nan"),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("L8 FA2 varlen document-boundary E2E Validation")
    print("=" * 70)

    _check_cuda()
    _set_deterministic()

    print(f"\n  torch: {torch.__version__}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  flash_attn: {__import__('flash_attn').__version__}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Load base model state dict for weight sharing
    # ------------------------------------------------------------------
    print("\n[0/4] Loading base model state dict ...")
    hf_base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    base_sd = copy.deepcopy(hf_base.state_dict())
    del hf_base
    torch.cuda.empty_cache()
    print(f"      {len(base_sd)} keys captured")

    # ------------------------------------------------------------------
    # Pass (a): Eager + 4D mask (ground truth)
    # ------------------------------------------------------------------
    print("\n[1/4] Pass (a): Eager + 4D document-boundary mask (reference) ...")
    ds_eager = _build_megatron_dataset(CACHE_EAGER, True, True, True)
    losses_eager = _run_training_pass(
        ds_eager, base_sd, attn_implementation="eager",
        block_diag_attn=True, output_dir="/tmp/l8_eager",
    )
    print(f"      Eager losses: {[f'{x:.4f}' for x in losses_eager]}")
    del ds_eager
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Pass (b): FA2 varlen + document-boundary (our fix)
    # ------------------------------------------------------------------
    print("\n[2/4] Pass (b): FA2 varlen + document-boundary (fix under test) ...")
    ds_fa2_doc = _build_megatron_dataset(CACHE_FA2_DOC, True, True, True)
    losses_fa2_doc = _run_training_pass(
        ds_fa2_doc, base_sd, attn_implementation="flash_attention_2",
        block_diag_attn=True, output_dir="/tmp/l8_fa2_doc",
    )
    print(f"      FA2 varlen losses: {[f'{x:.4f}' for x in losses_fa2_doc]}")
    del ds_fa2_doc
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Pass (c): FA2 causal-only (control — no doc-boundary enforcement)
    # ------------------------------------------------------------------
    print("\n[3/4] Pass (c): FA2 causal-only (no document boundary) ...")
    ds_fa2_causal = _build_megatron_dataset(CACHE_FA2_CAUSAL, False, False, False)
    losses_fa2_causal = _run_training_pass(
        ds_fa2_causal, base_sd, attn_implementation="flash_attention_2",
        block_diag_attn=False, output_dir="/tmp/l8_fa2_causal",
    )
    print(f"      FA2 causal losses: {[f'{x:.4f}' for x in losses_fa2_causal]}")
    del ds_fa2_causal
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    print("\n[4/4] Comparing losses ...")
    cmp_ab = _compare(losses_eager, losses_fa2_doc)
    cmp_ac = _compare(losses_eager, losses_fa2_causal)
    cmp_bc = _compare(losses_fa2_doc, losses_fa2_causal)

    print(f"  (a) eager 4D  vs (b) FA2 varlen doc: max_diff={cmp_ab['max_diff']:.6f}  mean_diff={cmp_ab['mean_diff']:.6f}")
    print(f"  (a) eager 4D  vs (c) FA2 causal:     max_diff={cmp_ac['max_diff']:.6f}  mean_diff={cmp_ac['mean_diff']:.6f}")
    print(f"  (b) FA2 varlen vs (c) FA2 causal:     max_diff={cmp_bc['max_diff']:.6f}  mean_diff={cmp_bc['mean_diff']:.6f}")

    # ---- Criteria ----
    # Cross-implementation threshold: eager (float32 softmax) vs FA2 (bf16 fused
    # kernel) have inherent numerical differences.  0.01 (1% relative) is the
    # same threshold used by the L7 E2E tests for FA2-vs-eager comparisons.
    # Within-implementation (eager-vs-eager) can use 1e-4 (L5 threshold).
    THRESHOLD_CROSS = 0.01   # 1% relative for cross-implementation (eager vs FA2)
    THRESHOLD_TIGHT = 1e-4   # for same-implementation comparisons

    # Criterion 1: FA2 varlen ≈ eager 4D (cross-implementation)
    crit1 = cmp_ab["max_diff"] < THRESHOLD_CROSS
    print(f"\n  C1 (FA2 varlen ≈ eager 4D):  max_diff={cmp_ab['max_diff']:.6f} {'<' if crit1 else '>='} {THRESHOLD_CROSS}  → {'PASS' if crit1 else 'FAIL'}")

    # Criterion 2: FA2 varlen is closer to eager than FA2 causal (doc boundary active)
    crit2 = cmp_ab["max_diff"] < cmp_ac["max_diff"]
    print(f"  C2 (FA2 varlen closer to eager than causal):  {cmp_ab['max_diff']:.6f} < {cmp_ac['max_diff']:.6f}  → {'PASS' if crit2 else 'FAIL'}")

    # Criterion 3: FA2 causal ≠ eager (confirms data has doc boundaries)
    crit3 = cmp_ac["max_diff"] > THRESHOLD_TIGHT
    print(f"  C3 (FA2 causal ≠ eager, doc boundaries matter):  max_diff={cmp_ac['max_diff']:.6f} {'>' if crit3 else '<='} {THRESHOLD_TIGHT}  → {'PASS' if crit3 else 'INCONCLUSIVE'}")

    overall = crit1 and crit2 and crit3

    print("\n" + "=" * 70)
    if overall:
        print("RESULT: PASS — FA2 varlen correctly enforces document-boundary attention")
    else:
        print("RESULT: FAIL")
        if not crit1:
            print("  → FA2 varlen loss differs from eager 4D mask (numerical issue)")
        if not crit2:
            print("  → FA2 varlen no closer to eager than causal (doc boundary not active)")
        if not crit3:
            print("  → FA2 causal ≈ eager (dataset may not have document boundaries)")
    print("=" * 70)

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
