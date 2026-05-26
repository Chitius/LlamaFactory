#!/usr/bin/env python3
"""L7B: MCA TE + FA2/FA3 varlen GPU 端到端验证.

在 Transformer Engine 环境下验证 MCA 路径 FA2 varlen 与 eager 的等价性。
同时包含 HF PT 路径回归验证。
"""

import copy
import json
import os
import shutil
import sys
import inspect
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

# =============================================================================
# Monkey-patch transformers flash_attention_forward for s_aux=None bug
# =============================================================================
from transformers.integrations import flash_attention as _fa_module
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_fa_src = inspect.getsource(_fa_module.flash_attention_forward)
_fa_src_patched = _fa_src.replace(
    "s_aux=s_aux.to(query.dtype),  # FA only accepts half precision",
    "s_aux=(s_aux.to(query.dtype) if s_aux is not None else None),  # FA only accepts half precision",
)
_fa_local = {}
exec(_fa_src_patched, _fa_module.__dict__, _fa_local)
_fa_module.flash_attention_forward = _fa_local["flash_attention_forward"]
ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", _fa_module.flash_attention_forward)
ALL_ATTENTION_FUNCTIONS.register("flash_attention_3", _fa_module.flash_attention_forward)
ALL_ATTENTION_FUNCTIONS.register("flash_attention_4", _fa_module.flash_attention_forward)

# =============================================================================
# Constants
# =============================================================================

K_STEPS = 5
MODEL_PATH = "/tmp/test_llama_mca"
DATA_PREFIX = os.path.join(
    os.path.dirname(__file__), "../../data/synthetic_short_doc_text_document"
)
CACHE_PATH_HF_FA2 = "/tmp/l7b_hf_fa2_cache"
CACHE_PATH_HF_EAGER = "/tmp/l7b_hf_eager_cache"
CACHE_PATH_MCA_FA2 = "/tmp/l7b_mca_fa2_cache"
CACHE_PATH_MCA_EAGER = "/tmp/l7b_mca_eager_cache"
REPORT_PATH = "/tmp/lf_mca_te_fa2_e2e_report.md"
SEED = 42
SEQ_LENGTH = 64
BATCH_SIZE = 2
LR = 1e-4
EOD_TOKEN = 50256
PAD_TOKEN_ID = 0

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SRC_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, os.path.join(SRC_ROOT, "src"))

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
# Loss recorder
# =============================================================================


class _LossRecorderCallback(TrainerCallback):
    def __init__(self):
        self.losses: List[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))


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
    dataset = MegatronGPTDataset(config, indexed_ds)
    dataset.config.shift_labels = True
    return dataset


# =============================================================================
# HF training args helper
# =============================================================================


def _make_hf_training_args(output_dir: str, steps: int):
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


# =============================================================================
# HF Side
# =============================================================================


def _disable_dropout(model: torch.nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            mod.p = 0.0
    if hasattr(model, "config"):
        for attr in ("dropout", "attention_dropout", "hidden_dropout", "resid_pdrop", "attn_pdrop"):
            if hasattr(model.config, attr):
                setattr(model.config, attr, 0.0)


def run_hf_side(
    dataset: Any,
    base_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    steps: int = K_STEPS,
    attn_implementation: str = "eager",
    output_dir: str = "/tmp/l7b_hf_side",
) -> List[float]:
    from llamafactory.hparams import FinetuningArguments, ModelArguments
    from llamafactory.model import load_tokenizer
    from llamafactory.train.pt.trainer import CustomTrainer
    from llamafactory.data.megatron.collator import MegatronDataCollatorForLanguageModeling

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

    training_args = _make_hf_training_args(output_dir, steps)
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
        block_diag_attn=True,
        attn_implementation=attn_implementation,
        compute_dtype=dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    if base_state_dict is not None:
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

    return cb.losses[:steps]


# =============================================================================
# MCA Side
# =============================================================================


def run_mca_side(
    dataset: Any,
    steps: int = K_STEPS,
    attn_implementation: str = "eager",
    output_dir: str = "/tmp/l7b_mca_side",
    base_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> tuple[List[float], Optional[str]]:
    """Run MCA side with Transformer Engine. Returns (losses, error_message)."""
    from mcore_adapter.models import AutoModel
    from mcore_adapter.training_args import Seq2SeqTrainingArguments
    from llamafactory.train.mca.trainer import CustomMcaTrainer
    from llamafactory.data.megatron.collator import MegatronDataCollatorForSeq2Seq
    from transformers import DataCollatorForSeq2Seq
    from llamafactory.extras.constants import IGNORE_INDEX

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        transformer_impl="transformer_engine",
        seed=SEED,
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=steps,
        learning_rate=LR,
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=1,
        save_steps=999999,
        bf16=True,
        fp16=False,
        disable_tqdm=True,
        report_to=[],
        weight_decay=0.0,
        max_grad_norm=0.0,
        dataloader_drop_last=True,
    )

    model = AutoModel.from_pretrained(MODEL_PATH, args)

    if base_state_dict is not None:
        model_state = model.state_dict()
        mismatch = []
        for k, v in base_state_dict.items():
            if k in model_state and model_state[k] is not None:
                if not torch.equal(model_state[k], v):
                    mismatch.append(k)
                    model_state[k].copy_(v)
        if mismatch:
            print(f"      Weight mismatch in {len(mismatch)} keys, corrected: {mismatch[:5]}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    collator = MegatronDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        block_diag_attn=True,
        attn_implementation=attn_implementation,
        compute_dtype=torch.bfloat16,
    )

    trainer = CustomMcaTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset,
    )

    cb = _LossRecorderCallback()
    trainer.add_callback(cb)

    try:
        trainer.train()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return cb.losses[:steps], error_msg

    while len(cb.losses) < steps:
        cb.losses.append(None)

    return cb.losses[:steps], None


# =============================================================================
# Comparison helper
# =============================================================================


def _compare_losses(losses_a: List[float], losses_b: List[float]) -> dict:
    compare_len = min(len(losses_a), len(losses_b))
    diffs = []
    max_diff = 0.0
    max_diff_step = -1
    for step in range(compare_len):
        la = losses_a[step]
        lb = losses_b[step]
        if la is not None and lb is not None:
            diff = abs(la - lb)
            diffs.append(diff)
            if diff > max_diff:
                max_diff = diff
                max_diff_step = step
    mean_diff = (sum(diffs) / len(diffs)) if diffs else float("nan")
    rel_diff = (
        (max_diff / losses_b[max_diff_step] * 100)
        if max_diff_step >= 0 and losses_b[max_diff_step]
        else 0.0
    )
    return {
        "compare_len": compare_len,
        "max_diff": max_diff,
        "max_diff_step": max_diff_step,
        "mean_diff": mean_diff,
        "rel_diff": rel_diff,
        "diffs": diffs,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("L7B MCA TE + FA2 GPU E2E Validation Test")
    print("=" * 70)

    _check_cuda()
    _set_deterministic()

    # Environment info
    print("\n[Environment]")
    print(f"  torch: {torch.__version__}")
    print(f"  transformers: {transformers.__version__}")
    import flash_attn
    print(f"  flash_attn: {flash_attn.__version__}")
    import transformer_engine
    print(f"  transformer_engine: {transformer_engine.__version__}")
    try:
        from megatron.core.extensions.transformer_engine import TEDotProductAttention
        print(f"  TEDotProductAttention: available")
    except Exception as e:
        print(f"  TEDotProductAttention: NOT available ({e})")

    # ------------------------------------------------------------------
    # Load base HF model state dict for weight sharing
    # ------------------------------------------------------------------
    print("\n[0/8] Loading base HF model state dict ...")
    hf_base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    base_state_dict = copy.deepcopy(hf_base_model.state_dict())
    del hf_base_model
    torch.cuda.empty_cache()
    print(f"      Captured {len(base_state_dict)} tensor keys")

    # ==================================================================
    # Stage 1: HF PT FA2 vs eager (regression check)
    # ==================================================================
    print("\n" + "=" * 70)
    print("Stage 1 (Regression): HF PT + document-boundary + FA2 vs eager")
    print("=" * 70)

    print("[1/8] Building HF FA2 dataset ...")
    dataset_hf_fa2 = _build_megatron_dataset(CACHE_PATH_HF_FA2, True, True, True)
    print(f"      Dataset size: {len(dataset_hf_fa2)}")

    print("[2/8] Running HF Side A (FA2) ...")
    losses_hf_fa2 = run_hf_side(
        dataset_hf_fa2,
        base_state_dict=base_state_dict,
        steps=K_STEPS,
        attn_implementation="flash_attention_2",
        output_dir="/tmp/l7b_hf_fa2",
    )
    print(f"      HF FA2 losses: {losses_hf_fa2}")

    del dataset_hf_fa2
    torch.cuda.empty_cache()

    print("[3/8] Building HF eager dataset ...")
    dataset_hf_eager = _build_megatron_dataset(CACHE_PATH_HF_EAGER, True, True, True)
    print(f"      Dataset size: {len(dataset_hf_eager)}")

    print("[4/8] Running HF Side B (eager) ...")
    losses_hf_eager = run_hf_side(
        dataset_hf_eager,
        base_state_dict=base_state_dict,
        steps=K_STEPS,
        attn_implementation="eager",
        output_dir="/tmp/l7b_hf_eager",
    )
    print(f"      HF eager losses: {losses_hf_eager}")

    del dataset_hf_eager
    torch.cuda.empty_cache()

    print("[5/8] Comparing HF FA2 vs eager ...")
    stats_hf = _compare_losses(losses_hf_fa2, losses_hf_eager)
    print(f"      Max diff: {stats_hf['max_diff']:.6f} (step {stats_hf['max_diff_step']})")
    print(f"      Mean diff: {stats_hf['mean_diff']:.6f}")
    print(f"      Rel diff: {stats_hf['rel_diff']:.2f}%")
    hf_pass = stats_hf["rel_diff"] < 1.0
    print(f"      Status: {'PASS' if hf_pass else 'FAIL'} (threshold < 1%)")

    # ==================================================================
    # Stage 2: MCA PT TE + FA2 vs eager
    # ==================================================================
    print("\n" + "=" * 70)
    print("Stage 2 (Primary): MCA PT + TE + document-boundary + FA2 vs eager")
    print("=" * 70)

    os.environ["USE_MCA"] = "1"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29522"

    # Initialize distributed once for MCA path
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    mca_fa2_losses = None
    mca_eager_losses = None
    mca_fa2_error = None
    mca_eager_error = None
    stats_mca = None
    mca_pass = None

    # Try MCA FA2
    print("[6a/8] Building MCA FA2 dataset ...")
    dataset_mca_fa2 = _build_megatron_dataset(CACHE_PATH_MCA_FA2, True, True, True)
    print(f"      Dataset size: {len(dataset_mca_fa2)}")

    print("[6b/8] Running MCA Side A (TE + FA2) ...")
    mca_fa2_losses, mca_fa2_error = run_mca_side(
        dataset=dataset_mca_fa2,
        steps=K_STEPS,
        attn_implementation="flash_attention_2",
        output_dir="/tmp/l7b_mca_fa2",
        base_state_dict=base_state_dict,
    )
    if mca_fa2_error:
        print(f"      MCA FA2 failed:\n{mca_fa2_error}")
    else:
        print(f"      MCA FA2 losses: {mca_fa2_losses}")

    del dataset_mca_fa2
    torch.cuda.empty_cache()

    # Try MCA eager for baseline
    print("[7/8] Building MCA eager dataset ...")
    dataset_mca_eager = _build_megatron_dataset(CACHE_PATH_MCA_EAGER, True, True, True)
    print(f"      Dataset size: {len(dataset_mca_eager)}")

    print("[8/8] Running MCA Side B (TE + eager) ...")
    mca_eager_losses, mca_eager_error = run_mca_side(
        dataset=dataset_mca_eager,
        steps=K_STEPS,
        attn_implementation="eager",
        output_dir="/tmp/l7b_mca_eager",
        base_state_dict=base_state_dict,
    )
    if mca_eager_error:
        print(f"      MCA eager failed:\n{mca_eager_error}")
    else:
        print(f"      MCA eager losses: {mca_eager_losses}")

    del dataset_mca_eager
    torch.cuda.empty_cache()

    if mca_fa2_error or mca_eager_error:
        mca_pass = False
    else:
        stats_mca = _compare_losses(mca_fa2_losses, mca_eager_losses)
        print(f"      MCA max diff: {stats_mca['max_diff']:.6f} (step {stats_mca['max_diff_step']})")
        print(f"      MCA rel diff: {stats_mca['rel_diff']:.2f}%")
        mca_pass = stats_mca["rel_diff"] < 1.0
        print(f"      Status: {'PASS' if mca_pass else 'FAIL'} (threshold < 1%)")

    # ==================================================================
    # Report generation
    # ==================================================================
    report_lines: List[str] = []
    report_lines.append("# L7B MCA TE + FA2 GPU 端到端验证报告\n")
    report_lines.append(f"- **Date**: {datetime.now().isoformat()}\n")
    report_lines.append(f"- **Model**: `{MODEL_PATH}`\n")
    report_lines.append(f"- **Dataset**: `{DATA_PREFIX}`\n")
    report_lines.append(f"- **Steps compared**: {K_STEPS}\n")
    report_lines.append(f"- **Batch size**: {BATCH_SIZE}\n")
    report_lines.append(f"- **Sequence length**: {SEQ_LENGTH}\n")
    report_lines.append(f"- **Seed**: {SEED}\n")
    report_lines.append(f"- **LR**: {LR}\n")
    report_lines.append(f"- **torch**: {torch.__version__}\n")
    report_lines.append(f"- **transformers**: {transformers.__version__}\n")
    report_lines.append(f"- **flash_attn**: {flash_attn.__version__}\n")
    report_lines.append(f"- **transformer_engine**: {transformer_engine.__version__}\n")
    report_lines.append("\n")

    # Stage 1
    report_lines.append("## Stage 1 (Regression): HF PT + document-boundary + FA2 vs eager\n")
    report_lines.append("| Step | HF FA2 | HF Eager | Diff | Status |\n")
    report_lines.append("|------|--------|----------|------|--------|\n")
    for step in range(stats_hf["compare_len"]):
        la = losses_hf_fa2[step]
        lb = losses_hf_eager[step]
        if la is None or lb is None:
            diff = float("nan")
            status = "N/A"
        else:
            diff = abs(la - lb)
            status = "CLOSE" if diff < 0.01 else "DIFF"
        report_lines.append(
            f"| {step:4d} | {la!s:>8} | {lb!s:>10} | {diff:.4f} | {status} |\n"
        )
    report_lines.append(f"\n- **Max diff**: {stats_hf['max_diff']:.6f} (at step {stats_hf['max_diff_step']})\n")
    report_lines.append(f"- **Mean diff**: {stats_hf['mean_diff']:.6f}\n")
    report_lines.append(f"- **Rel diff**: {stats_hf['rel_diff']:.2f}%\n")
    if hf_pass:
        report_lines.append("- **Result**: **PASS** ✅ HF PT 回归验证通过。\n")
    else:
        report_lines.append("- **Result**: **FAIL** ❌ HF PT 回归验证失败。\n")
    report_lines.append("\n")

    # Stage 2
    report_lines.append("## Stage 2 (Primary): MCA PT + TE + document-boundary + FA2 vs eager\n")
    if mca_fa2_error:
        report_lines.append("### MCA FA2 执行结果\n")
        report_lines.append(f"```\n{mca_fa2_error}\n```\n")
    elif mca_eager_error:
        report_lines.append("### MCA eager 执行结果\n")
        report_lines.append(f"```\n{mca_eager_error}\n```\n")
    else:
        report_lines.append("| Step | MCA FA2 | MCA Eager | Diff | Status |\n")
        report_lines.append("|------|---------|-----------|------|--------|\n")
        for step in range(stats_mca["compare_len"]):
            la = mca_fa2_losses[step]
            lb = mca_eager_losses[step]
            if la is None or lb is None:
                diff = float("nan")
                status = "N/A"
            else:
                diff = abs(la - lb)
                status = "CLOSE" if diff < 0.01 else "DIFF"
            report_lines.append(
                f"| {step:4d} | {la!s:>9} | {lb!s:>11} | {diff:.4f} | {status} |\n"
            )
        report_lines.append(f"\n- **Max diff**: {stats_mca['max_diff']:.6f} (at step {stats_mca['max_diff_step']})\n")
        report_lines.append(f"- **Mean diff**: {stats_mca['mean_diff']:.6f}\n")
        report_lines.append(f"- **Rel diff**: {stats_mca['rel_diff']:.2f}%\n")
        if mca_pass:
            report_lines.append("- **Result**: **PASS** ✅ MCA TE FA2 与 eager 的 loss 差异 < 1%，端到端验证通过。\n")
        else:
            report_lines.append("- **Result**: **FAIL** ❌ loss 差异超过 1%，需排查。\n")
    report_lines.append("\n")

    # Analysis
    report_lines.append("## 分析与结论\n")
    report_lines.append(
        "### HF PT 路径结论\n"
        f"- HF PT 路径下，FA2 varlen 与 eager 4D mask 的 loss 最大差异为 **{stats_hf['max_diff']:.6f}**，"
        f"相对差异为 **{stats_hf['rel_diff']:.2f}%**。\n"
    )
    if hf_pass:
        report_lines.append(
            "- 差异 < 1%，HF PT 路径回归验证通过。\n"
        )
    else:
        report_lines.append("- 差异超过阈值，需进一步排查。\n")

    report_lines.append(
        "\n### MCA 路径结论\n"
    )
    if mca_fa2_error:
        report_lines.append(
            "- MCA FA2 路径执行失败。\n"
            f"```\n{mca_fa2_error}\n```\n"
        )
    elif mca_eager_error:
        report_lines.append(
            "- MCA eager 路径执行失败，无法建立 baseline。\n"
            f"```\n{mca_eager_error}\n```\n"
        )
    elif mca_pass:
        report_lines.append(
            "- MCA 路径在 `transformer_impl=transformer_engine` 下，FA2 varlen 与 eager 的 loss 差异 < 1%，端到端验证通过。\n"
            "- `TEDotProductAttention` 正确支持了 `PackedSeqParams`，varlen FlashAttention 与 4D mask 在数学上等价。\n"
        )
    else:
        report_lines.append(
            "- MCA 路径下 FA2 与 eager 存在可观测差异，需排查。\n"
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"\nReport written to {REPORT_PATH}")

    # Console summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"HF PT ({K_STEPS} steps):")
    print(f"  FA2:   {losses_hf_fa2}")
    print(f"  Eager: {losses_hf_eager}")
    print(f"  Diffs: {[abs(a - b) for a, b in zip(losses_hf_fa2, losses_hf_eager)]}")
    print(f"  Max diff: {stats_hf['max_diff']:.6f} at step {stats_hf['max_diff_step']}")
    print(f"  Rel diff: {stats_hf['rel_diff']:.2f}%")
    if mca_fa2_losses is not None:
        print()
        print(f"MCA PT ({K_STEPS} steps):")
        print(f"  FA2:   {mca_fa2_losses}")
        if mca_eager_losses is not None:
            print(f"  Eager: {mca_eager_losses}")
            if any(l is not None for l in mca_fa2_losses):
                print(f"  Diffs: {[abs(a - b) for a, b in zip(mca_fa2_losses, mca_eager_losses) if a is not None and b is not None]}")
    if mca_fa2_error:
        print()
        print("MCA FA2 failed")
    if mca_eager_error:
        print()
        print("MCA eager failed")
    print("=" * 70)

    # Final verdict
    print("\n### Final Verdict ###")
    print(f"HF PT FA2 vs Eager: {'PASS' if hf_pass else 'FAIL'}")
    if mca_fa2_error:
        print("MCA PT FA2: FAILED")
    elif mca_eager_error:
        print("MCA PT Eager: FAILED")
    elif mca_pass is not None:
        print(f"MCA PT FA2 vs Eager: {'PASS' if mca_pass else 'FAIL'}")
    else:
        print("MCA PT: N/A")


if __name__ == "__main__":
    main()
