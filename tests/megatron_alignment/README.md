# Megatron-LM Alignment Tests

This directory contains a tiered alignment test suite that validates Llama-Factory's
Megatron-compatible data pipeline against the reference Megatron-LM implementation.

## Overview

| Level | Name | What it checks | Needs GPU | Needs Megatron-LM |
|-------|------|----------------|-----------|-------------------|
| **L0** | Trainer sampler integration | `get_dataset()` sets `disable_shuffling` and `CustomTrainer` uses `SequentialSampler` | No | No |
| **L1** | IndexedDataset alignment | `__getitem__`, document indices, sequence lengths | No | No (reference code loaded directly) |
| **L2** | GPTDataset index alignment | `document_index`, `sample_index`, `shuffle_index` | No | No (reference code loaded directly) |
| **L3** | GPTDataset sample alignment | Raw tokens and `input_ids`/`labels` from `__getitem__` | No | No (reference code loaded directly) |
| **L4** | Batch-level alignment | First N batches via `DataLoader` are identical | No | No (reference code loaded directly) |
| **L5** | **Loss alignment** | Step-by-step training losses match within `1e-4` | **Yes** | **Yes (or loss file)** |

> **Note:** L1-L4 load Megatron-LM reference source files directly via `importlib`
> (no package install or GPU required). L5 requires a real model and training loop.

---

## Prerequisites

### All levels

- Python 3.10+
- Llama-Factory installed (or `src/` on `PYTHONPATH`)
- PyTorch (CPU build is sufficient for L1-L4)
- NumPy

### L1-L4 only

- Megatron-LM source code available at the default relative path:
  `../../../Megatron-LM` (from `tests/megatron_alignment/`)
  - Override via `MEGATRON_LM_ROOT` environment variable if your copy lives elsewhere.
- The compiled helpers extension:
  `megatron/core/datasets/helpers_cpp.cpython-311-x86_64-linux-gnu.so`
- A `.bin` + `.idx` indexed dataset:
  `data/c4_demo_text_document` (generated from `data/c4_demo.jsonl` with GPT-2 tokenizer)

### L5 only

- **GPU** (CUDA-capable, enough VRAM for a ~500 M parameter model)
- Full Llama-Factory training dependencies (`transformers`, `accelerate`, etc.)
- **Megatron-LM training framework** set up (or pre-generated loss file)
- A small causal-LM checkpoint, default:
  `gpt2` (HuggingFace built-in, ~124 M parameters)

---

## Running the tests

### L1-L4 (CI-friendly)

Run all alignment levels automatically:

```bash
python -m pytest tests/megatron_alignment/ -v
```

Run a specific level:

```bash
python -m pytest tests/megatron_alignment/test_l0_trainer_sampler.py -v
python -m pytest tests/megatron_alignment/test_l1_indexed_dataset.py -v
python -m pytest tests/megatron_alignment/test_l2_l3_gpt_dataset.py -v
python -m pytest tests/megatron_alignment/test_l4_batch_alignment.py -v
```

### L5 (manual only)

L5 is **skipped by default** because it needs GPU + Megatron-LM.

#### Option A: pytest with `--run-l5`

```bash
python -m pytest tests/megatron_alignment/test_l5_loss_alignment.py -v --run-l5
```

#### Option B: direct Python execution

```bash
python tests/megatron_alignment/test_l5_loss_alignment.py --run-l5 \
    --k-steps 10 \
    --model-path /path/to/your/500m_model \
    --data-prefix /path/to/your/dataset_prefix
```

#### Option C: separate LF / Megatron runs

1. Generate LF losses first (the script caches them to `/tmp/l5_lf_losses.json`):
   ```bash
   python tests/megatron_alignment/test_l5_loss_alignment.py --run-l5
   ```

2. Generate Megatron losses independently and save to:
   ```
   /tmp/l5_megatron_losses.json
   ```
   (see the docstring in `test_l5_loss_alignment.py::run_megatron_training_loop()`
   for two recommended approaches).

3. Re-run the test to produce the comparison report:
   ```bash
   python tests/megatron_alignment/test_l5_loss_alignment.py --run-l5
   ```

The markdown report is written to `/tmp/l5_loss_alignment_report.md` by default.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `L5_K_STEPS` | `5` | Number of training steps to compare |
| `L5_MODEL_PATH` | `gpt2` | HF model checkpoint name or path |
| `L5_DATA_PREFIX` | `data/c4_demo_text_document` | Megatron `.bin`/`.idx` prefix |
| `L5_OUTPUT_DIR` | `/tmp/l5_loss_alignment_report.md` | Markdown report path |
| `L5_FORCE_RUN` | (unset) | Set to `1` to bypass the `--run-l5` flag check |

---

## Expected results

- **L1-L4**: Must pass with 100 % exact match (no tolerance).
- **L5**: Loss difference per step should be `< 1e-4`. Small deviations are expected
due to floating-point ordering, fused optimizers, or minor implementation differences
between HF and Megatron model code. If diffs exceed `1e-4`, inspect the report to
identify the first failing step and check LR schedule / weight-init / dropout settings.
