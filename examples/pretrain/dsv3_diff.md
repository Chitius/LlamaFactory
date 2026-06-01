# DeepSeek-3B (V3) Megatron -> LlamaFactory Diff

Implemented config names:

- `LlamaFactory/examples/pretrain/deepseekv3_megatron.yaml`
- `LlamaFactory/examples/pretrain/deepseekv3_megatron_debug.yaml`

Local architecture config:

- `dsv3_megatron_config/deepseek_3bv3_megatron_like_config.json`

## Architecture and Config Mapping (Sorted by Status)


| Megatron item                                              | Megatron value   | LlamaFactory / HF mapping                                                       | Status     | Notes                                                                       |
| ---------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------- |
| `--max-position-embeddings` / `--seq-length`               | `4096`           | `cutoff_len: 4096`                                                              | mapped     | Runtime training length matched.                                            |
| `--position-embedding-type rope`                           | `rope`           | native in DeepSeek config                                                       | mapped     | DeepSeek V3 uses RoPE by default.                                           |
| `--rotary-base`                                            | `10000`          | native in DeepSeek config                                                       | mapped     | Already in config as `rope_theta`.                                          |
| `--normalization RMSNorm`                                  | `RMSNorm`        | native in DeepSeek config                                                       | mapped     | Preserved in local config.                                                  |
| `--norm-epsilon`                                           | `1e-6`           | `rms_norm_eps: 1e-6`                                                            | mapped     | Preserved in local config.                                                  |
| `--swiglu`                                                 | `true`           | native in DeepSeek config                                                       | mapped     | DeepSeek MLP activation is `silu`/SwiGLU style.                             |
| `--untie-embeddings-and-output-weights`                    | `true`           | `tie_word_embeddings: false`                                                    | mapped     | Preserved in local config.                                                  |
| `--multi-latent-attention`                                 | `true`           | DeepSeek-V3 architecture                                                        | mapped     | Provided by model family implementation.                                    |
| `--moe-router-load-balancing-type`                         | `seq_aux_loss`   | `moe_aux_loss_coef: 1e-4` + `seq_aux: true`                                     | mapped     | Closest exposed LF control + local config flag.                             |
| `--moe-aux-loss-coeff`                                     | `1e-4`           | `moe_aux_loss_coef: 1.0e-4`                                                     | mapped     | Directly supported.                                                         |
| `--moe-layer-recompute`                                    | `true`           | `disable_gradient_checkpointing: false`                                         | mapped     | Closest equivalent recompute behavior.                                      |
| `--qk-head-dim` / `--qk-pos-emb-head-dim` / `--v-head-dim` | `128 / 64 / 128` | `qk_nope_head_dim / qk_rope_head_dim / v_head_dim`                              | mapped     | In local config.                                                            |
| `--use-flash-attn`                                         | `true`           | `flash_attn: fa2`                                                               | mapped     | Explicitly enabled.                                                         |
| `--use-distributed-optimizer`                              | `true`           | `deepspeed: ds_z3_offload_config.json`                                          | mapped     | Closest upstream distributed optimizer strategy.                            |
| `--bf16`                                                   | `true`           | `bf16: true`                                                                    | mapped     | Directly matched.                                                           |
| `--adam-beta1/2`                                           | `0.9 / 0.95`     | `adam_beta1: 0.9`, `adam_beta2: 0.95`                                           | mapped     | Directly matched.                                                           |
| `--weight-decay`                                           | `0.1`            | `weight_decay: 0.1`                                                             | mapped     | Directly matched.                                                           |
| `--clip-grad`                                              | `1.0`            | `max_grad_norm: 1.0`                                                            | mapped     | Directly matched.                                                           |
| `--lr`                                                     | `8.6e-4`         | `learning_rate: 8.6e-4`                                                         | mapped     | Directly matched.                                                           |
| `--min-lr`                                                 | `7e-6`           | `lr_scheduler_type: cosine_with_min_lr` + `lr_scheduler_kwargs: {min_lr: 7e-6}` | mapped     | Uses HF scheduler floor to match Megatron min-LR behavior.                  |
| `--lr-warmup-iters`                                        | `2000`           | `warmup_steps: 2000`                                                            | mapped     | Iter-based warmup matched.                                                  |
| `--train-iters`                                            | `48000`          | `max_steps: 48000`                                                              | mapped     | Direct equivalent in LF.                                                    |
| `--num-layers`                                             | `12`             | `num_hidden_layers: 12` (local config)                                          | mapped     | Achieved via local `config_name_or_path`.                                   |
| `--hidden-size`                                            | `1280`           | `hidden_size: 1280` (local config)                                              | mapped     | Achieved via local config.                                                  |
| `--ffn-hidden-size`                                        | `7168`           | `intermediate_size: 7168` (local config)                                        | mapped     | Achieved via local config.                                                  |
| `--num-attention-heads`                                    | `16`             | `num_attention_heads: 16` (local config)                                        | mapped     | Achieved via local config.                                                  |
| `--kv-channels`                                            | `128`            | implied by qk/v dims in local config                                            | mapped     | Achieved via `qk_nope_head_dim: 128` and `v_head_dim: 128` in local config. |
| `--num-experts`                                            | `64`             | `n_routed_experts: 64` (local config)                                           | mapped     | Routed expert count is directly matched at config level.                    |
| `--moe-layer-freq`                                         | `([0]*1+[1]*11)` | `first_k_dense_replace: 1`, `moe_layer_freq: 1`                                 | mapped     | Equivalent 12-layer pattern is represented in HF DeepSeek config style.     |
| `--moe-ffn-hidden-size`                                    | `896`            | `moe_intermediate_size: 896` (local config)                                     | mapped     | Per-expert FFN width is directly matched.                                   |
| `--moe-shared-expert-intermediate-size`                    | `1792`           | derived from `moe_intermediate_size * n_shared_experts`                         | mapped     | In V3 remote code, shared expert width is computed as `896 * 2 = 1792`.     |
| `--moe-router-topk`                                        | `6`              | `num_experts_per_tok: 6` (local config)                                         | mapped     | Directly matched via DeepSeek router config.                                |
| `--moe-router-topk-scaling-factor`                         | `2.5`            | `routed_scaling_factor: 2.5` (local config)                                     | mapped     | Directly matched via DeepSeek router config.                                |
| `--moe-router-score-function`                              | `sigmoid`        | `scoring_func: sigmoid` (local config)                                          | mapped     | Directly matched via DeepSeek router config.                                |
| `--moe-router-dtype`                                       | `fp32`           | implicit in DeepSeekV3 `MoEGate` (`hidden_states` and gate `weight` cast to `torch.float32` before routing linear) | mapped | Router math is enforced in fp32 in model code; no YAML knob required.       |
| `--q-lora-rank`                                            | `384`            | `q_lora_rank: 384` (local config)                                               | mapped     | Directly matched in local V3 config.                                        |
| `--kv-lora-rank`                                           | `512`            | `kv_lora_rank: 512` (local config)                                              | mapped     | Directly matched in local V3 config.                                        |
| `--qk-layernorm`                                           | `true`           | implicit via DeepSeekV3 `q_a_layernorm`/`kv_a_layernorm` path                  | mapped     | Enabled when `q_lora_rank` and `kv_lora_rank` are set (as in local config). |
| `topk_method`                           | default `noaux_tc` | `topk_method: noaux_tc` in local config                                         | mapped     | Aligned with current DeepSeekV3 codepath requirement (`noaux_tc`). |
| `ep_size (expert parallel)`             | default `1`      | `ep_size` in model config (optional, used by MoE runtime when `>1`)            | partial    | Implemented in model code, but current local config does not set it (effective `1`). |
| Monitoring/profiler flags                                  | many             | `logging_steps`, `plot_loss`, `report_to`                                       | partial    | LF supports base logging, not all Megatron monitor hooks.                   |
| `--use-sandwich-norm`                                      | `true`           | N/A                                                                             | not mapped | Not DSV3 feature.                                       |
| `--attn-post-norm-scale`                                   | `0.03`           | N/A                                                                             | not mapped | Not DSV3 feature.                                                      |
| `--ffn-post-norm-scale`                                    | `0.03`           | N/A                                                                             | not mapped | Not DSV3 feature.                                                      |
| `MTP (num_nextn_predict_layers)`        | default `1`      | config field exists, but no active modeling/training path in this LF codepath  | not mapped | Present in `configuration_deepseek.py`, but not used in current model forward/training flow. |
| `--sequence-parallel`                                      | `true`           | N/A                                                                             | not mapped | Managed by distributed backend.                                               |
| `--cross-entropy-loss-fusion` / TE fusion flags            | mixed            | N/A                                                                             | not mapped | Kernel-fusion internals are not exposed via LF YAML.                        |


## Training Dataset Policy

- **Current runnable configs use** `dataset: c4_demo` (from upstream `LlamaFactory/data/dataset_info.json`) for smoke/pretrain verification.
- **Sequence settings**:
  - main: `cutoff_len: 4096`, `packing: true`
  - debug: `cutoff_len: 512`, `max_samples: 64`, `packing: true`
- **Tokenizer source**: inherited from `model_name_or_path` (`deepseek-ai/DeepSeek-V3-Base` tokenizer assets).
- **If you want Megatron-like production data**: replace `dataset` with your registered corpus key(s) in `dataset_info.json`, keeping the same model config path.

## Distributed Policy

- **Framework**: `torchrun` + upstream LlamaFactory distributed launcher behavior.
- **Memory/distributed optimizer policy**: DeepSpeed ZeRO-3 CPU offload (`examples/deepspeed/ds_z3_offload_config.json`).
- **Precision policy**: bf16 mixed precision (`bf16: true`).
- **Gradient checkpointing**: enabled (`disable_gradient_checkpointing: false`).
- **Single-node 4-GPU example**: `CUDA_VISIBLE_DEVICES=4,5,6,7` with `--nproc_per_node=4`.

## LR Schedule Equivalence

- **Megatron source schedule**:
  - `--lr 8.6e-4`
  - `--lr-warmup-iters 2000`
  - `--train-iters 48000`
  - `--lr-decay-style cosine`
  - `--min-lr 7e-6`
- **LlamaFactory mapping in `examples/pretrain/deepseekv3_megatron.yaml`**:
  - `learning_rate: 8.6e-4`
  - `warmup_steps: 2000`
  - `max_steps: 48000`
  - `lr_scheduler_type: cosine_with_min_lr`
  - `lr_scheduler_kwargs: {min_lr: 7e-6}`
- **Resulting behavior**: linear warmup to peak LR, then cosine decay from peak to `7e-6` floor across remaining steps.

## Run It Yourself

### 1) Debug smoke test (recommended first)

```bash
cd /public/wangyiding/LlamaFactory
source /home/miniconda3/bin/activate llamafactory
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=29667 --nproc_per_node=1 src/train.py examples/pretrain/deepseekv3_megatron_debug.yaml
```

### 2) Multi-GPU debug test (GPUs 4-7)

```bash
cd /public/wangyiding/LlamaFactory
source /home/miniconda3/bin/activate llamafactory
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29677 --nproc_per_node=4 src/train.py examples/pretrain/deepseekv3_megatron_debug.yaml
```

### 3) Main run config (long run)

```bash
cd /public/wangyiding/LlamaFactory
source /home/miniconda3/bin/activate llamafactory
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29687 --nproc_per_node=8 src/train.py examples/pretrain/deepseekv3_megatron.yaml
```

Fineweb log path: `/public/wangyiding/LlamaFactory/saves/deepseekv3_moe/full/logs/pretrain_epoch1_20260529_171340.log`