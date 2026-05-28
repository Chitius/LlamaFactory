from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.get_logger(__name__)


try:
    import grouped_gemm.ops as gg_ops
    _HAS_GROUPED_GEMM = True
except ImportError:
    _HAS_GROUPED_GEMM = False


def _grouped_gemm_forward(
    x: torch.Tensor,            # [num_tokens, hidden] sorted by expert
    w_gate_up: torch.Tensor,    # [num_experts, hidden, 2 * intermediate]
    w_down: torch.Tensor,       # [num_experts, intermediate, hidden]
    tokens_per_expert: torch.Tensor,  # [num_experts]
) -> torch.Tensor:
    """Two grouped GEMMs covering SwiGLU MLP for all experts at once."""
    gate_up = gg_ops.gmm(x, w_gate_up, tokens_per_expert, trans_b=False)
    intermediate = w_gate_up.shape[-1] // 2
    gate, up = gate_up.split(intermediate, dim=-1)
    activated = F.silu(gate) * up
    return gg_ops.gmm(activated, w_down, tokens_per_expert, trans_b=False)


def _patched_deepseek_v3_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for DeepseekV3MoE.forward."""
    identity = hidden_states
    bsz, seq_len, hidden = hidden_states.shape
    flat = hidden_states.view(-1, hidden)

    # Defensive gate unpacking: V3 gate may return 2-tuple or 3-tuple depending on
    # transformers version (older returns (idx, weight, aux_loss), newer returns (idx, weight)).
    gate_out = self.gate(flat)
    if isinstance(gate_out, (tuple, list)) and len(gate_out) >= 3:
        topk_idx, topk_weight, _aux_loss = gate_out[0], gate_out[1], gate_out[2]
    else:
        topk_idx, topk_weight = gate_out[0], gate_out[1]
    topk_weight = topk_weight.to(flat.dtype)

    # Replicate tokens per topk route
    num_tokens = flat.shape[0]
    topk = topk_idx.shape[-1]
    flat_rep = flat.repeat_interleave(topk, dim=0)
    flat_idx = topk_idx.reshape(-1)

    # Sort by expert id
    sorted_idx, sort_perm = flat_idx.sort()
    sorted_x = flat_rep[sort_perm]

    # tokens_per_expert
    n_routed = self.config.n_routed_experts
    tokens_per_expert = torch.bincount(sorted_idx, minlength=n_routed).to(torch.int64)
    # grouped_gemm.ops.gmm requires batch_sizes on CPU
    tokens_per_expert_cpu = tokens_per_expert.cpu()

    # Stack expert weights into (E, ...)
    w_gate_up = torch.stack(
        [torch.cat([e.gate_proj.weight, e.up_proj.weight], dim=0).t() for e in self.experts]
    )  # [E, hidden, 2I]
    w_down = torch.stack([e.down_proj.weight.t() for e in self.experts])  # [E, I, hidden]

    if _HAS_GROUPED_GEMM:
        out_sorted = _grouped_gemm_forward(sorted_x, w_gate_up, w_down, tokens_per_expert_cpu)
    else:
        # Fallback: still vectorized but without true grouped kernel
        out_sorted = torch.zeros_like(sorted_x)
        offset = 0
        intermediate = w_gate_up.shape[-1] // 2
        for e_idx in range(n_routed):
            n = int(tokens_per_expert[e_idx])
            if n == 0:
                continue
            chunk = sorted_x[offset : offset + n]
            gate = chunk @ w_gate_up[e_idx, :, :intermediate]
            up = chunk @ w_gate_up[e_idx, :, intermediate:]
            mlp_out = (F.silu(gate) * up) @ w_down[e_idx]
            out_sorted[offset : offset + n] = mlp_out
            offset += n

    # Unsort back to original order
    out_rep = torch.empty_like(out_sorted)
    out_rep[sort_perm] = out_sorted

    # Reshape and apply topk weights
    out_rep = out_rep.view(num_tokens, topk, hidden)
    weighted = (out_rep * topk_weight.unsqueeze(-1)).sum(dim=1)
    routed_output = weighted.view(bsz, seq_len, hidden)

    # Add shared experts (DeepseekV3 always has shared experts)
    if hasattr(self, "shared_experts") and self.shared_experts is not None:
        routed_output = routed_output + self.shared_experts(identity)

    return routed_output


def patch_deepseek_v3_moe(config: "PretrainedConfig") -> bool:
    """Apply the monkey-patch. Returns True if applied, False otherwise."""
    if getattr(config, "model_type", None) != "deepseek_v3":
        return False

    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
    except ImportError:
        logger.warning_rank0("[grouped_gemm_moe] DeepseekV3MoE not importable; skipping patch.")
        return False

    if getattr(DeepseekV3MoE, "_dsv3_patched", False):
        return True

    DeepseekV3MoE.forward = _patched_deepseek_v3_moe_forward
    DeepseekV3MoE._dsv3_patched = True

    if _HAS_GROUPED_GEMM:
        logger.info_rank0("[grouped_gemm_moe] DeepseekV3MoE.forward patched (grouped_gemm backend)")
    else:
        logger.warning_rank0(
            "[grouped_gemm_moe] DeepseekV3MoE.forward patched (fallback loop). "
            "Install grouped-gemm for full speedup: pip install grouped-gemm"
        )
    return True
