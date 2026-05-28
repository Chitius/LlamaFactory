from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)


try:
    import grouped_gemm.ops as gg_ops
    _HAS_GROUPED_GEMM = True
except ImportError:
    _HAS_GROUPED_GEMM = False


# Cache for the dynamic AddAuxiliaryLoss class.  Captured at patch time from the
# same module the DeepseekV2MoE class came from, so it works for both the
# trust_remote_code path and the transformers-builtin path.
_AUX_LOSS_FN = None


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


def _patched_deepseek_v2_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for DeepseekV2MoE.forward.

    Supports both gate conventions:
      - trust_remote_code: gate expects 2-D, returns (idx, w, aux_loss)
      - builtin transformers: gate expects 3-D, returns (idx, w)
    Passing 3-D works for both (the trust_remote_code variant flattens internally).
    """
    identity = hidden_states  # for shared_experts
    bsz, seq_len, hidden = hidden_states.shape

    # Defensive gate unpacking: V2 gate may return 2-tuple or 3-tuple depending on
    # whether trust_remote_code or builtin transformers is used.
    gate_out = self.gate(hidden_states)
    if isinstance(gate_out, (tuple, list)) and len(gate_out) >= 3:
        topk_idx, topk_weight, aux_loss = gate_out[0], gate_out[1], gate_out[2]
    else:
        topk_idx, topk_weight = gate_out[0], gate_out[1]
        aux_loss = None

    # Normalize topk_idx/topk_weight to flat [B*S, topk]
    if topk_idx.dim() == 3:
        topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
        topk_weight = topk_weight.reshape(-1, topk_weight.shape[-1])

    flat = hidden_states.view(-1, hidden)
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

    # Re-attach aux_loss for trust_remote_code path
    if self.training and aux_loss is not None and _AUX_LOSS_FN is not None:
        routed_output = _AUX_LOSS_FN.apply(routed_output, aux_loss)

    # Add shared experts iff configured
    if getattr(self.config, "n_shared_experts", None) is not None and hasattr(self, "shared_experts"):
        routed_output = routed_output + self.shared_experts(identity)

    return routed_output


def _resolve_v2_moe_class(model: "PreTrainedModel"):
    """Find the DeepseekV2MoE class by walking the model — handles both
    trust_remote_code (dynamic transformers_modules.*) and transformers-builtin paths.
    Returns (cls, module_name) or (None, None)."""
    for module in model.modules():
        cls = type(module)
        if cls.__name__ == "DeepseekV2MoE":
            return cls, cls.__module__
    return None, None


def patch_deepseek_v2_moe(model: "PreTrainedModel") -> bool:
    """Apply the monkey-patch. Must be called *after* model instantiation."""
    global _AUX_LOSS_FN

    if getattr(model.config, "model_type", None) != "deepseek_v2":
        return False

    cls, mod_name = _resolve_v2_moe_class(model)
    if cls is None:
        logger.warning_rank0("[grouped_gemm_moe_v2] DeepseekV2MoE not found on model; skipping patch.")
        return False

    if getattr(cls, "_dsv2_patched", False):
        return True

    # Capture AddAuxiliaryLoss from the same module so .apply works regardless
    # of whether the class came from transformers builtin or trust_remote_code.
    try:
        import importlib
        owning_module = importlib.import_module(mod_name)
        _AUX_LOSS_FN = getattr(owning_module, "AddAuxiliaryLoss", None)
    except Exception as e:
        logger.warning_rank0(f"[grouped_gemm_moe_v2] could not resolve AddAuxiliaryLoss from {mod_name}: {e}")
        _AUX_LOSS_FN = None

    cls.forward = _patched_deepseek_v2_moe_forward
    cls._dsv2_patched = True

    aux_tag = "with AddAuxiliaryLoss" if _AUX_LOSS_FN is not None else "no aux-loss hook"
    if _HAS_GROUPED_GEMM:
        logger.info_rank0(
            f"[grouped_gemm_moe_v2] DeepseekV2MoE.forward patched "
            f"(grouped_gemm backend, {aux_tag}, source={mod_name})"
        )
    else:
        logger.warning_rank0(
            f"[grouped_gemm_moe_v2] DeepseekV2MoE.forward patched "
            f"(fallback loop, {aux_tag}, source={mod_name}). "
            "Install grouped-gemm for full speedup: pip install grouped-gemm"
        )
    return True
