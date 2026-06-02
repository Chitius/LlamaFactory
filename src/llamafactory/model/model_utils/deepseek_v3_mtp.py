# model/model_utils/deepseek_v3_mtp.py

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast


try:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3DecoderLayer,
        DeepseekV3ForCausalLM,
        DeepseekV3RMSNorm,
    )
except ImportError:
    # For older remote-code style DeepSeek-V3 repos.
    from transformers.models.deepseek_v3.modeling_deepseek import (  # type: ignore
        DeepseekV3DecoderLayer,
        DeepseekV3ForCausalLM,
        DeepseekV3RMSNorm,
    )


IGNORE_INDEX = -100


class DeepseekV3MTPModule(nn.Module):
    """One sequential MTP module.

    For depth k:

        concat([E(t_{i+k}), h_i^{k-1}])
            -> projection
            -> one DeepSeek-V3 decoder block
            -> final RMSNorm
            -> h_i^k

        h_i^k -> shared lm_head -> predict t_{i+k+1}

    This follows the Megatron-style MTP concat order:
        [future token embedding, previous hidden states]
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config

        self.hnorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        # Use a new DeepSeek-V3 decoder layer as the MTP transformer block.
        # layer_idx >= num_hidden_layers makes it MoE if first_k_dense_replace is small.
        self.block = DeepseekV3DecoderLayer(config, layer_idx=layer_idx)

        # Megatron-style MTP applies a final norm after the MTP transformer block
        # before the shared output head.
        self.final_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        prev_hidden_states: torch.Tensor,
        future_embeds: torch.Tensor,
        rotary_emb: nn.Module,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = prev_hidden_states.shape

        # Megatron concat order:
        #   concat([E(t_{i+k}), h_i^{k-1}])
        #
        # Note:
        #   Previous version used [hnorm(prev_hidden_states), enorm(future_embeds)].
        #   This version uses [enorm(future_embeds), hnorm(prev_hidden_states)].
        future_embeds = self.enorm(future_embeds)
        prev_hidden_states = self.hnorm(prev_hidden_states)

        hidden_states = torch.cat(
            [
                future_embeds,
                prev_hidden_states,
            ],
            dim=-1,
        )
        hidden_states = self.eh_proj(hidden_states)

        if position_ids is None:
            position_ids = (
                torch.arange(
                    seq_len,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(bsz, -1)
            )
        else:
            if position_ids.shape[-1] != seq_len:
                raise ValueError(f"MTP position_ids length {position_ids.shape[-1]} does not match seq_len {seq_len}.")

        position_embeddings = rotary_emb(hidden_states, position_ids=position_ids)
        cache_position = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        hidden_states = self.block(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            use_cache=False,
        )

        # Be robust to decoder layers that return either a tensor or a tuple.
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        # Megatron-style final norm after the MTP transformer block.
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class DeepseekV3ForCausalLMMTP(DeepseekV3ForCausalLM):
    """DeepseekV3ForCausalLM + sequential MTP training objective.

    During training:
        loss = main_lm_loss + mtp_loss_weight * avg_mtp_loss

    During generation / inference:
        if labels is None, MTP loss is skipped and the model behaves like normal CausalLM.
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_nextn_predict_layers = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
        self.mtp_loss_weight = float(getattr(config, "mtp_loss_weight", 0.1))

        self.mtp_modules = nn.ModuleList(
            [
                DeepseekV3MTPModule(
                    config=config,
                    layer_idx=config.num_hidden_layers + i,
                )
                for i in range(self.num_nextn_predict_layers)
            ]
        )

        # Initialize newly added MTP params.
        self.post_init()

        # For lightweight logging/debugging.
        self.last_main_loss = None
        self.last_mtp_loss = None
        self.last_mtp_acc = None
        self.last_mtp_correct = None
        self.last_mtp_total = None

    def _compute_mtp_loss(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        num_items_in_batch: Optional[torch.Tensor | int] = None,
    ) -> torch.Tensor:
        mtp_losses = []
        prev_hidden_states = hidden_states

        batch_size, seq_len = input_ids.shape
        vocab_size = self.config.vocab_size
        mtp_correct = hidden_states.new_zeros((), dtype=torch.float32)
        mtp_total = hidden_states.new_zeros((), dtype=torch.float32)

        for depth, mtp_module in enumerate(self.mtp_modules, start=1):
            # MTP depth k:
            #   h_i^{k-1} + E(t_{i+k}) -> h_i^k -> predict t_{i+k+1}
            target_len = seq_len - depth - 1
            if target_len <= 0:
                break

            # h_i^{k-1}, positions i = 0 ... target_len - 1
            h_ctx = prev_hidden_states[:, :target_len, :]

            # E(t_{i+k}), positions i+k = depth ... depth + target_len - 1
            future_ids = input_ids[:, depth : depth + target_len]
            future_embeds = self.model.embed_tokens(future_ids)

            # We intentionally keep the fixed/shifted position_ids behavior:
            # depth=1: future token positions are 1,2,3,...
            # depth=2: future token positions are 2,3,4,...
            #
            # This does NOT reproduce the old Megatron roll-position bug.
            if position_ids is not None:
                mtp_position_ids = position_ids[:, depth : depth + target_len]
            else:
                mtp_position_ids = (
                    torch.arange(
                        depth,
                        depth + target_len,
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

            # The MTP sequence index still corresponds to h_i, so the padding mask
            # follows valid h positions i = 0 ... target_len - 1.
            if attention_mask is not None:
                mtp_attention_mask = attention_mask[:, :target_len]
            else:
                mtp_attention_mask = None

            mtp_hidden_states = mtp_module(
                prev_hidden_states=h_ctx,
                future_embeds=future_embeds,
                rotary_emb=self.model.rotary_emb,
                attention_mask=mtp_attention_mask,
                position_ids=mtp_position_ids,
            )

            # h_i^k predicts t_{i+k+1}
            mtp_logits = self.lm_head(mtp_hidden_states)
            mtp_labels = labels[:, depth + 1 : depth + 1 + target_len]

            # ===== MTP shift debug: print once per depth per model instance =====
            if not hasattr(self, "_mtp_shift_debug_done_depths"):
                self._mtp_shift_debug_done_depths = set()

            if depth not in self._mtp_shift_debug_done_depths:
                self._mtp_shift_debug_done_depths.add(depth)

                try:
                    import torch.distributed as dist

                    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
                except Exception:
                    is_rank0 = True

                if is_rank0:
                    b = 0
                    max_show = min(8, target_len)

                    print("\n[MTP SHIFT DEBUG]", flush=True)
                    print(f"seq_len      : {seq_len}", flush=True)
                    print(f"depth        : {depth}", flush=True)
                    print(f"target_len   : {target_len}", flush=True)

                    print(
                        "input_ids pos:",
                        list(range(0, max_show + depth + 2)),
                        flush=True,
                    )
                    print(
                        "input_ids    :",
                        input_ids[b, : max_show + depth + 2].detach().cpu().tolist(),
                        flush=True,
                    )

                    print(
                        "h positions  :",
                        list(range(0, max_show)),
                        flush=True,
                    )
                    print(
                        "future pos   :",
                        list(range(depth, depth + max_show)),
                        flush=True,
                    )
                    print(
                        "future_ids   :",
                        future_ids[b, :max_show].detach().cpu().tolist(),
                        flush=True,
                    )
                    print(
                        "mtp pos ids  :",
                        mtp_position_ids[b, :max_show].detach().cpu().tolist(),
                        flush=True,
                    )
                    print(
                        "label pos    :",
                        list(range(depth + 1, depth + 1 + max_show)),
                        flush=True,
                    )
                    print(
                        "mtp_labels   :",
                        mtp_labels[b, :max_show].detach().cpu().tolist(),
                        flush=True,
                    )
                    print("[/MTP SHIFT DEBUG]\n", flush=True)
            # ==================================================================

            flat_logits = mtp_logits.reshape(-1, vocab_size)
            flat_labels = mtp_labels.reshape(-1).to(flat_logits.device)
            valid_labels = flat_labels.ne(IGNORE_INDEX)
            if torch.any(valid_labels):
                # Restrict the CE graph to supervised tokens. Ignored/padded MTP
                # positions can otherwise poison gradients if their activations
                # become non-finite in an attention backend.
                mtp_loss = F.cross_entropy(
                    flat_logits[valid_labels].float(),
                    flat_labels[valid_labels],
                    reduction="sum" if num_items_in_batch is not None else "mean",
                )
                if num_items_in_batch is not None:
                    if torch.is_tensor(num_items_in_batch):
                        num_items_in_batch = num_items_in_batch.to(mtp_loss.device)

                    # Match HuggingFace causal LM loss scaling. Trainer may pass
                    # the globally gathered main-token count in DDP, then scale
                    # the returned loss before backward/logging.
                    mtp_loss = mtp_loss / num_items_in_batch

                with torch.no_grad():
                    mtp_preds = flat_logits[valid_labels].argmax(dim=-1)
                    mtp_targets = flat_labels[valid_labels]
                    mtp_correct = mtp_correct + mtp_preds.eq(mtp_targets).sum().to(mtp_correct.dtype)
                    mtp_total = mtp_total + valid_labels.sum().to(mtp_total.dtype)
            else:
                # Keep this depth connected to autograd and DDP even when a
                # short batch has no target for the requested prediction depth.
                mtp_loss = torch.nan_to_num(flat_logits).sum() * 0.0

            mtp_losses.append(mtp_loss)

            # Sequential causal chain:
            # depth k+1 uses h_i^k, not the original main-model h_i^0.
            prev_hidden_states = mtp_hidden_states

        if len(mtp_losses) == 0:
            return hidden_states.new_zeros(()), mtp_correct, mtp_total

        # DeepSeek / Megatron style: average over MTP depths.
        return torch.stack(mtp_losses).mean(), mtp_correct, mtp_total

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if labels is not None and input_ids is None:
            raise ValueError("MTP training requires input_ids because it uses future token embeddings.")

        # Training should not use KV cache.
        # This avoids bad interactions with gradient checkpointing / Trainer behavior.
        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # During training, compute full logits for normal shifted LM loss.
        # During generation, keep HuggingFace's logits_to_keep behavior.
        if labels is not None:
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            main_loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

            mtp_loss_weight = float(os.getenv("LLAMAFACTORY_MTP_LOSS_WEIGHT", self.mtp_loss_weight))
            if self.num_nextn_predict_layers > 0 and mtp_loss_weight != 0.0:
                mtp_loss, mtp_correct, mtp_total = self._compute_mtp_loss(
                    input_ids=input_ids,
                    labels=labels,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    num_items_in_batch=kwargs.get("num_items_in_batch"),
                )
            else:
                mtp_loss = hidden_states.new_zeros(())
                mtp_correct = hidden_states.new_zeros((), dtype=torch.float32)
                mtp_total = hidden_states.new_zeros((), dtype=torch.float32)

            loss = main_loss + mtp_loss_weight * mtp_loss

            self.last_main_loss = main_loss.detach()
            self.last_mtp_loss = mtp_loss.detach()
            self.last_mtp_correct = mtp_correct.detach()
            self.last_mtp_total = mtp_total.detach()
            self.last_mtp_acc = (mtp_correct / mtp_total.clamp_min(1.0)).detach()

            if not hasattr(self, "_mtp_debug_step"):
                self._mtp_debug_step = 0

            self._mtp_debug_step += 1

            if self._mtp_debug_step <= 10:
                try:
                    import torch.distributed as dist

                    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
                except Exception:
                    is_rank0 = True

                if is_rank0:
                    print(
                        f"[MTP DEBUG] step={self._mtp_debug_step} "
                        f"main_loss={float(main_loss.detach().float()):.4f} "
                        f"mtp_loss={float(mtp_loss.detach().float()):.4f} "
                        f"mtp_acc={float(self.last_mtp_acc.float()):.4f} "
                        f"weight={mtp_loss_weight} "
                        f"total_loss={float(loss.detach().float()):.4f}",
                        flush=True,
                    )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
