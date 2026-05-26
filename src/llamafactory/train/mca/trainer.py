# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Iterator

import torch
import torch.nn.functional as F
from mcore_adapter.trainer import McaTrainer
from mcore_adapter.trainer.utils import get_ltor_masks_and_position_ids
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX


class CustomMcaTrainer(McaTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def _prepare_train_inputs(self, data_iterator: Iterator) -> dict[str, Tensor | Any]:
        inputs = next(data_iterator)
        inputs = {**inputs}  # avoid repeated modifications
        if self.args.sequence_packing:
            inputs = self._packing_sequence(inputs)
        else:
            attention_mask = inputs.get("attention_mask")
            # If collator already produced a 4D document-boundary mask, preserve it.
            # Megatron local DotProductAttention expects a *boolean* mask where
            # True = masked, so we convert the float mask (0 / min_dtype) to bool.
            if isinstance(attention_mask, Tensor) and attention_mask.dim() == 4:
                inputs["attention_mask"] = attention_mask == torch.finfo(attention_mask.dtype).min
                if "position_ids" not in inputs:
                    inputs["position_ids"] = None
            elif inputs.get("packed_seq_params") is not None:
                # Varlen FA2/FA3 path: collator already set attention_mask=None and provided
                # packed_seq_params. Flatten inputs to (1, -1) for THD format if needed,
                # and migrate packed_seq_params tensors to the model device.
                input_ids = inputs.get("input_ids")
                if input_ids is not None and isinstance(input_ids, Tensor) and input_ids.dim() == 2 and input_ids.size(0) != 1:
                    for key in ("input_ids", "labels", "position_ids"):
                        val = inputs.get(key)
                        if val is not None and isinstance(val, Tensor) and val.dim() == 2:
                            inputs[key] = val.view(1, -1)
                if "position_ids" not in inputs:
                    inputs["position_ids"] = None
                packed_seq_params = inputs.get("packed_seq_params")
                if packed_seq_params is not None:
                    device = self.args.device
                    if hasattr(packed_seq_params, "cu_seqlens_q") and packed_seq_params.cu_seqlens_q is not None:
                        packed_seq_params.cu_seqlens_q = packed_seq_params.cu_seqlens_q.to(device)
                    if hasattr(packed_seq_params, "cu_seqlens_kv") and packed_seq_params.cu_seqlens_kv is not None:
                        packed_seq_params.cu_seqlens_kv = packed_seq_params.cu_seqlens_kv.to(device)
                    if hasattr(packed_seq_params, "cu_seqlens_q_padded") and packed_seq_params.cu_seqlens_q_padded is not None:
                        packed_seq_params.cu_seqlens_q_padded = packed_seq_params.cu_seqlens_q_padded.to(device)
                    if hasattr(packed_seq_params, "cu_seqlens_kv_padded") and packed_seq_params.cu_seqlens_kv_padded is not None:
                        packed_seq_params.cu_seqlens_kv_padded = packed_seq_params.cu_seqlens_kv_padded.to(device)
                    if hasattr(packed_seq_params, "seq_idx") and packed_seq_params.seq_idx is not None:
                        packed_seq_params.seq_idx = packed_seq_params.seq_idx.to(device)
            else:
                attention_mask, position_ids = get_ltor_masks_and_position_ids(
                    inputs["input_ids"],
                    build_attention_mask=self.model_impl != "transformer_engine",
                    attn_mask_1D=attention_mask,
                )
                if not self.model.config.num_moe_experts and self.model_impl == "transformer_engine":
                    attention_mask = None
                inputs["attention_mask"] = attention_mask
                if "position_ids" not in inputs:
                    inputs["position_ids"] = position_ids if self.model.config.mtp_num_layers else None

        if "position_ids" not in inputs:
            inputs["position_ids"] = None
        inputs = self._get_batch_on_this_cp_rank(inputs)
        return inputs

    @override
    def _pad_batched_inputs(self, inputs: dict[str, Tensor | Any], seq_length: int):
        r"""Override to avoid padding error when handling 3d posids."""
        packed_seq_params = inputs.get("packed_seq_params")
        input_ids = inputs.get("input_ids")
        if (
            packed_seq_params is not None
            and input_ids is not None
            and isinstance(input_ids, Tensor)
            and input_ids.dim() == 2
        ):
            # Varlen FA2/FA3 path: flatten inputs if not already flattened,
            # then skip seq_length padding.
            if input_ids.size(0) != 1:
                for key in ("input_ids", "labels", "position_ids"):
                    val = inputs.get(key)
                    if val is not None and isinstance(val, Tensor) and val.dim() == 2:
                        inputs[key] = val.view(1, -1)
            for key in list(inputs.keys()):
                val = inputs[key]
                if isinstance(val, Tensor):
                    inputs[key] = val.to(self.args.device)
            # Handle packed_seq_params device migration
            packed_seq_params = inputs.pop("packed_seq_params", None)
            if packed_seq_params is not None:
                if hasattr(packed_seq_params, "cu_seqlens_q") and packed_seq_params.cu_seqlens_q is not None:
                    packed_seq_params.cu_seqlens_q = packed_seq_params.cu_seqlens_q.to(self.args.device)
                if hasattr(packed_seq_params, "cu_seqlens_kv") and packed_seq_params.cu_seqlens_kv is not None:
                    packed_seq_params.cu_seqlens_kv = packed_seq_params.cu_seqlens_kv.to(self.args.device)
                if hasattr(packed_seq_params, "cu_seqlens_q_padded") and packed_seq_params.cu_seqlens_q_padded is not None:
                    packed_seq_params.cu_seqlens_q_padded = packed_seq_params.cu_seqlens_q_padded.to(self.args.device)
                if hasattr(packed_seq_params, "cu_seqlens_kv_padded") and packed_seq_params.cu_seqlens_kv_padded is not None:
                    packed_seq_params.cu_seqlens_kv_padded = packed_seq_params.cu_seqlens_kv_padded.to(self.args.device)
                if hasattr(packed_seq_params, "seq_idx") and packed_seq_params.seq_idx is not None:
                    packed_seq_params.seq_idx = packed_seq_params.seq_idx.to(self.args.device)
                inputs["packed_seq_params"] = packed_seq_params
            return inputs

        padding_inputs = {
            k: v.tolist() if v is not None and isinstance(v, Tensor) else v
            for k, v in inputs.items()
            if k in self._language_input_names
        }

        position_ids_3d = None
        if isinstance(inputs.get("position_ids"), Tensor) and inputs["position_ids"].dim() == 3:
            position_ids_3d = inputs["position_ids"]
            padding_inputs.pop("position_ids", None)

        if "labels" in padding_inputs:
            padding_inputs["labels"] = [
                labels + [IGNORE_INDEX] * (seq_length - len(labels)) for labels in padding_inputs["labels"]
            ]
        tokenizer = (
            self.processing_class
            if isinstance(self.processing_class, PreTrainedTokenizerBase)
            else getattr(self.processing_class, "tokenizer", self.processing_class)
        )
        padding_side = getattr(tokenizer, "padding_side", "right")
        padding_inputs = tokenizer.pad(
            padding_inputs,
            padding="max_length",
            max_length=seq_length,
            return_tensors="pt",
        ).to(self.args.device)
        inputs.update(padding_inputs)

        for key in ("document_ids", "position_ids"):
            tensor = inputs.get(key)
            if tensor is not None and isinstance(tensor, Tensor) and tensor.dim() <= 2:
                current_seq_len = tensor.size(-1)
                if current_seq_len < seq_length:
                    pad_len = seq_length - current_seq_len
                    if padding_side == "left":
                        tensor = F.pad(tensor, (pad_len, 0), value=0)
                    else:
                        tensor = F.pad(tensor, (0, pad_len), value=0)
                    inputs[key] = tensor.to(self.args.device)

        if position_ids_3d is not None:
            current_seq_len = position_ids_3d.size(-1)
            if current_seq_len < seq_length:
                pad_len = seq_length - current_seq_len
                if padding_side == "left":
                    position_ids_3d = F.pad(position_ids_3d, (pad_len, 0), value=0)
                else:
                    position_ids_3d = F.pad(position_ids_3d, (0, pad_len), value=0)

            inputs["position_ids"] = position_ids_3d.to(self.args.device)

        # Handle packed_seq_params (Megatron-style varlen FlashAttention)
        packed_seq_params = inputs.pop("packed_seq_params", None)
        if packed_seq_params is not None:
            if hasattr(packed_seq_params, "cu_seqlens_q") and packed_seq_params.cu_seqlens_q is not None:
                packed_seq_params.cu_seqlens_q = packed_seq_params.cu_seqlens_q.to(self.args.device)
            if hasattr(packed_seq_params, "cu_seqlens_kv") and packed_seq_params.cu_seqlens_kv is not None:
                packed_seq_params.cu_seqlens_kv = packed_seq_params.cu_seqlens_kv.to(self.args.device)
            if hasattr(packed_seq_params, "cu_seqlens_q_padded") and packed_seq_params.cu_seqlens_q_padded is not None:
                packed_seq_params.cu_seqlens_q_padded = packed_seq_params.cu_seqlens_q_padded.to(self.args.device)
            if hasattr(packed_seq_params, "cu_seqlens_kv_padded") and packed_seq_params.cu_seqlens_kv_padded is not None:
                packed_seq_params.cu_seqlens_kv_padded = packed_seq_params.cu_seqlens_kv_padded.to(self.args.device)
            if hasattr(packed_seq_params, "seq_idx") and packed_seq_params.seq_idx is not None:
                packed_seq_params.seq_idx = packed_seq_params.seq_idx.to(self.args.device)
            inputs["packed_seq_params"] = packed_seq_params

        return inputs
