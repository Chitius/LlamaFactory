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

from dataclasses import dataclass
from typing import Any, Literal

import torch
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

try:
    from megatron.core.packed_seq_params import PackedSeqParams
except Exception:
    # Fallback for environments where Megatron is not installed or importable
    @dataclass
    class PackedSeqParams:
        qkv_format: str = None
        cu_seqlens_q: torch.Tensor = None
        cu_seqlens_kv: torch.Tensor = None
        cu_seqlens_q_padded: torch.Tensor = None
        cu_seqlens_kv_padded: torch.Tensor = None
        max_seqlen_q: int = None
        max_seqlen_kv: int = None
        local_cp_size: int = None
        cp_group: Any = None

from ...data.collator import apply_document_boundary_mask, prepare_4d_attention_mask


def _compute_packed_seq_params_for_document_boundary(
    document_ids: torch.Tensor,
) -> PackedSeqParams:
    """Compute Megatron-style PackedSeqParams from document_ids for varlen FlashAttention.

    Args:
        document_ids: [batch_size, seq_len] tensor of document segment indices.
                     Padding positions have value 0.

    Returns:
        PackedSeqParams with qkv_format='thd', cu_seqlens_q/kv, max_seqlen_q/kv.
    """
    batch_size, seq_len = document_ids.shape
    segment_lengths = []
    max_seqlen = 0

    for i in range(batch_size):
        row = document_ids[i]
        nonzero = (row != 0).nonzero(as_tuple=False)
        if len(nonzero) == 0:
            # All padding sample
            segment_lengths.append(seq_len)
            max_seqlen = max(max_seqlen, seq_len)
            continue

        valid_len = nonzero[-1].item() + 1
        valid_row = row[:valid_len]
        padding_len = seq_len - valid_len

        # Compute contiguous segments for the valid part
        if len(valid_row) > 0:
            diff = valid_row[1:] != valid_row[:-1]
            change_positions = torch.nonzero(diff, as_tuple=False).flatten() + 1
            boundaries = [0] + change_positions.tolist() + [len(valid_row)]
            lengths = [boundaries[j + 1] - boundaries[j] for j in range(len(boundaries) - 1)]
            segment_lengths.extend(lengths)
            if lengths:
                max_seqlen = max(max_seqlen, max(lengths))

        # Add trailing padding segment
        if padding_len > 0:
            segment_lengths.append(padding_len)
            max_seqlen = max(max_seqlen, padding_len)

    if not segment_lengths:
        # This shouldn't happen now since all-padding adds seq_len, but keep as safeguard
        cu_seqlens = torch.zeros(2, device=document_ids.device, dtype=torch.int32)
    else:
        cu = [0]
        for length in segment_lengths:
            cu.append(cu[-1] + length)
        cu_seqlens = torch.tensor(cu, device=document_ids.device, dtype=torch.int32)

    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


@dataclass
class MegatronDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Data collator for Megatron GPT dataset with document-boundary support.

    Extends DataCollatorForLanguageModeling to:
    1. Preserve dataset-provided labels (do not regenerate them).
    2. Manually pad ``document_ids`` (custom field not handled by tokenizer.pad).
    3. Call ``apply_document_boundary_mask`` to generate 4D block-diagonal + causal mask.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2", "fa2", "fa3", "disabled"] = "eager"
    compute_dtype: torch.dtype = torch.float32

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # 1. 分离 dataset 提供的自定义字段，避免被 base collator 误处理
        # 使用 shallow copy 消除 pop 对传入 examples 的副作用
        examples = [dict(ex) for ex in examples]
        custom_labels = [ex.get("labels") for ex in examples]
        custom_document_ids = [ex.pop("document_ids", None) for ex in examples]
        custom_position_ids = [ex.pop("position_ids", None) for ex in examples]

        # 2. 先调用 base collator，它会 pad input_ids, attention_mask(0/1), labels
        #    但 base collator 会覆盖 labels，所以我们稍后恢复
        batch = super().torch_call(examples)

        # 3. 恢复 dataset 提供的 labels（覆盖 base collator 生成的 labels）
        if any(cl is not None for cl in custom_labels):
            max_len = batch["input_ids"].size(1)
            padded_labels = []
            for cl in custom_labels:
                if cl is None:
                    # fallback: regenerate like base collator
                    pl = batch["input_ids"][len(padded_labels)].clone()
                    pl[pl == self.tokenizer.pad_token_id] = -100
                    padded_labels.append(pl)
                else:
                    if isinstance(cl, list):
                        cl = torch.tensor(cl, dtype=torch.long)
                    pad_len = max_len - cl.size(0)
                    if pad_len > 0:
                        if self.tokenizer.padding_side == "right":
                            cl = torch.cat([cl, torch.full((pad_len,), -100, dtype=torch.long)])
                        else:
                            cl = torch.cat([torch.full((pad_len,), -100, dtype=torch.long), cl])
                    padded_labels.append(cl)
            batch["labels"] = torch.stack(padded_labels)

        # 4. 手动 pad document_ids（tokenizer.pad 不会自动处理自定义字段）
        if any(d is not None for d in custom_document_ids):
            max_len = batch["input_ids"].size(1)
            padded_document_ids = []
            for doc_ids in custom_document_ids:
                if doc_ids is None:
                    # fallback: 所有有效 token 属于 doc 1，pad 为 0
                    seq_len = int(batch["attention_mask"][len(padded_document_ids)].sum().item())
                    doc_ids = torch.cat([
                        torch.ones(seq_len, dtype=torch.long),
                        torch.zeros(max_len - seq_len, dtype=torch.long),
                    ])
                else:
                    if isinstance(doc_ids, list):
                        doc_ids = torch.tensor(doc_ids, dtype=torch.long)
                    pad_len = max_len - doc_ids.size(0)
                    if pad_len > 0:
                        if self.tokenizer.padding_side == "right":
                            doc_ids = torch.cat([doc_ids, torch.zeros(pad_len, dtype=torch.long)])
                        else:
                            doc_ids = torch.cat([torch.zeros(pad_len, dtype=torch.long), doc_ids])
                padded_document_ids.append(doc_ids)
            batch["document_ids"] = torch.stack(padded_document_ids)

        # 5. 手动 pad position_ids
        if any(p is not None for p in custom_position_ids):
            max_len = batch["input_ids"].size(1)
            padded_position_ids = []
            for pid in custom_position_ids:
                if pid is None:
                    pid = torch.arange(max_len, dtype=torch.long)
                else:
                    if isinstance(pid, list):
                        pid = torch.tensor(pid, dtype=torch.long)
                    pad_len = max_len - pid.size(0)
                    if pad_len > 0:
                        if self.tokenizer.padding_side == "right":
                            pid = torch.cat([pid, torch.zeros(pad_len, dtype=torch.long)])
                        else:
                            pid = torch.cat([torch.zeros(pad_len, dtype=torch.long), pid])
                padded_position_ids.append(pid)
            batch["position_ids"] = torch.stack(padded_position_ids)

        # 6. 应用 document-boundary mask（生成 4D mask 或 FA2 预处理）
        batch = apply_document_boundary_mask(
            batch,
            self.compute_dtype,
            self.attn_implementation,
            self.block_diag_attn,
        )

        # 7. FA2/FA3 varlen: flatten batch to (1, total_tokens) so that
        #    flash_attn_varlen_func receives query/key/value in the expected
        #    (total_tokens, nheads, head_dim) layout.  cu_seq_lens_q/k from
        #    step 6 already partitions the full batch — they are unchanged.
        #
        #    After flattening, label positions that were "first token" of
        #    samples 1..bsz-1 in the 2D layout become prediction targets of
        #    the preceding sample's last logit.  These cross-sample boundary
        #    predictions have no counterpart in the 2D path and would dilute
        #    the loss.  We mask them to -100.
        if self.block_diag_attn and self.attn_implementation in ("flash_attention_2", "fa2", "fa3"):
            bsz = batch["input_ids"].shape[0]
            seq_len = batch["input_ids"].shape[1]
            for key in ("input_ids", "labels", "position_ids"):
                if key in batch and batch[key] is not None and isinstance(batch[key], torch.Tensor):
                    if batch[key].dim() == 2:
                        batch[key] = batch[key].view(1, -1)
            if bsz > 1 and "labels" in batch and batch["labels"] is not None:
                for k in range(1, bsz):
                    batch["labels"][0, k * seq_len] = -100

        return batch


@dataclass
class MegatronDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """Data collator for Megatron GPT dataset in MCA path with document-boundary support.

    Extends DataCollatorForSeq2Seq to:
    1. Handle ``document_ids`` and ``position_ids`` custom fields.
    2. Convert tensors to lists before passing to base collator.
    3. Manually pad custom fields after base collator.
    4. Generate Megatron-style ``PackedSeqParams`` for FA2/FA3 varlen,
       or 4D block-diagonal mask for eager/SDPA.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2", "fa2", "fa3", "disabled"] = "eager"
    compute_dtype: torch.dtype = torch.float32

    def __call__(self, features: list[dict[str, Any]], return_tensors=None) -> dict[str, Any]:
        # 1. 分离 dataset 提供的自定义字段，避免被 base collator 误处理
        examples = [dict(ex) for ex in features]
        custom_document_ids = [ex.pop("document_ids", None) for ex in examples]
        custom_position_ids = [ex.pop("position_ids", None) for ex in examples]

        # 2. DataCollatorForSeq2Seq.__call__ 会对 labels 做 list 拼接，
        #    如果 labels/document_ids/position_ids 是 tensor 会出错，先转 list
        for ex in examples:
            for key in ("labels", "document_ids", "position_ids"):
                if key in ex and torch.is_tensor(ex[key]):
                    ex[key] = ex[key].tolist()

        # 3. 调用 base collator，它会 pad input_ids, attention_mask, labels
        batch = super().__call__(examples, return_tensors=return_tensors)

        # 4. 手动 pad document_ids（tokenizer.pad 不会自动处理自定义字段）
        if any(d is not None for d in custom_document_ids):
            max_len = batch["input_ids"].size(1)
            padded_document_ids = []
            for doc_ids in custom_document_ids:
                if doc_ids is None:
                    # fallback: 所有有效 token 属于 doc 1，pad 为 0
                    seq_len = int(batch["attention_mask"][len(padded_document_ids)].sum().item())
                    doc_ids = torch.cat([
                        torch.ones(seq_len, dtype=torch.long),
                        torch.zeros(max_len - seq_len, dtype=torch.long),
                    ])
                else:
                    if isinstance(doc_ids, list):
                        doc_ids = torch.tensor(doc_ids, dtype=torch.long)
                    pad_len = max_len - doc_ids.size(0)
                    if pad_len > 0:
                        if self.tokenizer.padding_side == "right":
                            doc_ids = torch.cat([doc_ids, torch.zeros(pad_len, dtype=torch.long)])
                        else:
                            doc_ids = torch.cat([torch.zeros(pad_len, dtype=torch.long), doc_ids])
                padded_document_ids.append(doc_ids)
            batch["document_ids"] = torch.stack(padded_document_ids)

        # 5. 手动 pad position_ids
        if any(p is not None for p in custom_position_ids):
            max_len = batch["input_ids"].size(1)
            padded_position_ids = []
            for pid in custom_position_ids:
                if pid is None:
                    pid = torch.arange(max_len, dtype=torch.long)
                else:
                    if isinstance(pid, list):
                        pid = torch.tensor(pid, dtype=torch.long)
                    pad_len = max_len - pid.size(0)
                    if pad_len > 0:
                        if self.tokenizer.padding_side == "right":
                            pid = torch.cat([pid, torch.zeros(pad_len, dtype=torch.long)])
                        else:
                            pid = torch.cat([torch.zeros(pad_len, dtype=torch.long), pid])
                padded_position_ids.append(pid)
            batch["position_ids"] = torch.stack(padded_position_ids)

        # 6. 应用 document-boundary mask：MCA 路径生成 PackedSeqParams 而非 transformers cu_seq_lens
        document_ids = batch.pop("document_ids", None)
        if self.block_diag_attn and document_ids is not None:
            if self.attn_implementation in ("flash_attention_2", "fa2", "fa3"):
                batch["packed_seq_params"] = _compute_packed_seq_params_for_document_boundary(document_ids)
                batch["attention_mask"] = None
                # NOTE: We intentionally keep inputs 2D here.
                # Flattening to (1, -1) for varlen FlashAttention will be handled
                # by CustomMcaTrainer._pad_batched_inputs AFTER the MCA base
                # trainer's batch_size check has passed.
            else:
                batch["attention_mask"] = prepare_4d_attention_mask(document_ids, self.compute_dtype)

        # cast data dtype
        for key, value in batch.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(self.compute_dtype)

        return batch
