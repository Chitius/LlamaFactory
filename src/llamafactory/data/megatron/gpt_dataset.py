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

"""
Standalone Megatron GPT dataset implementation with zero Megatron dependencies.

This module re-implements the core indexing logic from
``megatron/core/datasets/gpt_dataset.py`` without importing Megatron,
falling back to a pure-Python :mod:`.sample_idx_builder` when the
Megatron C++ helpers are unavailable.
"""

import hashlib
import os
from dataclasses import dataclass
from math import ceil
from typing import Optional, Tuple

import numpy
import torch

from .cache_utils import find_megatron_cache, get_cache_paths, load_cached_indices, save_cached_indices
from .sample_idx_builder import build_sample_idx


@dataclass
class MegatronGPTDatasetConfig:
    """Configuration object for :class:`MegatronGPTDataset`.

    Corresponds to ``megatron/core/datasets/gpt_dataset.py::GPTDatasetConfig``
    but only exposes the fields needed for a minimal, standalone GPT dataset.
    """

    path_prefix: str
    """Path prefix to the IndexedDataset ``.bin`` / ``.idx`` files."""

    seq_length: int
    """Target sequence length for each sample."""

    seed: int = 42
    """Random seed used for shuffling documents and samples."""

    num_samples: Optional[int] = None
    """Total number of training samples to draw. When ``None``, the dataset
    exposes exactly one epoch of samples."""

    data_cache_path: Optional[str] = None
    """Directory where built indices are cached. When ``None``, indices are
    built in-memory and not persisted."""

    add_extra_token: bool = True
    """Whether to draw ``seq_length + 1`` tokens so that inputs and labels
    are both of length ``seq_length`` (standard causal-LM shift)."""

    drop_last_partial_sequence: bool = True
    """Whether to drop the final partial sequence when it does not fill
    ``seq_length``."""

    reuse_megatron_cache: bool = False
    """If ``True``, scan *data_cache_path* for pre-existing Megatron cache
    files and reuse them when possible."""

    split: str = "train"
    """Dataset split name, used for cache discovery and descriptions."""

    split_ratios: Optional[str] = None
    """Comma-separated train/valid/test ratios, e.g. '0.9,0.05,0.05'."""

    pad_token_id: int = -1
    """Token ID used for padding when a sample spans document boundaries
    and is shorter than the required length."""

    reset_attention_mask: bool = False
    """Whether to reset attention mask at document boundaries.
    When True, ``__getitem__`` returns an extra ``document_ids`` field."""

    reset_position_ids: bool = False
    """Whether to reset position ids at document boundaries (restart from 0)."""

    eod_mask_loss: bool = False
    """Whether to mask EOD token loss (set labels to -100 at EOD positions)."""

    eod_token_id: Optional[int] = None
    """EOD token ID. Required when ``eod_mask_loss`` is True."""

    shift_labels: bool = False
    """Whether to shift labels by one position for causal LM training.
    When False, labels = tokens.clone() (HuggingFace convention, model internally shifts).
    When True and add_extra_token=True, labels = text[1:] (Megatron convention, model does NOT internally shift).
    When True and add_extra_token=False, labels = roll(text, -1) with last position masked.
    """

    def __post_init__(self):
        if self.reset_attention_mask or self.reset_position_ids or self.eod_mask_loss:
            if self.pad_token_id < 0:
                raise ValueError(
                    f"pad_token_id must be a non-negative integer when "
                    f"reset_attention_mask, reset_position_ids, or eod_mask_loss is enabled, got {self.pad_token_id}. "
                    f"Ensure loader.py passes tokenizer.pad_token_id."
                )
        if self.eod_mask_loss and self.eod_token_id is None:
            raise ValueError("eod_token_id must be provided when eod_mask_loss is enabled.")


class MegatronGPTDataset(torch.utils.data.Dataset):
    """The GPT dataset, compatible with Megatron-LM's GPTDataset.

    Builds the document index, sample index, and shuffle index using the
    same algorithms as Megatron, but with no Megatron import dependency.

    Reference:
        ``megatron/core/datasets/gpt_dataset.py::GPTDataset``
    """

    def __init__(self, config: MegatronGPTDatasetConfig, indexed_dataset) -> None:
        """
        Args:
            config: The dataset configuration.
            indexed_dataset: A ``MegatronIndexedDataset``-like object that
                exposes ``sequence_lengths`` (``numpy.ndarray``) and a
                ``get(idx, offset, length)`` method.
        """
        self.config = config
        self.indexed_dataset = indexed_dataset

        # Attempt cache reuse / build
        self._document_index: Optional[numpy.ndarray] = None
        self._sample_index: Optional[numpy.ndarray] = None
        self._shuffle_index: Optional[numpy.ndarray] = None

        if config.data_cache_path:
            if config.reuse_megatron_cache:
                cache_paths = find_megatron_cache(config.data_cache_path, config.split)
            else:
                cache_paths = None

            if cache_paths is None:
                base_name = self._get_cache_base_name()
                cache_paths = get_cache_paths(base_name, config.data_cache_path)

            cached = load_cached_indices(cache_paths)
            if cached is not None:
                self._document_index, self._sample_index, self._shuffle_index = cached

            if self._document_index is None:
                self._document_index, self._sample_index, self._shuffle_index = self._build_indices()
                save_cached_indices(
                    cache_paths,
                    self._document_index,
                    self._sample_index,
                    self._shuffle_index,
                    self._get_description(),
                )
        else:
            self._document_index, self._sample_index, self._shuffle_index = self._build_indices()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_description(self) -> str:
        """Return a deterministic description string used for cache hashing.

        Reference:
            ``megatron/core/datasets/megatron_dataset.py::MegatronDataset.unique_description``
        """
        lines = [
            f"path_prefix: {self.config.path_prefix}",
            f"split: {self.config.split}",
            f"split_ratios: {self.config.split_ratios}",
            f"seq_length: {self.config.seq_length}",
            f"num_samples: {self.config.num_samples}",
            f"seed: {self.config.seed}",
            f"add_extra_token: {self.config.add_extra_token}",
            f"drop_last_partial_sequence: {self.config.drop_last_partial_sequence}",
            f"num_sequences: {len(self.indexed_dataset)}",
            f"total_tokens: {int(self.indexed_dataset.sequence_lengths.sum())}",
        ]
        return "\n".join(lines)

    def _get_cache_base_name(self) -> str:
        """Return the base name for cache files.

        Reference:
            ``megatron/core/datasets/gpt_dataset.py::_build_document_sample_shuffle_indices``
        """
        description = self._get_description()
        hash_val = hashlib.md5(description.encode("utf-8")).hexdigest()
        return f"{hash_val}-MegatronGPTDataset-{self.config.split}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples.

        Reference:
            ``megatron/core/datasets/gpt_dataset.py::GPTDataset.__len__``
        """
        return int(self._sample_index.shape[0] - 1)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample.

        The output dictionary follows Llama-Factory conventions:
        ``{"input_ids": ..., "labels": ..., "attention_mask": ...}``.

        Reference:
            ``megatron/core/datasets/gpt_dataset.py::GPTDataset.__getitem__``
        """
        text, document_ids, position_ids, padding_mask = self._get_sample(idx)
        text = torch.from_numpy(text).long()
        document_ids = torch.from_numpy(document_ids).long()
        padding_mask = torch.from_numpy(padding_mask)
        if position_ids is not None:
            position_ids = torch.from_numpy(position_ids).long()

        if self.config.add_extra_token:
            tokens = text[:-1].contiguous()
            if self.config.shift_labels:
                labels = text[1:].clone()
                labels_pad = padding_mask[1:]  # aligned with text[1:]
            else:
                labels = tokens.clone()
                labels_pad = padding_mask[: self.config.seq_length]  # aligned with text[:-1]
        else:
            tokens = text
            if self.config.shift_labels:
                labels = torch.roll(text, shifts=-1, dims=0).clone()
                labels[-1] = -100  # mask last position since there is no next token
                # labels[i] corresponds to text[i+1] for i < seq_length-1
                labels_pad = torch.cat([padding_mask[1:], torch.tensor([False])])
            else:
                labels = tokens.clone()
                labels_pad = padding_mask

        labels[labels_pad] = -100

        if self.config.eod_mask_loss and self.config.eod_token_id is not None:
            labels[labels == self.config.eod_token_id] = -100

        attention_mask = torch.ones(self.config.seq_length, dtype=torch.long)

        result = {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if self.config.reset_attention_mask:
            result["document_ids"] = document_ids[: self.config.seq_length]

        if position_ids is not None:
            result["position_ids"] = position_ids[: self.config.seq_length]

        return result

    @property
    def document_index(self) -> numpy.ndarray:
        """The ordered array of document (sequence) IDs repeated for each epoch."""
        return self._document_index

    @property
    def sample_index(self) -> numpy.ndarray:
        """The 2-D array of ``(document_idx_index, doc_offset)`` marking the
        start of every sample."""
        return self._sample_index

    @property
    def shuffle_index(self) -> numpy.ndarray:
        """The random permutation used to shuffle samples."""
        return self._shuffle_index

    @property
    def is_shuffled_internally(self) -> bool:
        """Whether this dataset performs its own shuffling."""
        return True

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _get_documents_for_split(self) -> numpy.ndarray:
        """Return the document IDs for the current split, optionally sliced by split_ratios.

        If ``split_ratios`` is provided (e.g. ``"0.9,0.05,0.05"``), the total document
        range is partitioned into train/valid/test and only the slice corresponding to
        ``config.split`` is returned.
        """
        total_docs = len(self.indexed_dataset)
        split_ratios = self.config.split_ratios
        if split_ratios is None:
            return numpy.arange(total_docs, dtype=numpy.int32)

        ratios = [float(x) for x in split_ratios.split(",")]
        if len(ratios) != 3:
            raise ValueError(f"split_ratios must have exactly 3 values, got {ratios}")

        # Normalize to sum=1
        ratio_sum = sum(ratios)
        if ratio_sum <= 0:
            raise ValueError(f"split_ratios must sum to > 0, got {ratio_sum}")
        ratios = [r / ratio_sum for r in ratios]

        train_end = int(total_docs * ratios[0])
        valid_end = train_end + int(total_docs * ratios[1])

        if self.config.split == "train":
            return numpy.arange(0, train_end, dtype=numpy.int32)
        elif self.config.split == "valid":
            return numpy.arange(train_end, valid_end, dtype=numpy.int32)
        elif self.config.split == "test":
            return numpy.arange(valid_end, total_docs, dtype=numpy.int32)
        else:
            raise ValueError(f"Unknown split '{self.config.split}'. Expected 'train', 'valid', or 'test'.")

    def _build_indices(self) -> tuple:
        """Build document, sample, and shuffle indices.

        Reference:
            ``megatron/core/datasets/gpt_dataset.py::_build_document_sample_shuffle_indices``
        """
        # Build document index with optional split slicing first, so that
        # num_tokens_per_epoch reflects only the documents in this split.
        documents = self._get_documents_for_split()
        if len(documents) == 0:
            raise ValueError(
                f"No documents available for split '{self.config.split}' with "
                f"split_ratios='{self.config.split_ratios}'. Cannot build a dataset from 0 documents."
            )

        num_tokens_per_epoch = int(numpy.sum(self.indexed_dataset.sequence_lengths[documents]))
        num_epochs = self._get_num_epochs(num_tokens_per_epoch)

        # Determine whether the final epoch should be shuffled separately
        if num_epochs == 1:
            separate_final_epoch = False
            num_samples_sans_final_epoch = 0
        else:
            add_extra = 1 if self.config.add_extra_token else 0
            num_samples_sans_final_epoch = (
                (num_epochs - 1) * num_tokens_per_epoch - add_extra
            ) // self.config.seq_length
            num_samples_from_final_epoch = self.config.num_samples - num_samples_sans_final_epoch
            num_samples_per_epoch = (
                num_tokens_per_epoch - add_extra
            ) // self.config.seq_length

            assert num_samples_from_final_epoch >= 0
            assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

            threshold = 0.80
            separate_final_epoch = num_samples_from_final_epoch < int(threshold * num_samples_per_epoch)

        numpy_random_state = numpy.random.RandomState(self.config.seed)

        document_index = self._build_document_index(
            documents, num_epochs, numpy_random_state, separate_final_epoch
        )

        # Build sample index
        drop_last = self.config.drop_last_partial_sequence
        add_extra = 1 if self.config.add_extra_token else 0

        # Try Megatron C++ helpers first, fall back to pure Python
        sample_index = self._try_build_sample_idx_cpp(
            document_index, num_epochs, num_tokens_per_epoch, drop_last, add_extra
        )
        if sample_index is None:
            sample_index = build_sample_idx(
                self.indexed_dataset.sequence_lengths,
                document_index,
                self.config.seq_length,
                num_epochs,
                num_tokens_per_epoch,
                drop_last,
                add_extra,
            )

        # Build shuffle index
        total_samples = int(sample_index.shape[0] - 1)
        if separate_final_epoch:
            shuffle_index = self._build_shuffle_index(
                num_samples_sans_final_epoch, total_samples, numpy_random_state
            )
        else:
            shuffle_index = self._build_shuffle_index(
                total_samples, total_samples, numpy_random_state
            )

        return document_index, sample_index, shuffle_index

    def _try_build_sample_idx_cpp(
        self,
        document_index: numpy.ndarray,
        num_epochs: int,
        num_tokens_per_epoch: int,
        drop_last: bool,
        add_extra: int,
    ) -> Optional[numpy.ndarray]:
        """Attempt to build the sample index with Megatron's C++ helpers.

        Returns ``None`` on any failure so the caller can fall back to the
        Python implementation.

        Reference:
            ``megatron/core/datasets/helpers.cpp::build_sample_idx``
        """
        try:
            from megatron.core.datasets import helpers
        except Exception:
            return None

        if document_index.dtype != numpy.int32:
            return None
        if self.indexed_dataset.sequence_lengths.dtype != numpy.int32:
            return None

        # Force-load mmap when access density is high (same heuristic as Megatron)
        if len(document_index) * 2 > len(self.indexed_dataset.sequence_lengths):
            sequence_lengths_for_cpp = self.indexed_dataset.sequence_lengths.copy()
        else:
            sequence_lengths_for_cpp = self.indexed_dataset.sequence_lengths

        sample_idx_max = max(
            document_index.shape[0], int(self.indexed_dataset.sequence_lengths.max())
        )
        if sample_idx_max <= numpy.iinfo(numpy.int32).max:
            sample_index = helpers.build_sample_idx_int32(
                sequence_lengths_for_cpp,
                document_index,
                self.config.seq_length,
                num_epochs,
                num_tokens_per_epoch,
                drop_last,
                add_extra,
            )
        else:
            sample_index = helpers.build_sample_idx_int64(
                sequence_lengths_for_cpp,
                document_index,
                self.config.seq_length,
                num_epochs,
                num_tokens_per_epoch,
                drop_last,
                add_extra,
            )
        return sample_index

    def _get_num_tokens_per_epoch(self) -> int:
        """Calculate the number of tokens in a single epoch.

        Reference:
            ``megatron/core/datasets/gpt_dataset.py::GPTDataset._get_num_tokens_per_epoch``
        """
        return int(numpy.sum(self.indexed_dataset.sequence_lengths))

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs required to satisfy *num_samples*.

        Reference:
            ``megatron/core/datasets/gpt_dataset.py::GPTDataset._get_num_epochs``
        """
        num_epochs = 1
        num_tokens = num_tokens_per_epoch
        if self.config.num_samples is None:
            return num_epochs

        num_tokens_requested = (
            self.config.num_samples * self.config.seq_length
        ) + (1 if self.config.add_extra_token else 0)
        while num_tokens < num_tokens_requested:
            num_epochs += 1
            num_tokens += num_tokens_per_epoch
        return num_epochs

    @staticmethod
    def _build_document_index(
        documents: numpy.ndarray,
        num_epochs: int,
        numpy_random_state: numpy.random.RandomState,
        separate_final_epoch: bool,
    ) -> numpy.ndarray:
        """Build an array with length ``num_epochs * len(documents)``.

        Documents are repeated *num_epochs* times and then globally shuffled.
        When *separate_final_epoch* is ``True``, the first ``num_epochs-1``
        epochs are shuffled together and the final epoch is shuffled
        separately.

        Reference:
            ``megatron/core/datasets/gpt_dataset.py::_build_document_index``
        """
        if not separate_final_epoch or num_epochs == 1:
            document_index = numpy.mgrid[0:num_epochs, 0 : len(documents)][1]
            document_index[:] = documents
            document_index = document_index.reshape(-1)
            document_index = document_index.astype(numpy.int32)
            numpy_random_state.shuffle(document_index)
            return document_index

        doc_idx_first = MegatronGPTDataset._build_document_index(
            documents, num_epochs - 1, numpy_random_state, False
        )
        doc_idx_last = MegatronGPTDataset._build_document_index(
            documents, 1, numpy_random_state, False
        )
        return numpy.concatenate((doc_idx_first, doc_idx_last))

    @staticmethod
    def _build_shuffle_index(
        num_samples: int,
        total_size: int,
        numpy_random_state: numpy.random.RandomState,
    ) -> numpy.ndarray:
        """Build the range ``[0, size)`` and shuffle it.

        If *num_samples* < *total_size*, the two sub-ranges are shuffled
        independently with the same random state.

        Reference:
            ``megatron/core/datasets/gpt_dataset.py::_build_shuffle_index``
        """
        dtype_ = numpy.uint32
        if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
            dtype_ = numpy.int64

        shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
        numpy_random_state.shuffle(shuffle_idx_first)
        if num_samples == total_size:
            return shuffle_idx_first

        shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
        numpy_random_state.shuffle(shuffle_idx_last)

        return numpy.concatenate((shuffle_idx_first, shuffle_idx_last))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _get_sample(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray, Optional[numpy.ndarray], numpy.ndarray]:
        """Retrieve the raw token ids, document ids, position ids, and padding mask for sample *idx*.

        This performs the shuffle mapping, looks up the sample index, and
        concatenates tokens from one or more documents, padding if needed.
        Document ids mark each token with an incrementing segment index
        (starting from 1; padding positions are 0).
        The padding mask is ``True`` at positions that were artificially padded
        and ``False`` for real document tokens.

        Reference:
            ``megatron/core/datasets/gpt_dataset.py::GPTDataset._query_document_sample_shuffle_indices``
        """
        idx = int(self._shuffle_index[idx])

        doc_index_beg, doc_index_beg_offset = self._sample_index[idx]
        doc_index_end, doc_index_end_offset = self._sample_index[idx + 1]

        sample_parts = []
        document_ids_parts = []
        document_id = 1

        if doc_index_beg == doc_index_end:
            # Sample spans a single document
            length = int(
                doc_index_end_offset
                - doc_index_beg_offset
                + (1 if self.config.add_extra_token else 0)
            )
            sample_parts.append(
                self.indexed_dataset.get(
                    int(self._document_index[doc_index_beg]),
                    offset=int(doc_index_beg_offset),
                    length=length,
                )
            )
            document_ids_parts.append(
                numpy.full(len(sample_parts[-1]), document_id, dtype=numpy.int64)
            )
        else:
            # Sample spans multiple documents
            for i in range(doc_index_beg, doc_index_end + 1):
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                if i < doc_index_end:
                    length = None
                else:
                    length = int(doc_index_end_offset + (1 if self.config.add_extra_token else 0))
                sample_parts.append(
                    self.indexed_dataset.get(
                        int(self._document_index[i]),
                        offset=int(offset),
                        length=length,
                    )
                )
                document_ids_parts.append(
                    numpy.full(len(sample_parts[-1]), document_id, dtype=numpy.int64)
                )
                document_id += 1

        length = sum(map(len, sample_parts))
        target_length = self.config.seq_length + (1 if self.config.add_extra_token else 0)

        # Build position-based padding mask: True = artificially padded, False = real data
        padding_mask_parts = [numpy.zeros(len(p), dtype=bool) for p in sample_parts]

        if length < target_length:
            pad_length = target_length - length
            sample_parts.append(
                numpy.full(pad_length, self.config.pad_token_id, dtype=numpy.int64)
            )
            document_ids_parts.append(
                numpy.zeros(pad_length, dtype=numpy.int64)
            )
            padding_mask_parts.append(
                numpy.ones(pad_length, dtype=bool)
            )

        text = numpy.concatenate(sample_parts, dtype=numpy.int64)
        document_ids = numpy.concatenate(document_ids_parts, dtype=numpy.int64)
        padding_mask = numpy.concatenate(padding_mask_parts)

        position_ids = None
        if self.config.reset_position_ids:
            position_ids = numpy.arange(target_length, dtype=numpy.int64)
            boundary_indices = numpy.where(numpy.diff(document_ids, prepend=document_ids[0]))[0]
            for b in boundary_indices:
                if b > 0 and document_ids[b] != 0:
                    position_ids[b:] -= position_ids[b]

        return text, document_ids, position_ids, padding_mask

    def _get_text(self, idx: int) -> numpy.ndarray:
        """Retrieve the raw token ids for sample *idx*.

        Backward-compatible wrapper around :meth:`_get_sample`.
        """
        text, _, _, _ = self._get_sample(idx)
        return text
