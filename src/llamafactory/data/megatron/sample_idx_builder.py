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
Python fallback for Megatron's C++ build_sample_idx and build_blending_indices.

Reference: megatron/core/datasets/helpers.cpp
"""

from math import ceil
from typing import Tuple

import warnings

import numpy


def build_sample_idx(
    sizes: numpy.ndarray,
    document_idx: numpy.ndarray,
    sequence_length: int,
    num_epochs: int,
    tokens_per_epoch: int,
    drop_last_partial_sequence: bool = True,
    add_extra_token_to_sequence: bool = True,
) -> numpy.ndarray:
    """Build the sample index.

    This is a pure-Python reimplementation of
    megatron/core/datasets/helpers.cpp::build_sample_idx.

    The sample index is a 2D array with shape (num_samples + 1, 2) where
    sample_idx[i] = (document_idx_index, doc_offset) marks the START of sample i.

    Args:
        sizes: Array of sequence lengths (int32).
        document_idx: Array of document indices (int32), shuffled.
        sequence_length: The sequence length per sample.
        num_epochs: The number of epochs.
        tokens_per_epoch: The number of tokens per epoch.
        drop_last_partial_sequence: Whether to drop the last partial sequence.
        add_extra_token_to_sequence: Whether to add an extra token for shift.

    Returns:
        numpy.ndarray: Sample index array of shape (num_samples + 1, 2).
    """
    assert sequence_length > 1
    assert num_epochs > 0
    assert tokens_per_epoch > 1

    if drop_last_partial_sequence:
        num_samples = (num_epochs * tokens_per_epoch - add_extra_token_to_sequence) // sequence_length
    else:
        num_samples = int(ceil(float(num_epochs * tokens_per_epoch - add_extra_token_to_sequence) / sequence_length))

    # Choose dtype to match Megatron's logic
    sample_idx_max = max(document_idx.shape[0], int(sizes.max()))
    if sample_idx_max <= numpy.iinfo(numpy.int32).max:
        dtype = numpy.int32
    else:
        dtype = numpy.int64

    sample_idx = numpy.zeros(2 * (num_samples + 1), dtype=dtype)

    sample_idx_index = 0
    document_idx_index = dtype(0)
    doc_offset = dtype(0)

    sample_idx[2 * sample_idx_index] = document_idx_index
    sample_idx[2 * sample_idx_index + 1] = doc_offset
    sample_idx_index += 1

    while sample_idx_index <= num_samples:
        remaining_seq_length = sequence_length + add_extra_token_to_sequence
        while remaining_seq_length != 0:
            document_index = document_idx[int(document_idx_index)]
            document_length = sizes[document_index] - doc_offset
            remaining_seq_length -= int(document_length)

            if remaining_seq_length <= 0:
                doc_offset = dtype(
                    int(doc_offset) + remaining_seq_length + int(document_length) - add_extra_token_to_sequence
                )
                remaining_seq_length = 0
            else:
                if int(document_idx_index) == document_idx.shape[0] - 1:
                    assert sample_idx_index == num_samples
                    doc_offset = dtype(sizes[document_idx[int(document_idx_index)]] - add_extra_token_to_sequence)
                    break
                document_idx_index = dtype(int(document_idx_index) + 1)
                doc_offset = dtype(0)

        sample_idx[2 * sample_idx_index] = document_idx_index
        sample_idx[2 * sample_idx_index + 1] = doc_offset
        sample_idx_index += 1

    return sample_idx.reshape(-1, 2)


def build_blending_indices(
    dataset_index: numpy.ndarray,
    dataset_sample_index: numpy.ndarray,
    weights: numpy.ndarray,
    num_datasets: int,
    size: int,
    verbose: bool = False,
) -> None:
    """Build blending indices using the greedy max-error algorithm.

    Reference: megatron/core/datasets/helpers.cpp::build_blending_indices

    Args:
        dataset_index: Output array of shape (size,) dtype int16.
        dataset_sample_index: Output array of shape (size,) dtype int64.
        weights: Normalized weights array of shape (num_datasets,) dtype float64.
        num_datasets: Number of datasets.
        size: Total number of blended samples.
        verbose: Whether to print info.
    """
    # Prefer Megatron C++ helpers for large-scale datasets
    try:
        from megatron.core.datasets import helpers
        helpers.build_blending_indices(
            dataset_index, dataset_sample_index, weights, num_datasets, size, verbose
        )
        return
    except Exception:
        warnings.warn(
            "Failed to import Megatron C++ helpers. Falling back to pure-Python blending index builder, "
            "which may be significantly slower for large-scale datasets.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Pure-Python fallback (slow for large datasets)
    current_samples = numpy.zeros(num_datasets, dtype=numpy.int64)

    for sample_idx in range(size):
        sample_idx_double = max(float(sample_idx), 1.0)
        max_error_index = 0
        max_error = weights[0] * sample_idx_double - current_samples[0]

        for dataset_idx in range(1, num_datasets):
            error = weights[dataset_idx] * sample_idx_double - current_samples[dataset_idx]
            if error > max_error:
                max_error = error
                max_error_index = dataset_idx

        dataset_index[sample_idx] = numpy.int16(max_error_index)
        dataset_sample_index[sample_idx] = current_samples[max_error_index]
        current_samples[max_error_index] += 1

    if verbose:
        print("> sample ratios:")
        for dataset_idx in range(num_datasets):
            ratio = float(current_samples[dataset_idx]) / float(size)
            print(f"   dataset {dataset_idx}, input: {weights[dataset_idx]}, achieved: {ratio}")


def build_exhaustive_blending_indices(
    dataset_index: numpy.ndarray,
    dataset_sample_index: numpy.ndarray,
    sizes: numpy.ndarray,
    num_datasets: int,
) -> None:
    """Build exhaustive blending indices.

    Reference: megatron/core/datasets/helpers.cpp::build_exhaustive_blending_indices

    Args:
        dataset_index: Output array of shape (sum(sizes),) dtype int16.
        dataset_sample_index: Output array of shape (sum(sizes),) dtype int64.
        sizes: Array of exact sample counts per dataset.
        num_datasets: Number of datasets.
    """
    # Prefer Megatron C++ helpers for large-scale datasets
    try:
        from megatron.core.datasets import helpers
        helpers.build_exhaustive_blending_indices(
            dataset_index, dataset_sample_index, sizes, num_datasets
        )
        return
    except Exception:
        warnings.warn(
            "Failed to import Megatron C++ helpers. Falling back to pure-Python exhaustive blending index builder, "
            "which may be significantly slower for large-scale datasets.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Pure-Python fallback (slow for large datasets)
    total_size = int(sizes.sum())
    weights = sizes.astype(numpy.float64) / float(total_size)
    dataset_sample_counts = numpy.zeros(num_datasets, dtype=numpy.int64)
    unspent = set(range(num_datasets))

    index_sample = 0
    while len(unspent) > 0:
        index_sample_double = max(float(index_sample), 1.0)
        error_argmax = -1
        error_max = float("-inf")

        for dataset_idx in unspent:
            error = weights[dataset_idx] * index_sample_double - dataset_sample_counts[dataset_idx]
            if error > error_max:
                error_max = error
                error_argmax = dataset_idx

        assert error_argmax >= 0
        dataset_index[index_sample] = numpy.int16(error_argmax)
        dataset_sample_index[index_sample] = dataset_sample_counts[error_argmax]
        dataset_sample_counts[error_argmax] += 1

        if sizes[error_argmax] - dataset_sample_counts[error_argmax] == 0:
            unspent.remove(error_argmax)

        index_sample += 1
