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
Cache path utilities for Megatron GPT dataset indices.

Reference: megatron/core/datasets/gpt_dataset.py (cache path logic)
"""

import os
from typing import Optional, Tuple

import numpy


def get_cache_paths(base_name: str, data_cache_path: str) -> dict:
    """Return the cache file paths for a given base name.

    In Megatron, the indices are saved as:
    ``{base_name}-document_index.npy``, ``{base_name}-sample_index.npy``,
    ``{base_name}-shuffle_index.npy``, and ``{base_name}-description.txt``.

    Reference:
        ``megatron/core/datasets/gpt_dataset.py::_build_document_sample_shuffle_indices``

    Args:
        base_name: The base file name prefix (e.g. ``{hash}-GPTDataset-train``).
        data_cache_path: The directory that holds the cached indices.

    Returns:
        A dictionary with keys ``document_index``, ``sample_index``,
        ``shuffle_index``, and ``description`` mapping to absolute file paths.
    """
    return {
        "document_index": os.path.join(data_cache_path, f"{base_name}-document_index.npy"),
        "sample_index": os.path.join(data_cache_path, f"{base_name}-sample_index.npy"),
        "shuffle_index": os.path.join(data_cache_path, f"{base_name}-shuffle_index.npy"),
        "description": os.path.join(data_cache_path, f"{base_name}-description.txt"),
    }


def load_cached_indices(cache_paths: dict) -> Optional[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]:
    """Load cached indices from disk if all files exist.

    Reference:
        ``megatron/core/datasets/gpt_dataset.py::_build_document_sample_shuffle_indices``
        (cache loading path)

    Args:
        cache_paths: Dictionary returned by :func:`get_cache_paths`.

    Returns:
        A tuple ``(document_index, sample_index, shuffle_index)`` if all
        cache files exist, otherwise ``None``.
    """
    required_keys = ("document_index", "sample_index", "shuffle_index", "description")
    if not all(os.path.isfile(cache_paths[k]) for k in required_keys):
        return None

    document_index = numpy.load(cache_paths["document_index"], allow_pickle=True)
    sample_index = numpy.load(cache_paths["sample_index"], allow_pickle=True)
    shuffle_index = numpy.load(cache_paths["shuffle_index"], allow_pickle=True)
    return document_index, sample_index, shuffle_index


def save_cached_indices(
    cache_paths: dict,
    document_index: numpy.ndarray,
    sample_index: numpy.ndarray,
    shuffle_index: numpy.ndarray,
    description: str,
) -> None:
    """Save indices to disk.

    Reference:
        ``megatron/core/datasets/gpt_dataset.py::_build_document_sample_shuffle_indices``
        (cache saving path)

    Args:
        cache_paths: Dictionary returned by :func:`get_cache_paths`.
        document_index: The document index array.
        sample_index: The sample index array.
        shuffle_index: The shuffle index array.
        description: Description string written to the ``description.txt`` file.
    """
    cache_dir = os.path.dirname(cache_paths["document_index"])
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    with open(cache_paths["description"], "wt") as writer:
        writer.write(description)

    numpy.save(cache_paths["document_index"], document_index, allow_pickle=True)
    numpy.save(cache_paths["sample_index"], sample_index, allow_pickle=True)
    numpy.save(cache_paths["shuffle_index"], shuffle_index, allow_pickle=True)


def find_megatron_cache(data_cache_path: str, split: str) -> Optional[dict]:
    """Scan *data_cache_path* for an existing Megatron-style cache.

    Megatron names cache files as ``{hash}-{ClassName}-{split}-{affix}``.
    This function looks for a ``*-{split}-document_index.npy`` file and,
    if the corresponding ``sample_index``, ``shuffle_index``, and
    ``description`` files also exist, returns their paths.

    Reference:
        ``megatron/core/datasets/gpt_dataset.py::_build_document_sample_shuffle_indices``

    Args:
        data_cache_path: Directory to scan.
        split: The split name (e.g. ``"train"``) to match.

    Returns:
        A dictionary in the same format as :func:`get_cache_paths` if a
        complete cache set is found, otherwise ``None``.
    """
    if not os.path.isdir(data_cache_path):
        return None

    suffix = f"{split}-document_index.npy"
    for fname in sorted(os.listdir(data_cache_path)):
        if fname.endswith(suffix):
            base_name = fname[: -len("-document_index.npy")]
            cache_paths = get_cache_paths(base_name, data_cache_path)
            if all(os.path.isfile(p) for p in cache_paths.values()):
                return cache_paths

    return None
