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
Standalone Megatron BlendedDataset implementation with zero Megatron dependencies.

This module re-implements the core blending logic from
``megatron/core/datasets/blended_dataset.py`` without importing Megatron.
"""

import hashlib
import json
import os
from typing import List, Optional

import numpy
import torch

from .gpt_dataset import MegatronGPTDataset
from .sample_idx_builder import build_blending_indices, build_exhaustive_blending_indices


class MegatronBlendedDataset(torch.utils.data.Dataset):
    """Blend multiple :class:`MegatronGPTDataset` instances into a single dataset.

    This re-implements the core logic from
    ``megatron/core/datasets/blended_dataset.py`` without importing Megatron.

    The blended dataset builds two index arrays:

    * ``dataset_index[i]`` - which dataset to draw sample *i* from.
    * ``dataset_sample_index[i]`` - which sample index to request from that
      dataset.

    Two construction modes are supported:

    1. **Weighted blending** - ``size`` is given and ``weights`` is a list of
       positive floats. The weights are normalized and ``size`` samples are
       drawn proportionally.
    2. **Exhaustive blending** - ``size`` is ``None`` and ``weights`` is a
       list of positive integers. Exactly ``weights[idx]`` samples are drawn
       from ``datasets[idx]`` (all samples are consumed in a deterministic
       interleaved order).

    Args:
        datasets: List of :class:`MegatronGPTDataset` instances to blend.
        weights: Blend weights. Floats for weighted mode, integers for
            exhaustive mode.
        size: Total number of blended samples. When ``None``, exhaustive
            blending is performed.
        data_cache_path: Directory where built indices are cached. When
            ``None``, indices are built in-memory and not persisted.
        seed: Random seed used for cache consistency (reserved for future
            use; the blending algorithm itself is deterministic).
        split: Dataset split name, used for cache file naming.
    """

    def __init__(
        self,
        datasets: List[MegatronGPTDataset],
        weights: Optional[List[float]],
        size: Optional[int],
        data_cache_path: Optional[str],
        seed: int,
        split: str = "train",
    ) -> None:
        if weights is not None:
            assert len(datasets) == len(weights), (
                f"Number of datasets ({len(datasets)}) must match number of weights ({len(weights)})"
            )
            assert len(datasets) < 32767, (
                f"Number of datasets ({len(datasets)}) must be less than 32767"
            )
            assert all(w > 0 for w in weights), "All weights must be positive"
            assert all(type(w) == type(weights[0]) for w in weights), (
                "All weights must be of the same type"
            )

            if size is None and isinstance(weights[0], float):
                # Megatron compatibility: float weights with size=None must all be integral
                assert all(w == int(w) for w in weights), (
                    "When size is None, all float weights must be integral values"
                )

        self.datasets = datasets
        self.weights = weights
        self.size = size
        self.data_cache_path = data_cache_path
        self.seed = seed
        self.split = split

        self._dataset_index: Optional[numpy.ndarray] = None
        self._dataset_sample_index: Optional[numpy.ndarray] = None

        self._build_and_cache_indices()

    def _get_description(self) -> str:
        """Return a deterministic JSON description used for cache hashing.

        The description aggregates the unique descriptions of all underlying
        datasets together with the blend parameters so that any change
        invalidates the cache.

        Returns:
            A JSON string suitable for MD5 hashing.
        """
        description = {
            "class": type(self).__name__,
            "datasets": [dataset._get_description() for dataset in self.datasets],
            "split": self.split,
            "weights": self.weights,
            "size": self.size,
        }
        return json.dumps(description, sort_keys=True, indent=2)

    def _get_cache_base_name(self) -> str:
        """Return the base name for BlendedDataset cache files.

        Returns:
            A string of the form ``{hash}-BlendedDataset-{split}``.
        """
        description = self._get_description()
        hash_val = hashlib.md5(description.encode("utf-8")).hexdigest()
        return f"{hash_val}-BlendedDataset-{self.split}"

    def _get_cache_paths(self) -> dict:
        """Return paths for the two index cache files and description.

        Returns:
            Dictionary with keys ``dataset_index``, ``dataset_sample_index``,
            and ``description`` mapping to absolute file paths. Returns an
            empty dictionary when ``data_cache_path`` is ``None``.
        """
        base_name = self._get_cache_base_name()
        if self.data_cache_path is None:
            return {}
        return {
            "dataset_index": os.path.join(
                self.data_cache_path, f"{base_name}-dataset_index.npy"
            ),
            "dataset_sample_index": os.path.join(
                self.data_cache_path, f"{base_name}-dataset_sample_index.npy"
            ),
            "description": os.path.join(
                self.data_cache_path, f"{base_name}-description.txt"
            ),
        }

    def _load_cached_indices(self) -> bool:
        """Load indices from cache if all files exist.

        Returns:
            ``True`` if indices were loaded successfully, ``False`` otherwise.
        """
        cache_paths = self._get_cache_paths()
        if not cache_paths:
            return False

        required = ("dataset_index", "dataset_sample_index", "description")
        if not all(os.path.isfile(cache_paths[k]) for k in required):
            return False

        self._dataset_index = numpy.load(cache_paths["dataset_index"], allow_pickle=True)
        self._dataset_sample_index = numpy.load(
            cache_paths["dataset_sample_index"], allow_pickle=True
        )
        return True

    def _save_cached_indices(self) -> None:
        """Save indices and description to cache."""
        cache_paths = self._get_cache_paths()
        if not cache_paths:
            return

        cache_dir = os.path.dirname(cache_paths["dataset_index"])
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        with open(cache_paths["description"], "wt") as writer:
            writer.write(self._get_description())

        numpy.save(cache_paths["dataset_index"], self._dataset_index, allow_pickle=True)
        numpy.save(
            cache_paths["dataset_sample_index"], self._dataset_sample_index, allow_pickle=True
        )

    def _build_and_cache_indices(self) -> None:
        """Build blending indices, using cache when available."""
        if self._load_cached_indices():
            return

        if self.size is not None and self.weights is not None:
            # Weighted blending mode
            weights_array = numpy.array(self.weights, dtype=numpy.float64)
            weights_array = weights_array / weights_array.sum()

            dataset_index = numpy.zeros(self.size, dtype=numpy.int16)
            dataset_sample_index = numpy.zeros(self.size, dtype=numpy.int64)

            build_blending_indices(
                dataset_index,
                dataset_sample_index,
                weights_array,
                len(self.datasets),
                self.size,
                verbose=False,
            )
        elif self.size is None and self.weights is not None:
            # Exhaustive blending mode
            sizes = numpy.array(self.weights, dtype=numpy.int64)
            total_size = int(sizes.sum())

            dataset_index = numpy.zeros(total_size, dtype=numpy.int16)
            dataset_sample_index = numpy.zeros(total_size, dtype=numpy.int64)

            build_exhaustive_blending_indices(
                dataset_index,
                dataset_sample_index,
                sizes,
                len(self.datasets),
            )
        else:
            raise ValueError(
                "Invalid combination of size and weights. "
                "Either provide both size and weights for weighted blending, "
                "or provide weights (as integers) with size=None for exhaustive blending."
            )

        # Validation: check no dataset is oversampled
        dataset_indices, dataset_sizes = numpy.unique(dataset_index, return_counts=True)
        for _index, _size in zip(dataset_indices, dataset_sizes):
            if len(self.datasets[_index]) < _size:
                raise IndexError(
                    f"The {self.split} blend oversamples dataset {_index}: "
                    f"requests {_size} samples but dataset has {len(self.datasets[_index])} samples."
                )

        self._dataset_index = dataset_index
        self._dataset_sample_index = dataset_sample_index

        self._save_cached_indices()

    def __len__(self) -> int:
        """Return the number of samples in the blended dataset."""
        return len(self._dataset_index)

    def __getitem__(self, idx: int):
        """Return a single sample from the blended dataset.

        Args:
            idx: Sample index.

        Returns:
            The sample from the underlying dataset at the mapped position.
        """
        dataset_id = int(self._dataset_index[idx])
        dataset_sample_id = int(self._dataset_sample_index[idx])
        return self.datasets[dataset_id][dataset_sample_id]
