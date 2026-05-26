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

from .blend_parser import parse_blend_list
from .blended_dataset import MegatronBlendedDataset
from .cache_utils import find_megatron_cache, get_cache_paths, load_cached_indices, save_cached_indices
from .collator import MegatronDataCollatorForLanguageModeling, MegatronDataCollatorForSeq2Seq
from .gpt_dataset import MegatronGPTDataset, MegatronGPTDatasetConfig
from .indexed_dataset import MegatronIndexedDataset
from .sample_idx_builder import build_blending_indices, build_exhaustive_blending_indices, build_sample_idx

SUPPORTED_MEGATRON_VERSIONS = ("megatron-core >= 0.5.0",)

__all__ = [
    "MegatronBlendedDataset",
    "MegatronDataCollatorForLanguageModeling",
    "MegatronDataCollatorForSeq2Seq",
    "MegatronGPTDataset",
    "MegatronGPTDatasetConfig",
    "MegatronIndexedDataset",
    "parse_blend_list",
    "find_megatron_cache",
    "get_cache_paths",
    "load_cached_indices",
    "save_cached_indices",
    "build_sample_idx",
    "build_blending_indices",
    "build_exhaustive_blending_indices",
    "SUPPORTED_MEGATRON_VERSIONS",
]
