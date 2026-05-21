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
Parser for Megatron blend list files.

Reference: megatron/core/datasets/utils.py::get_blend_from_list
"""

from typing import List, Optional, Tuple


def parse_blend_list(list_path: str) -> Tuple[List[str], Optional[List[float]]]:
    """Parse a Megatron-style ``.list`` file into dataset prefixes and optional weights.

    The ``.list`` file contains whitespace-separated tokens. The tokens are
    interpreted as either:

    1. A flat list of dataset prefixes (odd number of tokens), or
    2. Alternating ``(weight, prefix)`` pairs (even number of tokens).

    When the token count is even, each odd-positioned token is parsed as a
    float weight. If *any* weight fails parsing, the entire token list is
    treated as prefixes and ``weights`` is returned as ``None``.

    Args:
        list_path: Path to the ``.list`` file.

    Returns:
        A tuple ``(prefixes, weights)`` where *prefixes* is a list of
        stripped path strings and *weights* is either a list of ``float``
        values or ``None``.
    """
    with open(list_path, "r", encoding="utf-8") as f:
        tokens = f.read().split()

    if not tokens:
        return [], None

    if len(tokens) % 2 == 1:
        weights = None
        raw_prefixes = tokens
    else:
        raw_weights = []
        raw_prefixes = []
        for i in range(0, len(tokens), 2):
            raw_weights.append(tokens[i])
            raw_prefixes.append(tokens[i + 1])

        weights = []
        for rw in raw_weights:
            try:
                weights.append(float(rw))
            except ValueError:
                weights = None
                break

        if weights is None:
            raw_prefixes = tokens

    prefixes = [rp.strip() for rp in raw_prefixes]
    return prefixes, weights
