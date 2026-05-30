"""Tests for Megatron blend list parser.

Covers:
  - Basic prefix-only and weighted list parsing
  - Comment / blank-line skipping
  - Empty-file handling
  - Behavioural alignment with Megatron-LM ``get_blend_from_list``
"""

import os
import tempfile
from typing import List, Optional, Tuple

import pytest

from llamafactory.data.megatron.blend_parser import parse_blend_list
from utils import check_megatron_source, load_megatron_module

# ---------------------------------------------------------------------------
# Load Megatron-LM reference implementation for cross-check
# ---------------------------------------------------------------------------

_megatron_dir = check_megatron_source(require_helpers_cpp=False)
_megatron_utils = load_megatron_module(
    "megatron.core.datasets.utils",
    os.path.join(_megatron_dir, "megatron/core/datasets/utils.py"),
)
get_blend_from_list = _megatron_utils.get_blend_from_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_list(content: str) -> str:
    """Write *content* to a temporary ``.list`` file and return its path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".list", delete=False, encoding="utf-8") as f:
        f.write(content)
        return f.name


def _parse(content: str) -> Tuple[List[str], Optional[List[float]]]:
    """Helper: write *content* to a temp file and parse it."""
    path = _write_list(content)
    try:
        return parse_blend_list(path)
    finally:
        os.unlink(path)


def _megatron_parse(tokens: List[str]) -> Tuple[List[str], Optional[List[float]]]:
    """Run Megatron-LM reference parser on a raw token list."""
    result = get_blend_from_list(tokens)
    if result is None:
        return [], None
    return result


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestBasicParsing:
    def test_prefix_only_flat_list(self):
        content = "/path/to/a\n/path/to/b\n/path/to/c"
        prefixes, weights = _parse(content)
        assert prefixes == ["/path/to/a", "/path/to/b", "/path/to/c"]
        assert weights is None

    def test_weighted_pairs(self):
        content = "0.3 /path/to/a\n0.7 /path/to/b"
        prefixes, weights = _parse(content)
        assert prefixes == ["/path/to/a", "/path/to/b"]
        assert weights == pytest.approx([0.3, 0.7])

    def test_inline_tokens(self):
        """Tokens may be separated by arbitrary whitespace on a single line."""
        content = "0.25 /path/a  0.75 /path/b"
        prefixes, weights = _parse(content)
        assert prefixes == ["/path/a", "/path/b"]
        assert weights == pytest.approx([0.25, 0.75])


class TestCommentAndBlankLineHandling:
    def test_hash_comments_are_skipped(self):
        content = (
            "# This is a comment\n"
            "/path/to/a\n"
            "# Another comment\n"
            "/path/to/b\n"
        )
        prefixes, weights = _parse(content)
        assert prefixes == ["/path/to/a", "/path/to/b"]
        assert weights is None

    def test_blank_lines_are_skipped(self):
        content = "\n\n/path/to/a\n\n/path/to/b\n\n"
        prefixes, weights = _parse(content)
        assert prefixes == ["/path/to/a", "/path/to/b"]
        assert weights is None

    def test_comments_and_blank_lines_mixed(self):
        """Simulates a real-world .list file with headers and spacing."""
        content = (
            "############################################################\n"
            "# total_tokens(B)=1025.995341757 total_tokens=1025995341757\n"
            "############################################################\n"
            "\n"
            "10.676108168 /public/Datasets/part0/data_0_000000\n"
            "10.666388789 /public/Datasets/part0/data_10_000000\n"
            "# trailing comment\n"
            "10.670542935 /public/Datasets/part0/data_11_000000\n"
        )
        prefixes, weights = _parse(content)
        assert prefixes == [
            "/public/Datasets/part0/data_0_000000",
            "/public/Datasets/part0/data_10_000000",
            "/public/Datasets/part0/data_11_000000",
        ]
        assert weights == pytest.approx([10.676108168, 10.666388789, 10.670542935])

    def test_empty_file(self):
        prefixes, weights = _parse("")
        assert prefixes == []
        assert weights is None

    def test_only_comments(self):
        prefixes, weights = _parse("# comment 1\n# comment 2\n")
        assert prefixes == []
        assert weights is None

    def test_only_blank_lines(self):
        prefixes, weights = _parse("\n\n\n")
        assert prefixes == []
        assert weights is None


class TestEdgeCases:
    def test_single_prefix(self):
        prefixes, weights = _parse("/only/one")
        assert prefixes == ["/only/one"]
        assert weights is None

    def test_single_weighted_pair(self):
        prefixes, weights = _parse("1.0 /only/one")
        assert prefixes == ["/only/one"]
        assert weights == pytest.approx([1.0])

    def test_weight_parse_failure_fallback(self):
        """Even token count but non-numeric first token -> treat all as prefixes."""
        content = "/path/a not_a_number /path/b"
        prefixes, weights = _parse(content)
        assert prefixes == ["/path/a", "not_a_number", "/path/b"]
        assert weights is None

    def test_paths_with_whitespace_trimmed(self):
        content = "  /path/to/a  "
        prefixes, weights = _parse(content)
        assert prefixes == ["/path/to/a"]


class TestMegatronAlignment:
    """Ensure parse_blend_list behaves identically to Megatron's get_blend_from_list
    once file-level comments/blanks are stripped."""

    @pytest.mark.parametrize(
        "tokens",
        [
            ["/a", "/b", "/c"],
            ["0.3", "/a", "0.7", "/b"],
            ["1.0", "/only"],
            ["/a", "not_a_weight", "/b", "not_a_weight"],
            [],
        ],
    )
    def test_alignment_with_megatron(self, tokens: List[str]):
        content = " ".join(tokens)
        lf_prefixes, lf_weights = _parse(content)
        mg_prefixes, mg_weights = _megatron_parse(tokens)

        assert lf_prefixes == mg_prefixes
        if lf_weights is None:
            assert mg_weights is None
        else:
            assert lf_weights == pytest.approx(mg_weights)
