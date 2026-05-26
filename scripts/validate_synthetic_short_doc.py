#!/usr/bin/env python
"""Validate high document-boundary impact of the synthetic short-doc Megatron dataset."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset
from llamafactory.data.megatron.gpt_dataset import MegatronGPTDataset, MegatronGPTDatasetConfig


def main() -> None:
    prefix = str(Path(__file__).parent.parent / "data" / "synthetic_short_doc_text_document")
    seq_length = 64
    add_extra = True

    indexed_dataset = MegatronIndexedDataset(prefix)
    config = MegatronGPTDatasetConfig(
        path_prefix=prefix,
        seq_length=seq_length,
        seed=42,
        add_extra_token=add_extra,
        drop_last_partial_sequence=True,
        data_cache_path=None,
    )
    dataset = MegatronGPTDataset(config, indexed_dataset)

    sample_index = dataset.sample_index
    num_samples = len(dataset)
    print(f"Total documents: {len(indexed_dataset)}")
    print(f"Total tokens: {int(indexed_dataset.sequence_lengths.sum())}")
    print(f"Number of samples (seq_length={seq_length}, add_extra_token={add_extra}): {num_samples}")

    doc_spans = []
    total_sample_tokens = 0
    total_boundary_near_tokens = 0
    # Define "near boundary" as within 5 tokens of any internal document boundary.
    # (First/last document segments in a sample are not internal boundaries.)
    boundary_window = 5

    for i in range(num_samples):
        doc_beg_idx, offset_beg = sample_index[i]
        doc_end_idx, offset_end = sample_index[i + 1]

        span_docs = int(doc_end_idx - doc_beg_idx + 1)
        doc_spans.append(span_docs)

        # Reconstruct per-sample document segment lengths exactly as _get_sample does.
        part_lengths = []
        for di in range(doc_beg_idx, doc_end_idx + 1):
            offset = 0 if di > doc_beg_idx else int(offset_beg)
            if di < doc_end_idx:
                length = None  # read to end of document
            else:
                length = int(offset_end) + (1 if add_extra else 0)
            seq = indexed_dataset.get(int(dataset.document_index[di]), offset=offset, length=length)
            part_lengths.append(len(seq))

        total_len = sum(part_lengths)
        total_sample_tokens += total_len

        # Count tokens near internal boundaries.
        # For each boundary between part j and part j+1, count the last `boundary_window`
        # tokens of part j and the first `boundary_window` tokens of part j+1.
        for j in range(len(part_lengths) - 1):
            total_boundary_near_tokens += min(boundary_window, part_lengths[j])
            total_boundary_near_tokens += min(boundary_window, part_lengths[j + 1])

    doc_spans_arr = np.array(doc_spans)

    print("\n=== Document Span Statistics ===")
    print(f"  Min docs per sample: {int(doc_spans_arr.min())}")
    print(f"  Max docs per sample: {int(doc_spans_arr.max())}")
    print(f"  Mean docs per sample: {doc_spans_arr.mean():.2f}")
    print(f"  % samples spanning >=2 docs: {(doc_spans_arr >= 2).mean() * 100:.2f}%")
    print(f"  % samples spanning >=3 docs: {(doc_spans_arr >= 3).mean() * 100:.2f}%")
    print(f"  % samples spanning >=4 docs: {(doc_spans_arr >= 4).mean() * 100:.2f}%")

    print("\n=== Boundary Impact Statistics ===")
    print(f"  Boundary window: +/-{boundary_window} tokens around each internal boundary")
    print(f"  Total tokens in all samples: {total_sample_tokens}")
    print(f"  Tokens near an internal boundary: {total_boundary_near_tokens}")
    print(f"  Ratio: {total_boundary_near_tokens / total_sample_tokens * 100:.2f}%")

    # Also compute global stream boundary proximity.
    global_boundary_near = 0
    global_total = 0
    for doc_idx in range(len(indexed_dataset)):
        doc_len = len(indexed_dataset[doc_idx])
        global_total += doc_len
        # first boundary_window and last boundary_window tokens of each document
        near_in_doc = min(boundary_window, doc_len) + min(boundary_window, doc_len)
        # avoid double-counting if doc_len < 2*boundary_window
        near_in_doc = min(near_in_doc, doc_len)
        global_boundary_near += near_in_doc

    print(f"\n  Global stream ratio (within +/-{boundary_window} of any doc start/end): "
          f"{global_boundary_near / global_total * 100:.2f}%")

    assert (doc_spans_arr >= 2).mean() >= 0.90, (
        f"Less than 90% of samples span >=2 documents! "
        f"Actual: {(doc_spans_arr >= 2).mean() * 100:.2f}%"
    )
    print("\n[PASS] All assertions passed.")


if __name__ == "__main__":
    main()
