#!/usr/bin/env python
"""Generate synthetic short-document Megatron-format .bin/.idx for testing document-boundary masks."""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset

_INDEX_HEADER = b"MMIDIDX\x00\x00"


def write_index(
    idx_path: str,
    lengths: np.ndarray,
    pointers: np.ndarray,
    doc_indices: np.ndarray,
    dtype_code: int = 4,
) -> None:
    """Write the .idx file in Megatron format."""
    with open(idx_path, "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", dtype_code))  # dtype code (4 = int32)
        f.write(struct.pack("<Q", len(lengths)))  # sequence count
        f.write(struct.pack("<Q", len(doc_indices)))  # document count
        f.write(lengths.astype(np.int32).tobytes("C"))
        f.write(pointers.astype(np.int64).tobytes("C"))
        f.write(doc_indices.astype(np.int64).tobytes("C"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic short-document Megatron dataset"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "synthetic_short_doc_text_document"),
        help="Output prefix for .bin and .idx files",
    )
    parser.add_argument(
        "--num_docs", type=int, default=200, help="Number of documents to generate"
    )
    parser.add_argument(
        "--doc_length", type=int, default=20, help="Number of tokens per document (excluding EOS)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: gpt2, vocab_size={vocab_size}, eos_id={eos_id}")

    rng = np.random.RandomState(args.seed)
    all_ids = []
    lengths = []
    total_tokens = 0

    for _ in range(args.num_docs):
        ids = rng.randint(0, vocab_size, size=args.doc_length).tolist()
        ids.append(int(eos_id))
        arr = np.array(ids, dtype=np.int32)
        all_ids.append(arr)
        lengths.append(len(arr))
        total_tokens += len(arr)

    lengths_arr = np.array(lengths, dtype=np.int32)
    pointers_arr = (
        np.cumsum(np.concatenate(([0], lengths_arr[:-1]))) * np.dtype(np.int32).itemsize
    )
    doc_indices_arr = np.arange(args.num_docs + 1, dtype=np.int64)

    idx_path = str(args.output_prefix) + ".idx"
    bin_path = str(args.output_prefix) + ".bin"

    print(f"Writing {idx_path} ...")
    write_index(idx_path, lengths_arr, pointers_arr, doc_indices_arr)

    print(f"Writing {bin_path} ...")
    with open(bin_path, "wb") as f:
        for arr in all_ids:
            f.write(arr.tobytes("C"))

    print(f"Done. Documents={args.num_docs}, Total tokens={total_tokens}")

    # Verification
    print("\n--- Verification ---")
    dataset = MegatronIndexedDataset(str(args.output_prefix))
    assert len(dataset) == args.num_docs, "Dataset length mismatch!"
    print(f"Dataset length: {len(dataset)} (expected {args.num_docs})")

    for i in range(min(5, args.num_docs)):
        seq = dataset[i]
        expected = all_ids[i]
        assert len(seq) == len(expected), f"Length mismatch at index {i}"
        assert np.array_equal(seq, expected), f"Content mismatch at index {i}"
        print(f"  doc[{i}] length={len(seq)} -> OK")

    if args.num_docs >= 3:
        seqs = dataset[0:3]
        assert len(seqs) == 3, "Slice length mismatch"
        for j in range(3):
            assert np.array_equal(seqs[j], all_ids[j]), f"Slice content mismatch at {j}"
        print("  slice[0:3] -> OK")

    print("\nAll verifications passed.")


if __name__ == "__main__":
    main()
