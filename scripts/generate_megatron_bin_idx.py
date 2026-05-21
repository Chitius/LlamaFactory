#!/usr/bin/env python
"""Generate Megatron-format .bin/.idx files from text data using a HF tokenizer."""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llamafactory.data.megatron.indexed_dataset import MegatronIndexedDataset

_INDEX_HEADER = b"MMIDIDX\x00\x00"


def load_text_data(input_path: str):
    """Load text data from JSONL or Parquet."""
    path = Path(input_path)
    if path.suffix in (".jsonl", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                yield obj["text"]
    elif path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        for text in df["text"]:
            yield text
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")


def write_index(idx_path: str, lengths: np.ndarray, pointers: np.ndarray, doc_indices: np.ndarray, dtype_code: int = 4):
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


def main():
    parser = argparse.ArgumentParser(description="Generate Megatron .bin/.idx from text data")
    parser.add_argument("--input_path", required=True, help="Path to input JSONL or Parquet file")
    parser.add_argument("--tokenizer_path", required=True, help="Path to HF tokenizer directory")
    parser.add_argument("--output_prefix", required=True, help="Output prefix for .bin and .idx files")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for tokenization")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    print(f"Tokenizer vocab_size={len(tokenizer)}, eos_token_id={eos_id}")

    print(f"Loading text data from {args.input_path} ...")
    texts = list(load_text_data(args.input_path))
    n_docs = len(texts)
    print(f"Loaded {n_docs} documents.")

    print("Tokenizing ...")
    all_ids = []
    lengths = []
    total_tokens = 0
    batch_size = args.batch_size
    for i in range(0, n_docs, batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False)
        for ids in encoded["input_ids"]:
            ids.append(eos_id)
            arr = np.array(ids, dtype=np.int32)
            all_ids.append(arr)
            lengths.append(len(arr))
            total_tokens += len(arr)
        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= n_docs:
            print(f"  processed {min(i + batch_size, n_docs)}/{n_docs} docs")

    lengths_arr = np.array(lengths, dtype=np.int32)
    pointers_arr = np.cumsum(np.concatenate(([0], lengths_arr[:-1]))) * np.dtype(np.int32).itemsize
    # Megatron expects document_indices[-1] == sequence_count.
    # When each sequence is its own document: [0, 1, 2, ..., N]
    doc_indices_arr = np.arange(n_docs + 1, dtype=np.int64)

    idx_path = args.output_prefix + ".idx"
    bin_path = args.output_prefix + ".bin"

    print(f"Writing {idx_path} ...")
    write_index(idx_path, lengths_arr, pointers_arr, doc_indices_arr)

    print(f"Writing {bin_path} ...")
    with open(bin_path, "wb") as f:
        for arr in all_ids:
            f.write(arr.tobytes("C"))

    print(f"Done. Sequences={n_docs}, Total tokens={total_tokens}")

    # Verification
    print("\n--- Verification ---")
    dataset = MegatronIndexedDataset(args.output_prefix)
    print(f"Dataset length: {len(dataset)} (expected {n_docs})")
    assert len(dataset) == n_docs, "Dataset length mismatch!"

    num_check = min(5, n_docs)
    for i in range(num_check):
        seq = dataset[i]
        expected = all_ids[i]
        assert len(seq) == len(expected), f"Length mismatch at index {i}"
        assert np.array_equal(seq, expected), f"Content mismatch at index {i}"
        print(f"  doc[{i}] length={len(seq)}  -> OK")

    # Check a contiguous slice
    if n_docs >= 3:
        seqs = dataset[0:3]
        assert len(seqs) == 3, "Slice length mismatch"
        for j in range(3):
            assert np.array_equal(seqs[j], all_ids[j]), f"Slice content mismatch at {j}"
        print("  slice[0:3] -> OK")

    print("\nAll verifications passed.")


if __name__ == "__main__":
    main()
