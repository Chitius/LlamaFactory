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
Standalone reader for Megatron-LM IndexedDataset (.bin/.idx) format.

This module re-implements megatron/core/datasets/indexed_dataset.py without
importing Megatron, ensuring zero external dependencies beyond numpy and torch.

Reference: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py
"""

import struct
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple, Type, Union

import numpy
import torch


_INDEX_HEADER = b"MMIDIDX\x00\x00"


class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices."""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[numpy.number]) -> int:
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[numpy.number]]) -> int:
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError(f"Invalid dtype key: {key}")

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[numpy.number]:
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        else:
            return numpy.int32


class _IndexReader:
    """Object class to read the index (.idx) file.

    Args:
        idx_path: The path to the index file.
        multimodal: Whether the dataset is multimodal.
        sequences_per_dataset: The sequences per dataset (fast path).
        dtype_code: The dtype code of the tokenized documents (fast path).
    """

    def __init__(
        self,
        idx_path: str,
        multimodal: bool = False,
        sequences_per_dataset: Optional[Tuple[int, int]] = None,
        dtype_code: Optional[int] = None,
    ) -> None:
        if sequences_per_dataset is not None:
            assert dtype_code is not None
            self.dtype = DType.dtype_from_code(dtype_code)
            self.dtype_size = DType.size(self.dtype)
            self.sequence_count = sequences_per_dataset[0]
            self.document_count = sequences_per_dataset[1]
            offset = 34
        else:
            with open(idx_path, "rb") as stream:
                header = stream.read(9)
                assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

                version = struct.unpack("<Q", stream.read(8))[0]
                assert version == 1, f"bad version, cannot read: {idx_path}"

                code = struct.unpack("<B", stream.read(1))[0]
                self.dtype = DType.dtype_from_code(code)
                self.dtype_size = DType.size(self.dtype)

                self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
                self.document_count = struct.unpack("<Q", stream.read(8))[0]

                offset = stream.tell()

        self._idx_path = idx_path
        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )

        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )

        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )

        self.sequence_modes = None
        if multimodal:
            self.sequence_modes = numpy.frombuffer(
                self.bin_buffer,
                dtype=numpy.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )

    def __del__(self) -> None:
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()  # type: ignore[attr-defined]
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[numpy.int64, numpy.int32, Optional[numpy.int8]]:
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


class _MMapBinReader:
    """A bin reader that memory maps the data (.bin) file.

    Args:
        bin_path: The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_file_reader = open(bin_path, mode="rb")
        self._bin_buffer_mmap = numpy.memmap(self._bin_file_reader, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap.data)

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        return numpy.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self) -> None:
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()  # type: ignore[attr-defined]
        if self._bin_file_reader is not None:
            self._bin_file_reader.close()
        del self._bin_buffer_mmap
        del self._bin_file_reader


class MegatronIndexedDataset(torch.utils.data.Dataset):
    """The IndexedDataset, compatible with Megatron-LM's IndexedDataset.

    Args:
        path_prefix: The path prefix to the .bin and .idx files.
        multimodal: Whether the dataset is multimodal.
        mmap: Whether to memory map the .bin file.
    """

    def __init__(
        self,
        path_prefix: str,
        multimodal: bool = False,
        mmap: bool = True,
    ) -> None:
        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self.mmap = mmap

        idx_path = path_prefix + ".idx"
        bin_path = path_prefix + ".bin"

        self.index = _IndexReader(idx_path, multimodal=multimodal)
        if mmap:
            self.bin_reader = _MMapBinReader(bin_path)
        else:
            raise NotImplementedError("Non-mmap bin reader is not implemented.")

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> numpy.ndarray:
        return self.index.document_indices

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: Union[int, numpy.integer]) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = self.bin_reader.read(
                dtype=self.index.dtype,
                count=int(sequence_length),
                offset=int(sequence_pointer),
            )
            if self.multimodal:
                return sequence, sequence_mode
            return sequence
        elif isinstance(idx, slice):
            if idx.step is not None and idx.step != 1:
                raise ValueError("Slices with step != 1 are not supported.")
            start = idx.start or 0
            stop = idx.stop or len(self)
            if start >= stop:
                return [] if not self.multimodal else ([], [])

            # Read a single contiguous block
            sequence_pointer_start, _, _ = self.index[start]
            sequence_lengths = self.index.sequence_lengths[start:stop]
            total_length = int(sequence_lengths.sum())
            sequences = self.bin_reader.read(
                dtype=self.index.dtype,
                count=total_length,
                offset=int(sequence_pointer_start),
            )
            offsets = numpy.concatenate(([0], numpy.cumsum(sequence_lengths)[:-1]))
            result = numpy.split(sequences, offsets[1:])
            if self.multimodal:
                sequence_modes = self.index.sequence_modes[start:stop]
                return result, sequence_modes
            return result
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        sequence_pointer, sequence_length, _ = self.index[idx]
        if length is None:
            length = int(sequence_length) - offset
        sequence_pointer += offset * DType.size(self.index.dtype)
        return self.bin_reader.read(
            dtype=self.index.dtype,
            count=int(length),
            offset=int(sequence_pointer),
        )
