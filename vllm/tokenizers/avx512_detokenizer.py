# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# IDE_016 / SUB_201 §5 B1 PoC — ctypes wrapper for the AVX-512 detokenizer
# kernel (libavx512_tokenizer.so).
#
# Design:
#   * 코어 vLLM 인터페이스 변경을 최소화하기 위해 외부 .so 를 ctypes 로 로드한다.
#   * `AVX512Detokenizer` 는 HF tokenizer 로부터 vocab table 한 번 build 후
#     `batch_detokenize(token_ids, seq_offsets) -> list[bytes]` 호출을 제공.
#   * 통합은 detokenizer.py 의 env flag `VLLM_USE_AVX512_DETOK=1` 로 gated.

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_DEFAULT_LIB_REL = (
    "shadow_assists/features/IDE_016_avx512_amx_pool/"
    "build/avx512_tokenizer/libavx512_tokenizer.so"
)


def _default_lib_path() -> str:
    """Resolve the .so relative to the repo root (works in worktree)."""
    here = Path(__file__).resolve()
    # vllm/tokenizers/avx512_detokenizer.py → repo root is 2 parents up.
    repo_root = here.parents[2]
    candidate = repo_root / _DEFAULT_LIB_REL
    return str(candidate)


class AVX512Detokenizer:
    """Thin ctypes wrapper around the AVX-512 batch detokenize kernel.

    Parameters
    ----------
    lib_path : str | None
        Path to ``libavx512_tokenizer.so``. If None, falls back to the
        in-tree build artifact resolved relative to the repo root.
    vocab_table : dict
        Pre-built vocab table with keys ``pieces`` (bytes), ``offsets``
        (int32 ndarray of length V+1), ``sizes`` (int32 ndarray of length V).
        Use :meth:`from_hf_tokenizer` to build one from a HuggingFace
        ``PreTrainedTokenizerFast`` (or any object exposing ``get_vocab`` and
        ``convert_tokens_to_string``-compatible bytes via
        ``convert_ids_to_tokens``).
    use_avx512 : bool
        If True, use AVX-512 code path; otherwise scalar fallback.
    """

    def __init__(
        self,
        lib_path: str | None,
        vocab_table: dict,
        use_avx512: bool = True,
    ) -> None:
        self.lib_path = lib_path or _default_lib_path()
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(
                f"AVX-512 detokenizer .so not found at {self.lib_path}"
            )
        self._lib = ctypes.CDLL(self.lib_path)
        self._configure_signatures()

        # Vocab table layout (keep ndarrays alive on the instance).
        self._pieces: np.ndarray = np.frombuffer(
            vocab_table["pieces"], dtype=np.uint8
        )
        self._offsets: np.ndarray = np.ascontiguousarray(
            vocab_table["offsets"], dtype=np.int32
        )
        self._sizes: np.ndarray = np.ascontiguousarray(
            vocab_table["sizes"], dtype=np.int32
        )
        if self._offsets.shape[0] != self._sizes.shape[0] + 1:
            raise ValueError(
                f"offsets length must be sizes length + 1; got "
                f"{self._offsets.shape[0]} vs {self._sizes.shape[0]}"
            )
        self._V = int(self._sizes.shape[0])
        self._total_bytes = int(self._pieces.shape[0])
        self.use_avx512 = bool(use_avx512)

    # ------------------------------------------------------------------
    # ctypes signature wiring
    # ------------------------------------------------------------------
    def _configure_signatures(self) -> None:
        u8p = ctypes.POINTER(ctypes.c_uint8)
        i32p = ctypes.POINTER(ctypes.c_int32)

        self._lib.avx512_batch_detokenize_bytes.restype = None
        self._lib.avx512_batch_detokenize_bytes.argtypes = [
            u8p,                # pieces
            i32p,               # offsets
            i32p,               # sizes
            ctypes.c_int32,     # V
            ctypes.c_int32,     # total_bytes
            i32p,               # token_ids
            i32p,               # seq_offsets
            ctypes.c_int,       # B
            u8p,                # out_bytes
            i32p,               # out_byte_offsets
            i32p,               # out_byte_lengths
            ctypes.c_int,       # use_avx512
        ]

        self._lib.avx512_batch_detokenize_byte_total.restype = ctypes.c_int64
        self._lib.avx512_batch_detokenize_byte_total.argtypes = [
            u8p, i32p, i32p, ctypes.c_int32, ctypes.c_int32,
            i32p, ctypes.c_int,
        ]

    # ------------------------------------------------------------------
    # Vocab table construction from HF tokenizer
    # ------------------------------------------------------------------
    @classmethod
    def build_vocab_table(cls, hf_tokenizer) -> dict:
        """Build the flat (pieces, offsets, sizes) table from HF tokenizer.

        Each token i's bytes are obtained via
        ``hf_tokenizer.decode([i]).encode('utf-8')``. This matches the
        downstream output bytes that the kernel reproduces by concatenation
        of piece bytes, modulo BPE merge / leading-space heuristics handled
        upstream of detokenize.
        """
        V = len(hf_tokenizer)
        piece_bytes: list[bytes] = []
        for tid in range(V):
            try:
                s = hf_tokenizer.decode([tid])
            except Exception:
                s = ""
            piece_bytes.append(s.encode("utf-8", errors="replace"))

        sizes = np.fromiter(
            (len(b) for b in piece_bytes), dtype=np.int32, count=V
        )
        offsets = np.zeros(V + 1, dtype=np.int32)
        np.cumsum(sizes, dtype=np.int32, out=offsets[1:])
        pieces = b"".join(piece_bytes)
        return {"pieces": pieces, "offsets": offsets, "sizes": sizes}

    @classmethod
    def from_hf_tokenizer(
        cls,
        hf_tokenizer,
        lib_path: str | None = None,
        use_avx512: bool = True,
    ) -> "AVX512Detokenizer":
        table = cls.build_vocab_table(hf_tokenizer)
        return cls(lib_path=lib_path, vocab_table=table, use_avx512=use_avx512)

    # ------------------------------------------------------------------
    # Batch detokenize
    # ------------------------------------------------------------------
    def batch_detokenize(
        self,
        token_ids,
        seq_offsets=None,
    ) -> list[bytes]:
        """Detokenize a batch of sequences.

        Parameters
        ----------
        token_ids : list[list[int]] | np.ndarray
            Either a list of per-sequence token id lists, or a flat int32
            ndarray combined with ``seq_offsets``.
        seq_offsets : np.ndarray | None
            If ``token_ids`` is a flat ndarray, this must be an int32 array
            of length ``B+1`` giving the prefix offsets per sequence.

        Returns
        -------
        list[bytes]
            ``B`` sequences of UTF-8 bytes.
        """
        if seq_offsets is None:
            # List-of-lists input — flatten.
            if not isinstance(token_ids, (list, tuple)):
                raise TypeError(
                    "token_ids must be list[list[int]] when seq_offsets is None"
                )
            B = len(token_ids)
            lengths = np.fromiter(
                (len(s) for s in token_ids), dtype=np.int32, count=B
            )
            seq_offsets_arr = np.zeros(B + 1, dtype=np.int32)
            np.cumsum(lengths, dtype=np.int32, out=seq_offsets_arr[1:])
            total = int(seq_offsets_arr[-1])
            token_arr = np.empty(total, dtype=np.int32)
            cursor = 0
            for seq in token_ids:
                n = len(seq)
                if n:
                    token_arr[cursor:cursor + n] = np.asarray(seq, dtype=np.int32)
                cursor += n
        else:
            token_arr = np.ascontiguousarray(token_ids, dtype=np.int32)
            seq_offsets_arr = np.ascontiguousarray(seq_offsets, dtype=np.int32)
            B = seq_offsets_arr.shape[0] - 1
            total = int(seq_offsets_arr[-1])

        # Conservative output buffer: sum of all piece sizes for those ids.
        # Use the cheap kernel helper to get an exact tight upper bound.
        total_bytes_out = self._byte_total(token_arr, total)
        out_bytes = np.zeros(max(int(total_bytes_out), 1), dtype=np.uint8)
        out_byte_offsets = np.zeros(B + 1, dtype=np.int32)
        out_byte_lengths = np.zeros(B, dtype=np.int32)

        u8p = ctypes.POINTER(ctypes.c_uint8)
        i32p = ctypes.POINTER(ctypes.c_int32)

        self._lib.avx512_batch_detokenize_bytes(
            self._pieces.ctypes.data_as(u8p),
            self._offsets.ctypes.data_as(i32p),
            self._sizes.ctypes.data_as(i32p),
            ctypes.c_int32(self._V),
            ctypes.c_int32(self._total_bytes),
            token_arr.ctypes.data_as(i32p),
            seq_offsets_arr.ctypes.data_as(i32p),
            ctypes.c_int(B),
            out_bytes.ctypes.data_as(u8p),
            out_byte_offsets.ctypes.data_as(i32p),
            out_byte_lengths.ctypes.data_as(i32p),
            ctypes.c_int(1 if self.use_avx512 else 0),
        )

        results: list[bytes] = []
        for b in range(B):
            lo = int(out_byte_offsets[b])
            hi = int(out_byte_offsets[b + 1])
            results.append(out_bytes[lo:hi].tobytes())
        return results

    # ------------------------------------------------------------------
    # Helper for sizing the output buffer
    # ------------------------------------------------------------------
    def _byte_total(self, token_arr: np.ndarray, total_tokens: int) -> int:
        if total_tokens == 0:
            return 0
        u8p = ctypes.POINTER(ctypes.c_uint8)
        i32p = ctypes.POINTER(ctypes.c_int32)
        return int(self._lib.avx512_batch_detokenize_byte_total(
            self._pieces.ctypes.data_as(u8p),
            self._offsets.ctypes.data_as(i32p),
            self._sizes.ctypes.data_as(i32p),
            ctypes.c_int32(self._V),
            ctypes.c_int32(self._total_bytes),
            token_arr.ctypes.data_as(i32p),
            ctypes.c_int(int(total_tokens)),
        ))
