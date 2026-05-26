# IDE_016 / TSK_024 / SUB_171 — Python wrapper for AVX-512 batch detokenize.
#
# Usage:
#   from avx512_amx_pool.tokenizer import BatchDetokenizer
#   d = BatchDetokenizer.from_hf_tokenizer(hf_tok)
#   out_strs = d.batch_decode([[101, 202, ...], [303, 404, ...]])
#
# Fallback path: 빌드 안 됐거나 cpu_has_avx512 == False 면 Python tokenizer
# .decode 로 직접 fallback. unit test 에서 fallback 경로도 검증.

from __future__ import annotations

from typing import Sequence

import numpy as np


# Try to import the compiled module; if it's missing, only fallback is available.
try:
    from avx512_amx_pool import _core as _C  # type: ignore[attr-defined]

    _C_AVAILABLE = True
    _HAVE_AVX512 = bool(_C.cpu_has_avx512())
except Exception:  # pragma: no cover - dev fallback
    _C = None
    _C_AVAILABLE = False
    _HAVE_AVX512 = False


class BatchDetokenizer:
    """Wraps a HuggingFace tokenizer so that batch detokenize uses the
    AVX-512 kernel.  Falls back to ``tokenizer.decode`` when the kernel is
    unavailable.

    Construction is one-shot per tokenizer because the vocab table flattening
    pass is O(V·avg_piece_bytes); the resulting numpy arrays are reused on
    every call.
    """

    def __init__(
        self,
        pieces: np.ndarray,
        offsets: np.ndarray,
        sizes: np.ndarray,
        py_fallback_decode=None,
    ):
        assert pieces.dtype == np.uint8 and pieces.ndim == 1
        assert offsets.dtype == np.int32 and offsets.ndim == 1
        assert sizes.dtype == np.int32 and sizes.ndim == 1
        self._pieces = pieces
        self._offsets = offsets
        self._sizes = sizes
        self._vocab_size = int(sizes.shape[0])
        self._py_fallback = py_fallback_decode

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def have_avx512(self) -> bool:
        return _HAVE_AVX512

    # ────────────────────────────────────────────────────────────────
    # Vocab table construction
    # ────────────────────────────────────────────────────────────────

    @classmethod
    def from_hf_tokenizer(cls, hf_tok) -> "BatchDetokenizer":
        """Build a vocab table from a HuggingFace ``PreTrainedTokenizerFast``.

        Each piece is encoded with ``convert_ids_to_tokens`` to get the raw
        token string, then UTF-8 encoded.  Special-char handling (BPE markers
        like ``Ġ`` for spaces) matches what HF tokenizer.decode does — that is
        token-level exact match, byte-for-byte.
        """
        vocab_size = hf_tok.vocab_size if hasattr(hf_tok, "vocab_size") else len(hf_tok)
        # `convert_ids_to_tokens` returns the raw token (may contain BPE space
        # marker characters).  This matches detokenize_incrementally byte layout.
        pieces_list = []
        sizes = np.zeros(vocab_size, dtype=np.int32)
        offsets = np.zeros(vocab_size + 1, dtype=np.int32)
        running = 0
        for tid in range(vocab_size):
            tok = hf_tok.convert_ids_to_tokens(tid)
            if tok is None:
                b = b""
            else:
                b = tok.encode("utf-8", errors="replace")
            pieces_list.append(b)
            sizes[tid] = len(b)
            running += len(b)
            offsets[tid + 1] = running

        pieces = np.frombuffer(b"".join(pieces_list), dtype=np.uint8).copy()

        def _fallback(token_ids_batch: Sequence[Sequence[int]]):
            return [hf_tok.decode(ids, skip_special_tokens=False)
                    for ids in token_ids_batch]

        return cls(pieces, offsets, sizes, py_fallback_decode=_fallback)

    @classmethod
    def from_vocab_strings(cls, vocab_strings: Sequence[str]) -> "BatchDetokenizer":
        """Build directly from a list of token strings (tests / synthetic vocab)."""
        sizes = np.zeros(len(vocab_strings), dtype=np.int32)
        offsets = np.zeros(len(vocab_strings) + 1, dtype=np.int32)
        chunks = []
        running = 0
        for i, s in enumerate(vocab_strings):
            b = s.encode("utf-8", errors="replace")
            chunks.append(b)
            sizes[i] = len(b)
            running += len(b)
            offsets[i + 1] = running
        pieces = np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()

        def _fallback(token_ids_batch):
            return [
                "".join(vocab_strings[t] if 0 <= t < len(vocab_strings) else ""
                        for t in ids)
                for ids in token_ids_batch
            ]

        return cls(pieces, offsets, sizes, py_fallback_decode=_fallback)

    # ────────────────────────────────────────────────────────────────
    # Batch decode
    # ────────────────────────────────────────────────────────────────

    def _flatten(self, token_ids_batch):
        """Flatten nested list/array to (token_ids_flat, seq_offsets)."""
        seq_offsets = np.zeros(len(token_ids_batch) + 1, dtype=np.int32)
        for i, ids in enumerate(token_ids_batch):
            seq_offsets[i + 1] = seq_offsets[i] + len(ids)
        total = int(seq_offsets[-1])
        flat = np.empty(total, dtype=np.int32)
        cur = 0
        for ids in token_ids_batch:
            n = len(ids)
            flat[cur:cur + n] = np.asarray(ids, dtype=np.int32)
            cur += n
        return flat, seq_offsets

    def batch_decode(self, token_ids_batch, *, force_fallback: bool = False):
        """Decode a batch of token id sequences to UTF-8 strings.

        token_ids_batch : Sequence[Sequence[int]]
        force_fallback  : if True, use the python tokenizer.decode path
                          (used as oracle in tests).
        """
        if force_fallback or not _C_AVAILABLE or not _HAVE_AVX512:
            if self._py_fallback is not None:
                return self._py_fallback(token_ids_batch)
            # last-resort manual concat using vocab table
            return self._scalar_decode(token_ids_batch)

        token_ids_flat, seq_offsets = self._flatten(token_ids_batch)
        return _C.batch_detokenize_strings(
            self._pieces,
            self._offsets,
            self._sizes,
            token_ids_flat,
            seq_offsets,
            True,
        )

    def batch_decode_bytes(self, token_ids_batch, *, use_avx512: bool = True):
        """Return raw bytes per sequence (bypass UTF-8 decode cost).

        Useful when the caller wants to feed bytes directly into the next
        pipeline stage (e.g. vLLM detokenizer where downstream is BufferedString).
        """
        if not _C_AVAILABLE:
            return [s.encode("utf-8", errors="replace")
                    for s in self._scalar_decode(token_ids_batch)]

        token_ids_flat, seq_offsets = self._flatten(token_ids_batch)
        out_bytes, out_offs, out_lens = _C.batch_detokenize_bytes(
            self._pieces,
            self._offsets,
            self._sizes,
            token_ids_flat,
            seq_offsets,
            bool(use_avx512 and _HAVE_AVX512),
        )
        # slice into per-sequence bytes
        result = []
        ob = out_bytes.tobytes()
        for i in range(len(out_lens)):
            result.append(ob[out_offs[i]:out_offs[i + 1]])
        return result

    def _scalar_decode(self, token_ids_batch):
        """Pure-Python decode using the flattened vocab table.  Last-resort
        fallback when neither the kernel nor a HF tokenizer is available."""
        out = []
        for ids in token_ids_batch:
            chunks = []
            for tid in ids:
                if 0 <= tid < self._vocab_size:
                    lo = int(self._offsets[tid])
                    hi = int(self._offsets[tid + 1])
                    chunks.append(self._pieces[lo:hi].tobytes())
            out.append(b"".join(chunks).decode("utf-8", errors="replace"))
        return out


__all__ = ["BatchDetokenizer"]
