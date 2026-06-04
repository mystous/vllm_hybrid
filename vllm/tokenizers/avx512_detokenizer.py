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


# ---------------------------------------------------------------------------
# GPT-2 / ByteLevel byte_decoder
# ---------------------------------------------------------------------------
# Llama-3, Qwen2.5, GPT-2, Mistral, … all use HF ``ByteLevel`` decoder. Their
# token strings are *unicode-mapped bytes*: each char in the piece string maps
# back to a single raw byte via the standard GPT-2 bytes_to_unicode() table.
#
# For correct byte-exact detokenize, the vocab table the AVX-512 kernel sees
# must contain the *raw byte sequence* per token (e.g. the Korean syllable
# pieces "ì","ķ","Ī" → 0xEC 0x95 0x88), NOT each piece's UTF-8 encoding which
# would emit U+FFFD when a piece is only a fragment of a multi-byte codepoint.
def _gpt2_byte_decoder() -> dict[str, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _gpt2_byte_decoder()


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
        # Incremental streaming state (see DESIGN_B1_DETOK_INC.md §2).
        #
        # ``incremental_append(token_id)`` accumulates raw piece bytes into
        # ``_inc_buf`` and emits the longest valid-UTF-8 prefix; any trailing
        # incomplete codepoint bytes stay buffered until later tokens
        # complete them. This mirrors HF ``DecodeStream.step()`` semantics
        # and is what allows ByteLevel BPE (Llama-3 / Qwen-2.5 / GPT-2) to
        # be detokenized one token at a time without splitting multi-byte
        # characters.
        # ------------------------------------------------------------------
        self._inc_buf: bytearray = bytearray()
        self._inc_emit_chars: int = 0

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
    def _detect_bytelevel(cls, hf_tokenizer) -> bool:
        """Return True if the HF tokenizer uses a ByteLevel decoder.

        Production targets (Llama-3 / Qwen2.5 / GPT-2 / Mistral) all do.
        """
        bt = getattr(hf_tokenizer, "backend_tokenizer", None)
        if bt is None:
            return False
        dec = getattr(bt, "decoder", None)
        if dec is None:
            return False
        return type(dec).__name__ == "ByteLevel"

    @classmethod
    def build_vocab_table(cls, hf_tokenizer) -> dict:
        """Build the flat (pieces, offsets, sizes) table from HF tokenizer.

        For ByteLevel tokenizers (GPT-2 / Llama-3 / Qwen2.5 / Mistral) the
        per-token raw bytes are obtained via the standard GPT-2
        bytes_to_unicode() decoder applied to ``convert_ids_to_tokens(i)``.
        This is byte-exact w.r.t. ``hf_tokenizer.decode(seq)`` for any
        sequence whose tokens are not in the added/special set, and matches
        what the downstream vLLM detokenize stream accumulates anyway.

        For non-ByteLevel tokenizers we fall back to the previous
        per-id decode → utf-8 path. (Pure-SentencePiece models with `▁`
        normalization are not in the immediate target set; flagged in §3 of
        the report for follow-up.)

        Added / special tokens (BOS, EOS, `<|im_start|>`, …) are emitted
        with the **literal piece bytes** (i.e. the displayed marker UTF-8)
        when they fall outside the base byte-level vocab, which matches
        ``hf_tokenizer.decode(skip_special_tokens=False)``.
        """
        V = len(hf_tokenizer)
        bytelevel = cls._detect_bytelevel(hf_tokenizer)

        special_ids = set(hf_tokenizer.all_special_ids)
        added_ids = set(getattr(hf_tokenizer, "added_tokens_decoder", {}).keys())

        piece_bytes: list[bytes] = []
        if bytelevel:
            try:
                toks_all = hf_tokenizer.convert_ids_to_tokens(list(range(V)))
            except Exception:
                toks_all = [hf_tokenizer.convert_ids_to_tokens(i)
                            for i in range(V)]
            for tid, t in enumerate(toks_all):
                if t is None:
                    piece_bytes.append(b"")
                    continue
                if tid in special_ids or tid in added_ids:
                    # Special tokens: keep the literal text (UTF-8) so that
                    # decode(skip_special=False) byte-equality holds. When
                    # the upper layer chooses to skip them, it'll do so
                    # before calling batch_detokenize (the kernel itself is
                    # vocab-agnostic).
                    piece_bytes.append(t.encode("utf-8", errors="replace"))
                    continue
                # ByteLevel raw byte reconstruction.
                try:
                    piece_bytes.append(
                        bytes(_BYTE_DECODER[ch] for ch in t)
                    )
                except KeyError:
                    # Token contains a char not in the byte_decoder map →
                    # treat as literal UTF-8 (matches HF fallback).
                    piece_bytes.append(t.encode("utf-8", errors="replace"))
        else:
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
        return {
            "pieces": pieces,
            "offsets": offsets,
            "sizes": sizes,
            "bytelevel": bytelevel,
        }

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
    # Incremental streaming path (see DESIGN_B1_DETOK_INC.md §2)
    # ------------------------------------------------------------------
    def _piece_bytes(self, token_id: int) -> bytes:
        """Return raw piece bytes for ``token_id`` (silent on OOB)."""
        if token_id < 0 or token_id >= self._V:
            return b""
        lo = int(self._offsets[token_id])
        hi = int(self._offsets[token_id + 1])
        if hi <= lo:
            return b""
        return self._pieces[lo:hi].tobytes()

    @staticmethod
    def _split_valid_utf8_prefix(buf: bytearray) -> tuple[str, bytearray]:
        """Split ``buf`` at the latest UTF-8 codepoint boundary.

        Returns ``(text, hold)`` where ``text`` is the decoded prefix that
        is safe to emit and ``hold`` is the trailing incomplete-codepoint
        bytes that must remain buffered until further tokens arrive.

        Mirrors HF ``DecodeStream`` 's incremental UTF-8 boundary policy.
        """
        n = len(buf)
        if n == 0:
            return ("", bytearray())

        # Back-scan up to 4 bytes (max UTF-8 codepoint length) for the
        # latest valid boundary.
        split = -1
        max_scan = min(4, n)
        for k in range(1, max_scan + 1):
            b = buf[n - k]
            if b < 0x80:
                # ASCII byte at position (n-k) — boundary is right AFTER it.
                split = n - k + 1
                break
            if b >= 0xC0:
                # Lead byte. Determine expected length.
                if b < 0xE0:
                    need = 2
                elif b < 0xF0:
                    need = 3
                else:
                    need = 4
                if k >= need:
                    # Complete sequence — emit everything.
                    split = n
                else:
                    # Incomplete — boundary is BEFORE this lead byte.
                    split = n - k
                break
            # Continuation byte (0x80-0xBF) — keep scanning back.
        if split < 0:
            # 4 continuation bytes in a row → invalid; flush all so the
            # caller sees U+FFFD now rather than holding indefinitely.
            split = n

        emit_bytes = bytes(buf[:split])
        hold_bytes = bytearray(buf[split:])
        # ``errors="replace"`` matches HF .decode() default for legacy
        # bytes that turn out invalid (e.g. lone continuation from
        # special-token literal pieces).
        text = emit_bytes.decode("utf-8", errors="replace")
        return (text, hold_bytes)

    def incremental_append(self, token_id: int) -> str:
        """Append one token and return any newly emit-able text.

        Maintains per-instance UTF-8 byte buffer so multi-byte codepoints
        that span multiple tokens (common in ByteLevel BPE for CJK / emoji)
        are reassembled correctly. Returned string concatenated across all
        calls is byte-equal to ``hf_tokenizer.decode(all_ids)``.
        """
        raw = self._piece_bytes(int(token_id))
        if raw:
            self._inc_buf.extend(raw)
        text, hold = self._split_valid_utf8_prefix(self._inc_buf)
        self._inc_buf = hold
        if text:
            self._inc_emit_chars += len(text)
        return text

    def incremental_flush(self) -> str:
        """Emit any remaining buffered bytes (used at sequence end).

        Trailing invalid bytes become U+FFFD via ``errors='replace'``,
        matching HF ``.decode()`` behavior when fed an incomplete sequence.
        """
        if not self._inc_buf:
            return ""
        text = bytes(self._inc_buf).decode("utf-8", errors="replace")
        self._inc_buf = bytearray()
        if text:
            self._inc_emit_chars += len(text)
        return text

    def incremental_reset(self) -> None:
        """Reset the per-instance incremental buffer (new sequence)."""
        self._inc_buf = bytearray()
        self._inc_emit_chars = 0

    # ------------------------------------------------------------------
    # Micro-batch decode path (see DESIGN_B1_DETOK_INC.md §3)
    # ------------------------------------------------------------------
    def decode_batch(self, token_id_lists) -> list[str]:
        """Decode N concurrent sequences in one AVX-512 batch call.

        Same semantics as ``[hf.decode(ids) for ids in token_id_lists]``
        but vocab is loaded once → better cache locality + fewer Python ↔
        C transitions when N is large.

        Invalid UTF-8 bytes are emitted as U+FFFD (matches HF
        ``.decode()`` default ``errors='replace'``).
        """
        raw_bytes_list = self.batch_detokenize(token_id_lists)
        return [b.decode("utf-8", errors="replace") for b in raw_bytes_list]

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
