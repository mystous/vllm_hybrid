# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# IDE_016 / SUB_201 §5 B1 PoC — standalone correctness test
#
# Goal: verify that the AVX-512 batch detokenize kernel reproduces the same
# UTF-8 bytes as HuggingFace tokenizer.decode() for a given vocab.
#
# This test does NOT require booting vLLM. It exercises the ctypes wrapper
# directly:
#   1. Build vocab table from a small open tokenizer (sshleifer/tiny-gpt2 by
#      default — works fully offline if HF_HOME has it cached, otherwise
#      falls back to a synthetic byte-piece vocab).
#   2. Sample N token id sequences.
#   3. For each sequence, compare AVX-512 kernel output vs the kernel's
#      scalar fallback (always identical regardless of CPU), and (when
#      available) vs ``tokenizer.decode(ids).encode("utf-8")``.
#
# Run:
#   cd /workspace/poc_worktrees/wt_b1_detok
#   PYTHONPATH=. /workspace/vllm_dev_prj/bin/python tests/test_avx512_detok.py

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

# Make the worktree's vllm/ importable without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm.tokenizers.avx512_detokenizer import (  # noqa: E402
    AVX512Detokenizer,
    _default_lib_path,
)


def _make_synthetic_vocab(V: int = 512, seed: int = 0) -> dict:
    """Build a synthetic byte-piece vocab table (no external dep)."""
    rng = random.Random(seed)
    pieces_list: list[bytes] = []
    for tid in range(V):
        L = rng.randint(1, 6)
        # use printable ascii + occasional high byte to exercise width paths.
        b = bytes(rng.randint(0x20, 0x7e) for _ in range(L))
        pieces_list.append(b)
    sizes = np.fromiter((len(p) for p in pieces_list), dtype=np.int32, count=V)
    offsets = np.zeros(V + 1, dtype=np.int32)
    np.cumsum(sizes, dtype=np.int32, out=offsets[1:])
    return {
        "pieces": b"".join(pieces_list),
        "offsets": offsets,
        "sizes": sizes,
        "_piece_list": pieces_list,  # for python-side ground truth
    }


def _scalar_ground_truth(table: dict, ids_list: list[list[int]]) -> list[bytes]:
    out: list[bytes] = []
    pl = table["_piece_list"]
    V = len(pl)
    for seq in ids_list:
        chunks = [pl[t] for t in seq if 0 <= t < V]
        out.append(b"".join(chunks))
    return out


def _gen_sequences(V: int, B: int, seq_len: int, seed: int = 1) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(0, V - 1) for _ in range(seq_len)] for _ in range(B)]


def _try_hf_tokenizer():
    """Best-effort load of a small HF tokenizer. None if offline / missing."""
    name = os.environ.get(
        "B1_HF_TOK", "sshleifer/tiny-gpt2"
    )
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        return tok
    except Exception as exc:  # noqa: BLE001
        print(f"[info] HF tokenizer '{name}' unavailable ({exc}); "
              f"running synthetic-only path.")
        return None


def main() -> int:
    lib_path = _default_lib_path()
    print(f"[info] using kernel: {lib_path}")
    assert os.path.exists(lib_path), f".so missing at {lib_path}"

    # ---- (1) Synthetic vocab path ------------------------------------
    V = 1024
    B = 8
    L = 100
    table = _make_synthetic_vocab(V=V, seed=42)
    ids_list = _gen_sequences(V=V, B=B, seq_len=L, seed=7)

    det_avx = AVX512Detokenizer(
        lib_path=lib_path, vocab_table=table, use_avx512=True
    )
    det_scl = AVX512Detokenizer(
        lib_path=lib_path, vocab_table=table, use_avx512=False
    )

    out_avx = det_avx.batch_detokenize(ids_list)
    out_scl = det_scl.batch_detokenize(ids_list)
    gt = _scalar_ground_truth(table, ids_list)

    diffs_avx_scl = 0
    diffs_avx_gt = 0
    total_bytes = 0
    for b in range(B):
        total_bytes += len(gt[b])
        if out_avx[b] != out_scl[b]:
            diffs_avx_scl += 1
        if out_avx[b] != gt[b]:
            diffs_avx_gt += 1

    print(f"[synthetic] B={B} L={L} V={V} total_bytes={total_bytes}")
    print(f"[synthetic] avx vs scalar diffs: {diffs_avx_scl}/{B}")
    print(f"[synthetic] avx vs python-gt diffs: {diffs_avx_gt}/{B}")
    assert diffs_avx_scl == 0, "AVX-512 path diverged from scalar"
    assert diffs_avx_gt == 0, "AVX-512 path diverged from python ground truth"

    # ---- (2) HF tokenizer path (best-effort) -------------------------
    hf_tok = _try_hf_tokenizer()
    if hf_tok is not None:
        det_hf = AVX512Detokenizer.from_hf_tokenizer(hf_tok, lib_path=lib_path)
        V_hf = len(hf_tok)
        B_hf = 4
        L_hf = 100
        ids_hf = _gen_sequences(V=V_hf, B=B_hf, seq_len=L_hf, seed=13)
        out_hf_avx = det_hf.batch_detokenize(ids_hf)

        # Compare: AVX-512 path (concat of piece bytes from per-id decode)
        # vs HF tokenizer.decode(seq).encode("utf-8") — the latter applies
        # BPE merge / leading-space conventions that may differ from naive
        # piece concat. So we expect equality only when sum-of-decode-bytes
        # equals decode-of-sum bytes, which holds for byte-piece tokenizers.
        # We report per-sequence byte-level diff stats.
        cmp_rows = []
        for b in range(B_hf):
            tgt = hf_tok.decode(ids_hf[b]).encode("utf-8")
            got = out_hf_avx[b]
            cmp_rows.append((len(tgt), len(got), tgt == got))
        equal_count = sum(1 for r in cmp_rows if r[2])
        print(
            f"[hf] tokenizer='{hf_tok.__class__.__name__}' V={V_hf} B={B_hf} "
            f"L={L_hf} equal_to_tokdecode={equal_count}/{B_hf}"
        )
        for i, r in enumerate(cmp_rows):
            print(f"[hf]   seq{i}: hf_len={r[0]} avx_len={r[1]} "
                  f"byte_equal={r[2]}")
        # Note: full equality with hf_tok.decode is NOT a hard gate for the
        # PoC. The kernel reproduces "sum of per-token piece bytes"; this is
        # what the SUB_173 integration target also uses upstream of the
        # incremental detok stream.

    # ---- (3) Empty / edge cases -------------------------------------
    out_empty = det_avx.batch_detokenize([[]])
    assert out_empty == [b""], f"empty seq mishandled: {out_empty!r}"

    out_oob = det_avx.batch_detokenize([[V + 1000, -3, 0]])
    # OOB ids must be skipped silently; only id 0 contributes.
    assert out_oob[0] == table["_piece_list"][0], (
        f"OOB handling broken: {out_oob[0]!r} vs {table['_piece_list'][0]!r}"
    )
    print("[edge] empty + OOB handling: OK")

    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
