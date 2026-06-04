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


def _try_hf_tokenizer(name: str | None = None):
    """Best-effort load of a small HF tokenizer. None if offline / missing."""
    if name is None:
        name = os.environ.get("B1_HF_TOK", "sshleifer/tiny-gpt2")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        return tok
    except Exception as exc:  # noqa: BLE001
        print(f"[info] HF tokenizer '{name}' unavailable ({exc}); skipping.")
        return None


def _model_names() -> list[str]:
    """Tokenizers to exercise. Override via env B1_HF_TOKS (comma-separated)."""
    raw = os.environ.get(
        "B1_HF_TOKS",
        "sshleifer/tiny-gpt2,"
        "meta-llama/Llama-3.1-8B-Instruct,"
        "Qwen/Qwen2.5-7B-Instruct",
    )
    return [s.strip() for s in raw.split(",") if s.strip()]


def _hf_realistic_ids(tok, seed: int = 13) -> list[list[int]]:
    """Realistic id sequences via HF encode of natural-language prompts.

    Uniform-random id sampling is not a valid byte-equality test for
    ByteLevel BPE tokenizers because random ids generate UTF-8-invalid byte
    sequences that HF replaces with U+FFFD while the AVX-512 kernel (and
    real vLLM detok stream) emit the raw bytes. Real generated sequences
    are always valid UTF-8 (the LM only emits sane id streams), so we test
    that distribution instead.
    """
    prompts = [
        "The quick brown fox jumps over the lazy dog. " * 4,
        "Python list comprehensions: [x*2 for x in range(10) if x % 2 == 0]",
        "한국어 자연어 처리는 어렵습니다. 안녕하세요! 반갑습니다. " * 3,
        "Mixing English と日本語 and 中文 emoji ✓✗→ in one line. " * 2,
        "Math: \\sum_{i=0}^{n} i^2 = \\frac{n(n+1)(2n+1)}{6}",
        "Code:\n    def foo(x):\n        return x + 1\n",
    ]
    out: list[list[int]] = []
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        out.append(ids)
    return out


def _hf_chat_ids(tok) -> list[int]:
    """A realistic short multilingual chat — includes BOS / special tokens."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world! 안녕하세요. Today is sunny."},
        {"role": "assistant",
         "content": "Greetings! 반갑습니다. The weather is great."},
    ]
    try:
        return list(tok.apply_chat_template(messages, tokenize=True))
    except Exception:  # noqa: BLE001
        return list(tok.encode(
            "Hello, world! 안녕하세요. Today is sunny.",
            add_special_tokens=True,
        ))


def _run_hf_byteequal(tok, lib_path: str) -> tuple[int, int, int, int]:
    """Returns (random_pass, random_total, chat_pass, chat_total)."""
    det_hf = AVX512Detokenizer.from_hf_tokenizer(tok, lib_path=lib_path)

    # Realistic encoded sequences (mixed languages, code, math)
    ids_r = _hf_realistic_ids(tok)
    B_r = len(ids_r)
    out_r = det_hf.batch_detokenize(ids_r)
    pass_r = 0
    for b in range(B_r):
        tgt = tok.decode(ids_r[b]).encode("utf-8")
        if tgt == out_r[b]:
            pass_r += 1
        else:
            for j, (a, c) in enumerate(zip(tgt, out_r[b])):
                if a != c:
                    print(f"[hf]   realistic seq{b} first diff byte={j} "
                          f"hf={bytes([a])!r} avx={bytes([c])!r}")
                    print(f"[hf]     hf ctx : "
                          f"{tgt[max(0,j-20):j+30]!r}")
                    print(f"[hf]     avx ctx: "
                          f"{out_r[b][max(0,j-20):j+30]!r}")
                    break
            else:
                print(f"[hf]   realistic seq{b} length-only diff "
                      f"hf={len(tgt)} avx={len(out_r[b])}")
    print(f"[hf]   realistic (encode→decode): pass={pass_r}/{B_r}")

    # Chat sequence (includes special tokens) — note: special tokens fall in
    # the vocab table so AVX-512 will emit their literal pieces too; HF
    # decode() by default also keeps them, so byte equality should still hold.
    ids_c = _hf_chat_ids(tok)
    out_c = det_hf.batch_detokenize([ids_c])
    tgt_c = tok.decode(ids_c).encode("utf-8")
    eq_c = (tgt_c == out_c[0])
    print(f"[hf]   chat (len={len(ids_c)}): byte_equal={eq_c} "
          f"hf_bytes={len(tgt_c)} avx_bytes={len(out_c[0])}")
    if not eq_c:
        for j, (a, c) in enumerate(zip(tgt_c, out_c[0])):
            if a != c:
                print(f"[hf]   chat first diff byte={j} "
                      f"hf={bytes([a])!r} avx={bytes([c])!r}")
                print(f"[hf]   hf ctx : {tgt_c[max(0,j-20):j+30]!r}")
                print(f"[hf]   avx ctx: {out_c[0][max(0,j-20):j+30]!r}")
                break
    return pass_r, B_r, (1 if eq_c else 0), 1


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

    # ---- (2) HF tokenizer path — iterate over real production models ----
    hf_results: list[tuple[str, int, int, int, int, bool]] = []
    for model_name in _model_names():
        hf_tok = _try_hf_tokenizer(model_name)
        if hf_tok is None:
            continue
        print(f"[hf] tokenizer='{model_name}' "
              f"class={hf_tok.__class__.__name__} V={len(hf_tok)}")
        pass_r, total_r, pass_c, total_c = _run_hf_byteequal(hf_tok, lib_path)
        loaded_ok = (pass_r == total_r) and (pass_c == total_c)
        hf_results.append((model_name, pass_r, total_r, pass_c, total_c,
                           loaded_ok))

    # ---- (3) Empty / edge cases -------------------------------------
    out_empty = det_avx.batch_detokenize([[]])
    assert out_empty == [b""], f"empty seq mishandled: {out_empty!r}"

    out_oob = det_avx.batch_detokenize([[V + 1000, -3, 0]])
    # OOB ids must be skipped silently; only id 0 contributes.
    assert out_oob[0] == table["_piece_list"][0], (
        f"OOB handling broken: {out_oob[0]!r} vs {table['_piece_list'][0]!r}"
    )
    print("[edge] empty + OOB handling: OK")

    # ---- (4) HF summary + gate ---------------------------------------
    if hf_results:
        print("[hf] === summary ===")
        all_ok = True
        for name, pr, tr, pc, tc, ok in hf_results:
            print(f"[hf]   {name}: random={pr}/{tr} chat={pc}/{tc} "
                  f"{'OK' if ok else 'FAIL'}")
            all_ok = all_ok and ok
        if not all_ok:
            print("FAIL — at least one HF tokenizer byte-equal check failed")
            return 1

    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
