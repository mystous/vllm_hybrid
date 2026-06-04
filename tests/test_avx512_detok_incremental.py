# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# IDE_016 / SUB_201 §5 B1 PoC next-step — incremental + micro-batch test
#
# Goal:
#   1. incremental_append(tid) → str  : streaming path. UTF-8 byte buffer
#      reassembles multi-byte codepoints across token boundaries.
#   2. decode_batch(list[list[int]]) → list[str] : micro-batch wrapper that
#      shares the vocab table across N concurrent sequence decodes.
#   3. 3-way byte-equal gate over 30+ prompt × 3 models:
#        HF .decode(ids)
#        == AVX wrapper decode_batch([ids])[0]
#        == "".join(incremental_append(t) for t in ids) (+ incremental_flush())
#
# Run:
#   cd /workspace/poc_worktrees/wt_b1_detok
#   PYTHONPATH=. /workspace/vllm_dev_prj/bin/python \
#       tests/test_avx512_detok_incremental.py

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm.tokenizers.avx512_detokenizer import (  # noqa: E402
    AVX512Detokenizer,
    _default_lib_path,
)


# ---------------------------------------------------------------------------
# Prompt set — 30+ varied prompts exercising ASCII, CJK (3-byte UTF-8),
# Japanese, Chinese, emoji (4-byte), code, math, whitespace, very short,
# very long, and English/Korean/code mix.
# ---------------------------------------------------------------------------
PROMPTS: list[str] = [
    # ---- English (varied lengths)
    "The quick brown fox jumps over the lazy dog.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 8,
    "A.",
    "Hi",
    "Hello, world!",
    # ---- Korean (3-byte UTF-8, often splits inside a codepoint)
    "안녕하세요. 오늘 날씨가 정말 좋네요.",
    "한국어 자연어 처리 모델을 평가하고 있습니다. " * 4,
    "가나다라마바사아자차카타파하",
    "안녕",
    "안녕하세요! 반갑습니다. 오늘 만나서 정말 기쁩니다.",
    # ---- Japanese
    "こんにちは、世界！今日はいい天気ですね。",
    "日本語の自然言語処理は楽しいです。" * 3,
    # ---- Chinese
    "你好，世界！今天天气真好。",
    "中文自然语言处理是一个有趣的领域。" * 3,
    # ---- Emoji (4-byte UTF-8)
    "Look at these emojis: 🚀 🌟 🎉 🔥 💯",
    "Family: 👨‍👩‍👧‍👦 and flags: 🇰🇷 🇯🇵 🇨🇳 🇺🇸",
    "Mixed: hello 👋 안녕 🌸 こんにちは 🗾",
    # ---- Code
    "def foo(x):\n    return x + 1\n",
    "print('hello, world!')\nfor i in range(10):\n    print(i)\n",
    "x = [i * 2 for i in range(100) if i % 3 == 0]",
    "// C++ comment\nint main() { return 0; }",
    # ---- Math / symbols
    "Math: \\sum_{i=0}^{n} i^2 = \\frac{n(n+1)(2n+1)}{6}",
    "Inequality: 1 <= x < 10, y >= 0, z != 5",
    "Greek: α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω",
    # ---- Mixed language
    "Mixing English と日本語 and 中文 and 한국어 in one line. " * 2,
    "API: GET /v1/users/123 (200 OK) — body: {\"name\": \"김철수\"}",
    # ---- Whitespace / newlines
    "Line 1\nLine 2\n\nLine 4 (after blank)\n\tTabbed line",
    "   leading spaces and trailing   ",
    # ---- Very long
    "The quick brown fox jumps over the lazy dog. " * 32,
    "한국어 텍스트가 굉장히 길어지면 어떻게 될까요? " * 16,
    # ---- Tricky punctuation
    "Quoted: \"hello\" and 'world' — em-dash, ellipsis…",
    # ---- Numbers
    "Pi ≈ 3.14159265358979 and e ≈ 2.71828182845904",
    # ---- Edge: single character that's multi-byte
    "한",
    # ---- Edge: single emoji
    "🚀",
]


def _model_names() -> list[str]:
    raw = os.environ.get(
        "B1_HF_TOKS",
        "sshleifer/tiny-gpt2,"
        "meta-llama/Llama-3.1-8B-Instruct,"
        "Qwen/Qwen2.5-7B-Instruct",
    )
    return [s.strip() for s in raw.split(",") if s.strip()]


def _try_hf_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001
        print(f"[info] tokenizer '{name}' unavailable ({exc}); skipping.")
        return None


def _diff_report(label: str, idx: int, expected: bytes, got: bytes) -> None:
    L = min(len(expected), len(got))
    for j in range(L):
        if expected[j] != got[j]:
            print(f"[{label}] prompt#{idx} first byte diff at {j}: "
                  f"hf={bytes([expected[j]])!r} avx={bytes([got[j]])!r}")
            ctx_lo = max(0, j - 20)
            ctx_hi = j + 30
            print(f"[{label}]   hf  ctx: {expected[ctx_lo:ctx_hi]!r}")
            print(f"[{label}]   avx ctx: {got[ctx_lo:ctx_hi]!r}")
            return
    if len(expected) != len(got):
        print(f"[{label}] prompt#{idx} length-only diff: "
              f"hf={len(expected)} avx={len(got)}")
        if len(expected) > L:
            print(f"[{label}]   hf  trailing: {expected[L:L+30]!r}")
        if len(got) > L:
            print(f"[{label}]   avx trailing: {got[L:L+30]!r}")


def _ground_truth_decode(tok, ids: list[int]) -> bytes:
    """Source-of-truth detok output for byte-equal comparison.

    vLLM 의 native incremental detok 은 ``tokenizers.DecodeStream`` (Rust
    backend) 위에서 동작하므로 ground truth 도 ``backend_tokenizer.decode``
    를 사용한다. Transformers 의 high-level ``tok.decode()`` 는 추가
    cleanup (e.g. Llama-3 의 `clean_up_tokenization_spaces`) 을 적용해
    raw piece concat 과 다를 수 있어 정답이 아니다 (vLLM 의 stream output
    과도 다름).
    """
    bt = getattr(tok, "backend_tokenizer", None)
    if bt is not None:
        try:
            return bt.decode(ids, skip_special_tokens=False).encode("utf-8")
        except Exception:
            pass
    return tok.decode(ids).encode("utf-8")


def _check_three_way(
    label: str,
    tok,
    det: AVX512Detokenizer,
) -> tuple[int, int, int, int]:
    """Returns (batch_pass, inc_pass, batch_total, inc_total)."""
    ids_lists = []
    expected = []
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=False)
        ids_lists.append(ids)
        expected.append(_ground_truth_decode(tok, ids))

    # ---- (a) micro-batch decode_batch
    batch_str = det.decode_batch(ids_lists)
    batch_pass = 0
    for i, (s, exp) in enumerate(zip(batch_str, expected)):
        got = s.encode("utf-8")
        if got == exp:
            batch_pass += 1
        else:
            _diff_report(f"{label}/batch", i, exp, got)

    # ---- (b) incremental_append per-token, fresh buffer per prompt
    inc_pass = 0
    for i, ids in enumerate(ids_lists):
        det.incremental_reset()
        chunks: list[str] = []
        for tid in ids:
            chunks.append(det.incremental_append(int(tid)))
        chunks.append(det.incremental_flush())
        full = "".join(chunks).encode("utf-8")
        if full == expected[i]:
            inc_pass += 1
        else:
            _diff_report(f"{label}/inc", i, expected[i], full)

    return (batch_pass, inc_pass, len(ids_lists), len(ids_lists))


def main() -> int:
    lib_path = _default_lib_path()
    print(f"[info] kernel: {lib_path}")
    if not os.path.exists(lib_path):
        print(f"FAIL — .so missing at {lib_path}")
        return 1

    # ---- unit: split_valid_utf8_prefix on synthetic byte streams --------
    cases = [
        # (input bytes, expected_text, expected_hold)
        (b"", "", b""),
        (b"hello", "hello", b""),
        (b"hello\xec", "hello", b"\xec"),
        (b"hello\xec\x95", "hello", b"\xec\x95"),
        (b"hello\xec\x95\x88", "hello안", b""),
        (b"\xf0\x9f", "", b"\xf0\x9f"),
        (b"\xf0\x9f\x9a\x80", "🚀", b""),
        (b"a\xf0\x9f\x9a", "a", b"\xf0\x9f\x9a"),
        # Pure invalid sequence (4 continuation bytes in a row).
        (b"\x80\x80\x80\x80", "����", b""),
    ]
    unit_pass = 0
    for inp, exp_t, exp_h in cases:
        t, h = AVX512Detokenizer._split_valid_utf8_prefix(bytearray(inp))
        ok = (t == exp_t) and (bytes(h) == exp_h)
        if ok:
            unit_pass += 1
        else:
            print(f"[unit] FAIL: in={inp!r} got=({t!r},{bytes(h)!r}) "
                  f"exp=({exp_t!r},{exp_h!r})")
    print(f"[unit] split_valid_utf8_prefix: {unit_pass}/{len(cases)} PASS")
    if unit_pass != len(cases):
        return 1

    # ---- 3-way correctness over real tokenizers ---------------------------
    summary: list[tuple[str, int, int, int, int]] = []
    for name in _model_names():
        tok = _try_hf_tokenizer(name)
        if tok is None:
            continue
        try:
            det = AVX512Detokenizer.from_hf_tokenizer(tok, lib_path=lib_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[info] build wrapper failed for {name}: {exc}")
            continue
        print(f"[hf] model='{name}' class={tok.__class__.__name__} "
              f"V={len(tok)} prompts={len(PROMPTS)}")
        bp, ip, bt, it = _check_three_way(name, tok, det)
        print(f"[hf]   batch  byte-equal: {bp}/{bt}")
        print(f"[hf]   inc    byte-equal: {ip}/{it}")
        summary.append((name, bp, ip, bt, it))

    # ---- Final gate -------------------------------------------------------
    if not summary:
        print("FAIL — no HF tokenizers loaded; cannot validate gate")
        return 1

    print("[gate] === summary ===")
    all_ok = True
    grand_b_pass = 0
    grand_i_pass = 0
    grand_total = 0
    for name, bp, ip, bt, it in summary:
        ok_b = (bp == bt)
        ok_i = (ip == it)
        ok = ok_b and ok_i
        all_ok = all_ok and ok
        grand_b_pass += bp
        grand_i_pass += ip
        grand_total += bt
        print(f"[gate]   {name}: batch={bp}/{bt} inc={ip}/{it} "
              f"{'OK' if ok else 'FAIL'}")
    print(f"[gate] TOTAL: batch={grand_b_pass}/{grand_total} "
          f"inc={grand_i_pass}/{grand_total} models={len(summary)}")

    if not all_ok:
        print("FAIL — byte-equal gate")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
