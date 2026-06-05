# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# IDE_016 / SUB_201 §5 B1 PoC Phase A4-exclusive — NATIVE_EXCLUSIVE smoke test.
#
# Goal
# ----
#   FastIncrementalDetokenizer._protected_step 의
#   ``VLLM_USE_AVX512_DETOK_EXCLUSIVE=1`` 분기 (native stream.step skip + AVX-only
#   fast path) 와 그 fallback (예외 시 lazy reconstruct + native downgrade) 가
#   기대대로 동작하는지 단위 수준에서 검증한다.
#
# Coverage
# --------
#   1) default-off (EXCLUSIVE/NATIVE/INC 모두 0) → 기존 동작 그대로,
#      exclusive 카운터 미증가.
#   2) EXCLUSIVE=1 정상 path: 5 prompt × byte-equal vs HF backend,
#      step_count == total_tokens, fallback_count == 0,
#      reconstruct_count == 0.
#   3) EXCLUSIVE=1 + 강제 예외: monkeypatched ``incremental_append`` 가
#      특정 step 에서 RuntimeError 를 던지도록 만든 뒤, 그 step 부터 native
#      downgrade 가 일어나고 최종 string 이 여전히 byte-equal 인지 확인.
#      reconstruct_count == 1, fallback_count == 1.
#   4) EXCLUSIVE=1 + sanity fail: ``incremental_append`` 가 str 가 아닌
#      bytes 를 돌려주는 패치 → 동일하게 downgrade + byte-equal.
#
# Run
# ---
#   /workspace/vllm_dev_prj/bin/python tests/test_avx512_detok_native_exclusive.py
#
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROMPTS = [
    "Hello, world!",
    "안녕하세요. 오늘 날씨가 정말 좋네요.",
    "Mixing English と日本語 and 中文 and 한국어 in one line.",
    "Look at these emojis: 🚀 🌟 🎉",
    "def foo(x):\n    return x + 1\n",
]

MODEL_NAME = os.environ.get("B1_EXCLUSIVE_TOK", "sshleifer/tiny-gpt2")


def _build(env: dict[str, str]):
    """Reload detokenizer module under a clean env so the gating flags are
    re-evaluated at module import (they are module-level globals)."""
    for k, v in env.items():
        os.environ[k] = v
    if "vllm.v1.engine.detokenizer" in sys.modules:
        del sys.modules["vllm.v1.engine.detokenizer"]
    return importlib.import_module("vllm.v1.engine.detokenizer")


def _make_fast(det_module, tok):
    from types import SimpleNamespace
    sp = SimpleNamespace(
        stop=None,
        min_tokens=0,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )
    req = SimpleNamespace(
        request_id="exclusive-smoke",
        prompt_token_ids=[],
        sampling_params=sp,
    )
    return det_module.FastIncrementalDetokenizer(tok, req)


def _drive(fast, ids):
    parts = []
    for tid in ids:
        t = fast._protected_step(int(tid))
        if t is not None:
            parts.append(t)
    return "".join(parts)


def _ground_truth(tok, ids):
    bt = getattr(tok, "backend_tokenizer", None)
    if bt is not None:
        try:
            return bt.decode(ids, skip_special_tokens=False)
        except Exception:
            pass
    return tok.decode(ids)


def _load_tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:
        print(f"[skip] tokenizer '{MODEL_NAME}' unavailable: {exc}")
        return None


def main() -> int:
    tok = _load_tok()
    if tok is None:
        print("SKIP — tokenizer missing; cannot exercise exclusive wire-in.")
        return 0

    # ----- (1) default-off — EXCLUSIVE counters must be zero -----
    mod = _build({
        "VLLM_USE_AVX512_DETOK_INC": "0",
        "VLLM_USE_AVX512_DETOK_NATIVE": "0",
        "VLLM_USE_AVX512_DETOK_EXCLUSIVE": "0",
        "VLLM_AVX512_DETOK_VERIFY": "0",
    })
    ids = tok.encode(PROMPTS[0], add_special_tokens=False)
    fast = _make_fast(mod, tok)
    text = _drive(fast, ids)
    expected = _ground_truth(tok, ids)
    if text != expected:
        print(f"FAIL [1] default-off mismatch: got={text!r} exp={expected!r}")
        return 1
    snap = mod.avx512_detok_exclusive_snapshot()
    if snap["enabled"] or snap["step_count"] != 0:
        print(f"FAIL [1] default-off exclusive snap should be zero: {snap}")
        return 1
    print(f"[1/4] default-off: byte-equal OK; snap={snap}")

    # ----- (2) EXCLUSIVE on + normal path -----
    mod = _build({
        "VLLM_USE_AVX512_DETOK_INC": "0",
        "VLLM_USE_AVX512_DETOK_NATIVE": "0",
        "VLLM_USE_AVX512_DETOK_EXCLUSIVE": "1",
        "VLLM_AVX512_DETOK_VERIFY": "0",
    })
    total_tokens = 0
    pass_count = 0
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=False)
        total_tokens += len(ids)
        fast = _make_fast(mod, tok)
        text = _drive(fast, ids)
        exp = _ground_truth(tok, ids)
        if text == exp:
            pass_count += 1
        else:
            print(f"FAIL [2] EXCLUSIVE mismatch on prompt: {p!r}")
            print(f"  got={text!r}")
            print(f"  exp={exp!r}")
            return 1
    snap = mod.avx512_detok_exclusive_snapshot()
    if not snap["enabled"]:
        print(f"FAIL [2] enabled flag false: {snap}")
        return 1
    if snap["step_count"] != total_tokens:
        print(f"FAIL [2] step_count={snap['step_count']} != "
              f"total_tokens={total_tokens}")
        return 1
    if snap["fallback_count"] != 0 or snap["reconstruct_count"] != 0:
        print(f"FAIL [2] unexpected fallback in normal path: {snap}")
        return 1
    print(f"[2/4] EXCLUSIVE normal: {pass_count}/{len(PROMPTS)} prompts "
          f"byte-equal; snap={snap}")

    # ----- (3) EXCLUSIVE on + forced exception at step 2 -----
    mod = _build({
        "VLLM_USE_AVX512_DETOK_INC": "0",
        "VLLM_USE_AVX512_DETOK_NATIVE": "0",
        "VLLM_USE_AVX512_DETOK_EXCLUSIVE": "1",
        "VLLM_AVX512_DETOK_VERIFY": "0",
    })
    ids = tok.encode(PROMPTS[2], add_special_tokens=False)
    fast = _make_fast(mod, tok)
    # Patch the wrapper to raise on a chosen step.
    wrapper = fast._avx512_detok_inc
    assert wrapper is not None, "wrapper must attach when EXCLUSIVE on"
    orig = wrapper.incremental_append
    raise_at = min(2, max(1, len(ids) // 2))
    call_ix = {"n": 0}

    def patched(tid):
        call_ix["n"] += 1
        if call_ix["n"] == raise_at:
            raise RuntimeError("forced: AVX-only path simulated failure")
        return orig(tid)

    wrapper.incremental_append = patched  # type: ignore[attr-defined]
    text = _drive(fast, ids)
    exp = _ground_truth(tok, ids)
    if text != exp:
        print(f"FAIL [3] forced-exception byte-equal failed:")
        print(f"  got={text!r}")
        print(f"  exp={exp!r}")
        return 1
    if not fast._native_downgrade:
        print(f"FAIL [3] expected _native_downgrade=True")
        return 1
    snap = mod.avx512_detok_exclusive_snapshot()
    if snap["fallback_count"] < 1 or snap["reconstruct_count"] < 1:
        print(f"FAIL [3] fallback/reconstruct counters not bumped: {snap}")
        return 1
    print(f"[3/4] EXCLUSIVE forced-exception @step={raise_at}/"
          f"{len(ids)}: byte-equal OK; downgrade=True; snap={snap}")

    # ----- (4) EXCLUSIVE on + sanity fail (non-str return) -----
    mod = _build({
        "VLLM_USE_AVX512_DETOK_INC": "0",
        "VLLM_USE_AVX512_DETOK_NATIVE": "0",
        "VLLM_USE_AVX512_DETOK_EXCLUSIVE": "1",
        "VLLM_AVX512_DETOK_VERIFY": "0",
    })
    ids = tok.encode(PROMPTS[3], add_special_tokens=False)
    fast = _make_fast(mod, tok)
    wrapper = fast._avx512_detok_inc
    orig = wrapper.incremental_append
    fail_at = min(1, len(ids) - 1)
    call_ix = {"n": 0}

    def patched_sanity(tid):
        call_ix["n"] += 1
        if call_ix["n"] == fail_at + 1:
            return b"bytes-not-str-sanity-fail"  # wrong type
        return orig(tid)

    wrapper.incremental_append = patched_sanity  # type: ignore[attr-defined]
    text = _drive(fast, ids)
    exp = _ground_truth(tok, ids)
    if text != exp:
        print(f"FAIL [4] sanity-fail byte-equal failed:")
        print(f"  got={text!r}")
        print(f"  exp={exp!r}")
        return 1
    if not fast._native_downgrade:
        print(f"FAIL [4] expected _native_downgrade=True (sanity)")
        return 1
    print(f"[4/4] EXCLUSIVE sanity-fail @step={fail_at + 1}: "
          f"byte-equal OK; downgrade=True")

    print("ALL PASS — NATIVE_EXCLUSIVE smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
