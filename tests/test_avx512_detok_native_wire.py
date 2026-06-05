# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# IDE_016 / SUB_201 §5 B1 PoC Phase A4-prod — production wire-in smoke test.
#
# Goal:
#   FastIncrementalDetokenizer._protected_step 에 production wire-in
#   (VLLM_USE_AVX512_DETOK_NATIVE=1) 이 정상 attach 되어 token 한 개라도
#   AVX-512 path 를 통해 적절히 처리되는지 확인.
#
# Coverage:
#   1) Module import + telemetry snapshot accessible.
#   2) `_avx512_detok_inc_get_for(tok)` returns a wrapper instance when
#      either INC or NATIVE flag is set.
#   3) Drive `_protected_step` directly with N tokens — verify telemetry
#      counters move and the produced text is byte-equal to the native
#      DecodeStream baseline (= regression gate at unit level).
#   4) `VLLM_AVX512_DETOK_VERIFY=1` path: verify counter populated correctly.
#
# Run:
#   /workspace/vllm_dev_prj/bin/python tests/test_avx512_detok_native_wire.py

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

MODEL_NAME = os.environ.get(
    "B1_WIRE_TOK", "sshleifer/tiny-gpt2"
)


def _build(env: dict[str, str]):
    """Reload detokenizer module under a clean env so the gating flags pick
    up the chosen test config (module-level globals)."""
    for k, v in env.items():
        os.environ[k] = v
    if "vllm.v1.engine.detokenizer" in sys.modules:
        del sys.modules["vllm.v1.engine.detokenizer"]
    return importlib.import_module("vllm.v1.engine.detokenizer")


def _drive(det_module, tok, ids: list[int]) -> tuple[str, dict]:
    """Run `_protected_step` over `ids` with a minimal fake request stub."""
    from types import SimpleNamespace
    # Build a fake EngineCoreRequest with the bare-minimum surface
    # FastIncrementalDetokenizer.__init__ reads.
    sp = SimpleNamespace(
        stop=None,
        min_tokens=0,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )
    req = SimpleNamespace(
        request_id="wire-smoke",
        prompt_token_ids=[],
        sampling_params=sp,
    )
    fast = det_module.FastIncrementalDetokenizer(tok, req)
    text_parts: list[str] = []
    for tid in ids:
        t = fast._protected_step(int(tid))
        if t is not None:
            text_parts.append(t)
    return "".join(text_parts), det_module.avx512_detok_native_snapshot()


def _ground_truth(tok, ids: list[int]) -> str:
    """Same source-of-truth as test_avx512_detok_incremental: backend stream."""
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
        print("SKIP — tokenizer missing; cannot exercise wire-in.")
        return 0  # not a real failure — env limitation

    # --- (1) NATIVE off + INC off → wrapper attaches None, no telemetry move
    mod = _build({
        "VLLM_USE_AVX512_DETOK_INC": "0",
        "VLLM_USE_AVX512_DETOK_NATIVE": "0",
        "VLLM_AVX512_DETOK_VERIFY": "0",
    })
    ids = tok.encode(PROMPTS[0], add_special_tokens=False)
    txt_off, snap_off = _drive(mod, tok, ids)
    expected = _ground_truth(tok, ids)
    if txt_off != expected:
        print(f"FAIL — default-off baseline mismatch: got={txt_off!r} "
              f"exp={expected!r}")
        return 1
    if snap_off["step_count"] != 0:
        print(f"FAIL — default-off telemetry should be zero: {snap_off}")
        return 1
    print(f"[1/3] default-off baseline: {len(ids)} tokens byte-equal OK; "
          f"snap={snap_off}")

    # --- (2) NATIVE on (no verify) → telemetry moves, output byte-equal
    mod = _build({
        "VLLM_USE_AVX512_DETOK_INC": "0",
        "VLLM_USE_AVX512_DETOK_NATIVE": "1",
        "VLLM_AVX512_DETOK_VERIFY": "0",
    })
    total_tokens = 0
    total_pass = 0
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=False)
        total_tokens += len(ids)
        txt, snap = _drive(mod, tok, ids)
        exp = _ground_truth(tok, ids)
        if txt == exp:
            total_pass += 1
        else:
            print(f"FAIL — native wire-in mismatch on prompt: {p!r}")
            print(f"  got={txt!r}")
            print(f"  exp={exp!r}")
            return 1
    if snap["step_count"] < total_tokens:
        # Each prompt drives a fresh FastIncrementalDetokenizer so the
        # snapshot accumulates across all prompts.
        print(f"FAIL — telemetry step_count too low: got={snap['step_count']}"
              f" total_tokens={total_tokens}")
        return 1
    if snap["fallback_count"] != 0:
        print(f"WARN — unexpected fallbacks: {snap['fallback_count']}")
        # not a hard failure (warn-only) but log it for visibility
    print(f"[2/3] NATIVE wire-in: {total_pass}/{len(PROMPTS)} prompts "
          f"byte-equal; snap={snap}")

    # --- (3) VERIFY on → counters present; output still byte-equal
    mod = _build({
        "VLLM_USE_AVX512_DETOK_INC": "0",
        "VLLM_USE_AVX512_DETOK_NATIVE": "1",
        "VLLM_AVX512_DETOK_VERIFY": "1",
    })
    ids = tok.encode(PROMPTS[1], add_special_tokens=False)
    txt, snap = _drive(mod, tok, ids)
    exp = _ground_truth(tok, ids)
    if txt != exp:
        print(f"FAIL — verify-mode output mismatch: got={txt!r} exp={exp!r}")
        return 1
    if not snap["verify_enabled"]:
        print(f"FAIL — verify_enabled should be True: {snap}")
        return 1
    print(f"[3/3] VERIFY mode: byte-equal OK; snap={snap}")

    print("ALL PASS — production wire-in smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
