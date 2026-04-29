# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke for the NEO-into-vLLM wiring (IDE_006 4 차 재정의).

Runs two short generations with a small local model:
1. ``enable_neo_asymmetric=False`` — vanilla baseline.
2. ``enable_neo_asymmetric=True`` — NEO scheduler adapter active.

Verifies:
* Both runs complete without exceptions.
* Token outputs match (the current wiring routes the data path to
  vanilla even when the flag is on, so results must agree).
* NEO gate log lines appear in the subprocess stdout (grep'd from
  the redirected log file passed via ``--log-file``).

Run with:
    HF_HUB_OFFLINE=1 LD_PRELOAD=/usr/lib64/libcuda.so.1 \\
        python -u eval/run_neo_e2e_smoke.py 2>&1 | tee /tmp/neo_smoke.log
    python eval/run_neo_e2e_smoke.py --verify-only --log-file /tmp/neo_smoke.log

Or in one shot (the default ``--log-file`` matches the suggested redirect):
    bash eval/run_neo_e2e_smoke.sh
"""

from __future__ import annotations

# torch must load before vllm so that the dynamic loader caches
# torch's libtorch_*.so paths (vllm._C imports them).
import torch  # noqa: F401

import argparse
import sys
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS = [
    "Hello, my name is",
    "The quick brown fox",
    "Once upon a time",
]
MAX_TOKENS = 16
DEFAULT_LOG_FILE = "/tmp/neo_smoke.log"


def _run(enable_neo: bool):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        enable_neo_asymmetric=enable_neo,
        tensor_parallel_size=1,
        max_model_len=512,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        disable_log_stats=True,
    )
    params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, seed=0,
    )
    outputs = llm.generate(PROMPTS, params)
    return [out.outputs[0].token_ids for out in outputs], \
           [out.outputs[0].text for out in outputs]


def _verify(vanilla_tokens, vanilla_text, neo_tokens, neo_text,
            log_file: Path | None) -> int:
    """Return 0 on PASS, 1 on FAIL."""
    print("=" * 60)
    print("VERIFICATIONS")
    print("=" * 60)

    ok_tokens = vanilla_tokens == neo_tokens
    print(f"  token-id equality (vanilla vs NEO):  {'PASS' if ok_tokens else 'FAIL'}")
    if not ok_tokens:
        for i, (a, b) in enumerate(zip(vanilla_tokens, neo_tokens)):
            if a != b:
                print(f"    diff at prompt {i}: {a} vs {b}")

    # NEO gate logs come from the EngineCore subprocess and are written
    # to the captured log file. The current main process' logger never
    # sees them, so we grep the file passed in by the caller.
    log_text = log_file.read_text() if log_file and log_file.exists() else ""
    ok_adapter = "enable_neo_asymmetric activated" in log_text
    ok_gate = "execute_model: enable_neo_asymmetric=True observed" in log_text
    has_attach = "NEO scheduler attached" in log_text

    if log_file and not log_text:
        print(f"  log file empty or unreadable:        WARN ({log_file})")
        ok_adapter = ok_gate = None  # cannot verify

    def _label(state):
        if state is None:
            return "SKIP (no log file)"
        return "PASS" if state else "FAIL"

    print(f"  NeoSchedulerAdapter activation log:  {_label(ok_adapter)}")
    print(f"  execute_model NEO gate log:          {_label(ok_gate)}")
    print(f"  sub-batch attachment log:            "
          f"{'PASS (attached)' if has_attach else 'WARN (no attach yet)'}")

    binding = ok_tokens
    if log_file is not None:
        # If the caller asked us to inspect the log, those checks are
        # also binding.
        binding = binding and (ok_adapter is True) and (ok_gate is True)
    print(f"\nOVERALL: {'PASS' if binding else 'FAIL'}")
    return 0 if binding else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-file",
        type=Path,
        default=Path(DEFAULT_LOG_FILE),
        help="Path to the redirected stdout/stderr log file. "
             "Used by --verify-only or by the smoke wrapper.",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip the two LLM runs; only inspect the log file.",
    )
    ap.add_argument(
        "--no-log-verify",
        action="store_true",
        help="Skip the log-file verification (token equality only).",
    )
    args = ap.parse_args()

    if args.verify_only:
        # Re-run verification against an existing log file. This needs
        # the token outputs from a prior run — we cannot reconstruct
        # them, so we just inspect the log lines and report.
        log_text = args.log_file.read_text() if args.log_file.exists() else ""
        ok_adapter = "enable_neo_asymmetric activated" in log_text
        ok_gate = "execute_model: enable_neo_asymmetric=True observed" in log_text
        print(f"NeoSchedulerAdapter activation log:  "
              f"{'PASS' if ok_adapter else 'FAIL'}")
        print(f"execute_model NEO gate log:          "
              f"{'PASS' if ok_gate else 'FAIL'}")
        print(f"OVERALL (log-only): "
              f"{'PASS' if ok_adapter and ok_gate else 'FAIL'}")
        return 0 if ok_adapter and ok_gate else 1

    print("=" * 60)
    print("VANILLA RUN (enable_neo_asymmetric=False)")
    print("=" * 60)
    vanilla_tokens, vanilla_text = _run(False)
    for prompt, text in zip(PROMPTS, vanilla_text):
        print(f"  [vanilla] {prompt!r} → {text!r}")

    print("=" * 60)
    print("NEO RUN (enable_neo_asymmetric=True)")
    print("=" * 60)
    neo_tokens, neo_text = _run(True)
    for prompt, text in zip(PROMPTS, neo_text):
        print(f"  [neo]     {prompt!r} → {text!r}")

    log_file = None if args.no_log_verify else args.log_file
    return _verify(vanilla_tokens, vanilla_text, neo_tokens, neo_text,
                   log_file)


if __name__ == "__main__":
    sys.exit(main())
