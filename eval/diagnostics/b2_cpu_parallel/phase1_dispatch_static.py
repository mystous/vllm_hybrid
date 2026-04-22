#!/usr/bin/env python3
"""Phase 1 — 정적 코드 분석으로 CPU attention dispatch tree 를 프린트.

Heavy workload (batch=1, num_tokens=1, num_kv_heads=8, ctx_len=16384) 가
cpu_attn.py 에서 어느 decode path 를 타는지 소스 읽어서 확정.

사용:
    python eval/diagnostics/b2_cpu_parallel/phase1_dispatch_static.py

출력:
    - 각 `_trace_decode_path` 호출 site 의 gating 조건
    - IPEX vs TorchSDPA 분기점
    - Heavy input shape 추론 결과 (어느 path 로 감)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "vllm/v1/attention/backends/cpu_attn.py"

HEAVY_SHAPE = {
    "batch_size": 1,          # CPU engine 1 seq
    "num_tokens": 1,          # decode step
    "num_seqs": 1,            # same as batch
    "num_kv_heads": 8,        # Qwen2.5-32B GQA
    "num_query_heads": 40,    # 40 query heads
    "head_dim": 128,
    "ctx_len": 16384,         # long context
    "block_size": 16,         # typical paged attention
}

LIGHT_SHAPE = {**HEAVY_SHAPE, "ctx_len": 128}


def banner(title: str, ch: str = "=") -> None:
    print(f"\n{ch * 3} {title} {ch * max(3, 70 - len(title))}")


def read_text() -> list[str]:
    if not TARGET.exists():
        sys.exit(f"[ERROR] {TARGET} not found — run from repo root")
    return TARGET.read_text().splitlines()


def find_trace_sites(lines: list[str]) -> list[tuple[int, str, list[str]]]:
    """Return list of (line_no, path_label, preceding_condition_lines)."""
    sites = []
    pattern = re.compile(r'_trace_decode_path\(\s*"(\w+)"')
    for i, line in enumerate(lines):
        m = pattern.search(line)
        if not m:
            continue
        path = m.group(1)
        # Walk back up to 30 lines to capture surrounding `if` / `try`
        start = max(0, i - 30)
        context = lines[start:i + 1]
        sites.append((i + 1, path, context))
    return sites


def find_ipex_entry(lines: list[str]) -> list[tuple[int, str]]:
    """Find where _IPEXPagedAttention.forward_decode is called/dispatched."""
    result = []
    for i, line in enumerate(lines):
        if "_IPEXPagedAttention" in line or "use_ipex_paged_attention" in line or "ipex_paged_attention" in line.lower():
            result.append((i + 1, line.rstrip()))
    return result


def find_num_tokens_branches(lines: list[str]) -> list[tuple[int, str]]:
    """Find dispatch conditions mentioning num_tokens / num_seqs / context_lens."""
    result = []
    patterns = [
        r'if\s+num_tokens\s*[!=<>]=?\s*num_seqs',
        r'if\s+num_seqs\s*[!=<>]=?\s*1',
        r'if\s+num_tokens\s*[!=<>]=?\s*1',
        r'max_context_len\s*[!=<>]',
        r'if\s+context_lens',
    ]
    for i, line in enumerate(lines):
        for pat in patterns:
            if re.search(pat, line):
                result.append((i + 1, line.rstrip()))
                break
    return result


def emit_site(site: tuple[int, str, list[str]]) -> None:
    line_no, path, context = site
    # only show `if` / `try` / `else` / `return` lines from context for brevity
    print(f"\n  [{path}]  called at line {line_no}")
    print(f"  {'-' * 76}")
    for j, cline in enumerate(context):
        stripped = cline.strip()
        if not stripped:
            continue
        if re.match(r'^(if|elif|else|try|except|for|while|return)\b', stripped) or \
           '_trace_decode_path' in stripped:
            abs_no = line_no - len(context) + j + 1
            print(f"  L{abs_no:5d} | {cline.rstrip()}")


def infer_heavy_path(sites: list[tuple[int, str, list[str]]]) -> None:
    """Rule-based inference for heavy shape."""
    print("  추론 기준:")
    print(f"    num_tokens == num_seqs == 1  → sdpa_loop edge case 아님")
    print(f"    num_seqs == 1                → batch 일괄 처리 이점 없음")
    print(f"    ctx_len == 16384 (vs 128 light) → KV cache 크기 차이")
    print(f"    num_kv_heads == 8            → per-head parallel 상한 8")
    print()
    print("  주의: 이 스크립트는 조건을 자동으로 완전히 해석하지 못함.")
    print("        code 를 직접 읽어서 IPEX entry 조건 (is_ipex_available() 등)")
    print("        이 heavy/light 에서 어떻게 달라지는지 확인 필요.")


def main() -> int:
    banner("Phase 1 — CPU attention dispatch 정적 분석")
    print(f"Target file : {TARGET.relative_to(ROOT)}")
    print(f"Heavy shape : {HEAVY_SHAPE}")
    print(f"Light shape : {LIGHT_SHAPE}")

    lines = read_text()
    print(f"파일 줄 수  : {len(lines)}")

    banner("Section A — _trace_decode_path 호출 site (각 path 의 gating)")
    sites = find_trace_sites(lines)
    if not sites:
        print("  (없음) — dispatch tree 구조 변경되었을 수 있음")
    else:
        print(f"  총 {len(sites)} 개 path site 발견: {[p for _, p, _ in sites]}")
        for s in sites:
            emit_site(s)

    banner("Section B — IPEX entry point 언급")
    ipex_refs = find_ipex_entry(lines)
    if not ipex_refs:
        print("  IPEX 관련 라인 없음 — _IPEXPagedAttention 별도 파일일 수 있음")
    else:
        for no, line in ipex_refs[:30]:
            print(f"  L{no:5d} | {line}")

    banner("Section C — num_tokens / num_seqs / context_lens branching")
    branches = find_num_tokens_branches(lines)
    for no, line in branches[:30]:
        print(f"  L{no:5d} | {line}")

    banner("Section D — Heavy shape 추론")
    infer_heavy_path(sites)

    banner("Next", ch="—")
    print("  Phase 2: VLLM_HYBRID_TRACE=1 로 재실행해 counter 실측")
    print("    → eval/diagnostics/b2_cpu_parallel/g0_h100x8_qwen32b_longctx_trace.env")
    print("  Phase 3: stuck 프로세스 실시간 introspection")
    print("    → eval/diagnostics/b2_cpu_parallel/phase3_live_introspect.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
