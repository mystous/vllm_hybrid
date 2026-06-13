#!/usr/bin/env python3
"""SUB_213 정확도 게이트 — IDE_006/TST_003 D-i/D-ii 방식.

collect 모드: 서버에 greedy + logprobs 요청 → JSONL 저장
compare 모드: A/B JSONL 비교 → D-i (token divergence, informational) /
              D-ii (logprob max-abs-diff ≤ atol, PPL rel diff ≤ rtol — binding)
"""
import argparse
import asyncio
import json
import math
import sys

import httpx
import pyarrow.parquet as pq

PARQUET = "vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet"


def load_prompts(n: int) -> list[str]:
    t = pq.read_table(PARQUET).to_pylist()
    # mix 와 동일한 shuffle (seed 0)
    import random

    random.Random(0).shuffle(t)
    out = []
    for row in t:
        p = row.get("raw_text") or ""
        if p and len(p) < 8000:
            out.append(p)
        if len(out) >= n:
            break
    return out


async def collect(port: int, model: str, n: int, max_tokens: int, out: str):
    prompts = load_prompts(n)
    results = []
    async with httpx.AsyncClient(timeout=600) as client:
        sem = asyncio.Semaphore(8)

        async def one(i, p):
            async with sem:
                r = await client.post(
                    f"http://127.0.0.1:{port}/v1/completions",
                    json={
                        "model": model,
                        "prompt": p,
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "logprobs": 1,
                        "seed": 0,
                    },
                )
                r.raise_for_status()
                d = r.json()["choices"][0]
                lp = d.get("logprobs") or {}
                return {
                    "idx": i,
                    "tokens": lp.get("tokens", []),
                    "token_logprobs": lp.get("token_logprobs", []),
                    "text": d.get("text", "")[:200],
                }

        results = await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    with open(out, "w") as f:
        for r in sorted(results, key=lambda x: x["idx"]):
            f.write(json.dumps(r) + "\n")
    print(f"[collect] {len(results)} prompts → {out}")


def compare(a_path: str, b_path: str, atol: float, rtol: float):
    A = [json.loads(l) for l in open(a_path)]
    B = [json.loads(l) for l in open(b_path)]
    assert len(A) == len(B)
    worst_lp, worst_ppl, n_pass, div_stats = 0.0, 0.0, 0, []
    for a, b in zip(A, B):
        ta, tb = a["tokens"], b["tokens"]
        la = [x for x in a["token_logprobs"] if x is not None]
        lb = [x for x in b["token_logprobs"] if x is not None]
        # D-i: 첫 발산 위치
        div = next(
            (i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
            min(len(ta), len(tb)),
        )
        n_div = max(len(ta), len(tb)) - div
        div_stats.append(n_div)
        # D-ii: 공통 prefix 의 logprob 비교 + 시퀀스 PPL
        m = min(div, len(la), len(lb))
        max_abs = max((abs(x - y) for x, y in zip(la[:m], lb[:m])), default=0.0)
        ppl_a = math.exp(-sum(la) / len(la)) if la else 1.0
        ppl_b = math.exp(-sum(lb) / len(lb)) if lb else 1.0
        ppl_rel = abs(ppl_a - ppl_b) / max(ppl_a, 1e-9)
        worst_lp = max(worst_lp, max_abs)
        worst_ppl = max(worst_ppl, ppl_rel)
        if max_abs <= atol and ppl_rel <= rtol:
            n_pass += 1
    n = len(A)
    identical = sum(1 for d in div_stats if d == 0)
    print(f"[D-i ] 완전 일치 {identical}/{n}, 발산 토큰수 분포: "
          f"max={max(div_stats)}, mean={sum(div_stats)/n:.1f} (informational)")
    print(f"[D-ii] worst_max_abs_logprob={worst_lp:.4f} (atol {atol}) / "
          f"worst_ppl_rel={worst_ppl:.4f} (rtol {rtol}) / pass {n_pass}/{n}")
    verdict = worst_lp <= atol and worst_ppl <= rtol
    print(f"[VERDICT] {'PASS' if verdict else 'FAIL'} (binding = D-ii)")
    return 0 if verdict else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--port", type=int, required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--n", type=int, default=32)
    c.add_argument("--max-tokens", type=int, default=128)
    c.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--atol", type=float, default=0.5)
    p.add_argument("--rtol", type=float, default=0.1)
    args = ap.parse_args()
    if args.cmd == "collect":
        asyncio.run(collect(args.port, args.model, args.n, args.max_tokens, args.out))
    else:
        sys.exit(compare(args.a, args.b, atol=args.atol, rtol=args.rtol))
