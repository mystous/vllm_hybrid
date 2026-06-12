"""정확도 게이트 (TST_003 운영 해석) — vanilla bf16 KV vs fp8 KV.

분포 유사성 게이트:
  - top-1 token agreement: 두 backend 가 동일 위치에서 동일 argmax 를 낸 비율
  - per-token logprob max abs diff: 각 위치 top-1 logprob 의 |Δ| 분포
  - sequence-level PPL relative diff: exp(-mean(top1_logprob)) 비교

50 prompts(짧은 sharegpt) × max-tok 64, greedy(temperature=0), logprobs=5.

운영 게이트:
  - top-1 agreement(50 prompt 평균) ≥ 95%
  - PPL relative diff < 5%

사용:
  python accuracy_gate.py --capture --backend vanilla --port 8001 --model meta-llama/Llama-3.1-8B-Instruct --out vanilla_caps.jsonl
  python accuracy_gate.py --capture --backend fp8     --port 8002 --model meta-llama/Llama-3.1-8B-Instruct --out fp8_caps.jsonl
  python accuracy_gate.py --analyze --a vanilla_caps.jsonl --b fp8_caps.jsonl --out gate.json

prompt source: sharegpt200.parquet 의 짧은(<800 char) prompt 중 앞 N 개.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq


def load_prompts(parquet_path: str, n: int, max_chars: int = 800):
    rows = pq.read_table(parquet_path).to_pylist()
    short = [r for r in rows if len(r.get("raw_text", "")) < max_chars]
    return short[:n]


async def capture_one(client, base_url, model, rec, max_tokens, logprobs):
    """결정론적(greedy) 호출, /v1/completions, logprobs=5 → 위치별 top1 token + logprob 시퀀스 캡처."""
    payload = {
        "model": model,
        "prompt": rec["raw_text"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "logprobs": logprobs,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = await client.post(f"{base_url}/v1/completions", json=payload,
                          timeout=httpx.Timeout(600.0, connect=10.0))
    wall = (time.perf_counter() - t0) * 1000.0
    if r.status_code != 200:
        return {"prompt_id": rec["prompt_id"], "ok": False, "err": f"HTTP {r.status_code}: {r.text[:200]}",
                "wall_ms": wall}
    data = r.json()
    ch = data["choices"][0]
    # OpenAI completions logprobs payload: {tokens, token_logprobs, top_logprobs}
    lp = ch.get("logprobs") or {}
    return {
        "prompt_id": rec["prompt_id"],
        "ok": True,
        "wall_ms": wall,
        "text": ch.get("text", ""),
        "tokens": lp.get("tokens", []),
        "token_logprobs": lp.get("token_logprobs", []),
        "top_logprobs": lp.get("top_logprobs", []),  # list of dict[str,float]
        "finish_reason": ch.get("finish_reason"),
    }


async def capture_all(args):
    rows = load_prompts(args.parquet, args.n, args.max_chars)
    base = f"http://{args.host}:{args.port}"
    print(f"[capture] backend={args.backend} model={args.model} n={len(rows)} → {args.out}")
    async with httpx.AsyncClient() as cl:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for i, rec in enumerate(rows):
                cap = await capture_one(cl, base, args.model, rec, args.max_tokens, args.logprobs)
                cap["backend"] = args.backend
                f.write(json.dumps(cap) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(rows)}] {rec['prompt_id']} ok={cap['ok']}")
    print(f"[capture] done → {args.out}")


def analyze(a_path: str, b_path: str, out_path: str):
    a_recs = {}
    with open(a_path) as f:
        for ln in f:
            r = json.loads(ln)
            a_recs[r["prompt_id"]] = r
    b_recs = {}
    with open(b_path) as f:
        for ln in f:
            r = json.loads(ln)
            b_recs[r["prompt_id"]] = r

    common = sorted(set(a_recs) & set(b_recs))
    per_prompt = []
    all_top1_match = []
    all_logprob_diffs = []
    a_logppl, b_logppl = [], []

    for pid in common:
        a, b = a_recs[pid], b_recs[pid]
        if not (a.get("ok") and b.get("ok")):
            continue
        a_tok = a.get("tokens", [])
        b_tok = b.get("tokens", [])
        a_lp = a.get("token_logprobs", []) or []
        b_lp = b.get("token_logprobs", []) or []
        L = min(len(a_tok), len(b_tok))
        if L == 0:
            continue
        match = sum(1 for i in range(L) if a_tok[i] == b_tok[i])
        # logprob max abs diff over positions where both have a value
        diffs = []
        for i in range(L):
            la = a_lp[i] if i < len(a_lp) else None
            lb = b_lp[i] if i < len(b_lp) else None
            if la is not None and lb is not None:
                diffs.append(abs(la - lb))
        per_prompt.append({
            "prompt_id": pid,
            "L_compared": L,
            "L_a": len(a_tok),
            "L_b": len(b_tok),
            "top1_match": match,
            "top1_agreement": round(match / L, 4),
            "logprob_max_abs_diff": round(max(diffs), 4) if diffs else None,
            "logprob_mean_abs_diff": round(sum(diffs) / len(diffs), 4) if diffs else None,
        })
        all_top1_match.append(match / L)
        if diffs:
            all_logprob_diffs.extend(diffs)
        # sequence-level PPL: exp(-mean(top1_logprob)) on each side over its own tokens
        valid_a = [x for x in a_lp if x is not None]
        valid_b = [x for x in b_lp if x is not None]
        if valid_a:
            a_logppl.append(-sum(valid_a) / len(valid_a))
        if valid_b:
            b_logppl.append(-sum(valid_b) / len(valid_b))

    n_eff = len(per_prompt)
    mean_top1 = round(sum(all_top1_match) / n_eff, 4) if n_eff else None
    mean_logppl_a = sum(a_logppl) / len(a_logppl) if a_logppl else None
    mean_logppl_b = sum(b_logppl) / len(b_logppl) if b_logppl else None
    ppl_a = math.exp(mean_logppl_a) if mean_logppl_a is not None else None
    ppl_b = math.exp(mean_logppl_b) if mean_logppl_b is not None else None
    ppl_rel_diff = abs(ppl_b - ppl_a) / ppl_a if ppl_a else None

    summary = {
        "n_prompts_compared": n_eff,
        "mean_top1_agreement": mean_top1,
        "min_top1_agreement": round(min(all_top1_match), 4) if all_top1_match else None,
        "logprob_max_abs_diff": round(max(all_logprob_diffs), 4) if all_logprob_diffs else None,
        "logprob_mean_abs_diff": round(sum(all_logprob_diffs) / len(all_logprob_diffs), 4) if all_logprob_diffs else None,
        "logprob_p99_abs_diff": round(sorted(all_logprob_diffs)[int(0.99 * (len(all_logprob_diffs) - 1))], 4) if all_logprob_diffs else None,
        "ppl_a": round(ppl_a, 4) if ppl_a else None,
        "ppl_b": round(ppl_b, 4) if ppl_b else None,
        "ppl_rel_diff": round(ppl_rel_diff, 4) if ppl_rel_diff is not None else None,
        "gate_top1_pass": (mean_top1 is not None and mean_top1 >= 0.95),
        "gate_ppl_pass": (ppl_rel_diff is not None and ppl_rel_diff < 0.05),
        "gate_overall_pass": (
            mean_top1 is not None and mean_top1 >= 0.95
            and ppl_rel_diff is not None and ppl_rel_diff < 0.05
        ),
        "per_prompt": per_prompt,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze] n={n_eff} top1_agree={mean_top1} ppl_rel_diff={summary['ppl_rel_diff']} "
          f"gate={summary['gate_overall_pass']} → {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture")
    cap.add_argument("--backend", required=True, choices=["vanilla", "fp8"])
    cap.add_argument("--model", required=True)
    cap.add_argument("--port", type=int, required=True)
    cap.add_argument("--host", default="127.0.0.1")
    cap.add_argument("--parquet", default="/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_8gpu_full/sharegpt200.parquet")
    cap.add_argument("--n", type=int, default=50)
    cap.add_argument("--max-chars", type=int, default=800)
    cap.add_argument("--max-tokens", type=int, default=64)
    cap.add_argument("--logprobs", type=int, default=5)
    cap.add_argument("--out", required=True)

    an = sub.add_parser("analyze")
    an.add_argument("--a", required=True, help="vanilla captures jsonl")
    an.add_argument("--b", required=True, help="fp8 captures jsonl")
    an.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "capture":
        asyncio.run(capture_all(args))
    elif args.cmd == "analyze":
        analyze(args.a, args.b, args.out)


if __name__ == "__main__":
    main()
