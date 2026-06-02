"""TSK_042 oracle runner — sampled prompt 를 단일 method 백엔드에 concurrency=1 로
실행하고 per-prompt output_tps 측정 → per_request_raw.jsonl append.

oracle = per-prompt best method 이므로 **격리 측정(concurrency=1)** 이 핵심
(batch 면 다른 prompt 간섭이 method 우열을 오염).

사용:
  python oracle_runner.py --in sampled_prompts.parquet --method suffix \
     --port 8002 --model Qwen/Qwen2.5-32B-Instruct --max-tokens 512 \
     --out per_request_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

import httpx
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--method", required=True)       # vanilla/suffix/ngram/eagle
    ap.add_argument("--model", required=True)         # 실제 모델명 (필수)
    ap.add_argument("--model-tag", default=None)      # oracle_table model 컬럼 (없으면 --model)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)   # 0 = 전체
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = pq.read_table(args.inp).to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    url = f"http://{args.host}:{args.port}/v1/completions"
    model_tag = args.model_tag or args.model
    n_ok = 0
    t0all = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(3600.0, connect=10.0)) as client, \
            open(args.out, "a") as fout:
        for r in rows:
            t0 = time.perf_counter()
            rec = {"prompt_id": r["prompt_id"], "corpus": r["corpus"],
                   "lang": r["lang"], "n_input_tok": r["n_input_tok"],
                   "model": model_tag, "method": args.method, "seed": args.seed}
            try:
                resp = client.post(url, json={
                    "model": args.model, "prompt": r["raw_text"],
                    "max_tokens": args.max_tokens, "temperature": 0.0,
                    "top_p": 1.0, "stream": False,
                })
                wall = time.perf_counter() - t0
                if resp.status_code == 200:
                    u = resp.json().get("usage", {})
                    ctok = u.get("completion_tokens", 0)
                    rec.update(ok=True, wall_ms=wall * 1000.0,
                               completion_tokens=ctok,
                               prompt_tokens=u.get("prompt_tokens", 0),
                               output_tps=(ctok / wall if wall > 0 else 0.0))
                    n_ok += 1
                else:
                    rec.update(ok=False, wall_ms=wall * 1000.0,
                               error=f"HTTP {resp.status_code}: {resp.text[:120]}")
            except Exception as e:  # noqa: BLE001
                rec.update(ok=False, wall_ms=(time.perf_counter() - t0) * 1000.0,
                           error=repr(e)[:120])
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
    dt = time.perf_counter() - t0all
    print(f"[oracle] {model_tag} × {args.method}: {n_ok}/{len(rows)} ok in {dt:.0f}s "
          f"→ {args.out}")


if __name__ == "__main__":
    main()
