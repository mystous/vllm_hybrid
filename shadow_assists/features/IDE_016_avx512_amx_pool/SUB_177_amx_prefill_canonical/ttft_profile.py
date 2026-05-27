"""SUB_177 — TTFT profile (medium-context 512 / 1024 / 2048 prefill).

vllm GPU prefill latency 측정 — input_tokens 별 TTFT 분포.
직접 OpenAI completion API 를 비동기로 호출하여 첫 토큰 도착 시간 측정.

목적:
  - GPU prefill 의 latency 가 (a) compute-bound 인지 (b) memory-bound 인지 가설 검증
  - AMX CPU 보조의 latency budget upper bound 확인 (TTFT 의 일부 만 보조 가능)
  - paper §4 TSK_027 의 −15% target 의 feasibility input
"""
from __future__ import annotations

import argparse, asyncio, json, os, statistics, sys, time

import aiohttp


def gen_prompt(n_tokens: int) -> str:
    """길이가 n_tokens 정도인 prompt. 단어 평균 1.3 토큰 가정."""
    base = "The quick brown fox jumps over the lazy dog and then runs " \
           "into the forest where many other animals live including " \
           "deer rabbits squirrels and birds singing in the trees. "
    word_count = int(n_tokens / 1.3)
    return (base * (word_count // 30 + 2))[:word_count * 7]


async def one_request(session, url, model, prompt, max_tokens):
    """첫 토큰 도착 시간 측정 — stream=True 로 chunk timing 수집."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_send = time.perf_counter()
    ttft = None
    n_chunks = 0
    n_tokens_returned = 0
    prompt_tokens = None
    async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as resp:
        async for chunk in resp.content.iter_any():
            if ttft is None:
                ttft = time.perf_counter() - t_send
            n_chunks += 1
            # parse SSE for token count
            text = chunk.decode("utf-8", errors="ignore")
            for line in text.split("\n"):
                if line.startswith("data: "):
                    body = line[6:].strip()
                    if body == "[DONE]":
                        continue
                    try:
                        obj = json.loads(body)
                        if obj.get("choices") and obj["choices"][0].get("text"):
                            n_tokens_returned += 1
                        if obj.get("usage"):
                            prompt_tokens = obj["usage"].get("prompt_tokens")
                    except Exception:
                        pass
    t_end = time.perf_counter() - t_send
    return {"ttft": ttft, "total": t_end, "n_chunks": n_chunks,
            "n_tokens": n_tokens_returned, "prompt_tokens": prompt_tokens}


async def sweep_one(url, model, n_input_tokens, n_requests, max_tokens, concurrency):
    """concurrency 만큼 동시 send → n_requests 회 누적."""
    prompt = gen_prompt(n_input_tokens)
    print(f"  prompt length (chars): {len(prompt)}")

    sem = asyncio.Semaphore(concurrency)
    results = []

    async with aiohttp.ClientSession() as session:
        async def bound_req(i):
            async with sem:
                try:
                    r = await one_request(session, url, model, prompt, max_tokens)
                    return r
                except Exception as e:
                    return {"err": str(e)}

        tasks = [bound_req(i) for i in range(n_requests)]
        for f in asyncio.as_completed(tasks):
            r = await f
            results.append(r)

    return results


def summarize(results, label):
    ttfts = [r["ttft"] for r in results if r.get("ttft") is not None]
    if not ttfts:
        return {"label": label, "n": 0}
    ttfts.sort()
    n = len(ttfts)
    return {
        "label": label,
        "n": n,
        "ttft_min_ms": ttfts[0] * 1000,
        "ttft_p50_ms": ttfts[n // 2] * 1000,
        "ttft_p90_ms": ttfts[int(n * 0.9)] * 1000,
        "ttft_p99_ms": ttfts[int(n * 0.99)] * 1000,
        "ttft_max_ms": ttfts[-1] * 1000,
        "ttft_mean_ms": (sum(ttfts) / n) * 1000,
        "prompt_tokens_actual": [r.get("prompt_tokens") for r in results[:3]],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/completions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--n-requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"[ttft_profile] url={args.url} model={args.model}")
    print(f"  n_req={args.n_requests} max_tok={args.max_tokens} conc={args.concurrency}")

    all_summary = []
    for nt in args.input_tokens:
        print(f"\n=== input_tokens ≈ {nt} ===")
        results = await sweep_one(args.url, args.model, nt,
                                   args.n_requests, args.max_tokens, args.concurrency)
        s = summarize(results, f"input_{nt}")
        all_summary.append({"input_tokens_target": nt, "stats": s, "raw": results})
        print(f"  TTFT min/p50/p90/p99/max = "
              f"{s.get('ttft_min_ms', -1):.1f} / "
              f"{s.get('ttft_p50_ms', -1):.1f} / "
              f"{s.get('ttft_p90_ms', -1):.1f} / "
              f"{s.get('ttft_p99_ms', -1):.1f} / "
              f"{s.get('ttft_max_ms', -1):.1f} ms")
        print(f"  prompt_tokens actual (first 3): {s.get('prompt_tokens_actual')}")

    with open(args.out, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
