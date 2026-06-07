"""Run measure_client.py against N vllm instances concurrently.

For Step 1 (cluster capacity hypothesis B). All N runs start at roughly the
same time so they share host noise and CPU/Pi pressure faithfully.

Usage:
    python measure_concurrent.py \
        --endpoints 8011 8012 ... 8018 \
        --n 100 --conc 8 --max-tokens 256 \
        --out-dir logs/step1/run1
"""
from __future__ import annotations
import argparse, asyncio, json, time, random
from pathlib import Path
import httpx


def build_prompts(n: int, seed: int = 0) -> list[str]:
    sonnet_path = Path("/workspace/host_vllm_hybrid/benchmarks/sonnet.txt")
    if sonnet_path.exists():
        lines = [ln.strip() for ln in sonnet_path.read_text().splitlines() if ln.strip()]
    else:
        lines = [
            "The sun rose slowly above the silent hills.",
            "Each step forward whispered secrets to the wind.",
            "Old machines hummed in the corners of the warehouse.",
            "Beneath the pavement, a forgotten river still ran.",
        ]
    prompts: list[str] = []
    for i in range(n):
        rng = random.Random(seed * 1000 + i)
        head = (
            f"[doc {i:03d}] Continue the following passage in vivid prose for "
            f"several long paragraphs. Stay coherent and elaborate fully.\n\n"
        )
        body = "\n".join(rng.choice(lines) for _ in range(4))
        prompts.append(head + body)
    return prompts


async def _req(client, url, model, prompt, max_tokens, sem):
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"{url}/completions",
                json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                      "temperature": 0.0, "top_p": 1.0, "stream": False},
                timeout=httpx.Timeout(1800.0, connect=10.0),
            )
            wall = (time.perf_counter() - t0) * 1000.0
            if r.status_code == 200:
                u = r.json().get("usage", {})
                return {"ok": True, "wall_ms": wall,
                        "ctok": u.get("completion_tokens", 0),
                        "ptok": u.get("prompt_tokens", 0)}
            return {"ok": False, "wall_ms": wall, "ctok": 0, "ptok": 0,
                    "err": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"ok": False, "wall_ms": (time.perf_counter()-t0)*1000.0,
                    "ctok": 0, "ptok": 0, "err": repr(e)}


async def run_one_endpoint(port: int, prompts: list[str], max_tokens: int, conc: int):
    url = f"http://127.0.0.1:{port}/v1"
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=conc*2, max_keepalive_connections=conc*2)
    ) as cli:
        # Warmup once.
        warm = await _req(cli, url, "moe-offload-test", prompts[0][:200], 16, sem)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(
            _req(cli, url, "moe-offload-test", p, max_tokens, sem)
            for p in prompts
        ))
        wall = time.perf_counter() - t0
    ok = [r for r in results if r["ok"]]
    err = [r for r in results if not r["ok"]]
    walls = sorted(r["wall_ms"] for r in ok)
    p50 = walls[len(walls)//2] if walls else 0.0
    p99 = walls[min(len(walls)-1, int(len(walls)*0.99))] if walls else 0.0
    total_out = sum(r["ctok"] for r in ok)
    total_in = sum(r["ptok"] for r in ok)
    return {
        "port": port, "url": url,
        "wall_total_s": wall,
        "n_ok": len(ok), "n_err": len(err),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "decode_tps": total_out / wall if wall > 0 else 0.0,
        "req_per_s": len(ok) / wall if wall > 0 else 0.0,
        "p50_ms": p50, "p99_ms": p99,
        "warm_wall_ms": warm.get("wall_ms"),
        "first_errs": [r.get("err") for r in err[:3]],
        "raw_walls_ms": walls,
    }


async def run(ports: list[int], n: int, max_tokens: int, conc: int, seed: int, out_dir: Path):
    prompts = build_prompts(n, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # Launch all endpoints at the same time so they truly share host noise.
    tasks = [asyncio.create_task(run_one_endpoint(p, prompts, max_tokens, conc)) for p in ports]
    per_ep = await asyncio.gather(*tasks)
    wall_total = time.perf_counter() - t0

    total_out = sum(e["total_output_tokens"] for e in per_ep)
    total_in = sum(e["total_input_tokens"] for e in per_ep)
    cluster_tps = total_out / wall_total if wall_total > 0 else 0.0
    summary = {
        "n_endpoints": len(ports),
        "ports": ports,
        "n_per_endpoint": n,
        "conc_per_endpoint": conc,
        "max_tokens": max_tokens,
        "wall_total_s": wall_total,
        "cluster_total_output_tokens": total_out,
        "cluster_total_input_tokens": total_in,
        "cluster_decode_tps": cluster_tps,
        "tps_per_gpu_avg": cluster_tps / len(ports) if ports else 0.0,
        "tps_per_endpoint": {e["port"]: e["decode_tps"] for e in per_ep},
        "tps_per_endpoint_summary": {
            "min": min((e["decode_tps"] for e in per_ep), default=0.0),
            "max": max((e["decode_tps"] for e in per_ep), default=0.0),
            "mean": sum(e["decode_tps"] for e in per_ep) / max(1, len(per_ep)),
        },
        "n_ok_total": sum(e["n_ok"] for e in per_ep),
        "n_err_total": sum(e["n_err"] for e in per_ep),
    }
    out = {"summary": summary, "per_endpoint": per_ep}
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoints", nargs="+", type=int, required=True,
                   help="Ports of running vllm servers, e.g. 8011 8012 ...")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--conc", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    asyncio.run(run(a.endpoints, a.n, a.max_tokens, a.conc, a.seed, Path(a.out_dir)))


if __name__ == "__main__":
    main()
