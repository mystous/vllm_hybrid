#!/usr/bin/env python3
"""SUB_181 — aggregate 4-method e2e results across 3 mixes + compute speedup."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/vllm_hybrid/shadow_assists/features/IDE_019_multi_source_drafter/SUB_181_jacobi_router_e2e/measurements")
    ap.add_argument("--suffix", default="4method")
    args = ap.parse_args()

    root = Path(args.root)
    out = {"suffix": args.suffix, "mixes": {}, "summary": {}}
    tps_by_scenario = {"vanilla-only": [], "trident-only": [], "agsd-gated": []}
    jacobi_summary = {"total_calls": 0, "total_ms": 0.0, "calls_by_workload_chat": 0}

    for mix in ("balanced", "sonnet-heavy", "code-heavy"):
        bench_path = root / f"{args.suffix}_500p" / mix / f"benchmark_{mix}.json"
        stats_path = root / f"{args.suffix}_500p" / mix / "router_stats.json"
        if not bench_path.exists():
            print(f"[warn] missing {bench_path}")
            continue
        bench = json.loads(bench_path.read_text())
        out["mixes"][mix] = {"scenarios": {}, "router_stats": {}}
        for sc in bench.get("scenarios", []):
            name = sc.get("name") or sc.get("scenario")
            tps = sc.get("throughput_tps") or sc.get("tps")
            lat = sc.get("p50_latency_ms")
            ntok = sc.get("total_tokens")
            row = {"tps": tps, "p50_latency_ms": lat, "total_tokens": ntok}
            out["mixes"][mix]["scenarios"][name] = row
            if tps is not None and name in tps_by_scenario:
                tps_by_scenario[name].append(tps)
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            out["mixes"][mix]["router_stats"] = stats
            jacobi_summary["total_calls"] += stats.get("jacobi_calls", 0)
            jacobi_summary["total_ms"] += stats.get("jacobi_draft_time_total_ms", 0.0)
            wl_chat = stats.get("by_workload", {}).get("chat", 0)
            jacobi_summary["calls_by_workload_chat"] += wl_chat

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    out["summary"]["avg_tps"] = {k: avg(v) for k, v in tps_by_scenario.items()}
    out["summary"]["jacobi"] = jacobi_summary
    if jacobi_summary["total_calls"] > 0:
        out["summary"]["jacobi"]["avg_call_ms"] = (
            jacobi_summary["total_ms"] / jacobi_summary["total_calls"]
        )

    # speedup of agsd-gated vs vanilla-only
    if (out["summary"]["avg_tps"].get("agsd-gated")
            and out["summary"]["avg_tps"].get("vanilla-only")):
        out["summary"]["agsd_vs_vanilla_speedup"] = (
            out["summary"]["avg_tps"]["agsd-gated"]
            / out["summary"]["avg_tps"]["vanilla-only"]
        )
    if (out["summary"]["avg_tps"].get("agsd-gated")
            and out["summary"]["avg_tps"].get("trident-only")):
        out["summary"]["agsd_vs_trident_speedup"] = (
            out["summary"]["avg_tps"]["agsd-gated"]
            / out["summary"]["avg_tps"]["trident-only"]
        )

    out_path = root / f"{args.suffix}_aggregate.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[aggregate] wrote {out_path}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
