#!/usr/bin/env python3
"""IDE_076 comparison analyzer (PLAN.md §9.2, §10.6, 부록 A.5). 입력: eval/results/IDE_076_20260917/qwen/OPT4/<boot>/<session>/metrics.json + <boot>/variant_manifest.json
출력: <FEAT>/e2e_comparison.csv (세션 원값·유효성), <FEAT>/e2e_stats.json (variant×workload 통계: n, boots, mean, median, sd(ddof=1), min/max, pooled tps, p95 평균과 pooled p95 구분), 화면 요약.
이득: 총 개선 = throughput(F)/throughput(R0)−1, 한계 이득 = 짝 비교 (같은 블록 순서). 퍼센트 합산 없음. 실패/0-request 는 유효 tok/s 로 평균하지 않음."""
import json, glob, os, csv, statistics as st, sys
HOME = os.path.expanduser("~"); REPO = f"{HOME}/projects/vllm_hybrid"; CAMP = f"{REPO}/eval/results/IDE_076_20260917"; FEAT = f"{REPO}/shadow_assists/features/IDE_076"


EV = [json.loads(l) for l in open(f"{CAMP}/state/execution_events.jsonl")] if os.path.exists(f"{CAMP}/state/execution_events.jsonl") else []


def main():
    rows = []
    for bd in sorted(glob.glob(f"{CAMP}/qwen/OPT4/*/")):
        vm = json.load(open(f"{bd}variant_manifest.json")) if os.path.exists(f"{bd}variant_manifest.json") else {"variant_id": "R0_REF" if "R0_REF" in bd else "?", "binary_sha256": None}
        rq = json.load(open(f"{bd}requested_config.json")) if os.path.exists(f"{bd}requested_config.json") else {}
        for sd in sorted(glob.glob(f"{bd}*/")):
            if not os.path.exists(f"{sd}metrics.json") or sd.rstrip("/").endswith("warmup"): continue
            m = json.load(open(f"{sd}metrics.json")); s = m.get("summary") or {}
            valid = bool(s) and m.get("rc") == 0 and s.get("failed") == 0 and s.get("completed") == m.get("n")
            died = os.path.exists(f"{bd}server.full.log.gz") and any(e.get("event") == "server_died" and e.get("boot_id") == m["boot_id"] for e in EV)
            if died: valid = False   # 정상성 미확인 부팅(서버 사망) 의 세션은 성능 판정에서 제외 (PLAN.md §12.1)
            # 모집단 (PLAN §14): callback-free 전환 수정(experts_base.py CF-fix, 2026-09-18 01:02 적용) 전/후 분리
            pop = {"rearm": "post_fix", "sync_only": "sync_only"}.get(vm.get("cf_fix"), "sync_only" if str(vm.get("variant_id", "")).startswith("CFFIX") else "pre_fix")
            import re as _re
            _mb = _re.search(r"\((?:post-fix )?([pf]\d+)\)", str(vm.get("tag", ""))); block = _mb.group(1) if _mb else ""
            rows.append({"campaign_id": "IDE_076_20260917", "population": pop, "block": block, "variant_id": vm.get("variant_id"), "boot_id": m["boot_id"], "session_id": m["session"], "order_index": len([r for r in rows if r["boot_id"] == m["boot_id"]]),
                         "binary_sha256": (rq.get("kt_kernel_so_sha256") or "")[:16], "a1": (rq.get("env") or {}).get("KT_OPT_A1_ENABLE", "unset"), "a2": (rq.get("env") or {}).get("KT_OPT_A2_ENABLE", "unset"), "hotmap": (rq.get("server_args") or {}).get("init-expert-location"), "budget": (rq.get("env") or {}).get("KT_GPU_EXPERTS_PER_LAYER"),
                         "client": m.get("client"), "workload_id": m["workload"], "mode": m["mode"], "sent": m["n"], "completed": s.get("completed"), "failed": s.get("failed"), "input_tokens": s.get("total_input_tokens"), "output_tokens": s.get("total_output_tokens"), "duration_s": s.get("duration"),
                         "output_tps": s.get("output_throughput"), "ttft_p50_ms": s.get("median_ttft_ms"), "ttft_p95_ms": s.get("p95_ttft_ms"), "ttft_p99_ms": s.get("p99_ttft_ms"), "tpot_p50_ms": s.get("median_tpot_ms"), "tpot_p95_ms": s.get("p95_tpot_ms"), "tpot_p99_ms": s.get("p99_tpot_ms"), "itl_p95_ms": s.get("p95_itl_ms"), "e2el_p95_ms": s.get("p95_e2el_ms"),
                         "execution_valid": valid, "invalid_reason": ("server_died_in_boot" if died else ("" if valid else "rc/failed/completed")), "t_start": m["t_start"]["wall_kst"], "raw_metrics_path": os.path.relpath(sd, REPO)})
    os.makedirs(FEAT, exist_ok=True)
    with open(f"{FEAT}/e2e_comparison.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    stats = {}
    for (pop, v, wl, mode) in sorted(set((r["population"], r["variant_id"], r["workload_id"], r["mode"]) for r in rows)):
        rr = [r for r in rows if r["population"] == pop and r["variant_id"] == v and r["workload_id"] == wl and r["mode"] == mode and r["execution_valid"]]
        if not rr: continue
        tps = [r["output_tps"] for r in rr]; boots = sorted(set(r["boot_id"] for r in rr))
        stats[f"{pop}|{v}|{wl}|{mode}"] = {"population": pop, "variant": v, "workload": wl, "mode": mode, "n_sessions": len(rr), "n_boots": len(boots), "boots": boots, "tps_mean": st.mean(tps), "tps_median": st.median(tps), "tps_sd_ddof1": st.stdev(tps) if len(tps) > 1 else None, "tps_min": min(tps), "tps_max": max(tps),
                                      "tps_pooled": sum(r["output_tokens"] for r in rr) / sum(r["duration_s"] for r in rr), "ttft_p95_mean": st.mean(r["ttft_p95_ms"] for r in rr), "tpot_p95_mean": st.mean(r["tpot_p95_ms"] for r in rr), "tpot_p50_mean": st.mean(r["tpot_p50_ms"] for r in rr), "ttft_p50_mean": st.mean(r["ttft_p50_ms"] for r in rr), "per_boot_mean": {b: st.mean(r["output_tps"] for r in rr if r["boot_id"] == b) for b in boots}, "invalid_sessions": len([r for r in rows if r["population"] == pop and r["variant_id"] == v and r["workload_id"] == wl and r["mode"] == mode and not r["execution_valid"]])}
    # 이득 (같은 모집단의 R0 대비, MAIN OFF): 총 개선 + 부팅 블록 짝 비교 (부팅 순서대로 짝)
    gains = {}
    for k, v in stats.items():
        base = stats.get(f"{v['population']}|R0|M073_QWEN_MAIN_SHORT|OFF")
        if v["workload"] != "M073_QWEN_MAIN_SHORT" or v["mode"] != "OFF" or v["variant"] == "R0" or not base: continue
        # 짝 = 같은 블록의 R0 (블록 태그가 있으면); 없으면 부팅 순서 짝
        blk = {r["boot_id"]: r["block"] for r in rows}
        base_by_blk = {blk.get(b, ""): base["per_boot_mean"][b] for b in base["boots"]}
        if any(blk.get(b) for b in v["boots"]):
            pairs = [(v["per_boot_mean"][b], base_by_blk[blk[b]]) for b in v["boots"] if blk.get(b) in base_by_blk]
        else:
            bb = [base["per_boot_mean"][b] for b in base["boots"]]; cb = [v["per_boot_mean"][b] for b in v["boots"]]; pairs = list(zip(cb, bb))
        gains[f"{v['population']}|{v['variant']}"] = {"population": v["population"], "pair_blocks": [blk.get(b, "") for b in v["boots"] if blk.get(b) in base_by_blk] if any(blk.get(b) for b in v["boots"]) else None,"total_gain_vs_R0_mean": v["tps_mean"] / base["tps_mean"] - 1, "paired_block_gains": [c / b_ - 1 for c, b_ in pairs], "all_pairs_positive": all(c > b_ for c, b_ in pairs) if pairs else None, "n_pairs": len(pairs), "ttft_p95_ratio": v["ttft_p95_mean"] / base["ttft_p95_mean"], "tpot_p95_ratio": v["tpot_p95_mean"] / base["tpot_p95_mean"]}
    json.dump({"stats": stats, "gains_vs_R0": gains}, open(f"{FEAT}/e2e_stats.json", "w"), indent=1, ensure_ascii=False)
    for k, v in stats.items(): print(k.ljust(44), "n", v["n_sessions"], "boots", v["n_boots"], "mean", round(v["tps_mean"], 2), "sd", (round(v["tps_sd_ddof1"], 2) if v["tps_sd_ddof1"] else None), "min/max", round(v["tps_min"], 1), round(v["tps_max"], 1), "ttft95", round(v["ttft_p95_mean"]), "tpot95", round(v["tpot_p95_mean"], 1))
    for k, g in gains.items(): print("GAIN", k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in g.items()}))


if __name__ == "__main__": main()
