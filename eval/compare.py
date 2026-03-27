#!/usr/bin/env python3
"""
compare.py — GPU-only vs Hybrid 벤치마크 비교 리포트 생성

사용법:
    python compare.py [--results-dir results] [--gpu-label gpu_only] [--hybrid-label hybrid]

출력:
    results/comparison.txt  — 텍스트 리포트
    results/comparison.json — JSON 요약
    (콘솔에도 출력)
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# 벤치마크 JSON 로드
# ---------------------------------------------------------------------------

BENCH_KEYS = {
    "request_throughput":    ("Request throughput (req/s)", "higher_better"),
    "output_throughput":     ("Output token throughput (tok/s)", "higher_better"),
    "total_token_throughput":("Total token throughput (tok/s)", "higher_better"),
    "duration":              ("Benchmark duration (s)", "lower_better"),
    "mean_ttft_ms":          ("Mean TTFT (ms)", "lower_better"),
    "median_ttft_ms":        ("Median TTFT (ms)", "lower_better"),
    "p99_ttft_ms":           ("P99 TTFT (ms)", "lower_better"),
    "mean_tpot_ms":          ("Mean TPOT (ms)", "lower_better"),
    "median_tpot_ms":        ("Median TPOT (ms)", "lower_better"),
    "p99_tpot_ms":           ("P99 TPOT (ms)", "lower_better"),
    "mean_itl_ms":           ("Mean ITL (ms)", "lower_better"),
    "median_itl_ms":         ("Median ITL (ms)", "lower_better"),
    "p99_itl_ms":            ("P99 ITL (ms)", "lower_better"),
}


def load_bench(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] 파일 없음: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 모니터링 CSV 요약 통계
# ---------------------------------------------------------------------------

def summarize_monitor_csv(csv_path: Path) -> dict | None:
    """GPU 또는 CPU CSV를 읽어 평균/최대 요약 반환."""
    if not csv_path.exists():
        return None
    try:
        import csv
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            return None

        headers = list(rows[0].keys())
        util_cols = [h for h in headers if h.endswith("_util_pct") or h.endswith("_power_w")]

        summary = {}
        for col in util_cols:
            vals = [float(r[col]) for r in rows if r.get(col, "") not in ("", "N/A")]
            if vals:
                summary[col] = {
                    "mean": round(sum(vals) / len(vals), 2),
                    "max": round(max(vals), 2),
                    "min": round(min(vals), 2),
                }
        summary["sample_count"] = len(rows)
        summary["duration_s"] = round(float(rows[-1]["elapsed_s"]) - float(rows[0]["elapsed_s"]), 1) if len(rows) > 1 else 0.0
        return summary
    except Exception as e:
        print(f"[WARN] CSV 분석 실패 ({csv_path.name}): {e}")
        return None


# ---------------------------------------------------------------------------
# 비교 리포트 생성
# ---------------------------------------------------------------------------

def build_report(gpu_data: dict, hyb_data: dict,
                 gpu_mon_gpu: dict | None, gpu_mon_cpu: dict | None,
                 hyb_mon_gpu: dict | None, hyb_mon_cpu: dict | None,
                 args) -> tuple[str, dict]:

    lines = []
    result_json = {
        "generated_at": datetime.now().isoformat(),
        "gpu_only": {},
        "hybrid": {},
        "comparison": {},
        "gpu_utilization": {},
        "cpu_utilization": {},
    }

    # --- 헤더 ---
    lines.append("=" * 70)
    lines.append("  vLLM Hybrid Benchmark Comparison Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # --- 환경 정보 ---
    def _env(data: dict) -> str:
        parts = []
        if "model_id" in data:
            parts.append(f"model={data['model_id']}")
        if "num_prompts" in data:
            parts.append(f"prompts={data['num_prompts']}")
        return ", ".join(parts) if parts else "N/A"

    lines.append(f"  GPU-only  : {_env(gpu_data)}")
    lines.append(f"  Hybrid    : {_env(hyb_data)}")
    lines.append("")

    # --- 메트릭 비교 테이블 ---
    lines.append("-" * 70)
    lines.append(f"  {'지표':<40} {'GPU Only':>10} {'Hybrid':>10} {'차이':>8}")
    lines.append("-" * 70)

    for key, (label, direction) in BENCH_KEYS.items():
        g_val = gpu_data.get(key)
        h_val = hyb_data.get(key)
        if g_val is None or h_val is None:
            continue

        diff_pct = (h_val - g_val) / abs(g_val) * 100 if g_val != 0 else 0.0
        if direction == "higher_better":
            marker = "▲" if diff_pct > 1 else ("▼" if diff_pct < -1 else "~")
        else:
            marker = "▼" if diff_pct > 1 else ("▲" if diff_pct < -1 else "~")

        # 단위별 포맷
        if "throughput" in key:
            fmt = ".1f"
        elif "duration" in key:
            fmt = ".2f"
        else:
            fmt = ".2f"

        lines.append(
            f"  {label:<40} {g_val:{fmt}>10} {h_val:{fmt}>10} {diff_pct:+.1f}% {marker}"
        )

        result_json["gpu_only"][key] = g_val
        result_json["hybrid"][key] = h_val
        result_json["comparison"][key] = {"diff_pct": round(diff_pct, 2), "direction": direction}

    lines.append("-" * 70)
    lines.append("")

    # --- 핵심 요약 ---
    req_g = gpu_data.get("request_throughput", 0)
    req_h = hyb_data.get("request_throughput", 0)
    tok_g = gpu_data.get("output_throughput", 0)
    tok_h = hyb_data.get("output_throughput", 0)
    ttft_g = gpu_data.get("mean_ttft_ms", 0)
    ttft_h = hyb_data.get("mean_ttft_ms", 0)

    speedup = req_h / req_g if req_g > 0 else 0
    tok_speedup = tok_h / tok_g if tok_g > 0 else 0
    ttft_gain = (1 - ttft_h / ttft_g) * 100 if ttft_g > 0 else 0

    lines.append("  [핵심 요약]")
    lines.append(f"  Request throughput: {req_g:.2f} → {req_h:.2f} req/s  ({speedup:.1%})")
    lines.append(f"  Output tok/s:       {tok_g:.0f} → {tok_h:.0f} tok/s  ({tok_speedup:.1%})")
    lines.append(f"  Mean TTFT gain:     {ttft_gain:+.1f}%")
    lines.append("")

    result_json["comparison"]["request_throughput_speedup"] = round(speedup, 4)
    result_json["comparison"]["output_tok_speedup"] = round(tok_speedup, 4)
    result_json["comparison"]["ttft_gain_pct"] = round(ttft_gain, 2)

    # --- GPU/CPU 활용률 요약 ---
    def _format_mon(label: str, mon: dict | None, lines: list, key: str):
        if not mon:
            lines.append(f"  {label}: 데이터 없음 (monitor CSV 미존재)")
            return
        lines.append(f"  {label} ({mon.get('sample_count', 0)} samples, {mon.get('duration_s', 0)}s):")
        util_avg_keys = [k for k in mon if k.endswith("_util_pct") and "avg" in k]
        util_card_keys = sorted([k for k in mon if k.endswith("_util_pct") and "avg" not in k])
        power_keys = [k for k in mon if k.endswith("_power_w")]

        for k in util_avg_keys:
            v = mon[k]
            lines.append(f"    {k:<30} avg={v['mean']:5.1f}%  max={v['max']:5.1f}%  min={v['min']:5.1f}%")
        for k in util_card_keys:
            v = mon[k]
            lines.append(f"    {k:<30} avg={v['mean']:5.1f}%  max={v['max']:5.1f}%")
        for k in power_keys:
            v = mon[k]
            lines.append(f"    {k:<30} avg={v['mean']:6.1f}W  max={v['max']:6.1f}W")

    lines.append("-" * 70)
    lines.append("  [GPU 활용률]")
    _format_mon("GPU-only", gpu_mon_gpu, lines, "gpu_only")
    lines.append("")
    _format_mon("Hybrid  ", hyb_mon_gpu, lines, "hybrid")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  [CPU 활용률]")
    _format_mon("GPU-only", gpu_mon_cpu, lines, "gpu_only")
    lines.append("")
    _format_mon("Hybrid  ", hyb_mon_cpu, lines, "hybrid")
    lines.append("")
    lines.append("=" * 70)

    result_json["gpu_utilization"] = {
        "gpu_only": gpu_mon_gpu or {},
        "hybrid": hyb_mon_gpu or {},
    }
    result_json["cpu_utilization"] = {
        "gpu_only": gpu_mon_cpu or {},
        "hybrid": hyb_mon_cpu or {},
    }

    return "\n".join(lines), result_json


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU-only vs Hybrid comparison report")
    default_rd = os.environ.get("EVAL_RUN_DIR", "results")
    parser.add_argument("--results-dir", default=default_rd, help="Results directory (default: EVAL_RUN_DIR or 'results')")
    parser.add_argument("--gpu-label", default="gpu_only", help="GPU-only result label")
    parser.add_argument("--hybrid-label", default="hybrid", help="Hybrid result label")
    args = parser.parse_args()

    rd = Path(args.results_dir)

    gpu_bench = load_bench(rd / f"{args.gpu_label}.json")
    hyb_bench = load_bench(rd / f"{args.hybrid_label}.json")

    gpu_mon_gpu = summarize_monitor_csv(rd / f"{args.gpu_label}_monitor_gpu.csv")
    gpu_mon_cpu = summarize_monitor_csv(rd / f"{args.gpu_label}_monitor_cpu.csv")
    hyb_mon_gpu = summarize_monitor_csv(rd / f"{args.hybrid_label}_monitor_gpu.csv")
    hyb_mon_cpu = summarize_monitor_csv(rd / f"{args.hybrid_label}_monitor_cpu.csv")

    report_txt, report_json = build_report(
        gpu_bench, hyb_bench,
        gpu_mon_gpu, gpu_mon_cpu,
        hyb_mon_gpu, hyb_mon_cpu,
        args,
    )

    print(report_txt)

    txt_path = rd / "comparison.txt"
    json_path = rd / "comparison.json"

    txt_path.write_text(report_txt)
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)

    print(f"\n[compare] 저장 완료:")
    print(f"  텍스트 → {txt_path}")
    print(f"  JSON   → {json_path}")


if __name__ == "__main__":
    main()
