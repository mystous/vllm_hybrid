"""SUB_199 B200 sweep 결과 finalize.

run dir 의 {scenario}__{workload}.json 들을 읽어
  (1) RESULTS.md 용 markdown 표 출력
  (2) _ALL_MEASUREMENTS.csv 에 sub=199 행 append (중복 가드)

사용: python finalize_results.py <run_dir> [--write-csv]
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path

WORKLOADS = ["sonnet", "chat", "code", "balanced", "sonnet-heavy", "code-heavy"]
SCENARIOS = ["vanilla", "trident", "AGSD"]
CSV_PATH = Path("/workspace/host_vllm_hybrid/shadow_assists/features/IDE_006/"
                "TSK_020/measurements/_ALL_MEASUREMENTS.csv")


def load(run_dir: Path):
    data = {}
    for wl in WORKLOADS:
        for sc in SCENARIOS:
            f = run_dir / f"{sc}__{wl}.json"
            if f.exists():
                s = json.load(open(f))["summary"]
                data[(wl, sc)] = s
    return data


def md_table(data) -> str:
    lines = ["| workload | vanilla | trident | AGSD | trident vs vanilla | AGSD vs vanilla |",
             "|---|---:|---:|---:|---:|---:|"]
    for wl in WORKLOADS:
        v = data.get((wl, "vanilla"), {}).get("output_tps")
        t = data.get((wl, "trident"), {}).get("output_tps")
        a = data.get((wl, "AGSD"), {}).get("output_tps")
        def fmt(x): return f"{x:,.1f}" if isinstance(x, (int, float)) else "⏳"
        def pct(x, base):
            if isinstance(x, (int, float)) and isinstance(base, (int, float)) and base:
                return f"{(x/base-1)*100:+.1f}%"
            return "—"
        lines.append(f"| {wl} | {fmt(v)} | {fmt(t)} | {fmt(a)} | "
                     f"{pct(t, v)} | {pct(a, v)} |")
    return "\n".join(lines)


def csv_rows(data):
    rows = []
    for (wl, sc), s in data.items():
        # vanilla/trident = TP=8 단일 인스턴스, AGSD = TP=4×2 라우터
        tp = "4" if sc == "AGSD" else "8"
        kind = "router-2inst-b200" if sc == "AGSD" else "single-tp8-b200"
        rows.append(["b200", "Qwen2.5-32B-B200", tp, wl, sc, kind,
                     f"{s['output_tps']:.2f}", f"{s['wall_total_s']:.2f}", "", ""])
    return rows


def main():
    run_dir = Path(sys.argv[1])
    write_csv = "--write-csv" in sys.argv
    data = load(run_dir)
    print(f"# loaded {len(data)}/18 cells from {run_dir}\n")
    print(md_table(data))
    if write_csv:
        existing = CSV_PATH.read_text().splitlines()
        if any(ln.startswith("b200,") for ln in existing):
            print("\n[csv] sub=b200 rows already present — skip append")
        else:
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerows(csv_rows(data))
            print(f"\n[csv] appended {len(data)} rows (sub=b200) → {CSV_PATH.name}")


if __name__ == "__main__":
    main()
