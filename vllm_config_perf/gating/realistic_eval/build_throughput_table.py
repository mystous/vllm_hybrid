"""TSK_042 (B) — summ_*.json → 집계 + 셀별 상세 문서 + raw 링크.

stdout = 마크다운(처리량/oracle/util/TTFT·TPOT/α 표 + 셀상세·raw 링크 인덱스). SUMMARY.md 로 tee.
옵션:
  --parquet PATH     168셀 × 전 메트릭 long-format parquet (+ .csv 동시)
  --cells-dir DIR    셀마다 상세 md (summ 메트릭 + raw per-request 분포) 생성
  --raw-file PATH    per_request_raw.jsonl (셀별 분포 계산 + 링크)
  --results-md PATH  같은 마크다운을 헤더 붙여 feature RESULTS.md 로 아카이브(절대경로 링크)

사용: python build_throughput_table.py --dir <OUTDIR> --parquet m.parquet \
        --cells-dir <OUTDIR>/cells --raw-file <OUTDIR>/per_request_raw.jsonl --results-md R.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

CONDS = ["sharegpt", "swebench", "humaneval", "mbpp", "wildchat", "lmsys", "mix"]
FIELDS = ["model", "method", "condition", "n", "n_ok", "n_err", "concurrency",
          "max_tokens", "stream", "wall_total_s", "total_completion_tokens", "output_tps",
          "ttft_ms_p50", "ttft_ms_p99", "tpot_ms_p50", "tpot_ms_p99",
          "accept_rate", "accept_tokens", "draft_tokens",
          "gpu_util", "gpu_mem_mib", "cpu_util"]


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 2)


def _slug(*parts):
    return "__".join(str(p).replace("/", "-") for p in parts)


def load(d):
    recs = []
    for f in sorted(glob.glob(os.path.join(d, "summ_*.json"))):
        try:
            s = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        rec = {k: s.get(k) for k in FIELDS}
        rec["_summ"] = os.path.basename(f)
        pc = s.get("per_corpus_reqtps") or {}
        rec["reqtps_avg"] = round(sum(pc.values()) / len(pc), 1) if pc else None
        recs.append(rec)
    return recs


def load_raw(path):
    rows = []
    if path and os.path.exists(path):
        with open(path) as fh:
            for ln in fh:
                try:
                    rows.append(json.loads(ln))
                except Exception:  # noqa: BLE001
                    pass
    return rows


def cell_raw(rows, m, me, cond):
    """셀 (model,method,condition) 의 per-request raw 행. condition 태그 없는 초기 행은
    corpus==cond 로 보강(mix 는 태그로 분리)."""
    out = []
    for r in rows:
        if r.get("model") != m or r.get("method") != me:
            continue
        c = r.get("condition")
        if cond == "mix":
            if c == "mix":
                out.append(r)
        else:
            if c == cond or (c is None and r.get("corpus") == cond):
                out.append(r)
    return out


def lut(recs, field):
    return {(r["model"], r["method"], r["condition"]): r.get(field) for r in recs}


def metric_table_md(rec):
    o = [f"# {rec['model']} × {rec['method']} × {rec['condition']}", "", "## 메트릭 (집계)",
         "| 메트릭 | 값 |", "|---|---|"]
    pretty = [
        ("output_tps", "output_tps"), ("n_ok/n", f"{rec['n_ok']}/{rec['n']} (err {rec['n_err']})"),
        ("wall_total_s", "wall_total_s"), ("total_completion_tokens", "total_completion_tokens"),
        ("TTFT p50/p99 ms", f"{rec['ttft_ms_p50']}/{rec['ttft_ms_p99']}"),
        ("TPOT p50/p99 ms", f"{rec['tpot_ms_p50']}/{rec['tpot_ms_p99']}"),
        ("accept α (acc/draft)",
         f"{rec['accept_rate']} ({rec['accept_tokens']}/{rec['draft_tokens']})"
         if rec['accept_rate'] is not None else "None (vanilla)"),
        ("GPU util / mem MiB", f"{rec['gpu_util']} / {rec['gpu_mem_mib']}"),
        ("CPU util", "cpu_util"), ("reqtps_avg", "reqtps_avg"),
        ("concurrency / max_tokens / stream",
         f"{rec['concurrency']} / {rec['max_tokens']} / {rec['stream']}"),
    ]
    for label, key in pretty:
        v = rec.get(key) if isinstance(key, str) and key in rec else key
        o.append(f"| {label} | {v} |")
    return o


def write_cell_docs(recs, raw_rows, cells_dir):
    os.makedirs(cells_dir, exist_ok=True)
    index = []   # (rec, cell_md_basename)
    for rec in recs:
        m, me, cond = rec["model"], rec["method"], rec["condition"]
        name = f"cell_{_slug(m, me, cond)}.md"
        o = metric_table_md(rec)
        cr = [r for r in cell_raw(raw_rows, m, me, cond) if r.get("ok")]
        if cr:
            o += ["", f"## per-request 분포 (raw 기반, n={len(cr)})",
                  "| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |",
                  "|---|---|---|---|---|"]
            ct = [r.get("completion_tokens") for r in cr if r.get("completion_tokens") is not None]
            wl = [r.get("wall_ms") for r in cr if r.get("wall_ms") is not None]
            tf = [r.get("ttft_ms") for r in cr if r.get("ttft_ms") is not None]
            tp = [r.get("tpot_ms") for r in cr if r.get("tpot_ms") is not None]
            for lab, q in [("min", 0.0), ("p50", 0.5), ("p99", 0.99), ("max", 1.0)]:
                o.append(f"| {lab} | {_pct(ct, q)} | {_pct(wl, q)} | {_pct(tf, q)} | {_pct(tp, q)} |")
        o += ["", "## raw / 원시 데이터",
              f"- 집계 원본: [`{rec['_summ']}`](../{rec['_summ']})",
              "- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터",
              "  ```bash",
              f'  jq -c \'select(.model=="{m}" and .method=="{me}" and '
              + (f'.condition=="{cond}")\'' if cond == "mix" else
                 f'(.condition=="{cond}" or (.condition==null and .corpus=="{cond}")))\'')
              + " ../per_request_raw.jsonl",
              "  ```"]
        with open(os.path.join(cells_dir, name), "w") as fh:
            fh.write("\n".join(o) + "\n")
        index.append((rec, name))
    return index


def build_md(recs, index, link_base="", cells_rel="cells"):
    models = sorted({r["model"] for r in recs})
    methods = sorted({r["method"] for r in recs})
    tps = lut(recs, "output_tps")
    o, p = [], None
    o = []
    def p(*a):
        o.append(" ".join(str(x) for x in a))

    p("# TSK_042 (B) 측정 결과 — 실 trace, conc=32, max_tokens=8192, stream, TP=8(7B Qwen만 4)\n")
    p(f"모델 {len(models)} × method {len(methods)} × 조건 {len(CONDS)}. 셀 {len(recs)}개.\n")

    p("## 처리량 (output_tps) — 조건별 model×method\n")
    for cond in CONDS:
        if not any((m, me, cond) in tps for m in models for me in methods):
            continue
        p(f"### {cond}")
        p("| model | " + " | ".join(methods) + " | best | suffix vs van |")
        p("|---" * (len(methods) + 3) + "|")
        for m in models:
            vals = {me: tps.get((m, me, cond)) for me in methods}
            cells = [f"{vals[me]:,.0f}" if vals[me] else "—" for me in methods]
            best = max((me for me in methods if vals[me]), key=lambda me: vals[me], default="—")
            van, suf = vals.get("vanilla"), vals.get("suffix")
            sv = f"{(suf/van-1)*100:+.0f}%" if van and suf else "—"
            p(f"| {m} | " + " | ".join(cells) + f" | {best} | {sv} |")
        p("")

    p("## condition-level oracle (model × condition → best method) — regret 입력")
    p("| model | " + " | ".join(CONDS) + " |")
    p("|---" * (len(CONDS) + 1) + "|")
    for m in models:
        row = [max((me for me in methods if tps.get((m, me, c))),
                   key=lambda me: tps.get((m, me, c)), default="—") for c in CONDS]
        p(f"| {m} | " + " | ".join(row) + " |")
    p("")

    for label, f50, f99 in [("TTFT", "ttft_ms_p50", "ttft_ms_p99"),
                            ("TPOT", "tpot_ms_p50", "tpot_ms_p99")]:
        l50, l99 = lut(recs, f50), lut(recs, f99)
        p(f"## 지연 {label} (mix, p50/p99 ms)")
        p("| model | " + " | ".join(methods) + " |")
        p("|---" * (len(methods) + 1) + "|")
        for m in models:
            cells = []
            for me in methods:
                a, b = l50.get((m, me, "mix")), l99.get((m, me, "mix"))
                cells.append(f"{a}/{b}" if a is not None else "—")
            p(f"| {m} | " + " | ".join(cells) + " |")
        p("")

    spec = [me for me in methods if me != "vanilla"]
    if spec:
        ar = lut(recs, "accept_rate")
        p("## accept α (mix, accepted/draft) — spec method")
        p("| model | " + " | ".join(spec) + " |")
        p("|---" * (len(spec) + 1) + "|")
        for m in models:
            cells = [f"{ar.get((m, me, 'mix')):.3f}" if ar.get((m, me, "mix")) is not None
                     else "—" for me in spec]
            p(f"| {m} | " + " | ".join(cells) + " |")
        p("")

    gu, cu, gm = lut(recs, "gpu_util"), lut(recs, "cpu_util"), lut(recs, "gpu_mem_mib")
    p("## util (mix, gpu%/cpu%, gpu_mem GiB)")
    p("| model | " + " | ".join(methods) + " |")
    p("|---" * (len(methods) + 1) + "|")
    for m in models:
        cells = []
        for me in methods:
            g, c, mem = gu.get((m, me, "mix")), cu.get((m, me, "mix")), gm.get((m, me, "mix"))
            cells.append(f"{g}/{c} ({mem/1024:.0f}G)" if g is not None and mem else
                         (f"{g}/{c}" if g is not None else "—"))
        p(f"| {m} | " + " | ".join(cells) + " |")
    p("")

    # raw 데이터 + 셀 상세 인덱스 (링크)
    p("## raw 데이터")
    p(f"- per-request 전체 로그: [`per_request_raw.jsonl`]({link_base}per_request_raw.jsonl) "
      "(corpus·ok·wall_ms·ttft_ms·tpot_ms·completion/prompt_tokens·model·method·condition)")
    p(f"- long-format 메트릭: [`metrics_table.parquet`]({link_base}metrics_table.parquet) / "
      f"[`metrics_table.csv`]({link_base}metrics_table.csv)")
    p(f"- 백엔드/측정 로그: `{link_base}_logs/`\n")
    if index:
        p("## 셀 상세 문서 (셀별 메트릭 + raw 분포)")
        p("| model | method | condition | tps | 상세 | summ |")
        p("|---|---|---|---|---|---|")
        for rec, name in index:
            t = rec.get("output_tps")
            p(f"| {rec['model']} | {rec['method']} | {rec['condition']} | "
              f"{t:,.0f} | [상세]({link_base}{cells_rel}/{name}) | "
              f"[json]({link_base}{rec['_summ']}) |")
        p("")
    return "\n".join(o)


def write_parquet(recs, path):
    cols = [c for c in FIELDS] + ["reqtps_avg"]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist([{c: r.get(c) for c in cols} for r in recs]), path)
    except Exception as e:  # noqa: BLE001
        print(f"<!-- parquet 실패: {e} -->")
    with open(os.path.splitext(path)[0] + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({c: r.get(c) for c in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--cells-dir", default=None)
    ap.add_argument("--raw-file", default=None)
    ap.add_argument("--results-md", default=None)
    args = ap.parse_args()

    recs = load(args.dir)
    if not recs:
        print("(no summaries)"); return
    raw_rows = load_raw(args.raw_file)
    index = write_cell_docs(recs, raw_rows, args.cells_dir) if args.cells_dir else []

    print(build_md(recs, index, link_base="", cells_rel="cells"))   # SUMMARY.md (상대경로)

    if args.parquet:
        write_parquet(recs, args.parquet)
    if args.results_md:
        os.makedirs(os.path.dirname(args.results_md), exist_ok=True)
        base = os.path.abspath(args.dir).rstrip("/") + "/"   # RESULTS.md 는 절대경로 링크
        hdr = ("# TSK_042 워크로드 활용 실험 — 측정 결과 (아카이브)\n\n"
               f"> parent IDE_022. source: `{args.dir}`. 셀 {len(recs)}개. "
               "원시 summ_*.json + per_request_raw.jsonl + metrics_table.parquet + cells/ 동봉.\n\n---\n\n")
        with open(args.results_md, "w") as fh:
            fh.write(hdr + build_md(recs, index, link_base=base, cells_rel="cells") + "\n")


if __name__ == "__main__":
    main()
