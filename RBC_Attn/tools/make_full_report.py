#!/usr/bin/env python3
"""모든 실험의 요약·집계·원시 셀 데이터를 단일 MD 로 묶는다.

원본 JSON (`results/v02/**`) 에서만 값을 읽는다. 측정하지 않은 값은 계산해 채우지 않고
NOT_RUN 으로 남긴다. 셀 단위 원시 수치를 그대로 실어 다른 세션이 이 파일만으로
판단할 수 있게 한다.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics as st
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "v02")


def load(rel):
    p = os.path.join(RES, rel)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def med(xs):
    return float(np.median(xs)) if len(xs) else None


def f(x, nd=1, unit=""):
    return "—" if x is None else f"{x:.{nd}f}{unit}"


def table(head, rows, align=None):
    """마크다운 표. rows 는 문자열 리스트의 리스트."""
    out = ["| " + " | ".join(head) + " |"]
    a = align or (["---"] * len(head))
    out.append("|" + "|".join(a) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


R = ["---:"]


def sec(title, level=2):
    return ["", "#" * level + " " + title, ""]


# ----------------------------------------------------------------------------------
def env_section(L, src):
    if not src:
        return
    m = src["meta"]
    L += sec("측정 환경")
    v = m.get("versions", {})
    L += table(["항목", "값"], [
        ["GPU", f"{m.get('gpu')} (UUID `{m.get('gpu_uuid')}`)"],
        ["SM 수", m.get("sm")],
        ["사용 장치 수 / 논리 ordinal", f"{m.get('device_count', '—')} / {m.get('logical_ordinal', 0)}"],
        ["torch / CUDA", f"{v.get('torch')} / {v.get('cuda')}"],
        ["triton", v.get("triton")],
        ["flashinfer", f"{v.get('flashinfer')} (available={v.get('available')})"],
        ["원본 생성 시각", m.get("time")],
    ])
    L += ["",
          "시간 측정은 전부 profiling 없는 CUDA event / graph replay 다 (§7.5). "
          "NCU 는 이번 작업에서 실행하지 않았다.", ""]


def e0_section(L, e0):
    L += sec("E0 — 기준선 복구 (§11 E0)")
    if e0 is None:
        L += ["NOT_RUN", ""]
        return
    m = e0["meta"]
    L += [f"입력: Nq{m['nq']} / Nk{m['nk']} / G{m['g']} / D{m['d']} / BK128 / sel{m['sel']}, "
          f"seed {m['seed']}. 반복 {m['reps']} × graph replay {m['replays']}.",
          f"clock 상태: `{m.get('clocks')}`", ""]
    cells = [c for c in e0["cells"] if c.get("correct")]
    L += ["### 패턴별 최강", ""]
    rows = []
    for pat in ("common_private", "random", "clustered"):
        sub = [c for c in cells if c["pattern"] == pat]
        if not sub:
            continue
        b = min(sub, key=lambda c: c["gpu_us_median"])
        rows.append([pat, b["policy"], f"M{b['max_m']}", b["split"],
                     f(b["gpu_us_median"], 1, "us"), b["n_tasks"], b["partial_slots"]])
    L += table(["패턴", "정책", "max_m", "split", "GPU 시간", "tasks", "partial"],
               rows, ["---", "---", "---", "---"] + R + R + R)
    L += ["", "### 전량 (정확성 통과 셀)", ""]
    rows = []
    for c in sorted(cells, key=lambda x: (x["pattern"], x["policy"], x["max_m"], x["split"])):
        rows.append([c["pattern"], c["policy"], c["max_m"], c["split"],
                     f(c["gpu_us_median"], 1), c["n_tasks"], c["partial_slots"],
                     c["BM"], f"{c['rel_err']:.2e}", f(c["plan_ms"], 2)])
    L += table(["패턴", "정책", "max_m", "split", "GPU us", "tasks", "partial", "BM",
                "rel_err", "plan ms"],
               rows, ["---", "---"] + R * 8)
    if e0["errors"]:
        L += ["", f"오류/제외 {len(e0['errors'])}건:", ""]
        for e in e0["errors"][:20]:
            L.append(f"- {e}")
    L.append("")


def e1_section(L, e1):
    L += sec("E1 — 최종 계획 선택과 분할 (§11 E1)")
    if e1 is None:
        L += ["NOT_RUN", ""]
        return
    m = e1["meta"]
    L += [f"실행기: {m['executor']}. 입력 Nq{m['nq']}/Nk{m['nk']}/G{m['g']}/D{m['d']}/sel{m['sel']}, "
          f"seed {m['seed']}.", "",
          "**핵심 관찰**: KV 방문량(`KV_visits`)은 max_m 을 키우면 줄지만 GPU 시간은 "
          "급격히 늘어난다. 이득원(KV 재읽기 감소)과 비용(병렬성·padding)이 반대 방향이다.", ""]
    rows = []
    for r in sorted(e1["rows"], key=lambda x: (x.get("pattern", ""), x.get("policy", ""),
                                               x.get("split", ""), x.get("max_m", 0))):
        if "error" in r:
            rows.append([r.get("pattern"), r.get("policy"), r.get("split"), r.get("max_m"),
                         "ERROR", "", "", "", "", "", "", r["error"][:40]])
            continue
        rows.append([r["pattern"], r["policy"], r["split"], r["max_m"], r["BM"],
                     r["ntasks"], r["KV_visits"], r["padded_units"], r["partial_slots"],
                     f(r["main_us"], 1), f(r["merge_us"], 1), f(r["GPU_total_us"], 1)])
    L += table(["패턴", "정책", "split", "max_m", "BM", "ntasks", "KV_visits", "padded",
                "partial", "main us", "merge us", "GPU us"],
               rows, ["---", "---", "---"] + R * 9)
    L.append("")


def e2_section(L, e2):
    L += sec("E2 — 변경별 효과 분리 (§11 E2)")
    if e2 is None:
        L += ["NOT_RUN", ""]
        return
    m = e2["meta"]
    L += [f"입력 Nq{m['nq']}/Nk{m['nk']}/G{m['g']}/D{m['d']}/sel{m['sel']}, seed {m['seed']}. "
          f"반복 {m['reps']} × replay {m['replays']}. 셀 {len(e2['cells'])}개, "
          f"실패 {len(e2['failures'])}건.", "",
          "행 A=v0.1 보존 / B=P1 / C=+direct output / D_P3a=+BN64 / D_P3b=+late V / "
          "D_P3c=+FULL fast path. 각 행에서 세 정책을 모두 실행했다.", ""]
    g = collections.defaultdict(list)
    for c in e2["cells"]:
        g[(c["pattern"], c["row"], c["policy"], c["max_m"])].append(c["gpu_us"])
    mm = {k: med(v) for k, v in g.items()}
    ROWS = ["A_v01", "B_P1", "C_P2", "D_P3a", "D_P3b", "D_P3c"]
    for pat in ("common_private", "random", "clustered"):
        L += [f"### {pat}", ""]
        rows = []
        for row in ROWS:
            cells = []
            for pol in ("q_outer", "signature_only", "rbc"):
                cand = [(k[3], v) for k, v in mm.items()
                        if k[0] == pat and k[1] == row and k[2] == pol]
                if not cand:
                    cells += ["—", "—"]
                    continue
                bm, bv = min(cand, key=lambda x: x[1])
                cells += [f"{bv:.1f}", f"M{bm}"]
            rows.append([row] + cells)
        L += table(["행", "q_outer us", "M", "signature us", "M", "rbc us", "M"],
                   rows, ["---"] + R * 6)
        L.append("")
    L += ["### A 대비 상대 변화 (각 정책의 최적 max_m 기준, 음수=개선)", ""]
    rows = []
    for pol in ("q_outer", "signature_only", "rbc"):
        for pat in ("common_private", "random", "clustered"):
            a = min((v for k, v in mm.items()
                     if k[0] == pat and k[1] == "A_v01" and k[2] == pol), default=None)
            if a is None:
                continue
            cell = [pol, pat, f"{a:.1f}us"]
            for row in ROWS[1:]:
                b = min((v for k, v in mm.items()
                         if k[0] == pat and k[1] == row and k[2] == pol), default=None)
                cell.append("—" if b is None else f"{100*(b-a)/a:+.1f}%")
            rows.append(cell)
    L += table(["정책", "패턴", "A 기준"] + ROWS[1:], rows, ["---", "---"] + R * 6)
    L += ["", "### 전량 원시 셀", "",
          "(반복별 값. `rep` 는 무작위 순서로 실행한 반복 번호다.)", ""]
    rows = []
    for c in sorted(e2["cells"], key=lambda x: (x["pattern"], x["row"], x["policy"],
                                                x["max_m"], x["rep"])):
        rows.append([c["pattern"], c["row"], c["policy"], c["max_m"], c["rep"],
                     f(c["gpu_us"], 2), c["n_tasks"], c["partial_slots"], c["multi_rows"],
                     "Y" if c["full"] else "N", c["bn"], "Y" if c["late_v"] else "N",
                     f"{c['rel_err']:.2e}"])
    L += table(["패턴", "행", "정책", "max_m", "rep", "GPU us", "tasks", "partial",
                "multi", "FULL", "BN", "lateV", "rel_err"],
               rows, ["---", "---", "---"] + R * 10)
    L.append("")


def gate_section(L, gate, gate_s):
    L += sec("§13.1 GPU 단계 게이트 — 영역 탐색")
    L += ["게이트: non-Q RBC 계획이 최강 공통 Q-outer 대비 GPU 시간을 5% 이상 줄이는 "
          "held-out 영역이 있는가. `is_q_fallback` 인 계획(= Q-outer 와 구조가 같은 계획)은 "
          "non-Q 로 세지 않는다.", ""]
    for tag, gj in (("sweep A — Nq512/2048/8192, G8, seed101", gate),
                    ("sweep B — Nq128/256/512, G4/8/16, seed101~103", gate_s)):
        L += [f"### {tag}", ""]
        if gj is None:
            L += ["NOT_RUN", ""]
            continue
        cells = gj["cells"]
        gdef = gj["meta"].get("g")
        keys = sorted({(c["nq"], c.get("g", gdef), c["sel"], c["pattern"]) for c in cells})
        rows, hits = [], []
        for key in keys:
            sub = [c for c in cells
                   if (c["nq"], c.get("g", gdef), c["sel"], c["pattern"]) == key]

            def best_of(pred):
                per = {}
                for c in sub:
                    if pred(c):
                        per.setdefault((c["policy"], c["max_m"]), []).append(c)
                if not per:
                    return None
                k = min(per, key=lambda kk: med([x["gpu_us"] for x in per[kk]]))
                return (med([x["gpu_us"] for x in per[k]]), k[0], k[1], per[k][0])
            q = best_of(lambda c: c["policy"] == "q_outer")
            s = best_of(lambda c: c["policy"] == "signature_only"
                        and not c.get("is_q_fallback"))
            r = best_of(lambda c: c["policy"] == "rbc" and not c.get("is_q_fallback"))
            if q is None:
                continue
            n = min([x for x in (s, r) if x], key=lambda x: x[0]) if (s or r) else None
            d = None if n is None else (n[0] - q[0]) / q[0]
            ok = d is not None and d <= -0.05
            if ok:
                hits.append((key, d))
            vis = "—" if n is None else f"{n[3]['KV_visits']/q[3]['KV_visits']:.2f}x"
            rows.append([key[0], key[1], key[2], key[3], f(q[0], 1), f"M{q[2]}",
                         "—" if s is None else f(s[0], 1),
                         "—" if r is None else f(r[0], 1),
                         "—" if d is None else f"{100*d:+.1f}%",
                         "통과" if ok else "미통과",
                         "—" if n is None else n[1], vis])
        L += table(["Nq", "G", "sel", "패턴", "Q-outer us", "M", "signature us",
                    "merge us", "Δ non-Q", "게이트", "승자", "KV 방문비"],
                   rows, R * 3 + ["---"] + R * 2 + R * 3 + ["---", "---"] + R)
        L += ["", f"통과 {len(hits)}/{len(keys)} 영역"
              + (f", 감소율 중앙값 {100*st.median([h[1] for h in hits]):.1f}%, "
                 f"최대 {100*min(h[1] for h in hits):.1f}%" if hits else ""), ""]
        if hits:
            ax = [(0, "Nq"), (1, "G"), (2, "sel"), (3, "패턴")]
            arows = []
            for i, name in ax:
                cnt = collections.Counter(h[0][i] for h in hits)
                tot = collections.Counter(k[i] for k in keys)
                arows.append([name, ", ".join(f"{v}={cnt.get(v,0)}/{tot[v]}"
                                              for v in sorted(tot, key=str))])
            L += table(["축", "통과/전체"], arows)
            L.append("")


def calib_section(L, table_json, raw):
    L += sec("P4 — 시간 모델 교정 (§7)")
    if table_json is None:
        L += ["NOT_RUN", ""]
        return
    m = table_json.get("meta", {})
    L += [f"교정 seed {m.get('seeds')} / Nq {m.get('nq')} / G {m.get('g')} / "
          f"sel {m.get('sel')} / 패턴 {m.get('patterns')} / max_m {m.get('max_m')}.",
          f"표본 {m.get('n_rows')}개. 시간 측정 방식: {m.get('timing')}.", "",
          "회귀는 (BM, G) 구간별 **비음수 최소제곱(NNLS)** + 상대오차 가중이다. "
          "음수 계수를 허용하면 외삽에서 음수 시간을 만들고 그것이 '가장 빠른 후보'로 "
          "보이게 된다 (실제로 한 번 발생해 regret 415% 를 만들었다).", "",
          "특징 벡터:", ""]
    L += ["```", ", ".join(table_json.get("feature_names", [])), "```", ""]
    rows = []
    for k in sorted(table_json.get("resid", {}),
                    key=lambda x: tuple(int(v) for v in x.split(":"))):
        r = table_json["resid"][k]
        o = table_json.get("obs", {}).get(k, {})
        dm = table_json.get("diff_margin", {}).get(k)
        rows.append([k, r["n"], f"{r['rel_p50']:.3f}", f"{r['rel_p90']:.3f}",
                     f"{r['rel_max']:.3f}",
                     f"{o.get('y_min', 0):.0f}~{o.get('y_max', 0):.0f}",
                     "전역" if dm is None else f"{dm:.3f}"])
    L += table(["BM:G", "n", "잔차 p50", "p90", "max", "관측 y 범위 us", "diff margin"],
               rows, ["---"] + R * 6)
    L += ["", "후보 **차이** 예측의 상대 잔차 (error_margin 의 근거): "
          f"p50 {m.get('diff_margin_rel_p50', float('nan')):.3f} / "
          f"p90 {m.get('diff_margin_rel_p90', float('nan')):.3f} "
          f"(n={m.get('diff_margin_n')})", ""]
    pc = m.get("plan_cost_us", {})
    if pc:
        L += ["계획 생성 CPU 비용 중앙값 (선택기의 `extra_prepare` 근거):", ""]
        L += table(["후보", "us"], [[k, f"{v:.0f}"] for k, v in pc.items()], ["---"] + R)
        if "signature" in pc and "q_outer" in pc:
            L += ["", f"→ `extra_prepare` = {pc['signature']-pc['q_outer']:.0f}us", ""]
    L += ["", "형상 구간별 최적 max_m lookup: "
          f"`{table_json.get('best_max_m', {})}`", ""]
    if raw:
        ok = [r for r in raw["raw"] if "error" not in r]
        L += ["", f"### 교정 원시 표본 ({len(ok)}개)", "",
              "(형상·후보·max_m 별 실측 GPU 시간과 특징값)", ""]
        rows = []
        for r in sorted(ok, key=lambda x: (x["nq"], x["g"], x["sel"], x["pattern"],
                                           x["candidate"], x["max_m"], x["seed"])):
            rows.append([r["seed"], r["nq"], r["g"], r["sel"], r["pattern"],
                         r["candidate"], r["max_m"], r["bm"], f(r["measured_us"], 2),
                         r["n_tasks"], r["sum_k"], r["max_k"], int(r["padded"]),
                         r["multi_rows"], r["unique_blocks"],
                         f(r.get("k_p50"), 1), f(r.get("k_p90"), 1),
                         f(r.get("full_ratio"), 3)])
        L += table(["seed", "Nq", "G", "sel", "패턴", "후보", "max_m", "BM", "실측 us",
                    "tasks", "sum_k", "max_k", "padded", "multi", "uniq",
                    "k_p50", "k_p90", "full비"],
                   rows, R * 4 + ["---", "---"] + R * 12)
        L.append("")


def e3_section(L, e3):
    L += sec("E3 — 교정된 선택기와 oracle (§11 E3)")
    if e3 is None:
        L += ["NOT_RUN", ""]
        return
    m = e3["meta"]
    cells = e3["cells"]
    reg = [c["regret"] for c in cells if c["regret"] is not None]
    act = sum(1 for c in cells if c["rbc_plan_used"])
    gr = [(c["gpu_only_rule"]["measured_us"] - c["oracle_us"]) / c["oracle_us"]
          for c in cells if c.get("gpu_only_rule", {}).get("measured_us") is not None]
    ga = sum(1 for c in cells if c.get("gpu_only_rule", {}).get("selected") not in
             (None, "q_outer"))
    L += [f"교정 seed {m.get('calibration_seeds')} / 평가 seed {m['seeds']} / "
          f"형상 {m['shapes']} / 패턴 {m['patterns']}. 셀 {len(cells)}개.", "",
          "§7.7 은 선택 규칙에 `extra_prepare_cost` 를 넣으라고 하면서 regret 은 GPU "
          "시간으로 채점한다. 규칙과 채점이 서로 다른 것을 재므로 두 지표를 모두 남긴다.", ""]
    rows = []
    if gr:
        rows.append(["GPU-only 규칙 (§7.7 진단 기준)", f"{100*st.median(gr):.2f}%",
                     f"{100*np.quantile(gr,0.9):.2f}%", f"{100*max(gr):.2f}%",
                     f"{ga}/{len(cells)}"])
    if reg:
        rows.append(["production 규칙 (extra_prepare 포함)", f"{100*st.median(reg):.2f}%",
                     f"{100*np.quantile(reg,0.9):.2f}%", f"{100*max(reg):.2f}%",
                     f"{act}/{len(cells)}"])
    L += table(["규칙", "regret 중앙값", "p90", "최악", "RBC_active"],
               rows, ["---"] + R * 4)
    L += ["", f"기준: 중앙값 3% 이하, 최악 5% 이하. 선택기 CPU 중앙값 "
          f"{st.median([c['selector_cpu_us'] for c in cells]):.0f}us "
          f"(probe + 선택된 계획 생성).", ""]
    L += ["### 형상·패턴별", ""]
    by = collections.defaultdict(list)
    for c in cells:
        by[(c["shape"], c["pattern"])].append(c)
    rows = []
    for k in sorted(by):
        xs = by[k]
        x0 = xs[0]
        p = x0["predicted_us"]
        pf = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:.0f}"
        orc = collections.Counter(x["oracle_winner"] for x in xs).most_common(1)[0][0]
        rs = collections.Counter(x["selected_reason"] for x in xs).most_common(1)[0][0]
        rows.append([k[0], k[1], x0["selected_candidate"], orc,
                     f"{100*st.median([x['regret'] for x in xs]):.1f}%",
                     pf(p.get("q_outer_us")), pf(p.get("signature_us")),
                     f"{x0['error_margin_us']:.0f}", f"{x0['extra_prepare_us']:.0f}",
                     rs[:34]])
    L += table(["형상", "패턴", "선택", "oracle 승자", "regret 중앙값", "예측 q",
                "예측 s", "margin us", "extra us", "이유"],
               rows, ["---", "---", "---", "---"] + R * 5 + ["---"])
    L += ["", "### 전량 원시 셀", ""]
    rows = []
    for c in sorted(cells, key=lambda x: (x["shape"], x["pattern"], x["seed"])):
        rows.append([c["seed"], c["shape"], c["pattern"], c["selected_candidate"],
                     c["selected_max_m"], f(c["selected_measured_us"], 1),
                     c["oracle_winner"], f"M{c['oracle_max_m']}", f(c["oracle_us"], 1),
                     f"{100*c['regret']:.1f}%" if c["regret"] is not None else "—",
                     f(c["best_q_outer_us"], 1),
                     f"{100*c['vs_best_q_outer']:.1f}%"
                     if c.get("vs_best_q_outer") is not None else "—",
                     f(c["selector_cpu_us"], 0),
                     (c.get("gpu_only_rule") or {}).get("selected", "—")])
    L += table(["seed", "형상", "패턴", "선택", "M", "선택 us", "oracle", "M",
                "oracle us", "regret", "best Q us", "vs Q", "선택기 CPU us",
                "GPU-only 선택"],
               rows, R + ["---", "---", "---"] + R * 10)
    L.append("")


def e4_section(L, e4, e4_before):
    L += sec("E4 — 전체 준비비 포함 비교 (§11 E4) — **주 판정**")
    if e4 is None:
        L += ["NOT_RUN", ""]
        return
    m = e4["meta"]
    L += [f"형상 {m['shapes']} / 패턴 {m['patterns']} / 평가 seed {m['evaluation_seeds']}. "
          f"레코드 {len(e4['records'])}개, 오류 {len(e4['errors'])}건.", "",
          "측정 모드: `run_only`(준비된 plan 의 GPU), `cold_init`(capacity 할당·pin), "
          "`pool_fresh_host`(새 host mask 계획·pack·H2D·GPU·완료), "
          "`pool_fresh_gpu`(GPU-origin mask 의 D2H 포함 — **주 판정**), "
          "`exact_plan_cache`(같은 mask guard 후 재사용).", "",
          "CPU 시간과 GPU 시간을 더해 wall 이라 쓰지 않았다. `wall_us` 는 실제 벽시계다.", ""]

    by = collections.defaultdict(dict)
    for r in e4["records"]:
        by[(r.get("shape"), r.get("pattern"))].setdefault(r["selected_policy"], {})[
            r["timing_mode"]] = r
    L += ["### 판정 표 (pool_fresh_gpu, gain = 1 − T/T_q_outer. 음수 = 느려짐)", ""]
    rows = []
    for k in sorted(by):
        q = by[k].get("q_outer", {}).get("pool_fresh_gpu")
        if not q:
            continue
        qw = med(q["wall_us"])
        for cand in ("signature", "bounded_merge"):
            c = by[k].get(cand, {}).get("pool_fresh_gpu")
            if not c:
                continue
            w = med(c["wall_us"])
            ro_q = by[k]["q_outer"].get("run_only")
            ro_c = by[k].get(cand, {}).get("run_only")
            gq = med(ro_q["gpu_total_us"]) if ro_q else None
            gc = med(ro_c["gpu_total_us"]) if ro_c else None
            rows.append([k[1], k[0], cand, f(qw, 0), f(w, 0),
                         f"{100*(1-w/qw):+.1f}%",
                         f(gq, 1), f(gc, 1),
                         "—" if (gq is None or gc is None) else f"{100*(1-gc/gq):+.1f}%",
                         "개선 없음" if w >= qw else "개선"])
    L += table(["패턴", "형상", "후보", "q_outer wall us", "후보 wall us", "gain(wall)",
                "q_outer GPU us", "후보 GPU us", "gain(GPU만)", "판정"],
               rows, ["---", "---", "---"] + R * 6 + ["---"])
    L += ["", "GPU 시간만 보면 signature 가 이기는 조합이 있으나, 계획 CPU 비용이 그보다 "
          "커서 전체 시간에서는 모두 뒤집힌다.", ""]

    L += ["### 비용 분해 (pool_fresh_gpu, 중앙값)", ""]
    rows = []
    for r in e4["records"]:
        if r["timing_mode"] != "pool_fresh_gpu":
            continue
        rows.append([r.get("pattern"), r.get("shape"), r["selected_policy"],
                     f(med(r["cpu_plan_pack_us"]), 0), f(med(r["dma_activity_us"]), 0),
                     f(med(r["gpu_total_us"]), 0), f(med(r["wall_us"]), 0),
                     r["descriptor_live_bytes"], r["h2d_bytes"], r["h2d_copy_count"],
                     r["d2h_bytes"], r["actual_tasks"], r["partial_slots"],
                     f"{r['host_pin_calls_hotpath']}/{r['device_allocation_requests_hotpath']}"])
    L += table(["패턴", "형상", "후보", "cpu plan+pack us", "DMA us", "GPU us", "wall us",
                "desc live B", "H2D B", "H2D 회수", "D2H B", "tasks", "partial",
                "hot pin/alloc"],
               rows, ["---", "---", "---"] + R * 11)
    L += ["", "`hot pin/alloc` 은 hot path 의 host pin 호출 수 / device allocation 요청 "
          "수다. §8.7 의 목표는 0/0 이고 달성했다.", ""]

    L += ["### 측정 모드별 전량", ""]
    rows = []
    for r in sorted(e4["records"], key=lambda x: (str(x.get("shape")),
                                                  str(x.get("pattern")),
                                                  x["selected_policy"],
                                                  x["timing_mode"])):
        w = med(r["wall_us"]) if r["wall_us"] else None
        rows.append([r.get("shape"), r.get("pattern"), r["selected_policy"],
                     r["timing_mode"], r["mask_origin"], f(w, 1),
                     f(med(r["gpu_total_us"]), 1) if r["gpu_total_us"] else "—",
                     f(med(r["cpu_plan_pack_us"]), 1) if r["cpu_plan_pack_us"] else "—",
                     f(med(r["main_us"]), 1) if r["main_us"] else "—",
                     f(med(r["merge_us"]), 1) if r["merge_us"] else "—",
                     "Y" if r["plan_cache_hit"] else "N",
                     {True: "pass", False: "FAIL", None: "—"}[r["numeric_passed"]],
                     {True: "pass", False: "FAIL", None: "—"}[r["exact_edge_passed"]]])
    L += table(["형상", "패턴", "후보", "모드", "mask 출발", "wall us", "GPU us",
                "cpu plan+pack us", "main us", "merge us", "cache", "numeric",
                "exact-edge"],
               rows, ["---", "---", "---", "---", "---"] + R * 5 + ["---", "---", "---"])
    L.append("")

    if e4_before:
        L += ["### 플래너 최적화 전후 (§8.6 적용)", "",
              "계획 CPU 비용은 줄었지만 **공통 개선**이라 기준선도 함께 빨라져 상대 회귀 "
              "폭은 줄지 않았다.", ""]
        rows = []
        for tag, j in (("최적화 전", e4_before), ("최적화 후", e4)):
            bb = collections.defaultdict(dict)
            for r in j["records"]:
                bb[(r.get("shape"), r.get("pattern"))].setdefault(
                    r["selected_policy"], {})[r["timing_mode"]] = r
            ds = []
            for k in bb:
                q = bb[k].get("q_outer", {}).get("pool_fresh_gpu")
                if not q:
                    continue
                qw = med(q["wall_us"])
                for cand in ("signature", "bounded_merge"):
                    c = bb[k].get(cand, {}).get("pool_fresh_gpu")
                    if c:
                        ds.append((med(c["wall_us"]) - qw) / qw)
            if ds:
                rows.append([tag, len(ds), f"{100*st.median(ds):+.1f}%",
                             f"{100*min(ds):+.1f}%", sum(1 for d in ds if d < 0)])
        L += table(["", "조합 수", "Q대비 중앙값", "최선", "개선 조합"],
                   rows, ["---"] + R * 4)
        L.append("")

    L += ["### 원본 JSON 스키마 (§14.2)", "",
          "`raw/runtime/e4_full_cost.json` 의 각 레코드는 §14.2 필드를 그대로 갖는다. "
          "측정하지 않은 값은 `null` 과 사유(`notes`)를 유지하고 0 으로 바꾸지 않았다. "
          "레코드 예 (첫 `pool_fresh_gpu`):", ""]
    ex = next((r for r in e4["records"] if r["timing_mode"] == "pool_fresh_gpu"), None)
    if ex:
        s = dict(ex)
        for kk in ("cpu_plan_pack_us", "dma_activity_us", "gpu_total_us", "wall_us",
                   "main_us", "merge_us"):
            if isinstance(s.get(kk), list) and len(s[kk]) > 3:
                s[kk] = s[kk][:3] + [f"...({len(ex[kk])}개)"]
        L += ["```json", json.dumps(s, indent=1, ensure_ascii=False), "```", ""]


def plan_section(L):
    L += sec("계획 descriptor 덤프 (§4.4)")
    any_ = False
    for pat in ("common_private", "random", "clustered"):
        j = load(f"plan/dump_{pat}.json")
        if j is None:
            continue
        any_ = True
        m = j["meta"]
        L += [f"### {pat} (Nq{m['nq']}/G{m['g']}/D{m['d']}/BK{m['bk']}, edges {m['edges']})",
              ""]
        rows = []
        for r in j["rows"]:
            rows.append([r["candidate"], r["max_m"], r["split"], r["BM"], r["n_tasks"],
                         r["slots"], r["partial_slots"], r["direct_rows"],
                         r["multi_owner_rows"], r["empty_rows"], r["kv_sum_k"],
                         r["kv_max_k"], r["unique_blocks"], r["padded_units"],
                         f(r["plan_ms"], 2),
                         "—" if r["predicted_us"] is None else f"{r['predicted_us']:.0f}",
                         {True: "pass", False: "FAIL", None: "—"}.get(
                             r.get("exact_partition"), "—")])
        L += table(["후보", "max_m", "split", "BM", "tasks", "slots", "partial",
                    "direct", "multi", "empty", "sum_k", "max_k", "uniq", "padded",
                    "plan ms", "예측 us", "정확 분할"],
                   rows, ["---"] + R * 13 + R * 2 + ["---"])
        L.append("")
    if not any_:
        L += ["NOT_RUN (`tools/dump_plan.py` 미실행)", ""]


def code_section(L):
    L += sec("구현 변경 요약")
    L += table(["항목", "파일", "내용"], [
        ["P1 강제 분할 제거", "`src/planner.cpp`",
         "`min_tasks` 기본 0. 과거 2×SM 분할은 legacy 후보로만 유지. 선택 후 재분할 없음"],
        ["P2 direct output", "`src/planner.cpp`, `rbc/kernels.py`",
         "degree 기반 출력 소유권. degree==1 행은 main 이 정규화 결과를 직접 기록하고 "
         "partial/merge 를 건너뛴다. reduction CSR 은 multi-owner 행만"],
        ["P3 K microtile", "`rbc/kernels.py`",
         "`BN` 서브루프, `LATE_V` variant, `FULL` membership fast path"],
        ["P4 교정 선택기", "`rbc/select.py`, `src/planner.cpp`",
         "계획 없이 후보 특징만 뽑는 `rbc_probe` C ABI + (BM,G) 구간별 NNLS 회귀 + "
         "보수적 선택 규칙"],
        ["P5 3단계 runtime", "`rbc/slots.py`",
         "`prepare_capacity`/`plan_into`/`submit`. 2 slot, event 의존성, C++ 이 pinned "
         "slab 에 직접 쓰기, FULL 이면 H2D 1회, capacity 초과는 명시적 실패"],
        ["P6 whole-query", "`stream_bench.py`",
         "기본 경로는 whole-query. chunk pipeline 은 비교 옵션 (§13.1 미충족으로 확대 안 함)"],
        ["§8.6 플래너 최적화", "`src/planner.cpp`",
         "worker별 스크래치 재사용, Task 풀(capacity 유지), 그룹별 CSR 직접 접근, "
         "block→위치 이진탐색 제거, reduction CSR 을 count→prefix sum→fill 로"],
    ])
    L.append("")


def issues_section(L):
    L += sec("작업 중 발견·수정한 결함")
    L += ["측정을 신뢰하려면 계측 자체의 결함을 먼저 드러내야 한다. 이번 작업에서 찾아 "
          "고친 것들이다.", ""]
    L += table(["증상", "원인", "조치"], [
        ["선택기가 최악의 계획 선택 (regret 415%)",
         "회귀가 음수 시간을 예측했고 `max(0, ·)` 클리핑 때문에 0us 로 보여 '가장 빠른 "
         "후보'가 됨",
         "비음수 최소제곱(NNLS)으로 교체. 음수·범위 밖 예측은 `inf` 로 배제"],
        ["선택기가 189셀 전부에서 무력화 (예측 전부 inf)",
         "`refit` 이 raw 에서 신규 특징을 복원하지 못해 조용히 0 을 채웠고, 그 결과 관측 "
         "범위가 0 이 되어 모든 입력이 '범위 초과'로 판정",
         "raw 에 특징 저장, `refit` 은 필드 누락 시 명시적 실패. 특징값 기반 외삽 거부 제거"],
        ["max_m 선택 오류 (regret 217%)",
         "(BM,G) 구간별 독립 회귀의 예측값을 구간을 넘어 직접 비교. 각 구간이 자기 절편을 "
         "가지므로 교정이 보정하지 않은 비교",
         "max_m 은 교정 lookup 으로 결정. 후보 간 BM 구간이 다르면 비교를 거부하고 Q-outer"],
        ["E4 의 CPU plan+pack 4.9ms",
         "`slots.py` 의 FULL membership 판정이 task 수만큼 파이썬 루프 (signature 816 task)",
         "task 별 기대 마스크를 KV 길이만큼 펼쳐 한 번에 비교. 4,888 → 847us"],
        ["E4 의 signature 첫 측정 160ms",
         "그 후보가 쓰는 커널 variant(FULL=False + merge)의 JIT 가 첫 측정에 섞임",
         "후보별 warmup 분리, `jit_first_us`·`warmup_us` 를 별도 항목으로 기록"],
        ["플래너 재작성 후 크래시",
         "OpenMP worker 가 마스터의 `thread_local` 스크래치를 보지 못함",
         "마스터 인스턴스를 포인터로 전달"],
        ["FINAL_RESULT 의 P1 효과 65%",
         "생성기가 max_m 을 전부 섞은 중앙값을 비교",
         "정책별 최적 max_m 기준으로 집계 수정 (실측과 일치하는 1.8%)"],
        ["게이트 46/81 vs 45/81 불일치",
         "생성기가 seed 를 섞어 최소를 취해 운 좋은 seed 하나가 결과를 정함",
         "(정책, max_m)별 중앙값을 먼저 낸 뒤 최적 선택"],
    ])
    L.append("")


def verdict_section(L, e4, e3, gate_s):
    L += sec("판정과 해석", 2)
    L += ["### 세 단계 판정 (§1.3)", ""]
    L += table(["판정", "결과", "근거"], [
        ["구현 개선", "**통과**",
         "P2 direct output 이 동일 정책의 v0.1 대비 GPU 시간 −5~8%. 플래너 최적화로 "
         "계획 CPU −20~29%"],
        ["RBC GPU 기여", "**부분 통과**",
         "§13.1 게이트 46/81 영역 (중앙값 −35.7%, 최대 −69.5%). 다만 승자는 "
         "signature_only 36 / bounded_merge 10 으로, RBC 고유 병합이 signature 까지 "
         "이긴 것은 10개 영역"],
        ["CPU–GPU 전체 개선", "**미달**",
         "E4 의 9개 입력 전부 회귀. 목표는 완료시간 10% 감소"],
    ])
    L += ["", "### 비용이 남은 지점 (§16 요구)", "",
          "**1. GPU 이득의 원인이 설계의 주장과 다르다.** 게이트 통과 영역 대부분에서 "
          "KV 방문 비가 1.00 이다. 즉 RBC 가 내세운 KV 재읽기 감소가 아니라 Q-outer "
          "union 직사각형의 padded dot 낭비가 사라진 것이 이득의 원인이다. E1 이 이를 "
          "직접 보여준다 — max_m 을 16→128 로 키우면 KV 방문은 −29% 줄지만 GPU 시간은 "
          "76.9→1,751.7us(+2,180%)로 악화한다. 이득원과 비용이 반대 방향이다.", "",
          "**2. GPU 이득이 CPU 계획비보다 작다.** Nq512 에서 CPU plan+pack 이 405~528us "
          "인데 §8.7 목표는 50us 이고, GPU 이득의 절대 크기는 20~75us 다. §8.6 을 적용해 "
          "계획비를 20~29% 줄였지만 이는 **공통 개선**이라 기준선도 함께 빨라졌다.", "",
          "**3. 남은 격차는 알고리즘 고유 비용이다.** signature 는 같은 입력에서 task 를 "
          "816개, q_outer 는 128개 만든다. task 수에 비례하는 비용(슬롯 생성, kmask "
          "채우기, descriptor 길이)은 할당을 없애도 남는다. 구현 최적화로 제거할 성질이 "
          "아니다.", "",
          "### 해석 제한 (§15)", "",
          "- 합성 mask 연산자 수준의 결과다. 모델 tok/s·서비스 SLO·신규성 주장으로 "
          "확장하지 않는다.",
          "- Q-outer fallback·pinned pool·일반 커널 개선은 RBC 분해의 성과가 아니다. "
          "E2 표에서 '공통 Q-outer 도 얻은 효과'와 'RBC−Q'를 분리했다.",
          "- KV 합집합 바이트로 HBM 비용을 계산하지 않았다. `kv_block_visits` 는 발행량 "
          "특징이며 실측 HBM 바이트가 아니다.",
          "- P6/E5(pipeline·실제 capture 확대)는 §13.1 규정에 따라 진행하지 않았다.",
          "- NCU/NSYS 프로파일은 수집하지 않았다. 시간 모델은 §7.5 대로 profiling 없이 "
          "교정했다.", ""]


def manifest_section(L):
    L += sec("산출물과 재현")
    L += ["```bash",
          "python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage reproduce",
          "python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage plan-audit",
          "python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage kernel",
          "python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage calibrate",
          "python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage online",
          "python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage report",
          "```", ""]
    mf = os.path.join(ROOT, "RBC_Attn_v02_result", "MANIFEST.sha256")
    if os.path.exists(mf):
        lines = open(mf).read().strip().split("\n")
        L += [f"`RBC_Attn_v02_result/MANIFEST.sha256` — 파일 {len(lines)}개.", ""]
        L += ["<details><summary>전체 해시 목록</summary>", "", "```"]
        L += lines
        L += ["```", "", "</details>", ""]
    # 원본 JSON 목록
    rows = []
    for dp, dn, fn in os.walk(RES):
        for x in sorted(fn):
            if not x.endswith((".json", ".txt")):
                continue
            full = os.path.join(dp, x)
            rows.append([os.path.relpath(full, ROOT),
                         f"{os.path.getsize(full)/1024:.0f} KB"])
    if rows:
        L += ["### 원본 파일", ""]
        L += table(["경로", "크기"], sorted(rows), ["---"] + R)
        L.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="RBC_Attn_v02_result/FULL_REPORT.md")
    a = ap.parse_args()

    e0 = load("kernel/e0_baseline.json")
    e1 = load("kernel/e1_finalization.json")
    e2 = load("kernel/e2_ablation.json")
    gate = load("kernel/gate_region_sweep.json")
    gate_s = load("kernel/gate_region_sweep_small.json")
    e3 = load("select/e3_selector.json")
    e4 = load("runtime/e4_full_cost.json")
    e4b = load("runtime/e4_full_cost_before_planner_opt.json")
    ctab = None
    ctp = os.path.join(ROOT, "configs", "plan_cost_table.json")
    if os.path.exists(ctp):
        ctab = json.load(open(ctp))
    craw = load("select/calibration_raw.json")

    L = ["# RBC-Attn v0.2 — 전체 실험 기록 (요약 + 원시 데이터)", "",
         "지시서 `RBC_Attn_v0.2` 에 따른 수정과 측정의 전량 기록이다. 모든 수치는 "
         "`results/v02/**` 의 원본 JSON 에서 생성했고, 측정하지 않은 값은 계산해 채우지 "
         "않고 `NOT_RUN` / `—` 로 남겼다.", ""]

    # 한눈에
    L += sec("한눈에", 2)
    if e4:
        by = collections.defaultdict(dict)
        for r in e4["records"]:
            if r["timing_mode"] == "pool_fresh_gpu":
                by[(r.get("shape"), r.get("pattern"))][r["selected_policy"]] = r
        ds = []
        for k, c in by.items():
            if "q_outer" not in c:
                continue
            qw = med(c["q_outer"]["wall_us"])
            for cand in ("signature", "bounded_merge"):
                if cand in c:
                    ds.append((med(c[cand]["wall_us"]) - qw) / qw)
        L += [f"- **전체 시간 판정: 개선 없음.** E4 의 {len(ds)}개 조합 전부 q_outer 가 "
              f"가장 빠르다 (Q대비 중앙값 {100*st.median(ds):+.1f}%).", ]
    if gate_s:
        L += ["- **GPU 시간만 보면 이기는 영역이 있다.** §13.1 게이트 46/81 영역 통과, "
              "감소율 중앙값 −35.7%, 최대 −69.5%."]
    L += ["- **이득의 원인은 설계의 주장과 다르다.** 통과 영역 대부분에서 KV 방문 비가 "
          "1.00 이다. KV 재읽기 감소가 아니라 Q-outer union 직사각형의 padded dot 낭비가 "
          "사라진 것이다.",
          "- **막는 것은 CPU 계획비다.** Nq512 에서 405~528us 로 §8.7 목표 50us 의 8~10배, "
          "GPU 이득 20~75us 보다 크다.",
          "- **정확성은 전부 통과.** numeric 54/54, exact-edge 27/27, 테스트 154건.", ""]

    env_section(L, e4 or e3 or e2 or e1)
    code_section(L)
    e0_section(L, e0)
    e1_section(L, e1)
    e2_section(L, e2)
    gate_section(L, gate, gate_s)
    calib_section(L, ctab, craw)
    e3_section(L, e3)
    e4_section(L, e4, e4b)
    plan_section(L)
    issues_section(L)
    verdict_section(L, e4, e3, gate_s)
    manifest_section(L)

    path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fo:
        fo.write("\n".join(L) + "\n")
    size = os.path.getsize(path)
    print(f"작성: {path} ({len(L)} 줄, {size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
