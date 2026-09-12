#!/usr/bin/env python3
"""FINAL_RESULT.md 를 원본 JSON 에서만 생성한다 (지시서 §14.1).

없는 값은 계산해 채우지 않고 NOT_RUN / UNSUPPORTED / FAILED 로 구분한다.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
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
    return float(np.median(xs)) if xs else None


def fmt(x, unit="", nd=1):
    return "NOT_RUN" if x is None else f"{x:.{nd}f}{unit}"


def boot_ci(a, b, n=10000, seed=0):
    """paired ratio (b/a) 의 95% bootstrap 신뢰구간."""
    if not a or not b or len(a) != len(b):
        return None
    rng = np.random.default_rng(seed)
    pa, pb = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(pa), size=(n, len(pa)))
    r = pb[idx].mean(axis=1) / pa[idx].mean(axis=1)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def section_first_table(e4, gate_small, e3) -> list:
    """§14.1 첫 표. pool_fresh_gpu 기준."""
    out = ["| 입력 | mask 출발점 | 비교 모드 | 최강 기준선/시간 | RBC 선택/시간 | 감소율 "
           "| non-Q RBC 비율 | 정확성 | 판정 |",
           "|---|---|---|---:|---:|---:|---:|---|---|"]
    if e4 is None:
        out.append("| (E4 미수행) | — | pool-fresh | NOT_RUN | NOT_RUN | NOT_RUN "
                   "| NOT_RUN | NOT_RUN | NOT_RUN |")
        return out
    # (shape, pattern) 별로 pool_fresh_gpu 레코드를 모은다
    by = collections.defaultdict(dict)
    for r in e4["records"]:
        if r["timing_mode"] != "pool_fresh_gpu":
            continue
        by[(r.get("shape"), r.get("pattern"))][r["selected_policy"]] = r
    active = None
    if e3 is not None and e3["cells"]:
        n = sum(1 for c in e3["cells"] if c["rbc_plan_used"])
        active = f"{n}/{len(e3['cells'])} ({n/len(e3['cells']):.0%})"
    for key in sorted(by, key=lambda x: (str(x[0]), str(x[1]))):
        shape, pattern = key
        cands = by[key]
        q = cands.get("q_outer")
        if q is None:
            continue
        qw = med(q["wall_us"])
        nonq = [c for name, c in cands.items() if name != "q_outer"]
        best = min(nonq, key=lambda c: med(c["wall_us"]) or float("inf")) if nonq else None
        bw = med(best["wall_us"]) if best else None
        red = None if (bw is None or not qw) else (1 - bw / qw)
        acc = []
        for c in cands.values():
            if c.get("numeric_passed") is True:
                acc.append("pass")
            elif c.get("numeric_passed") is False:
                acc.append("FAIL")
        accs = "pass" if acc and all(x == "pass" for x in acc) else (
            "FAIL" if "FAIL" in acc else "NOT_CHECKED")
        verdict = "NOT_RUN"
        if red is not None:
            verdict = "10% 목표 통과" if red >= 0.10 else (
                "개선 있음(10% 미달)" if red > 0 else "개선 없음")
        out.append(f"| {pattern} {shape} | GPU | pool-fresh | q_outer / {fmt(qw,'us')} "
                   f"| {best['selected_policy'] if best else 'NOT_RUN'} / {fmt(bw,'us')} "
                   f"| {'NOT_RUN' if red is None else f'{100*red:+.1f}%'} "
                   f"| {active or 'NOT_RUN'} | {accs} | {verdict} |")
    out.append("| 실제 capture | 실제 위치 | 동일 경계 | NOT_RUN | — | — | — | — | "
               "NOT_RUN (capture 없음) |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="RBC_Attn_v02_result/FINAL_RESULT.md")
    a = ap.parse_args()

    e0 = load("kernel/e0_baseline.json")
    e1 = load("kernel/e1_finalization.json")
    e2 = load("kernel/e2_ablation.json")
    gate = load("kernel/gate_region_sweep.json")
    gate_s = load("kernel/gate_region_sweep_small.json")
    e3 = load("select/e3_selector.json")
    e4 = load("runtime/e4_full_cost.json")

    L = ["# RBC-Attn v0.2 — 최종 결과", "",
         "이 문서의 모든 수치는 `results/v02/**` 의 원본 JSON 에서 생성됐다. "
         "측정하지 않은 값은 `NOT_RUN` 으로 남기고 계산해 채우지 않았다.", ""]

    # 환경
    src = e4 or e3 or e2 or e1
    if src:
        m = src["meta"]
        L += ["## 측정 환경", "",
              f"- GPU: {m.get('gpu')} (UUID {m.get('gpu_uuid')}), SM {m.get('sm')}",
              f"- 사용 장치 수 {m.get('device_count', 'NOT_RUN')}, 논리 ordinal "
              f"{m.get('logical_ordinal', 0)}",
              f"- 버전: {m.get('versions')}",
              f"- 생성 시각 기준 원본: {m.get('time')}", ""]

    L += ["## 1. 최종 판정 표 (§14.1)", "",
          "감소율은 §1.3 의 `gain = 1 - T_new/T_best_baseline` 이다. 음수는 기준선보다 "
          "**느려졌다**는 뜻이다.", ""]
    L += section_first_table(e4, gate_s, e3)
    L.append("")

    # 게이트
    L += ["## 2. §13.1 GPU 단계 게이트", ""]
    for tag, gj in (("Nq512~8192 / G8", gate), ("Nq128~512 / G4·8·16", gate_s)):
        if gj is None:
            L.append(f"- {tag}: NOT_RUN")
            continue
        cells = gj["cells"]
        keys = sorted({(c["nq"], c.get("g", gj["meta"].get("g")), c["sel"], c["pattern"])
                       for c in cells})
        hits = []
        for key in keys:
            sub = [c for c in cells
                   if (c["nq"], c.get("g", gj["meta"].get("g")), c["sel"],
                       c["pattern"]) == key]
            # seed 반복이 있으면 (정책, max_m) 별 중앙값을 먼저 낸 뒤 최적을 고른다.
            # seed 를 섞어 최소를 취하면 운 좋은 seed 하나가 결과를 정한다.
            def best_of(pred):
                per = {}
                for c in sub:
                    if pred(c):
                        per.setdefault((c["policy"], c["max_m"]), []).append(c["gpu_us"])
                return min((med(v) for v in per.values()), default=None)
            q = best_of(lambda c: c["policy"] == "q_outer")
            n = best_of(lambda c: c["policy"] != "q_outer" and not c.get("is_q_fallback"))
            if q is None or n is None:
                continue
            d = (n - q) / q
            if d <= -0.05:
                hits.append((key, d))
        L.append(f"- {tag}: {len(hits)}/{len(keys)} 영역 통과"
                 + (f", 감소율 중앙값 {100*st.median([h[1] for h in hits]):.1f}%, "
                    f"최대 {100*min(h[1] for h in hits):.1f}%" if hits else ""))
    L.append("")

    # 선택기
    L += ["## 3. P4 선택기 품질 (§7.7 진단 기준)", ""]
    if e3 is None:
        L.append("NOT_RUN")
    else:
        reg = [c["regret"] for c in e3["cells"] if c["regret"] is not None]
        act = sum(1 for c in e3["cells"] if c["rbc_plan_used"])
        cpu = [c["selector_cpu_us"] for c in e3["cells"]]
        L += [f"- 교정 seed {e3['meta'].get('calibration_seeds')} / 평가 seed "
              f"{e3['meta']['seeds']}",
              f"- regret 중앙값 {100*st.median(reg):.2f}% (기준 3%), "
              f"최악 {100*max(reg):.2f}% (기준 5%)",
              f"- `RBC_active_fraction` = {act}/{len(e3['cells'])} "
              f"= {act/len(e3['cells']):.3f}",
              f"- 선택기 CPU 중앙값 {st.median(cpu):.0f}us (probe + 선택된 계획 생성)"]
    L.append("")

    # P1~P6 효과 분해
    L += ["## 4. 변경별 효과 분해 (§14.1 두 번째 표)", "",
          "GPU 실행시간의 상대 변화다. **음수가 시간 감소(개선)**, 양수가 악화다. "
          "각 행은 같은 탐색 예산 안에서 정책별 최적 max_m 을 고른 뒤 패턴별 값의 "
          "중앙값이다. 마지막 열은 같은 실행기에서 RBC 계획이 Q-outer 계획보다 "
          "얼마나 느린지이며, 양수면 RBC 분해가 손해라는 뜻이다.", "",
          "| 변경 | 단독 효과 | 누적 효과 | 공통 Q-outer 도 얻은 효과 | RBC−Q (양수=RBC 손해) |",
          "|---|---|---|---|---|"]
    if e2 is None:
        L.append("| (E2 미수행) | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |")
    else:
        cells = e2["cells"]

        def best(row, pol):
            """행·정책별로 **같은 탐색 예산 안의 최적 max_m** 을 고른 뒤 그 중앙값.

            max_m 을 섞어 중앙값을 내면 행마다 분포가 달라 비교가 성립하지 않는다.
            패턴이 여럿이면 패턴별 최적끼리의 중앙값을 쓴다.
            """
            pats = sorted({c["pattern"] for c in cells})
            vals = []
            for pat in pats:
                per_m = {}
                for c in cells:
                    if c["row"] == row and c["policy"] == pol and c["pattern"] == pat:
                        per_m.setdefault(c["max_m"], []).append(c["gpu_us"])
                if per_m:
                    vals.append(min(med(v) for v in per_m.values()))
            return med(vals)

        rows = [("P1 (강제 분할 제거)", "A_v01", "B_P1"),
                ("P2 (direct output)", "B_P1", "C_P2"),
                ("P3 (BN=64)", "C_P2", "D_P3a"),
                ("P3 (late V)", "C_P2", "D_P3b"),
                ("P3 (FULL fast path)", "C_P2", "D_P3c")]
        a0 = best("A_v01", "q_outer")
        for label, prev, cur in rows:
            dq_prev, dq_cur = best(prev, "q_outer"), best(cur, "q_outer")
            dr_prev, dr_cur = best(prev, "rbc"), best(cur, "rbc")
            solo = None if not (dq_prev and dq_cur) else 100 * (dq_cur - dq_prev) / dq_prev
            cum = None if not (a0 and dq_cur) else 100 * (dq_cur - a0) / a0
            extra = None
            if dq_cur and dr_cur:
                extra = 100 * (dr_cur - dq_cur) / dq_cur
            L.append(f"| {label} | {fmt(solo,'%',1)} | {fmt(cum,'%',1)} | "
                     f"{fmt(solo,'%',1)} (Q-outer 기준) | {fmt(extra,'%',1)} (RBC−Q) |")
        L += ["| P4 (교정 선택기) | 아래 §3 | — | — | — |",
              "| P5 (arena·단일 전송) | 아래 §5 | — | — | — |",
              "| P6 (whole-query 기본) | 기본값 변경 (pipeline 은 비교 옵션) | — | — | — |"]
    L += ["", "NCU 바이트와 일반 실행시간의 측정 조건은 다르다 "
          "(NCU 는 profiling 활성, 실행시간은 profiling 없음). 두 값을 같은 조건의 "
          "인과로 연결하지 않았다.", ""]

    # P5
    L += ["## 5. P5 runtime 비용 분해 (§8.7)", ""]
    if e4 is None:
        L.append("NOT_RUN")
    else:
        L += ["| 입력 | 후보 | cpu plan+pack | DMA | GPU | wall (fresh_gpu) | "
              "descriptor live/capacity | H2D | hot pin/alloc |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in e4["records"]:
            if r["timing_mode"] != "pool_fresh_gpu":
                continue
            L.append(f"| {r.get('pattern')} {r.get('shape')} | {r['selected_policy']} "
                     f"| {fmt(med(r['cpu_plan_pack_us']),'us',0)} "
                     f"| {fmt(med(r['dma_activity_us']),'us',0)} "
                     f"| {fmt(med(r['gpu_total_us']),'us',0)} "
                     f"| {fmt(med(r['wall_us']),'us',0)} "
                     f"| {r['descriptor_live_bytes']}/{r['descriptor_capacity_bytes']}B "
                     f"| {r['h2d_bytes']}B x{r['h2d_copy_count']} "
                     f"| {r['host_pin_calls_hotpath']}/"
                     f"{r['device_allocation_requests_hotpath']} |")
    L.append("")

    # 금지 사항 관련 자기 점검
    L += ["## 6. 해석 제한", "",
          "- 이 결과는 합성 mask 연산자 수준이다. 모델 tok/s·서비스 SLO·신규성 주장으로 "
          "확장하지 않는다.",
          "- Q-outer fallback·pinned pool·일반 kernel 개선은 RBC 분해의 성과가 아니다. "
          "§4 표에서 '공통 Q-outer 도 얻은 효과' 와 'RBC 추가 효과' 를 분리했다.",
          "- KV 합집합 바이트로 HBM 비용을 계산하지 않았다. 방문량(`kv_block_visits`) 은 "
          "발행량 특징이며 실측 HBM 바이트가 아니다.",
          "- pipeline 결과가 있어도 chunk serial 대비 개선을 whole-query 개선으로 쓰지 "
          "않는다.", ""]

    path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"작성: {path} ({len(L)} 줄)")


if __name__ == "__main__":
    main()
