#!/usr/bin/env python3
"""EXP-E11 3단계 판정: 사전등록 예측 (predictions.json) 과 전수측정 결과를 대조해
예측 오차 / 순위 일치 / 정책별 selection_regret 을 계산한다.

usage: analyze_e11.py <predictions.json> <oracle 결과 디렉토리>
"""
import json, re, sys, os, glob, itertools

pred_path, B = sys.argv[1], sys.argv[2]
P = json.load(open(pred_path))
NUM = r"([0-9]+\.?[0-9]*)"


def metrics(f):
    t = open(f, errors="ignore").read()
    g = lambda p: (float(m.group(1)) if (m := re.search(p, t)) else None)
    return dict(tok=g(r"Output token throughput \(tok/s\):\s*" + NUM),
                tpot=g(r"Mean TPOT \(ms\):\s*" + NUM),
                tpot99=g(r"P99 TPOT \(ms\):\s*" + NUM),
                ttft=g(r"Median TTFT \(ms\):\s*" + NUM),
                ttft99=g(r"P99 TTFT \(ms\):\s*" + NUM),
                dur=g(r"Benchmark duration \(s\):\s*" + NUM),
                ok=g(r"Successful requests:\s*" + NUM))


meas = {}   # (wl, cand) -> metrics
for f in sorted(glob.glob(os.path.join(B, "o_*.log"))):
    m = re.match(r"o_(.+)_(WH\d)\.log", os.path.basename(f))
    if not m:
        continue
    d = metrics(f)
    if d["tok"]:
        meas[(m.group(2), m.group(1))] = d

SLO_TPOT99 = P["slo"]["tpot_p99_ms"]

print(f"# EXP-E11 3단계 판정\n\n사전등록 예측: `{os.path.basename(pred_path)}`  측정: `{os.path.basename(B)}`")
print(f"SLO: TPOT p99 <= {SLO_TPOT99:.0f} ms\n")

summary = {}
for wl in P["workloads"]:
    wid = wl["id"]
    rows = {r["cand"]: r for r in P["predictions"][wid]}
    have = [(c, meas[(wid, c)]) for c in rows if (wid, c) in meas]
    if not have:
        print(f"## {wid} — 측정 없음\n")
        continue
    print(f"## {wid} — random {wl['Lin']}/{wl['Lout']}, 동시성 {wl['C']}, {wl['np']}요청\n")
    print("| 후보 | 예측 tok/s | 실측 tok/s | 오차 | 실측 TPOT p99 | SLO | 예측 순위 | 실측 순위 |")
    print("|---|---|---|---|---|---|---|---|")
    pr = sorted(rows, key=lambda c: -rows[c]["tok_s"])
    mr = sorted([c for c, _ in have], key=lambda c: -meas[(wid, c)]["tok"])
    for c, d in sorted(have, key=lambda t: -t[1]["tok"]):
        p = rows[c]["tok_s"]; a = d["tok"]
        slo = "통과" if (d["tpot99"] or 1e9) <= SLO_TPOT99 else "위반"
        print(f"| {c} | {p:.1f} | **{a:.1f}** | {100*(p-a)/a:+.1f}% | {d['tpot99']:.1f} | {slo} | "
              f"{pr.index(c)+1} | {mr.index(c)+1} |")
    errs = sorted(abs(100 * (rows[c]["tok_s"] - d["tok"]) / d["tok"]) for c, d in have)
    n = len(errs)
    print(f"\n예측 절대오차 중앙값 {errs[n//2]:.1f}% / 최대 {errs[-1]:.1f}% ({n} 후보)")

    # 순위 일치: 실측 차이가 10% 이상인 쌍에서 예측 순서가 맞는 비율
    pairs = [(a, b) for a, b in itertools.combinations([c for c, _ in have], 2)
             if abs(meas[(wid, a)]["tok"] - meas[(wid, b)]["tok"]) / max(meas[(wid, a)]["tok"], meas[(wid, b)]["tok"]) >= 0.10]
    agree = sum(1 for a, b in pairs
                if (rows[a]["tok_s"] > rows[b]["tok_s"]) == (meas[(wid, a)]["tok"] > meas[(wid, b)]["tok"]))
    print(f"순위 일치 (실측 차이 >=10% 인 쌍 {len(pairs)}개): {100*agree/len(pairs):.0f}%" if pairs else "순위 비교 가능한 쌍 없음")

    # oracle 과 정책별 regret
    feas = [(c, d) for c, d in have if (d["tpot99"] or 1e9) <= SLO_TPOT99]
    pool = feas if feas else have
    oc, od = max(pool, key=lambda t: t[1]["tok"])
    print(f"\n전수탐색 oracle: **{oc}** = {od['tok']:.1f} tok/s"
          f"{' (SLO 통과 후보 중)' if feas else ' (SLO 통과 후보 없음 -> 전체 중)'}")
    print("\n| 정책 | 선택 | 실측 tok/s | regret | 측정 예산(부팅 수) |")
    print("|---|---|---|---|---|")
    pol = P["policy_choice"][wid]
    budgets = dict(cost_model=0, fixed_epochb=0, memory_only=0)
    for name, key in (("비용모델 선택", "cost_model"), ("고정 EPOCH-B", "fixed_epochb"),
                      ("memory-only", "memory_only"), ("단일요인 탐색", "single_factor"),
                      ("무작위 탐색(예산6)", "random_budget6")):
        v = pol.get(key)
        cand = v["choice"] if isinstance(v, dict) else v
        budget = len(v["probed"]) if isinstance(v, dict) else budgets.get(key, 0)
        if cand is None or (wid, cand) not in meas:
            print(f"| {name} | {cand} | 측정 없음 | — | {budget} |")
            continue
        a = meas[(wid, cand)]["tok"]
        print(f"| {name} | {cand} | {a:.1f} | **{100*(od['tok']-a)/od['tok']:.1f}%** | {budget} |")
        if key == "cost_model":
            summary[wid] = 100 * (od["tok"] - a) / od["tok"]
    print()

if summary:
    v = sorted(summary.values())
    print("## 종합\n")
    print(f"비용모델 선택의 regret: " + ", ".join(f"{k} {x:.1f}%" for k, x in summary.items()))
    print(f"중앙값 {v[len(v)//2]:.1f}% / 최악 {v[-1]:.1f}%")
    print(f"\n목표 (§17.6): 중앙값 <=5%, 최악 <=10% -> "
          f"{'통과' if v[len(v)//2] <= 5 and v[-1] <= 10 else '미달'}")
