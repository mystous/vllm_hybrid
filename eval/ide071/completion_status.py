#!/usr/bin/env python3
"""IDE_071 — COMPLETION_STATUS.md (등록 항목별 종료 상태, 실행량, 미실행·OUT_OF_SCOPE 목록). 평가 없음."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
CAMP = f"{REPO}/eval/results/IDE_071_20260916"; CD = f"{CAMP}/compact"

def main():
    cs = json.load(open(f"{CD}/compact_state.json")); st = json.load(open(f"{CAMP}/state/state.json"))
    L = [f"# COMPLETION_STATUS — IDE_071 ({now()['wall_kst']})", "", "## compact (cpu_offload_no_02_compact) — 등록 항목별 종료 상태", "", "| 항목 | 셀 | 종료 상태 | 반복(valid/total) | 비고 |", "|---|---|---|---|---|"]
    reg = [("대조군 B3 (재사용)", "P2_cf1_skip1_pin1_def8", "REUSED", "3/3", "12단계 P2 셀, 같은 워크로드"), ("대조군 B4 (재사용)", "R04_d4_epoch", "REUSED", "3/3 (SHORT_COLD)", "12단계 P1 셀")]
    for cid, rec in cs["cells"].items():
        a = rec["attempts"][-1]; reps = a.get("reps") or []
        reg.append((cid.split("_")[0] if not cid.startswith("S2_b3") else ("확인 대상" if "confirm" in cid else ("연속 부하" if "load" in cid else "S2")), cid, a["exit_status"] + (f" (attempt {a['attempt']})" if a["attempt"] != "a1" else ""), f"{sum(1 for r in reps if r['valid'])}/{len(reps)}", ""))
    reg.append(("S6", "S6_b4_combo", "NOT_TRIGGERED", "—", "B4 계열 조건 충족 변경 0개 (selection_trace)"))
    reg.append(("확인 대상 B4 계열", "—", "NOT_TRIGGERED", "—", "확인 후보 없음 → B4 재확인·확인 3회 생략 (규칙 §4.1 '새 후보 없는 계열은 생략 가능')"))
    reg.append(("선택적 진단 1회", "—", "NOT_RUN", "—", "선택 항목, 미실행 (예산 잔여)"))
    for r in reg: L.append("| " + " | ".join(r) + " |")
    b = cs["budget"]
    L += ["", f"## 실행량 (compact)", "", f"- 일반 벤치 {b['bench']}/{b['bench_max']} (1차 5 + 부모 재확인 1 + 확인 3 + 별도 입력 1) · 연속 부하 {b['load']} (시도 {b['retry']+b['load']}) · 진단 {b['diag']} · 재시도 {b['retry']}/2 · GSM40 {sum(1 for f in glob.glob(CD + '/*/a*/gsm40.json'))}/2",
          f"- 부팅 {b['boot']}/14 · 워밍업 = 성공 부팅당 1회 (C16·32요청) · greedy4 smoke = 성공 부팅당 1세트", f"- 종료 상태: **{cs['status']}** (탐색 범위 내 등록 항목 처리 완료; 전역 최적·통계적 동등성 판단 아님)", ""]
    # 12단계 부분
    c = {"COMPLETED": 0, "FAILED": 0, "OTHER": 0}
    for cid, rec in st["cells"].items():
        ex = rec["attempts"][-1]["exit_status"]; c["COMPLETED" if ex == "COMPLETED" else ("FAILED" if ex.startswith("FAILED") else "OTHER")] += 1
    unrun = 0
    for ph in ("P3", "P4", "P5", "P6", "P7"):
        p = f"{CAMP}/manifests/cells_{ph}.jsonl"
        if os.path.exists(p): unrun += sum(1 for l in open(p) if json.loads(l)["cell_id"] not in st["cells"])
    L += ["## 12단계 계획 (cpu_offload_no_02) — 07:05 에 compact 로 대체", "", f"- 실행 셀 {len(st['cells'])}: COMPLETED {c['COMPLETED']}, FAILED {c['FAILED']}, 기타 {c['OTHER']} (P1 8, P2 20, P3 20/35, 앵커 재측정 포함)", f"- 등록 후 미실행 (OUT_OF_SCOPE): {unrun} 셀 (P3 잔여, P4~P7); P8~P11 은 manifest 미생성", "- 상세: FULL_REPORT.md §3.2~§3.3", ""]
    open(f"{FEAT}/COMPLETION_STATUS.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
