#!/usr/bin/env python3
"""v0.2 캠페인 드라이버 (지시서 §10.1, §10.3).

각 단계는 이미 구현된 도구를 호출하는 얇은 driver 다. kernel/planner 본체를 여기서
다시 구현하지 않는다.

  reproduce   E0  기준선 복구 (대표 세 입력, seed 42)
  plan-audit  E1  최종 계획 선택과 분할 + descriptor 덤프
  kernel      E2  P1/P2/P3 효과 분리 + 게이트 영역 판정
  calibrate   P4  시간 모델 교정 (교정 seed 0~4)
  online      E3  교정된 선택기 vs oracle (평가 seed 101~109)
  pipeline    E5  whole-query 대비 pipeline — §13.1 게이트 통과 시에만
  report          위 산출물을 모아 FINAL_RESULT.md 를 쓴다

`--stage report` 는 없는 값을 계산해 채우지 않고 NOT_RUN / UNSUPPORTED / FAILED 를 구분한다.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "v02")

# 단계별 (산출물 경로, 그 산출물을 만드는 명령)
STAGES = {
    "reproduce": [
        ("kernel/e0_baseline.json",
         "tools/e0_baseline.py --out results/v02/kernel/e0_baseline.json"),
    ],
    "plan-audit": [
        ("kernel/e1_finalization.json",
         "tools/e1_finalization.py --out results/v02/kernel/e1_finalization.json"),
        ("plan/dump_common_private.json",
         "tools/dump_plan.py --pattern common_private --verify "
         "--out results/v02/plan/dump_common_private.json"),
        ("plan/dump_random.json",
         "tools/dump_plan.py --pattern random --verify "
         "--out results/v02/plan/dump_random.json"),
        ("plan/dump_clustered.json",
         "tools/dump_plan.py --pattern clustered --verify "
         "--out results/v02/plan/dump_clustered.json"),
    ],
    "kernel": [
        ("kernel/e2_ablation.json",
         "tools/e2_ablation.py --out results/v02/kernel/e2_ablation.json"),
        ("kernel/gate_region_sweep.json",
         "tools/gate_region_sweep.py --nq 512 2048 8192 --sel 8 16 32 --seeds 101 "
         "--out results/v02/kernel/gate_region_sweep.json"),
        ("kernel/gate_region_sweep_small.json",
         "tools/gate_region_sweep.py --nq 128 256 512 --gs 4 8 16 --sel 8 16 32 "
         "--seeds 101 102 103 --out results/v02/kernel/gate_region_sweep_small.json"),
    ],
    "calibrate": [
        ("../../configs/plan_cost_table.json",
         "tools/calibrate_plan_cost.py --seeds 0 1 2 3 4 --nq 128 256 512 2048 "
         "--gs 4 8 --sel 8 16 32"),
    ],
    "online": [
        ("select/e3_selector.json",
         "tools/compare_v02.py --out results/v02/select/e3_selector.json"),
        ("runtime/e4_full_cost.json",
         "tools/e4_full_cost.py --out results/v02/runtime/e4_full_cost.json"),
    ],
    "pipeline": [
        ("pipeline/e5_pipeline.json", None),      # 게이트 판정 후에만
    ],
}


def run(cmd: str, *, dry=False) -> int:
    full = f"{sys.executable} {cmd}"
    print(f"\n$ {full}")
    if dry:
        return 0
    return subprocess.call(shlex.split(full), cwd=ROOT)


def exists(rel: str) -> bool:
    return os.path.exists(os.path.join(RES, rel))


def stage_status(name: str) -> dict:
    out = {}
    for rel, cmd in STAGES[name]:
        out[rel] = "PRESENT" if exists(rel) else ("NOT_RUN" if cmd else "UNSUPPORTED")
    return out


def do_report(args) -> int:
    """산출물을 모아 요약을 쓴다. 없는 값은 계산하지 않고 상태로 남긴다."""
    summary = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), stages={})
    for name in STAGES:
        summary["stages"][name] = stage_status(name)
    path = os.path.join(RES, "campaign_status.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(summary, open(path, "w"), indent=1, ensure_ascii=False)
    print("\n단계별 산출물 상태")
    for name, st in summary["stages"].items():
        print(f"  {name}")
        for rel, s in st.items():
            print(f"    {s:12} {rel}")
    print(f"\n저장: {path}")
    print("FINAL_RESULT.md 는 tools/make_final_result.py 가 이 상태와 원본 JSON 으로 만든다.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rbc_v02.yaml")
    ap.add_argument("--stage", required=True,
                    choices=list(STAGES) + ["report", "status"])
    ap.add_argument("--force", action="store_true", help="이미 있는 산출물도 다시 만든다")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = os.path.join(ROOT, args.config)
    if not os.path.exists(cfg):
        print(f"설정이 없다: {cfg}", file=sys.stderr)
        return 2
    print(f"설정 {args.config} / 단계 {args.stage}")

    if args.stage == "status":
        return do_report(args)
    if args.stage == "report":
        return do_report(args)

    # 선행 산출물 검사
    order = list(STAGES)
    idx = order.index(args.stage)
    for prev in order[:idx]:
        st = stage_status(prev)
        missing = [k for k, v in st.items() if v == "NOT_RUN"]
        if missing and prev != "pipeline":
            print(f"주의: 선행 단계 '{prev}' 의 산출물이 없다: {missing}")

    if args.stage == "pipeline":
        gate = os.path.join(RES, "kernel", "gate_region_sweep_small.json")
        if not os.path.exists(gate):
            print("게이트 판정 산출물이 없다. --stage kernel 을 먼저 실행한다.")
            return 3
        print("게이트 통과 영역에서만 pipeline 을 실행한다 (§13.1). "
              "현재 구현은 whole-query 기본이며 chunk pipeline 은 비교 옵션이다.")
        return run("stream_bench.py --policy signature_only --chunk-q 256 "
                   "--out results/v02/pipeline/e5_pipeline.json", dry=args.dry_run)

    rc = 0
    for rel, cmd in STAGES[args.stage]:
        if cmd is None:
            print(f"  {rel}: UNSUPPORTED (이 단계에서 만들지 않는다)")
            continue
        if exists(rel) and not args.force:
            print(f"  {rel}: PRESENT — 건너뛴다 (--force 로 재실행)")
            continue
        r = run(cmd, dry=args.dry_run)
        if r != 0:
            print(f"  FAILED rc={r}: {cmd}")
            rc = r
    return rc


if __name__ == "__main__":
    sys.exit(main())
