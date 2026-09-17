#!/usr/bin/env bash
# IDE_075 — 아직 v2 산출이 없는 세션을 모드별로 분석 (재실행 안전). 인자: 결과 루트(기본 IDE_075 qwen/OPT4)
cd ~/projects/vllm_hybrid; R=${1:-eval/results/IDE_075_20260917/qwen/OPT4}
for bd in $R/*/; do
  for sd in $bd*/; do
    [ -f $sd/metrics.json ] || continue; s=$(basename $sd); mode=$(python3 -c "import json;print(json.load(open('$sd/metrics.json'))['mode'])")
    if [ "$mode" = "CORR" ] && [ ! -f $sd/v2/dependency_summary_v2.json ]; then python3 eval/ide075/dependency_v2.py $sd --v2 > $sd/dependency_v2.log 2>&1; python3 eval/ide075/analyze_costs_v2.py $sd > $sd/costs_v2.log 2>&1; python3 eval/ide075/tp_collective.py $sd > $sd/tp_collective.log 2>&1; python3 eval/ide075/eager_layer_timeline.py $sd > $sd/eager.log 2>&1; fi
    if [ "$mode" = "RESOURCE" ] && [ ! -f $sd/v2/resource_summary.json ]; then python3 eval/ide075/resource_v2.py $sd > $sd/resource_v2.log 2>&1; python3 eval/ide075/tid_cpu_v2.py $sd > $sd/tid_cpu.log 2>&1; fi
    if [ "$mode" = "FOCUS" ] && [ ! -f $sd/v2/tail_offcpu_summary.json ]; then python3 eval/ide075/tid_cpu_v2.py $sd > $sd/tid_cpu.log 2>&1; python3 eval/ide075/focus_sched.py $sd > $sd/focus.log 2>&1; fi
    [ -f $sd/v2/validation_results_v2.json ] || python3 eval/ide075/validate_v2.py $sd > $sd/validate_v2.log 2>&1
  done
  [ -f $bd/expert_rows_summary.json ] || python3 eval/ide075/expert_samples.py $bd > $bd/expert_samples.log 2>&1
done
echo "$(date +%FT%T%z) analyze_all done ($R)" >> eval/results/IDE_075_20260917/state/chain.log
