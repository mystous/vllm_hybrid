#!/usr/bin/env bash
# 다음 셀 완료까지 대기(최대 30분). 완료시 EVENT=CELL_DONE+상세, 30분 무변화시 EVENT=HEARTBEAT.
OUT=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/routing_llmd_20260603
RE=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval
PY=/workspace/vllm_dev_prj/bin/python
before=$(ls $OUT/summ_*.json 2>/dev/null|sort)
for i in $(seq 1 120); do
  now=$(ls $OUT/summ_*.json 2>/dev/null|sort)
  new=$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$now"))
  if [ -n "$new" ]; then
    echo "EVENT=CELL_DONE"
    "$PY" "$RE/cell_report.py" $new
    echo "누적 셀: $(printf '%s\n' "$now"|grep -c .)"
    exit 0
  fi
  sleep 15
done
echo "EVENT=HEARTBEAT"
echo "30분 무변화 — 누적 $(ls $OUT/summ_*.json 2>/dev/null|wc -l) 셀:"
ls $OUT/summ_*.json 2>/dev/null | sed 's#.*/summ_##;s#.json##' | tr '\n' ' '; echo
