#!/usr/bin/env bash
# /proc/stat 기반 전체 CPU busy% 1Hz 샘플러 (mpstat 부재 환경). usage: cpu_sample.sh <outfile>
out=$1
prev_idle=0; prev_total=0
while true; do
  read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
  idle_all=$((idle + iowait)); total=$((user + nice + system + idle + iowait + irq + softirq + steal))
  if [ "$prev_total" -ne 0 ]; then
    dt=$((total - prev_total)); di=$((idle_all - prev_idle))
    [ "$dt" -gt 0 ] && echo "$(date +%s) busy=$(awk "BEGIN{printf \"%.2f\", 100*(1-$di/$dt)}")" >> "$out"
  fi
  prev_idle=$idle_all; prev_total=$total
  sleep 2
done
