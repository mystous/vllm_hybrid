# TSK_043 — AGSD: CPU 병렬성 최적화 + 측정 RESULTS

> **status**: ⚪ 대기 (구현 전) — 측정 후 채움
> **plan**: [`../../IDE_006/TSK_020/planning/TSK_043_cpu_classifier.md`](../../IDE_006/TSK_020/planning/TSK_043_cpu_classifier.md)

## 산출 예정
- 분류기 C0~C3 구현 + self-test 통과
- **CPU 병렬성 ablation 표** (§8 R1~R6 lever ON/OFF: classify req/s·p99 + AGSD routing e2e throughput)
- decision-regret 표 (corpus × classifier × {mean/p99 regret, zero율, catastrophic율})
- classify latency 표 (p50/p99, RE2·mimalloc on-off)
- §4 accept/kill 판정 (C0 mean regret>5% → 진입정당 / method spread<5% 또는 C2·C3 무개선 → kill)

(측정 미실시 — TSK_042 oracle_table 합류 필요)
