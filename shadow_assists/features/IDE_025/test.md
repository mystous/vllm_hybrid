# IDE_025 / TST_023 — 검증

| 항목 | 기준 |
|---|---|
| net win | 압박 구성에서 (c) 가 (b) 대비 TTFT p95 개선 또는 throughput 개선 (재계산 회피 실증) |
| 무회귀 | 비압박 구성에서 (c) 회귀 ≤1% |
| 동작 실증 | offload hit/miss 카운터 (connector 로그) 로 DRAM reload 발생 확인 — "카운터 0 이면 이득 주장 금지" (IDE_006 교훈) |
