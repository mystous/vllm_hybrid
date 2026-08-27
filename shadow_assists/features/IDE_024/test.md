# IDE_024 / TST_022 — 검증

| 항목 | 기준 |
|---|---|
| 무손실 | t2 (격리 BG) 의 GPU serving tps ≥ t1 × 0.99 |
| CPU 활용 | t2 의 CPU busy avg ≥ 50% |
| 대조 | t3 (미격리) 가 t2 보다 유의하게 나쁨 → 격리가 driver 임을 입증 |
| BG 산출 | BG 자체의 처리량 (hash/s 등) 기록 — 합산 가치 정량 |
