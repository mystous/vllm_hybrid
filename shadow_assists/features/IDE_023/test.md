# IDE_023 / TST_021 — 검증

| 항목 | 기준 | 방법 |
|---|---|---|
| regime 실증 | R1-0528 FP8 이 GPU-only (TP=8, gmu 0.95) 로 **OOM** | vLLM/SGLang GPU-only 기동 시도 로그 |
| 서빙 성립 | hybrid 로 기동 + 정상 completion 출력 | curl smoke + 출력 육안 검증 |
| 성능 | prefill tok/s, decode tok/s (C=1/8/32) | bench serve (STD profile 변형) |
| CPU 활용 | decode 구간 CPU busy ≥50% 구간 존재 | eval/cpu 모니터 (pidstat 계열) |
| 참조점 | 8×L20+Xeon 227 tok/s (KT 공식) 대비 오더 일치 | 비교표 |
