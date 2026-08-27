# IDE_025 / TSK_045 — 단계별 작업

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | vLLM 이미지의 kv_offload/OffloadingConnector 기동 flag 확인 | ✅ (`CPUOffloadingSpec`+`cpu_bytes_to_use`) |
| 2 | 공유 prefix workload 스크립트 작성 | ✅ (`prefix_repetition` 32×8K) |
| 3 | 압박 구성 캘리브레이션 | ✅ (`--num-gpu-blocks-override 9600`) |
| 4 | 셀 측정 | ✅ 4 cells (`eval/results/20260827_120530_tsk045_kv_tier/`) |
| 5 | 비압박 대조군 회귀 측정 | ✅ (−1.31%, 경계) |
| 6 | TST_023 판정 | ✅ net win +51.8% / reload 91GB·280회 / 무회귀 경계 → 압박 워크로드 한정 ON 권고 |
