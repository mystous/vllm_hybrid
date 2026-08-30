
## CUDA graph 트랙 (2026-08-30 오후 — 병목=동기화 대기 47% 프로파일 판정 후)

| 셀 | 구성 | out tok/s @C32 | 판정 |
|---|---|---:|---|
| G1 | hot64+spec + full decode graph | 부하에서 segfault | kt×full-graph 비호환 실체 확인 |
| G2 | 〃 + tc_piecewise | 부하에서 죽음 (spec verify 경로가 full-graph 강제) | spec×graph 결합은 미해결 과제 |
| **G3** | **hot64 + tc_piecewise (spec 없음)** | **226.4 / 232.5 (재현, 편차 2.7%)** | **신기록 — 기준 56.3 대비 +307%** |

- G3 품질: GSM8K 40문항 **95.0%** (기준 85.0%) — 게이트 통과
- 성립 조건 수리 2건 추가: deep_gemm cu12/cu13 불일치 (DSA 모듈 import 가드), spec 없는 순수 decode 경로 사용
- 해석: 프로파일이 지목한 "rank 간 동기화 대기 47%"를 부분-graph 가 제거. 초안 검증(+20%)보다 큰 이득이라 spec 은 현재 구성에서 제외 (graph×spec 결합은 향후 과제)

### 최신 확정 구성 (2026-08-30 16시 기준)

(96,2) + hot-64 (빈도 맵+mask) + `--cuda-graph-backend-decode tc_piecewise` = **229 tok/s @C32, GSM 95.0%** — 기준 대비 **+307%**
