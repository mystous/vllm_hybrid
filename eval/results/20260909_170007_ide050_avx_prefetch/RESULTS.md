# IDE_050 — AVX rb 커널 B 스트림 소프트웨어 prefetch (`KT_AVX_PF`) — **채택 (기본 2)**

`avx_rb_rows` 의 k_begin 루프에서 2 KB B 블록 PF개 앞을 `_mm_prefetch(T0)` (32 라인). 정수 산술 불변 → bit 동일 (4개 형상 max abs diff 0).

## 1행 expert (n_cold 32, vec 경로, µs/expert, 각 2회)
| 구성 | 1회 | 2회 |
|---|---|---|
| rb (prefetch 없음) | 69.7 | 70.6 |
| rb + PF 2 | 66.0 | 65.7 |
| rb + PF 4 | 66.0 | 65.6 |
| rb + PF 8 | 68.1 | 65.9 |
→ **−6%** (사전 등록 −7% 경계). 누적 (원본 75.3 → 65.7): 1행 스트리밍 **−13%**, 소켓당 ≈ 190 GB/s (torch 읽기 195, DDR 이론 282 의 67%).

서빙 효과는 final sweep 에서 통합 측정 (`KT_AVX_RB=1 KT_AVX_PF=2`).
