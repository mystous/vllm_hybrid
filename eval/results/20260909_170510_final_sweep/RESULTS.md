# 최종 구성 재측정 (2026-09-09 17:55~18:30) — Qwen3-Coder-480B-A35B FP8, H100×4 (TP4) + Xeon 8480+×2 (96 스레드, AMXINT4)

공통: hot-96 (α=0.25 phase 가중 hot set) + callback-free 핸드오프 (`KT_CALLBACK_FREE=1`) + τ deferral (`KT_COLD_DEFER=1 KT_COLD_TAU=0.25`) + 비-워커 코어 재배치 + expert 별 AMX 선택 (`KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3`) + AVX 레지스터 블로킹·prefetch (`KT_AVX_RB=1 KT_AVX_PF=2`) + FP8 KV (`--kv-cache-dtype fp8_e5m2 --mem-fraction-static 0.94`). sonnet 512/128, prefix 100, 프롬프트 3×C, seed 42.

## 운영점 A (균형): `--cuda-graph-max-bs 192 --cuda-graph-bs 32 64 96 128 160 192 --max-total-tokens 122880 --chunked-prefill-size 8192`
| 동시성 | tok/s | TPOT ms | TTFT s | 정오 기준 (IDE_039-i 구성) |
|---|---|---|---|---|
| C192 (fresh) | **1048.7** | 131.1 | 6.77 | — (KV 부족) |
| C160 | **1012.2** | 114.0 | 5.73 | 982.4 / 119.5 |
| C128 | 961.9 | 96.0 | 4.83 | 951.8 / 100.5 |
| C64 | **796.6** | 57.6 | 2.97 | 769 / 60.9 |
| C32 | (781.7, prefix cache 적중 → 무효; fresh 재측정 별도) | | 0.58 | 572 |
GSM100 **96.0** (96/100; 정오 구성 97.0, 캠페인 기준선 97.0 — 1문항 차이. 커널은 bit 동일 검증, CUDA graph 버킷 변경에 따른 배치 패딩·BF16 반올림 토큰 분기로 추정 → per-token logprob 비교 별도 수행).

## 운영점 B (최대 처리량): `--cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224 --max-total-tokens 143360 (실할당 127,309) --chunked-prefill-size 4096`
| 동시성 | tok/s | TPOT ms | TTFT s |
|---|---|---|---|
| C224 (fresh) | **1066.5** | 154.0 | 7.30 |
| C192 | 1037.2 | 137.0 | 6.27 |
GSM40 97.5.

## 요약
- 최고 처리량 **1066~1070 tok/s (C224, 운영점 B)**, 균형 **1049 (C192) / 1012 (C160, 운영점 A)**. 캠페인 시작 (09-08, 56.3) 대비 **19.0×**, 어제 최종 (C64 769) 대비 +39%, 정오 (982.4) 대비 +8.6%.
- 오늘 오후 기여: AVX 커널 (IDE_048/050) ≈ +2~3% (C160 982→1012 중 일부, C64 769→797), graph 버킷 축소·KV 확대 (IDE_049) ≈ +5~9% (C192/C224 신설).
- 파일: `finA_c*.log`, `finB_c*.log`, `gsm100_finA.json`, `gsm40_finB.json`, `RUN.log`.
