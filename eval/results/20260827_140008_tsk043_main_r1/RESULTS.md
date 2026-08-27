# TSK_043 본판 — DeepSeek-R1-0528 KT Hybrid 측정 결과 (2026-08-27)

- 노드: violet-h100-016 (Xeon 8480+×2 AMX / 2TB DDR5 / H100×8, **turbo OFF 2.0GHz**)
- 모델: deepseek-ai/DeepSeek-R1-0528 (native FP8, **642GB** — HBM 총량 640GB 초과)
- 스택: SGLang 0.5.18 (lmsysorg/sglang:latest) + kt-kernel 0.7.0.post2 (--no-deps) + 호환성 패치 4건
- CPU expert: AMXINT4 변환본 (328GB, `kt quant -m int4 -i fp8`, 변환 65분) — 257 experts/layer 전량 CPU

## r0 — GPU-only 불가 실증 (TST_021 게이트 1) ✅

vLLM 0.28, TP=8, gmu 0.95 → 90초 만에 `torch.OutOfMemoryError: CUDA out of memory` (worker 로드 중).
**642GB > usable HBM ~608GB — 이 머신에서 hybrid 가 "선택"이 아니라 "필수"인 regime 실증.**

## r1 — KT Hybrid 서빙 (GPU MLA attention TP=8 + CPU AMX experts)

| 항목 | 값 |
|---|---|
| 기동 | HEALTH OK **160s** (642GB expert → DRAM, page cache 덕) |
| 처리량 | **19.67 out tok/s** (32 req × 1024/128, C=8, 전 req 완료) |
| TTFT p50 / TPOT p50 | 11,199 ms / 305.5 ms |
| CPU busy | **avg 42.7% / max 49.5%** (cpuinfer 96 threads 포화 — CLAUDE.md "CPU 활용" 직접 달성) |
| GPU util | avg 75.6% (MLA attention + 통신) |

### ⚠ 품질 게이트 미통과 (후속 SUB_167)

greedy smoke 출력이 비문 (따옴표/기호 스팸). **판별 실험 완료**:

| 실험 | 결과 |
|---|---|
| chat template 경유 | 동일 깨짐 → template 아님 |
| Qwen3-30B-A3B-FP8 → AMXINT8 | 출력 정상 |
| **Qwen3-30B-A3B-FP8 → AMXINT4 (동일 변환기)** | **출력 정상** → INT4 경로 일반 결함 아님 |
| 변환 스크립트 main vs v0.7.0.post1 태그 | IDENTICAL → 버전 스큐 아님 |
| kt-kernel 0.7.0 스왑 | 무관한 import 오류로 부팅 불가 — 판정 불가 |

→ **원인 후보를 "DeepSeek 계열 특이 처리" (block-wise `weight_scale_inv` 128×128 dequant 또는 shared expert(257번째) 폴딩) 로 좁힘**. upstream (kvcache-ai/ktransformers) 이슈 조회/제보 대상.

## 해석

- 처리량 19.7 tok/s 는 turbo OFF (2.0GHz) + expert 전량 CPU (`kt-num-gpu-experts 0`) + deferral 미사용의 **하한**. KT 공식 참조치 (8×L20+Xeon, R1 227 tok/s total) 대비 개선 여지: turbo unlock, hot expert GPU 배치 (`--kt-num-gpu-experts` 수십~수백), expert deferral, cpuinfer 튜닝
- 그러나 성능 논의는 **품질 게이트 통과 후**에만 유효 (SUB_167 선행)
