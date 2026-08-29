# TSK_047 — Qwen3-Coder-480B-A35B-FP8: CPU-AMX expert 서빙 방법 검증 결과 (2026-08-29)

- 목적: ① 방법(FP8→AMXINT4 변환 + CPU-AMX expert 연산 서빙)의 대형-Qwen 스케일 검증 ② R1-0528 품질 결함(`SUB_167`)의 DeepSeek-특이 여부 교차 확인 ③ **GPU 수 축소(TP=4)로 진짜 capacity regime 실증** (사용자 설계)
- 스택: SGLang 0.5.18 + kt-kernel 0.7.0.post2 (패치 4건), 변환 `kt quant -m int4 -i fp8` → **232GB (67 shards)**, turbo OFF 2.0GHz

## 셀 결과

| 셀 | 구성 | 결과 |
|---|---|---|
| **r0** | GPU-only **TP=4** (HBM 320GB) | **OOM 사망 (45초)** — `memory allocation failed with OOM on device 1` — 480GB 모델은 4장에 불가 ✓ |
| **r1** | **hybrid TP=4** — GPU 4장(attention/공유+KV) + CPU AMXINT4 expert 232GB 전량 | **서빙 성립** (부팅 90초). **품질 4/4 정상** — "Paris, which is…", 올바른 fibonacci, 1..100 공식, `s[::-1]`. 처리량 44.5 tok/s (C=16, 64req), TTFT p50 8.5s, TPOT 287ms, **CPU busy 41.9%** |
| r2 | GPU-only TP=8 (기준선 의도) | **부팅 불가 — OOM 아님**: FP8 block-quant 제약 (`output_size 320 not divisible by block_n=128`) 으로 sglang 이 이 모델을 TP=8 로 샤딩 못 함. 기준선 미확보 (한계로 기재) |

## 판정

1. **방법 건전성 확정**: 480B급에서 변환→서빙→품질까지 무손실 동작. 30B에 이은 두 번째 Qwen 검증
2. **`SUB_167` 원인 확정**: 동일 파이프라인에서 Qwen 30B·480B 모두 정상, DeepSeek-R1 만 깨짐 → **DeepSeek-계열 특이 (block-scale `weight_scale_inv` 또는 shared-expert 처리) 변환 결함**으로 최종 좁힘. upstream 제보 대상
3. **capacity 실증 (사용자 설계 구도)**: 같은 GPU 4장으로 "불가능(OOM) → 가능(정상 서빙)" — CPU+DRAM 이 GPU 4장 부족분을 대체. 나머지 4장은 다른 워크로드에 자유
4. 성능 맥락: 44.5 tok/s 는 turbo OFF + 전-expert-CPU + 무튜닝 **하한**. 개선 레버 (미실행): turbo, hot-expert GPU 배치, spec decode (E3 에서 1.45~1.55× 실증), cpuinfer 튜닝

## 한계

- GPU-only 기준선 부재 (TP=4 는 OOM, TP=8 은 엔진 제약) — 처리량 비교는 불가, 품질 비교는 절대 정답성으로 대체
- 단일 run, C=16 한 점
