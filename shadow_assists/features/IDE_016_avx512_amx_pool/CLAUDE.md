# CLAUDE.md — IDE_016 구현 시 알아야 할 것

## 0. 핵심 규칙
- 본 feature 구현 시 **Phase A 측정 결과 (SUB_098~168)** 가 input baseline
- C++ kernel build 는 **prod 머신 (Sapphire Rapids + g++-12+)** 에서만 AMX 직접 verify
- 개발 머신 (Alder Lake) 은 AVX-512 만 가능 (AMX 부재) — AMX 코드는 cross-compile + Intel SDE simulator
- **물리 코어 0-111 만 사용**, HT 시블링 (112-223) 금지, **최대 100 core 활용** (12 core kernel 여유 보존)
- 측정 시간 **KST 표시** (`TZ=Asia/Seoul date`)
- commit/push 사용자 명시 지시 시에만
- 측정 default **1-run** (사용자 명시 multi-run 시만)

## 1. Phase A 측정 결과 input

| Phase A finding | implication |
|---|---|
| sampler.py 44.3% CPU (SUB_161) | TSK_025 의 main target |
| AMX 22.05 TFLOPS peak (SUB_106) | TSK_026 의 lower bound |
| 10.24 TFLOPS available (SUB_117) | total CPU compute budget |
| VLLM threads 96% S (SUB_162) | CPU 자원 idle gap 확인 |

## 2. 구현 우선순위

1. **TSK_025** (AVX-512 sampling) — SUB_161 의 44% lever, 가장 큰 lift potential
2. **TSK_026** (AMX draft head matmul) — IDE_019 의 dependency
3. TSK_024 (AVX-512 tokenizer) — TSK_032 attention-phase task 의 dependency
4. TSK_027 (AMX prefill assist) — IDE_002 operationalize, complexity 큼

## 3. 측정 protocol

- canonical: Qwen 32B TP=4×2 AGSD-gated × 500p × 32 concurrency (SUB_160 protocol)
- baseline: SUB_160 의 500p baseline (balanced AGSD 5,474 tps)
- target: 각 TSK 별 +5-10% net positive (TSK_025 alone)

## 4. 검증 게이트

CLAUDE.md 의 Constraint:
> GPU만 사용 했을 때와 결과 값이 달라져서는 안됨

운영 해석: token-level bit-exact 가 아니라 **분포·의도 수준의 유사성** (per-token logprob max abs diff, sequence PPL relative diff). 측정 시 본 게이트 적용.

## 5. dependencies (다른 IDE / feature)

- IDE_015 의 측정 결과 (Phase A) — 위 1번
- TSK_020 의 AGSD router (sub094_router.py) — 측정 환경
- spec_decoding/plan/README.md §IDE_016 — full plan

## 6. 알려진 risk + fallback

| risk | fallback |
|---|---|
| AVX-512 microcode fuse-off (CLAUDE.md) | AVX2 fallback path 필수 — `__attribute__((target("avx2")))` versioned function |
| AMX 가 prod 머신만 — 개발 머신 미지원 | dev 에선 build + unit test, prod 에서만 e2e 측정 |
| sampling 정확도 위반 (top-k/p edge cases) | reference GPU baseline 과 per-token logprob comparison 게이트 |
| TP=4×2 dual instance 영역 두 vllm 의 sampling overhead 공유 자원 | per-process AVX-512 pool + lockless ring buffer 권장 |
