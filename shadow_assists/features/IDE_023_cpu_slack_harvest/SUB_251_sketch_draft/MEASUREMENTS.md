# SUB_251 — Sketch/Hash 기반 spec draft (R4, 시야확대 B축) — no-go (2026-06-16)

> 10라운드 루프 R4. 신규성 기준(=vLLM 미적용): vLLM draft는 ngram=KMP 정확매칭(numba JIT)/
> suffix=arctic tree뿐, 확률적 자료구조 기반 draft 없음 → sketch-draft는 vLLM에 신규.
> 목표: O(1) hash/sketch로 draft 생성 가속 → host-bound 구간 tps↑.

## CPU 마이크로벤치 (`exp/bench_sketch_draft.py`, 8B 토크나이저 반복 코퍼스 3종)
| drafter | ms/call | accept | 비고 |
|---|---:|---:|---|
| KMP (vLLM 현행, 순수 Python 재현) | 0.0236 | 0.986 | 기준 |
| Hash 단일-n (n=3) | 0.0056 | 0.845 | 4.3× 빠름이나 accept −0.14 (최장매칭 손실) |
| **MultiOrder hash (n=3~8, 최장우선)** | 0.0153 | **0.986** | accept ±0 회복, 1.5× |

## 판정 = **no-go (신규지만 향상 없음)**
1. **1.5× < 게이트 2×** — 최장매칭 회복 위해 다중 n 조회.
2. **baseline 불리**: vLLM 실제 ngram은 **numba `@njit`** — Python dict MultiOrder는 numba-KMP 대비 우위 불확실.
3. **구조적 천장 (결정적)**: SUB_225 확정 — 70B GPU-bound, CPU draft가 GPU verify와 완전 오버랩
   (critical path 밖). draft 0.02ms vs GPU forward 수십 ms → draft 가속은 70B tps 무영향.
   (jemalloc draft −17%에도 70B tps 0변화로 이미 입증.)

## 함의
- sketch-draft는 **신규성은 있으나(vLLM 미적용) 측정 향상 없음** → 루프 규칙상 다음 라운드.
- 좁은 가능성: **소형모델(0.5~8B)+고동시성** host-bound 레짐에선 draft가 critical path일 수 있어
  numba/C MultiOrder가 의미 가능 — 단 이득 작고 표적이 70B 아님.
- 재확인: **CPU-side 레버는 70B에서 SUB_225 천장에 막힘** (233 prefetch·225 jemalloc·231 false-sharing과 동류).
산출물: `exp/bench_sketch_draft.py`.
