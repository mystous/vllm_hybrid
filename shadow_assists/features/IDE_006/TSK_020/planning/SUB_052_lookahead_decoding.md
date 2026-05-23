# SUB_052 — Lookahead Decoding (CPU Jacobi)

> **parent**: TSK_020 / 카테고리 A (Advanced spec decode, CPU draft)
> **status**: 대기 (plan only)
> **effort**: medium (2-3 일)
> **CPU% target**: 40-60% / **throughput 가설**: SUB_047 대비 +5-15%
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §1

---

## 1. Mechanism

Lookahead Decoding 은 Jacobi iteration 으로 **n-gram pool** 을 fully parallel 생성 한다. main GPU model 의 fixed-point iteration 형태 — 매 step 마다 n-gram lookahead window (W) 와 ngram_size (N) 에 대해 W × N 개 token 후보를 한꺼번에 ungreedy 추정 + n-gram cache lookup.

CPU 영역 친화: Jacobi iteration 자체 가 embarrassingly parallel (window position 별 독립) + n-gram cache 가 prompt-local + small.

```
[CPU: Jacobi iteration on lookahead window (parallel positions)]
  ↓ (n-gram pool 영역 W × N matrix)
[CPU: n-gram cache lookup (radix tree-like)]
  ↓ (validated n-gram chains 영역 GPU 영역 verify queue)
[main GPU forward + verify chain (1 forward)]
  ↓
[accept longest matching prefix]
```

## 2. 출처

| 자료 | 위치 |
|---|---|
| 본 paper | [arXiv 2402.02057](https://arxiv.org/abs/2402.02057) — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" |
| Reference impl | GitHub `hao-ai-lab/LookaheadDecoding` |
| vLLM 내장 | **없음** — port 필요 |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/spec_decode/lookahead.py` (신규) | Jacobi iteration kernel + n-gram pool lookup |
| `vllm/v1/spec_decode/__init__.py` | LookaheadProposer 등록 |
| `vllm/config/speculative.py` | `method="lookahead"` + window/ngram_size 영역 옵션 |
| `vllm/v1/spec_decode/utils.py` | CPU Jacobi 영역 numba/cython 영역 빠른 영역 path |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | LookaheadDecoding 원본 impl 검토 (jacobi iteration 영역 핵심 함수) | 0.5 일 |
| Phase 1 | vLLM `LookaheadProposer` skeleton (Proposer interface 정합) | 0.5 일 |
| Phase 2 | Jacobi iteration CPU 영역 numba 영역 빠른 path | 1 일 |
| Phase 3 | n-gram pool lookup + GPU verify integration | 0.5 일 |
| Phase 4 | 정확도 + throughput 측정 | 0.5 일 |
| 총 | | **3 일** |

## 5. CPU% target / throughput 가설

- Jacobi iteration: window W=15, ngram_size N=5 영역 W × N = 75 token 영역 parallel
- numba parallel + 8 thread/rank 영역 ~5-10ms/step CPU
- n-gram cache: hash table lookup ~1-2ms
- 가설: ngram (SUB_047) acceptance ~60% → Lookahead 영역 longer n-gram chain 영역 acceptance ~65-70% (sonnet 같은 repetitive workload)
- throughput: +5-15% (SUB_047 위 영역 추가 lever)

## 6. Risk

| 위험 | 완화 |
|---|---|
| Lookahead 영역 fixed-point convergence 영역 step 영역 다양 — 일관 throughput 보장 안 됨 | window/ngram_size sweep 영역 sweet spot 영역 |
| vLLM Proposer interface 영역 lookahead 영역 fit 안 함 (multi-position 영역 동시 verify 영역 필요) | rejection_sampler 영역 extension 필요 |
| 본 lever 영역 ngram 영역 위 영역 추가 영역 minimal gain 가능 (sonnet 영역 ngram 영역 이미 충분히 hit) | 일반 chat / code workload 영역 본격 효과 |

## 7. Dependencies

- vLLM `Proposer` interface (이미 존재)
- numba 또는 cython (n-gram lookup 영역 빠른 path)

## 8. Acceptance criteria

- [ ] LookaheadProposer 등록 + spec verify 정상
- [ ] sonnet throughput ≥ 10,956 tps (SUB_047 동등 이상)
- [ ] CPU busy ≥ 40%
- [ ] chat workload (vs sonnet) 영역 acceptance 추가 ≥ 5pp
