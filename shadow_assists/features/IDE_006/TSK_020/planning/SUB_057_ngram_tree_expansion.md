# SUB_057 — ngram tree expansion (multi-chain candidates)

> **parent**: TSK_020 / 카테고리 C (vLLM 내부 CPU lever, SUB_047 확장)
> **status**: 대기 (plan only)
> **effort**: medium (2-3 일)
> **CPU% target**: 15-25% / **throughput 가설**: acceptance 60→75% → +10-15%
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §3

---

## 1. Mechanism

현 SUB_047 의 ngram proposer 는 prompt 안에서 1 개의 best matching chain 영역 K token 영역 제안. 본 SUB 영역 영역 **top-M candidate chain** 영역 tree 구조 영역 동시 제안 + GPU 영역 tree attention 영역 영역 verify.

```
[ngram lookup, top-M chains (M=3-5)]
  ↓ (each chain K=7 token)
[merge → token tree (depth K, breadth M)]
  ↓ (tree H2D)
[GPU verify tree (1 forward, tree attention)]
  ↓ (accept longest valid path)
```

SUB_050 (Eagle CPU draft) / SUB_053 (SpecInfer tree) 영역 tree verify 영역 아이디어 영역 ngram 영역 적용.

## 2. 출처

| 자료 | 위치 |
|---|---|
| SpecInfer tree | [arXiv 2305.09781](https://arxiv.org/abs/2305.09781) §3 (tree verification mechanism) |
| Eagle2 dynamic tree | [arXiv 2406.16858](https://arxiv.org/abs/2406.16858) §4.2 (dynamic tree construction) |
| vLLM 내장 | `vllm/v1/spec_decode/ngram_proposer.py` (SUB_047 patch) — top-1 chain 만 |
| tree attention | vLLM Eagle 영역 tree attention 영역 leverage 가능 |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/spec_decode/ngram_proposer.py` | top-M chain 영역 출력 (numba 영역 top-M priority queue) |
| `vllm/v1/spec_decode/rejection_sampler.py` | tree path 영역 verify 영역 accept longest path logic |
| `vllm/v1/attention/backends/flash_attn.py` | tree attention mask (Eagle2 영역 pattern 영역 leverage) |
| `vllm/config/speculative.py` | `ngram_top_m` field 추가 (default 1 = 현 동작) |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | Eagle2 tree attention impl 검토 (vLLM eagle.py) | 0.5 일 |
| Phase 1 | ngram_proposer 영역 top-M chain output (numba multi-thread) | 1 일 |
| Phase 2 | rejection_sampler tree-aware verify | 1 일 |
| Phase 3 | env-gated sweep (M=1/3/5) + throughput 측정 | 0.5 일 |
| 총 | | **3 일** |

## 5. CPU% target / throughput 가설

- ngram lookup 영역 top-M chain (M=3-5) → CPU thread 영역 더 많은 영역 work
- SUB_047 cap=8 영역 위 영역 추가 work → CPU busy 5.51% → 15-25%
- 가설: acceptance rate 60% → 75% (tree 영역 다양성 영역 hit ↑) → throughput +10-15%
- vs SUB_047 best: 10,956 → ~12,000-12,500 tps 영역 가능

## 6. Risk

| 위험 | 완화 |
|---|---|
| tree expansion 영역 GPU verify forward 영역 batch 영역 커져 latency 영역 증가 | top-M 영역 작게 (M=3 start) |
| ngram top-M 영역 lookup 영역 numba 영역 implement 필요 | numba parallel + priority queue |
| 모든 chain 영역 동일 prefix 영역 시 redundancy | dedup logic |

## 7. Dependencies

- SUB_047 (cap=8 patch) — 본 lever 영역 위 영역 적재
- Eagle2 tree attention pattern (vLLM 영역 기존 leverage)

## 8. Acceptance criteria

- [ ] ngram top-M output 정상
- [ ] tree verify accept rate ≥ 70% (sonnet, M=3)
- [ ] throughput ≥ 11,500 tps (SUB_047 의 +5% 이상)
- [ ] CPU busy ≥ 15%
- [ ] 정확도: top-1 token-id ≥ 99% vs SUB_047
