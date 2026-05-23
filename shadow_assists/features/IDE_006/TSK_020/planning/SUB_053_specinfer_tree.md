# SUB_053 — SpecInfer tree spec decode

> **parent**: TSK_020 / 카테고리 A (Advanced spec decode, CPU draft)
> **status**: 대기 (plan only)
> **effort**: large (1-2 주)
> **CPU% target**: 35-50%
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §1

---

## 1. Mechanism

SpecInfer 는 multiple draft model 또는 multiple sampling temperature 가 **여러 branch 의 token tree** 를 생성 후, GPU 가 tree attention 으로 **모든 branch 를 1 forward 로 동시 verify** 한다. accept tree path 영역 longest valid prefix.

```
[CPU: multiple draft (model A, B, ... 또는 temperature 0.1/0.5/1.0)]
  ↓ (각 draft 영역 K token chain 생성)
[merge → token tree (depth K, breadth N_branch)]
  ↓ (tree H2D)
[main GPU tree attention verify (1 forward)]
  ↓
[accept longest valid path]
```

본 lever 의 CPU 친화 영역: 여러 draft 영역 parallel forward (각 draft model small) + tree merge 영역 CPU 영역.

## 2. 출처

| 자료 | 위치 |
|---|---|
| SpecInfer paper | [arXiv 2305.09781](https://arxiv.org/abs/2305.09781) — "SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification" |
| Reference impl | GitHub `flexflow/FlexFlow` |
| Related | OmniServe (KV mgmt), MMLU spec decode 영역 |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/spec_decode/specinfer.py` (신규) | multi-draft tree generator + tree merge |
| `vllm/v1/spec_decode/tree_attention.py` (신규) | tree attention verify kernel (또는 vLLM 내장 leveraging) |
| `vllm/config/speculative.py` | `method="specinfer"` + n_branch / draft_models list |
| `vllm/v1/worker/gpu_model_runner.py` | tree-aware forward path |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | SpecInfer impl 영역 FlexFlow 영역 검토 | 1-2 일 |
| Phase 1 | SpecInferProposer skeleton + Proposer interface 정합 | 2 일 |
| Phase 2 | multi-draft parallel forward (CPU N draft models) | 2-3 일 |
| Phase 3 | tree attention verify (vLLM 내장 leveraging 또는 신규 kernel) | 3-5 일 |
| Phase 4 | 정확도 + throughput 측정 | 1-2 일 |
| 총 | | **2 주** |

## 5. CPU% target / throughput 가설

- multi-draft (n_branch=3-5) × K=5 영역 token = 15-25 candidate
- 각 draft 영역 CPU small model 영역 forward — total CPU ~10-20ms/step
- 가설: tree verify 영역 accept rate ~75-85% (branch 다양성 영역 hit ↑)
- throughput: +15-25% over ngram, 단 tree attention GPU overhead 변수

## 6. Risk

| 위험 | 완화 |
|---|---|
| tree attention 영역 vLLM native 영역 미지원 — 신규 kernel 영역 큰 작업 | vLLM Eagle2 영역 tree attention 영역 leverage 가능 |
| multi-draft model 영역 별 영역 ckpt 영역 별도 영역 필요 | small ngram + Eagle head + Medusa head 영역 mix |
| n_branch ↑ 영역 GPU verify forward 영역 batch 영역 커져 latency 영역 증가 | n_branch=3 영역 시작 |

## 7. Dependencies

- SUB_050 (Eagle CPU) + SUB_051 (Medusa CPU) — multi-draft 영역 component 영역 활용
- SUB_057 (ngram tree expansion) — tree attention 영역 subset 영역 학습 가능

## 8. Acceptance criteria

- [ ] multi-draft tree generation 정상
- [ ] tree attention verify 정상
- [ ] sonnet throughput ≥ 10,956 tps
- [ ] CPU busy ≥ 35%
- [ ] chat workload 영역 acceptance ≥ 70%
