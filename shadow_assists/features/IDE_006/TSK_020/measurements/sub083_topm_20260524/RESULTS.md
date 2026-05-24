# SUB_083 — Phase 4: ngram top-M tree verify (design analysis, blocked)

> **parent**: TSK_020 (성능 향상 plan Phase 4)
> **plan**: [`../../planning/SUB_083_ngram_top_m_tree_verify.md`](../../planning/SUB_083_ngram_top_m_tree_verify.md)
> **measurement**: 2026-05-24 22:43 KST, **design analysis only — actual implementation 영역 본 session 범위 초과 (2-3 일 effort)**
> **status**: ◐ **design 정리, implementation 후속**

---

## 1. 본 fork 영역 현 상태 (SUB_057)

`vllm/v1/spec_decode/ngram_proposer.py` (690 lines) — `VLLM_NGRAM_TOP_M>1` 시 본 fork 영역 wired:

```python
self.ngram_top_m = max(1, int(os.environ.get("VLLM_NGRAM_TOP_M", "1")))
if self.ngram_top_m > 1:
    # Pre-allocate top-M buffer (3D: max_num_seqs × top_m × k)
    self.valid_ngram_draft_topm = np.zeros((max_num_seqs, self.ngram_top_m, self.k), dtype=np.int32)
    self.valid_ngram_topm_count = np.zeros((max_num_seqs), dtype=np.int32)
    warnings.warn("SUB_057: VLLM_NGRAM_TOP_M=N wired — top-M numba kernel active, "
                  "but only chain 0 (longest) used until rejection_sampler tree verify is implemented.")
```

- numba kernel (`batch_propose_numba_topm`) 영역 top-M chain 영역 buffer 영역 채움 ✓
- rejection_sampler 영역 chain 0 만 verify (나머지 unused) ✗

## 2. rejection_sampler.py 영역 현 single-chain path 분석

`vllm/v1/sample/rejection_sampler.py` (850 lines) — 핵심 entry:

| 함수 | 라인 | scope |
|---|---|---|
| `class RejectionSampler(nn.Module)` | 30 | main entry |
| `output_token_ids = rejection_sample(metadata.draft_token_ids, ...)` | 141 | 1D draft tensor (single chain per request) |
| `rejection_greedy_sample_kernel` | 395 | greedy mode triton kernel |
| `rejection_sample` | 350 | core function, `draft_token_ids.ndim == 1` 가정 |

**핵심 제약**: 현 `rejection_sample` 가 **1D draft tensor** (각 request 영역 single chain) 가정. tree verify 영역 다음 변경 필요:
- input shape: `[batch_size, top_m, k]` (3D) 또는 tree node-based encoding
- target model forward: tree mask 영역 attention 영역 multi-branch 영역 동시 처리
- rejection: 각 chain 영역 independent reject → longest accepted chain 영역 선택
- KV cache: 선택된 chain 만 commit, 나머지 rollback

## 3. 변경 surface area (estimate)

| file | 변경 종류 | 추정 라인 |
|---|---|---|
| `vllm/v1/sample/rejection_sampler.py` | tree verify path 추가 (multi-chain reject + longest accept selection) | ~150-250 |
| `vllm/v1/spec_decode/metadata.py` | `SpecDecodeMetadata` 영역 top_m field + tree encoding | ~30 |
| `vllm/v1/worker/gpu_model_runner.py` | target forward 영역 tree mask 영역 attention 영역 | ~80-150 |
| `vllm/v1/attention/backends/*.py` | tree attention mask 영역 backend 영역 | ~50-100 per backend |
| `vllm/v1/spec_decode/ngram_proposer.py` | metadata 영역 top_m chain 영역 packaging | ~30-50 |
| **합계** | | **~350-600 라인** |

→ **effort 2-3 일 + 정확도 검증 1-2 일 = 총 1 주** 추정. 본 session 영역 implementation 불가.

## 4. tree attention 영역 reference

vLLM 영역 이미 tree attention 지원 — EAGLE-2 (`vllm/v1/spec_decode/eagle.py`) 와 Medusa (`vllm/v1/spec_decode/medusa.py`) 영역 dynamic / fixed tree mask 영역 처리. 본 ngram top-M 영역 같은 framework 활용 가능 — 새 attention backend 영역 안 만들어도 됨.

## 5. expected gain analytical (SUB_075 acceptance 데이터 영역)

본 fork 영역 SUB_075 측정 (sonnet ngram_spec7_cap8): per-position α = [0.522, 0.407, 0.349, 0.296, 0.274, 0.246, 0.240] (position 1~7).

single-chain 영역 mean_accept_len = `1 + Σ (∏_i α_i)` (cumulative)... 실측 3.72.

tree verify 영역 top_m chains (예: top_m=3):
- 각 chain 영역 independent prefix → 각 position 영역 best chain 선택 가능
- 추정: position 1 영역 (1 − (1−α_1)^3) = (1 − 0.478^3) = 0.891 (single 0.522 의 1.7× 향상)
- 누적 K 향상: ~1.4-1.6×

→ **sonnet K 3.72 → ~5.5-6.0** 가능성 (acceptance ↑). tps 향상 추정:
- ngram (current best): 10,956.5 tps (K=3.72)
- ngram tree verify (가설, top_m=3): 10,956.5 × (5.5/3.72) / (1 + tree overhead 0.1) = ~14,800 tps
- vs vanilla 4,679.8 = **+216%** (현 +134% 영역 +82 pp 추가)

→ **Phase 4 영역 sonnet workload 영역 +80 pp 추가 향상 가능성** (현실 영역 50% 정도만 달성 가정도 +40 pp).

## 6. 본 session 진행 한계 정리

| 항목 | 본 session 가능성 |
|---|---|
| design analysis (현 single-chain path 분석) | ✅ 완료 (본 doc §2-3) |
| variation surface estimate | ✅ 완료 (350-600 라인) |
| expected gain analytical | ✅ 완료 (+80 pp sonnet 가능성) |
| actual rejection_sampler patch | ✗ vLLM core large surface (1 주 effort) |
| 측정 검증 | ✗ patch 후 가능 |

## 7. 후속 SUB candidate

- SUB_088 (제안, 1 주): rejection_sampler tree verify path 구현 + 정확도 검증
- SUB_089 (제안, 0.5 일): tree verify patch 후 sonnet workload 영역 top_m sweep (1/3/5/8)
- SUB_090 (제안): chat/code workload 영역 측정 (단 code 영역 α≈0 영역 효과 없음 예상)

## 8. risk

- tree attention 영역 vLLM v1 영역 backends 호환성 영역 backend-by-backend test 필요
- 정확도 issue 가능 (multi-chain rejection 영역 sampling distribution 영역 변경)
- ngram drafter 영역 top-M chains 영역 prefix common 영역 적어 tree attention 영역 가치 작을 수도 (vs EAGLE 영역 deep tree)
