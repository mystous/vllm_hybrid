# SUB_083 — Phase 4: ngram top-M tree verify 구현

> **parent**: TSK_020 (성능 향상 plan Phase 4)
> **status**: 활성 (2026-05-24 신설)
> **effort**: 2-3 일
> **based on**: SUB_057 (top-M numba kernel wired, rejection_sampler tree verify TODO)

## 1. 목표

`VLLM_NGRAM_TOP_M>1` 시 본 fork 영역 wired 영역 top-M numba kernel 의 rejection_sampler 영역 tree verify path 구현. sonnet K 3.72 → ~4.5 가능성 (acceptance ↑).

## 2. 현재 상태 (SUB_057)

`vllm/v1/spec_decode/ngram_proposer.py` 영역:
```python
self.ngram_top_m = max(1, int(os.environ.get("VLLM_NGRAM_TOP_M", "1")))
if self.ngram_top_m > 1:
    # Pre-allocate top-M buffer
    self.valid_ngram_draft_topm = np.zeros((max_num_seqs, self.ngram_top_m, self.k), ...)
    warnings.warn("SUB_057: VLLM_NGRAM_TOP_M=N wired — top-M numba kernel active, "
                  "but only chain 0 (longest) used until rejection_sampler tree verify is implemented.")
```

→ numba kernel 영역 top-M chain 영역 buffer 영역 채움 (`batch_propose_numba_topm`), 단 rejection_sampler 영역 chain 0 만 verify. 나머지 chains 영역 unused.

## 3. 진행 절차

### Step 1 — rejection_sampler tree verify path 분석

`vllm/v1/sample/rejection_sampler.py` 영역 현재 chain (single sequence) 영역 verify. tree (multi-chain) verify 영역 다음 영역 변경 필요:
- input: top-M chains (각 길이 γ)
- target model forward: 각 chain 영역 target logit 계산 (chain 들 영역 prefix common 영역 attention mask 영역 신중히)
- rejection: 각 chain 영역 independent reject sampling, longest accepted chain 영역 선택
- KV cache: 선택된 chain 만 commit, 나머지 rollback

### Step 2 — 구현

vLLM 영역 tree-attention 영역 지원 path 확인 — EAGLE-2 / Medusa 영역 tree mask 영역 어떻게 처리하는지 reference. ngram tree verify 영역 같은 framework 활용 가능.

### Step 3 — 측정 sweep

`VLLM_NGRAM_TOP_M` 영역 1 / 3 / 5 / 8 × 3 workload (sonnet/chat/code). expected:
- sonnet: K 3.72 → 4.5+ (top-M chain 영역 더 긴 chain 선택 가능)
- chat: K 6.69 → 비슷 (이미 매우 높음)
- code: K 1.10 → 거의 변화 없음 (acceptance 자체가 0 인 환경)

## 4. risk

- rejection_sampler core 변경 영역 정확도 issue 가능 — 광범위 정확도 test (canonical 3-run + variance) 필수
- tree attention mask 영역 vLLM v1 의 batch_attn 영역 호환 가능한지 확인
- effort 큼 (vLLM core 변경 영역 large surface)

## 5. 산출물

- `vllm/v1/sample/rejection_sampler.py` 영역 tree verify path
- `/tmp/run_sub083_topm_sweep.sh`
- `measurements/sub083_topm_<TS>/RESULTS.md`
