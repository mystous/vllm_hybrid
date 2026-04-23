# g0_11_qwen2.5_32b_phase1(fail) — §11 Phase 1 측정 (regression 확인)

## 정체성 및 결과 요약

§11 Phase 1 (Option A wire-up): CPU decode attention 을 IPEX `single_query_cached_kv_attention` 대신 `_PagedAttention.forward_decode` → AVX-512 `batch16_paged_attention_v1` 로 강제 라우팅. 기존 kernel 그대로, C++ 변경 0 줄, Python + infra 만 수정.

**결과**: 측정된 모든 seqs (1/2/4/8) 에서 §06-1 v1 대비 **0~12% outTP regression**. §11 Phase 1 **기각**. v1 상태로 롤백, §11 Phase 2 (v2 신규 kernel) 로 가기 전에 선행 연구 기반 상위 기법 (§13 T-MAC / §16 SparAMX / §22 NEO / §28 xFT) 로 방향 재검토.

- 측정일: 2026-04-20
- Branch 시점: `ninja-gap/11-batch-aware-decode-attn` (commit `1d5a7a711` 기반)
- env: `g0_h100x8_qwen32b_11.env` (§06 + §06-1 v1 stack + §11 on)
- 환경: H100x8 + Qwen2.5-32B + TP=8 + 500 req × 128/128

## 측정 범위

gpu_only baseline + hybrid sweep seqs 1/2/4/8 까지. seqs 16 부터는 §06-1 v1 과 §11 Phase 1 의 차이가 seqs=8 까지 모두 regression 으로 굳어져 **측정 중단**. seqs 16 이 유일하게 `batch16_paged_attention_v1` 의 full-batch path 를 활성화하는 구간이었으나, 하위 구간 전체 regression 이므로 진행 의미 없음.

## outTP 비교 (§06-1 v1 vs §11 Phase 1)

| seqs | §06-1 v1 | §11 Phase 1 | Δ (§11 − v1) | gpu_only |
|---:|---:|---:|---:|---:|
| 1 | 1,196.3 | 1,056.5 | **−11.7%** | 11,522.95 |
| 2 | 794.0 | 735.3 | **−7.4%** | 11,522.95 |
| 4 | 496.2 | 501.1 | +1.0% | 11,522.95 |
| 8 | 272.3 | 258.4 | **−5.1%** | 11,522.95 |

## 왜 실패했는가 — 추정 원인

### 1. 대상 구간 (seqs 2/4/8) 이 kernel 의 remainder path

`batch16_paged_attention_v1` 는 `num_seqs ≥ 16` 이어야 full-batch OMP-parallel `(batch×head)` 경로가 켜짐. seqs 2/4/8 에서는 `num_full_batches = 0`, remainder 단일-seq loop 로 떨어짐. 이 구간은 IPEX `single_query_cached_kv_attention` 와 memory access / 병렬화 구조가 본질적으로 동일. 이득이 나올 구조가 아니었음.

### 2. prefill SDPA fallback 오버헤드

§11 은 layout 일관성 유지 위해 flag on 시 IPEX `flash_attn_varlen_func` (prefill path) 도 스킵하고 pure SDPA 로 route. IPEX prefill kernel 이 SDPA 보다 빠른 구간이 있었으면 이 전환이 순손실.

### 3. seqs=1 regression (−11.7%)

M=1 GEMV path 로 attention kernel 과 무관한 경로. 그럼에도 regression 관측. 원인은:
- **측정 노이즈** (single-run variance) 가능성 — §06-1 v2(fail) 도 seqs=1 에서 동일한 -10% 관측했음
- 또는 _PagedAttention 경로 강제 전환이 간접적으로 메모리 allocator / cache behavior 에 영향

### 4. 근본 — CPU batch 병렬화 자체가 불작동

더 큰 구조적 문제 (§11 과 무관): §06-1 v1 자체가 seqs=1 (1196) → seqs=8 (272) 로 **4.4× 감소**. warmup 시간도 seqs 에 기하급수 증가. M>1 에서 kernel 이 병렬 이득 없이 serialize 하고 있다는 증거. `q8_0_gemm_vnni_impl` 의 weight reuse 가 이론대로 작동 안 함.

## gpu_only 대비 절대 수치

gpu_only outTP = **11,522.95 tok/s** (단일 레퍼런스, seqs 축 무관).  
hybrid 최고치 = seqs=1 §06-1 v1 **1,196.3 tok/s** = gpu_only 의 **10.4%**.  
hybrid 어느 설정에서도 gpu_only 를 초과 못함. α (CPU engine 기여) 는 현재 구간에서 음수.

## 결론

§11 Phase 1 (Option A, kernel 교체 없이 기존 `batch16_paged_attention_v1` dispatch 활성) 은 의미 있는 이득 없음. Phase 2 (v2 신규 kernel, {1,4,8,16,32} template specialization + block coalescing) 로 진행할지 재평가 필요.

더 중요한 판단: **CPU batch 경로 전체의 근본 결함** 이 §11 으로 해결될 성질의 것이 아님. 선행 연구 실측 수치 있는 Tier 1 기법 (§13 T-MAC INT4 4× / §16 SparAMX SPR 1.42× / §22 NEO H100 70B 14.3% / §28 xFT 포팅) 으로 방향 전환이 더 근거 있음.

## 구조

```
g0_11_qwen2.5_32b_phase1(fail)/
├── README.md                     # 본 파일
├── g0_h100x8_qwen32b_11.env      # env snapshot
├── gpu_only_baseline/            # TP=8 gpu_only
└── seqs{1,2,4,8}/                # hybrid §06 + §06-1 v1 + §11 sweep (seqs 16 미측정)
```

## 관련

- **§06-1 v1 (직전 스택)**: `../g0_06_1_qwen2.5_32b_v1/` — §11 은 이 위에 얹힘
- **§06-1 v2(fail)**: `../g0_06_1_qwen2.5_32b_v2(fail)/` — 같은 실패 패턴 (kernel 변경 regression)
- **base**: `../g0_00_qwen2.5_32b_base/` — TP=8 §06 off baseline
- 기법 문서: `NinjaGap_Todo/11_batch_aware_decode_attn.md`
- 이 실패 데이터가 유용한 시점: §11 Phase 2 재시도 또는 §13/§16/§22/§28 로 전환 시 bottleneck 재진단 기준
