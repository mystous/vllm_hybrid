# Ninja Gap Backlog — Tier 2 보류 / 장거리 기법 (2026-04-20)

Tier 1 후보 4개 (§13 T-MAC / §16 SparAMX / §22 NEO / §28 xFT) 실측 검증 이후 재평가 대상. 현재 작업 축에서 제외.

## Tier 2 (원리만, 실측 수치 없음)

우리 환경에 보고된 실측 수치가 없어 근거 약한 기법들. 각 문서에 메커니즘 / 예상 이득 / 위험 상세 유지.

| § | 기법 | 문서 |
|---|---|---|
| §07 | ISA Binary Dispatch (AVX-512 ↔ AMX) | [07_isa_binary_dispatch.md](../NinjaGap_Todo/07_isa_binary_dispatch.md) |
| §08 | Kernel Fusion (QKV / Gate-Up / Residual+Norm) | [08_kernel_fusion.md](../NinjaGap_Todo/08_kernel_fusion.md) |
| §09 | Softmax + SiLU LUT | [09_softmax_silu_lut.md](../NinjaGap_Todo/09_softmax_silu_lut.md) |
| §10 | Head Folding (GEMV → GEMM) | [10_head_folding.md](../NinjaGap_Todo/10_head_folding.md) |
| §12 | Barrier / Sync 감소 (OMP persistent region) | [12_barrier_sync_reduction.md](../NinjaGap_Todo/12_barrier_sync_reduction.md) |
| §14 | AVX / AMX Cascade Pipeline | [14_avx_amx_cascade.md](../NinjaGap_Todo/14_avx_amx_cascade.md) |
| §15 | AMX Weight Pre-pack (독자 제어) | [15_amx_weight_prepack.md](../NinjaGap_Todo/15_amx_weight_prepack.md) |
| §17 | Core Group Systolic Pipeline | [17_core_group_pipeline.md](../NinjaGap_Todo/17_core_group_pipeline.md) |
| §23 | CPU Native Quantization (llama.cpp Q8_0/Q4_K) | [23_cpu_native_quantization.md](../NinjaGap_Todo/23_cpu_native_quantization.md) |
| §24 | Activation Quant (W8A8 full) | [24_activation_quantization_w8a8.md](../NinjaGap_Todo/24_activation_quantization_w8a8.md) |
| §25 | GQA-aware Batched Paged Attention (§11 확장) | [25_gqa_batched_attention.md](../NinjaGap_Todo/25_gqa_batched_attention.md) |

## 강등 (2026-04-20)

| § | 기법 | 강등 사유 | 문서 |
|---|---|---|---|
| §18 | Spec Decode CPU Drafter (DuoDecoding) | DuoDecoding 2× 는 **GPU drafter** 실측, CPU drafter balance 조건 (경로 1 batch scaling) 미충족. CPU baseline 통과 후 재평가 | [18_spec_decode_cpu_drafter.md](../NinjaGap_Todo/18_spec_decode_cpu_drafter.md) |

## 장거리 (context 의존, 현재 workload 128/128 기여 제한적)

| § | 기법 | 문서 |
|---|---|---|
| §19 | P/D Disaggregation | [19_pd_disaggregation.md](../NinjaGap_Todo/19_pd_disaggregation.md) |
| §20 | KV Cache CPU Tier Offload (InfiniGen) | [20_kv_offload.md](../NinjaGap_Todo/20_kv_offload.md) |
| §21 | ScoutAttention Layer-Ahead | [21_scout_attention.md](../NinjaGap_Todo/21_scout_attention.md) |

## 기각 (확정, 재시도 없음)

| § | 기법 | 기각 사유 | 측정 데이터 |
|---|---|---|---|
| §03 Phase 2 | 1GB hugetlb | SPR TLB 구조상 역효과 +22% (2026-04-19) | — |
| §04 | IPEX WoQ INT8 | vLLM Linear 클래스 비호환, §23 편입 (2026-04-19) | — |
| §06-1 v2 | VNNI `vpdpbusd` intrinsic + compensation | half-tile waste + compensation overhead, v1 대비 −7~−13% | `../measurement_results/H100x8/g0_06_1_qwen2.5_32b_v2(fail)/` |
| §11 Phase 1 | IPEX 우회 + batch16 dispatch | remainder path 이득 없음 + prefill SDPA 오버헤드, §06-1 v1 대비 −12~−5% | `../measurement_results/H100x8/g0_11_qwen2.5_32b_phase1(fail)/` |

---

**재평가 조건**: Tier 1 후보 4개 중 **최소 1개라도 실측 검증** (우리 환경에서 hybrid outTP 개선) 된 후. 그 이전에 이 backlog 에서 항목을 본 작업축으로 끌어올리지 않음.

SSOT: `../Tech_done.md` v8 §SSOT-3 (원인 트리).
