# HWC Round 2 — KV fp8 stack 5-sweep

## Context

- Round 1 결과 H8 (KV fp8) = +4.02% ± 0.61% (5 sweep, 22,966 vs 22,078).
- Round 2 는 fp8 위에 다른 lever 5개 결합으로 +10% 이상 도달 목표.

## Levers

| # | Lever | Stack on fp8 | mean tps | Δ% vs base | Δ% vs fp8 | verdict |
|---|---|---|---|---|---|---|
| H8 (R1 ref) | KV fp8 | — | 22965.6 ± 139.4 | +4.02% | 0% | accept gate ↑ |
| R2-4 | + pass_config.fuse_norm_quant+act_quant+attn_quant | 22399.8 ± 216.8 | +1.46% | -2.46% | 음수 → 기각 |
| R2-5 | + pass_config.enable_sp + fuse_gemm_comms | 23015.8 ± 235.6 | **+4.25%** | +0.22% | best 동등 |
| R2-6 | + attention_config.use_prefill_query_quantization | 22934.9 ± 363.5 | +3.88% | -0.14% | 동등 |
| R2-3 | + VLLM_ATTENTION_BACKEND=TRITON_ATTN (1 sweep) | 22837.5 | +3.44% | -0.56% | 동등 (5sw 미진행) |
| R2-1b | KV fp8_e5m2 (대안 fp8 variant) | — | (진행 중) | TBD | TBD |

## 발견

1. **fp8 단독** 이 winner — additional lever 들 stacking 효과 없음 또는 약간 음수.
2. **enable_sp (R2-5)** 가 fp8 단독과 std 안에 동등 (+0.22%) — Llama-8B 의 작은 RMSNorm 으로 SP 의 통신 절감 효과 미미.
3. **fuse_norm_quant (R2-4)** 가 음수 — RMSNorm 자체가 충분히 작아서 fusion 오버헤드가 더 큼.
4. **use_prefill_query_quantization (R2-6)** 가 효과 없음 — prefill phase 가 전체 work 의 작은 부분.
5. **TRITON_ATTN (R2-3)** 가 FlashInfer 와 거의 동등 — TRITON 의 fp8 attention kernel 도 sm_100 에서 좋음.

## Round 3 plan

R2-5 (fp8 + enable_sp) = +4.25% 를 base 로:
- R3-A: + async-scheduling — host overlap
- R3-B: + max-num-batched-tokens=16384 — larger batch
- R3-C: + cudagraph FULL_DECODE_ONLY — graph capture overhead 줄임
- R3-D: + max-num-seqs=256 — 더 많은 concurrent seqs

각 5-sweep, paired Δ% vs baseline 측정. winner 가 +10% 도달하면 multi-model + accuracy gate.
