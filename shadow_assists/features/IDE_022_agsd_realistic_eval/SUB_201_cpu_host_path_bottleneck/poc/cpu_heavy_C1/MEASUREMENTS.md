# cpu_heavy_C1 — NGram spec decode (CPU heavy 시도) measurements

## Baseline (재측정)

`shadow_assists/.../cpu_heavy_baseline/summary.json`:

| metric | value |
|---|---|
| output_tps | 22,007.87 ± 143.44 |
| gpu_util | 96.17% |
| cpu_util (mpstat) | **5.44% ± 0.03%** |

## C-1a — ngram K=3, single CPU thread (vLLM 기본), no precompute

NGram proposer 기본값 (NUM_THREADS_CAP=1, PRECOMPUTE=0). 검증용 — single-thread overhead floor.

| sweep | output_tps | gpu_util | cpu_util (top) | cpu_util (mpstat) | accept_rate α |
|---:|---:|---:|---:|---:|---:|
| 1 | 16,903.3 | 66.6% | 4.5% | 4.56% | 0.7131 |
| 2 | 17,327.1 | 68.0% | 4.5% | 4.58% | 0.7087 |
| 3 | 17,020.1 | 68.3% | 4.5% | 4.54% | 0.7079 |

- **mean ≈ 17,083.5 tps** → **Δ vs baseline = -22.4%**
- accept_rate 71% 매우 양호 (대부분 prompt의 prefix가 repeat — sharegpt 특성)
- 그러나 cpu_util 5.44% → 4.56% **오히려 감소** (GPU 일이 늘어 CPU가 더 wait)
- gpu_util 96% → 67% (verify batch 크기 증가로 step 효율 감소)

**Verdict**: REJECT. spec decoding 자체가 (이미 GPU bound인) baseline에 대해 throughput penalty 부과. CPU 활용 lever 의 목적과 정반대.

## C-1b/c — multi-thread 변형 (CANCELLED)

C-1a 가 -22.4% 로 통계적으로 명확한 negative 였고, multi-thread (32 CPU threads + precompute) 도 GPU forward latency 가 critical path 라 throughput 같은 수준일 가능성이 매우 높음. 25-30 분의 측정 cost 회피를 위해 cancel.

근거: NGram proposer 의 CPU 일 (numba batch_propose 호출) 은 미리 측정된 결과로도 step 당 ~수십 μs 수준. 따라서 thread cap 늘려도 critical path 가 아니라 CPU util 만 살짝 증가 + throughput 동일.

## C-1 결론

NGram spec decoding 은 **CPU 활용도 lever 가 아니라 GPU verify 의 batch size 를 K 배로 증폭시키는 lever**. baseline 이 이미 GPU bound (96%) 인 환경에서는 wallclock 증가가 spec speedup 보다 커서 net-negative. 이전 50+ lever fail 과 동일 매커니즘.

Brief 의 C-1 (AMX BF16 sampler) 와는 매커니즘이 다르므로 C-1 결론 자체는 본 결과로 close 되지 않으나, 본 round 에서는 **NGram spec decode 가 winning lever 아님** 을 확인.
