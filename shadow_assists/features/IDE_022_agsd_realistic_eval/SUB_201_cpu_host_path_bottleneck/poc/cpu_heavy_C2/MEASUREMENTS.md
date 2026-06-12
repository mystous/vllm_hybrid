# cpu_heavy_C2 — fp8 KV (re-confirmation, mpstat CPU monitor)

이전 measurement (#48/49/50) 가 식별한 유일 양수 lever (`--kv-cache-dtype fp8` +4.02%) 의
재현 + mpstat 으로 CPU util 변동 직접 측정.

## Configuration

- Llama-3.1-8B-Instruct TP=8, sharegpt 500 prompts × conc=64 × max-tok=2048
- Base env: `VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1`
- Base CLI: `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`
- + `--kv-cache-dtype fp8`

## Result

| metric | baseline | C2_fp8_kv | Δ |
|---|---:|---:|---:|
| output_tps (n=5) | 22,007.87 ± 143.44 | **22,874.34 ± 378.57** | **+3.93%** |
| gpu_util | 96.17% | 95.86% | -0.31pp |
| cpu_util (mpstat) | 5.44% ± 0.03% | 5.47% ± 0.03% | **+0.03pp (≈0)** |

### Per-sweep (raw, sharegpt mix)

| sweep | tps | gpu_util | cpu_util (top) | cpu_util (mpstat) |
|---:|---:|---:|---:|---:|
| s1 | 22,216.3 | 96.0% | 5.3% | 5.44% |
| s2 | 23,068.8 | 94.6% | 5.4% | 5.48% |
| s3-5 | ~ | ~ | ~ | ~ |

## Verdict

- Throughput: **+3.93%** — 이전 #50 측정 (+4.02%) 과 동일 재현. statistically significant
  (Δ/σ ≈ 866/(0.5√(143²+379²)) ≈ 4.3, p < 0.001).
- **CPU util 변화 없음 (5.44 → 5.47 %, < 0.1pp).** brief 의 "CPU 활용도 상승" gate 미통과.
- gpu_util 96 → 96 — GPU bound regime 유지. fp8 은 KV memory bandwidth 만 절감, CPU 일은
  변하지 않음.

**결론**: fp8 KV 는 throughput +3.93% gain 의 유효한 lever 이지만, brief 의 **CPU 활용 lever 가 아님**.
"≥+10% throughput + CPU 활용도 상승" 동시 조건 미충족. building block 으로만 유효.
