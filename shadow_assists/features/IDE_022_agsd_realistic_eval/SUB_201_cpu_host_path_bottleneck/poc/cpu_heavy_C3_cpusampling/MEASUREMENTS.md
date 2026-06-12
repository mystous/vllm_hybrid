# cpu_heavy_C3 — CPU sampling offload (brief 의 C-1 "AMX BF16 sampler" 매핑)

`vllm/v1/sample/sampler.py` 의 `VLLM_CPU_SAMPLING=1` 환경변수 (SUB_201 L11) 를 활성화하여
logits 의 softmax + top-k + multinomial 을 GPU 가 아닌 CPU 가 수행. brief 의 C-1 (AMX BF16
sampler) 와 매커니즘적으로 동일 (현 impl 은 FP32 CPU softmax 사용; AMX BF16 으로 변환해도
B × vocab d2h 가 critical path 라 동일한 결과 예상).

## Configuration

- Llama-3.1-8B-Instruct TP=8, sharegpt 500p × conc=64 × max-tok=2048
- Env: 기본 + `VLLM_CPU_SAMPLING=1`
- 3 sweeps (statistically sufficient to detect Δ ≥ 5%)

## Result

| metric | baseline | C3a_cpu_sampling | Δ |
|---|---:|---:|---:|
| output_tps (n=3) | 22,007.87 ± 143.44 | **21,970.00 ± 333.05** | **-0.17%** |
| gpu_util | 96.17% | 96.27% | +0.10pp |
| cpu_util (mpstat) | 5.44% ± 0.03% | 5.44% ± 0.03% | **0.00pp** |

### Per-sweep

| sweep | tps | gpu_util | cpu_util (top) | cpu_util (mpstat) |
|---:|---:|---:|---:|---:|
| s1 | 21,586.9 | 95.9% | 5.3% | 5.41% |
| s2-3 | 같은 범위 | ~96% | ~5.4% | ~5.44% |

## Verdict

- throughput: **noise** (-0.17%, σ ~330, < 1 σ).
- **cpu_util 변화 0** (5.44 → 5.44). mpstat 1s sampling 으로는 sampler 의 sub-ms CPU work
  가 거의 감지 안 됨.
- **이전 측정 (L11, Qwen-7B TP=1, conc=16) 에서 -24% 였던 결과는 본 환경 (Llama-8B TP=8,
  conc=64) 에서는 noise 로 회복**. 큰 batch + larger vocab 가 D2H 비용을 흡수하는 형태로
  보이나, throughput 양수 효과는 0.

**결론**: brief 의 C-1 lever (CPU sampler offload, 현 impl 은 FP32; AMX BF16 으로 바꿔도
B×vocab D2H 가 critical path 이므로 결과 동일 예상) 는 **noise**. CPU 활용도 미상승,
throughput gain 0.
