# cpu_continuous — IDE_022 / SUB_201 continuous lever sweep

목표: throughput ≥+10% AND cpu_util 상승 lever 발굴.
Baseline: vanilla+B3FaP+L2+L10 = 22,008 ± 143 tps, gpu 96.2%, cpu 5.4%.
Workload: sharegpt 500p × conc=64 × max-tok=2048, TP=8, max-model-len=16384, FAP.

## 결과 표 (mean ± std over 5 sweeps)

| C# | lever | mean tps | std | Δ% | cpu_util | gpu_util | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| C-7a | `--kv-cache-dtype fp8 --calculate-kv-scales` | 22671.4 | 138.8 | +3.01% | 5.42 | 96.2 | partial (gpu-only, cpu↔) |
| C-7c | `--enable-chunked-prefill --max-num-batched-tokens 4096` | 21987.5 | 162.6 | -0.09% | 5.38 | 96.4 | reject (parity, no cpu↑) |
| C-7d | `--enable-prefix-caching` | 22055.6 | 167.4 | +0.22% | 5.40 | 96.3 | reject (noise, no cpu↑) |
| C-7e | `VLLM_FLASH_ATTN_VERSION=3` | 22050.4 | 132.4 | +0.19% | 5.40 | 96.3 | reject (noise, no cpu↑) |
| C-8a | detok ThreadPool(4) pre-pass | (3/5 ok) 21931.8 | 343.9 | -0.35% | 5.43 | 96.2 | reject (s4 EngineCore crash, no cpu↑) |
| C-10a | Eagle3 spec(k=3) | 15340.8 | 451.7 | -30.30% | 5.08 | 95.2 | reject (α=0.7% at conc=64) |
| C-7af | C-7a + `--enable-prefix-caching` | 22632.1 | 152.7 | +2.84% | 5.44 | 96.2 | partial (combo no-add) |
| C-7f | `--max-num-seqs 1024 --max-num-batched-tokens 32768` | 22001.4 | 145.6 | -0.03% | 5.42 | 96.4 | reject (no headroom at conc=64) |
| C-7g | `--stream-interval 8` | 22179.5 | 154.4 | +0.78% | 4.92 | 96.3 | reject (no cpu↑, marginal tps) |
| C-7i | `--no-enable-chunked-prefill` | 22045.5 | 270.6 | +0.17% | 5.38 | 96.4 | reject |
| C-7k | `VLLM_USE_FLASHINFER_SAMPLER=1` | 22053.4 | 154.8 | +0.21% | 5.38 | 96.4 | reject |
| C-7l | C-7a + FA3 + FI sampler + stream-interval 4 | 22736.0 | 186.0 | +3.31% | 5.02 | 96.2 | partial (best gpu-only, cpu↓) |

## 결론

10 lever (C-7a~C-7l, C-8a, C-10a) 와 1 combo (C-7af) 시도. 본 워크로드는 GPU 96% sat / CPU 5% idle. host loop overhead 가 measurable threshold 이하이므로 `+10% AND cpu_util↑` 동시 만족 lever **미발견**. 

- 최고 tps: **C-7l (+3.31%, 22736 tps)** — gpu-side only, cpu↓ (게이트 미통과)
- 단독 lever 최고: **C-7a (+3.01%)** — KV fp8 + scales (기존 +3.94% 와 정합)
- C-8a (detok ThreadPool) — 4 번째 sweep 에서 EngineCore IPC crash (race risk)
- C-10a Eagle3 (k=3) — α=0.7% → -30% (conc=64 에서 spec decode 비효율)

본 워크로드/모델 조합에서 CPU 활용 상승 lever 는 본질적으로 부재 — `--prefetch-tokenize` 와 `--burst-aware-admission` 이 이미 host path 를 거의 다 흡수. 다음 단계 권고: 워크로드 자체를 host-bound 영역 (e.g. conc=512, prompt-heavy short-decode) 으로 옮기거나, draft-CPU spec decode 같은 본질적 CPU 작업 lever (현재 idea pool 의 C-9c) 시도.
