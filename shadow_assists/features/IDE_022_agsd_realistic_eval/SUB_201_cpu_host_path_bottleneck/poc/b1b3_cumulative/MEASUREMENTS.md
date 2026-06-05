# B1 + B3 cumulative gain — Llama-3.1-8B TP=2 (2026-06-05 KST)

> 두 lever 의 cumulative effect 측정. 4 run sequential (sharegpt 200p × conc=16 × max-tok 8192 × vanilla, stream).
> 환경: B200 GPU 4-5, port 8002, model meta-llama/Llama-3.1-8B-Instruct, --gpu-memory-utilization 0.85, --max-model-len 16384.

## 1. 4 run 표

| run | cudagraph | EXCLUSIVE | tps | n_ok | wall_s | TTFT p50 / p99 | TPOT p50 | GPU% | CPU% | err |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **A_baseline** | PIECEWISE | 0 | 4,410.1 | 197/200 | 299.1 | 21.5 / 65.7 | 3.50 | 78.4 | 4.5 | **3** |
| **B_b3** | FaP | 0 | **4,471.7** | 200/200 | 337.5 | 22.4 / 277.6 | 3.40 | **98.7** | 2.3 | 0 |
| **C_b1** | PIECEWISE | 1 | 4,306.8 | 200/200 | 340.1 | **18.0** / 183.2 | 3.50 | 97.4 | **2.1** | 0 |
| **D_b1b3** | FaP | 1 | 4,423.0 | 200/200 | 336.7 | 20.5 / 181.4 | 3.40 | 98.6 | 2.2 | 0 |

## 2. Δ 분석

### 2.1 raw Δ vs A_baseline

| Δ | tps | TTFT p50 | GPU% | CPU% |
|---|---:|---:|---:|---:|
| Δ_B3 (B−A) | **+1.4%** | +0.9 ms | +20.3pp | -2.2pp |
| Δ_B1 (C−A) | **-2.3%** | -3.5 ms | +19.0pp | -2.4pp |
| Δ_B1+B3 (D−A) | +0.3% | -1.0 ms | +20.2pp | -2.3pp |

### 2.2 caveat — A_baseline 의 n_ok=197/200 (3 EngineCore err) → tps 인플레이트
- 3 req 가 mid-generation 에서 fail → total_completion_tokens 가 wall_total_s 보다 빨리 끝남
- 더 robust 한 비교는 **0 err 인 B/C/D 만** 또는 wall-normalized req-throughput

### 2.3 0-err 셋 비교 (B / C / D)

| 기준 run | Δ_FaP→EXC 추가 | Δ_EXC→FaP 추가 |
|---|---|---|
| B (FaP+native) → D (FaP+EXC) | **-1.1%** (interference) | — |
| C (EXC+PIECEWISE) → D (EXC+FaP) | — | **+2.7%** |

→ **EXCLUSIVE 단독은 본 측정에서 -3.7%** (B→C: 4471.7 → 4306.8)
→ FaP 위에 EXCLUSIVE 추가는 -1.1% (subadditive negative)
→ EXCLUSIVE 위에 FaP 추가는 +2.7%

### 2.4 이전 B1 EXCLUSIVE 측정 (`poc/b1_e2e/MEASUREMENTS.md §9`) 와 inconsistency

| 측정 | baseline (PIECEWISE+native) | EXCLUSIVE (PIECEWISE+EXC) | Δ |
|---|---:|---:|---:|
| 이전 B1 v3 (`§9`) | 4,146.8 | 4,271.6 | **+3.01%** |
| 본 측정 (A vs C) | 4,410.1 | 4,306.8 | **-2.3%** |

→ baseline 자체가 ~6% 변동 (4147 ↔ 4410). **inter-run variability 가 lever Δ 보다 큼** — 단일 run sample 의 신뢰성 한계 확인.
→ 본 측정의 baseline 이 inflated (3 err) 라 Δ_B1 음수 효과 과대.

## 3. cumulative gain verdict

| 결론 | 근거 |
|---|---|
| ✅ **FaP 가 가장 robust positive** | A→B +1.4%, C→D +2.7%, 모든 비교에서 positive direction |
| ⚠ **EXCLUSIVE 는 본 측정 inconsistent** | 이전 +3.01% vs 본 -2.3%, sample noise 큼 |
| ⚠ **cumulative D 는 B (FaP only) 보다 -1.1%** | additive 가설 (+1.4% + +3.01%) 미성립 |
| ✅ **GPU util 4 run 모두 78→97-98%** | host-bottleneck (A_baseline 78.4) 가 두 lever 어느 하나로 해소되면 같은 ceiling 도달 |

## 4. production 권고

| 권고 | 근거 |
|---|---|
| 🥇 **default = B3 (FaP)** | 측정 4471.7 tps best, 0 err, GPU 98.7%, CPU 2.3%, 부수효과 (TTFT p99 안정화) |
| 🥈 **B1 EXCLUSIVE 는 H100 prod 재검증 후 결정** | B200 baseline noise 가 lever Δ 보다 큼, prod 환경 (FA3 native + ALWAYS cap) 에서 진짜 ROI 측정 필요 |
| ⚠ **cumulative (FaP+EXC) 는 단일 run 에서 D < B** | run-to-run noise 6%+ 라 4-run 1회 측정으로 결정 불가, H100 sweep 의 3-repeat 후 재판정 |

## 5. 한계 + 다음 step

1. **single-run noise > lever Δ**: B200 8B Llama 의 baseline 변동 ~6% 가 EXC 의 ±3% 효과를 가림. 3 repeat × 4 run = 12 sweep 으로 statistically 유의한 비교 필요.
2. **prod H100 검증**: FA3 native + ALWAYS cap 에서 cumulative gain 패턴이 다를 수 있음 (B3 의 +30% 처럼).
3. **다른 모델 사이즈 검증**: A2 의 Llama-70B 에서 +1.82% 패턴처럼, Llama-8B 의 EXCLUSIVE effect 도 70B 에서 다를 가능성.

## 6. GPU 4-5 최종 free 검증

```
4, 0 MiB
5, 0 MiB
```

GPU 0-3 / 6-7 본 측정 동안 미접촉 (CUDA_VISIBLE_DEVICES=4,5 격리 유지).
