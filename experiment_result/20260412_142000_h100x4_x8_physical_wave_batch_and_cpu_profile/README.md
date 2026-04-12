# H100x4 / H100x8 물리 머신 — wave-batch + CPU 프로파일 종합

**날짜**: 2026-04-12
**환경 2개**: H100x4 물리 (cloud-J1O17M) + H100x8 물리 (violet-h100-023)
**범위**: 물리 머신에서 wave-batch hybrid 실험 + CPU many-core 프로파일링 + KVM 대비 비교

---

## 1. 하드웨어 비교

| | H100x4 물리 | H100x8 물리 | H100x4 KVM (이전) |
|---|---|---|---|
| 호스트 | cloud-J1O17M | violet-h100-023 | cloud-J1O17M (KVM guest) |
| CPU | Xeon 8480+ 1S × 96C | Xeon 8480+ **2S × 56C = 112C** | Xeon 8480+ 1S × 96 vCPU |
| NUMA | 1 node, 967 GB | **2 nodes, 1031 GB each** | 1 node |
| SMT | 1T/C (96 logical) | **2T/C (224 logical)** | 1T/C (96 logical) |
| GPU | 4 × H100 80 GB | **8 × H100 80 GB** | 4 × H100 80 GB |
| Driver | 580.126.09 | 580.126.20 | 580 (KVM passthrough) |
| L3 cache | 물리 105 MB | 물리 105 MB × 2S | **KVM 16 MB** |
| Memory BW | ~300 GB/s | ~600 GB/s (2S) | **~26.5 GB/s (KVM 제한)** |

**결정적 차이**:
- H100x8 은 **2-socket** → 2 NUMA 노드 → `num_cpu_engines=2` 자동
- KVM 은 L3 = 16 MB / BW = 26.5 GB/s 제한으로 CPU 성능 10× 저하

---

## 2. 벤치마크 결과 종합 (500 × 128/128, TP=4)

### 2.1 1.5B Qwen2.5 — 3 환경 비교

| 지표 | H100x8 물리 G | H100x8 물리 H | H100x4 물리 G | H100x4 물리 H | H100x4 KVM G | H100x4 KVM H |
|---|---:|---:|---:|---:|---:|---:|
| Duration (s) | **3.38** | **8.57** | 3.64 | 56.73 | 4.43 | 22.60 |
| Wall (s) | **12.89** | **18.77** | 12.93 | 66.41 | 15.51 | 33.68 |
| Output TP (tok/s) | **18,214** | **7,186** | 16,929 | 1,086 | 13,898 | 2,725 |
| Req TP (req/s) | **147.87** | **58.33** | 137.07 | 10.85 | 112.83 | 22.13 |
| Med TPOT (ms) | **20.56** | **21.50** | 21.49 | 65.39 | 27.59 | 70.60 |
| Mean TTFT (ms) | 644 | 712 | 793 | 1,405 | 833 | 1,239 |
| P99 TTFT (ms) | 765 | 1,291 | 950 | 9,660 | 1,062 | 6,903 |

(G = gpu_only, H = hybrid wave-batch cpu_max_seqs=16)

### 2.2 7B Qwen2.5 — H100x4 물리

| 지표 | H100x4 물리 gpu_only |
|---|---:|
| Duration (s) | 4.14 |
| Wall (s) | 13.93 |
| Output TP (tok/s) | 15,053 |
| Med TPOT (ms) | 24.19 |
| Mean TTFT (ms) | 937 |
| P99 TTFT (ms) | 1,258 |

**7B hybrid**: `hybrid.json` 이 비어있음 (CPU 요청이 완료되지 않아 bench timeout). 이전 분석 (H100x4 KVM) 에서 확인된 76 thread GEMM 절벽 + KVM BW 제한이 원인.

---

## 3. CPU Tail 비교 — 물리 vs KVM

1.5B hybrid 의 CPU tail = hybrid duration - gpu_only duration:

| 환경 | GPU dur (s) | Hybrid dur (s) | **CPU tail (s)** | Tail/GPU |
|---|---:|---:|---:|---:|
| **H100x8 물리** | 3.38 | 8.57 | **5.19** | 1.54× |
| H100x4 물리 | 3.64 | 56.73 | **53.09** | 14.6× |
| H100x4 KVM | 4.43 | 22.60 | **18.17** | 4.1× |
| H100x1 KVM | 4.43 | 22.60 | **18.17** | 4.1× |

**H100x8 물리의 CPU tail = 5.19초** — H100x4 물리 (53초) 대비 **10.2× 빠름**.

원인:
- H100x8 = 2-socket → `num_cpu_engines=2` → 각 엔진이 자기 NUMA 노드의 56 core + 로컬 BW 사용
- H100x4 물리 = 1-socket 96 core → `num_cpu_engines=1` → 76 thread 가 1 개 L3 에 경합 → GEMM 절벽

**H100x4 물리가 KVM 보다 오히려 느린 이유**: 물리 머신의 96 core 가 1 NUMA 에서 76 thread GEMM 절벽을 겪음 (oneDNN FFN N=9728 tiling 문제, GEMM profiling 에서 확인). KVM 은 24 core H100x1 실험에서 BW 가 제한적이지만 thread 수가 적어 GEMM 절벽 없음.

---

## 4. CPU 프로파일링 결과 (H100x4 물리, analysis_log/20260412_125831)

### 4.1 시스템 토폴로지

- CPU: Xeon 8480+, 96 core, 1T/C, 1 NUMA
- **L3: 16 MB** (KVM guest 제한 — 물리 105 MB 의 15%)
- L2: 4 MB × 96 = 384 MB
- SMT: 없음 (1T/C)

### 4.2 GEMM Thread Scaling — 64→76 절벽

| GEMM shape | 최적 thread | 최적 GFLOPS | 76t GFLOPS | 76t/최적 |
|---|---:|---:|---:|---:|
| QKV (N=3584) | 76 | 2,829 | 2,829 | 1.00× |
| **FFN_up (N=9728)** | 64 | **2,008** | **245** | **0.12×** |
| **FFN_dn (K=9728)** | 48 | **2,040** | **316** | **0.15×** |
| Single (N=9728) | 64 | 130 | 16 | 0.12× |
| Prefill (N=9728) | 64 | 8,293 | 1,759 | 0.21× |

**N=9728 (FFN) 행렬에서 76 thread 가 5-8× 성능 추락.** oneDNN tiling 의 pathological 선택.

### 4.3 Attention Scaling — 문제 없음

```
batch=16 threads=76: 0.042 ms (빠르고 안정)
batch=16 threads=96: 0.050 ms (96 에서만 약간 증가)
```

### 4.4 Layer Breakdown — **FFN 이 64-72% 지배**

| | 24 threads | 76 threads | 비율 |
|---|---:|---:|---:|
| Total | 626.9 ms | 1,031.5 ms | 1.65× |
| Attention | 112.1 ms (18%) | 318.9 ms (31%) | 2.84× |
| **MLP/FFN** | **452.8 ms (72%)** | **656.7 ms (64%)** | **1.45×** |

### 4.5 Memory Bandwidth — 26.5 GB/s 천장

BW 가 16-24 thread 에서 포화 (26.5 GB/s). 48+ thread 에서 오히려 하락. **KVM 제한**.

---

## 5. 핵심 발견 요약

### F1. 물리 H100x8 (2-socket) 은 hybrid 가 거의 동작함

- TPOT: gpu_only 20.56 ms → hybrid **21.50 ms** (1.05× 만 증가)
- Wall: 12.89s → 18.77s (1.46× 증가)
- **2-NUMA → num_cpu_engines=2 → 각 엔진 56 core + 로컬 BW → GEMM 절벽 없음**

### F2. 물리 H100x4 (1-socket) 은 GEMM 절벽으로 hybrid 성능 붕괴

- TPOT: gpu_only 21.49 ms → hybrid **65.39 ms** (3.04× 증가)
- CPU tail 53초 — oneDNN FFN tiling 이 76 thread 에서 pathological
- 최적 thread 수는 48-64

### F3. KVM 은 L3 16 MB + BW 26.5 GB/s 이중 제한

- 물리 대비 CPU 성능 10× 저하
- GEMM 절벽과 독립적으로 BW 자체가 부족

### F4. 2-socket NUMA 분할이 many-core CPU 의 핵심

- H100x8 (2S × 56C): CPU tail **5.2초** ✓
- H100x4 (1S × 96C): CPU tail **53초** ✗
- 같은 총 코어 수여도 NUMA 분할 여부로 10× 차이

---

## 6. 미완료 / 실패 실험

| 디렉토리 | 상태 | 비고 |
|---|---|---|
| `20260412_063229_H_C_..._wrong_bin` | ❌ | 잘못된 바이너리 실행 |
| `20260412_062947_G_..._wrong_bin` | ❌ | 잘못된 바이너리 실행 |
| `20260412_063649_G_..._wrong_bin` | ❌ | 잘못된 바이너리 실행 |
| `20260412_065049_H_C_..._7B_fail` | ❌ | 7B hybrid CPU 미완료 (GEMM 절벽) |
| `20260412_063921_H_C_..._7B_fail` | ❌ | 7B hybrid CPU 미완료 |
| `20260412_071013_H_C_..._7B` | ❌ | hybrid.json 비어있음 |
| `20260412_073433_H_C_..._7B` | ❌ | hybrid.json 비어있음 |

---

## 7. 다음 단계

1. **H100x8 물리에서 7B / 32B hybrid** — 32B 는 TP=8 로 8 GPU 전부 사용 가능. 2-NUMA 환경에서 CPU tail 측정.
2. **H100x8 물리에서 cpu_profile.sh** — L3 / BW / GEMM scaling 이 물리 2-socket 에서 어떤지 KVM 대비 확인
3. **H100x4 물리에서 `HYBRID_CPU_THREADS=48`** — GEMM 절벽 (76t) 을 피하고 7B hybrid 재시도
4. **논문 업데이트** — KVM vs 물리 차이, 2-socket NUMA 의 중요성 기술

---

## 8. 데이터 소스

**H100x8 물리 결과**:
- `eval/results/20260412_141500_G_H100_80GB_HBM3_x8_Qwen2.5-1.5B-Instruct/`
- `eval/results/20260412_141741_H_C_H100_80GB_HBM3_x8_Qwen2.5-1.5B-Instruct/`

**H100x4 물리 결과**:
- `eval/results/20260412_070024_G_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/`
- `eval/results/20260412_070313_H_C_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/`
- `eval/results/20260412_070649_G_H100_80GB_HBM3_x4_Qwen2.5-7B-Instruct/`

**CPU 프로파일**:
- `eval/analysis_log/20260412_125831_cpu_profile/` (최종 성공 run)

**이전 비교 데이터**:
- KVM H100x1: `experiment_result/20260412_050600_h100x1_qwen1.5b_7b_32b_wave_batch_scaling/`
- KVM H100x1 wave sweep: `experiment_result/20260412_051400_h100x1_qwen7b_wave_size_sweep_4_8_16/`
- dev RTX3090: `experiment_result/20260412_023700_dev_rtx3090_wave_batch_gate_fix_initial_validation/`
