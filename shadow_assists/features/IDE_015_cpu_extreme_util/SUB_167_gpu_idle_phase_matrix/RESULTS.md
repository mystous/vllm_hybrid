# SUB_167 — GPU 20pp idle window phase-별 categorize matrix (paper Table 1a)

> **parent**: IDE_015 / TSK_022 (원 SUB_102) — 분석 SUB
> **scope**: 2026-05-26 (analysis only, no new measurement)
> **status**: ✅ 완료 — SUB_161 (py-spy) + SUB_117 (per-worker util) + SUB_098 (baseline) 데이터 결합 → phase matrix
> **note**: nsys 부재로 GPU kernel timeline 직접 측정 불가. CPU-side proxy (py-spy) + system-wide GPU util 로 indirect 추정

---

## 0. 두괄식 — phase 별 GPU idle window 추정 matrix

### Paper Table 1a 후보 (canonical Qwen 32B TP=4×2 AGSD-gated balanced)

| Phase | GPU active % | CPU active % | idle gap (GPU) | CPU task 가능 작업 |
|---|---:|---:|---:|---|
| **attention** (memory-bound) | ~50-70% | < 5% (idle) | **~30-50%** | scheduling / detokenize / grammar check |
| **linear matmul** (compute-bound) | ~85-95% | ~ sampler.py 44% | ~10% | KV prefetch / draft head / cold-KV decompress |
| **sampling + post-process** | ~10-20% | **sampler.py 44.3% (SUB_161)** | **~80%** ⭐ | tokenize next batch / AGSD route decision / metadata assemble |
| **TP all-reduce** (NCCL) | ~50% | communication_op 19% | ~50% | logits processor pre-compute |
| **inter-step idle (queue wait)** | 0% | EngineCore 0.2% R (SUB_162) | **~100%** ⭐⭐ | **모든 CPU task 가능** |

→ **largest idle window = sampling phase + inter-step idle**. 두 영역이 IDE_018 phase-burst 의 main fill target.

---

## 1. Source 데이터

### 1.1 GPU util (SUB_160 30-min sustained, N=32 fill 시)

| GPU | role | avg util | max | active 추정 |
|---:|---|---:|---:|---|
| 0-3 (vanilla NUMA0) | vanilla backend | 65-71% | 100% | linear-heavy |
| 4-7 (trident NUMA1) | trident backend | 25-28% | 91-94% | sampling-heavy + spec |
| **8-GPU avg** | — | **47.6%** | — | — |

→ trident backend (GPU 4-7) 의 25-28% util — spec decoding 의 lightweight verify phase + sampling overhead 가 큰 비중.

### 1.2 CPU-side phase distribution (SUB_161 py-spy, trident TP0, 40s)

| Phase (CPU file) | sample share | 의미 |
|---|---:|---|
| sampler.py | **44.3%** | sampling + top-k/top-p |
| gpu_model_runner.py | 79.5% | model dispatcher |
| logits_processor.py | 27.0% | logit bias/temp/penalty |
| penalties.py | 23.4% | penalty ops |
| qwen2.py | 27.7% | model layers (attention + linear) |
| communication_op.py | 19.0% | NCCL all-reduce |
| update_async_output_token_ids | 17.6% | output post-processing |
| IPC dequeue + wait | ~25% | worker idle wait |

### 1.3 CPU thread state (SUB_162, vllm 30 thread, 60s)

- **VLLM Worker_TP**: 1.0% R / 96.0% S
- **VLLM EngineCore**: 0.2% R / 99.8% S
- avg 5.04 running threads / sample

---

## 2. Phase categorize 정량 (paper Table 1a 본 matrix)

| Phase | duration estimate | GPU active | CPU active | **idle gap** | dominant CPU work |
|---|---:|---:|---:|---:|---|
| pre-step (dequeue + schedule) | 5-10 ms | 0% | dequeue 25% | **100%** | metadata assemble |
| attention forward | 5-15 ms | 60-80% | qwen2.py 27% | 30-50% | detokenize (TSK_024) |
| linear matmul + sampling | 8-20 ms | 70-90% (compute), 15-25% (sample) | sampler 44%, logits 27% | 10-80% | CPU sampling rewrite (TSK_025) |
| TP all-reduce | 2-5 ms | 50% (network-bound) | communication_op 19% | 50% | logit pre-compute, schedule next |
| post-step (output token id) | 1-3 ms | 0% | update_async 17.6% | **100%** | tokenize, grammar check |

→ **per-step total ~25-50 ms** (각 phase 의 합). 가장 큰 idle gap 은 pre-step + post-step = ~30% 의 step time.

---

## 3. paper Figure 후보 — phase-별 throughput lift 잠재력

| Phase | 현재 lift 가능 잠재력 | IDE_018 main task |
|---|---|---|
| sampling phase | sampler 44% CPU 자체가 lever → AVX-512 sampling → ~30-40% latency 감소 가능 | TSK_025 → IDE_018 attention-phase task |
| linear phase | KV prefetch + AMX draft head | TSK_026/030 → IDE_018 linear-phase task |
| pre/post-step idle | 30% step time available | TSK_032 attention-phase task pool (full lever) |
| TP all-reduce | logits pre-compute | TSK_032 |

→ **expected lift estimate**: sampling rewrite (TSK_025) alone +5-10% / phase-burst integration (TSK_034) +10-20% / total (Phase 2+3) +15-30% throughput Δ 예상.

---

## 4. nsys 부재의 한계 + 본 SUB 의 confidence

| 측정 | confidence |
|---|---|
| GPU util per-GPU avg (full duration) | **high** (1Hz nvidia-smi sampling) |
| CPU-side phase distribution (file-level) | **medium-high** (py-spy 100 Hz × 40s) |
| **kernel-level GPU timing** (attention vs linear 정량 분리) | **low** — nsys 부재 |
| **per-step sub-ms timing** (single decode step 의 phase 별 ms) | **low** — nsys 부재 |

→ paper main figure 시 nsys 환경에서 후속 measurement 권장. 본 matrix 는 **first-order estimate** 로 IDE_018 design 입력 용.

---

## 5. 다음 step

- (별도 turn — kernel dev) TSK_031 CUDA event hook 으로 per-step phase timing 정량 — 본 matrix 의 **GPU 측 정확한 값** 확정
- (별도 turn — 가능 env) nsys 가능 환경에서 SUB_161 protocol 재실행 → per-kernel timing 캡처
- SUB_168 (다음 SUB) — CPU task candidate matrix (paper Table 1b)
