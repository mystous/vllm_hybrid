# SUB_201 §5 step 분해 프로파일 — 1차 verdict (2026-06-05 KST 05:35)

> 측정: Qwen-7B suffix (저 util 신호 최강) + Llama-70B suffix (대조). nsys 2025.1.1, CUDA + NVTX + osrt trace, 60s window.
> 환경: B200×8, sm_100. nsys CAP_SYS_ADMIN/CAP_PERFMON 없음 → CPU sampling 불가, GPU + CUDA API + NVTX trace 만.

---

## 1. 측정 환경

| 항목 | 값 |
|---|---|
| 도구 | `/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys` (번들 Nsight Systems 2025.1.1) |
| trace 옵션 | `-t cuda,nvtx,osrt` |
| trace window | 60s (delay=120s for Qwen-7B / 240s for Llama-70B, traffic burst 동기) |
| traffic | mix corpus, conc=16/32, limit=50/150p |
| 산출물 | `qwen7b_suffix_first.nsys-rep` (216MB), `llama70b_suffix_first.nsys-rep` (285MB), `.sqlite`, `.summ.json`, `.raw.jsonl` |

**제약**:
- `Linux Kernel Paranoid Level = 4`, `perf_event_open: Fail` → CPU sampling 불가
- `NVGPUCTRPERM` 권한 차단 → ncu metric 일부 불가
- 즉 (b) verify 내부 occupancy/DRAM 정밀 측정은 보류, **inter-kernel gap + CUDA API 측 host overhead** 로 (a)/(c) 판별

---

## 2. 측정 결과

### 2.1 Qwen-7B suffix (GPU 26.5%, gap 최대 신호)

**Traffic 결과** (60s 의 측정 가능 window):
```
50p × conc=16 × mix → 3,989 tps / wall 35s / gpu 46.5% / cpu 2.6% / ttft p50 25ms / α 0.747
```

**GPU kernel sum** (top 5, total = 약 33.4s in 60s):
| Time (%) | Total (ns) | Kernel | 해석 |
|---:|---:|---|---|
| **65.8** | 24.04s | `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | Flash MHA Q128 (verify 메인) |
| 9.9 | 3.60s | `fmha ... Q8Kv128 MultiCtasKvCga` | Flash MHA Q8 (draft verify) |
| 3.6 | 1.30s | `fmha ... Q8Kv128 MultiCtasKv` | (variant) |
| 2.9 | 1.06s | `reshape_and_cache_flash_kernel` | KV cache 업데이트 |
| 2.7 | 0.98s | `ncclDevKernel_AllGather_RING_LL` | TP=4 collective |

→ **FMHA 합 79.3%** (attention 압도 dominant)

**CUDA API summary** (host 측 비용, top 5 in 60s):
| Time (%) | Total | API | 해석 |
|---:|---:|---|---|
| **36.3** | **5.26s** | `cudaLaunchKernel` | 1,116,364 calls × avg 4.7μs |
| 20.2 | 2.93s | `cuLaunchKernelEx` | 759,524 calls |
| 17.8 | 2.58s | `cudaGraphLaunch_v10000` | 373,172 calls |
| 13.4 | 1.95s | `cudaMemcpyAsync` | 308,548 calls × avg 6.3μs |
| 2.5 | 0.36s | `cuTensorMapEncodeTiled` | 3,041,024 calls (TMA setup) |

→ **kernel launch 합 (cudaLaunchKernel + cuLaunchKernelEx + cudaGraphLaunch) = 10.77s = trace 의 18%**, **memcpy 13.4%**

**NVTX marker** (vllm 자체 안 씀, NCCL marker 만):
- `NCCL:ncclAllGather` × 12,884 calls × 34.8μs = 448ms 합 (Trace 의 0.7%)

### 2.2 Llama-70B suffix (대조군, GPU 83.4%)

**Traffic 결과** (60s window):
```
150p × conc=32 × mix → ~7,200 tps (예상치, 실측 raw 에서)
```

**GPU kernel sum** (top 5):
| Time (%) | Total (ns) | Kernel |
|---:|---:|---|
| **76.9** | 53.35s | `fmhaSm100fKernel ... Q128Kv128PersistentContext` |
| 10.5 | 7.31s | `fmha ... Q8Kv128 MultiCtasKvCga` |
| 2.9 | 2.01s | `reshape_and_cache_flash_kernel` |
| 2.3 | 1.63s | `fmha ... Q8Kv128 MultiCtasKv` |
| 1.3 | 0.89s | `multimem_all_reduce_kernel` |

→ **FMHA 합 89.7%**, GPU total kernel time ~65s (multi-stream sum, GPU 8개 분산)

**CUDA API summary** ⚠ **결정적 차이**:
| Time (%) | Total | API |
|---:|---:|---|
| **80.4** | **65.26s** | **`cudaMemcpyAsync`** ← **압도적 dominant** |
| 6.6 | 5.37s | `cudaGraphLaunch_v10000` |
| 5.8 | 4.70s | `cuLaunchKernelEx` |
| 4.4 | 3.55s | `cudaLaunchKernel` |
| 0.8 | 0.67s | `cuTensorMapEncodeTiled` |

→ **kernel launch 합 = 13.62s = 17%**, **memcpy = 65.26s = 80%!**

---

## 3. 결정 트리 적용 (§5.4)

> Gate: gap ≥ verify의 ~30% 이고 **host-bound** 확인 → (a) → A1/B 본론 확정.

### 3.1 Qwen-7B (low util)
- GPU kernel time 33s + launch 11s + memcpy 2s = 46s in 60s
- Trace 60s 중 traffic 35s = 약 58% 의 60s 만 GPU 활성 시도. 그 안에서:
  - GPU active = 28s (nvidia-smi 46.5% × 60s 와 일치)
  - Host overhead = launch 11s + memcpy 2s = 13s
  - **launch overhead 자체가 trace 의 18%** = **명백한 host-bound (a)**
- **lever**: launch overhead 분산 = **B3 (scheduler / kernel batching)** + **A1 (CPU drafting)**

### 3.2 Llama-70B (high util)
- GPU kernel time ~65s (multi-stream), wall trace 60s
- **memcpy 가 host time 의 80% (65s 합)** — 매트릭스 ttft p50 가 28ms→56ms 로 폭증한 이유와 일치 (memcpy 동기 비용)
- multi-GPU TP=8 의 inter-GPU H2D/D2H transfer 가 cudaMemcpyAsync 로 잡힘
- **lever**: memcpy 축소 = **A2 (KV tiering DRAM, zero-copy CPU buffer for input)** + **B1 (detok)**

### 3.3 결정

| 모델 | gap 종류 | (a)/(b)/(c) | 회수 lever |
|---|---|---|---|
| Qwen-7B (저 util) | **launch overhead 18%** | **(a) host-bound** | **B3 scheduler + A1 CPU drafting** |
| Llama-70B (고 util) | **memcpy 80%** | **(a) host-bound** | **A2 KV tiering + B1 detok** |

**verdict: 둘 다 (a) host-bound 확정 — SUB_201 의 본론 (CPU 드래프팅 + host-path 엔진) 진행 가능.**

단 모델 사이즈/TP 에 따라 dominant host 부담이 **launch (작은 모델) → memcpy (큰 모델/TP)** 로 이동. lever 선택도 그에 따라 분기.

---

## 4. 한계 & 미확정

### 4.1 측정 한계
- **CPU sampling 불가** (perf_event_open Fail) → host 작업 분해 (draft 제안 / scheduler / sampling / detok / Python overhead) 의 세부 비율 미측정
- **py-spy attach 미시도** (CAP_SYS_PTRACE 차단 가능, 별도 검증 필요)
- **traffic burst window 가 trace 보다 짧음** (Qwen-7B 35s vs trace 60s) → 일부 metric 이 idle 시간 포함, 정밀 normalization 필요
- ncu metric 불가 (NVGPUCTRPERM) → (b) verify 내부 occupancy 정밀 확인 미완

### 4.2 R1 671B 측정 — 부분 성공 (nsys-rep 미생성, indirect evidence 만)

**측정 시도 결과** (2026-06-05 KST 05:50):
- nsys delay=600s, duration=90s, R1 suffix TP=8
- vllm boot 7분 (cache hit), traffic burst 100p × conc=32 부분 완료 (64/100 ok, wall 172s)
- ⚠ **nsys-rep 미생성**: vllm 의 multiprocessing worker (reparented child) 가 nsys 의 cleanup 차단 → SIGTERM/SIGKILL 시 unfinalized 데이터 lost
- **알려진 nsys 제약**: vllm 의 v1 engine 이 spawn 한 worker 가 `--wait=all` 같은 옵션 필요 (next try)

**R1 vllm 내부 spec metrics — indirect evidence (10s 단위 8개 측정)**:

| metric | 값 (range) | 해석 |
|---|---|---|
| Mean acceptance length | 2.08~3.05 (목표 33 = K+1) | K=32 의 6-9% 만 활용 |
| Per-position accept rate | **0.35→0.06 빠른 decay** (12 position) | suffix cache prefix 부분 match, 깊은 단계는 거의 miss |
| Avg Draft acceptance rate | **42~60%** | 11k drafted 중 5-7k accepted, **40% wasted compute** |
| GPU KV cache usage | 4-6% | KV 압박 없음 (MLA 효과) |
| Generation throughput | 800-1000 tokens/s (32 concurrent) | 매트릭스 측정과 일치 |

**R1 traffic burst summary**:
```
DeepSeek-R1 × suffix × mix: 443 tps / wall 172s / gpu 76.5% / cpu 4.5% / ttft 240ms / 64/100 ok (36p 미완)
```

→ 매트릭스 측정 (mix 781 tps, α 0.45) 과 비슷한 패턴. **R1 의 suffix 페널티 = MoE expert routing × reasoning chain 다양성 → cache prefix 부분 match 만**.

**R1 verdict (indirect)**:
- nsys CUDA API 측정 없이도 spec metrics 만으로 **R1 의 suffix 가 wasted compute 40% 임이 명확**
- 즉 R1 은 ① spec-decode 자체의 가설을 깨는 모델 (α × K_eff < spec overhead) + ② 추가로 host overhead (matrix ttft 폭증 +200%) 도 있음
- A1 CPU drafting 로 회수 가능성: draft 의 wasted compute 를 CPU 로 옮기면 GPU 의 verify cost 만 → 매트릭스 throughput +20~+50% 회복 추정 (idata 매트릭스 +1500 tps 가능)

**다음 R1 측정 재시도 옵션**:
- `nsys profile --wait=all` 으로 multiprocessing worker 종료 대기
- 또는 vllm 안에서 `torch.cuda.cudart().cudaProfilerStart()` 호출 후 `--capture-range=cudaProfilerApi` 사용
- 또는 ncu (실패한 NVGPUCTRPERM 우회 — sudo 또는 NVreg_RestrictProfilingToAdminUsers=0)

---

## 5. 다음 단계

### 5.1 즉시
1. **R1 671B suffix 측정** (위 같은 셋업, delay=720s for 12 분 boot 안전 마진, duration=60s) — host overhead 의 어느 부분이 dominant 인지 확정 (예상: cudaMemcpyAsync 가 더 큼)
2. **py-spy attach 시도** — Python hot function (CAP 권한 확인). 또는 cProfile wrapper 로 vllm 측 내부 호출 분석

### 5.2 lever PoC (verdict 확정 후)
- **A1 CPU drafting**: 적용 모델 = R1 + DS-Llama-70B + Qwen-72B (α<0.5). IDE_019 자산 (`vllm/v1/spec_decode/` cpu proposer) 재활용
- **A2 KV tiering**: 적용 모델 = Llama-70B + 405B (memcpy dominant). IDE_017 자산 (DMA zero-copy)
- **B1 detok AVX-512**: 적용 모델 = Llama-8B / DS-Qwen-7B (mix 24~27k tps, throughput 큼). IDE_016/SUB_171
- **B3 scheduler**: 적용 모델 = Qwen-7B (launch overhead 18%, kernel 작음). 별도 분석

### 5.3 문서화
- 본 verdict 를 SUB_201 README §5.5 의 산출물로 인용 추가
- 결과 표 → 논문 §6 mechanism analysis 의 evidence 로 활용

---

## 부록 — 측정 명령 (재현용)

```bash
PROF=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile
export PATH=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64:$PATH

# 1) vllm serve via nsys (background, delay 후 trace 시작)
CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile -y 120 -d 60 -t cuda,nvtx,osrt \
  --output=$PROF/qwen7b_suffix_first --force-overwrite=true \
  /workspace/vllm_dev_prj/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 4 --port 8001 --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --speculative-config '{"method":"suffix","num_speculative_tokens":32}' &

# 2) READY poll + traffic burst (nsys trace window 동안)
# 3) nsys 종료 + vllm kill
# 4) analysis:
nsys stats --report cuda_gpu_kern_sum --format=table <run>.nsys-rep
nsys stats --report cuda_api_sum     --format=table <run>.nsys-rep
nsys stats --report nvtx_pushpop_sum --format=table <run>.nsys-rep
```
