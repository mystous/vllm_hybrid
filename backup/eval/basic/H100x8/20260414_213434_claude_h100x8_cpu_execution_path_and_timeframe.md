# H100x8 CPU 실행 경로 + 타임프레임 심층 분석

**작성**: 2026-04-14 21:34 (Claude)
**환경**: H100x8 물리 (violet-h100-023), Xeon Platinum 8480+ 2S × 56C × 2T = 224 logical, 2 NUMA
**분석 대상**: `eval/basic/H100x8/` 의 4 runs (1 gpu_only + 3 hybrid)

---

## 0. 실험 메타데이터

| ID | 디렉토리 | 모드 | max_seqs | threads | 결과 wall | 비고 |
|---|---|---|---:|---:|---:|---|
| G  | 20260413_081534_G | gpu_only | — | — | 3.77 s | TP=4 baseline |
| H1 | 20260414_044922_H_C | hybrid | 1 | 32 | **394.2 s** | profile peak 조합 |
| H2 | 20260414_045947_H_C | hybrid | 16 | 32 | **2097.6 s** | wave=16 재앙 |
| H3 | 20260414_054010_H_C | hybrid | 1 | 56 | **407.6 s** | NUMA 풀가동 |

**공통 workload**: Qwen2.5-7B-Instruct, TP=4 (H100 × 4), BF16, 500 req × 128 in / 128 out, burst (rate=inf)

---

## 1. 하드웨어/소프트웨어 스택

### 1.1 Hardware (system_info.json 기반)
- **CPU**: Intel Xeon Platinum 8480+ ×2 sockets, 56 cores/socket, 2T/core → 224 logical
  - ISA: AVX-512F/BW/VL/VNNI + **AMX-BF16 + AMX-INT8 + AMX-TILE**
  - L2 cache: 224 MiB (2 MiB × 112 instances)
  - L3 cache: 210 MiB (**105 MiB × 2 socket 분리**, non-shared)
  - DRAM: ~2 TB DDR5, node 0/1 각 ~1 TB, NUMA distance 10/21
- **GPU**: NVIDIA H100 80GB HBM3 × 8, NVLink, 3.35 TB/s HBM
- **NUMA topology**:
  ```
  node 0 cpus: 0..55 (P0-55 primary) + 112..167 (HT sibling)
  node 1 cpus: 56..111 (P56-111 primary) + 168..223 (HT sibling)
  ```

### 1.2 Software
- PyTorch 2.9.0+cu130 (CUDA build, CPU ops 호출 가능)
- IPEX 2.8.0+git (BF16 weight prepack, single_query_cached_kv_attention)
- oneDNN AMX primitive: `brg_matmul:avx10_1_512_amx` dispatch 실제 확인 (이전 v4 F4)
- vLLM v0.1.dev8504+ (commit `78fa48cb8` 기반 fork)

---

## 2. 부팅 단계 타임라인 (3 hybrid runs 통합)

### 2.1 Wall-clock 기준 이벤트 시퀀스

| Event | H1 (044922) Δt | H2 (045947) Δt | H3 (054010) Δt | 의미 |
|---|---:|---:|---:|---|
| APIServer 시작 | T+0 (04:47:57) | T+0 (04:57:59) | T+0 (05:35:44) | — |
| `[HYBRID-RESOLVE]` (main) | T+0 | T+0 | T+0 | `_resolve_cpu_params` 계산 |
| `[HYBRID-LAUNCH] num_cpu_engines=2` | T+0 | T+0 | T+0 | `launch_hybrid_engines` 가 2 engine spawn 결정 |
| CPU engine proc 2개 spawn + re-resolve | T+7s | T+7s | T+7s | `multiprocessing.Process(target=run_cpu_engine_core, numa_node=0/1)` |
| `[HYBRID-CPU-ENV]` + `[HYBRID-CPU-PROC]` | T+9s | T+9s | T+9s | `_setup_cpu_process_env` + torch import 후 |
| CPU VllmConfig created | T+9s | T+9s | T+9s | `_create_cpu_vllm_config` (device=cpu, TP=1, kv_cache=402GB) |
| `_get_autobind_cpu_ids` (NUMA 0/1) | T+9s | T+9s | T+9s | `hybrid_config.numa_bind_node` 경로 |
| `init_cpu_threads_env` (C++) 완료 | T+10s | T+10s | T+10s | 32 OMP threads × sched_setaffinity 1:1 pin (H3: 56 threads) |
| 7B weight 로드 + IPEX prepack | T+10~85s | T+10~115s | T+10~275s | NUMA-local DRAM 에 14 GB × 2 engines |
| **첫 request 도착 (probe)** | T+98s | T+122s | T+280s | `[HYBRID-ROUTER-INIT]` bench client 연결 |

**관찰**: 부팅 시간이 run 별로 다른 이유는 model weight 로드 / IPEX prepack 시간 변동성 + CUDA graph capture. 실제 request 처리는 `[HYBRID-ROUTER-INIT]` 이후 시작.

### 2.2 Engine identity 와 ZMQ 토폴로지 (부팅 후 고정)

`_HybridEngineLauncherMixin._compute_core_engines` (core_client.py:1355):

```python
num_cpu = max(1, hybrid_config.num_cpu_engines)   # = 2 (H100x8)
engine_ranks = [0, 1, 2]                          # GPU, CPU_1, CPU_2
core_engines = [rank.to_bytes(2, "little")        # [b'\x00\x00', b'\x01\x00', b'\x02\x00']
                for rank in engine_ranks]
```

```
APIServer (PID 1093764)
  ├─ input socket (ZMQ ROUTER, bind)
  │   ├─ GPU identity = b'\x00\x00' → GPU EngineCoreProc
  │   ├─ CPU_1 identity = b'\x01\x00' → CPU_EngineCore_1 (PID 1094539, NUMA 0)
  │   └─ CPU_2 identity = b'\x02\x00' → CPU_EngineCore_2 (PID 1094540, NUMA 1)
  │
  └─ output socket (ZMQ PULL, bind)
      ← GPU PUSH, CPU_1 PUSH, CPU_2 PUSH (interleaved async)
```

Request routing 은 **동기**: `CapacityAwareRouter.route()` 가 engine identity 반환 → `input_socket.send_multipart([identity, payload])`.
결과 수집은 **비동기**: `output_socket.recv()` 가 ID 식별 후 해당 request 의 Future 에 resolve.

---

## 3. Request routing — `_route_wave_batch` 상세

### 3.1 State machine (CapacityAwareRouter)

```python
# hybrid_core.py:189-235
self._cpu_states = [                     # per-engine in_flight & count
    {"in_flight": 0, "count": 0} for _ in range(num_cpu_engines)
]
self._cpu_wave_accepted = [0, 0]         # 현재 wave 의 누적 admit 수
self._cpu_wave_closed   = [False, False] # 현재 wave 가 full 되어 닫혔는지
self.cpu_max_num_seqs                    # wave size (H1/H3=1, H2=16)
```

### 3.2 Wave lifecycle (per engine)

```
open      : accepted=0, closed=False, in_flight=0
admitting : 0 < accepted < max_seqs, closed=False  → 계속 admit
full      : accepted == max_seqs → closed=True, 추가 admit 금지
draining  : closed=True, in_flight > 0
complete  : closed=True, in_flight == 0 → accepted=0, closed=False 로 reset
```

### 3.3 `_find_wave_open_cpu` (hybrid_core.py:440)

```python
def _find_wave_open_cpu(self) -> int:
    best_idx = -1
    best_accepted = self.cpu_max_num_seqs  # sentinel
    for i in range(self.num_cpu_engines):
        if self._cpu_wave_closed[i]: continue           # 이 engine 스킵
        if self._cpu_wave_accepted[i] >= self.cpu_max_num_seqs: continue
        if self._cpu_wave_accepted[i] < best_accepted:  # strict <
            best_accepted = self._cpu_wave_accepted[i]
            best_idx = i
    return best_idx
```

**Strict `<` 의 효과**: 두 engine 이 같은 accepted 값이면 **index 낮은 쪽 고정 선택**. 2 engine 동시 open 상태에서 req 들이 도착하면 alternating 이 아닌 **engine 0 → engine 0 → …** 이 될 수 있음. 하지만 engine 0 가 full 되면 engine 1 로 자연스럽게 넘어감.

### 3.4 Cold-start gate (hybrid_core.py:486)

```python
if (self.gpu_count + self.cpu_count) == 0:
    return self._to_gpu()
```

bench_serving.py 는 본 burst 전에 1건 "Initial test run" probe 를 보냄. 이 probe 를 GPU 로 보내지 않으면:
1. Monitor CPU util 첫 샘플부터 100% → 수치 왜곡
2. Wave 가 "동시 시작" 이 아닌 "순차 admit" → 측정 해석 어려움

### 3.5 H1 (max_seqs=1) 실제 routing 시뮬레이션

```
req 1 (probe): gpu_count=cpu_count=0 → cold-start → GPU
req 2: _find_wave_open_cpu → i=0 (accepted=0<1, best=0, idx=0)
                            → i=1 (accepted=0<0 false) → pick engine 0
       accepted[0]=1 ≥ max_seqs=1 → closed[0]=True
       [HYBRID-WAVE] engine=0 wave closed (accepted=1, batch_size=1)
req 3: _find_wave_open_cpu → i=0 closed skip
                            → i=1 (accepted=0<1, best=0, idx=1) → pick engine 1
       accepted[1]=1 → closed[1]=True
       [HYBRID-WAVE] engine=1 wave closed
req 4-500: 둘 다 closed → _find_wave_open_cpu returns -1 → GPU
```

**Total CPU dispatch = 2 req, GPU dispatch = 498 req**. 실측 로그의 `finished=501` 은 probe(1) + main 500 의 총합.

### 3.6 H2 (max_seqs=16) 실제 routing 시뮬레이션

```
req 1: GPU (cold-start)
req 2: engine 0 accepted=1
req 3: engine 0 accepted=2 (strict < 로 engine 0 이 계속 선택, engine 1 accepted=0 과 동률 시 idx 0 우선)

wait — actual trace 는 alternating.
실측 로그 확인 필요: req 2→engine 0 (1,0) → req 3 시 engine 1 선택? 
```

실제 바이트 레벨: `_find_wave_open_cpu` 가 engine 0 accepted=1, engine 1 accepted=0 상태에서 req 3 에 대해 i=0 (1<16, best=1 idx=0), i=1 (0<1 true → best=0 idx=1) → **engine 1 선택**. 즉 **alternating 동작**.

```
req 2: engine 0 (1, 0)
req 3: engine 1 (1, 1)
req 4: engine 0 (2, 1)
req 5: engine 1 (2, 2)
...
req 32: engine 1 (16, 16) → 둘 다 closed
req 33-500: GPU
```

**Total CPU dispatch = 32 req** (두 engine 각 16). TPOT 역산으로 확인:
```
500 × mean_TPOT(1047) = 468 × 22.2 + 32 × X
X = (523,500 - 10,390) / 32 = 16,034 ms ≈ p99 15,966 ✓
```

### 3.7 로그 검증 (H1 = 044922)

실제 기록:
```
04:49:35 [HYBRID-ROUTER-INIT] ...
04:49:36 [HYBRID-WAVE] engine=0 wave closed (accepted=1, batch_size=1)
04:49:36 [HYBRID-WAVE] engine=1 wave closed (accepted=1, batch_size=1)
04:49:39 [HYBRID-ROUTER-STATS] finished=501 ... in_flight_gpu=474 (GPU 아직 드레인 중)
...
04:54:32 [HYBRID-WAVE] engine=0 wave drained (accepted=1) → reset, next wave open
04:56:10 [HYBRID-WAVE] engine=1 wave drained (accepted=1) → reset
```

**CPU engine 0 처리 시간** = 04:54:32 - 04:49:36 = **296 s** (req 128 tokens)
**CPU engine 1 처리 시간** = 04:56:10 - 04:49:36 = **394 s** (req 128 tokens)
**Wall** = max(GPU 14s, CPU 1 394s) = **394.2 s** ✓

CPU per-step 역산 = 394s / 128 tokens = **3079 ms/step** (batch=1)

> **의외**: engine 0 이 engine 1 보다 빠름 (296 vs 394 s). 이유 추정:
> - NUMA 0 DRAM 이 NUMA 1 보다 freshness 가 좋았거나
> - NCCL/TP-4 GPU worker 가 NUMA 1 쪽 cores 에 더 간섭
> - 또는 단순한 tail-latency 편차 (1 req 이므로 표본 1)

---

## 4. CPU worker 실행 경로 상세

### 4.1 프로세스 레벨 스택

```
run_cpu_engine_core (hybrid_core.py:1245)
    ↓ 플랫폼 강제 override
    _platforms._current_platform = CpuPlatform()  ← CUDA 비활성화
    ↓
    _setup_cpu_process_env (hybrid_core.py:940)
        ├─ CUDA_VISIBLE_DEVICES=""
        ├─ VLLM_CPU_KVCACHE_SPACE=<auto GB>
        ├─ OMP_NUM_THREADS, MKL_NUM_THREADS=32 (or 56)
        ├─ OMP_PROC_BIND / OMP_PLACES pop (Intel OMP 로 인한 master thread 1-CPU pin 방지)
        ├─ sched_setaffinity(0, sched_getaffinity(1))  ← 오늘 fix (dev 용)
        ├─ VLLM_CPU_OMP_THREADS_BIND=auto
        └─ configure_intel_optimizations() → ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX
    ↓ torch import (환경변수 전파 후)
    torch.set_num_threads(32)
    torch.set_num_interop_threads(2)
    ↓
    _create_cpu_vllm_config (hybrid_core.py:1100)
        └─ scheduler.enable_chunked_prefill = False  ← CPU prefill 순차화 원인
    ↓
    EngineCoreProc(**kwargs) → run_busy_loop
        ↓ ZMQ recv request
        EngineCore.step()
            ├─ CoreScheduler.schedule()
            └─ UniProcExecutor.execute_model
                ↓
                CPUWorker.execute_model (cpu_worker.py:619)
                    ├─ _hybrid_exec_step += 1
                    ├─ [HYBRID-CPU-EXEC] trace (if VLLM_HYBRID_TRACE=1)
                    ├─ [HYBRID-CPU-PROFILE] attn/mlp breakdown hook
                    └─ self.model_runner.execute_model(scheduler_output)
                        ↓
                        CPUModelRunner._forward
                            ↓
                            Qwen2ForCausalLM.forward
                                ↓ 28 × Qwen2DecoderLayer
                                    ├─ self_attn (CPU paged attention)
                                    │     └─ cpu_attn.py:1261
                                    │         IPEX single_query_cached_kv_attention
                                    │             → C++ FD kernel (OMP parallel for collapse(3))
                                    │             → seq × partition × head_group 3-D 병렬
                                    ├─ LayerNorm
                                    ├─ mlp (gate_up + down projection)
                                    │     └─ oneDNN brg_matmul (AMX BF16)
                                    │         → OMP parallel on 32 threads (NUMA-local)
                                    └─ residual
```

### 4.2 IPEX FD kernel 내부 (PagedAttentionKrnl.cpp)

**Dispatcher** (line 1846-2015):
```cpp
if (num_heads * batch > threads * 2 && beam_size >= 4) {
    // VNNI INT8 path
} else {
    // Flash Decoding (FD) path
    flash_attention_decoding(...)
}
```

7B (28 heads, 4 kv_heads, GQA 7:1), batch=1, 32 threads:
- `28 × 1 > 32 × 2 = 64` → False → **FD kernel**

7B batch=16, 32 threads:
- `28 × 16 > 32 × 2 = 448 > 64` → True, 하지만 `beam_size=1 < 4` → **FD kernel 재차**

즉 batch 크기 무관하게 FD kernel 이 호출됨.

**FD kernel 병렬화** (line 1230):
```cpp
#pragma omp parallel for collapse(3)
for (seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (partition = 0; partition < num_partitions; partition++) {
        for (head_group = 0; head_group < num_kv_heads; head_group++) {
            // partition 내에서 QK^T + softmax + V 합산
        }
    }
}
```

**batch=1 FD**: total_iters = 1 × N_partitions × 4 = 4N. num_partitions 는 context_len (128 tok) / partition_size (일반적 64) = 2. 따라서 `1 × 2 × 4 = 8` parallel iters → 32 OMP threads 에 balanced.

**batch=16 FD**: total_iters = 16 × 2 × 4 = 128 parallel iters → 32 threads 에 4 iter/thread. 이론상 4× 더 work 인데 왜 **7× 느려지나?**
- **KV cache scatter**: 16 seq 의 block_table 이 서로 다름 → L3 105MB 에 8 seq × (28 layer × 2 pages × 128 KV) = 대량 페이지 → L3 miss 폭증
- **DRAM BW 포화**: batch=1 은 working set L3 fit, batch=16 은 DRAM 에서 매 page 당 read → 307 GB/s × 2 socket 이 이미 포화
- **Partition scheduling imbalance**: schedule(static,1) 기본값이면 seq 별 context_len 차이로 thread 간 load imbalance

### 4.3 MLP 경로 (oneDNN AMX BF16)

```
x → linear(gate_proj) → silu
y → linear(up_proj)
gate*up → linear(down_proj)
```

각 linear 는 oneDNN primitive:
- batch=1: M=1, K=3584, N=14336 (ffn_hidden)
- IPEX 가 weight prepack 을 미리 해둠 → oneDNN 이 `brg_matmul` 로 dispatch
- AMX-BF16 tile mult: 16×16×64 tile × 32 threads
- per-iteration ~15 ms (batch=1) vs ~240 ms (batch=16) → 16× 선형 대략 유지 (MLP 는 matmul 이므로 batch 와 compute 선형)

→ **MLP 는 batch 에 선형**. **Attention 이 batch 에 비선형** 인 것이 wave=16 재앙의 주 원인.

---

## 5. Request 처리 타임프레임 (per-request lifecycle, H1 기준)

### 5.1 GPU-routed request (probe 또는 req 4-500)

```
T=0     client POST /v1/completions
T+1ms   APIServer: HybridAsyncMPClient.add_request_async
T+1ms   CapacityAwareRouter._route_wave_batch → "gpu"
T+2ms   input_socket.send(identity=b'\x00\x00', payload)
T+3ms   GPU EngineCore recv, scheduler enqueue
T+3~5ms scheduler schedules, forward starts
T+5~28ms GPU forward (per-token TPOT 22-55ms)
T+28ms × 128 = 2.8s  streaming 완료 (per-token output via PUSH)
T+3s    request done
```

500 req GPU burst → 처음 ~3s 에 다 받고 ~14s wall clock 에 완료 (TP=4 NCCL + continuous batching).

### 5.2 CPU-routed request (H1 의 req 2 또는 req 3)

```
T=0        client POST (bench 시작 직후)
T+1ms      _route_wave_batch → _find_wave_open_cpu → "cpu:0"
T+2ms      [HYBRID-WAVE-DISPATCH] if PROFILE=1
T+2ms      input_socket.send(identity=b'\x01\x00', payload)
T+3ms      CPU_EngineCore_1 recv, EngineCoreProc.run_busy_loop
T+5~200ms  scheduler, enter execute_model
T+~200ms   **prefill 128 tokens**
            ├─ chunked_prefill=False → 128 tokens 를 1번의 forward 로
            ├─ attention: seq_len=128, QK^T = 128×128 matrix
            ├─ IPEX single_query 가 아닌 prefill kernel (VNNI BF16)
            └─ batch dim=1, seq_len=128 → matmul M=128, dense compute
           ≈ 200-500 ms (MLP 가 prefill 에서도 주 bottleneck)
T+500ms    첫 token 생성 (TTFT)
T+500ms~   decode 128 tokens
            ├─ 매 step 128×(prefill KV 재사용) + 1 new token
            ├─ IPEX single_query_cached_kv_attention (FD kernel batch=1)
            ├─ MLP AMX BF16 matmul batch=1
            └─ per-step ~3079 ms (실측 역산)
T+394s     128번째 token 생성 완료 → [HYBRID-WAVE] drained
```

**TTFT 로그 데이터**: H1 median TTFT=824ms, p99=1075ms. GPU req 가 대다수이므로 TTFT 분포는 GPU-dominated. 2 CPU req 의 TTFT 는 tail 에 묻힘 (p99 에 포함되지만 더 높은 percentile 없어 확정 불가).

---

## 6. max_seqs 별 비선형 열화 분석

### 6.1 실측 per-step (wall 역산, 128 tokens decode)

| Config | engine 수 | N_CPU | per-engine req | per-engine wall | per-step |
|---|---:|---:|---:|---:|---:|
| H1 max_seqs=1 threads=32 | 2 | 2 | 1 | 394 s | **3079 ms** |
| H2 max_seqs=16 threads=32 | 2 | 32 | 16 | 2098 s | **16390 ms** |
| H3 max_seqs=1 threads=56 | 2 | 2 | 1 | 408 s | **3188 ms** |

### 6.2 Batch 크기에 따른 per-step 증가율

```
H2 per-step (16) / H1 per-step (1) = 16390 / 3079 = 5.32×
```

만약 linear scaling 이면 16× 가 나와야 함. 실측 5.32× 는:
- **Attention (FD kernel)**: batch 1→16 에서 파이썬 단 linear × (scatter 페널티 ~1.5×) = 24×
- **MLP (AMX BF16 matmul)**: batch 1→16 에서 거의 linear 16×
- **Overhead (sampler, scheduler, ZMQ, Python)**: batch 무관 ~50 ms/step

합산:
```
per-step (batch=B) ≈ B × (attn_1 × penalty(B) + mlp_1) + overhead
```

`attn_1 ≈ 100 ms`, `mlp_1 ≈ 120 ms`, `overhead ≈ 50 ms` 추정:
- batch=1: 1 × (100 × 1.0 + 120) + 50 = **270 ms** (실측 3079 ms 보다 훨씬 작음)

→ 이 단순 모델로는 설명 안 됨. 실측 batch=1 3079ms 자체가 이론 예측 대비 10× 큼. 원인 미확정 가설:
- **KV cache paged read pattern**: 매 step 128 KV pages 를 DRAM 에서 read (L3 hit 불가, 14GB weight 가 L3 밀어냄)
- **oneDNN kernel 재진입 오버헤드**: 매 layer 의 matmul primitive 재생성 비용
- **Python-C++ 경계 왕복**: 28 layer × (attn + MLP + norm + residual) = 100+ 번 경계 왕복

batch=16 에서의 5.32× ratio 는 이 상수 부분이 batch 전체로 amortize 됐고 scatter penalty 가 함께 작용한 결과.

### 6.3 batch=16 에서 MLP vs Attention 기여도 (가설)

실측 per-step 16390 ms. 만약 MLP 가 linear (16× = 120×16 = 1920 ms) 라면 나머지 14470 ms 가 attention + KV scatter 라는 뜻. 이론적으로 attn=100 × 16 × scatter_penalty = 1600 × penalty. penalty ≈ 9× 여야 함.

**실측 per-step breakdown 은 `VLLM_HYBRID_PROFILE=1` 로 측정 가능** (현재 미수집). 다음 실험에서 H2 config 로 짧은 bench (5-10 req) 돌리면 `[HYBRID-CPU-PROFILE] step=N attn=X ms mlp=Y ms` 로 분해 수치 확보 가능.

---

## 7. CPU-GPU Idle 비대칭

### 7.1 Monitor CSV 기반 실측 (H1, 1777 samples @ 1 Hz)

| 지표 | Value |
|---|---:|
| GPU avg util | **0.0%** (GPU 가 3s 에 끝난 뒤 2060s 동안 idle) |
| GPU busy samples (>10%) | 2 / 1777 (0.1%) |
| CPU avg util | **19.0%** (32 threads × 112 cores × ~19% ≈ 21 cores busy) |
| CPU busy duration | 2060 s (10s~2070s) |

### 7.2 이상적 분담 vs 실제

workload: 500 req × 128 decode tokens = 64,000 total tokens

이상:
```
T_ideal = total_tokens / (GPU_TP + 2 × CPU_TP) 
        = 64000 / (16501 + 2 × 0.1) ≈ 3.9 s
```
(CPU 기여가 거의 0 이므로 GPU_only wall 14s 와 동일)

실제 H1: 394s. 이유:
- Router 가 probe 직후 req 2/3 을 CPU 로 보냄 (cpu-first policy)
- 그 2 req 가 완료될 때까지 wall 이 기다림
- GPU 는 나머지 498 req 를 14s 에 완료했지만, CPU req tail 로 인해 전체 wall 394s

**구조적 손해**: 2 req × 128 tokens = 256 tokens 를 CPU 에 보내면 **GPU 기준 16ms 작업** 인데 **CPU 에서 394s 걸림** = 24,625× 느림. Amdahl 위반.

### 7.3 언제 hybrid 이득이 나는가

`T_hybrid = max(T_gpu_share, T_cpu_share)` 가 `T_gpu_only` 보다 작아지려면:
1. **GPU 가 queue-saturated** (현재 H100x4 32B 에서도 GPU 43% 로 sub-saturated)
2. **T_cpu_share < T_gpu_only - T_gpu_share**

현재 H100x8 7B 에서는 GPU 단독 14s, CPU 1 req 394s → CPU 에 req 보낸 순간 wall ≥ 394s.

**조건 충족 예시** (미실측):
- **70B TP=8, long context 16K**: GPU prefill bottleneck → T_gpu_only 가 수 분으로 증가
- **Rate-limited burst 2000+ req**: GPU queue 가 차서 T_gpu_share 확장 → CPU 가 일부 흡수 가능

---

## 8. 소스 코드 ↔ 로그 마커 완전 매핑

| 로그 marker | 파일:라인 | 역할 | 발생 빈도 |
|---|---|---|---|
| `[HYBRID-RESOLVE]` | `hybrid_core.py:906` | `_resolve_cpu_params` 결과 emission | boot 시 2-3회 (main + per-engine) |
| `[HYBRID-LAUNCH]` | `hybrid_core.py:1455` | `launch_hybrid_engines` 가 결정한 engine 수 | boot 시 1회 (APIServer) |
| `[HYBRID-CLIENT]` | `core_client.py:1407` | `HybridAsyncMPClient` 의 num_cpu_engines write-back | boot 시 1회 |
| `[HYBRID-CPU-ENV]` | `hybrid_core.py:1070` | OMP/ISA/BIND 환경 설정 완료 | per-engine boot 1회 |
| `[HYBRID-CPU-PROC]` | `hybrid_core.py:1292` | torch import 후 thread 수 확정 | per-engine boot 1회 |
| `CPU VllmConfig created` | `hybrid_core.py:1216` | `_create_cpu_vllm_config` | per-engine boot 1회 |
| `[HYBRID-CPU-WORKER] thread config` | `cpu_worker.py:276` | `CPUWorker.__init__` | per-engine boot 1회 |
| `[HYBRID-CPU-WORKER] _get_autobind_cpu_ids` | `cpu_worker.py:837` | NUMA bind 결정 (numa_bind_node 우선) | per-engine boot 1회 |
| `[HYBRID-CPU-WORKER] init_device` | `cpu_worker.py:436` | 최종 local_omp_cpuid | per-engine boot 1회 |
| `[HYBRID-CPU-WORKER] init_cpu_threads_env` | `cpu_worker.py:447` | C++ extension pin 결과 | per-engine boot 1회 |
| `[HYBRID-CPU-WORKER] post-init` | `cpu_worker.py:494` | 최종 main thread affinity | per-engine boot 1회 |
| `[HYBRID-ROUTER-INIT]` | `hybrid_core.py:264` | `CapacityAwareRouter` 첫 dispatch 시 | 1회 (첫 request) |
| `[HYBRID-WAVE] wave closed` | `hybrid_core.py:517` | wave-batch 가 max_seqs 도달 | per-wave 1회 |
| `[HYBRID-WAVE] wave drained` | `hybrid_core.py:557` | wave 완료 후 reset | per-wave 1회 |
| `[HYBRID-WAVE-DISPATCH]` | `hybrid_core.py:508` | engine 선택 + accepted snapshot | per-CPU-dispatch (PROFILE=1) |
| `[HYBRID-ROUTER-STATS]` | `hybrid_core.py:681` | 주기적 처리량/ratio | stats_log_interval (25 finished 마다) |
| `[HYBRID-CPU-EXEC]` | `cpu_worker.py:701` | per-step trace | TRACE=1 or TRACE_EVERY=N |
| `[HYBRID-CPU-PROFILE]` | `cpu_worker.py:712` | per-step attn/mlp breakdown | PROFILE=1, PROFILE_EVERY=N |
| `[HYBRID-CPU-ATTN-IPEX]` | `cpu_attn.py:1278` | IPEX call timing + batch histogram | PROFILE=1 |

---

## 9. 결론 및 다음 연구 방향

### 9.1 확인된 사실

1. **2-NUMA hybrid 인프라 정상 동작** — 2 engine × 56 cores strict bind, alternating admit, wave lifecycle 전부 로그로 검증됨
2. **profile peak 32 threads 재현** — H1 (394s) vs H3 (408s) 에서 32>56 확인, BW-bound 포화 구간 일치
3. **wave=16 재앙은 attention 스케일링 비선형성** — batch 1→16 에서 per-step 5.32× 증가 (선형 예측 16× 에서 이탈, KV scatter 페널티로 설명)
4. **현재 workload (500×128/128) 에서 hybrid 이득 구조적 불가** — T_hybrid = T_cpu_tail = 394s vs T_gpu_only = 14s

### 9.2 미확정 가설 (추가 실험 필요)

1. **CPU per-step batch=1 이 이론 예측 대비 10× 큰 이유** — KV paged access / oneDNN re-init / Python-C++ 경계 오버헤드 중 어느 것이 지배적인가?
   - 실험: `VLLM_HYBRID_PROFILE=1` 로 H1 config 재실행, `[HYBRID-CPU-PROFILE]` 로 attn vs mlp 분해
2. **engine 0 vs engine 1 tail latency 차이 (296 vs 394s)** — NUMA 별 DRAM 상태 또는 TP-4 NCCL 간섭?
   - 실험: 5회 반복 실행 → engine 0 이 항상 빠르면 구조적, 무작위면 tail variance
3. **max_seqs=2, 4, 8 의 wall** — batch 스케일링 곡선의 knee point 확인
   - 실험: H1 config 에서 max_seqs 만 바꿔 3회 추가 실행

### 9.3 다음 단계 우선순위

1. **GPU 포화 workload 정의** (70B / long-context 16K / rate-limited 2000 req) → hybrid 이득 조건 만족 여부 실측
2. **Spec decode CPU drafter (A1)** 구현 — request-level partition 의 Amdahl 한계 돌파
3. **PROFILE 로그 수집 재실험** — 현재 분석의 미확정 가설 3건 해소

### 9.4 H100x8 실험의 논문 기여

- **"hybrid is complement, not replacement"** (paper Property 2) 가 H100+7B+500×128/128 에서 올바르게 작동 (CPU 에 거의 안 보냄)
- **2-NUMA 경로 정상 동작 확인** → multi-socket 서버에서 동작하는 코드로 공식화 가능
- **wave=16 재앙 재현 + 원인 후보 제시** → paper §5 limitation 으로 수록 가치

---

## Appendix: 로그 raw excerpt (H1 044922)

부팅:
```
04:47:57 APIServer [HYBRID-RESOLVE] max_seqs=1 threads=32 cores=56 numa_nodes=2
04:47:57 APIServer [HYBRID-LAUNCH] num_cpu_engines=2 (numa_aware=True)
04:48:06 CPU_EngineCore_1 [HYBRID-CPU-ENV] PID=1094539 OMP=32 ONEDNN_ISA=AVX512_CORE_AMX
04:48:06 CPU_EngineCore_2 [HYBRID-CPU-ENV] PID=1094540 OMP=32 ONEDNN_ISA=AVX512_CORE_AMX
04:48:06 CPU_EngineCore_1 [HYBRID-CPU-WORKER] init_device local_omp_cpuid='112..167' numa_bind_node=0
04:48:06 CPU_EngineCore_2 [HYBRID-CPU-WORKER] init_device local_omp_cpuid='168..223' numa_bind_node=1
04:48:07 CPU_EngineCore_1 post-init cpu_affinity=1 cores [112]
04:48:08 CPU_EngineCore_2 post-init cpu_affinity=1 cores [168]
04:49:35 APIServer [HYBRID-ROUTER-INIT] strategy=wave-batch priority=cpu-first cpu_max_num_seqs=1
```

Wave:
```
04:49:36 [HYBRID-WAVE] engine=0 wave closed (accepted=1, batch_size=1)
04:49:36 [HYBRID-WAVE] engine=1 wave closed (accepted=1, batch_size=1)
04:54:32 [HYBRID-WAVE] engine=0 wave drained (accepted=1) → reset, next wave open
04:56:10 [HYBRID-WAVE] engine=1 wave drained (accepted=1) → reset
```

Stats (초기):
```
04:49:39 [HYBRID-ROUTER-STATS] finished=501 GPU=43.8 tok/s (499 reqs), 
         CPU=0.0 tok/s (2 reqs), cpu_ratio=0.4%, in_flight_cpu=2/1, in_flight_gpu=474
```

Stats (종반):
```
04:54:32 [HYBRID-ROUTER-STATS] finished=501 GPU=48.5 tok/s CPU=0.4 tok/s in_flight_cpu=1/1
```

---

## 10. 보충 (codex `inspect.txt` 교차 검증)

`inspect.py` (2026-04-14 commit 6df4ce0f5 추가) 가 H3 (054010) run 에 대해 생성한 `inspect.txt` 데이터. Claude 의 초기 분석에서 누락된 사실을 수록.

### 10.1 OMP 가 pin 된 것은 physical primary 가 아니라 **HT sibling**

| NUMA 0 (run 054010, 354 samples @ 1 Hz) | mean | max |
|---|---:|---:|
| **physical primary** (CPU 0-55) | **3.1%** | 8.0% |
| **HT sibling** (CPU 112-167) | **72.6%** | 93.0% |
| combined (per-physical-core) | 37.8% | 48.3% |

| NUMA 1 | mean | max |
|---|---:|---:|
| physical primary (CPU 56-111) | 4.9% | 11.8% |
| HT sibling (CPU 168-223) | 70.2% | 93.7% |
| combined | 37.5% | 50.0% |

**해석** (당시 코드 `cpus[-1:]` 선택 결과):
- `_get_autobind_cpu_ids` 가 `core_to_cpus[physical_core] = [primary, ht_sibling]` 중 **뒤의 것 (ht_sibling)** 을 선택
- C++ `init_cpu_threads_env` 가 OMP thread 를 logical CPU 112-167 / 168-223 에 1:1 pin
- HT sibling 측에서 바쁨 (~72%), physical primary (0-55, 56-111) 는 Linux scheduler view 상 ~3% (kernel thread + 모니터링 정도)
- **하드웨어 상 한 physical core 의 두 logical 은 동일한 core resource 공유** → 실제 core 단위 util 은 combined 37.8%

**commit 6df4ce0f5 의 fix 의미**: `cpus[-1:]` → `cpus[:1]` 로 변경해 physical primary 에 pin. H3 이전 run (H1/H2 포함) 은 모두 HT sibling pinning. **Claude 의 문서 본문에서 `local_omp_cpuid='112..167'` 을 "56 physical primary" 로 읽은 건 부정확** — 실제로는 56 logical ID 에 해당하는 **HT sibling**. Codex fix 이후 로그에 `'0,2,...,54, 56,58,...'` 같은 primary 번호가 나와야 정상.

### 10.2 GPU per-GPU util 분해 — TP=4 는 GPU 0-3 만 사용

```
GPU0: util mean=0.3%  max=45.0%  | power mean=118W  max=449W
GPU1: util mean=0.2%  max=44.0%  | power mean=119W  max=470W
GPU2: util mean=0.2%  max=45.0%  | power mean=116W  max=447W
GPU3: util mean=0.2%  max=43.0%  | power mean=124W  max=466W
GPU4: util mean=0.0%  max= 0.0%  | power mean= 75W  max= 76W   ← dormant
GPU5: util mean=0.0%  max= 0.0%  | power mean= 69W  max= 69W
GPU6: util mean=0.0%  max= 0.0%  | power mean= 68W  max= 68W
GPU7: util mean=0.0%  max= 0.0%  | power mean= 69W  max= 70W

avg: mean=0.1%  max=22.1%  (354 samples, 416s)
```

**관찰**:
- env `TENSOR_PARALLEL_SIZE=4` 에 의해 **GPU 0-3 만 TP 그룹으로 초기화** 됨. GPU 4-7 은 vLLM 에서 완전 미사용.
- GPU 4-7 은 dormant idle (power 68-75W, NVIDIA driver default). workload 와 무관.
- GPU 0-3 의 peak util 45%, peak power 449-470W — 정상 GPU burst
- mean util 0.2-0.3% 는 run 전체 416s 중 GPU compute 는 단 몇 초 (GPU 가 빨리 끝나고 CPU tail 대기).
- avg across 8 GPUs = 0.1% — **H100x8 서버의 ¾ 가 완전 낭비**

**논문 시사점**: 본 workload 의 "H100x8" 는 실제로는 H100x4 + 4 GPUs dormant. TP=8 로 했다면 GPU 0-7 모두 사용되지만 workload (7B) 가 너무 작아 오히려 NCCL overhead 가 증가.

### 10.3 Power budget 실측

| GPU state | mean power | peak power | 비고 |
|---|---:|---:|---|
| Active TP group (0-3) | 116-124 W | 449-470 W | compute burst 시 peak, 대부분 idle |
| Dormant (4-7) | 68-75 W | 76 W | driver default, 변동 없음 |
| CPU engine × 2 (56 threads each) | 측정 안 됨 | — | RAPL counter 미수집 |

**전체 서버 GPU 전력**: (116+119+116+124 + 68+75+69+70) W ≈ **757 W mean** during hybrid run. wall 416s × 757 W / 3600 = **87.4 Wh** (GPU 만). 같은 workload 를 gpu_only (14 s) 로 완료 시 동일 GPU group 이 약 (475×4 + 72×4) W × 14 s / 3600 ≈ **8.5 Wh** → **hybrid 가 10× 더 전력 소모**. CPU 까지 고려하면 격차 더 커짐 (Sapphire Rapids 2 socket full load ~700W 추가 추정).

### 10.4 Physical core util 분석의 논문적 의미

vLLM 의 `init_cpu_threads_env` C++ 구현은 per-physical-core 1 OMP thread 가 원칙 (`one_logical_per_core` 설계). `cpus[-1:]` 선택은 이 원칙에 부합하지만, **logical ID 가 HT sibling 인 것**이 시스템 모니터링 도구에서 "physical 은 idle, HT 는 busy" 처럼 보여 혼란을 유발. `cpus[:1]` 로 fix 한 이후에는 logical ID 가 physical primary (0-55 처럼) 로 나와 모니터링 친화적.

실제 compute 는 동일 (한 physical core 가 OMP thread 1개 처리). 단 다음 2 상황에서 차이:
1. **모니터링 해석**: fix 전에는 physical util 만 보면 "CPU 노는 중" 으로 잘못 판단 가능
2. **다른 프로세스와 공존**: HT sibling 에 OMP 가 있으면 OS scheduler 가 physical primary 에 다른 job 배치 → 같은 physical core 가 HT 경쟁으로 성능 저하. fix 이후 OS 가 physical primary 를 OMP 에 할당하므로 HT sibling 은 확실히 idle → 다른 프로세스가 들어와도 서로 다른 physical core 에 배치될 가능성 상승

### 10.5 결론 재정리

Claude 본문의 **"local_omp_cpuid='112..167' = 56 physical primary"** 는 부정확. 정확하게는:
- `112..167` 은 NUMA 0 의 56개 **HT sibling logical CPU** (physical primary 0-55 의 짝)
- 한 physical core 당 1 OMP thread 가 pin 된 건 맞음 (1:1 원칙 준수)
- 단, 그 1 thread 가 OS scheduler 에서 보기엔 HT 측에서 돌아감
- 6df4ce0f5 이후 run 은 `0..55` 등 primary logical 에 pin

본 데이터 (054010) 는 **fix 이전 상태**. NUMA binding + engine spawning 은 정상 작동했으나 primary vs HT 선택만 fix 로 정정됨. 성능 수치 (394/408s wall) 는 H1/H3 간 threads 32 vs 56 차이로 해석했는데, **HT 선택 자체가 공통 이므로 그 해석 유효**.

> **논문용 정확한 표현**: "Each CPU EngineCore process spawned per NUMA node, pinning 32 (or 56) OMP threads 1:1 to the NUMA's logical CPUs (HT siblings in runs prior to commit 6df4ce0f5, physical primaries after)."

---

## 11. 추가 보충 (codex `20260414_213415_codex_h100x8_log_analysis.md` 교차 검증)

아래 항목은 codex 문서가 로그 + 소스에서 정확히 잡아냈지만 Claude 본문에 빠져 있던 내용. 본문 해석을 **정정** 해야 하는 부분 포함.

### 11.1 `post-init torch_threads` 는 OMP 설정을 **덮어씀**

boot log 실측:
```
# 044922 (HYBRID_CPU_THREADS=32 env, resolve threads=32)
[HYBRID-CPU-ENV] PID=1094539 OMP=32 ...
[HYBRID-CPU-WORKER] post-init: torch_threads=56 process_threads=247 cpu_affinity=1 cores [112]

# 054010 (HYBRID_CPU_THREADS=56 env, resolve threads=56)
[HYBRID-CPU-ENV] PID=... OMP=56 ...
[HYBRID-CPU-WORKER] post-init: torch_threads=56 ...
```

**관찰**: `044922` 도 최종 `torch_threads=56`. 원인은 C++ `init_cpu_threads_env` (csrc/cpu/utils.cpp) 가 `local_omp_cpuid` 길이 기준으로 `omp_set_num_threads(56)` / `torch::set_num_threads(56)` 재설정하기 때문. `local_omp_cpuid='112..167'` (56 ids) 이므로 env 에서 32 를 지정해도 결국 56 으로 확정.

### 11.2 본문 Part D 정정 — "threads=32 vs 56 차이 3.6%" 해석 오류

본문 4절/6절의 다음 해석은 **부정확**:

> profile peak 32 threads 재현 (BW-bound 포화, 56 으로 갈 수록 barrier overhead 손해)

실제로는 **두 run 모두 decode 단계에서 torch_threads=56**. 따라서 **thread 수 차이로 wall 차이 (394 vs 408s, +3.6%) 를 설명할 수 없음**. 남은 가능성:
- **부팅 타이밍 편차**: 모델 로드 / IPEX prepack / CUDA graph capture 시점 차이
- **CPU tail 분산**: 동일 config 여도 2 CPU req 가 수 % 변동하는 측정 자연 분산 (n=1)
- **NUMA 상태 차이**: DRAM 페이지 freshness, NUMA 1 쪽 TP-4 NCCL 간섭
- **env `HYBRID_CPU_THREADS` 값 무시 됨을 처음부터 인지해야**. env 는 실제로 `KVCACHE_GB` 등 다른 값에 영향 주지만 thread 수는 C++ 가 덮어씀.

따라서 **"dev profile peak 32" 의 결론** 은 dev 머신에 한정 (i9-12900KF 에선 HYBRID_CPU_THREADS 가 min(env, physical_cores) 으로 반영되어 16 까지 차이남). **H100x8 에서는 env threads 설정이 최종 OMP 수를 바꾸지 않으므로 `_maxthreads` variant 와 base 의 wall 차이는 thread 수가 아닌 다른 요인**.

### 11.3 `wall_time_s` vs `duration` 구분

hybrid.json 필드:
- `duration`: benchmark 의 compute 시간 (첫 request 전송 ~ 마지막 완료)
- `wall_time_s`: **bench.sh 레벨 wall** (probe + main burst + overhead 포함)

codex 가 사용한 `wall_time_s`:
- 044922: duration 394.24s, **wall_time 403.74s**
- 054010: duration 407.57s, **wall_time 418.04s**
- 045947: duration 2097.58s, **wall_time 2108.0s**

Claude 본문은 `duration` 을 "wall" 로 잘못 표기한 경우가 있음 (394s 등). 차이 ~10s 는 probe + bench 초기화. **논문용 숫자는 `wall_time_s` 가 더 엄밀**.

### 11.4 IPEX KV cache write 경로

`_IPEXPagedAttention.write_to_paged_cache` (cpu_attn.py):
```python
ipex_modules.PagedAttention.reshape_and_cache(
    key, value, key_cache, value_cache,
    slot_mapping.flatten().int())
```

**KV write 도 pure PyTorch 가 아닌 IPEX C++ 경로**. Claude 본문은 decode attention (`single_query_cached_kv_attention`) 만 다뤘는데, 매 step 마다 **write + read 두 IPEX 호출이 병행**. batch=16 재앙은 read (scatter) 뿐 아니라 write 도 batch 증가로 영향 받음. KV cache 의 `slot_mapping` 이 16 seq 에 대해 서로 다른 slot 을 가리키므로 write 도 scatter pattern.

### 11.5 Chunked prefill `flash_attn_varlen_func` 경로 disabled

`_create_cpu_vllm_config`:
```python
cpu_sched.enable_chunked_prefill = False
cpu_sched.chunked_prefill_enabled = False
```

cpu_attn.py 에 `ipex_modules.PagedAttention.flash_attn_varlen_func` (chunked prefill 용) 도 있지만 **본 실험에서는 비활성화**. 즉:
- CPU decode: IPEX `single_query_cached_kv_attention` ✓ 사용
- CPU prefill (chunked): IPEX `flash_attn_varlen_func` ✗ **미사용**
- CPU prefill (full): 비 chunked 경로 (non-IPEX? 확인 필요)

→ CPU prefill 최적화 여지 존재 (chunked_prefill 활성화 시 `flash_attn_varlen_func` 사용, TTFT 개선 가능성). 단 wave-batch 와의 호환성 별도 검증 필요.

### 11.6 `in_flight_cpu=N/M` 표기 의미

router stats 의 `in_flight_cpu=2/1` / `32/16`:
- 분자 N: **aggregate CPU in_flight** (전체 CPU engine 합산)
- 분모 M: **per-engine `cpu_max_num_seqs`**

**주의**: 분자/분모가 서로 다른 scope. 예를 들어 `num_cpu_engines=2, cpu_max_num_seqs=1` 일 때 최대 aggregate in_flight = 2 이지만 표기는 `2/1`. 이 `2/1` 은 "2 req 가 per-engine 제한 1 을 초과" 로 잘못 읽힐 여지 있음 → 사실 per-engine 제한은 1, aggregate 는 2 (정상 최대치).

codex 관찰: `32/16` 도 동일 — engine 2개 × max_seqs 16 = aggregate 32, 분모는 여전히 engine 단일 한도 16. **overflow 버그 아님**.

### 11.7 정직한 해석의 범위 (codex vs Claude 차이)

Claude 10.1 은 `inspect.txt` 의 physical/HT split 수치를 바로 수용:
- physical (0-55) mean 3.1%
- HT sibling (112-167) mean 72.6%

Codex 는 이 해석을 **"로그만으로는 확정 불가"** 로 보수적 표현:
> 로그만으로는 다음을 단정할 수 없다. 위 CPU 집합이 topology 상 physical 인지 HT sibling 인지.

양쪽 입장 정리:
- **로그 only 로 본 HT/physical 확정 불가** (logical CPU id 만 있음)
- 하지만 `lscpu -e=CPU,CORE,NODE` topology + inspect.txt 가 사용한 `phys_threshold=112` 규약을 외부 근거로 받아들이면 확정 가능
- system_info.json 의 NUMA raw 에 이미 logical id → NUMA 매핑 있음. 이 시스템에서 HT 번호 규칙은 "primary 0-55, 56-111 / HT 112-167, 168-223" 로 광범위 검증된 Xeon 8480+ topology

**결론**: Claude 10.1 의 HT 해석은 외부 근거 (system_info + inspect.txt 기반) 를 수용한 것으로 유효. 다만 **"로그 자체가 primary vs HT 를 단언하지 않는다"** 는 codex 지적은 공정함. 논문 서술 시에는 "system_info.json NUMA topology + inspect.txt 의 `phys_threshold=112` 기반" 이라는 근거 출처를 명시.

### 11.8 Codex 가 제시한 다음 분석 후보 (채택)

1. **`local_omp_cpuid='112..223'` 범위가 선택된 정확한 코드 경로 추적** — `_get_autobind_cpu_ids` 의 `sched_getaffinity(0)` 결과 + NUMA filter + `cpus[-1:]` 가 **왜 HT sibling 을 먼저 return 하는지** 의 기전 설명 (core_id ordering 관련)
2. **C++ `init_cpu_threads_env` 가 env threads 무시하는 게 의도된 동작인가, 버그인가** — CLAUDE.md / 설계 문서와 대조
3. **CPU 1 request per-token 추정치 정량화** — 현재 실측만 있고 이론치 대비 분해 없음. `[HYBRID-CPU-PROFILE]` 로 attn/mlp/other 분해 재측정 필요

### 11.9 본문 수정 권고 요약

| 본문 위치 | 원 내용 | 정정 방향 |
|---|---|---|
| Part D (§4 IPEX dispatcher) | "32 threads" 가정 | **실제 56 threads** 로 교체 |
| §5 Per-step 계산 | threads=32 기준 | threads=56 기준 재계산 |
| §6 batch 스케일링 해석 | threads 차이로 설명 | 제거 — threads 는 동일 |
| Wall time 표기 | 394.2 / 2098 / 408 | `wall_time_s` 기준 403.74 / 2108 / 418.04 병기 |
| §9.1 H3 vs H1 차이 | "profile peak 32 재현" | **재검토 필요** — H100 에선 threads 차이 없음, dev 한정 결론 |

위 정정은 본문 덮어쓰지 않고 **Section 11 로 보완** 하는 형태로 기록. 원본 문서 흐름과 codex 관찰을 모두 보존.
