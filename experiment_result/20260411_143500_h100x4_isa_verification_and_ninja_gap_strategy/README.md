# H100x4 + SPR 8480+ — ISA/IPEX 동작 검증 + Ninja Gap 전략 분석

`20260411_143500_h100x4_isa_verification_and_ninja_gap_strategy`

> 형식: 실험 run 이 아니라 **검증 + 전략 분석 노트**.
> (1) 현재 AMX / AVX-512 / IPEX 가 정말로 실행되고 있는지 4 단계로 empirical
> 검증, (2) 그 위에서 paper §3 Property 2 의 ninja gap 을 실제 wall-time
> 단축으로 전환하기 위해 어떤 architectural change 가 필요한지 정리.

## 1. ISA / IPEX / oneDNN 동작 검증 (4-단계 evidence)

질문: "AMX, AVX-512, IPEX 가 다 제대로 동작하고 있는건 맞아?"

답: **네, 4 단계 모두 통과.**

| 단계 | 측정 | 결과 |
|---|---|---|
| **1. CPU feature detect** | `vllm.platforms.intel_cpu_utils.detect_intel_cpu_features()` | Intel Xeon Platinum 8480+ — AVX-512 ✓, AVX-512 VNNI ✓, AVX-512 BF16 ✓, **AMX-BF16 ✓**, **AMX-INT8 ✓** (96 cores, Sapphire Rapids+) |
| **2. 환경변수** | server 부팅 `[HYBRID-CPU-ENV]` 로그 | `ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX` 정상 적용 |
| **3. Extension/module load** | `vllm._custom_ops`, `torch.ops._C_utils`, `intel_extension_for_pytorch` | `HAS_CPU_OPS=True`, `HAS_CPU_UTILS=True`, `init_cpu_threads_env` 등록 ✓, IPEX 2.8.0+gitcb81bf2 ✓, mkldnn enabled ✓ |
| **4. Decode path 실측** | 1.5B 회귀 run 의 1900 회 attention 호출 | 100% `[HYBRID-CPU-ATTN] path=ipex` (custom_avx / sdpa_batched / sdpa_loop fallback **0건**) |
| **5. oneDNN 커널 실측** | `ONEDNN_VERBOSE=1` 로 BF16 matmul 트레이스 | `brg_matmul:avx10_1_512_amx` — **실제 AMX BF16 brgemm 커널 dispatch** |

`ONEDNN_VERBOSE` raw output (BF16 64×128 × 128×256 matmul):

```
onednn_verbose,v1,info,cpu,isa:Intel AVX-512 with float16, Intel DL Boost
  and bfloat16 support and Intel AMX with bfloat16 and 8-bit integer support
onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx10_1_512_amx,...
  src:bf16::blocked:ab::f0 wei:bf16::blocked:ab::f0 dst:bf16::blocked:ab::f0
```

→ ISA / IPEX / oneDNN 스택은 **빈틈없이 실행 경로에 들어옴**. 본 라우팅
fix 로 H100+1.5B/7B 에서 CPU 분배가 0 건이 되었지만, 그건 **Property 2 가
정확히 작동해서 CPU 가 더 느린 경로로 안 보내는 것**이지 ISA 가 안 도는
것이 아님.

### 1.1 잔존 minor item

`_setup_cpu_process_env` 에서 `OMP_PROC_BIND` / `OMP_PLACES` 가 미설정 (`None`).
1:1 코어 pin 은 C++ `init_cpu_threads_env` 가 `sched_setaffinity` 로 직접
처리해서 동작은 하지만, 향후 large CPU workload (예: spec decode drafter,
KV offload) 시나리오에서는 OpenMP 자체에 binding hint 도 같이 주는 게 안전.

별도 PR 후보:

```python
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")
```

## 2. Ninja Gap 의 한계 — 현재 접근의 천장

**현재 hybrid 의 decomposition**: request-level partition. 한 요청 통째로
GPU 또는 CPU 에 보내고 양쪽이 독립 실행. 본 fix (`_route_throughput_adaptive`
의 expected-finish 비교) 는 이 모델 안에서 wall-optimal routing 을 한다.

**천장 (정량화)**:

T_hybrid = max(T_gpu_share, T_cpu_share)

GPU 는 `tput_gpu_per_req × N_gpu_share` tokens/sec, CPU 는 `tput_cpu_per_req
× N_cpu_share`. 같은 요청을 어디 보내든 그 요청 자체가 끝나는 시각의 max
가 wall. **같은 모델 같은 워크로드**에서는 CPU per-req latency 가 GPU 보다
크면 (H100+1.5B 에서 13×, H100+7B 에서 20×+), CPU 에 보낼수록 max() 가 커짐
→ ninja gap 음수.

T_hybrid < T_gpu 가 성립하려면 둘 중 하나가 필요:

**Case (a)** — *같은 일을 다른 자원에 분산*: GPU 가 queue-saturated 되어
GPU wait > CPU per-req 시간이 되는 영역. 본 fix 가 자동으로 cover.
H100+1.5B+500-burst 에서는 발생 안 함 (GPU wait 는 ~4s, CPU per-req 47s).

**Case (b)** — *다른 일을 다른 자원에서*: CPU 와 GPU 가 같은 요청을 처리
하지 않고 **본질적으로 다른 layer / 다른 step / 다른 precision** 을 처리.
이게 진짜 ninja gap. 본 인프라는 (b) 를 아직 구현하지 않음.

## 3. (b) 를 어떻게 구현할까 — 5가지 후보, 우선순위

| # | 항목 | 예상 wall 단축 (H100+1.5B/7B 기준) | 구현 비용 | 본 레포 기존 자산 |
|---|---|---|---|---|
| 1 | **Speculative decode CPU drafter** | 30~50% | 중 | `vllm/v1/spec_decode/`, `ngram_proposer_dynamic.py` |
| 2 | **KV cache CPU tier offload** | 단독 0% → batch ↑ 시 2~3× | 큼 | PagedAttention block table |
| 3 | **Long-context P/D disaggregation** | 32K+ context only, 0~80% | 중 | `vllm/engine/disaggregated/` (stub) |
| 4 | **AMX-INT8 CPU path 활성화** | 단독 0% → 1+3 의 곱셈 인자 | 작음 | `csrc/cpu/gemm_vnni.cpp`, `csrc/cpu/quant_q8_0.cpp` |
| 5 | TP across CPU/GPU | (-) PCIe BW 압살 | n/a | — (안 됨) |

### 3.1. **A1 — Speculative decode with CPU drafter** ⭐⭐⭐⭐⭐

핵심 구조:
- CPU EngineCore 에 작은 draft 모델 (예: Qwen2.5-0.5B, ~0.5B params, AMX BF16 로 ~50 tok/s) 을 띄움.
- GPU EngineCore 에는 메인 모델 (1.5B/7B/32B) 을 띄움.
- Draft 가 K=4~8 토큰 propose → GPU 가 single-step verify → accept 된 토큰 commit, reject 된 토큰부터 재draft.
- accept rate 60~80% 면 GPU TPOT 의 effective rate 가 1.5~2× 단축.

왜 이게 H100+SPR 환경에 잘 맞는가:
- CPU 의 "느림" 이 문제 안 됨 — draft 모델이 메인의 1/30~1/100 이라 AMX BF16 한 번 호출에 ~10ms. K=8 토큰 generate 가 80ms.
- 같은 80ms 사이에 GPU 가 K+1 토큰 verify (1 step ~22ms) + 다음 batch step → CPU 가 GPU 를 전혀 안 막음.
- 본 hybrid 의 dual-process 인프라 (별 ZMQ socket, identity dispatch, CapacityAwareRouter) 위에 draft engine 을 third process 로 추가하는 형태로 확장.
- AMX/IPEX 가 verified 상태이므로 CPU 쪽 디버깅 거의 없음.

구현 작업 (rough):
1. `HybridConfig` 에 `spec_decode_draft_model: str | None` 추가.
2. `launch_hybrid_engines` 가 num_cpu_engines + num_draft_engines 만큼 spawn.
3. `_route_throughput_adaptive` 옆에 `_route_speculative` 를 추가 (모든 요청은 GPU + draft 양쪽으로 fanout).
4. `process_engine_outputs` 에서 GPU verify result 와 CPU draft tokens 를 combine.
5. accept/reject 로직 (vLLM 의 V0 spec_decode 코드 차용).

가장 빠른 ninja gap 경로.

### 3.2. **A2 — KV cache CPU tier offload** ⭐⭐⭐⭐

핵심 구조:
- HBM (320 GB total) vs DDR5 (944 GB) — DRAM 이 ~3× 큼.
- Hot KV (recent N decode steps) → HBM, Cold KV (paused / long-context) → DRAM.
- Attention 시점에 cold block 을 DMA 로 HBM 으로 swap (또는 CPU 에서 attention 일부 수행 후 partial result 를 GPU 로 보냄).

왜 ninja gap 인가:
- 현재 H100+1.5B 에서 GPU 가 빠른 이유는 batch size 가 제한적 (~480) 인데
  실제 GPU compute 는 batch 1024+ 에서야 saturated. **HBM 이 KV 를 더 이상
  못 담아서 batch 가 안 늘어나는 것**이 천장.
- KV 를 CPU 로 빼면 동시 시퀀스 ~3000 까지 가능. GPU 가 진짜 saturated 되면
  total throughput 이 2~3× 증가. wall 단축.

구현 비용 큼:
- PagedAttention 의 block table 에 tier 추가.
- Eviction policy (LRU? recency-based?).
- DMA path (cudaMemcpyAsync + pinned host memory).
- Attention kernel 이 cold block 을 만나면 swap-in trigger.

paper 의 Property 2 를 가장 honestly 정량화할 수 있는 메커니즘. 하지만
구현 난이도는 spec decode 보다 훨씬 큼.

### 3.3. **A3 — Long-context Prefill/Decode disaggregation** ⭐⭐⭐⭐

- prefill: 큰 matmul (compute-bound). AMX BF16 가 H100 대비 2~5x 느리지만,
  long context (32K+) 에서는 GPU 도 prefill 동안 decode 를 못 함 → CPU 가
  prefill 떠맡으면 GPU TPOT 보호.
- decode: HBM3 BW 가 압도적. 무조건 GPU.
- 본 레포의 `vllm/engine/disaggregated/` 는 stub 만 있음. 본 hybrid 의
  process isolation + ZMQ routing 을 그 위에 얹으면 됨.

단점: input length 32K+ 워크로드가 아니면 효과 거의 없음. 워크로드 의존적.

### 3.4. **A4 — AMX-INT8 CPU path 활성화** ⭐⭐⭐

- AMX-INT8 peak TOPS 는 AMX-BF16 의 **2×** (Sapphire Rapids 기준).
- 본 레포에 이미 `csrc/cpu/gemm_vnni.cpp` (VNNI INT8 6×16 micro-kernel) +
  `csrc/cpu/quant_q8_0.cpp` 가 있음. **빌드는 되는데 런타임 dispatch 가
  안 됨** (`cpu_attn.py` 가 IPEX BF16 path 만 사용, `_C_cpu_ops` 의 INT8
  GEMM 호출 경로 없음).
- INT8 path 를 활성화하면 같은 CPU EngineCore 가 ~2× 빨라져 expected-finish
  비교에서 CPU 가 이기는 영역이 넓어짐 → A1/A2/A3 의 곱셈 인자.

작은 PR 로 측정 가능한 첫 단계.

### 3.5. **B1 — TP across CPU/GPU** ❌

- PCIe Gen5 ×16 = 64 GB/s. NVLink ~900 GB/s 의 1/14.
- per-layer all-reduce 비용이 compute 절약을 압살.

### 3.6. **B2 — Pipeline (CPU front layers, GPU rest)** ❌

- 직렬 dependency 라 wall 두 배. 다수 요청 pipelining 으로 amortize 가능
  하지만 FIFO depth + stall 관리가 끔찍.

## 4. 권고 — 다음 한 가지를 한다면

**A1 (speculative decode CPU drafter)** 을 우선.

이유:
- 본 hybrid 의 dual-process 인프라 + ZMQ routing + AMX/IPEX 검증 완료 →
  추가 디버깅 거의 없음.
- 효과가 정량적으로 측정 가능 (accept rate, TPOT delta).
- H100+1.5B/7B 같이 GPU 가 fast 한 환경에서도 작동 (vs A2 는 GPU saturated
  되어야만 의미).
- 본 routing fix (Property 2 expected-finish gate) 와 직교적 — 라우팅 fix
  를 망치지 않음.

**그 다음**: A4 (AMX-INT8 path 활성화) 로 CPU per-req 속도 2× 측정 → A1 의
draft model 을 INT8 로 돌리면 draft throughput 이 다시 2× 더 증가.

A2 / A3 는 워크로드 특화라 별도 demo workload 정의가 먼저 필요 (long
context, large batch).

## 5. Demo 워크로드 — 30B 급 모델 후보

`Qwen2.5-1.5B/7B` 는 H100x4 에 너무 작아서 ninja gap 측정의 의미가 약함
(GPU 가 압도적). **30B 급** 으로 올리면:
- HBM 메모리 압력이 의미 있게 발생 → A2 (KV offload) demo 가능
- AMX BF16 에서 30B prefill 이 H100 대비 ~5x 느리지만 의미 있는 비율 →
  A1 (spec decode) 의 draft model 도 큰 모델 가능
- GPU step time 이 길어져 spec decode accept rate 가 critical path

추천 후보 (다음 섹션 별도).

## 6. 다음 단계 액션

1. **A1 implementation plan** 별도 문서 작성 — `HybridConfig` 확장 필드,
   `launch_hybrid_engines` 의 third process spawn, `_route_speculative` 의
   fanout 로직, accept/reject path.
2. **A4 빠른 활성화 PR** — `cpu_attn.py` 에 INT8 dispatch path 추가, INT8
   per-req throughput 측정.
3. **30B demo 워크로드 정의** — 모델, num_prompts, input/output length 분포.
4. (별도) `OMP_PROC_BIND=close, OMP_PLACES=cores` 환경변수 추가.

## 7. 참조

| 항목 | 경로 |
|---|---|
| 본 fix (라우팅 expected-finish gate) | `vllm/v1/engine/hybrid_core.py:347-405` |
| 회귀 추적 + 수정 분석 | `experiment_result/20260411_141500_h100x4_qwen1.5b_routing_regression_root_cause_fix/` |
| 4-run 검증 (1.5B/7B × G/H) | `experiment_result/20260411_142900_h100x4_qwen1.5b_7b_gpu_only_vs_hybrid_4runs/` |
| AMX/AVX-512 CPU 커널 (활성화 안 됨) | `csrc/cpu/gemm_vnni.cpp`, `csrc/cpu/quant_q8_0.cpp`, `csrc/cpu/decode_gemv.cpp` |
| Spec decode 인프라 (stub) | `vllm/v1/spec_decode/ngram_proposer_dynamic.py` |
| Disaggregated serving (stub) | `vllm/engine/disaggregated/` |
| paper Property 2 정의 | `docs/paper/main.tex` §3 |
