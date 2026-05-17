# v1.6 1-step Timeline (Option A — sync result wait) — 2026-05-16 KST 재측정

> 목적: revert 후 정확한 **Option A (sync result wait)** 영역에서 timeline 재측정 + 도식 재생성.
> 측정 시각: 2026-05-16 18:17 (py-spy chrometrace) + 18:30 (nsys profile)
> 워크로드: 100p × 8192 (timeline 측정용 short workload)
> base HEAD: `7730f61dc` (Phase 3.1 + 3.3, sync mode)

## 0. cdec 의 "async" 정의 (용어 명확화)

NEO 의 cdec dispatch 는 **항상 2 단계 async**:

| 단계 | 의미 | 우리 measurement 영역 |
|---|---|---|
| **(1) cdec submit** | `cdec_executor.submit(pacpu_compute, ...)` — CPU pacpu 를 **별도 worker process** 에서 비동기 실행. main thread 즉시 return + future 객체 받음 | **항상 async ✓** (2주 측정 전체) |
| **(2) cdec result wait** | `cdec_future.result()` — main thread 가 CPU pacpu 결과를 받는 시점 | **두 옵션** ↓ |

### 두 옵션

| 옵션 | 위치 | 코드 | 행동 |
|---|---|---|---|
| **Option A (sync wait)** | same layer 의 `unified_attention_with_output` 안 | `attention.py:1148` | `cdec_future.result()` main thread blocking → wall path 위. **= NEO 원본 source 의 layer 안 sync 정합** |
| Option B (우리 자체 cross-layer drain 시도, 미완성) | pending queue 에 push, next layer 가 drain | `attention.py:1133-1145`, env gate `VLLM_NEO_ASYNC_CDEC=1` | `_neo_pending_cdec_queue` 에 push, next attention 호출 시 drain |

→ **2주 측정 전체 = Option A** (env unset). Option B 시도 (env=1) 시 starvation 으로 step 멈춤 (`drain timing` 미완성).
**중요**: 시도 단계 narrative 에서 Option B 를 "NEO §4.4 algorithm-correct path" 로 잘못 label 한 곳이 있음. 실제로는 **NEO 원본 source 에 cross-layer drain mechanism 없음** (`pending_cdec_queue` 는 우리 자체 추가 abstraction). NEO 원본 정합 = Option A. Option B 는 우리 자체 가속 시도일 뿐.

## 1. 측정 fact 비교 (Option A 재측정 vs 이전 sync vs async 시도)

### GPU kernel summary (60s capture, 8 worker 합)

| 항목 | sync 첫 측정 (16:33) | async 시도 (17:46) | **Option A 재측정 (18:30)** |
|---|---:|---:|---:|
| NCCL AllReduce count | 42,384 | 26,888 (−37%) | **43,360** ✓ sync 일치 |
| NCCL AllReduce avg time/call | 687 μs | 2,690 μs (+292%) | **691 μs** ✓ sync 일치 |
| NCCL AllReduce total | 29.13 s | 72.33 s | **29.96 s** ✓ |
| GEMM `nvjet_tst_256x152` count | 20,424 | 12,720 | **20,904** ✓ |
| FlashAttn count | 21,064 | 13,360 | **21,552** ✓ |

### CUDA API

| 항목 | sync 첫 측정 | async 시도 | **Option A 재측정** |
|---|---:|---:|---:|
| cudaEventSynchronize % | 9.6% | **21.5%** | **9.8%** ✓ sync 일치 |
| cudaLaunchKernel | 2.6% | 1.6% | 2.7% ✓ |
| cudaMemcpyAsync % | 0.6% | 0.4% | 0.6% ✓ |

### NVTX NCCL ranges

| 항목 | sync 첫 측정 | async 시도 | **Option A 재측정** |
|---|---:|---:|---:|
| `NCCL:ncclAllReduce` count | 42,392 | 26,888 | **43,367** ✓ |
| `NCCL:ncclAllReduce` avg μs | 53 | 54 | 53 ✓ |

→ **Option A 재측정 = sync 첫 측정과 완전 정합**. revert 정상 작동.

## 2. 1-step Timeline 도식

![Option A 1-step Timeline](./timeline_schematic.svg)

### 도식의 async 동작 영역 (sync wait 외)

| lane | async 영역 | 표현 |
|---|---|---|
| **GPU stream 0 (b0) ↔ stream 1 (b1)** | s0 위 batch[0] preproj + s1 위 batch[1] attention 의 **동시 launch** (NEO §4.4 forward_double Stage 0) | "s0 stream ↔ s1 동시 launch" 텍스트 |
| **GPU stream 1 (b1)** | b1 의 preproj/postproj/MLP **GPU 에서 b0 와 concurrent**. attention 만 cdec rows 면 GPU SKIP → cdec_executor 로 dispatch | "b0 와 concurrent ✓" 텍스트 |
| **cdec worker #1 + #2** | `max_workers=2` 의 **2 process concurrent pacpu**. layer i, i+2, i+4 ↔ i+1, i+3, i+5 분배 (소켓 라운드 로빈) | 2 sub-lane (worker #1, #2) 으로 분리 표시 |
| **cdec submit** | main thread 에서 `cdec_executor.submit(...)` 호출 즉시 return + future 받음, CPU pacpu 는 worker process 에서 비동기 시작 | cdec worker lane 의 시작 시점 |
| **async_output thread** | 별도 background thread. previous step 의 GPU event poll + token id emit, **main thread / cdec 와 concurrent** | 60s 전체 lane (background concurrent) |
| **Swap stream** D2H/H2D | `cudaMemcpyAsync` 별도 stream → GPU compute 와 wall overlap. **wall hide ✓** (PCIe bw 1.3% 만 사용) | "GPU compute 와 overlap → wall hide" 텍스트 |

### sync wait 영역 (★ Option A 의 비용)

| 위치 | 코드 | 시간 |
|---|---|---:|
| **`cdec_future.result()`** | `attention.py:1148`, same layer 의 attention 안 main thread blocking | **+24 ms / step** (★②) |
| `_drain_pending()` | `sub_batch_executor.py:269` (`forward_double` Stage 1 default stream) | Option A queue empty 라 no-op |
| `cudaEventSynchronize` | next layer 시작 시 prev layer GPU event wait | step time 의 9.8% (29.5s / 60s) |

(timeline 의 정량 fact 는 [`../timeline_v16_20260516/README.md`](../timeline_v16_20260516/README.md) 의 §1-3 와 동일 — Option A 동등 영역.)

### Option A 의 wall path 분해 (NEO 추가 61 ms / step)

| # | 영역 | 추가 시간 | 원인 |
|---|---|---:|---|
| ① | `Python attention.py hot path × 80 layer` | **+12 ms** | Python overhead, overlap mechanism 과 무관 |
| **②** | CPU pacpu time **>** GPU concurrent work time → cumulative GPU IDLE 누적 | **+24 ms** | overlap mechanism 작동 중. layer 당 CPU pacpu (~2.3ms) > GPU concurrent work (preproj+postproj+gdec, ~0.4ms) → 차이 누적 → GPU stream queue 빈 시점 발생. v1.6 의 ThreadPool 의 추가 overhead (max_workers=2 cap, GIL race) 가 S1-S9 대비 +6 ms 가중 |
| ③ | `swap_in launch + Python overhead + emit` | +25 ms | `_neo_handle_kv_swap` Python loop, ATen `index_kernel` GOMP, overlap 끝난 step 마감 |
| | **합** | **+61 ms** | vanilla 54 + 61 = NEO 115 ms |

## 3. Overlap mechanism 의 실제 동작 (정확)

v1.6 Option A 도 `cpu_communication_stream` (S2 이전부터 존재) 사용. transfer + swap + result_copy 가 GPU compute 와 진짜 동시 진행. cdec dispatch 가 ThreadPool 위 진행 동안 default stream queue 도 자유롭게 GPU 에서 실행 → **mechanism 자체는 NEO 원본 정합**. 단:

- ThreadPool 의 max_workers=2 cap → 56 cdec arrival vs 2 worker × 2.3ms/cdec → CPU pacpu wall ~64 ms
- 같은 시간에 GPU concurrent work (다른 sub-batch 의 preproj/postproj/gdec) wall ~40 ms
- 차이 ~24 ms 가 GPU IDLE 로 누적 (overlap 의 win 다 챙겨도 cdec wall 보다 GPU work 가 짧아서 남는 영역)

**즉 ② 의 +24 ms 의 원인 = "cdec wait blocking" 이 아니라 "CPU pacpu wall 이 GPU concurrent work wall 보다 길어서 남는 GPU IDLE 누적"**. v1.6 → S1-S9 의 −6 ms 단축 = ThreadPool overhead (queue contention + GIL race) 제거 → 같은 CPU pacpu 시간이지만 GPU stream queue 처리가 더 원활.

### Option B (우리 자체 cross-layer drain 시도, 미완성)

### Option B (우리 자체 cross-layer drain 시도, 미완성)

cdec_future 를 `_neo_pending_cdec_queue` 에 push, next layer 의 attention 시작 시 `_neo_drain_pending_cdec()` 가 oldest drain. layer 영역 wall path 가 cdec result wait 와 overlap 시도 (자체 가속).

**NEO 원본과의 관계**: `_neo_pending_cdec_queue` / cross-layer drain 의 **소스 코드가 NEO 원본 (`swiftllm/worker/layers/transformer_layer.py`) 에 없음**. 즉 Option B 는 우리가 추가한 자체 abstraction, NEO 원본 정합 작업이 아닌 자체 가속 시도. 시도 단계 narrative 에서 잘못 "NEO §4.4 algorithm-correct" label 붙임 — 실제 NEO §4.4 는 batch interleave / asymmetric pipeline 영역만 다루고 cross-layer cdec drain 은 다루지 않음.

**우리 implement 의 한계** (실측):
- Option B 활성 시 (env `VLLM_NEO_ASYNC_CDEC=1` 또는 module default True) → step 진행 거의 멈춤 (FORK STAT total 47,500 → 500, −99%)
- 가설: drain timing 동기화 미완성, NEO 원본의 `forward_first_stage` / `forward_double` / `forward_last_stage` 의 정확한 ordering 과 다른 drain 점 선택
- 코드 comment 도 정직히 적시: "Empirically OMP pool saturates ... regresses throughput"
- S3 (S1-S9 rewrite) 에서 Option B path 제거 → NEO 원본 source 100% 정합 회복

### Option B 완성 시 추정 효과 (NEO 원본 정합과 무관, 자체 가속 가능성)

NEO 원본대로 drain timing 정확히 동기화 후 Option B 활성 시:
- ② 영역 (+24 ms) 제거 가능 → NEO step 115 → 91 ms
- throughput 2,197 → **2,775 tps** (+26%)
- vanilla 4,690 의 59% 도달

단 우리 implement 의 미완성 영역 추가 작업 필요. 후순위 lever.

## 4. 파일

| file | 내용 | size |
|---|---|---:|
| [`timeline_schematic.svg`](timeline_schematic.svg) | Option A 영역의 1-step timeline 도식 (vanilla vs NEO) | ~10 KB |
| `nsys_stats/cuda_*.txt` | nsys profile summary (kernel, API, memcpy, NVTX, osrt) | ~30 KB |

raw 측정 data (heavier):
- py-spy chrometrace: `eval/results/20260516_181723_v16_timeline_pyspy/chrometrace/` (각 tp 130-148 MB)
- nsys profile: `eval/results/20260516_183037_v16_timeline_nsys/v16_timeline.nsys-rep` (178 MB)

## 4.1 ★ 결정적 원리 — sync wait = barrier (모든 async 효과 무력화)

`cdec_future.result()` 가 main thread 의 wall path 위에 있어 **barrier cascade** 발생:

```
main thread Python forward (1 layer)
  ├─ preproj GEMM launch (s0) ──┐
  ├─ cdec_future = executor.submit(pacpu) ── (worker process 비동기 시작)
  ├─ b1 preproj GEMM launch (s1) ──┤ s0/s1 concurrent
  ├─ skip_gpu_attn check
  └─ ★ cdec_future.result() ← BARRIER (sync wait)
        ↓ 이 시점에 다음 영역들이 대기:
        ├─ GPU stream0 → next layer preproj launch 못 함
        ├─ GPU stream1 → next layer 동시 launch 못 함
        ├─ async_output thread → next step GPU event poll 못 함
        ├─ swap_in launch → next step prep 못 함
        └─ 8 TP worker 모두 같은 NCCL all_reduce 직전에 stuck
```

### Barrier 가 wall 에 미치는 영향

| 항목 | NEO §4.4 의도 (Option B) | 우리 measurement (Option A) |
|---|---|---|
| **wall 공식** | `wall = max(b0 GPU, b1 CPU pacpu)` | `wall = b0 GPU + b1 CPU pacpu_remaining (sync wait)` |
| b0 GPU compute | 54 ms | 54 ms |
| b1 CPU pacpu (max_workers=2) | 64 ms | 64 ms 중 40 ms 가 GPU 와 overlap, **24 ms 가 barrier 뒤로 누적** |
| theoretical wall | max(54, 64) = **64 ms** | 54 + 24 = **78 ms** + 추가 (Python overhead + swap launch) +37 ms = **115 ms** |
| concurrent 효과 | b0/b1 동시 → win | barrier 가 cdec submit + s0/s1 stream + swap async + output async **모두 직렬화** → win 0 |

→ **코드의 모든 async 메커니즘 (cdec submit, s0/s1 concurrent stream, swap async stream, async_output thread) 가 wall 의미상 sequential**. barrier 한 곳이 chain 전체 묶음.

### Option B (NEO §4.4) 가 본질적으로 필요한 이유

Option B = `cdec_future.result()` 를 same layer 가 아니라 **next layer 의 시작 시 drain**:

```
main thread Python forward (layer i)
  ├─ s0 launches
  ├─ executor.submit(pacpu) → queue.append(future)
  ├─ s1 launches
  └─ continue → next layer (i+1)
      ├─ queue.popleft() → drain previous layer's future ← barrier 가 layer offset 만큼 뒤로
      ├─ s0 launches (i+1)
      └─ ...
```

→ barrier 가 *layer i+1 의 GPU compute 와 overlap* → CPU pacpu 가 GPU 와 진짜 concurrent → `wall = max(b0 GPU, b1 CPU)` 실현.

우리 implement 는 코드 구조 (queue + drain) 는 있되 *drain timing 동기화 미완성* → 활성 시 starvation. lever 약 +26%.

## 5. 결론

1. **2주 측정 전체 = Option A 영역** (cdec submit async + result wait sync). 정확한 fact.
2. **NEO 원본 source 정합 = Option A**. NEO 원본의 `_attention` 함수가 same-layer sync wait — 우리 Option A 가 그 정합. Option B = 우리 자체 cross-layer drain 시도, NEO 원본에 없는 추가 abstraction, 시도 단계 narrative 에서 잘못 "NEO §4.4" label 붙음.
3. **현재 v1.6 best 2,197 tps avg = Option A 영역의 best**. 추가 lever:
   - swap path Python+ATen overhead 제거 (③ 영역, After-NEO plan ★ Top Priority): +11-25%
   - cross-layer drain 자체 abstraction 완성 (NEO 원본 정합과 무관, 우리 자체 가속): +26% 가능성 (단 drain timing 동기화 fix 필요)
   - 두 lever 합산 시 vanilla 의 ~60-70% 도달 가능 (paper claim H100 +14% 는 vanilla 보다 *증가*, 실현 어려움)
