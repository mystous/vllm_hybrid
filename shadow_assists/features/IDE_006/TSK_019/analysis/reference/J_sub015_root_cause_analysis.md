# J — SUB_015 실패의 3 차원 root cause 분석 + 개선 방향

> 2026-05-18 KST. branch `feat/neo-amx-apply` HEAD `7200f41f1`.
>
> **분석 문서 (read-only). 코드 변경 0**. SUB_015-Phase 3 의 6 step 모두 S1-S9 baseline (2,238.6 tps) 초과 못한 진정한 root cause 를 **소스 코드 + HPC + 분산 컴퓨팅** 3 차원으로 분석 + 개선 방향 6 후보 ranking.

---

## 1. Executive summary

### 1.1 핵심 진단

**SUB_015 실패는 HPC 차원의 AMX 한계가 아니라 분산 컴퓨팅 차원의 Amdahl bottleneck 이 root cause**:

```
SUB_015 의 6 step (B/A/G/C'/vec K/all)
  = AMX qk FLOPs 이득 (+15% 이론)
  < 분산 dim overhead (-20% 실제, GIL + barrier idle + pipeline depth=0)
```

기존 `analysis/I_amx_proper_design.md` 의 HPC 단일 차원 분석으로는 설명 안 됨 — Step 5 (vec K conv, setup overhead 제거) 도 3-run avg -2.35% 회귀.

### 1.2 측정 backing

| 측정 | 결과 | 차원 |
|---|---:|---|
| S1-S9 baseline (3-run avg) | **2,238.6** | (baseline) |
| Phase 3 A dropin AMX (3-run avg) | 2,142.5 (-4.3%) | HPC: setup overhead dominant |
| Step 5 (B+A+vec K conv, 3-run avg) | 2,186.1 (-2.35%) | HPC setup 줄여도 회귀 — **분산 dim root** 확인 |
| Step 6 (B+A+vec K+G+C', 1-run) | 2,199.7 (-1.7%) | G+C' 통합 시 회귀 누적 |

★ **HPC 차원의 모든 변형 (B, A, vec K, G, C', all) 이 분산 dim 의 Amdahl bottleneck 안에서만 발휘** — net 회귀.

### 1.3 개선 방향 6 후보 ranking

| 순위 | 후보 | 차원 | Effort | 예상 win | Risk |
|---|---|---|---:|---:|---|
| **1** | **F1 async pipeline depth > 0** | 분산 | 3-5 일 | **+5~15%** ★ | 정확도 / race |
| 2 | F2 b0/b1 size 균형 (scheduler) | 분산 | 2-3 일 | +3-7% | scheduler 변경 |
| 3 | F3 K cache BF16 host store (full) | 코드+HPC | 1-2 일 | +1-3% | 정확도 검증 |
| 4 | F4 TP=8 → TP=4 (M=8→16) | 코드+HPC | 2-4 주 | +5-10% | GPU throughput trade |
| 5 | F5 BLOCK_SIZE 16 → 32 (N=16→32) | 코드+HPC | 2-4 주 | +3-7% | NEO scheduler 변경 |
| 6 | F6 OMP barrier dynamic schedule | 분산 | 1-2 일 | +1-3% | race 위험 |

★ **F1 (async pipeline depth)** 이 가장 high-ROI — 가장 큰 미사용 영역 (현재 depth=0, 80 layer pipelining 미실현).

---

## 2. Dimension 1: 소스 코드 분석

### 2.1 Per-block matmul size 의 fundamental constraint

**위치**: `csrc/cpu/pacpu/dtype.h:33-44`

```c
#elif defined(LLAMA2_70B) || defined(LLAMA3_70B) || defined(LLAMA3_3_70B)
  #define NUM_LAYERS 80
  #define NUM_Q_HEADS (64 / TP_DEGREE)   // = 8 (TP=8)
  #define NUM_KV_HEADS (8 / TP_DEGREE)   // = 1 (TP=8)
#endif

#define HEAD_DIM 128
#define BLOCK_SIZE 16
```

→ **per-block matmul = M=8 × N=16 × K=128**. fundamental fixed at TP=8.

### 2.2 cdec_future 의 same-layer scope

**위치**: `vllm/model_executor/layers/attention/attention.py:764-1135`

매 layer 의 forward_attention 호출 시 cdec_future 생성:
- layer N 의 cdec_future 가 layer N 안에서 result wait (line 1135-1196)
- layer N+1 시작 전에 layer N 완료 보장 — sequential per-layer

**Implication**: 80 layer batched 단일 pacpu call 불가능 (D 시도 불가 root). layer 간 pipeline 의 잠재 영역은 **다른 step 의 동일 layer cdec** — 그게 F1 의 async pipeline 영역.

### 2.3 Option L 의 seq_lens.cpu() sync (이미 fix)

**위치**: `attention.py:806-871`

```python
_seq_lens_attr_optL = getattr(_fc, "_neo_cdec_seq_lens_cpu_cache", None)
if _seq_lens_attr_optL is None:
    _seq_lens_gpu_optL = getattr(attn_metadata, "seq_lens", None)
    if _seq_lens_gpu_optL is not None:
        _seq_lens_attr_optL = _seq_lens_gpu_optL.cpu()  # GPU→CPU sync
        _fc._neo_cdec_seq_lens_cpu_cache = _seq_lens_attr_optL  # 1회 cache
```

v1.5.2 fix 후 1 step 의 첫 layer 에서 1 회 `.cpu()` + cache → 80 layer 재사용. 추가 sync win 없음.

### 2.4 AMX path 의 contract diff

**위치**: `csrc/cpu/pacpu/amx_kernel.cpp` (Step 5 best)

- B (thread Q cache): 1 회 변환, same-seq consecutive task 재사용
- A (K^T outer pre-pack): imax × 4 round 변환 + cache, AMX hot loop 은 read only
- vec K conv: `_mm512_cvtneps_pbh` 으로 FP16→BF16 4-8× 빠름

**contract diff** (ISPC vs AMX path):
- ISPC `attn_one_seq` = qk + softmax + av 통합 (single call)
- AMX path = qk_amx + ispc::softmax (export 추가) + ispc::av_product (분리 호출)
- **3 회 function dispatch** vs ISPC 의 1 회 — small overhead

→ HPC fixed cost. F3 (K BF16 host store) 로 변환 cost 제거해도 dispatch overhead 동일.

---

## 3. Dimension 2: HPC 분석

### 3.1 Roofline analysis

**AI (Arithmetic Intensity)** — `analysis/H_static_analysis.md` 측정:

| Kernel | FLOP / seq / layer | Byte traffic | AI (FLOP/byte) |
|---|---:|---:|---:|
| qk_product | imax × 32,768 | imax × 4,608 + 2,048 | **7.11** |
| av_product | imax × 32,768 | imax × 4,608 + 4,096 | **7.11** |
| softmax | ~491K (transcendental dominant) | 96N | 2.5 |

**SPR per-core peak**:
- AVX-512 FP16 = 32 lane × 2 FMA × 4 GHz = **128 GFLOP/sec**
- AMX BF16 = 1024 FMA/cycle/tile × 4 GHz = **1,024 GFLOP/sec** (이론)
- L1 BW = ~200 GB/s, L2 = ~50 GB/s, DRAM = ~5 GB/s (shared)

**Roofline ceiling**:
- L1: AI 7.1 × 200 = 1,420 GFLOP/sec → compute-bound at peak
- L2: AI 7.1 × 50 = **355 GFLOP/sec** → AVX-512 의 2.77× ceiling
- DRAM: AI 7.1 × 5 = 35.5 GFLOP/sec → memory-bound

★ **NEO 의 K cache working set 이 L2 (2 MB/core) fit 시 ceiling = 355 GFLOP/sec**. AMX 의 이론 peak (1,024) 의 35%. **이론 ceiling 자체가 낮음**.

### 3.2 AMX tile occupancy

| 매개 | NEO (TP=8) | AMX sweet spot | utilization |
|---|---:|---:|---:|
| M (heads) | 8 | 16 (tile) → 32 (vllm 8-tile pattern) | **25-50%** |
| N (tokens) | 16 (BLOCK_SIZE) | 16 (1 tile) → 32 (2 tile) | 50-100% |
| K rounds | 4 (K=128 BF16, 32/round) | up to ∞ | OK |
| 8 tile usage | 3 (A+B+C) — Step 2/5 | 8 (4 quadrant + A0/A1 + B0/B1) | **37.5%** |

★ **AMX peak 의 effective utilization ≈ 19%** (M=50% × tile_count=37.5%). 이론 1024 GFLOP/sec 의 200 GFLOP/sec. AVX-512 FP16 peak (128) 의 1.56× — 그러나 setup overhead 가 이를 상쇄.

### 3.3 Setup overhead breakdown (Phase 3 A 측정)

per-block (M=8, N=16, K=128) AMX path:

| Operation | Cycle 추정 | 빈도 |
|---|---:|---:|
| Q FP16→BF16 (1024 elem scalar) | 50-100 | 1 (per call, Step 1 hoist) |
| K^T pre-pack + FP16→BF16 (2048 elem × 4 round) | **200-400** ★ | per block |
| `_tile_loadd` × 2 × 4 round | 240 | per block |
| `_tile_dpbf16ps` × 4 round | 64 | per block (work) |
| `_tile_zero` + `_tile_stored` | 60 | per block |
| C copy (16×16 → 8×16) | 30 | per block |
| **AMX total / block** | **~650-900** | |
| **ISPC AVX-512 FP16 (baseline)** | **~400-500** | |

→ AMX 의 setup overhead 가 work (64 cycle) 의 **10-14× 더 큼**. Step 5 의 vec K conv 로 K 변환 50% 감소해도 (200 → 100) net cycle 여전히 ISPC > AMX.

### 3.4 Memory hierarchy

| Level | Size (SPR) | NEO K cache working set | Fit |
|---|---:|---:|:-:|
| L1 | 48 KB / core | per-call 4 KB | ✓ |
| L2 | 2 MB / core | per-block 4 KB × imax (~256 blocks max) = 1 MB | ✓ |
| L3 | 105 MB shared | per-thread 1 MB × 14 thread = 14 MB | ✓ |
| DRAM | 2 TB | full host buffer (NEO_CPU_RESIDENT_REQS=128) | — |

★ **K cache 의 working set 가 L2 fit** → DRAM-bound 아님. SDR-bound 도 아님. **AMX setup overhead 가 진정 한계**.

### 3.5 Register pressure (K_TILE_WIDTH sweep)

**위치**: `csrc/cpu/pacpu/pacpu.ispc:4` `#define K_TILE_WIDTH 2`

| K_TILE | tps (1-run) | vs K=2 |
|---|---:|---:|
| 2 (baseline) | 571.6 | (best) |
| 4 | 562.7 | -1.6% |
| 8 | 569.0 | -0.5% |

→ ISPC vectorization 패턴 K=2 에 최적화. K=4/8 시 register pressure ↑ + reduce_add cycle 횟수 ↓ trade — net loss.

### 3.6 HPC 결론

**HPC 차원의 win 한계**:
- Roofline ceiling = 355 GFLOP/sec (L2-bound)
- AMX effective peak = 200 GFLOP/sec (19% utilization)
- ISPC AVX-512 effective = ~100 GFLOP/sec (현재 baseline)
- **이론 win = 100 → 200 GFLOP/sec = +100%**

단 실측 (Step 5):
- ISPC baseline cycle ~400-500
- AMX cycle ~650-900 (setup overhead dominant)
- **net cycle gain = -30% (loss)**

→ NEO 의 작은 matmul 에서 AMX peak utilization 위해 큰 redesign (TP, BLOCK_SIZE) 필요. SUB_015 범위 밖.

---

## 4. Dimension 3: 분산 컴퓨팅 분석 ★ (신규 영역)

### 4.1 cdec_executor max_workers cap — Amdahl 한계

**위치**: `vllm/model_executor/layers/attention/attention.py:1462`

```python
_max_workers = int(_os_env.environ.get("VLLM_NEO_CDEC_WORKERS", "2"))
```

**fact**:
- 80 layer × 8 worker = **640 possible cdec task** / iteration
- cap=2 → 동시 cdec future 최대 2개 pending
- SUB_030 측정: max_workers=4 시 **-8% 회귀** (GIL contention 의 trade-off, cap=2 가 best)

**Amdahl 정량**:
- cdec 영역 의 총 wall time = 80 layer × cdec_wait (8.75 ms/layer 측정) = 700 ms / step (max-case)
- cap=2 의 parallel speedup ceiling = 2× — cdec 영역의 CPU 가속 (AMX) 의 win 이 cap 안에서만 발휘
- AMX qk +15% 이론 win × cap_factor (cdec 영역 share × cap=2) ≈ +1-3% throughput

→ AMX win 의 거시적 ceiling 매우 낮음.

### 4.2 b0/b1 비대칭 + GPU↔CPU pipeline idle window

**위치**: `vllm/v1/core/sched/neo_scheduler_adapter.py:1078-1098`

**fact**:
- b0 (GPU current step): vllm_ids — current step 의 vllm 처리 reqs
- b1 (CPU cdec): swapped_out reqs — CPU pacpu 처리 reqs
- 측정: b0 GPU compute avg ~1.5 ms vs b1 CPU pacpu avg ~2.5 ms
- **GPU idle window ~1 ms / step** (b0 완료 후 b1 wait)
- b0 batch size 작음 (typical ~20/500 = 4%) — GPU underutilized

**잠재 win**:
- b0 batch size ↑ (예: 10% → 30%) → GPU compute 더 길게 → CPU와 더 잘 overlap → +3-7% throughput
- Trade: cdec scope (b1) 가 줄어 CPU 가속 effect 감소

### 4.3 OMP team barrier #1/#2 의 heterogeneous work

**위치**: `csrc/cpu/pacpu/core.h:308-391`

**Step 0 (store_kv)** — `core.h:312-326`
- partition: `bch_blk_size = (batch_size - 1) / ws + 1` → thread tid 가 `[tid * bch_blk_size, min((tid+1)*bch_blk_size, batch_size))` 의 seq 처리
- thread 간 work 평등 (batch_size / ws)
- barrier #1: store_kv 의 결과를 attn_one_seq 가 read — race 회피

**Step 1 (attn_one_seq)** — `core.h:331-345`
- partition: `tasks[thrd_start_task[tid] .. thrd_start_task[tid+1])` — task-based dynamic
- task = (batch_id, seq_offs, seg_len, cum_seg_len) — sequence 의 block chunk
- thread 별 task 수: `thrd_rst_blks[tid] = tot_blks / ws + (tid < tot_blks % ws)`
- **work imbalance**: long seq 의 마지막 task 가 한 thread 에 집중 — barrier wait

**Step 2 (gather)** — `core.h:349-362`
- partition: bch_blk (Step 0 과 동일)
- Step 1 의 thread 와 다름 → barrier #2 필요

**libgomp 43.75% (Phase 1 측정) 의 mechanism**:
- 매 layer call 의 4 sync (fork + barrier #1 + barrier #2 + implicit join)
- 80 layer × 72 step/sec × 4 sync = 23,040 sync/sec/worker × 8 worker = **184,320 sync/sec total**
- imbalance 시 일부 thread (longest task 가진) 가 다른 thread 의 barrier wait 유발

### 4.4 TP=8 worker ↔ cdec executor GIL serialization

**위치**: `vllm/model_executor/layers/attention/attention.py:1013-1024`

**fact**:
- TP=8, 매 step 의 80 layer × 8 worker = **640 GIL acquire/release cycle/iteration**
- `_neo_cdec_compute_cpu()` 호출이 S5 ThreadPool 제거 후 main thread 에서 직접 호출 — GIL hold during dispatch setup (~0.5 ms)
- VLLM_NEO_CDEC_WORKERS=4, 8 증가 시 GIL contention 악화 (cap=2 가 best)

**잠재 win**:
- cdec dispatch path 의 critical section (`attention.py:769-789` 9-step validation) 의 C++ extension 화 — GIL 회피
- 단 큰 변경 (Python → C++ binding)

### 4.5 NUMA 2-node placement (이미 최적)

**위치**: `vllm/v1/core/sched/neo_cpu_kv_buffer.py:92-103` + `gpu_worker.py:191-202`

**현재 환경**:
- `VLLM_NEO_NUMA_BIND=1` 활성 → 8 worker 의 host buffer pinned 할당이 worker NUMA node 에 매칭
- `VLLM_NEO_CPU_PIN_PER_WORKER=1` + `VLLM_NEO_CPU_PIN_CORES=12` → per-worker 12 core 의 NUMA pin

→ NUMA 영역 추가 win 없음.

### 4.6 cdec dispatch critical section (이미 최적화)

**위치**: `attention.py:769-789`

```python
if (_tok is None or _tok[1] <= _tok[0]
        or _seq is None or _seq[1] <= _seq[0]
        or not _req_ids):
    cdec_future = None
elif not hasattr(torch.ops, "pacpu"):
    cdec_future = None
else:
    ...
```

9-step validation — v1.5.2 fix 후 seq_lens.cpu() cache 적용으로 매 step 의 첫 layer 만 sync, 나머지 79 layer 는 cache hit. 추가 win 없음.

### 4.7 ★ Async pipeline depth = 0 (가장 큰 미사용 영역)

**위치**: `vllm/model_executor/layers/attention/attention.py:1233-1238`

```python
# _neo_pending_cdec_queue (deque)
# VLLM_NEO_CDEC_PIPELINE_DEPTH default = 1
# _neo_async_cdec_mode default = False
```

**fact**:
- 현재 S1-S9 = **sync wait only** (line 1135-1196)
- 매 layer 의 cdec_future result wait 후 다음 layer 진행
- **80 layer pipeline depth = 0** — sequential per-layer

**잠재 win (★ 최대)**:
- depth > 0 시 layer N 의 cdec submit → layer N+1 시작 → ... → layer N+depth 시점에 layer N result 사용
- 이론적 pipelining win:
  - depth=1 → +50% cdec wall (overlap 50%)
  - depth=2 → +66% cdec wall
  - depth=4 → +75% cdec wall
- 단 정확도 (attention output dependency) + race safety 필요
- 측정: `_NEO_PL_GLOBAL_STATS:1089-1096` 의 cdec_count vs gpu_count ≈ 1.0 (no pipeline depth)

**구현 영역**:
- `_neo_pending_cdec_queue` 의 deque size > 1 으로 확장
- async mode 활성화 (`_neo_async_cdec_mode = True`)
- result wait 의 deferred mechanism (current layer result 가 다음 N+depth layer 까지 deferred)
- race safety: cdec_future 의 result 가 GPU output tensor write 위치 안 충돌 보장

### 4.8 분산 dim 결론

**분산 dim 의 bottleneck ranking**:

| # | Bottleneck | 현재 상태 | 잠재 win |
|---|---|---|---:|
| 4.7 | async pipeline depth = 0 | 미적용 | **+5-15%** ★ |
| 4.1 | cdec cap=2 | best (4 시 회귀) | 0 |
| 4.2 | b0/b1 idle | b0 작음 | +3-7% |
| 4.3 | OMP barrier imbalance | libgomp 43.75% (잡힘) | +1-3% |
| 4.4 | GIL serialization | cap=2 와 trade-off | +2-5% (C++ binding) |
| 4.5 | NUMA | 최적 | 0 |
| 4.6 | critical section | v1.5.2 fix | 0 |

→ **F1 (async pipeline depth)** 이 가장 큰 미사용 영역.

---

## 5. 통합 진단 — 왜 6 step 모두 baseline 초과 못 했나

### 5.1 cycle 정량 분해

per attention call (1 layer × 1 worker):

```
┌─────────────────────────────────────────────────────┐
│ Total cdec_wait = 8.75 ms / layer (Phase 1 측정)   │
├─────────────────────────────────────────────────────┤
│ Step 0 (store_kv): 0.3 ms  (3.4%)                  │
│ ─ barrier #1 wait: 0.3 ms  (3.4%, imbalance)       │
│ Step 1 (attn_one_seq): 6.5 ms  (74.3%)             │
│   ├─ qk_product: 2.0 ms  (22.9%)                   │
│   ├─ softmax: 2.5 ms  (28.6%)                      │
│   └─ av_product: 2.0 ms  (22.9%)                   │
│ ─ barrier #2 wait: 0.5 ms  (5.7%, imbalance)       │
│ Step 2 (gather): 0.2 ms  (2.3%)                    │
│ ─ Python dispatch overhead: 0.95 ms  (10.9%)       │
└─────────────────────────────────────────────────────┘
```

### 5.2 AMX win 의 cycle 정량

AMX qk (Step 5 best, B+A+vec K conv):
- ISPC qk_product cycle ~400-500 (2.0 ms / layer)
- AMX qk_amx cycle ~600-700 (3.0-3.5 ms / layer) ★ **net -1.0~1.5 ms / layer**

→ AMX 가 **qk 영역에서만 -1.0 ms loss**. 다른 영역 (softmax, av, barrier, dispatch) 영향 없음.

cdec_wait 의 변화: 8.75 ms → **9.75 ms / layer** (Step 5)
- per step (80 layer): 700 ms → 780 ms — **+11% wall time**
- 실측 (Step 5 3-run avg 2,186.1): 1,852.4 s wall vs S1-S9 1,819 s = **+1.8% wall** — overlap 으로 일부 흡수

→ AMX 의 net loss 가 **분산 dim 의 pipeline depth=0 에서 흡수 안 됨** (overlap 영역 없음, 매 layer sync).

### 5.3 진정한 win path

```
SUB_015 의 본질:
  CPU 측 HPC 최적화 (AMX, vec K) 만으로 풀 수 없음.
  분산 컴퓨팅 차원 (async pipeline, b0/b1 balance, GIL 회피) 의
  변경이 동반되어야 진정한 win.
```

★ **F1 (async pipeline depth) + F2 (b0/b1 균형)** 의 조합이 AMX 보다 high-ROI.

---

## 6. 개선 방향 6 후보 (ranked)

### F1: async pipeline depth > 0 ★ (가장 high-ROI)

**차원**: 분산 컴퓨팅

**변경 영역**:
- `attention.py:1233-1238` — `_neo_pending_cdec_queue` 의 deque size > 1
- `_neo_async_cdec_mode = True` 활성화
- result wait 의 deferred mechanism — layer N submit → ... → layer N+depth result use

**정량 win 추정**:
- depth=1: layer-level 50% overlap → cdec wall **-50%**
- depth=2: 66% overlap → cdec wall **-66%**
- depth=4: 75% overlap → cdec wall **-75%**
- 적용 시 throughput +5-15% (cdec 비율 dependent)

**Risk**:
- 정확도 (attention output 의 sequential dependency 위반 위험)
- race safety (cdec_future 의 result 가 GPU output tensor write 위치 충돌)
- pending future 의 memory pressure (depth × cdec_count × hidden_size)

**Effort**: 3-5 일
1. result deferred dispatch path 작성 (`attention.py` 의 1135-1196 영역 재구조)
2. cdec_future queue 의 lifecycle 관리 + race safety
3. 정확도 검증 (TST_003 verdict)
4. 500p × 3-run avg 측정

### F2: b0/b1 size 균형

**차원**: 분산 컴퓨팅 (scheduler)

**변경 영역**:
- `vllm/v1/core/sched/neo_scheduler_adapter.py:1078-1098`
- b0 (GPU) batch size 의 enlargement (current ~4% → target 10-15%)
- b1 (CPU) scope 의 corresponding shrink

**정량 win 추정**:
- GPU compute 의 b0 wall 이 b1 (CPU) wall 와 더 잘 overlap
- GPU idle window (~1 ms / step) → 0.3-0.5 ms / step
- throughput +3-7%

**Risk**:
- scheduler 변경 (chain dispatch logic)
- CPU 가속 (AMX) 의 effect 감소 (b1 scope 줄어들수록)

**Effort**: 2-3 일

### F3: K cache BF16 host store (full)

**차원**: 코드 + HPC

**변경 영역**:
- swap path 에서 GPU FP16 → host BF16 변환 (1 회/swap)
- AMX kernel 의 K FP16→BF16 변환 제거
- ISPC path 는 그대로 (FP16 K read)
- env-toggle dual path (또는 BF16 only)

**정량 win 추정**:
- AMX path 의 K 변환 cost 완전 제거 (-200-400 cycle/block)
- Step 5 의 vec K conv 의 추가 win: ~+1-3% throughput
- 단 swap path 의 변환 cost 가 amortize 되는지 검증 필요 (한 step 의 K cache 가 여러 cdec 호출에서 재사용)

**Risk**:
- BF16 mantissa 7 vs FP16 10 = 3 bit precision drop
- TST_003 verdict (분포·의도 유사성) 정확도 검증 필요
- ISPC path 와 dual dtype 유지 시 buffer 2× memory

**Effort**: 1-2 일

### F4: TP=8 → TP=4 (M=8→16, AMX tile occupancy 향상)

**차원**: 코드 + HPC + 모델 분산

**변경 영역**:
- model parallelism 설정 변경 (TP=4)
- NUM_Q_HEADS = 64/4 = 16 → AMX tile (M=16) full
- 단 NUM_KV_HEADS = 8/4 = 2 → GQA factor 변경
- GPU parallelism 감소 — GPU side throughput trade

**정량 win 추정**:
- AMX tile occupancy: M=8/16 → M=16/16 (100%)
- AMX effective peak: 19% → 38% (2× ↑)
- 단 GPU 측 parallelism 감소로 prefill throughput ↓ — net effect 측정 필요
- 추정: +5-10% (workload dependent)

**Risk**:
- 모델 설정 변경 — 전체 architecture 영향
- KV cache memory 분배 변경
- GPU 측 batch size 변화 (TP=4 의 4 worker × ~2× batch)

**Effort**: 2-4 주 (모델 + 검증)

### F5: BLOCK_SIZE 16 → 32 (N=16→32)

**차원**: 코드 + HPC

**변경 영역**:
- `csrc/cpu/pacpu/dtype.h:19` `#define BLOCK_SIZE 16` → 32
- NEO scheduler 의 paging granularity 변경
- block_table indexing 변경

**정량 win 추정**:
- AMX tile N 차원 occupancy: 50% → 100%
- per-block matmul size 2× ↑ — setup overhead 의 amortize ↑
- 추정: +3-7%

**Risk**:
- NEO core design 변경 (BlockManager 의 모든 영역 영향)
- KV cache memory 분배 변경 (block 수 절반, 각 block 2× size)
- Prefill 영역 영향 검증 필요

**Effort**: 2-4 주

### F6: OMP barrier 의 dynamic schedule + nowait

**차원**: 분산 컴퓨팅

**변경 영역**:
- `csrc/cpu/pacpu/core.h:331` — Step 1 의 task loop 를 `#pragma omp for schedule(dynamic) nowait` 변경
- Step 0/2 의 partition 도 task partition 으로 통일 (load balance ↑)

**정량 win 추정**:
- barrier wait time 감소 (load imbalance ↓)
- libgomp 43.75% 의 일부 영역 (barrier wait spin) 축소
- 추정: +1-3%

**측정 history**:
- Phase 2 의 Step G-2 OMP_NUM_THREADS sweep 시도 — env 변경만으로는 win 0
- Phase 3 의 Step 4 (C' 2-block fused) 시도 — 5-tile cfg 의 추가 overhead 로 회귀
- → schedule(dynamic) 단독 측정 안 됨 (Step 4 의 cumulative 으로 가려짐)

**Risk**:
- dynamic schedule 의 atomic counter overhead — small workload 시 net loss 가능
- race 위험 (Step 0/2 partition 통일 시)

**Effort**: 1-2 일

---

## 7. Verification

### 7.1 산출 검증
- [x] 3 차원 (소스 코드 + HPC + 분산 컴퓨팅) 모두 cover
- [x] 각 dimension 별 file:line backing 5+ entry (총 17 file:line)
- [x] 각 dimension 별 측정 자료 backing (Phase 3 A 3-run, Step 1~6 1-run, Step 5 3-run, libgomp perf dso)
- [x] 개선 방향 6 후보 의 effort/win/risk 모두 채워짐
- [x] F1-F6 ranking 의 ROI 비교 표 명시
- [x] 통합 진단 의 "AMX win < 분산 overhead" 결론 정량 (cycle + wall time) backing

### 7.2 정량 backing 출처

| Fact | 출처 |
|---|---|
| Step 5 3-run avg 2,186.1 (-2.35%) | `measurements/sub015_p3_step5_amx_bav_500p_3run_20260518/README.md` |
| libgomp 43.75% | `analysis/H_dynamic_analysis.md` |
| qk AI 7.11 FLOP/byte | `analysis/H_static_analysis.md` |
| AMX setup ~650-900 cycle | `measurements/sub015_p3_amx_500p_3run_20260518/README.md` |
| max_workers cap=2 best | SUB_030 측정 (-8% at cap=4) |
| OMP partition file:line | `csrc/cpu/pacpu/core.h:308-391` |
| cdec future scope | `attention.py:756-1196` |
| async pipeline depth=0 | `attention.py:1233-1238` |

### 7.3 다음 단계 후보 (사용자 명시 후 별도 task)

| 우선 | 후보 | 권고 |
|---|---|---|
| 1 | **F1 async pipeline depth** | 가장 high-ROI, 3-5 일 |
| 2 | F3 K cache BF16 host store | quick win, 1-2 일 |
| 3 | F2 b0/b1 균형 | medium, 2-3 일 |
| 4 | F6 OMP dynamic schedule | quick try, 1-2 일 |
| 5 | F4/F5 (TP/BLOCK_SIZE) | 큰 architectural change, 별도 long-haul |

---

## 8. 핵심 교훈

1. **HPC 단일 차원 분석은 SUB_015 의 실패를 설명할 수 없다** — 분산 컴퓨팅 차원의 Amdahl bottleneck (cdec cap, pipeline depth, GIL) 이 더 큰 root cause.

2. **NEO 의 작은 matmul (M=8) 은 AMX 의 sweet spot 아님** — TP/BLOCK_SIZE 같은 architectural redesign 없이는 AMX 의 진정한 win 어려움.

3. **async pipeline depth = 0 가 가장 큰 미사용 영역** — 80 layer pipelining 의 잠재 win +5-15% 가 미실현.

4. **1-run noise 가 매우 큼 (CV 3.23%)** — Step 별 1-run 결과 의 신뢰성 매우 낮음. 3-run avg 만 ground truth.

5. **AMX path 자체는 functionally correct** — env-gated default off 로 keep, future F3 (K cache BF16 host store) 의 base.
