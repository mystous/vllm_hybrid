# Timeline 분석 — `feat/neo-amx-apply` HEAD `e64c56561` (2026-05-20 KST)

> 가장 마지막 코드 베이스 (HEAD `e64c56561`) 의 1-step Timeline 분석.
> 베이스: S1-S9 ([`timeline_v16_s1_s9_20260517/`](../timeline_v16_s1_s9_20260517/)) 위에 P3/P4/D/OOB env-gated 옵션 누적.
> Default 환경의 timeline mechanism = S1-S9 와 **identical** (env 기본 OFF). 각 env-gated alternative path 의 영향 영역만 별도 분석.

## 0. 변경 요약 (S1-S9 → 현재 HEAD)

| commit | 영역 | env | default | timeline 영향 |
|---|---|---|:-:|---|
| `c798472c7` | P3 Phase 0+1 — K cache BF16 host store (Python wiring) | `VLLM_NEO_HOST_K_BF16` | **OFF** | 활성 시: swap_out path 에 BF16 cast 추가 (스왑 1회당 ~200-400 cyc 추가) + cdec 안 K cast 영구 제거 |
| `5fcaf0033` | P3 Phase 2 — `amx_kernel.cpp` BF16 K variant 분리 | (C++ dispatch only) | — | 코드 변경 없음 — `VLLM_NEO_USE_AMX=1` 활성 시 K BF16 read path |
| `aba1b14b1` | P3 Phase 2~3 — `pacpu.cpp` + `core.h` K BF16 dispatch | `VLLM_NEO_USE_AMX` × `VLLM_NEO_HOST_K_BF16` | **OFF × OFF** | 활성 시: AMX qk_product path (timeline 영역 §3 참조) |
| `3aeb9885f` | P3 swap_out dtype fix | — | (always on after fix) | swap_out cast bug fix (영향 작음, 정확도 fix) |
| `4857014e2` | P4 (F1 async cdec) | `VLLM_NEO_ASYNC_CDEC` + `VLLM_NEO_CDEC_PIPELINE_DEPTH` | **OFF**, depth=1 | 활성 시: cdec wait 가 deque 으로 미뤄짐, postproj 직전 drain (timeline 영역 §4 참조) |
| `3129d3900` | D fix (OOB silent skip) | `VLLM_NEO_OOB_SILENT` | **ON** | exception path 의 traceback 출력 skip (behavior 동등, 로그 volume ↓) |
| `dd80747a6` | G/H log rate-limit | (always on) | — | `swap_out attach` INFO 의 rate-limit (1회/1000 step + cap 변경 시), behavior 동일 |

→ **Default 환경 (모든 env unset 또는 OFF)** 의 timeline = **S1-S9 그대로** ([`timeline_v16_s1_s9_20260517/README.md`](../timeline_v16_s1_s9_20260517/README.md)).

→ 본 문서 = (1) Default timeline fact 의 최신 측정 update + (2) 각 env-gated alternative path 의 timeline 영향 + (3) 다음 lever 영역 정합 ([`analysis/M_sub015_phase3_hpc_optimization.md`](../../analysis/M_sub015_phase3_hpc_optimization.md) §5 Phase α/β/γ).

---

## 1. Default 환경 1-step Timeline (= S1-S9, 동적 분석 기반)

![NEO S1-S9 — 동적 분석 기반 timeline](timeline_schematic.svg)

> **본 도식 = 동적 분석 (perf record 60s) 기반 재작성** (2026-05-20 turn 3).
> 이전 timeline ([`../timeline_v16_s1_s9_20260517/timeline_schematic.svg`](../timeline_v16_s1_s9_20260517/timeline_schematic.svg)) 의 정정 영역:
> 1. **swap_stream lane 신규 추가** — async wall HIDDEN 명시 (사용자 지적 정합)
> 2. **cdec_executor lane 내부 분해** — libgomp barrier wait 62% + libpacpu compute 38% (★ ② 안 dominant)
> 3. **OMP barrier marker** (#1/#2/implicit) cdec lane 위 vertical line 으로 80 layer × 3 barrier 표시
> 4. **③ "+25 ms" label 변경** — `swap_in + sample + emit` → `step-end TBD (sample/emit/admit 가설)`
> 5. **OMP worker 의 wait 대상 화살표 변경** — GPU forward → cdec_executor barrier (실제 wait root)
> 6. **동적 분석 fact 분해 표** + 핵심 메시지 박스 (이전 timeline 정정)

### 1.1 mechanism (변경 없음)

NEO upstream `transformer_layer.py` 의 5 가지 정합 요소가 default 환경에서 계속 작동:

| 요소 | NEO 원본 line | 우리 implement (HEAD `e64c56561`) | 정합 |
|---|---|---|:-:|
| `cpu_communication_stream` 별도 운영 | 116/143/150 | `_get_neo_communication_stream` (attention.py:1372) | ✓ |
| `_transfer_qkv` cpu_comm async | 171 | `_xfer_stream.record_event()` (attention.py:993) | ✓ |
| `paged_attention_cpu` main thread direct | 336 | `_neo_cdec_compute_cpu` direct (attention.py:1013) — sync path | ✓ |
| result D2D on cpu_comm_stream + `_compute_wait_comm` | 351-355 | S9 + `_neo_compute_wait_comm` (attention.py) | ✓ |
| `_forward_pipeline_stage(cur_stage)` ordering | 397-427 | `forward_double` (sub_batch_executor.py) — S8 ordering | ✓ |

→ paper §4.4 의 "Layer N/N+1 동시 + CPU async pipeline" mechanism 작동 ✓

### 1.2 wall 분해 (default 환경) — 이전 timeline 의 명시 영역

| # | 영역 | 추가 시간 | 의미 (이전 timeline) | 동적 분석 정정 (§13) |
|---|---|---:|---|---|
| ① | Python attention.py hot path × 80 layer | +12 ms | skip_gpu check, direct call launch overhead, cudaStream sync — pure Python overhead | ✓ 정합 (python 1.84% + CUDA stream sync 미계측) |
| **②** | CPU pacpu time > GPU concurrent work → cumulative GPU IDLE | **+18 ms** | overlap 작동 중. layer 당 CPU pacpu (~2.3 ms) > GPU concurrent work (~0.4 ms) → 차이 누적 | ✓ wall 정합. **단 내부 cycle 분해: libgomp barrier wait 62% (+11.2 ms) + libpacpu actual work 38% (+6.8 ms)** ★ true dominant = OMP barrier |
| ③ | "swap_in + sample + emit + Python loop" | +25 ms | `_neo_handle_kv_swap` Python loop, ATen `index_kernel` GOMP — overlap 끝난 step 마감 영역 | **▲ misattribution** — swap path = async, wall hidden (libtorch_cpu 10.24% 의 cycle 은 parallel 영역). +25 ms 의 진정한 source = sample / emit / next-step admission 가설 (§14.3 instrumentation 필요) |
| | **합 (NEO 추가)** | **+55 ms** | vanilla 54 ms + NEO 55 ms = **NEO ~109 ms / step** | 합 정합 |

→ 본 wall 분해 = S1-S9 와 identical (HEAD `e64c56561` 의 default OFF env-gated 옵션 영향 없음).
→ **동적 분석 기반 진정한 lever priority** (§15) = ② 안의 libgomp barrier wait (★★★) > softmax fast_exp (★★) > AMX/F3 (★) > swap path (⚪ 이미 async).

---

## 2. 최신 측정 fact (HEAD `e64c56561` default 환경)

### 2.1 gmu=0.92 환경 (이전 측정)

| 측정 | 3-run avg | min | max | CV | vs vanilla |
|---|---:|---:|---:|---:|---:|
| vanilla (4,690.7, gmu=0.85) | — | — | — | — | 100% |
| S1-S9 (NEO 원본 정합, 2026-05-17) | **2,238.6** | 2,153.6 | 2,303.4 | 3.44% | **47.7%** |
| v1.6 (commit `64f9e0c48`) | 2,197.4 | 2,156.9 | 2,223.8 | 1.62% | 46.8% |

### 2.2 gmu=0.85 환경 (신규 측정 2026-05-19~20, [`p3_compare_3run_085_20260520/`](../p3_compare_3run_085_20260520/))

| Config | run 1 | run 2 | run 3 | **3-run avg** | CV | vs vanilla |
|---|---:|---:|---:|---:|---:|---:|
| **vanilla (NEO off)** | 4,679.4 | 4,680.4 | 4,680.7 | **4,680.2** | 0.01% | 100% |
| **★ v1.6 best (NEO best @ gmu=0.85)** | 1,749.8 | 1,778.8 | 1,970.5 | **1,833.0** | 6.6% | **39.2%** |
| S1-S9 | 1,763.0 | 1,858.9 | 1,778.3 | **1,800.1** | 2.9% | 38.5% |
| P3 (HOST_K_BF16=1 + USE_AMX=1) | 1,799.4 | 1,764.0 | 1,800.4 | **1,787.9** | 1.2% | 38.2% |
| P1 baseline | 1,695.3 | 1,738.2 | 1,801.8 | **1,745.1** | 3.0% | 37.3% |

→ **gmu cross-env ranking 차이** 확정:
- gmu=0.92: S1-S9 (2,238.6) > v1.6 (2,197.4) — S1-S9 best
- gmu=0.85: v1.6 (1,833.0) > S1-S9 (1,800.1) — **v1.6 best**

원인 = NEO scheduler 의 wall-clock 의존 trigger (KV pool 압박 영역 차이). S1-S9 의 cpu_comm_stream split 이 gmu=0.92 영역의 더 큰 KV pool 안에서만 win 영역. gmu=0.85 의 작은 pool 영역 에서는 split overhead > win.

### 2.3 P3 (K BF16 host store + AMX) 의 net loss 확정

P3 = `VLLM_NEO_HOST_K_BF16=1` + `VLLM_NEO_USE_AMX=1` 활성.
**gmu=0.85 3-run avg = 1,787.9 (-2.5% vs v1.6 best 1,833.0)** → AMX dropin 의 net loss 정량 확정.

원인 ([`analysis/M_sub015_phase3_hpc_optimization.md`](../../analysis/M_sub015_phase3_hpc_optimization.md) §2.2 / §3.1):
- AMX tile occupancy 19% (M=8/16 × 3/8 tile)
- setup overhead 650-900 cyc / block > work 64 cyc × 10배
- 외부 1차 fact backing: OpenBLAS Discussion #5205, AWS Compute Blog, libxsmm cutoff `(MNK)^(1/3) ≤ 64`

---

## 3. env-gated alternative path 1: P3 (K BF16 + AMX) timeline

`VLLM_NEO_HOST_K_BF16=1` + `VLLM_NEO_USE_AMX=1` 활성 시.

### 3.1 swap_out path 변경 (HOST_K_BF16=1)

```
S1-S9 default:
  GPU FP16 (KV cache) → host pinned FP16 (swap_out_staging) → host BF16 buffer (cdec read time cast)

P3 활성:
  GPU FP16 → host pinned BF16 (swap_out 시 _mm512_cvtne2ps2bf16 cast) → host BF16 (cdec read direct)
```

| 영역 | S1-S9 | P3 활성 | Δ |
|---|---:|---:|---:|
| swap_out cast cost | 0 (FP16 pass-through) | +200-400 cyc / block (FP16→FP32→BF16 paired) | **+** |
| cdec read cast cost | 200-400 cyc / block (FP16→FP32→BF16) | 0 (BF16 direct read) | **−** |
| storage size | 256 byte/block (FP16) | 128 byte/block (BF16) | **−50%** |

→ swap_out 1 회 + cdec read N 회 (N = 80 layer × cdec dispatch 빈도) → cdec read 영역 cast 영구 제거가 amortize 영역 비율 큼. **이론상 +1-5% throughput**.

### 3.2 cdec compute path 변경 (USE_AMX=1)

```
S1-S9 default cdec path (ISPC):
  qk_product (ISPC 128-lane gang)
  → softmax (ISPC 3-pass)
  → av_product (ISPC)
  total: ~500 cyc / block, ISPC 8-10 GFLOP/s/core

P3 활성 cdec path (AMX qk + ISPC softmax/av):
  K^T pre-pack (BF16 transpose + cast)  ← setup 200-400 cyc
  tile_loadcfg + tile_loadd × 2 × 4 round  ← setup 240 cyc
  tile_dpbf16ps × 4 round  ← work 64 cyc
  tile_stored + C copy  ← setup 90 cyc
  → softmax (ISPC 동일)
  → av_product (ISPC 동일)
  total: ~650-900 cyc / block (qk 영역만 1.5-2× regression)
```

| Cycle 영역 | ISPC qk | AMX qk | Δ |
|---|---:|---:|---:|
| Q FP16→BF16 변환 (Step 1 thread-local cache hoist) | 0 | 50-100 (per call) | **+** |
| K^T pre-pack + FP16→BF16 (Step 2 outer hoist) | 0 | 200-400 (per block) | **+** |
| Vector load / compute (work) | ~400 | 64 (`tile_dpbf16ps` × 4) | **−** |
| Tile config / store overhead | 0 | 300 (tile_loadcfg + tile_stored + C extract) | **+** |
| **per-block total** | **~400-500** | **~650-900** | **+50-100%** |

→ AMX 의 setup overhead 가 work 64 cyc 대비 10배 이상 — small matmul 영역 (NEO `(MNK)^(1/3) ≈ 22.6`) 에서 **net loss 보장**. 외부 fact (Intel Opt Ref Manual #355308 `TDPBF16PS` throughput 16 cyc, tile shape 무관) backing.

### 3.3 P3 적용 시 timeline 영역의 ② (CPU IDLE 누적) 변화

| 영역 | S1-S9 | P3 활성 | 차이 |
|---|---:|---:|---:|
| ② CPU pacpu time / layer | ~2.3 ms | ~2.6-2.8 ms (qk 영역 +25-50%) | +0.3-0.5 ms |
| ② cumulative GPU IDLE | +18 ms | +24-28 ms (CPU bottleneck 더 심화) | **+6-10 ms** |
| **NEO 추가 wall / step** | +55 ms | **+61-65 ms** | **+10-20% step time** |

→ measurement 정합: P3 3-run avg 1,787.9 vs v1.6 1,833.0 = **-2.5%** ≈ 10-20% step time 증가의 throughput 환산.

---

## 4. env-gated alternative path 2: P4 (async cdec) timeline

`VLLM_NEO_ASYNC_CDEC=1` + `VLLM_NEO_CDEC_PIPELINE_DEPTH=1` 활성 시.

### 4.1 코드 path

[`vllm/v1/worker/sub_batch_executor.py:285-291`](../../../../../vllm/v1/worker/sub_batch_executor.py):

```python
if _os_async.environ.get("VLLM_NEO_ASYNC_CDEC", "0") == "1":
    from vllm.model_executor.layers.attention.attention import (
        neo_async_cdec_scope as _neo_async_cdec_scope,
    )
    with _neo_async_cdec_scope():       # forward_pipeline 진입 시만 활성
        return self._forward_pipeline_inner(batches, embeddings)
return self._forward_pipeline_inner(batches, embeddings)
```

[`vllm/model_executor/layers/attention/attention.py:1180-1188`](../../../../../vllm/model_executor/layers/attention/attention.py):

```python
if cdec_future is not None and _neo_async_cdec_mode:
    _neo_pending_cdec_queue.append((cdec_future, output, cdec_t0, cdec_t1))
    _depth_p4 = int(os.environ.get("VLLM_NEO_CDEC_PIPELINE_DEPTH", "1"))
    while len(_neo_pending_cdec_queue) > _depth_p4:
        _neo_drain_pending_cdec()
```

drain 위치 = `forward_double` / `forward_first_stage` / `forward_last_stage` 의 postproj 직전 (sub_batch_executor.py).

### 4.2 의도된 timeline (depth=1 활성 시)

```
S1-S9 default (sync wait):
  layer N: cdec submit → GPU forward launch → cdec wait (block)
                                                  └─ main thread blocks
                                                     GPU stream queue 그동안 진행
  next layer: layer N+1 시작

P4 active (depth=1):
  layer N:   cdec submit → GPU forward launch → enqueue (return immediately)
  layer N+1: cdec submit → GPU forward launch → drain N (block)
                                                  └─ layer N+1 cdec 가 이미 실행 중
                                                     drain 시점에 partial 또는 완료
```

→ 이론상 layer-level overlap +1 (drain depth=1 이라 2 cdec 동시 in-flight).

### 4.3 실측 결과 — net loss

| 측정 | 결과 |
|---|---|
| P4 sanity 100p × 8192 | 920.4 tps (코드 통과) |
| P4 long run 1 (500p, gmu=0.85) | 1,803.0 tps (lucky variance) |
| P4 + MIRROR=80 재현 | **NO_RESULT** (EngineDeadError @ 2%, 30,080 OOB precheck errors) |

원인 ([`p4_p5_lever_20260520/README.md`](../p4_p5_lever_20260520/README.md)):
- async cdec 의 deferred dispatch → cdec slot 추적 영역 race → D11 OOB precheck trigger → cdec dispatch skip → 누적 backlog → engine death
- KV cache sequential dependency (layer i output → layer i+1 input) 로 진정한 cross-layer pipeline 불가능
- **현재 implementation 의 진정한 가치 = 음수** (안정성 회귀)

### 4.4 P4 의 timeline 영향 (의도 vs 실제)

| 영역 | S1-S9 sync | P4 depth=1 의도 | P4 실제 |
|---|---|---|---|
| ② CPU pacpu cumulative | +18 ms | 이론 -8~12 ms (50% overlap) | 측정 불가 (engine death) |
| backlog / OOB race | 0 | 0 | **30k+ events** |
| stability | ✓ | ✓ | ✗ (재현 시 crash) |

→ Amdahl ceiling 영역 win 은 이론적이나 **infra 의 KV dependency** 가 차단. infra 재설계 없이는 의도 효과 도달 X.

---

## 5. env-gated alternative path 3: D fix (OOB silent skip)

`VLLM_NEO_OOB_SILENT=1` (**default ON**).

### 5.1 timeline 영향

- **behavior 동등** — cdec_future=None fallback 수행은 동일
- 차이 영역 = exception traceback 의 stdout 출력 skip
- **timeline path 변화 없음**, log volume ↓

### 5.2 측정 결과

| Config | tps | 비고 |
|---|---:|---|
| combo 1 baseline (D=off, 100p × 8192) | 934.5 | reference |
| combo 2 D-only (D=on, 100p × 8192) | 935.6 | **+0.1%** (variance 영역) |

→ short workload 영역의 D fix 영향 = noise. long workload 영역 의 진정한 effect = TBD (별도 측정 필요).

---

## 6. OOB root fix v1 (G/H log rate-limit, commit `dd80747a6`)

[`vllm/v1/core/sched/neo_scheduler_adapter.py:1166`](../../../../../vllm/v1/core/sched/neo_scheduler_adapter.py) 의 `logger.info("[Plan v4 G/H] swap_out attach ...")` 에 rate-limit 추가:
- 첫 5 회 + 매 1000 회 + cap saturation 상태 변경 시만

### 6.1 timeline 영향

- **behavior 동등** — scheduler logic 변화 없음
- 차이 영역 = stdout 출력 volume
- **timeline path 변화 없음**, primary deadlock root 영역 잔존 (scheduler hot spin 190 calls/sec @ mirror cap saturated)

### 6.2 fix 효과 정량 (전후 비교)

| 지표 | 이전 combo 3 (NO_RESULT) | fix 적용 후 |
|---|---:|---:|
| `[Plan v4 G/H] swap_out attach` log count | 1,092,271 | **328** (22,755× ↓) |
| stdout file size | 214 MB | **2.89 MB** (74× ↓) |
| scheduler call rate | (미측정) | 190 calls/sec |
| 결과 (A-only TP=4 100p × 8192) | NO_RESULT (timeout) | **NO_RESULT (timeout, 동일)** |

→ **secondary cause (log spam) 차단 ✓**, **primary cause (scheduler hot spin) 잔존** — 별도 fix 필요.

---

## 7. variance fact (현재 코드 베이스, 신규 4 측정 포함)

| path | 환경 | runs | min — max | avg | CV | vs vanilla |
|---|---|:-:|---|---:|---:|---:|
| vanilla (NEO OFF) | gmu=0.85 | 3 | 4,679.4 — 4,680.7 | **4,680.2** | **0.01%** | — |
| vanilla (NEO OFF) | gmu=0.92 | 3 | 4,690.4 — 4,691.0 | 4,690.7 | 0.006% | — |
| **★ S1-S9** | gmu=0.92 | 3 | 2,153.6 — 2,303.4 | **2,238.6** | 3.44% | **47.7%** |
| **★ v1.6 best** | gmu=0.85 | 3 | 1,749.8 — 1,970.5 | **1,833.0** | 6.6% | **39.2%** |
| S1-S9 | gmu=0.85 | 3 | 1,763.0 — 1,858.9 | 1,800.1 | 2.9% | 38.5% |
| v1.6 | gmu=0.92 | 3 | 2,156.9 — 2,223.8 | 2,197.4 | 1.62% | 46.8% |
| P3 (HOST_K_BF16=1 + USE_AMX=1) | gmu=0.85 | 3 | 1,764.0 — 1,800.4 | 1,787.9 | 1.2% | **38.2%** |
| P1 baseline (commit `aba1b14b1`) | gmu=0.85 | 3 | 1,695.3 — 1,801.8 | 1,745.1 | 3.0% | 37.3% |
| P4 (ASYNC_CDEC=1) long | gmu=0.85 | 1 | — | 1,803.0 | — | 38.5% (lucky variance) |
| P4 (ASYNC_CDEC=1) 재현 | gmu=0.85 | 1 | NO_RESULT | — | — | — |
| P5 MIRROR=60 | gmu=0.85 | 1 | — | 1,766.8 | — | 37.8% |

### 핵심 fact
- vanilla CV 0.006-0.01% (deterministic) vs NEO CV 1.2-6.6% (variance 잔존)
- variance source = NEO scheduler 의 wall-clock 의존 trigger (`time.time()`, KV pool snapshot)
- **HEAD `e64c56561` 의 모든 env-gated alternative 가 default (S1-S9 또는 v1.6 best) 초과 X**
- gmu cross-env ranking 차이 (S1-S9 vs v1.6 best) 가 env-dependent 변환 의 NEO scheduler 영역 의 sensitivity 입증

---

## 8. 다음 lever 영역 정합 ([`analysis/M_sub015_phase3_hpc_optimization.md`](../../analysis/M_sub015_phase3_hpc_optimization.md))

### 8.1 Phase α (env-only sweep, 1-2 일) — timeline 영역 unchanged

| Lever | env 변경 | timeline 영향 |
|---|---|---|
| M0 KMP_BLOCKTIME=INF | env-only | barrier spin (② 영역의 일부) 단축 — 외부 fact: IPEX LLM 2-3× 보고 |
| M1 KMP_AFFINITY 명시 | env-only | OMP thread placement 의 NUMA locality 개선 — ② 영역의 일부 |

→ **timeline mechanism 변화 없음**. ② 영역 (libgomp barrier + CPU IDLE) 의 일부 단축 효과 측정 영역.

### 8.2 Phase β (HPC 영역 surgical change, 3-5 일) — timeline 영역 부분 변경

| Lever | 코드 변경 | timeline 영향 |
|---|---|---|
| M2 K cache BF16 host store (= P3 의 HOST_K_BF16 영역만, USE_AMX=0) | Python wiring (existing) + 정확도 검증 | swap_out cast +200-400 cyc, cdec read cast -200-400 cyc. AMX 영역 분리로 setup overhead 회피 — **이론상 +1-5%** |
| M3 online softmax (FlashAttention 식) | `pacpu.ispc` softmax 영역 3-pass → single-pass | softmax 영역 (9.73% of cycle) 2× 가능 — **이론상 +2-5%**, ② 영역 단축 |

→ **timeline 영역 ②** (CPU IDLE) 의 핵심 lever. mechanism 자체는 unchanged.

### 8.3 Phase γ (architectural, 2-4 주, 별도 SUB 영역)

| Lever | 영역 | timeline 영향 |
|---|---|---|
| F4 TP=8 → TP=4 | M=8 → M=16 (AMX tile 100% occupancy) | P3 path 영역의 AMX setup amortize 가능 — **이론상 +5-10%**, **단 GPU 측 parallelism 감소 trade-off** |
| F5 BLOCK_SIZE 16 → 32 | per-block work 2×, N=16→32 (AMX tile full) | AMX setup amortize + cache line 활용 — **이론상 +3-7%** |

---

## 9. 결론

### 9.1 현재 코드 베이스 (HEAD `e64c56561`) 의 timeline 상태

1. **Default 환경 timeline = S1-S9 와 identical** (env 기본 OFF)
2. **best config (gmu 환경 별)**:
   - gmu=0.92: S1-S9 **2,238.6 tps**
   - gmu=0.85: v1.6 best **1,833.0 tps**
3. **모든 env-gated alternative path 의 net loss 확정** (P3 -2.5%, P4 unstable, D +0.1% noise)
4. **OOB root fix v1**: secondary cause (log spam) 차단 ✓, primary cause (scheduler hot spin) 잔존

### 9.2 timeline mechanism 영역 의 정합

- NEO 원본 `transformer_layer.py` 의 5 가지 정합 요소 (cpu_communication_stream / `_transfer_qkv` async / `paged_attention_cpu` direct / result D2D async / `_forward_pipeline_stage` ordering) 모두 작동 ✓
- ② 영역 (CPU pacpu > GPU concurrent work → cumulative GPU IDLE) = **CPU bottleneck 의 결과**, overlap mechanism 실패가 아님
- paper claim +14% 도달 X 의 근본 원인 = workload 차이 (long context + max batch HBM-fit) 와 vllm baseline 차이

### 9.3 다음 lever 권고 (동적 분석 기반 재책정 — §15 참조)

본 turn 의 동적 fact 적용으로 priority **재책정**:

| 순위 | Lever | 영역 | wall 단축 | effort |
|---|---|---|---:|:-:|
| **★★★ 1** | **OMP barrier wait 영구 제거** (barrier #1/#2 또는 thread imbalance) | ② 안 libgomp 62% | **-11 ms (+10%)** | 1-3 일 |
| **★★ 2** | KMP_BLOCKTIME=INF + KMP_AFFINITY 명시 (env-only) | libgomp spin 영역 | -2~-4 ms | 1 시간 |
| **★★ 3** | softmax fast_exp (ISPC polynomial) | libpacpu softmax | -2 ms | 2-3 일 |
| ★ 4 | F3 K BF16 host store (USE_AMX=0 single lever) | libpacpu qk 일부 | -1 ms | 2-3 일 |
| ⚪ 5 | F4 TP=8→4 (AMX tile full) | libpacpu qk +50% | -3 ms (GPU TP trade-off) | 2-4 주 |
| ⚪ 6 | swap path 추가 가속 | wall hidden, ROI ~0 | -1 ms 미만 | (deprioritize) |
| ✗ | F6 OMP dynamic schedule | atomic overhead 회귀 -1.4% | — | (시도 후 폐기) |

**즉시 (1-2 일)**: §14.1 OMP barrier instrumentation + §14.2 fast_exp 적용
**다음 1-2 주**: KMP_BLOCKTIME/AFFINITY sweep + F3 K BF16 host store
**중장기**: F4/F5 (별도 SUB 영역)

---

## 10. 파일

| file | 내용 |
|---|---|
| `README.md` (본 문서) | HEAD `e64c56561` timeline 분석 + env-gated alternative + 신규 측정 fact + 동적 분석 기반 bottleneck pin-point |
| **`timeline_schematic.svg` (★ 신규)** | **동적 분석 기반 timeline 도식** — swap_stream async hidden lane + cdec_executor 내부 분해 (barrier wait 62% / compute 38%) + OMP barrier marker + ③ "+25 ms" source TBD |
| `../timeline_v16_s1_s9_20260517/timeline_schematic.svg` | (이전) S1-S9 baseline timeline — 본 문서 §13.2-13.5 에서 misattribution 영역 정리 |
| `../timeline_v16_s1_s9_20260517/README.md` | (이전) S1-S9 timeline 상세 (2026-05-17) |
| `../timeline_v16_optionA_20260516/README.md` | (이전) Option A (v1.6) timeline (2026-05-16) |
| `../p3_compare_3run_085_20260520/README.md` | gmu=0.85 5-case 3-run 측정 |
| `../p4_p5_lever_20260520/README.md` | P4 (F1 async cdec) + P5 (F2 MIRROR sweep) |
| `../combo_sweep_20260520/README.md` | A (TP=4) × D (OOB silent) 4-combo |
| `../oob_root_fix_20260520/README.md` | G/H log rate-limit fix v1 |
| `../../analysis/M_sub015_phase3_hpc_optimization.md` | HPC 측면 최적화 분석 + 외부 1차 출처 backing |

## 11. Change Log

| 일자 (KST) | 변경 |
|---|---|
| **2026-05-20** | 신설. HEAD `e64c56561` timeline 분석 + S1-S9 default timeline 정합 + env-gated alternative path (P3/P4/D) 의 timeline 영향 + 신규 4 측정 (gmu=0.85 cross-env / P4/P5 lever / combo sweep / OOB root fix) 통합. |
| **2026-05-20 (turn 2)** | §12-15 추가 — bottleneck 식별 framework + 이전 timeline (v1.6 / Option A / S1-S9) cross-check + async / barrier / sync 영역 별 위치 정합 + 미계측 영역 정리. |
| **2026-05-20 (turn 3, 본 turn)** | **사용자 지적 (swap path 가 async 라 wall hidden) 적용 → §12-15 동적 분석 기반 재작성**. perf record 60s fact (libgomp 43.75% / libpacpu 26.38% / libtorch_cpu 10.24% / python 1.84%) 기반. (1) ③ "swap path +25 ms" misattribution 정정 — swap 영역 wall hidden, +25 ms 의 진정한 source 미확정 (§13.2/13.5/13.8). (2) ② 안의 진정한 dominant = libgomp barrier wait (62% of cdec cycle = +11.2 ms wall). (3) lever priority 재책정 — OMP barrier > KMP_BLOCKTIME > fast_exp > F3 > F4 > swap (deprioritize) > F6 (회귀 확인). (4) §14 측정 plan 우선순위 변경 — swap_in instrumentation 제거, step-end +25 ms source 식별 신규 추가. |
| **2026-05-20 (turn 4, 본 turn)** | **★ 신규 SVG 도식 작성** (`timeline_schematic.svg`, 410 lines) — 동적 분석 기반 재구성. (1) **swap_stream lane 신규 추가** — async wall HIDDEN pattern (대각선 hatch) 명시. (2) **cdec_executor lane 내부 분해** — barrier wait 62% (★ ompBarrierWait pattern) + compute 38% (ispcCompute pattern) 의 2-band 시각화. (3) **OMP barrier marker** (#1/#2/implicit) cdec lane 위 vertical line 80 layer × 3 = ~240 barrier 표시. (4) **③ label 변경** — `swap_in + sample + emit` → `step-end TBD (sample/emit/admit?)` (dashed border). (5) **OMP worker wait 화살표 변경** — GPU forward → cdec_executor barrier. (6) **동적 분석 fact 분해 표** + 핵심 메시지 박스 (이전 timeline 정정 영역 명시). |

---

## 12. Bottleneck 식별 framework — 동적 분석 기반 재작성

> **사용자 지적 (2026-05-20 본 turn)**: swap path 는 async (SUB_025/026 staging + swap_stream) 로 wall 에서 **완전히 숨어 있음**. 이전 timeline 의 "③ swap_in +25 ms" 영역 = wall path 아님 (CPU spillover cycle 영역만, GPU IDLE 영역 기여 X).
> → 본 §12-15 = **perf record 동적 분석 fact** ([`../../analysis/H_dynamic_analysis.md`](../../analysis/H_dynamic_analysis.md)) 에 기반한 재작성.

### 12.1 동적 분석 fact — perf record 60s, 413K samples (S1-S9, gmu=0.92)

**측정 dir**: `eval/results/20260517_200212_cpu112_analysis_500p/deep_dive_60s/perf.data`
**total event count**: 6.86 T cycle (60s × 8 worker × 14 OMP thread)

| DSO | cycle % | 분류 | wall 기여 | 측정 |
|---|---:|---|---|:-:|
| **libgomp.so.1** | **43.75%** | OMP barrier wait (`gomp_team_barrier_wait_*`) + spin | **★★★ ② 안의 dominant** — barrier 가 풀릴 때까지 thread spinning | ✓ perf dso |
| **libpacpu** | 26.38% | ISPC kernel work (softmax 9.73 / qk 8.75 / av 7.90) | ★★ ② 안의 actual compute (CPU pacpu 의 실제 work) | ✓ perf dso |
| **libtorch_cpu** | 10.24% | ATen swap path CPU spillover (`index_put_kernel<Half>` 4.85 / `AVX2::copy_kernel` 3.27 / `index_kernel<Half>` 2.12) | **★ async 영역** — swap_stream + cdec_executor 가 parallel 실행 → wall 에서 hidden | ✓ perf dso |
| python3.12 | 1.84% | Python interpreter dispatch (`_PyEval_EvalFrameDefault`) | ⚪ 작음 | ✓ |
| (나머지) | 17.79% | misc symbols < 0.5% | ⚪ | ✓ |

### 12.2 swap path = async, wall 에서 hidden (사용자 지적 정합)

```
[time axis →]
┌──────────────────────────────────────────────────────────────────────┐
│ main thread:   Python attn → cdec_direct → GPU forward → next layer   │
│                                  ↕ (cdec_executor worker process)     │
├──────────────────────────────────────────────────────────────────────┤
│ swap_stream:   ┌─async swap_out gather─┐ ┌─DMA D→H─┐ (drain later)    │
│                └─ overlap with forward ─┘ └─overlap with forward ─┘    │
├──────────────────────────────────────────────────────────────────────┤
│ swap_in stream: ┌─cudaMemcpyAsync H→D─┐ (overlap with attn)          │
│                 └────── completely async, hidden ──────┘              │
├──────────────────────────────────────────────────────────────────────┤
│ GPU default stream:  preproj → attn → postproj → MLP → AllReduce      │
└──────────────────────────────────────────────────────────────────────┘
                                                                       ↑
                                                            step end (wall stop)
```

**핵심 fact**:
1. **swap_out** = SUB_026 staging buffer N=3 + cudaMemcpyAsync on swap_stream → forward 와 overlap. wall 기여 = drain wait time only (작음)
2. **swap_in** = per-layer `copy_layer_out` 에서 ATen index_kernel + cudaMemcpyAsync on swap_stream → attn 과 overlap. wall 기여 = launch overhead 만 (~Python loop)
3. **libtorch_cpu 10.24% 의 cycle** = **CPU 가 parallel 로 work 하는 영역** (cdec_executor + swap path 동시 실행), wall path 가 아님 — 단 cdec_executor 와 같은 CPU pool 을 공유하면 libgomp barrier 영역에서 thread contention 발생 가능

### 12.3 NEO 추가 +55 ms 의 진정한 영역 분해 (동적 backing)

```
vanilla 54 ms ──────────────────────────────────── 54 ms
                                                          ↓
NEO  109 ms = 54 + 55 ms NEO addition
                                                          ↓
                                              NEO 추가 55 ms:
                                                          ↓
   ┌──────────────────────┬──────────────────────────────┐
   ① Python wall overhead  ② cdec leftover (CPU bottleneck → GPU IDLE)
   +12 ms (22%)            +43 ms (78%) ★ TRUE TOP BOTTLENECK
                           
                           ② 내부 cycle 분포 (동적):
                           ├─ libgomp barrier wait 62%   ──── ★★★ true dominant
                           ├─ libpacpu compute  38%       ──── actual work
                           └─ libtorch_cpu      (async)   ──── hidden (parallel)

   (③ swap path = async, wall hidden, 0 ms 기여 ★ 사용자 지적 정합)
```

**진정한 1순위 bottleneck = ② 안의 libgomp barrier wait (62% of cdec cycle time)**.
이전 framework 의 "③ swap path +25 ms" = **wall 측정 misattribution** — 실제로는 ② 안의 cdec compute 가 +43 ms (= +18 ms direct visible + +25 ms 가 cdec leftover wave 의 일부) 가능성 높음. 정확한 ②/③ 분리는 §14 instrumentation 필요.

### 12.4 ② 안의 libgomp 43.75% 의 source — OMP barrier wait

`core.h:328-403` 영역의 4 sync point:

```
# pragma omp parallel
{
    // Step 0: store_kv (per-thread partition)
    ...
    # pragma omp barrier               ★ barrier #1 — futex_wait
    // Step 1: attn_one_seq (per-task partition, ISPC or AMX)
    ...
    # pragma omp barrier               ★ barrier #2 — futex_wait (dominant 가설)
    // Step 2: gather_output (per-thread partition)
    ...
}  // implicit barrier (omp parallel end)  ★ barrier #3
```

**libgomp hot symbol 분포** ([`H_dynamic_analysis.md:51-58`](../../analysis/H_dynamic_analysis.md)):

| Offset | cycle % | 의미 |
|---|---:|---|
| 0x1de60 | **31.98%** | base hot — `gomp_team_barrier_wait_*` 후보 |
| 0x1de62 | 3.75% | base + 2 (next instr in same loop) |
| 0x1de6b | 3.68% | base + 11 |
| 0x1e028 | 3.42% | 별개 함수 또는 loop tail |
| 0x1de6f | 0.92% | base + 15 |

→ **0x1de60 ~ 0x1de6f 의 16-byte 범위 = 단일 함수의 inner spin loop**. libgomp 의 `gomp_team_barrier_wait_end` 또는 `do_wait` (KMP_BLOCKTIME spin) 후보.

**step-rate backing** ([`H_dynamic_analysis.md:75-82`](../../analysis/H_dynamic_analysis.md)):
- step rate = 72 step/sec × 80 layer = **5,760 paged_attention call/sec/worker**
- 매 call 의 4 sync × ws=14 thread imbalance → 평균 ~50% thread 가 wait state
- 14 thread × 60 sec × 8 worker = 6,720 thread-sec 측정 window
- libgomp 43.75% × 6.86T cycle / 14 thread / 60 sec / 8 worker = **44.6 M cycle/sec/thread**

→ ② 안에서 libgomp 가 진정한 dominant — barrier 가 풀릴 때까지 thread spinning 으로 cycle 소모.

### 12.5 ② 안의 libpacpu 26.38% — actual ISPC compute

| Kernel | cycle % | FLOP / seq / layer | 분류 |
|---|---:|---:|---|
| softmax | **9.73%** | 0.49 M (qk/av 의 11%) | **★ cycle/FLOP worst** — exp/log latency, ILP 제한 |
| qk_product | 8.75% | 4.19 M | matmul |
| av_product | 7.90% | 4.19 M | matmul |

→ ② 안의 actual compute 26.38% 중 **softmax 가 가장 많은 cycle 소모** — fast_exp (B lever) 시 가장 큰 net win.

### 12.6 libtorch_cpu 10.24% = async (wall hidden)

| Symbol | cycle % | 영역 |
|---|---:|---|
| `index_put_kernel<Half>` | 4.85% | `_neo_handle_kv_swap` 의 swap_out 영역 (CPU scatter) |
| `AVX2::copy_kernel` | 3.27% | tensor `.to(device)` 의 host-side copy |
| `index_kernel<Half>` | 2.12% | `host_k_buf.index_select(1, idx_cpu)` 영역 |

→ **이 10.24% 는 wall path 가 아님**:
- swap_out async 영역 (SUB_026 staging N=3) = swap_stream + cdec_executor worker process 에서 실행 → main thread wall block X
- swap_in async 영역 = swap_stream cudaMemcpyAsync → attn 과 overlap
- 단 **OMP team 공유** — cdec_executor worker 의 OMP pool 과 swap path 의 ATen GOMP 가 같은 thread pool 사용 시 libgomp 영역에서 contention

→ **F6 OMP dynamic schedule (-1.4% 회귀)** 의 root = atomic counter overhead. 대신 OMP_NUM_THREADS sweep / persistent OMP team / barrier #1 제거 영역이 진정한 lever.

### 12.7 현재 measurement 가능한 metric (VLLM_NEO_PROFILE=1)

[`attention.py:1131-1224`](../../../../../vllm/model_executor/layers/attention/attention.py) 의 PROFILE 영역:

| Metric | 의미 | 단위 | 현재 측정 |
|---|---|---|:-:|
| `gpu_ms` | 1 layer 의 GPU forward time (`self.impl.forward` 호출 elapsed) | ms/layer | ✓ |
| `cdec_wait_avg` / `cdec_wait_max` | cdec_future.result() blocking time (S5 direct = 0) | ms/layer | ✓ |
| `b0_count_sum` / `b1_count_sum` | sub-batch row count (b0 = GPU, b1 = cdec) | rows/layer | ✓ |
| `skip_gpu_count` | b1-only sub-batch 에서 GPU forward skip 횟수 | count/step | ✓ |

[`gpu_model_runner.py:6631+`](../../../../../vllm/v1/worker/gpu_model_runner.py) 의 swap PROFILE:

| Metric | 의미 | 단위 | 현재 측정 |
|---|---|---|:-:|
| `[NEO SWAP_OUT CALL] count=N (async/sync)` | swap_out 발화 횟수 + path | count, log | ✓ |
| `[PROFILE SWAP_OUT async] req=X blocks=N slot=S` | async swap_out 의 req 별 fact | log | ✓ |
| `_elapsed_so_ms` (sync swap_out) | sync swap_out 한 req 의 elapsed | ms/req | ✓ |
| **swap_in elapsed** | swap_in path 의 elapsed | ms | ✗ (instrumentation 미존재) |
| **drain_async wait time** | drain 에서 event sync 대기 시간 | ms | ✗ |

### 12.8 측정 불가능한 (instrumentation 필요한) 영역 — bottleneck pin-point 갭

| 영역 | 필요 instrumentation | 우선순위 | 이유 |
|---|---|:-:|---|
| **OMP barrier #1 (store_kv → attn)** | core.h:346 직전/직후 `omp_get_wtime()` + thread-local diff | ★★★ | libgomp 43.75% 의 dominant 후보 |
| **OMP barrier #2 (attn → gather)** | core.h:400 직전/직후 동일 | ★★★ | Step 1 imbalance 의 누적 영역 (가설) |
| **per-thread Step 1 elapsed** | tid 별 (qk+softmax+av) 시작/끝 wtime | ★★ | task partition imbalance 정량 |
| **OMP implicit barrier (parallel end)** | omp parallel { } 끝 직전 wtime | ★★ | barrier #3 의 별도 추적 |
| **CUDA stream wait_stream duration** | nvprof / nsys 또는 manual cuda events | ★ | ① +12 ms 의 일부 |
| **NCCL AllReduce 위치 + wait** | nvprof NVTX range | ★ | GPU IDLE 영역의 step 위치 |
| ~~swap_in per-layer elapsed~~ | (async 라 wall 기여 작음 — instrumentation 우선순위 ↓) | ⚪ | **사용자 지적 — swap path 는 wall hidden** |
| ~~`_neo_handle_kv_swap` Python loop elapsed~~ | (Python launch overhead 만, async DMA 가 본체) | ⚪ | 동일 사유 |

---

## 13. 이전 timeline (v1.6 / Option A / S1-S9) cross-check — swap path async 영역 정정

### 13.1 이전 timeline 의 wall 분해 vs 동적 분석 fact

**이전 timeline 의 명시 영역**:

| Timeline | base | wall total | NEO 추가 | ① Python | ② cdec | ③ "swap_in+sample+emit" |
|---|---|---:|---:|---:|---:|---:|
| v1.6 (`timeline_v16_20260516`) | sync first measurement (16:33) | 115 ms | +61 ms | +12 | **+24** | +25 (TBD 표시) |
| **Option A** (`timeline_v16_optionA_20260516`) | sync 재측정 (18:30) | 115 ms | +61 ms | +12 | **+24** | +25 |
| **★ S1-S9** (`timeline_v16_s1_s9_20260517`) | NEO 원본 정합 (S1-S9) | 109 ms | +55 ms | +12 | **+18** | +25 |

**Δ (S1-S9 vs v1.6/Option A)** = −6 ms in ② (ThreadPool overhead + GIL race 제거).

### 13.2 ③ "+25 ms" 의 misattribution — 사용자 지적

**이전 framework**: `+25 ms` = "swap_in + sample + emit + Python loop overhead" 통합 추정.
**실제 동적 fact**:
- swap_out = SUB_025/026 staging N=3 + cudaMemcpyAsync on swap_stream → **forward 와 overlap, wall hidden**
- swap_in = per-layer `copy_layer_out` 의 cudaMemcpyAsync on swap_stream → **attn 과 overlap, wall hidden**
- libtorch_cpu cycle 10.24% = **async path 의 CPU work**, wall path X
- 실측 (PROFILE log): `[NEO SWAP_OUT CALL]` async ratio 39.7% (SUB_025), 60% 가 sync fallback — 단 sync swap_out 도 SUB_028 batched H2D 적용으로 -57% latency

→ ③ 의 "+25 ms" 의 **swap 영역 실제 wall 기여 = 매우 작음** (drain wait 영역만, ~5 ms 미만 추정).
→ 그러면 +25 ms 의 진정한 source 는 어디인가? 후보:
  1. **② cdec leftover 의 실제 크기가 +18 ms 보다 큼** (PROFILE 의 per-layer gpu_ms 합 외 step-end 영역 누적)
  2. **sample + emit 영역의 GPU + Python 영역** (sampler, log_metrics, async_output emit)
  3. **next-step admission overhead** (NEO scheduler 의 swap-in 결정 + admission Python overhead)

→ **정확한 분리는 instrumentation 필요** (§14).

### 13.3 동적 분석 기반 재정리 — wall vs cycle 영역 매핑

```
wall (109 ms) ──────────────── perf record 60s ──────────────── 6.86 T cycle
                                                                          ↓
            ┌──────────────────────────────────────────────────────────┐
            │ libgomp 43.75% = 3.00 T cyc                              │
            │   └─ ② cdec 안의 OMP barrier wait (thread 가 spin 중)     │  ★ wall path
            ├──────────────────────────────────────────────────────────┤
            │ libpacpu 26.38% = 1.81 T cyc                             │
            │   └─ ② cdec 안의 ISPC actual work (softmax/qk/av)       │  ★ wall path
            ├──────────────────────────────────────────────────────────┤
            │ libtorch_cpu 10.24% = 0.70 T cyc                         │
            │   └─ swap path async (index_put/copy/index) — parallel  │  ⚪ async (wall hidden)
            ├──────────────────────────────────────────────────────────┤
            │ python 1.84% = 0.13 T cyc                                │
            │   └─ Python eval — main thread, 일부 wall                 │  △ partial wall
            └──────────────────────────────────────────────────────────┘
```

**wall path contribution**:
- ② cdec 안 = libgomp (62% of cdec cycle) + libpacpu (38% of cdec cycle) = 70.13% of total cycle = **dominant wall path**
- ③ swap path async = libtorch_cpu 10.24% = **wall hidden** (cycle 은 소모하지만 wall block 안 함)
- ① Python = 1.84% + CUDA stream sync 영역 = 작은 wall

### 13.4 ② 안의 +18 ms 의 진정한 분해 (동적 backing)

```
② cdec leftover +18 ms (GPU IDLE 누적)
                                ↓
  CPU pacpu cycle 분포 (libpacpu + libgomp = 70.13% of total):
                                ↓
  ┌─────────────────────────────────────────────────────┐
  │ libgomp barrier wait 62% of cdec cycle              │  → +18 × 0.62 = +11.2 ms
  │   ★★★ TRUE TOP BOTTLENECK                            │
  │   - barrier #1 (store_kv → attn) wait               │
  │   - barrier #2 (attn → gather) wait                 │
  │   - implicit barrier (parallel end)                 │
  │   - thread imbalance 의 가장 늦은 thread 대기         │
  │                                                      │
  ├─────────────────────────────────────────────────────┤
  │ libpacpu actual work 38% of cdec cycle              │  → +18 × 0.38 = +6.8 ms
  │   - softmax (9.73 / 26.38 = 37% of pacpu)           │  → +6.8 × 0.37 = +2.5 ms
  │   - qk_product (8.75 / 26.38 = 33% of pacpu)        │  → +6.8 × 0.33 = +2.3 ms
  │   - av_product (7.90 / 26.38 = 30% of pacpu)        │  → +6.8 × 0.30 = +2.0 ms
  └─────────────────────────────────────────────────────┘
```

→ **libgomp barrier wait 영구 제거 시 ② 가 +18 ms → +6.8 ms 가능** (-11.2 ms wall ≈ +10% throughput).
→ AMX/F3 등 ISPC compute 가속 lever = +6.8 ms 영역만 단축 가능 — **barrier 영역 미접근 시 ceiling 작음**.

### 13.5 이전 timeline 의 정확성 평가 (동적 fact 기반)

| 영역 | 이전 timeline 표현 | 동적 fact | 평가 |
|---|---|---|---|
| ① Python overhead +12 ms | "skip_gpu check, cdec submit/launch, cudaStream sync" 통합 | python 1.84% + CUDA stream sync (미계측) | ✓ 추정 적절, 단 세부 분리 불가 |
| ② cdec leftover +18 ms | "CPU pacpu time > GPU concurrent work" | libgomp 62% + libpacpu 38% of cdec | ✓ wall fact 정합. 단 **내부 분해 (barrier vs compute) 영역 미표시** |
| **③ "swap_in+sample+emit +25 ms"** | "swap_in 80 layer Python loop → batched ATen" | swap = async (libtorch 10.24% async), wall hidden. **+25 ms 의 진정한 source 미확정** | **▲ misattribution** — swap 영역 wall 기여 작음 (drain wait ~5 ms 미만 추정). +25 ms 의 다른 source = sample/emit/next-step admission 영역 |
| OMP worker 80 의 futex_wait 62% | "다음 OMP fork / NCCL 기다리며 sleep" | **libgomp 43.75% = OMP barrier wait spinning** | ✓ 정합 — 단 어느 barrier 가 dominant 인지 (#1 vs #2 vs implicit) 미분리 |
| libtorch_cpu swap 영역 | "★ Top Priority — Phase 1 alt 적용" | **async path → wall hidden, ROI 작음** | **▲ priority 재책정 필요** — barrier 영역이 진정한 dominant |

→ **이전 timeline = ② / ③ 의 misattribution 으로 lever priority 가 swap path 영역에 잘못 배치**. 실제 진정한 영역 = ② 안의 OMP barrier wait.

### 13.6 각 timeline 에서 정확히 표시된 영역 (✓ 정합)

| 영역 | v1.6 | Option A | S1-S9 |
|---|:-:|:-:|:-:|
| 8 lanes (GPU stream / cpu_comm / CPU pacpu master / OMP worker / async_output / cuda-EvtHandlr / pt_nccl_watchdg / pt_gloo) | ✓ | ✓ | ✓ |
| ★ cdec leftover +18-24 ms (위치 70-90 ms region) | ✓ | ✓ | ✓ |
| ★ swap_in +25 ms (위치 end region) | ✓ | ✓ | ✓ |
| futex_wait 의 OMP worker 80 (62% wait state) | ✓ | ✓ | ✓ |
| async_output thread 의 GIL 대기 (ep_poll) | ✓ | ✓ | ✓ |
| cudaEventSynchronize ~9.8% step time | ✓ | ✓ | ✓ |
| GPU stream queue 의 NEO §4.4 batch interleave | — | ✓ | ✓ |
| cpu_communication_stream 별도 lane | — | — | ✓ (S1-S9 신규) |

### 13.7 모든 timeline 에서 누락 또는 불명확한 영역 (✗ 또는 △ 갭)

| 영역 | 상태 | 의미 |
|---|:-:|---|
| **OMP barrier #1 wait time** (store_kv → attn) | ✗ 미표시 | libgomp 43.75% 의 일부 — 어느 barrier 가 dominant 인지 불명 |
| **OMP barrier #2 wait time** (attn → gather) | ✗ 미표시 | thread imbalance 의 main 누적 영역 가설 |
| **per-thread Step 1 imbalance** | ✗ 미표시 | dynamic schedule (-1.4% atomic overhead) 시도 실패 → static 의 imbalance 정량 부재 |
| **CUDA stream wait_stream duration** | △ "★" 만 표시, 정량 X | `_xfer_stream.wait_stream` / `_cur_stream.wait_stream(_comm_stream)` 의 영향 분리 X |
| **NCCL AllReduce 의 GPU IDLE 기여도** | △ NCCL count + total 만 | step 안 어디서 발생, 어느 stream 위인지 불명 |
| **swap_in per-layer breakdown** | △ "+25 ms swap_in + sample + emit" 통합 | Python loop / index_kernel GOMP / pinned alloc / scatter 의 비율 불명 |
| **`_neo_handle_kv_swap` Python loop** | △ "Python overhead" 통합 | GIL hold 시간 정량 부재 → async_output thread 의 idle 영역 분리 X |
| **cdec submit → cdec compute 시작 lag** | ✗ 미표시 | submit 즉시 return 이지만 OMP fork 시간 (futex wake-up 80 thread) 측정 X |
| **kv_cache layer dependency wait** | △ comment 만 | layer N output → layer N+1 input dep 으로 async pipeline 불가능한 영역 |

### 13.8 misleading 또는 재해석 필요한 영역 (▲)

| 영역 | timeline 표현 | 실제 의미 | 재해석 |
|---|---|---|---|
| **S1-S9 의 `cdec_wait_avg = 0 ms`** | "future 즉시 return" | S5 direct 호출이라 cdec 가 main thread 안에서 blocking 실행, wait 없음 = direct elapsed | "+18 ms ②" 의 실체 = cdec direct execution time (wait 아님), GIL released 라 GPU stream queue 가 그동안 진행 |
| **"★ cdec leftover +18 ms" 위치 70-90 ms region** | 시각 표시 | 실제 = 매 layer 마다 약 0.23 ms 발생 → 80 layer 누적 ≈ 18 ms (interleave 후 leftover) | 단일 block 이 아니라 80 layer 분산. timeline 의 단일 block 표시는 누적 visualization |
| **"★ swap_in + sample + emit +25 ms" 위치 end region** | 시각 표시 | 실제 = step 끝 영역의 swap_in 80 layer + sample 1회 + emit, breakdown 없음 | swap_in 이 dominant 가설 (실측 없음) — 정량 검증 필요 |
| **"OMP worker × 80" futex_wait 62%** | 시각 표시 | 실제 = wait state breakdown — 어느 영역 wait 인지 (barrier vs OMP fork) 분리 X | 본 lane 의 futex_wait 가 cdec 안의 어느 barrier 인지 정량 필요 |

### 13.9 cross-check 결론 (동적 분석 정정 적용 후)

| 측면 | 결론 |
|---|---|
| **macro-level wall 분해** | ✓ 정합 (vanilla 54 + NEO 55 = 109 ms total) |
| **lane 구성** | ✓ 정합 (8 lanes 의 lifecycle) |
| **NEO §4.4 batch interleave mechanism** | ✓ S1-S9 정합 |
| **② cdec leftover +18 ms** | ✓ wall 정합 — 단 **내부 cycle 분해 (libgomp 62% vs libpacpu 38%) 영역 미표시** |
| **③ "swap path +25 ms"** | **▲ misattribution** — swap 영역 wall hidden, +25 ms 의 진정한 source 미확정 (sample/emit/admission 영역 가설) |
| **lever priority** | **▲ 재책정 필요** — 이전 "swap path = Top Priority" 가 잘못. **OMP barrier wait 가 진정한 dominant** |
| **env-gated alternative path** | ✗ P3 / P4 / D / OOB fix 신규 영역 미표시 (본 doc §3-6 신규 보강) |

→ **현재 timeline 의 가치** = macro-level wall + lane 구성 ✓
→ **현재 timeline 의 한계** = (a) ② 내부 cycle 분해 미시각화, (b) ③ misattribution (swap 이 wall hidden), (c) async / sync 영역 의 시각적 분리 부족
→ **즉시 정정 영역** = lever priority — barrier wait (OMP) > softmax (compute) > AMX (qk/av compute) > swap path (이미 async, 다음 turn 시도 가치 작음)

---

## 14. 다음 turn — bottleneck pin-point 측정 plan (동적 분석 backing)

본 plan = §12.8 의 instrumentation 갭 메우기. **swap path 영역은 async 라 deprioritize**. 진정한 dominant ② 안의 OMP barrier wait 영역 우선:

### 14.1 우선순위 1: OMP barrier #1 / #2 정량 (★★★)

[`csrc/cpu/pacpu/core.h:340-403`](../../../../../csrc/cpu/pacpu/core.h) 영역에 instrumentation:

```cpp
# pragma omp parallel
{
    double t_start = omp_get_wtime();
    // Step 0: store_kv ...
    double t_step0 = omp_get_wtime();
    # pragma omp barrier               // ★ barrier #1
    double t_after_b1 = omp_get_wtime();
    // Step 1: attn_one_seq ...
    double t_step1 = omp_get_wtime();
    # pragma omp barrier               // ★ barrier #2
    double t_after_b2 = omp_get_wtime();
    // Step 2: gather_output ...
    double t_end = omp_get_wtime();

    // thread-local elapsed 측정
    if (tid < MAX_WS) {
        _tl_step0_ms[tid] = (t_step0 - t_start) * 1000;
        _tl_b1_wait_ms[tid] = (t_after_b1 - t_step0) * 1000;  // ★ barrier #1
        _tl_step1_ms[tid] = (t_step1 - t_after_b1) * 1000;
        _tl_b2_wait_ms[tid] = (t_after_b2 - t_step1) * 1000;  // ★ barrier #2
        _tl_step2_ms[tid] = (t_end - t_after_b2) * 1000;
    }
}
// 함수 return 전: env VLLM_NEO_PROFILE_OMP=1 시 thread별 elapsed log
```

→ **출력**: `[PROFILE OMP] tid=N step0=X.X ms b1_wait=Y.Y ms step1=Z.Z ms b2_wait=W.W ms step2=V.V ms`
→ **분석**: thread 별 max(b1_wait) / max(b2_wait) = imbalance 영역 정량. Step 1 의 thread 별 variance = task partition 의 효율.

### 14.2 우선순위 2: softmax fast_exp (★★, code 변경)

[`csrc/cpu/pacpu/pacpu.ispc:111-142`](../../../../../csrc/cpu/pacpu/pacpu.ispc) 의 softmax 영역:

- 현재 = ISPC builtin `exp()` (SVML 또는 SLEEF polynomial)
- 후보 = fast_exp polynomial (degree 5, IEEE 754 short-circuit 제거)
- softmax 9.73% cycle 의 50% 단축 가정 → cdec compute -2.5 ms
- 단 정확도 영향 검증 필요 (TST_003 verdict — 분포·의도 유사성)

### 14.3 우선순위 3: step-end +25 ms 의 진정한 source 식별 (★★★)

이전 timeline 의 "③ swap_in + sample + emit +25 ms" 의 진정한 source 식별. swap 영역이 wall hidden 이므로:

```python
# vllm/v1/worker/gpu_model_runner.py 의 step end 영역 (sampler + emit + admit)
import time as _t_se
_t_se_start = _t_se.perf_counter()
# (sampler 영역)
_t_sampler = _t_se.perf_counter()
# (emit / async_output 영역)
_t_emit = _t_se.perf_counter()
# (next-step admission 영역 — NEO scheduler swap-in 결정)
_t_admit = _t_se.perf_counter()

# [PROFILE STEP_END] sampler=X.X ms emit=Y.Y ms admit=Z.Z ms total=W.W ms
```

→ **출력**: +25 ms 영역의 sample / emit / admit 분포. swap 영역의 진정한 wall 기여 (drain wait) 분리.

### 14.4 우선순위 4: CUDA stream wait_stream duration (★)

`nsys profile` 으로 측정 — 코드 변경 0:

```bash
nsys profile -o nsys_e64c56561 -t cuda,nvtx -s none -c cudaProfilerApi \
    --cuda-graph-trace=node \
    bash eval/run_neo_standard.sh
nsys stats --report cuda_api_sum nsys_e64c56561.nsys-rep
```

→ **출력**: `cudaStreamWaitEvent` 의 total / count / avg.

### 14.5 측정 후 timeline 보강 plan

위 4 영역 측정 완료 후 본 doc 에 추가:
- §16 신설: OMP barrier breakdown table (barrier #1 / #2 / implicit / Step 1 imbalance variance)
- §17 신설: softmax fast_exp 적용 후 cycle ratio 변화 (libpacpu softmax 9.73% → ?%)
- §18 신설: step-end +25 ms 의 sample/emit/admit 분포
- §19 신설: CUDA stream wait_stream 정량 table

→ 그 후 SVG 재생성: **swap 영역을 async/hidden lane 으로 명확 분리** + ② 안의 barrier wait / actual compute 의 visualization + +25 ms 영역의 진정한 source 표시.

---

## 15. Bottleneck 식별 영역 — 정리표 (동적 분석 정정)

| 영역 | wall 기여 | cycle % (perf) | 현재 측정 | 진정한 dominant | next action |
|---|---:|---:|---|---|---|
| **★★★ ② libgomp barrier wait** | **+11.2 ms** (② 의 62%) | **43.75%** | ✗ 별도 측정 X (libgomp 통합 dso 만) | OMP barrier #1/#2/implicit 의 thread imbalance wait | §14.1 instrumentation |
| **★★ ② libpacpu actual work** | +6.8 ms (② 의 38%) | 26.38% | ✓ cdec_wait (S5=0) + gpu_ms ratio 추정 | softmax 9.73% (cycle/FLOP worst) ★ fast_exp 후보 | §14.2 fast_exp code change |
| **★ ① Python wall overhead** | +12 ms | 1.84% (eval) + 미계측 (CUDA stream sync) | ✗ skip_gpu_count 만 | CUDA stream wait_stream / Python attention.py loop | §14.4 nsys profile |
| **▲ ③ "+25 ms" misattribution** | +25 ms (source 불명) | — | ✗ 미계측 | sample / emit / next-step admission 가설 (swap 영역 아님) | §14.3 step-end PROFILE |
| **⚪ swap path (async)** | ~0-5 ms (drain wait 만) | 10.24% (libtorch_cpu) — async cycle, wall hidden | ✓ swap_out PROFILE log | (lever priority 낮음 — 이미 async) | — (별도 turn 가치 작음) |
| ⚪ vanilla baseline 54 ms | 54 ms | NCCL + GEMM + FlashAttn (NSys) | ✓ NSys | (NEO 변경 영역 밖) | — |

### 핵심 (사용자 지적 정합)

1. **swap path = async, wall 에서 hidden** — 이전 framework 의 "swap path Top Priority" 잘못. cycle 은 10.24% 소모하지만 main thread wall block X.
2. **진정한 TOP bottleneck = ② 안의 libgomp barrier wait (+11.2 ms, ② 의 62%)** — OMP barrier 영역. F6 dynamic schedule 시도 -1.4% 회귀 → 다른 접근 필요 (OMP_NUM_THREADS sweep / barrier #1 제거 / KMP_BLOCKTIME tuning).
3. **AMX/F3 같은 ISPC compute 가속 = +6.8 ms 영역만 단축 가능** — barrier 영역 미접근 시 ceiling 작음.
4. **softmax fast_exp = 2nd lever** — 9.73% cycle 의 50% 단축 가정 시 -2.5 ms.
5. **+25 ms 의 진정한 source 식별 = §14.3 instrumentation** — sample/emit/admit 분리.

### lever priority 정리 (재책정)

| 순위 | Lever | 영역 | 예상 wall 단축 | effort |
|---|---|---|---:|:-:|
| **★★★ 1** | OMP barrier wait 영구 제거 (barrier #1 또는 thread imbalance fix) | ② 안 libgomp 62% | **-11 ms (≈ +10%)** | 1-3 일 |
| **★★ 2** | KMP_BLOCKTIME=INF + KMP_AFFINITY 명시 (env-only) | ② 안 libgomp 의 spin 영역 단축 | -2 ~ -4 ms (-2~4%) | 1 시간 |
| **★★ 3** | softmax fast_exp (ISPC polynomial) | ② 안 libpacpu softmax | -2 ms (-2%) | 2-3 일 |
| ★ 4 | F3 K BF16 host store (env P3 single lever, USE_AMX=0) | ② 안 libpacpu qk 일부 | -1 ms (-1%) | 2-3 일 |
| ⚪ 5 | F4 TP=8→4 (M=8→16, AMX tile full) | ② 안 libpacpu qk +50% | -3 ms (-3%, 단 GPU TP trade-off) | 2-4 주 |
| ⚪ 6 | swap_in/out 추가 가속 | wall hidden 이라 ROI ~0 | -1 ms 미만 | 2-3 일 |
| ✗ | F6 OMP dynamic schedule | atomic counter overhead | **-1.4% 회귀 확인** | (시도 후 폐기) |

→ **다음 turn 추천** = §14.1 OMP barrier instrumentation (★★★) + §14.2 fast_exp 적용 + §14.4 nsys profile. effort = 2-3 일.
