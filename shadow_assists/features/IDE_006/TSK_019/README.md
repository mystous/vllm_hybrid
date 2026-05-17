# TSK_019 — Best Configuration index

> Best configuration 영역 history + 개략 정보. 상세 fact = 각 best file 영역.

## Best history (3-run avg/min/max)

| 시각 (KST) | best file | commit | 3-run avg | min | max | CV | wall avg | workload | 22 strict | 비고 |
|---|---|---|---:|---:|---:|---:|---:|---|:-:|---|
| **2026-05-17 14:25** | [`Best_S1_S9_2238tps.md`](Best_S1_S9_2238tps.md) | `feat/neo-option-b` (v1.6 base + S1-S9) | **2,238.6** | **2,153.6** | **2,303.4** | **3.44%** | **1,819s** | 500p × 8192 | **19/19** | **★ NEO 원본 100% 정합** (10/10 함수). NEO §4.4 정통 implement 완성 |
| 2026-05-14 ~ 16 | [`Best_v1.6_2157tps.md`](Best_v1.6_2157tps.md) | `64f9e0c48` | 2,197.4 | 2,156.9 | 2,223.8 | 1.62% | 1,844s | 500p × 8192 | 19/19 | strict 19/19 + shape_mismatch=0. Run 1 (2026-05-14) + Run 2/3 (2026-05-16 회귀) |
| 2026-05-15 20:31 | [`Best_Phase3_1_kmp50.md`](Best_Phase3_1_kmp50.md) | `099d23e54` | 2,038.7 (1-run) | — | — | — | 1,591s | **400p** × 8192 | 19/19 | Phase 3.1 (OMP persist) + KMP_BLOCKTIME=50, single-trial |

## Reference measurements (3-run avg/min/max)

| 시각 (KST) | dir | tps avg / min / max | CV | wall avg | workload | runs | 용도 |
|---|---|---:|---:|---:|---|:-:|---|
| 2026-05-10 17:16 | [`measurements/vanilla_3run_20260510/`](measurements/vanilla_3run_20260510/) | **4,690.7** / 4,690.4 / 4,691.0 | **0.006%** | 873s | 500p × 8192 | 3 | vanilla 분모 (NEO OFF) |
| **2026-05-17 14:25** | [`measurements/neo_s1_s9_500p_3run_20260517/`](measurements/neo_s1_s9_500p_3run_20260517/) | **2,238.6** / 2,153.6 / 2,303.4 | **3.44%** | 1,819s | 500p × 8192 | 3 | **★ S1-S9 best (NEO 원본 100% 정합)** |
| 2026-05-14 ~ 16 | [`measurements/neo_v1_6_500p_3run_20260516/`](measurements/neo_v1_6_500p_3run_20260516/) | **2,197.4** / 2,156.9 / 2,223.8 | **1.62%** | 1,844s | 500p × 8192 | 3 | v1.6 best (commit `64f9e0c48`) |
| 2026-05-16 07:21 | [`measurements/neo_phase3_1_kmp200_500p_3run_20260516/`](measurements/neo_phase3_1_kmp200_500p_3run_20260516/) | 2,134.9 / 2,013.2 / 2,255.7 | 5.68% | 1,914s | 500p × 8192 | 3 | Phase 3.1 (Persistent OMP) KMP=200 |
| 2026-05-16 09:56 | [`measurements/neo_phase3_1_3_kmp200_500p_3run_20260516/`](measurements/neo_phase3_1_3_kmp200_500p_3run_20260516/) | 2,083.3 / 2,015.4 / 2,145.4 | 3.13% | 1,957s | 500p × 8192 | 3 | Phase 3.1+3.3 (cherry-pick `0717f4b8c`) |

## 개략 정보

### S1-S9 (500p best — 3-run avg 2,238.6 tps, CV 3.44%) ★ NEO 원본 100% 정합
- workload: 500p × 8192 in/out (4M total tokens)
- 핵심: NEO §4.4 algorithm-correct path 정통 implement (S1-S9 9 단계 rewrite)
  - S1: `_neo_comm_wait_compute` / `_neo_compute_wait_comm` helper (NEO 동등)
  - S3: Option B async deque 제거 → Option A sync path 만 유지
  - S4: `forward_double` 의 `with cuda.stream(s0/s1):` 제거
  - S5: `ThreadPoolExecutor.submit` → `_neo_cdec_compute_cpu` 직접 호출 + `_NeoDirectFuture`
  - S7: `_get_batch_streams()` (s0/s1 stream pair) dead code 제거
  - S8: `forward_double` NEO `_forward_pipeline_stage(cur_stage)` ordering 정합
  - S9: result D2D copy 를 `cpu_communication_stream` 위 async + `_compute_wait_comm()` 호출
- 3-run: 2,303.4 / 2,153.6 / 2,258.9 → avg 2,238.6
- vs vanilla 3-run avg 4,690.7: **47.7%** (+0.9pt vs v1.6 best)
- vs v1.6 best 3-run avg 2,197.4: **+1.9%**
- NEO 원본 `swiftllm/worker/layers/transformer_layer.py + model.py` 10/10 함수 정합 ✓
- branch: `feat/neo-option-b` (v1.6 base `64f9e0c48` + S1-S9)

### v1.6 (500p — 3-run avg 2,197.4 tps, CV 1.62%)
- workload: 500p × 8192 in/out (4M total tokens)
- 핵심: shape mismatch fix (`NeoCpuKvBuffer._in_flight_swap_out`)
- 3-run: 2,156.9 / 2,223.8 / 2,211.6 → avg 2,197.4
- vs vanilla 3-run avg 4,690.7: **46.8%**
- S1-S9 base (commit `64f9e0c48`)

### Phase 3.1+KMP=50 (400p best)
- workload: 400p × 8192 (3.2M total tokens — v1.6 보다 80% 영역)
- 핵심: OMP persistent (omp_set_dynamic(0)) + KMP_BLOCKTIME=50ms
- cdec_wait 영역 2.68→2.38ms (−11.2%)
- Phase 3.4 baseline 영역 (400p, 1,930.5 tps) 대비 +5.61%

### 직접 비교 한계

v1.6 best 2,157 tps (500p) vs Phase 3.1+KMP=50 2,038 tps (400p) 영역 의 workload 영역 다름 — 직접 throughput 영역 비교 X. 동일 workload 영역 (400p) 영역 의 비교:
- Phase 3.4 baseline (400p, env Phase 3.1 적용 X): 1,930.5 tps
- Phase 3.1+KMP=50 (400p): 2,038.7 tps (+5.61%)

## vanilla vs NEO 1-step Timeline

### S1-S9 (NEO 원본 100% 정합, 2026-05-17 측정) ★ 현재 best
상세: [`measurements/timeline_v16_s1_s9_20260517/README.md`](measurements/timeline_v16_s1_s9_20260517/README.md)

![vanilla vs NEO 1-step Timeline (S1-S9)](measurements/timeline_v16_s1_s9_20260517/timeline_schematic.svg)

S1-S9 의 차이 영역 (vs Option A):
- GPU stream: default + s0 + s1 (3 개) → default + cpu_communication_stream (2 개, NEO 원본 정합)
- cdec dispatch: `ThreadPoolExecutor.submit` → `_neo_cdec_compute_cpu` 직접 호출
- cdec result wait: `cdec_future.result()` blocking 24 ms → `_NeoDirectFuture.result()` 즉시 return (cdec 시간은 직접 호출에 흡수)
- result D2D copy: main stream sequential → `cpu_communication_stream` async + `_compute_wait_comm()` 정합
- forward_double ordering: `with cuda.stream(s0/s1)` 동시 launch → NEO `_forward_pipeline_stage(cur_stage)` ordering (batches[cur_stage] postproj+preproj 가 batches[other] attention *앞*)
- ② cdec compute (S5 직접 호출): Option A 의 +24 ms → S1-S9 의 +18 ms (−6 ms, batch interleave + cpu_comm_stream hide)

### Option A (v1.6, sync result wait, 2026-05-16 18:30)

상세: [`measurements/timeline_v16_optionA_20260516/README.md`](measurements/timeline_v16_optionA_20260516/README.md)

![vanilla vs NEO 1-step Timeline (Option A)](measurements/timeline_v16_optionA_20260516/timeline_schematic.svg)

### cdec dispatch 의 2 단계 "async" 정의

| 단계 | 의미 | 우리 measurement |
|---|---|---|
| **cdec submit** | `cdec_executor.submit(...)` — CPU pacpu 별도 worker process 비동기 실행 | **항상 async ✓** |
| **cdec result wait** | `cdec_future.result()` main thread 가 결과 받는 시점 | **두 옵션** |

- **Option A** (`attention.py:1148`): same layer 의 attention 안 main thread blocking → wall path 위
- Option B (`attention.py:1133`, env `VLLM_NEO_ASYNC_CDEC=1`): pending queue + next layer drain (NEO §4.4 algorithm-correct path). 우리 implement 활성 시 starvation 으로 step 멈춤 (drain timing 미완성)

→ **2주 측정 전체 = Option A**. NEO 의 "async" 라 부른 게 *submit 단계 async* — 단 result wait 는 sync.

### NEO 추가 61 ms / step 의 출처 (Option A 영역)

| # | 영역 (timeline 위치) | 추가 시간 | 원인 |
|---|---|---:|---|
| ① | `Python attention.py hot path × 80 layer` (54-66 ms) | **+12 ms** | skip_gpu_attn check, _neo_drain_pending_cdec (Option A 에서 queue empty 라 no-op), cdec submit, cudaStream sync |
| **②** | **`cdec_future.result()` Option A BLOCKING wait** (66-90 ms) | **+24 ms** | max_workers=2 cap 으로 56 cdec / 2 worker = 64 ms CPU work, 24 ms 가 wall 위로 |
| ③ | `swap_in launch + Python overhead + emit` (90-115 ms) | **+25 ms** | `_neo_handle_kv_swap` Python loop, ATen `index_kernel` GOMP, `copy_layer_out` advanced indexing |
| | **합** | **+61 ms** | vanilla 54 + 61 = NEO 115 ms ✓ |

vanilla 54 ms 이후 영역에서 **GPU 는 거의 idle (utilization 21%)**. CPU thread 도 90% idle wait.

### 가속 lever

| lever | 영역 | 예상 효과 |
|---|---|---:|
| swap path Python+ATen 제거 (After-NEO plan ★ Top Priority) | ③ +25 ms 의 절반 + ① 절반 | tps +11~25% |
| Option B (NEO §4.4) 완성 — drain timing fix | ② +24 ms 제거 | tps +26% (구현 필요) |
| 두 lever 합산 | ①②③ 의 대부분 | vanilla 의 ~60-70% 도달 |

## variance fact (vanilla vs NEO 측정 3-run avg/min/max)

| path | runs | min — max | avg | CV | vs vanilla |
|---|:-:|---|---:|---:|---:|
| vanilla (500p, NEO OFF) | 3 | 4,690.4 — 4,691.0 | 4,690.7 | **0.006%** | — |
| **★ NEO S1-S9 (branch `feat/neo-option-b`)** | 3 | 2,153.6 — 2,303.4 | **2,238.6** | **3.44%** | **47.7%** |
| NEO v1.6 (commit `64f9e0c48`) | 3 | 2,156.9 — 2,223.8 | 2,197.4 | 1.62% | 46.8% |
| NEO Phase 3.1 (KMP=200, 400p) | 3 | 1,918.6 — 2,251.4 | 2,044.0 | 8.85% | — |
| NEO Phase 3.1 (KMP=200, 500p) | 3 | 2,013.2 — 2,255.7 | 2,134.9 | 5.68% | 45.5% |
| NEO Phase 3.1+3.3 (KMP=200, 500p) | 3 | 2,015.4 — 2,145.4 | 2,083.3 | 3.13% | 44.4% |

→ **S1-S9 = 모든 NEO 측정 중 가장 높은 avg** (2,238.6 tps, +1.9% vs v1.6 best). v1.6 base 위 NEO 원본 정통 9 단계 rewrite (10/10 함수 정합).
→ S1-S9 의 CV 3.44% 는 v1.6 best (1.62%) 보다 큼 — variance source 가 작은 rewrite scope 의 stochastic 영향. 단 min 2,153.6 ≈ v1.6 best min 2,156.9 → variance bottom 안정.
→ Phase 3.1 (Persistent OMP) / Phase 3.3 (CUDA Stream Priority) 는 v1.6 baseline 보다 throughput avg −3% ~ −5% 하락 + variance 증가 → 회귀로 폐기.
→ vanilla = deterministic, NEO = run variance 잔존. variance source = NEO scheduler 의 wall-clock 의존 trigger (`time.time()`, KV pool snapshot 도달 시점) + predictor 의 rolling perfdata 의존. 상세 = `measurements/*/README.md`.

## 관련 영역 file

| 영역 | 위치 |
|---|---|
| ★ S1-S9 best fact + 재현 | [`Best_S1_S9_2238tps.md`](Best_S1_S9_2238tps.md) |
| S1-S9 rewrite plan (9 단계 + 정적 영향도) | [`analysis/G_neo_rewrite_plan.md`](analysis/G_neo_rewrite_plan.md) |
| v1.6 best fact | [`Best_v1.6_2157tps.md`](Best_v1.6_2157tps.md) |
| Phase 3.1+KMP=50 (400p best) | [`Best_Phase3_1_kmp50.md`](Best_Phase3_1_kmp50.md) |
| 5-phase 분석 산출 (Phase A-F + G_neo_rewrite_plan) | `analysis/` (14 .md) |
| 본 plan 구현 plan | `After_NEO_implementation_plan.md` |
| v1.6 성능 분석 | `Performance_analaysis_v1.6.md` |
| reference / log archive | `measurements/` |
