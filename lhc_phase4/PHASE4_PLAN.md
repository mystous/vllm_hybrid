# LHC Phase 4 — METRONOME-LHC integration

**날짜**: 2026-06-08
**상위**: `lhc_phase3/PHASE3_VERDICT.md` (§5.2 task list)

> **⛔ 기각 (2026-06-11, 사용자 판정)**: Metronome(METRONOME-LHC) 아이디어 기각.
> 논문에는 부정 결과로만 축소 보존 (paper §06 subsec:lhc-def "시도와 기각" +
> §08 subsec:res-metronome-lhc "기각 verdict"). 표 (`tbl_metronome_lhc.tex`) ·
> 그림 (`fig_lhc_pipeline.tex`) 삭제, **LHC_P4_004/005 측정 계획 폐기** —
> `run_metronome_sweep.sh` / `aggregate.py` 실행 금지. 측정 완료분 (Option A
> noise band / misuse / Option C) 은 부정 결과 증거로 논문에 잔존.

## 1. Phase 4 산출물

| ID | 임무 | 상태 |
|---|---|---|
| **LHC_P4_001** | NEO swap-out CUDA assert 디버그 | **DONE (OOB guard 적용)** |
| **LHC_P4_002** | OOB filter `_neo_handle_kv_swap` 진입점 + swap-in | DONE (옵션 (b) drop + warn) |
| **LHC_P4_003** | AMX C3 standalone hook | DONE (`vllm/v1/lhc/amx_c3_lane.py`) |
| **LHC_P4_003a** | METRONOME-LHC 5-stage 알고리즘 | DONE (`vllm/v1/lhc/metronome/`) |
| **LHC_P4_004** | W-D1/2/3 pilot 재실행 | **폐기 (2026-06-11 기각)** |
| **LHC_P4_005** | 9 wl × 7 cfg × 3 sweep | **폐기 (2026-06-11 기각)** |
| **LHC_P4_006** | paper §06/§08 갱신 | DONE → 2026-06-11 기각 verdict 로 재갱신 (§06 축약 + §08 기각 절) |

## 2. LHC_P4_001 — Root cause + fix

### 2.1 Root cause
W-D1 lhc_dsa pilot 의 CUDA device-side assert 의 원인:
- 각 worker 의 `num_gpu_blocks` 가 cudagraph profiling 단계에서
  `num_gpu_blocks_override=64` 로 override 됨 (`_init_minimal_kv_cache_for_profiling`).
- 정상 흐름에서는 profiling 끝나면 `_cleanup_profiling_kv_cache` →
  `self.cache_config.num_gpu_blocks = None` 으로 정리.
- 그러나 worker 측의 *실제 KV cache 텐서* shape[0] 이 어느 시점에 113371 으로
  alloc 되었는지 단언할 수 없음. swap-out path 에서 `kv[0][gpu_idx]` 시
  gpu_idx 가 worker KV cache shape[0] (=64) 을 초과 → CUDA device-side assert.

### 2.2 Fix
`vllm/v1/worker/gpu_model_runner.py::_neo_handle_kv_swap` 진입점에:
1. `_kv_cap = int(self.kv_caches[layer0][0].shape[0])` 추출
2. swap-out for-loop 안에서 `max(gpu_blocks) >= _kv_cap` 이면 warn-and-drop
3. swap-in for-loop 도 동일 guard
4. 첫 1회만 warn (cascade log 회피)

이 fix 는 옵션 (b) (OOB 검증 + drop). engine crash 회피가 최우선이며,
실제 swap path 가 작동해야 하는 환경에서는 worker 의 KV cache 가
113371 blocks 로 정상 alloc 된 상태이므로 guard 가 fire 하지 않음.

## 3. METRONOME-LHC 알고리즘 (LHC_P4_003a)

5-stage 구현 위치: `vllm/v1/lhc/metronome/`

| Stage | 파일 | 책임 |
|---|---|---|
| TEMPO | `tempo.py` | PMU sampler daemon (100 Hz ring buffer, DSA queue / mem_bw / cpu / AMX util) |
| CHORD | `chord.py` | cross-lane producer-consumer FIFO (sampler/scatter/detok) |
| METER | `meter.py` | NUMA-aware rank pin + libnuma preferred |
| ACCENT | `accent.py` | DSA budget gate (queue/llc/bw thresh) |
| RITARDANDO | `ritardando.py` | saturation → lane pause TTL fallback |
| Orchestrator | `orchestrator.py` | 5-stage 통합 entry: `metronome_start/step_end/stop` |

### 3.1 활성화

```bash
export VLLM_LHC_METRONOME=1       # master gate
export VLLM_LHC_DSA=1              # DSA lane (Phase 3)
export VLLM_LHC_DSA_WQ_PER_RANK=1  # PASID safe
export VLLM_LHC_AMX_C3=1           # AMX prefix scan
# optional tuning:
export VLLM_LHC_METRONOME_HZ=100
export VLLM_LHC_METRONOME_RING=1024
export VLLM_LHC_ACCENT_DSA_Q=12
export VLLM_LHC_RIT_DSA_SAT=32
```

### 3.2 Hook 위치

| hook | 파일 | event |
|---|---|---|
| `metronome_start()` | `vllm/v1/worker/gpu_worker.py::init_device` end | worker init 직후 |
| `metronome_step_end()` | `vllm/v1/core/sched/scheduler.py::schedule` end | 매 step 끝 |
| ACCENT/RITARDANDO gate | `vllm/v1/core/sched/neo_cpu_kv_buffer.py::copy_all_layers_in_from_staged` | DSA dispatch 판단 시 |

## 4. 측정 실행 (사용자 머신 — B200)

### 4.1 단위 검증 (W-D1 1 sweep — NEO fix 검증)

```bash
cd /workspace/host_vllm_hybrid
WORKLOADS=wd1 CONFIGS=vanilla SWEEPS=1 \
    bash lhc_phase4/run_metronome_sweep.sh
WORKLOADS=wd1 CONFIGS=metronome SWEEPS=1 \
    bash lhc_phase4/run_metronome_sweep.sh
```

검증 포인트:
- engine crash 없음 (이전엔 CUDA assert)
- `[NEO LHC_P4_001] swap-out OOB drop` 가 fire 한다면 worker KV cache
  실제 capacity 가 부족함 (다음 분석)
- `metronome_lhc_stats` 가 step end 마다 갱신

### 4.2 full sweep

```bash
bash lhc_phase4/run_metronome_sweep.sh    # 9 wl × 7 cfg × 3 sweep
python3 lhc_phase4/aggregate.py            # md/csv/tex 생성
```

총 189 cells, 각 cell 약 30-90 s + 30 s boot/cleanup, 예상 총 시간
약 5-6 시간 (B200).

## 5. Paper 통합 (LHC_P4_006 DONE)

| 위치 | 변경 |
|---|---|
| `paper/sections/06_ceres_algorithm.tex` | §06.6-9 신규 — Lane Separation / METRONOME-LHC 알고리즘 / Theorem 1-3 |
| `paper/sections/08_results.tex` | §subsec:res-metronome-lhc 신규 |
| `paper/figures/fig_lhc_pipeline.tex` | 5-stage 다이어그램 |
| `paper/tables/tbl_metronome_lhc.tex` | 9×7 placeholder (aggregate.py 후 자동 갱신) |

## 6. Phase 4 게이트

| 게이트 | 임계 | 측정 후 결과 |
|---|---|---|
| metronome ≥ +7% on ≥3/9 wl | $\Delta\!\ge\!+7\%$ | TBD |
| CPU util 증가 | +20 pp | TBD |
| metronome_sfx > Suffix 단독 | 가산성 | TBD |
| Theorem 1 (κ < 0.3) | 자원 dot product | TBD (DSA/AMX/NEO 단독 PMU) |
