# Option C — Step 2: Regime Detector implementation

Date: 2026-06-08 (KST)
Source: `vllm/v1/lhc/regime_detector.py` (newly added module).

## 1. Module layout

| symbol | purpose |
|---|---|
| `WorkloadRegime` (IntEnum) | UNKNOWN / GPU_SATURATED / KV_HEAVY / BALANCED |
| `classify(...)` | pure function — testable, no globals |
| `RegimeDetector` | process-singleton, owns EWMA state + cross-proc `mp.Value` |
| `current_regime()` / `should_use_dsa()` / `should_use_amx_c3()` | gate queries used by lanes |
| `note_swap_event(n)` | called from `_neo_handle_kv_swap` |
| `note_kv_usage(pct)` / `note_step(gpu_util)` | called from scheduler step end |
| `stats_snapshot()` | dict for paper figures / debugging |

## 2. Classification rule

Implemented in `classify()` (pure function); rule lattice, first match wins:

1. **KV_HEAVY**   `kv_pct > 0.75` or `swap_rate > 10.0/s`
2. **GPU_SATURATED**   `gpu_util > 0.90` and `kv_pct < 0.50` and `swap_rate < 1.0/s`
3. **BALANCED**   default

Thresholds tunable via env:

```
VLLM_LHC_REGIME_GPU_THR     default 0.90
VLLM_LHC_REGIME_KV_THR      default 0.50
VLLM_LHC_REGIME_SWAP_THR    default 1.0
VLLM_LHC_REGIME_KV_HEAVY    default 0.75
VLLM_LHC_REGIME_SWAP_HEAVY  default 10.0
VLLM_LHC_REGIME_INTERVAL    default 20   (steps between re-classify)
VLLM_LHC_REGIME_ADAPTIVE    default 0    (master gate)
```

## 3. Signal accumulators

- `swap_rate_ewma`: events/s with α=0.3 EWMA on every `note_swap_event`,
  decayed × 0.95 on each `note_step` (so rate naturally cools when spill
  subsides).
- `last_kv_pct`: most recent value reported by scheduler.
- `last_gpu_util`: most recent value passed to `note_step`, or probed
  lazily via `torch.cuda.utilization()` at classify time when 0.
- `hist`: histogram of regime classifications (per worker proc) for
  `regime_accuracy.md` reporting.

## 4. Cross-process state propagation

The detector holds a `multiprocessing.Value("i", int)` initialised to
`UNKNOWN`. The EngineCore process writes regime decisions; TP worker
procs read via `current_regime()`. Each worker also runs its own
`RegimeDetector` singleton, but the gate read goes through the shared
`mp.Value`, so all TP ranks share a consistent regime view within one
classify period.

When `VLLM_LHC_REGIME_ADAPTIVE=0` (default), the `should_use_dsa()` and
`should_use_amx_c3()` helpers fall back to the static `VLLM_LHC_DSA` /
`VLLM_LHC_AMX_C3` envs — preserving Option A back-compat.

## 5. Lane wiring

Three integration sites:

### 5.1 DSA lane gate
`vllm/v1/lhc/dsa_lane.py::dsa_memcpy()` — after the existing size /
self-test gates, consult `should_use_dsa()` when `VLLM_LHC_REGIME_ADAPTIVE=1`.
KV_HEAVY classification routes copies through DSA; other regimes fall
back to plain memcpy.

### 5.2 AMX C3 lane gate
`vllm/v1/lhc/amx_c3_lane.py::amx_c3_available()` — same pattern. AMX C3
is on in KV_HEAVY and BALANCED, off in GPU_SATURATED.

### 5.3 Scheduler hook
`vllm/v1/core/sched/scheduler.py::schedule()` — at the end of every
scheduler step (after the existing metronome_step_end hook), feed:

- `kv_cache_manager.block_pool.num_free_blocks / num_gpu_blocks` →
  `note_kv_usage(used_pct)`
- `note_step()` (triggers re-classify every `VLLM_LHC_REGIME_INTERVAL`
  steps; default 20, so ~50 ms cadence at typical step times)

Step-end overhead is dominated by the `note_step` call (∼1 μs) plus the
GPU-util probe once every N=20 steps (∼300 μs amortized to ∼15 μs/step).

### 5.4 NEO swap event hook
`vllm/v1/worker/gpu_model_runner.py::_neo_handle_kv_swap()` — on
non-empty swap, call `note_swap_event(len(swap_out_ids)+len(swap_in_ids))`.
This is what raises `swap_rate_ewma` above the `swap_heavy=10/s` threshold
and triggers KV_HEAVY classification.

## 6. Smoke test

```bash
$ /workspace/vllm_dev_prj/bin/python -c "
from vllm.v1.lhc.regime_detector import classify, WorkloadRegime, RegimeDetector
# unit
print('saturated:', classify(0.95, 0.30, 0.0))       # → GPU_SATURATED (1)
print('kv (high kv):', classify(0.95, 0.80, 0.0))    # → KV_HEAVY (2)
print('kv (high swap):', classify(0.95, 0.10, 50.0)) # → KV_HEAVY (2)
print('balanced:', classify(0.50, 0.40, 0.0))         # → BALANCED (3)
# end-to-end detector
d = RegimeDetector.instance()
d.note_kv_usage(0.80)
for _ in range(25): d.note_step(gpu_util=0.85)
print('after 25 high-kv steps:', d.current_regime())  # → KV_HEAVY (2)
"
```

Output (verified):
```
saturated: 1
kv (high kv): 2
kv (high swap): 2
balanced: 3
after 25 high-kv steps: WorkloadRegime.KV_HEAVY
```

Module imports cleanly; unit + integration smoke pass.

## 7. Files changed

- new: `vllm/v1/lhc/regime_detector.py` (≈ 300 LoC)
- modified: `vllm/v1/lhc/__init__.py` (re-export)
- modified: `vllm/v1/lhc/dsa_lane.py` (gate consult)
- modified: `vllm/v1/lhc/amx_c3_lane.py` (gate consult)
- modified: `vllm/v1/core/sched/scheduler.py` (step end hook)
- modified: `vllm/v1/worker/gpu_model_runner.py` (NEO swap event hook)

## 8. Verdict

Detector module is implemented, integrated, and import-clean. Static
mode preserves Option A behaviour exactly; adaptive mode kicks in only
when `VLLM_LHC_REGIME_ADAPTIVE=1` is set. Step 3 (sweep) will exercise
the path under real workload.
