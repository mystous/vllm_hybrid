# vLLM Patch Design — IDE_018 Phase-Burst Hook Insertion

> **status**: 설계 안 (별도 turn 에서 monkey-patch 또는 plugin entry point 로 적용).
> **scope**: `vllm/v1/worker/gpu_model_runner.py` 의 `_model_forward`, `_sample`, `execute_model` 세 지점에 `PhaseBurstRuntime.mark_phase()` 삽입.

---

## 1. Phase 매핑

| Phase enum | GPU 활동 | vllm 코드 위치 (현재 fork main) | hook 위치 (after-call) |
|---|---|---|---|
| `PHASE_ATTENTION` | attention layer matmul (memory-bound) | `_model_forward` 의 `self.model(...)` 진입 직후 — model.forward 내부 layer 단위 | **per-layer 정밀도는 invasive** → step-level 로 단순화: `_model_forward` 진입 직후 `mark_phase(ATTENTION)` |
| `PHASE_LINEAR` | FFN / MLP matmul (compute-bound) | model.forward 내부 → 외부에서 구분 불가 | step 단위에서는 PHASE_ATTENTION 으로 묶고, per-layer 분리는 별도 turn (layer hook 추가) |
| `PHASE_TP_ALLRED` | NCCL allreduce | model.forward 내부 distributed primitive | 외부 hook 불가 — `torch.distributed` `register_comm_hook` 가능 (별도 turn) |
| `PHASE_SAMPLE` | sampling kernel (logits, sampler) | `_sample()` line 3512–3540 | `_sample()` 진입 직후 `mark_phase(SAMPLE)` |
| `PHASE_POST_STEP` | bookkeeping, output detok queueing | `_bookkeeping_sync()` 진입 | `mark_phase(POST_STEP)` |
| `PHASE_IDLE` | inter-step (engine main loop) | `vllm/v1/engine/core.py` step 종료 후 | `mark_phase(IDLE)` 또는 PhaseBurstRuntime 측 timer 로 자동 |

> **practical compromise (Phase 1)**: per-layer attention/linear 구분은 model forward 코드 직접 수정 필요 (invasive, paper 검증 후 별도 turn). Phase 1 patch 는 `(attention+linear) → (sample) → (post)` 의 **3-phase coarse signal** 로 운영. paper §4 측정은 sampler.py 44% 의 sampling-phase 만으로도 CPU util elevate 충분 (SUB_161 비중).

---

## 2. Patch insertion points

### 2.1 `_model_forward` (line 3684+)

```python
def _model_forward(self, input_ids=None, positions=None, ...):
    # ── IDE_018 hook (patched) ──
    if _phase_burst_enabled:
        _phase_burst_rt.mark_phase(PHASE_ATTENTION, step_id=_current_step_id)
    # ────────────────────────────
    # existing dual sub-batch dispatch ...
    return self.model(...)
```

### 2.2 `_sample` (line 3512+)

```python
def _sample(self, logits, spec_decode_metadata):
    # ── IDE_018 hook (patched) ──
    if _phase_burst_enabled:
        _phase_burst_rt.mark_phase(PHASE_SAMPLE, step_id=_current_step_id)
    # ────────────────────────────
    sampling_metadata = self.input_batch.sampling_metadata
    ...
```

### 2.3 `_bookkeeping_sync` (line 3542+)

```python
def _bookkeeping_sync(self, ...):
    # ── IDE_018 hook (patched) ──
    if _phase_burst_enabled:
        _phase_burst_rt.mark_phase(PHASE_POST_STEP, step_id=_current_step_id)
    # ────────────────────────────
    ...
```

### 2.4 `execute_model` 진입 (line 4084) — step boundary + task enqueue

```python
def execute_model(self, scheduler_output, intermediate_tensors=None):
    # ── IDE_018 hook (patched) ──
    if _phase_burst_enabled:
        _phase_burst_step_id_increment()         # next step
        _phase_burst_enqueue_step_tasks(self, scheduler_output)
        _phase_burst_rt.mark_phase(PHASE_ATTENTION, step_id=_current_step_id)
    # ────────────────────────────
    if self.execute_model_state is not None:
        raise RuntimeError(...)
    ...
```

`_phase_burst_enqueue_step_tasks(self, scheduler_output)` 는 helper:

- Task A: prepare-next-batch (next iteration scheduler call) → enqueue with `MASK_ATTN | MASK_LINEAR`
- Task B: detokenize prev step → enqueue with `MASK_ATTN | MASK_LINEAR | MASK_POST`
- Task D: classify new requests → if `scheduler_output.new_request_data` 가 있으면 enqueue
- Task E: KV prefetch → 다음 step 의 attention 에서 access 될 cold chunk list 가 있으면 enqueue
- Task F: AMX draft → IDE_019 wiring 후 enqueue

---

## 3. monkey-patch vs explicit edit

| approach | pros | cons |
|---|---|---|
| **monkey-patch via plugin** (e.g., `vllm/plugins/phase_burst.py`) | vLLM main 파일 무수정 / upstream PR 불필요 / on-off toggle | function dispatch overhead (1-2 μs) / `_phase_burst_step_id` 같은 state 의 class-attr injection |
| **explicit edit of `gpu_model_runner.py`** | 0 overhead 경로 / state 관리 깔끔 | merge conflict 빈도 ↑ / upstream PR 어려움 |

**권장**: Phase 1 (dev/paper) = explicit edit (fork main 의 `_neo_*` 가 이미 fork-specific 이라 동일 정책), Phase 2 (upstream candidate) = plugin entry point 재작성.

---

## 4. step_id 관리

```python
# gpu_model_runner.py 의 instance attr
self._phase_burst_step_id: int = 0

def _phase_burst_step_id_increment(self):
    self._phase_burst_step_id += 1
    return self._phase_burst_step_id
```

`PhaseBurstRuntime.mark_phase(phase, step_id=self._phase_burst_step_id)` 의 step_id 는 atomic 갱신 → CPU pool 의 stats trace 와 vllm step counter 일치.

---

## 5. lifecycle 통합

### 5.1 모듈 import 단계

`vllm/v1/worker/gpu_model_runner.py` 상단:

```python
try:
    import phase_burst
    _phase_burst_enabled = phase_burst.is_enabled()
    if _phase_burst_enabled:
        _phase_burst_rt = phase_burst.PhaseBurstRuntime.global_instance()
    else:
        _phase_burst_rt = None
except ImportError:
    _phase_burst_enabled = False
    _phase_burst_rt = None
```

### 5.2 shutdown

`gpu_model_runner.py` 의 cleanup (worker shutdown) 또는 atexit:

```python
import atexit
atexit.register(phase_burst.PhaseBurstRuntime.shutdown_global)
```

---

## 6. CUDA event 정밀도 (별도 turn)

per-layer attention/linear boundary 는 phase signal 의 **fidelity vs invasiveness** 의 trade-off:

- **fidelity high**: model.forward 내부 각 `LlamaAttention.forward` / `LlamaMLP.forward` 진입/종료 마다 `cudaStreamAddCallback` 또는 `cudaEventRecord` + reader poll. p99 dispatch latency 50-100 μs.
- **fidelity low (Phase 1)**: `execute_model` step 단위만 → forward = attention 통합 phase. paper §4 Figure 5 의 CPU util elevate 는 sample phase 비중 (SUB_161 sampler 44%) 만으로도 baseline 4.1% → 25-30% 도달 가능.

**Phase 2 (per-layer)**: `vllm/model_executor/models/{llama,qwen2}.py` 의 layer.forward 에 cudaEvent 삽입. 본 turn 의 scope 외.

---

## 7. accuracy gate

- **OFF/ON 비교**: `_phase_burst_enabled = False/True` 의 두 run 의 per-token logprob max abs diff < 1e-3.
- `mark_phase` 자체는 atomic store + eventfd write 만 — model output 에 영향 0. 단, **task pool 의 task 가 vllm state 를 mutate 하면 발산**. 본 turn 의 stub task 들은 invocation counter 만 증가 → safe. 실제 task wiring (IDE_012 classifier 등) 시 별도 검증.

---

## 8. 적용 체크리스트 (별도 turn)

- [ ] `vllm/v1/worker/gpu_model_runner.py` 의 위 4 지점 patch (line ~3684, ~3512, ~3542, ~4084)
- [ ] `_phase_burst_step_id` instance attr 초기화 (`__init__`)
- [ ] `phase_burst` import + global runtime
- [ ] atexit shutdown
- [ ] ENV `VLLM_USE_PHASE_BURST=1` 활성화 시 sanity test (`tests/test_phase_burst_e2e.py::test_signal_round_trip`)
- [ ] baseline vs phase-burst tps + CPU util 비교 (canonical AGSD 500p)
