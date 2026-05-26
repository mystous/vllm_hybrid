# SUB_161 — py-spy sublayer profile (nsys 부재 alternative, 1-run)

> **parent**: IDE_015 / TSK_022 재정의 (nsys 없으므로 py-spy 만 사용)
> **scope**: 2026-05-26 23:20 ~ 23:24 KST (~4 min total — vllm boot 90s + 500p balanced 1-run + py-spy 40s record)
> **status**: ✅ 완료 — py-spy flamegraph (vanilla + trident TP0) + benchmark
> **tooling**: py-spy 0.4.2 (native + threads + idle). ncu 는 본 SUB 에서 skip (vllm 의 Python multiprocess 에 직접 attach 어려움)

---

## 0. 두괄식 — trident TP0 CPU 시간의 44% 가 sampling 단계 ⭐

### Top finding

**trident TP0 worker** 의 40s py-spy record 결과 — 파일별 CPU 시간 share:

| File | sample share | 의미 |
|---|---:|---|
| `vllm/v1/worker/gpu_model_runner.py` | **79.5%** | model dispatcher (전체 forward path) |
| `vllm/v1/executor/multiproc_executor.py` | 57.6% | IPC overhead |
| `torch/nn/modules/module.py` | 51.6% | torch nn.Module forward |
| **`vllm/v1/sample/sampler.py`** | **44.3%** ⭐ | sampling (greedy/top-k/top-p) |
| `vllm/model_executor/models/qwen2.py` | 27.7% | model layer (attention/linear) |
| **`vllm/model_executor/layers/logits_processor.py`** | **27.0%** | logits processing (pre-sampling) |
| **`vllm/v1/sample/ops/penalties.py`** | **23.4%** | sampling penalty ops |
| `vllm/distributed/communication_op.py` | 19.0% | TP collective ops |
| `vllm/v1/worker/gpu_input_batch.py` | 17.6% | batch metadata |

→ **CPU side total of sampler + logits + penalties ≈ 44% + 27% + 23% = ~94% (cumulative)** during the workload window — sampling 단계가 가장 큰 CPU 사용 phase.

### IDE_016 / IDE_018 직접 입력

| 의미 | 영향 IDE |
|---|---|
| sampler.py 44% — top-k/top-p + greedy choice 가 CPU-bound | **TSK_025 (AVX-512 sampling)** 의 가장 큰 lift 예상 영역 |
| logits_processor 27% — bias/temperature/penalty 적용 | TSK_025 의 logit processor vectorize 의 input |
| penalties 23% — repetition/presence penalty | TSK_025 가 IDE_018 의 phase-burst attention task 와 정합 |
| gpu_model_runner 79.5% (전체 dispatch chain) | model.forward 전체 — phase-burst 의 dispatch hook 위치 |
| communication_op 19% — TP collective | NCCL all-reduce 등 — phase boundary 의 marker 후보 |

---

## 1. Benchmark result (1-run 500p balanced)

| scenario | tps | wall (s) | p50 (s) | p99 (s) |
|---|---:|---:|---:|---:|
| vanilla-only | 2,267.9 | 54.9 | 3.53 | 3.74 |
| trident-only | 3,907.4 | 31.8 | 1.48 | 4.87 |
| **AGSD-gated** | **5,442.2** | 22.9 | 0.74 | 3.03 |

→ AGSD 5,442 — SUB_160 의 30-run mean (5,457) 와 0.3% 차이. 정합.

---

## 2. py-spy flamegraph data (trident TP0, n=415 samples × 100 Hz × 40s)

### 2.1 Function-level hot spots (full 415-sample roll-up)

flamegraph SVG: [`raw/pyspy_record_3904735.svg`](raw/pyspy_record_3904735.svg)

| Function | samples | share | category |
|---|---:|---:|---|
| (root: dispatch chain) | 415 | 100% | — |
| `_sample` (gpu_model_runner:3521) | 116 | 27.9% | sampling |
| `update_async_output_token_ids:1013` (gpu_input_batch) | 73 | 17.6% | output post-process |
| `worker_busy_loop` (multiproc_executor:963) | 56 | 13.5% | IPC wait |
| `dequeue` (shm_broadcast:766) | 30 | 7.2% | IPC dequeue |
| `dequeue` (shm_broadcast:755) | 19 | 4.6% | IPC dequeue |
| `update_async_output_token_ids:1024` | 26 | 6.3% | output post-process |
| `__enter__` (contextlib:137) | 10 | 2.4% | torch grad context |

### 2.2 vanilla TP0 — idle 비율 압도적

`raw/pyspy_record_3904726.svg` — 분석 결과 vanilla TP0 의 py-spy 가 의미있는 sample 캡처 못함 (worker 가 대부분 idle 상태).

→ AGSD 라우터 분배 (SUB_098 데이터 기준: balanced 66 vanilla + 134 trident) 으로 vanilla 가 trident 의 절반 미만 workload → idle 비율 큼.
→ **paper 입력**: AGSD 의 imbalance 영역도 IDE_018 phase-burst 의 입력 (vanilla 의 idle window 에 CPU 작업 fill 가능).

---

## 3. Idle vs Active state — py-spy dump 비교

### 3.1 vanilla TP0 — pre-workload (idle) + post-workload (idle 동일)

```
Thread 3904726 (idle): "MainThread"
    poll (zmq/sugar/poll.py:106)
    wait (vllm/distributed/device_communicators/shm_broadcast.py:186)
    acquire_read (vllm/distributed/device_communicators/shm_broadcast.py:674)
    dequeue (vllm/distributed/device_communicators/shm_broadcast.py:755)
    worker_busy_loop (vllm/v1/executor/multiproc_executor.py:963)
```

→ vanilla TP0 의 MainThread 는 **항상 shm_broadcast 의 acquire_read 에 blocked** — IPC 신호 대기.

### 3.2 EngineCore — engine input queue 에 blocked

```
Thread 3903451 (idle): "MainThread"
    wait (threading.py:355)
    get (queue.py:171)
    _process_input_queue (vllm/v1/engine/core.py:1480)
    run_busy_loop (vllm/v1/engine/core.py:1458)
```

→ EngineCore 는 input queue 에서 next batch 대기 — 이 wait window 가 **CPU 가 사용 가능한 idle gap** (IDE_018 의 attention-phase CPU task pool 의 입력 후보).

---

## 4. 핵심 finding (IDE_016/018 입력)

| finding | 의미 | 영향 IDE |
|---|---|---|
| **trident TP0 CPU 44.3% 가 sampling 단계** | TSK_025 AVX-512 sampling 의 가장 큰 lift 잠재력 — sampling 만 vectorize 해도 latency 감소 | IDE_016 / TSK_025 |
| logits_processor + penalties 합 50.4% | sampling chain (logit transform → sampling) 전체가 CPU-bound | IDE_016 / TSK_025 |
| communication_op 19% | TP collective (NCCL all-reduce) — phase boundary marker 후보 | IDE_018 / TSK_031 |
| update_async_output_token_ids 17.6% | output post-processing (tokenizer 호출 영역) | IDE_016 / TSK_024 |
| IPC dequeue + wait ~25% | worker idle wait — phase-burst CPU task pool 의 capacity | IDE_018 / TSK_032 |
| vanilla TP0 거의 idle (AGSD 분배 비대칭) | AGSD imbalance — vanilla 의 idle window 도 CPU 작업 fill 가능 | IDE_018 |

---

## 5. nsys 부재의 영향 + 한계

| 잃은 정보 | 대안 영역 |
|---|---|
| **GPU kernel timeline** (kernel 별 wall start/end) | py-spy 의 Python stack 만 캡처 — GPU timing 직접 측정 불가 |
| **attention vs linear phase boundary** (sublayer 별 GPU 시간) | py-spy 의 model_runner.py 시간 share 로 간접 추정만 가능 |
| **kernel-by-kernel detail** (e.g. flash attention 시간) | ncu 단독 캡처 가능하나 본 SUB 에서 skip (vllm multiproc 에 ncu attach 복잡) |

→ TSK_022 의 "GPU 20pp idle window phase 별 categorize" 는 nsys 없이 **완전한 정량 어려움**. 본 SUB 의 py-spy 결과로 CPU 측 phase distribution 은 확보 → IDE_018 의 한쪽 정량 입증.
→ paper 시 nsys 가능 머신 / 별도 환경에서 후속 측정 필요.

---

## 6. 다음 step

- **SUB_162** (queued — 1-run): /proc + py-spy CPU thread state sampling (perf 부재 alternative)
- (옵션) ncu kernel-by-kernel — vllm subprocess 에 직접 attach 가능여부 검증
- **TSK_025 AVX-512 sampling** (kernel dev — 별도 turn) — 본 SUB 의 44.3% sampler.py finding 이 lever
- **TSK_032 attention-phase CPU task pool** (IDE_018) — 본 SUB 의 IPC idle wait + sampler 정량을 입력

---

## 7. raw data

- `benchmark_balanced.json` — 500p × 3 scenario 측정값
- `raw/pyspy_record_3904735.svg` — trident TP0 flamegraph (284 KB, 40s × 100 Hz, 415 samples)
- `raw/pyspy_record_3904726.svg` — vanilla TP0 flamegraph (대부분 idle)
- `raw/pyspy_dump_pre_*.txt` / `pyspy_dump_post_*.txt` — 4 process snapshot (pre/post workload)
- `_monitor_{cpu,gpu}.csv` — full duration util
- `logs/{vanilla,trident,router,bench,main,monitor}.log`
- 소스: `/tmp/run_sub161_ncu_pyspy_profile.sh`
