# D-1 Async Pipelining — Diagnosis

## 결론 (사전 분석)

**D-1 의 본질 (prev step host op 와 next step GPU execute overlap) 은
vLLM 의 baseline (vanilla) 에서 이미 native 로 active.**

### 근거

1. **AsyncScheduler 자동 활성**
   - `vllm/config/vllm.py:789-832` — `scheduler_config.async_scheduling` 의
     default 가 `None` 이면, spec-decode 없고 pooling 아니면 자동으로 `True`.
   - 그래서 본 LHC baseline (vanilla, spec-decode off) 은 `True`.
   - log 확인: `INFO 06-09 06:40:18 [vllm.py:834] Asynchronous scheduling is enabled.`

2. **batch_queue_size=2 자동 활성**
   - `vllm/v1/executor/multiproc_executor.py:476-487` — TP-only (pp=1) +
     `async_scheduling=True` 면 자동으로 `max_concurrent_batches=2`.
   - `vllm/v1/engine/core.py:240-271` — `batch_queue_size > 1` 이면 `step_with_batch_queue`
     사용 → step N 의 host post-process 가 step N+1 의 GPU execute 와 overlap.
   - log 확인: `[NEO STEP DIAG] step=2 num_scheduled=847 batch_queue=1/2 running=16`

3. **GPU saturation 측정값**
   - baseline `mix_s1.json`: `gpu_util=99.6%`, `cpu_util=5.8%`.
   - GPU 가 거의 100% saturated → host critical path 가 GPU step 보다 짧음.
   - 이미 vLLM-native pipelining 이 GPU bubble 을 거의 제거함.

### 의미

D-1 의 원래 plan (`VLLM_LHC_ASYNC_PIPELINE=1` 으로 step end host op 을
background thread 로 분리) 은 vLLM-native 와 **중복**. 추가 양수 효과
극히 작거나 0.

### 잔여 lever 시도

baseline 과 중복되지 않는 host-overhead 감소 lever:

- **L1 — `--stream-interval 16`**: SSE event coalesce. host IPC 16 배 감소.
  - 잠재: streaming 시 vLLM-output_processor 의 enqueue/IPC 빈도 → throughput.
  - 단점: TTFT 영향 없음 (첫 token 발신 동일), TPOT 표시 변동 가능.

### 측정 결과 (s1 only, 2026-06-09)

`runs/L1_stream16/{corpus}_s1.json` vs `vanilla_runs/{corpus}_s1.json`.

| corpus    | baseline (tps) | L1_stream16 (tps) | Δtps    | Δ%      |
|-----------|---------------:|------------------:|--------:|--------:|
| sharegpt  | 3157.5         | 3166.4            | +8.9    | +0.28%  |
| swebench  | 3018.8         | 2893.2            | -125.6  | -4.16%  |
| humaneval | 3421.8         | 3428.3            | +6.5    | +0.19%  |
| mbpp      | 1900.2         | 1888.3            | -11.9   | -0.63%  |
| wildchat  | 3216.8         | 3220.4            | +3.6    | +0.11%  |
| lmsys     | 3089.5         | 3127.8            | +38.3   | +1.24%  |
| mix       | 3219.1         | 3209.0            | -10.1   | -0.31%  |

**Summary: n=7, mean Δ%=-0.47, std=1.60, wins(≥+5%)=0/7. STOP 조건 미달성.**

### 결론

1. baseline `vanilla` 가 이미 `async_scheduling=True` + `batch_queue_size=2`
   (vLLM-native) → D-1 의 본질 (prev step host op vs next step GPU execute
   overlap) 이 이미 active. serve log `[NEO STEP DIAG] batch_queue=1/2` 로
   확인.
2. baseline `gpu_util=99.6%`, `cpu_util=5.8%` → GPU saturate, host critical
   path 짧음. 추가 host-overhead 감소 lever (stream_interval=16) 효과 ~0.
3. swebench -4.16% 는 baseline 의 s1↔s2 variance (-7.36%) 와 같은 noise
   floor 안. 단일 sweep paired 의 분산이 ±5% 정도.
4. **D-1 algorithm 자체는 양수 lever 가 아님**: vLLM-native baseline 과
   기능 중복으로 추가 효과 측정 안 됨. plan 의 STOP 조건 (3/7 ≥+5%) 도달
   불가.

### Revert

본 PoC 는 코드 변경 없음 (`--stream-interval` 은 vLLM CLI flag). 추가
revert 필요 없음. PoC 산출물 (`runs/`, `logs/`, `DIAGNOSIS.md`,
`compare.py`, `run_d1.sh`) 는 기록 목적으로 유지.
