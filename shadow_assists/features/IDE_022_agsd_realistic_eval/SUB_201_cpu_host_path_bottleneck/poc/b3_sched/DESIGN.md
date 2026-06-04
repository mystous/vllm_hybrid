# B3 Scheduler / Launch Batching PoC — Design Doc

- **Worktree**: `/workspace/poc_worktrees/wt_b3_sched`
- **Branch**: `poc/b3_sched_260604_2208`
- **Source**: SUB_201 §5 verdict — Qwen-7B suffix launch overhead 74%
- **Scope**: 분석 + telemetry patch + CLI sweep plan (실 부팅 smoke X)

---

## 1. SUB_201 §5 evidence 요약

| Metric | Qwen-7B suffix |
|---|---|
| `cudaLaunchKernel` | 36.3% (5.26s / 60s wall) |
| `cuLaunchKernelEx` | 20.2% |
| `cudaGraphLaunch` | 17.8% |
| **합 launch overhead** | **~74%** |
| launch call rate | 18,606 calls/sec |

→ GPU idle 의 절반 이상이 launch 자체. kernel batching / graph fusion / scheduler iter 조정으로 회수 가능.

---

## 2. Scheduler step loop — launch 횟수 estimate

### 2.1 한 iter 의 forward kernel 횟수 (대략)

`Scheduler.schedule()` 한 번 호출 = 한 model forward pass 1번 launch.

`schedule()` 가 반환하는 `SchedulerOutput.total_num_scheduled_tokens` 가 그 iter 의 input token 수. Forward 는 한 번이지만, 다음의 kernel 다중 발화:

| 영역 | per-iter launch count (Qwen-7B 32 layer 기준) |
|---|---|
| Embedding | 1 |
| Per layer (32 layers): RMSNorm, QKV proj, RoPE apply, attention (FlashAttn), o-proj, post-RMSNorm, gate+up proj, SiLU mul, down proj | 32 × (~9~12 kernels) = ~300~380 |
| Final RMSNorm + lm_head | 2 |
| Sampling (greedy/top-k) | 3~5 |
| Spec decode (Eagle draft) verify path | + Eagle draft head forward (smaller layer count, 추가 ~50~100 kernels) |
| **합** | **~350~500 launch / iter** (cudagraph 없을 때) |

→ 18,606 calls/sec / 400 calls/iter ≈ **46.5 iter/sec** = ~21.5 ms/iter 의 host-bound steady state.

### 2.2 CUDA graph 활성화 시 (현재 default = `FULL_AND_PIECEWISE`)

- **PIECEWISE** = attention 만 graph 밖 (cudagraph 외부 op), 나머지 graph 내부.
  - Attention op 마다 별도 `cudaLaunchKernel` 발화 → 32 layers × ~1~2 attn kernel = ~32~64 attn launch / iter.
  - 나머지 graph 안 = 1번의 `cudaGraphLaunch` 로 fuse.
  - 결과 per iter: **1 (cudaGraphLaunch) + 32~64 (attention) + ~5 (samp) ≈ 38~70 launch / iter**.
- **FULL** = attention 도 graph 내부. per iter: **1~3 launch / iter** (decode-only batch).
- **`cudaGraphLaunch` 17.8%** = PIECEWISE 의 graph 발화. **`cuLaunchKernelEx` 20.2%** = piecewise 의 외부 attn (variable shape).

→ `cudagraph_mode=FULL` 로 가면 attn launch ~32~64 → ~0 = launch overhead **20.2% (cuLaunchKernelEx) + 17.8% / N (graph 자체 1번 발화로 줄어듦) ≈ 30~40% point 회수 가능** (이론치).

---

## 3. CLI option 후보 (launch 영향)

`vllm serve --help=CompilationConfig` + `--help=SchedulerConfig` 결과:

| Option | Default | Lever 효과 |
|---|---|---|
| `--cudagraph-mode` | `FULL_AND_PIECEWISE` (v1 default) | `FULL` 변경 시 attn 도 graph 안 → launch ↓↓↓ |
| `--max-num-batched-tokens` | model-dependent | ↑ 하면 한 iter 의 token 수 ↑ = iter 수 ↓ = launch 총수 ↓ |
| `--max-num-seqs` | 256 (default) | ↑ 시 batch size ↑, decode 의 iter 효율 ↑ |
| `--enable-chunked-prefill` | True | OFF 하면 prefill 1iter / req → host stall 증가, 그러나 mixed prefill/decode batching 위주 분석 후 결정 |
| `--enforce-eager` | False | True 면 cudagraph 비활성 → diagnostic 만 |
| `--num-speculative-tokens` (speculative.num_speculative_tokens) | 4 (AGSD 기본) | ↓ 시 verify launch ↓, 그러나 spec gain ↓ trade-off |

핵심 후보: **`cudagraph_mode=FULL`** (단일 lever).

---

## 4. `cudagraph_mode=FULL` Qwen-7B 호환성 분석

### 4.1 vllm 소스 검토

- `vllm/v1/attention/backend.py:495` `AttentionCGSupport` enum 정의:
  - `ALWAYS` — mixed-prefill-decode 도 graph OK
  - `UNIFORM_BATCH` — 모든 query_len 동일 시 graph OK (spec decode = 1+num_spec)
  - `UNIFORM_SINGLE_TOKEN_DECODE` — query_len==1 만 OK
  - `NEVER`
- `vllm/v1/attention/backends/flash_attn.py:292`:
  ```python
  _cudagraph_support = (
      AttentionCGSupport.ALWAYS
      if get_flash_attn_version() == 3
      else AttentionCGSupport.UNIFORM_BATCH
  )
  ```
  - **FA3 = `ALWAYS`** (Qwen-7B 가 H100 / SM90 → FA3 사용 → `FULL` 가능)
  - FA2 = `UNIFORM_BATCH` → mixed batch 시 `FULL` 안 됨 → `FULL_AND_PIECEWISE` fallback

### 4.2 결론

- Qwen-7B (dense, 32 layer, FA3 가능 환경 = H100 prod) → **`--cudagraph-mode FULL` 시도 가치 큼**.
- Spec decode (AGSD num_speculative_tokens=4) 와 결합 시: verify 의 batch shape = `(batch, 1+4)` = `UNIFORM_BATCH` 형 → FA3 의 ALWAYS path 에서도 graph capture 가능.
- 주의: prefill chunk 가 섞인 mixed batch 에서는 FA3 도 query_len 변동 → `FULL` 에서 dispatch miss 시 eager fallback 가능. 측정으로 확인 필요.
- Fallback path: `FULL_DECODE_ONLY` — decode-only batch 만 FULL, mixed 는 eager. Qwen-7B suffix workload 가 decode dominant 면 효과적.

---

## 5. Minimal patch — telemetry only

실 launch fusion 은 worker level (`gpu_model_runner.py` + cudagraph dispatch) 까지 들어가야 함. PoC scope (~2-3h) 로는 **scheduler 의 iter-level metric 만 측정 가능하게 hook** 추가.

### 5.1 적용 patch (이미 적용됨)

파일: `vllm/v1/core/sched/async_scheduler.py`

추가:
- env flag `VLLM_SCHED_LAUNCH_TELEMETRY=1` 활성 시 per-iter counter:
  - `_b3_iter_count` — 총 iter 수
  - `_b3_total_scheduled_tokens` — total token 누적
  - `_b3_total_scheduled_reqs` — req 수 누적
  - `_b3_total_spec_reqs` — spec decode 활성 req 수
- env `VLLM_SCHED_LAUNCH_TELEMETRY_LOG_EVERY=100` (default) iter 당 1번 `logger.info` dump
- counter 코드는 `_LAUNCH_TELEMETRY_ENABLED` False 시 attribute lookup 한 번만 → hot-path overhead 무시할 수준 (~ns)

### 5.2 Risk

- **None (default OFF)** — env 안 켜면 기존 behavior 동일.
- 켰을 때 logger.info 가 매 100 iter ≈ 2 sec 마다 1줄 → 무시 가능.
- 다른 scheduler 영역 미변경 → invariant 보존.

### 5.3 추가 patch (production-ready, 본 PoC scope 밖)

진짜 launch fusion 은 다음 영역 변경 필요 (~수일 dev):

1. **CUDA graph capture size 확장** (`CompilationConfig.cudagraph_capture_sizes`)
   - decode-only batch 의 모든 (batch_size, query_len) 조합 capture
   - Spec decode 의 `1+num_spec_tokens` query_len 추가 capture
2. **Mixed prefill-decode batch 의 FULL graph fallback 결정 로직** (`cudagraph_dispatcher.py`)
3. **Eagle draft verify 의 cudagraph 통합** (`vllm/v1/worker/gpu/spec_decode/eagle/cudagraph.py`)

---

## 6. CLI option sweep plan (실 부팅은 다른 turn)

### 6.1 Sweep matrix (Qwen-7B suffix, H100 prod)

| Run | `cudagraph_mode` | `max-num-batched-tokens` | num_spec_tokens | 예상 launch rate (calls/sec) | 예상 launch overhead (% of wall) |
|---|---|---|---|---|---|
| R0 (baseline) | `FULL_AND_PIECEWISE` | 8192 | 4 | 18,606 (측정값) | 74% |
| R1 | `FULL` | 8192 | 4 | ~3,000 (이론 ~5x ↓) | ~25~35% |
| R2 | `FULL_DECODE_ONLY` | 8192 | 4 | ~6,000~10,000 | ~40~55% |
| R3 | `FULL` | 16384 | 4 | ~1,500 (iter ↓ 2x) | ~15~25% |
| R4 | `FULL` | 8192 | 2 | ~2,500 (verify token ↓) | ~20~30% |

### 6.2 측정 방법

각 run 60s window, nsys 으로 launch rate 측정 + `VLLM_SCHED_LAUNCH_TELEMETRY=1` 로 scheduler iter rate 동시 측정:

```bash
VLLM_SCHED_LAUNCH_TELEMETRY=1 \
VLLM_SCHED_LAUNCH_TELEMETRY_LOG_EVERY=100 \
nsys profile --wait=all --duration=60 \
  -o qwen7b_suffix_R1_full.nsys \
  vllm serve Qwen/Qwen2.5-7B \
  --cudagraph-mode FULL \
  --speculative-config '{"method":"eagle","num_speculative_tokens":4,...}'
```

### 6.3 Net-win 판정

- **PASS 조건**: launch overhead % wall < 50% AND throughput (tok/s) ≥ baseline × 1.10
- **FAIL 조건**: FA3 dispatch miss / KV invariant 위반 / 정확도 (TST_003 분포 유사성) 회귀

---

## 7. 발견한 장애물 + 다음 dev step

### 7.1 장애물

1. **vllm runtime venv = main repo link** → worktree patch 가 venv 에 반영 안 됨 → 본 worktree 안에서 부팅 smoke 불가. 다른 turn 에서 main repo 로 patch cherry-pick 후 부팅 필요.
2. **`cudagraph_mode=FULL`** 자체는 코드 patch 없음 = CLI flag 만 → 가장 simple lever. 단, **prod 머신 (H100, FA3) 에서만 검증 가능** (RTX 3090 dev 머신은 FA3 unsupported, FA2 fallback → `UNIFORM_BATCH` → `FULL_AND_PIECEWISE` 강제 다운그레이드).
3. Spec decode + `FULL` graph 의 capture size matrix 가 폭증 (`max_num_seqs × (1+num_spec_tokens)` 조합) → 메모리 ↑ + warmup 시간 ↑.

### 7.2 다음 dev step (production-ready scheduler patch 복잡도)

| Step | 작업 | 추정 |
|---|---|---|
| 1 | Worktree patch → main repo cherry-pick + venv 재link | 30분 |
| 2 | Qwen-7B suffix `--cudagraph-mode FULL` 부팅 smoke (H100 prod) | 1h |
| 3 | nsys 60s sweep R1~R4 + scheduler telemetry 동시 측정 | 4~6h |
| 4 | Net-win 판정 + TST_003 정확도 회귀 검증 | 2~3h |
| 5 | (positive 시) `FULL` 을 AGSD default 로 promote + AGENTS.md 갱신 | 1h |
| 6 | (negative 시) cudagraph_dispatcher.py 의 mixed batch fallback 정교화 patch (~ 수일) | 2~3d |
| **합** | | **~1~2 days for CLI sweep**, ~ **수일** for production-grade patch |
