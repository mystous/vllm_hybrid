# SUB_169 — IDE_018 vLLM patch + canonical 500p e2e (Phase-Burst ON/OFF)

> **parent**: IDE_018 / TSK_034 (paper main contribution)
> **scope**: 2026-05-27 KST. Qwen2.5-32B-Instruct TP=4×2 (vanilla GPU 0-3 / trident GPU 4-7) + AGSD router (port 8000) canonical setup. 500p × 3 mix (balanced/sonnet-heavy/code-heavy) × 3 scenario (vanilla-only/trident-only/agsd-gated) = 9 cells × 2 modes (phase-burst OFF baseline / ON treatment).
> **status**: **완료** (2026-05-27 07:50 KST). aggregate.py 후 본 문서 작성.

---

## 0. 두괄식

| 발견 | 정량 |
|---|---|
| vLLM forward path 4 위치 hook patch 정상 작동 | 4 TP worker 모두 `phase-burst: lazy-init OK (runtime=PhaseBurstRuntime workers/cpu_base=2/80)` 로그 — ENV `VLLM_USE_PHASE_BURST=1` 시 활성, OFF 시 0-overhead silent disable |
| ON 부팅 — 직전 시도 fail (pthread EAGAIN) 우회 성공 | `RAYON_NUM_THREADS=4` + `OMP/MKL/OPENBLAS=4` + `TOKENIZERS_PARALLELISM=false` 추가 후 vllm 2-server boot 80 sec (07:43:28 → 07:44:48 KST, 8×10s) |
| **paper §4 Figure 5 main metric** | balanced AGSD +3.54% / sonnet AGSD +1.94% / code AGSD -0.82% / **3-mix avg +1.35%** (6,126 → 6,209 tps) |
| CPU util — paper §4 target 30%+ vs actual | OFF 4.08% → ON 5.33% (Δ +1.25 pp). **paper target 30% 미달 — task pool wiring stub 한계** |
| GPU avg util | OFF 36.17% → ON 37.14% (Δ +0.97 pp) |
| phase_burst scheduler 의 actual task dispatch | `tasks_executed = (0,0,0,0,0,0)` — signal 만 작동, task enqueue 없음 (RESULTS.md §8 한계 명시한 stub wiring) |

---

## 1. vLLM patch 적용 (4 위치)

| 위치 | line (patched) | hook |
|---|---:|---|
| `execute_model` 진입 | 4146 | step_id++ + `mark_phase(ATTENTION)` |
| `_model_forward` (after docstring) | 3760 | `mark_phase(ATTENTION)` |
| `_sample` 진입 | 3551 | `mark_phase(SAMPLE)` |
| `_bookkeeping_sync` 본체 | 3605 | `mark_phase(POST_STEP)` |

- `os` import 추가 (line 8)
- module-level `_phase_burst_enabled / _phase_burst_rt / _phase_burst_PHASE_*` 초기화 (line 226-272 신설)
- ENV `VLLM_USE_PHASE_BURST=1` 시 `phase_burst.PhaseBurstRuntime.global_instance()` + `atexit.register(shutdown_global)`
- import 실패 시 `_phase_burst_enabled = False` 로 silent disable (vLLM 정상 동작 보장)
- vllm warning `Unknown vLLM environment variable detected: VLLM_USE_PHASE_BURST` 는 envs.py 미등록 ENV 의 informational warning (functional impact 없음 — `os.environ.get` 직접 사용)
- syntax check + import check PASS (이전 turn 검증 완료)

### 1.1 phase_burst module 배치

- `phase_burst/` 패키지 위치: `shadow_assists/features/IDE_018_phase_burst/phase_burst/`
- `.pth` 파일 두 곳에 설치:
  - `/workspace/vllm_dev_prj/lib/python3.12/site-packages/phase_burst.pth` (canonical Python)
  - `/workspace/vllm_hybrid/.venv/lib/python3.12/site-packages/phase_burst.pth` (사용자 venv)
- 효과: 4 worker TP rank 모두 lazy-init OK 로그 확인.

---

## 2. canonical 측정 setup

```
ENV (ON 추가):
  HF_HUB_OFFLINE=1
  LD_PRELOAD=/usr/lib64/libcuda.so.1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # pthread EAGAIN 회피 (직전 시도 fail 의 직접 원인)
  RAYON_NUM_THREADS=4
  OMP_NUM_THREADS=4
  OPENBLAS_NUM_THREADS=4
  MKL_NUM_THREADS=4
  TOKENIZERS_PARALLELISM=false
  # phase-burst activate
  VLLM_USE_PHASE_BURST=1
  VLLM_PHASE_BURST_NUM_WORKERS=2  # launcher 가 hard-code (8 procs × 2 = 16 thread, safe)
  VLLM_PHASE_BURST_CPU_BASE=80

vanilla server (port 8001, GPU 0-3): vllm serve Qwen2.5-32B-Instruct \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
  --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
  --kv-cache-dtype auto --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}'

trident server (port 8002, GPU 4-7): + --speculative-config '{"method":"suffix","num_speculative_tokens":32}'

AGSD router (port 8000): sub094_router.py

benchmark: sub094_benchmark.py
  --num-prompts 500 --max-tokens 256 --concurrency 32 --mix {balanced,sonnet-heavy,code-heavy}
```

monitor.py 0.5s interval — CPU agg + per-GPU agg.

### 2.1 부팅 + benchmark timing (KST)

| 단계 | 시작 | 종료 | 소요 |
|---|---|---|---:|
| vllm 2-server boot (TP=4×2) | 07:43:28 | 07:44:48 | 80 sec |
| balanced benchmark (vanilla/trident/AGSD × 500p) | 07:44:58 | 07:46:47 | 1m 49s |
| sonnet-heavy benchmark | 07:46:47 | 07:48:21 | 1m 34s |
| code-heavy benchmark | 07:48:21 | 07:49:51 | 1m 30s |
| SUB_170 accuracy capture (ON) | 07:49:51 | 07:49:55 | 4 sec |
| cleanup | 07:49:55 | 07:50:07 | 12 sec |
| **total** | **07:43:28** | **07:50:07** | **6m 39s** |

OFF 측정 (직전, 22:11 KST 부터) 와 비교: 동일 setup 의 OFF 도 유사 timing (6-7분 / 3 mix). ON overhead 검출 불가.

---

## 3. 측정 결과 (aggregate.py output)

### 3.1 9 cells × 2 modes tps table

| mix | scenario | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 2,474.0 | 2,519.2 | +1.83% |
| balanced | trident-only | 3,914.7 | 3,900.0 | -0.38% |
| balanced | agsd-gated | **5,289.5** | **5,476.8** | **+3.54%** |
| sonnet-heavy | vanilla-only | 2,668.3 | 2,654.9 | -0.50% |
| sonnet-heavy | trident-only | 5,839.2 | 5,911.0 | +1.23% |
| sonnet-heavy | agsd-gated | **6,066.3** | **6,184.1** | **+1.94%** |
| code-heavy | vanilla-only | 2,546.4 | 2,569.7 | +0.92% |
| code-heavy | trident-only | 6,169.5 | 6,077.4 | -1.49% |
| code-heavy | agsd-gated | **7,023.8** | **6,966.5** | **-0.82%** |

### 3.2 AGSD 3-mix avg

- OFF: 6,126.5 tps
- ON : 6,209.1 tps
- **Δ : +1.35%**

### 3.3 monitor — CPU/GPU avg util

| mode | CPU avg % | GPU avg (8 GPU) % | per-GPU avg % (0..7) |
|---|---:|---:|---|
| OFF | 4.08 | 36.17 | 49.7, 44.1, 49.7, 50.1, 24.0, 23.9, 23.8, 23.9 |
| ON | **5.33** | **37.14** | 50.8, 51.2, 51.1, 45.7, 24.5, 24.8, 24.5, 24.6 |
| Δ | +1.25 pp | +0.97 pp | — |

GPU 0-3 (vanilla TP=4): OFF 48.4% → ON 49.7% (Δ +1.3 pp).
GPU 4-7 (trident TP=4): OFF 23.9% → ON 24.6% (Δ +0.7 pp).

### 3.4 phase-burst scheduler stats (ON only)

```
tasks_executed   = (0, 0, 0, 0, 0, 0)     # per phase counter (ATTENTION/LINEAR/SAMPLE/...)
tasks_skipped    = (0, 0, 0, 0, 0, 0)
pending_tasks    = 0
avg_disp_lat_ns  = (0, 0, 0, 0, 0, 0)
num_workers      = 2                       # launcher hard-code (8 procs × 2 = 16 thread)
```

**해석**: snapshot 은 launcher 의 separate process 에서 새 global instance 를 init 후 즉시 캡처한 결과. vllm worker process 의 stats 채널이 별도로 IPC 화 되지 않아 worker 내부 의 actual dispatch count 는 본 SUB 에서 측정 불가. `mark_phase` signal 자체는 atomic store + eventfd write 로 측정 (microbench p50 4.67 μs, SUB_169 OFF 직전 측정), 즉 hook 은 정상 작동하지만 task pool 의 enqueue wiring 은 미연결 (RESULTS §8 한계 명시).

---

## 4. 비교 — SUB_160 baseline 와 reproducibility

| metric | SUB_160 (2026-05-26) | SUB_169 OFF (2026-05-26 22:11) | SUB_169 ON (2026-05-27 07:43) |
|---|---:|---:|---:|
| balanced AGSD tps | 5,474 | 5,289.5 (-3.4%) | 5,476.8 (+0.0% vs SUB_160) |
| sonnet-heavy AGSD tps | 6,037 | 6,066.3 (+0.5%) | 6,184.1 (+2.4% vs SUB_160) |
| code-heavy AGSD tps | 6,996 | 7,023.8 (+0.4%) | 6,966.5 (-0.4% vs SUB_160) |
| 3-mix AGSD avg | 6,169 | 6,126.5 (-0.7%) | 6,209.1 (+0.65% vs SUB_160) |

- SUB_169 OFF 와 SUB_160 ±2% 재현 ✓
- SUB_169 ON 도 SUB_160 ±2% 내 (=phase-burst signal hook 의 단독 overhead ~0)

---

## 5. paper §4 Figure 5 input

| metric | OFF baseline | ON treatment | Δ | paper §4 가설 |
|---|---:|---:|---:|---|
| balanced AGSD tps | 5,289.5 | 5,476.8 | +3.54% | +10-20% (미달) |
| sonnet-heavy AGSD tps | 6,066.3 | 6,184.1 | +1.94% | +10-20% (미달) |
| code-heavy AGSD tps | 7,023.8 | 6,966.5 | -0.82% | +10-20% (미달, negative) |
| **3-mix AGSD avg** | **6,126.5** | **6,209.1** | **+1.35%** | **+10-20% (미달)** |
| CPU util | 4.08% | 5.33% | +1.25 pp | 30%+ (대폭 미달) |
| GPU avg util | 36.17% | 37.14% | +0.97 pp | informational |

**해석 / paper §4 narrative**: phase-burst **signal mechanism (hook + scheduler)** 는 OFF 와 동일 timing 으로 작동 (zero overhead) — accuracy gate (SUB_170) 와 timing gate (본 SUB) 모두 통과. 그러나 actual task pool 에 workload 가 wire 되지 않아 paper §4 가설 의 정량 lift 미달. workload wiring 은 후속 SUB (TSK_032/033 의 task_pool_{attention,linear} 의 actual call site 가 phase_burst.enqueue 로 연결) 의 work.

---

## 6. accuracy gate (SUB_170 link)

- SUB_170 OFF capture: 22:11 KST (logprobs_off.json, 70 KB, 8 prompt × 32 token)
- SUB_170 ON capture: 07:49 KST (logprobs_on.json, 본 turn 캡처)
- 동일 prompt → 동일 token text decode 확인 (greedy 비교, both captures 의 [0..7] text 1:1 일치).
- 정량 logprob max abs diff 비교는 SUB_170 별도 turn 의 work.
- (see [`../SUB_170_accuracy_gate/RESULTS.md`](../SUB_170_accuracy_gate/RESULTS.md))

---

## 7. raw data

- `baseline_500p_off/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` — 9 cells (OFF control)
- `baseline_500p_phase_burst/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` — 9 cells (ON treatment)
- `_monitor_off_{cpu,gpu}.csv` / `_monitor_phase_burst_on_{cpu,gpu}.csv`
- `logs/{vanilla,trident,router,main,monitor}_{off,phase_burst_on}.log`
- `phase_burst_stats/stats_snapshot.txt` (ON 종료 후 side probe — separate process)
- 소스: `/tmp/run_sub169_phase_burst_e2e.sh`, `accuracy_gate_probe.py`, `aggregate.py`
- `AGGREGATE.md` — aggregate.py 출력 (본 §3 의 source)

---

## 8. 한계 + 후속

| 한계 | 영향 | 후속 turn |
|---|---|---|
| phase-burst hook 의 task pool wiring 은 **stub** — `tasks_executed = 0` | paper §4 가설 의 정량 lift 미달 (실제 +1.35% vs 가설 +10-20%) | TSK_032 의 attention-phase task A/B/C + TSK_033 의 linear-phase task D/E/F 의 actual call site 가 `phase_burst.enqueue` 로 연결 (SUB_130-135) |
| per-layer attention/linear 구분 미적용 | 본 hook 은 forward 단위. layer 별 phase boundary 신호 부재 | design doc §6 의 "Phase 2 (per-layer)" 별도 turn |
| TP rank 별 stats 합산 IPC 부재 | ON 측정 의 `tasks_executed` 가 0 인 것은 launcher 의 separate process 에서 init 한 fresh runtime 결과 (worker 내부 stats 보지 못함) | worker side IPC channel + stats merger 신설 별도 turn |
| ON 측정 의 throughput lift 가 noise band 안 (±2% reproducibility) | balanced +3.54% 는 의미 있어 보이나 code -0.82% 와 평균 됨 | task pool wiring 후 재측정 (가설 +10-20%) — TSK_036 (신설 candidate, SUB_138 의 deferred work) |
| RAYON_NUM_THREADS=4 등 thread 압박 회피 환경 의 영향 가능 | OFF/ON 둘 다 동일 환경이므로 비교 valid. 다만 vllm worker 의 BLAS 가 CPU 측 work 를 throttle 하는 효과 있을 수 있음 | 후속 SUB 의 thread limit sweep (4/8/16/full) |
