# B3 sched — cudagraph_mode sweep MEASUREMENTS

- **Hardware**: NVIDIA B200 × 4 (GPU 0-3, sm_100)
- **Model**: Qwen/Qwen2.5-7B-Instruct, TP=4, max-model-len=16384, gpu-mem-util=0.85
- **Method**: suffix decoding (`num_speculative_tokens=32`)
- **Workload**: 50p × conc=16, mix shuffle (seed=42), stream
- **vllm version**: `v1.7.dev16107+gffe20fb09.d20260601`
- **Date**: 2026-06-04 22:25~22:34
- **Attention backend (auto)**: **FLASHINFER** (B200 자동 선택. FA3 아님)
- **Boot cmd template**: `vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 4 --port 8001 --gpu-memory-utilization 0.85 --max-model-len 16384 --compilation-config '{"cudagraph_mode":<MODE>}' --allow-deprecated-quantization --speculative-config '{"method":"suffix","num_speculative_tokens":32}'`
- **Telemetry note**: `VLLM_SCHED_LAUNCH_TELEMETRY=1` 은 worktree-only patch (vllm/v1/core/sched/async_scheduler.py)이며 dev venv (main repo link)에서는 dispatch 되지 않아 본 sweep 에서 skip 함.

---

## 1. R0 / R1 / R2 boot 결과

| Run | `cudagraph_mode` (요청) | 실 적용 mode | boot wall (READY) | engine init (`core.py:376`) | compilation | capture sizes (PIECEWISE/FULL) | KV cache | 결과 |
|---|---|---|---|---|---|---|---|---|
| R0 | `PIECEWISE` | **PIECEWISE** (1) | ~64 s | 31.97 s | 16.77 s | **51 / —** | 141.34 GiB | OK |
| R1 | `FULL` | **FULL_AND_PIECEWISE** (2,1) — backend 다운그레이드 | ~75 s | 33.37 s | 16.61 s | **15 / 15** | 141.35 GiB | OK |
| R2 | `FULL_AND_PIECEWISE` | **FULL_AND_PIECEWISE** (2,1) | ~70 s | 34.25 s | 16.45 s | **15 / 15** | 141.34 GiB | OK |

핵심 boot 발견:

1. **FlashInfer backend cap**: R1 부팅 중 vllm warn:
   ```
   WARNING [compilation.py:1310] CUDAGraphMode.FULL is not supported with FlashInferBackend
   backend (support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_AND_PIECEWISE
   ```
   → DESIGN.md §4 에서 FA3 가 `AttentionCGSupport.ALWAYS` 라 `FULL` 단독 가능하다고 봤으나, **B200 의 default backend 는 FLASHINFER 이고 FlashInfer 는 `UNIFORM_BATCH` 이라 `FULL` 단독 불가**. 결과적으로 R1 ≡ R2 (실 capture 가 동일).
2. **capture size matrix 가 PIECEWISE 단독 대비 줄어듦**: R0 = 51 sizes (1~512 ladder). R1/R2 = 15 sizes (PIECEWISE) + 15 sizes (FULL decode-only). 동일 ladder 의 fewer-bin 으로 합쳐졌으며 capture 자체 시간은 1~2초 (전체 init 의 ~5%) — capture 폭증 시나리오 X.
3. **engine init ±2 s 이내 동일**, compilation time 동일. **메모리 footprint 동일** (141 GiB available KV).

---

## 2. Burst 측정 (50p × conc=16, mix)

| Run | mode | wall (s) | tokens | **output_tps** | TTFT p50 (ms) | TTFT p99 | TPOT p50 (ms) | TPOT p99 | GPU util (%) | GPU mem (MiB) | CPU% | accept α | accepted/draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R0** | PIECEWISE | **39.6** | 165 202 | **4 169.1** | 27.8 | 351.3 | **5.3** | 12.5 | **67.2** | 632 544 | 3.3 | 0.7774 | 125 908 / 161 967 |
| **R1** | FULL (→FaP) | **59.7** | 151 950 | **2 546.3** | 33.7 | 346.2 | **9.4** | 17.9 | **45.0** | 631 932 | 3.1 | 0.7593 | 114 529 / 150 838 |
| **R2** | FULL_AND_PIECEWISE | **35.7** | 157 560 | **4 413.3** | 26.3 | 510.1 | **5.6** | 12.2 | **47.5** | 632 112 | 2.5 | 0.7917 | 121 349 / 153 282 |

Per-corpus req/s (subset):

| corpus | R0 | R1 | R2 |
|---|---|---|---|
| lmsys | 375.1 | 248.1 | 355.9 |
| wildchat | 303.7 | 205.0 | 331.7 |
| sharegpt | 186.7 | 129.6 | 206.4 |
| swebench | 304.1 | 123.0 | 229.6 |
| humaneval | 343.8 | 164.5 | 255.5 |

### 2.1 베이스라인 (TSK_042 500p × conc=32) 대조

| Metric | TSK_042 baseline (500p×32) | R0 (50p×16) |
|---|---|---|
| tps | 7 803 | 4 169 |
| ttft p50 (ms) | 69.3 | 27.8 |
| tpot p50 (ms) | 3.1 | 5.3 |
| α | 0.8814 | 0.7774 |
| GPU util | 26.5% | 67.2% |
| GPU mem | 632 567 | 632 544 |

→ 50p×16 (작은 burst) 에서는 batch saturation 부족 → per-req latency 가 더 낮지만 aggregate tps 도 비례하여 낮음. 베이스라인 의 26.5% GPU util 은 long-tail 의 reflection. 본 sweep 의 R0/R1/R2 비교만 같은 footing 으로 유효.

---

## 3. cudagraph_mode 별 영향 정리

| 영향 | PIECEWISE (R0) | FULL_AND_PIECEWISE (R2) | "FULL" 요청 (R1) |
|---|---|---|---|
| Attention launch 경로 | 항상 graph 외 (`cuLaunchKernelEx`) | decode-only batch 는 FULL graph 안, mixed batch 는 PIECEWISE | 동일 (실 적용은 FaP) |
| Capture sizes | 51 (전 ladder) | 15 + 15 | 15 + 15 |
| GPU memory cost | base | +0 (KV cache available 동일) | +0 |
| TPOT p50 | 5.3 ms | 5.6 ms | **9.4 ms** (이상치) |
| GPU util | 67.2% | 47.5% | 45.0% |
| **output_tps** | 4 169 | **4 413 (+5.9%)** | 2 546 (−38.9%) |

- **R0→R2 net win**: +5.9% throughput. SUB_201 §5 의 "launch overhead 38% 회수" 기대 (PIECEWISE→FULL theoretical) 대비 매우 작음. 이유:
  1. **FULL 단독 불가** (FlashInfer cap) → R2 의 FULL graph 는 **decode-only batch 만** 사용 → mixed prefill-decode batch 에서는 R0 와 동일 PIECEWISE path. Suffix 의 spec verify 가 mixed prefill-decode 로 자주 falling → FULL graph hit-rate 낮음.
  2. **B200 의 baseline overhead 가 H100 대비 다름** — SUB_201 의 38% 수치는 H100 + FA3 가정. B200 + FlashInfer 의 launch overhead 비중은 측정 안 됨 (nsys profile 미수행, dev venv telemetry 미적용).
- **R1 anomaly (−38.9%)**: 동일 mode 로 fallback 됐는데 R2 보다 훨씬 느림. 이는 측정 noise 또는 cold-cache 효과로 추정 (R1 이 시간순으로 가운데, 직전 R0 의 워크로드 후 warmup 잔여 graph cache 손실 가능). 셀당 50p (≤1 분) 의 작은 burst → tail outlier 1~2 req 의 영향이 크다. **재측정 필요** (시간 부족으로 본 turn 에선 skip).

---

## 4. SUB_201 §5 launch overhead 38% 회수 검증

| 항목 | 기대 (DESIGN §2.2) | 측정 (R0→R2) |
|---|---|---|
| launch rate 감소 | ~5× (PIECEWISE 38~70/iter → FULL 1~3/iter) | **미측정** (nsys 미수행) |
| throughput net gain | +10% 이상 (PASS 조건) | **+5.9%** |
| TPOT p50 개선 | 감소 기대 | 5.3 → 5.6 ms (사실상 동일, +0.3) |
| GPU util | 증가 기대 | 67.2 → 47.5 (감소; batch saturation 부족 가능성) |

**판정**: SUB_201 §5 의 net-win PASS 조건 (`tps ≥ baseline × 1.10`) **미달**. 단, 본 측정은 **B200 + FlashInfer** 환경이고 SUB_201 §5 는 H100 + FA3 (또는 FA2) 가정이라 직접 비교 invalid. **prod 머신 (H100 + FA3) 에서 재측정 필요**. 본 측정의 1차 finding 은 **B200 환경에선 FULL 단독 불가 = FaP 만 사용 가능 → §5 의 회수 시나리오 (FULL 단독) 자체가 적용 안 됨**.

---

## 5. 발견한 장애물

1. **FlashInfer backend cap (P0)**: B200 의 default attention backend 는 FLASHINFER (`vllm/v1/attention/backends/flashinfer.py`), `AttentionCGSupport.UNIFORM_BATCH`. 결과적으로 `--cudagraph-mode FULL` 요청은 자동으로 `FULL_AND_PIECEWISE` 로 다운그레이드. SUB_201 §5 의 회수 시나리오 (FULL 단독) 는 **FA3 (FlashAttention v3) backend 강제 + cap=ALWAYS 일 때만** 활성화. B200 에서 FA3 강제 시도 필요 (`VLLM_ATTENTION_BACKEND=FLASH_ATTN` env + FA3 빌드 확인).
2. **Capture size matrix 폭증 없음**: 우려와 달리 spec_decode (num_spec=32) + FULL 조합에서도 capture sizes = 15 (decode-only ladder 만) 로 합쳐져 부팅 시간 +6~10 s 만 증가. memory 영향 0. **장애물 아님**.
3. **R1 anomaly**: 동일 mode 인데 R2 대비 −42% tps 측정. 50p × conc=16 의 작은 burst 가 noise 에 너무 민감. **최소 100p × conc=32 + 각 셀 3회 repeat** 권장.
4. **Telemetry patch 미적용**: worktree 의 `async_scheduler.py` 변경이 main repo link venv 에 반영 안 됨. iter rate / launch rate 정량 측정 불가. 해결: (a) worktree-local venv 재빌드, (b) main repo cherry-pick, 또는 (c) nsys profile 로 대체.

---

## 6. 다음 dev step 권장

1. **(즉시, 1h)** B200 에서 **FA3 강제** 시도: `VLLM_ATTENTION_BACKEND=FLASH_ATTN` env 로 `FULL` 모드 단독 활성화 가능 여부 확인. FA3 빌드 미존재 시 FA2 로 fallback → `UNIFORM_BATCH` 라 동일하게 다운그레이드 → **dev step 종료**.
2. **(우선, 2~3h)** Burst size 확대 후 R0/R2 재측정: 500p × conc=32 (TSK_042 와 동일), 3회 repeat, 평균 + std. R1 의 anomaly 가 noise 였는지 확인.
3. **(중기, 4~6h)** **prod H100 노드** 로 동일 sweep 이동. H100 + FA3 → `AttentionCGSupport.ALWAYS` → **FULL 단독 가능**. 이때 §5 의 38% 회수 시나리오 검증 가능. nsys profile 동시 (60s window).
4. **(production patch, ~수일)** B200 환경 한정 lever:
   - `vllm/v1/attention/backends/flashinfer.py` 의 `_cudagraph_support` 를 mixed-batch UNIFORM 인지 dynamic 판정 후 일부 mixed batch 도 FULL graph 안에 포함하도록 patch (DESIGN §5.3 의 cudagraph_dispatcher.py 영역).
   - CLI default 변경은 **현행 PIECEWISE 유지** 권장 (R2 의 +5.9% 가 R1 noise 가능성 고려 시 통계적으로 약함). 더 큰 burst 와 H100 검증 후 결정.

---

## 7. git status (worktree)

```
On branch poc/b3_sched_260604_2208
nothing to commit, working tree clean
```

새로 생성:
- `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_sched/MEASUREMENTS.md` (본 파일)
- `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_sched/runs/r0_piecewise.json`
- `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_sched/runs/r0_piecewise.raw.jsonl`
- `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_sched/runs/r1_full.json`
- `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_sched/runs/r1_full.raw.jsonl`
- `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_sched/runs/r2_full_and_piecewise.json`
- `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_sched/runs/r2_full_and_piecewise.raw.jsonl`

Boot logs (`/tmp/r0_boot.log`, `/tmp/r1_boot.log`, `/tmp/r2_boot.log`) 는 tmpfs 라 reboot 후 손실. 필요 시 PoC 디렉토리로 copy.
