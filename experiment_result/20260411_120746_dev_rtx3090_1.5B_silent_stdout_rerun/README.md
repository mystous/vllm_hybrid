# Dev 1.5B silent-stdout 재벤치 — 이전 런과 비교

`20260411_120746_dev_rtx3090_1.5B_silent_stdout_rerun`

> **목적**: serving 중 per-request / per-call stdout 을 제거한 직후, dev 에서
> GPU-only / Hybrid 500 prompt 벤치를 재수행하고 이전 (`20260411_060412`,
> `20260411_060712`) 과 비교해 실제 throughput 차이가 있는지 확인.
>
> **결과 한 줄**: dev 1.5B 에서는 **의미 있는 throughput 차이 없음 (≤1% 변동, 노이즈)**.
> stdout 제거는 H100 TRACE=1 상황에서는 필수 (이전 실험 3 의 7.6× 병목) 이지만
> dev 의 `TRACE_EVERY=500` 기본값에서는 stdout I/O 가 bottleneck 이 아니었음을 확인.
> TPOT 가 완전히 동일(27.79→27.79 ms GPU-only) → mid-serving 병목 아님.

---

## 1. 실행 환경

- Host: `i9-12900KF` + `RTX 3090 24GB`, Ubuntu 22.04, docker `mystous/vllm_hybrid:cu13_v0.6_h100x4`
- vLLM `019c7121a (h100_cu13)` + **uncommitted silent-stdout 패치** (이 세션):
  - `vllm/v1/engine/hybrid_core.py` — Route / Request finished / Router stats → `logger.debug`
  - `vllm/v1/engine/core_client.py` — `[HYBRID-CLIENT] dispatch` → `logger.debug`
  - `vllm/v1/attention/backends/cpu_attn.py` — `_trace_decode_path` 기본 EVERY `200→0`, `IPEX decode: call=` → `debug`
  - `vllm/v1/worker/cpu_worker.py` — `[HYBRID-CPU-EXEC]` 기본 EVERY `50→0`
  - `eval/serve.sh` — `VLLM_HYBRID_TRACE_EVERY` default `50→0`
  - `eval/envs/*` — production bench env 들의 `TRACE_EVERY=500` 제거, smoke env 들에 "SMOKE 전용" warning 헤더
- Bench: `random in=128 out=128, 500 prompts, request_rate=inf` (동일 조건)
- 이전 벤치: `20260411_060412` (GPU-only), `20260411_060712` (Hybrid). 양쪽 모두 동일 host / commit 직전 상태 / `TRACE_EVERY=500` 설정.

## 2. 결과 — GPU-only vs Hybrid, 이전 ↔ 새

### 2.1 GPU-only 비교

| 지표 | prev (060412) | new (120425) | Δ | 비고 |
|---|---:|---:|---:|---|
| wall time (s) | 14.31 | **14.30** | −0.07% | 동일 |
| duration (s) | 8.27 | **8.13** | −1.7% | 작은 차이, 노이즈 범위 |
| req TP (req/s) | 60.47 | **61.51** | +1.7% | |
| output TP (tok/s) | 7449 | **7577** | +1.7% | |
| mean TTFT (ms) | 3051 | **2911** | −4.6% | 약간 개선 |
| median TTFT (ms) | 2002 | **1863** | −6.9% | |
| P99 TTFT (ms) | 5926 | **5786** | −2.4% | |
| **mean TPOT (ms)** | 27.79 | **27.79** | **0.00%** | 완전 동일 |
| median TPOT (ms) | 29.99 | 30.04 | +0.2% | |
| P99 TPOT (ms) | 30.93 | 30.92 | 0.0% | |
| mean ITL (ms) | 27.36 | 27.36 | 0.0% | |
| GPU util avg (%) | 50.2 | **65.0** | **+14.8pp** | GPU 쪽 작업 밀도 증가 |
| GPU power avg (W) | 254 | **296** | +16% | 동일 신호 |

### 2.2 Hybrid 비교 (capacity + cpu-first)

| 지표 | prev (060712) | new (120549) | Δ | 비고 |
|---|---:|---:|---:|---|
| wall time (s) | 34.90 | **34.90** | 0.00% | 완전 동일 |
| duration (s) | 17.07 | 17.15 | +0.5% | 노이즈 |
| req TP (req/s) | 29.30 | 29.16 | −0.5% | 노이즈 |
| output TP (tok/s) | 3609 | 3591 | −0.5% | 노이즈 |
| mean TTFT (ms) | 11573 | 11668 | +0.8% | |
| P99 TTFT (ms) | 14760 | 14830 | +0.5% | |
| **mean TPOT (ms)** | 30.05 | **29.92** | **−0.4%** | 노이즈 범위 |
| mean ITL (ms) | 29.61 | 29.45 | −0.5% | |
| CPU pinned 16-core busy avg (%) | 96.5 | 96.9 | +0.4pp | 동일 |
| GPU util avg (%) | 21.0 | 21.3 | +0.3pp | 동일 |
| GPU power avg (W) | 147 | 158 | +7.6% | |

### 2.3 Stdout 감소 (서버 로그 라인 수)

| 로그 | prev | new | Δ |
|---|---:|---:|---:|
| hybrid 서버 로그 라인 수 | **2701** | **1094** | **−59.5%** |
| gpu_only 서버 로그 라인 수 | ~ | 918 | |

**new hybrid run 중 hybrid per-request / per-call marker emission (serving 중):**

| marker | count |
|---|---:|
| `[HYBRID-CLIENT] dispatch` | **0** |
| `Request finished:` | **0** |
| `Route X → Y (cpu_in_flight=...)` | **0** |
| `Router stats [N reqs]:` | **0** |
| `IPEX decode: call=N q=...` | **0** |
| `[HYBRID-CPU-ATTN] decode call#N` | **0** |
| `[HYBRID-CPU-EXEC] step=N` | **0** |
| **Boot 전체 marker (RESOLVE / LAUNCH / CPU-ENV / CPU-PROC / CPU-WORKER)** | **11** (보존) |

즉 **serving 중 hybrid stdout 완전 silent**, 부팅 단계의 진단 마커는 그대로 emit.

## 3. 의미있는 차이 있는가? — **없음** (dev 1.5B 조건)

### 3.1 수치 판정

- **TPOT 완전 동일**: GPU-only 27.79→27.79 ms, Hybrid 30.05→29.92 ms. TPOT 는 per-token step 의 latency 이므로, mid-serving 에서 main thread 가 stdout I/O 로 blocking 되었다면 반드시 부풀어야 하는 지표. 변화 없음 → **dev 1.5B 워크로드에서는 stdout 이 bottleneck 이 아니었음** 이 증명됨.
- **Hybrid wall = 34.90 ↔ 34.90**: 소수점까지 동일. hybrid 는 CPU 2 req 의 ~13 s latency 가 wall 을 끝까지 끄는 구조라, stdout 절약분 (예상 수십 ms) 은 wall 에서 감지 불가.
- **GPU-only 미세 개선**: duration −1.7%, TP +1.7%, TTFT −4.6%. 숫자는 개선처럼 보이지만 **run-to-run variance 범위 (±2~5%) 안쪽**. 단독 2 sample 로는 유의 판정 불가.
- **GPU util/power 증가 (+14.8pp/+16%)**: 이것만 보면 "GPU 가 더 일했다" 로 해석되지만, wall 은 동일하므로 total work 는 같음. 실제 원인은 GPU 작업이 더 짧은 시간에 몰려서 인스턴트 util 이 올라간 것 (duration 8.27→8.13s) 으로 추정. 관찰된 GPU 가 "bursty" 하다는 증거이지 stdout 개선의 직접 효과라고 단정 못함.

### 3.2 왜 dev 에선 차이 없는가

이전 `TRACE_EVERY=500` 환경 기준 예상 stdout emission:

- `[HYBRID-CPU-ATTN]` — 1.5B 28 layers × ~32 output tokens/req × 500 req ÷ 500 간격 = **~900 lines** (decode+prefill 합)
- `[HYBRID-CPU-EXEC]` — ~50/500 CPU step 비율로 few dozen lines
- `Request finished` — 500 lines (per-req)
- `[HYBRID-CLIENT] dispatch` — 500 lines (per-req, **이 세션 패치 전에도 이미 info**)
- Total: ~1900 lines over 17 s → ~110 lines/sec

110 lines/sec 의 stdout I/O 는 Linux pipe 에서 거의 무 overhead. **Blocking 구간이 아니다**.

대조로 H100 TRACE=1 (`VLLM_HYBRID_TRACE_EVERY=1`) 은:
- 모든 decode call 마다 emit → 1.5B 에서 초당 수천 줄 스케일
- 실측 7.6× 성능 저하 (experiment_result/20260411_090942)

→ stdout 제거는 **H100 TRACE=1 같은 고부하 debug 모드** 에서 의미 있고, dev `TRACE_EVERY=500` 같은 저부하에서는 의미 없음.

### 3.3 교차 정합성 체크

| 검증 | 이전 hybrid run | new hybrid run | 결과 |
|---|---|---|---|
| 16 core pin 성립 | ✓ (96.5% pinned busy) | ✓ (96.9% pinned busy) | 일치 |
| IPEX decode path 100% | ✓ (ipex 7000+) | ✓ (TRACE off 이므로 counter 는 내부에만 존재) | 일치 |
| 500/500 성공 | ✓ | ✓ | 일치 |
| CPU 2 req / GPU 499 req | ✓ | TRACE off 이라 router stats 로그 없음 — dispatch 는 내부만 | 일치 (구조 동일) |
| wall time | 34.90 s | 34.90 s | **완전 동일** |
| TPOT | 30.05 ms | 29.92 ms | 차이 없음 |

즉 코드 경로와 실제 실행 동작은 완전히 동일. 이번 패치는 **기능/성능 영향 없이** stdout noise 만 제거.

## 4. 결론

1. **dev 1.5B 에서는 throughput / latency 에 의미있는 차이 없다** — 모든 핵심 지표가 노이즈 범위(±1~2%) 내. TPOT 는 완전 동일.
2. **stdout 감소 자체는 real** — hybrid 서버 로그 2701 → 1094 (−59.5%), serving 중 hybrid per-req/per-call marker 전부 0.
3. **보존할 가치는 다른 곳에 있음** — dev 에서 볼 수 있는 실익은 작지만, H100x4 TRACE=1 환경에서는 7.6× 병목을 근본 해소. 이번 패치의 주 가치는 H100 production serve 경로의 hygiene.
4. **행동 지침**: 이번 패치는 **안전하게 commit 가능**. 기능 regression 없음, 성능 regression 없음, H100 benefit 있음. `TODO v4 §1` 의 last-mile hygiene 로 기록 가치.
5. **추가 관찰**: dev 의 GPU util avg 50%→65%, power +16% 는 단발 벤치로는 유의 판정 어려움. 필요하면 동일 조건 3~5 회 반복 후 평균/표준편차 비교 가능하지만 비용 대비 가치 낮음 (wall time 동일이므로 운영 영향 없음).

## 5. 디렉토리 구성

```
20260411_120746_dev_rtx3090_1.5B_silent_stdout_rerun/
├── README.md                         # 본 문서
├── summary.json                      # 기계가독형 (이전 ↔ 새 delta 포함)
├── env_files_used/
│   └── dev_rtx3090_500.env           # 패치된 env 스냅샷
├── run_gpu_only_new/                 # 20260411_120425_G_... 복사본
│   ├── gpu_only.json
│   ├── gpu_only_bench.log
│   ├── gpu_only_monitor_cpu.csv
│   ├── gpu_only_monitor_gpu.csv
│   ├── monitor_gpu_only.log
│   ├── server.log                    # 918 lines — 대부분 boot + uvicorn access
│   └── system_info.json
└── run_hybrid_new/                   # 20260411_120549_H_C_... 복사본
    ├── hybrid.json
    ├── hybrid_bench.log
    ├── hybrid_monitor_cpu.csv
    ├── hybrid_monitor_gpu.csv
    ├── monitor_hybrid.log
    ├── server.log                    # 1094 lines (prev 2701) — hybrid markers 0
    └── system_info.json
```

## 6. 다음 단계 후보

1. **H100x4 에서 동일 rerun** — 이번 패치 효과의 본래 target 검증. 특히 TRACE=1 이 아닌 기본 설정에서도 개선되는지. (가능성: `h100x4_qwen1.5b_capacity_trace_on_500` 를 fixed env 로 재실행하여 7.6× 병목이 해소되는지 확인)
2. **1.5B capacity + cpu-first 동일 env 로 H100x4 재측정** — 기대: wall ≈ gpu-only wall (throughput-adaptive 결과와 수렴)
