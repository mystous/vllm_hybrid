# SUB_184 — IDE_018 / TSK_032 phase-burst Phase 2 task pool fill canonical 500p e2e

> **parent**: TSK_032 (IDE_018) — paper §4 main lever follow-up to SUB_169.
> **scope**: 2026-05-27 12:08 ~ 12:14 KST (~6 min wall, OFF + ON chained, max-tokens=32).
> **status**: ✅ 완료 — task pool dummy fill 적용 + canonical 500p × 3 mix × 3 scenario × OFF/ON 측정. **hypothesis rejected** ⚠.

---

## 0. 두괄식 — paper main lever core 가설 **reject** ⚠⚠

본 SUB 는 SUB_169 의 stub 상태 (+1.35% 3-mix avg AGSD, CPU util 5.33%) 을 **task pool 에 실제 CPU work 채워서** paper target 30%+ CPU util + +5% throughput 으로 끌어올리는 시도.

| 측정 | OFF | ON (phase-burst + dummy fill) | Δ |
|---|---:|---:|---:|
| **3-mix avg AGSD** | **4,306 tps** | **4,230 tps** | **−1.75%** ⚠ |
| balanced AGSD | 3,993 | 3,874 | −2.97% |
| sonnet AGSD | 4,383 | 4,338 | −1.02% |
| code AGSD | 4,542 | 4,479 | −1.39% |
| CPU util mean | 4.08% | 5.61% | +1.53 pp (paper target 30% 멀음) |
| paper §4 target | — | — | +5% AGSD / 30%+ CPU util (실패) |

→ **dummy task pool fill 단계에서도 throughput net positive 미발생**. paper §4 main lever 의 core 가설 ("CPU idle GPU phase 와 overlap 만 시키면 critical path 변하지 않고 CPU util 끌어올림 가능") 이 본 SUB scope 의 setup 에서 **reject**.

---

## 1. ON 모드 setup

| 항목 | 값 | 비고 |
|---|---|---|
| `VLLM_USE_PHASE_BURST` | =1 | mark_phase hook activate |
| `DUMMY_FILL` | =1 | task pool 의 dummy CPU work 실행 |
| `COUNT` | =8 | dummy operation count per phase |
| `ITERS` | =8 | inner loop iters |
| `NUM_WORKERS` | =8 | OMP / pthread workers in pool |
| `CPU_BASE` | =80 | dummy worker cores 80-87 (vllm CPU 와 격리) |
| lazy-init log | `IDE_018 phase-burst: lazy-init OK (runtime=PhaseBurstRuntime workers/cpu_base=8/80)` | 8 vllm worker process × 8 worker = 64 dummy thread |

본 SUB 의 dummy fill 은 light setup (8 worker × 8 iters × 8 count) — paper target 30% CPU util 도달 전 시작점.

## 2. 상세 결과 — 9 cell

| mix | scen | OFF | ON | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1,565 | 1,586 | **+1.36%** |
| balanced | trident-only | 1,646 | 1,310 | **−20.41%** ⚠ |
| balanced | **agsd-gated** | **3,993** | **3,874** | **−2.97%** |
| sonnet | vanilla-only | 1,988 | 2,007 | +0.92% |
| sonnet | trident-only | 3,469 | 2,971 | **−14.36%** ⚠ |
| sonnet | **agsd-gated** | **4,383** | **4,338** | **−1.02%** |
| code | vanilla-only | 1,863 | 1,877 | +0.75% |
| code | trident-only | 3,013 | 2,573 | **−14.62%** ⚠ |
| code | **agsd-gated** | **4,542** | **4,479** | **−1.39%** |

### 2.1 핵심 observation

1. **vanilla-only 모두 small positive (+0.7 ~ +1.4%)** — single-instance vllm 의 phase signal 활용 시 dummy fill 이 GPU phase 와 partial overlap.
2. **trident-only 모두 catastrophic regression (−14 ~ −20%)** ⚠ — spec decoding 의 cudagraph 가 더 dense 한 GPU phase 구성, dummy fill 의 CPU work 가 contending 발생. phase signal mark 위치가 spec decoding 의 inner loop 안에서 잘못된 시점에 fire.
3. **agsd-gated mixed (−1 ~ −3%)** — vanilla path 의 +0.9~+1.4% 와 trident path 의 −14~−20% 가 AGSD router 의 backend 분배에 따라 partial cancel. balanced 가 chat 분기 비율 (33%) 로 가장 trident-heavy → 가장 큰 regression.
4. **CPU util OFF 4.08% → ON 5.61%** — dummy work 실행 자체는 confirmed, 단 paper target 30% 의 1/6 수준.

### 2.2 SUB_169 와 비교

| 측정 | SUB_169 (stub) | SUB_184 (dummy fill) | 차이 |
|---|---:|---:|---:|
| 3-mix avg AGSD Δ | **+1.35%** (positive) | **−1.75%** (regression) | **−3.10 pp 악화** |
| CPU util ON | 5.33% | 5.61% | +0.28 pp (marginal) |
| trident-only Δ | minor | −14 ~ −20% | catastrophic worsening |

→ SUB_169 의 +1.35% 가 **stub 상태에서 phase-mark IPC overhead 만 보여준 noise positive** 였다는 honest interpretation. dummy work 가 더해진 본 SUB 에서 throughput 더 깎임 (특히 trident path) → **CPU work 가 overlap 이 아닌 contention 으로 작용**.

---

## 3. 가설 rejection 의 의미

| paper §4 main lever 가설 | 본 SUB 결과 |
|---|---|
| (a) GPU phase 동안 CPU 가 idle 인 시간이 존재 | confirmed (OFF CPU util 4% 가 그 증거) |
| (b) phase mark IPC 가 충분히 빠름 | confirmed (SUB_169 4.67 μs) |
| (c) CPU work 를 GPU phase 와 동시 실행하면 critical path 영향 없음 | **rejected** — trident-only 가 −14~−20% 회귀. spec decoding 의 GPU phase boundary 가 dense + CPU work 가 vllm worker 의 GIL / pinned alloc 와 contending |
| (d) 결과적으로 CPU util ↑ + throughput 유지 | **rejected** — CPU util +1.5 pp 에 throughput −1.75% |

본 SUB 의 dummy fill 은 8 worker × 80 cpu_base 로 vllm 의 active core (0-49 vanilla + 56-105 trident) 와 cpu_base=80 영역만 겹침 — 그래도 contention. **CPU work 자체의 cache pollution + Linux scheduler 의 worker migration + GPU<->CPU pinned alloc lock 영향** 등 indirect contention 으로 추정.

## 4. paper §4 implication

기존 honest assessment:
> "paper §4 main lever IDE_018 phase-burst 는 SUB_169 stub 측정에서 +1.35% 1-run 결과를 보였으나, SUB_184 dummy fill 단계에서 −1.75% 로 회귀하여 stub 결과의 +1.35% 가 phase mark IPC overhead 의 noise 였음을 확인. 따라서 본 fork 에서 IDE_018 의 throughput lever 자격 **상실**. CPU util 끌어올림 자체는 다른 lever (e.g., side-channel batch precompute, decoupled NEW workload) 로 paper-bound 가능."

## 5. 다음 step (별도 SUB)

- 본 lever 의 후속 SUB 권고 **신중**: paper main lever 자격 회복하려면 (a) phase mark site 의 정확한 GPU idle window 측정 + (b) CPU work 의 cache-aware partition + (c) GIL release 안전 보장 등 invasive vllm core 수정 필요 (heavy).
- 대안: **SUB_185 = SUB_178 cold-KV decompress long-context workload e2e** — NEW workload 후보의 e2e first signal. paper main lever 자격 상실한 IDE_018 대신 IDE_017 의 NEW workload 로 main 후보 이동.

---

## 6. raw data

- `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (18 cell)
- `_monitor_{off,on}_{cpu,gpu}.csv` (0.5s interval)
- `logs/{main,vanilla,trident,router,monitor,chain}_{off,on}.log`
- `logs/phase_burst_log_on.txt` — lazy-init confirmation
- `launcher.sh` — OFF/ON chained launcher
- `build/` — task pool dummy fill 빌드 산출물
