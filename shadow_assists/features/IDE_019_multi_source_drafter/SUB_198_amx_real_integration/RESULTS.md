# SUB_198 — IDE_019 AMX draft head proxy concurrent fire canonical 500p e2e

> **parent**: SUB_187 (AMX kernel microbench 0.524 ms ⭐) follow-up.
> **scope**: 2026-05-27 16:26 ~ 16:32 KST (~6 min wall, OFF + ON chain).
> **status**: ✅ 완료 — **proxy mode net loss** (−2.77%). real spec_decode integration 별도 SUB 필요.

---

## 0. 두괄식 — AMX proxy concurrent fire **net loss** ⚠

| 측정 | OFF | ON (AMX firer concurrent) | Δ |
|---|---:|---:|---:|
| **3-mix avg AGSD** | **4,363.8 tps** | **4,242.8 tps** | **−2.77%** ⚠ |
| balanced agsd | 4,032.6 | 4,030.0 | −0.06% |
| sonnet agsd | 4,459.2 | 4,333.7 | −2.81% |
| code agsd | 4,599.8 | 4,364.8 | **−5.11%** |
| AMX firer per-cycle | — | 375 ms / 1000ms cycle | duty 37.5% |
| paper main 기준 +5% | 미달 | (역방향 net loss) |

→ **proxy mode 에서 paper main lever 자격 미확보**. 단 real vllm spec_decode integration 별도 invasive SUB 가 valid 한 binding 평가 — 본 SUB 의 proxy 측정은 "concurrent CPU AMX work 가 critical path 와 contending" 의 first signal 일 뿐.

---

## 1. ON 모드 spec (proxy mode)

| 항목 | 값 |
|---|---|
| binary | `build/amx_firer` (SUB_187 의 .so 와 link, 64 OMP threads, cores 80-95) |
| AMX kernel | `amx_draft_qwen05b_step_ms(B=1, K=7)` — Qwen 0.5B small draft shape (hidden=896, vocab=152064) |
| cycle | 1000ms (1 Hz, work cost 375ms 보다 충분히 길게 설정) |
| per-cycle 측정 | 375.04 ms (microbench 0.524 ms × ~700 factor — microbench 와 e2e 격차) |
| duty cycle | ~37.5% (target 2-5% 보다 7× 높음, 단 cycle 길이로 1 Hz 만) |
| lifetime | 85 cycles in ON wall (~31.8s active fraction in 85s wall) |
| AMX hardware | active ✓ (`amx_hw=1`) |

**proxy mode 의미**: vllm spec_decode 의 draft proposer 로 통합 안 됨. 본 firer 는 standalone process 가 cores 80-95 에서 AMX kernel 을 concurrent fire — vllm 의 draft path 와 무관하게 CPU/cache/bus resource 만 contend.

## 2. 9-cell 상세

| mix | scen | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1,620.4 | 1,607.6 | −0.79% |
| balanced | trident-only | 1,598.2 | 1,572.9 | −1.59% |
| balanced | **agsd-gated** | **4,032.6** | **4,030.0** | **−0.06%** |
| sonnet | vanilla-only | 2,104.3 | 2,008.2 | −4.57% |
| sonnet | trident-only | 3,351.3 | 3,280.9 | −2.10% |
| sonnet | **agsd-gated** | **4,459.2** | **4,333.7** | **−2.81%** |
| code | vanilla-only | 1,901.0 | 1,904.3 | +0.17% |
| code | trident-only | 2,997.5 | 2,824.7 | **−5.76%** |
| code | **agsd-gated** | **4,599.8** | **4,364.8** | **−5.11%** ⚠ |
| **3-mix avg AGSD** | — | **4,363.8** | **4,242.8** | **−2.77%** |

### 2.1 핵심 observation

1. **code-heavy agsd −5.11%** — 9-cell worst. AMX work 가 code workload 의 critical path 와 가장 큰 contention.
2. **balanced agsd −0.06%** — noise floor. chat workload 의 cpu_jacobi 분기 비율 (33%) 로 AMX work 가 분산되어 영향 작음.
3. vanilla-only / trident-only 모두 small negative (−0.79% ~ −5.76%) — AMX work 의 CPU resource consumption 이 vllm worker thread 와 contention.

### 2.2 SUB_187 microbench 와 격차 분석

| 측정 | per-call latency |
|---|---:|
| SUB_187 microbench K=7 B=1 OMP=64 p50 | **0.524 ms** ⭐ |
| 본 SUB amx_firer per-cycle | **375 ms** |
| 격차 | **~700×** |

격차 원인 추정:
- microbench 의 OMP team 이 cache hot state 에서 측정 (warmup 후 100 iters mean)
- firer 의 매 cycle 1-call: OMP team spawn cost + AMX tile reload + thread migration cost
- 또는 SUB_187 의 per-step 계산식 (`total/K`) 의 분모 잘못 해석 가능성 (K=7 inner step 가 1 call 안에 포함된 거라면 단일 call 는 3.67 ms 정도, 100× 격차 여전)
- 또는 본 firer 의 single-call AMX step_ms 가 microbench 와 다른 path (inner loop replication 없는 raw call)

본 격차의 정확한 root cause = 별도 profiling SUB 필요. 단 본 SUB 의 proxy 측정 결과 (net loss) 는 valid signal.

## 3. real vllm spec_decode integration 의 필요성

본 SUB 는 **proxy** 측정 — vllm spec_decode 의 draft proposer 로 AMX kernel 통합 안 됨. real integration 의 경우:
- vllm 의 `vllm/v1/spec_decode/draft_model.py` 또는 `lookahead.py` 에 cpu_amx_proposer 추가
- draft candidate 가 vllm 의 verify pipeline 으로 자동 routing
- acceptance rate 측정 + token-level bit-exact 검증
- chat workload α=0.8 의 theoretical 4.12× spec speedup 가 e2e tps 로 변환 가능 여부 검증

본 SUB scope 외. 별도 invasive SUB 필요 — vllm core PR-level work.

## 4. accuracy gate

| gate | 결과 |
|---|---|
| token-level / 분포 정합성 | **PASS by construction** — proxy mode 라 vllm spec_decode path 미변경, AMX kernel 의 output 은 verify pipeline 에 inject 안 됨 |

## 5. paper §4 implication

- **SUB_187 microbench 0.524 ms / 4.12× theoretical spec speedup 는 isolated kernel-level metric 만 valid**
- proxy concurrent fire 단계 net loss = isolated kernel speedup 의 e2e 자동 변환 안 됨 (SUB_181 / SUB_185 / SUB_192 등 일관 패턴)
- paper §4 main lever 자격 회복 = real vllm spec_decode integration + canonical 500p × 3 mix × OFF/ON × multi-run 필수
- 본 fork 의 paper main +5% 도달 lever 검증 = **SUB_196 cellB (branchy × 100ms cycle, +5.28%) 만 남음**

## 6. 누적 패턴 final update

| 카테고리 | 시도 | 1-run net positive | paper main +5% 도달 (1-run) |
|---|---:|---:|---:|
| drop-in CPU kernel | 7 | 0 | 0 |
| environment individual | 2 | 1 | 0 |
| environment stack | 2 | 0 (destructive) | 0 |
| paper main IDE_018 | 1 | 0 (retract) | 0 |
| NEW workload e2e proxy | 3 (185, 192, 198) | 0 | 0 |
| AMX draft microbench | 1 (SUB_187) | feasibility PASS | 0 (e2e 미달성) |
| side-channel individual | 3 | 2 | 0 |
| side-channel × env pair stack | 1 (SUB_197) | 1 (+2.83%) | code-heavy +7.73% |
| side-channel stack triple | 1 (SUB_191) | 0 (destructive) | 0 |
| work-pattern ablation (2 cell) | 2 | 2 (cellA +0.98% / cellB +5.28%) | **cellB +5.28% 3-mix avg** ⭐ |
| multi-run variance verify | 1 (SUB_194) | retract 1 (SUB_190) | — |
| **누적** | **24** | **6** | **2 (SUB_197 code-only / SUB_196 cellB 3-mix avg)** |

→ paper main +5% 도달 lever 2 (SUB_196 cellB 3-mix avg, SUB_197 code-only). 둘 다 multi-run binding 검증 필요. AMX real integration 은 별도 invasive SUB 필요.

## 7. raw data

- `src/amx_firer.cpp` (SUB_187 .so 와 link, 64 OMP cores 80-95, B=1 K=7, 1000ms cycle)
- `build/amx_firer` binary
- `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (18 cell)
- `_monitor_{off,on}_{cpu,gpu}.csv`
- `logs/{main,vanilla,trident,router,monitor,precompute}_{off,on}.log`
- `launcher.sh` (SUB_188 launcher copy + AMX firer binary swap)
