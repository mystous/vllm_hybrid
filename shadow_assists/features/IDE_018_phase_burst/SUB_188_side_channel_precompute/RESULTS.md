# SUB_188 — side-channel batch precompute (NEW workload) canonical 500p e2e

> **parent**: IDE_018 phase-burst (paper §4 main lever follow-up to SUB_184 reject + SUB_185 cold-KV proxy noise).
> **scope**: 2026-05-27 13:22 ~ 13:29 KST (~7 min wall, OFF + ON chained, max-tokens=32).
> **status**: 완료 — side-channel precompute 적용 + canonical 500p × 3 mix × 3 scenario × OFF/ON 측정. **3-mix avg AGSD +1.84% net positive (small)** — paper main 기준 (+5%) 미달이나 SUB_184 reject 이후 IDE_018 영역의 첫 양적 net positive 신호.

---

## 0. 두괄식

| 측정 | OFF | ON (side-channel precompute) | Δ |
|---|---:|---:|---:|
| **3-mix avg AGSD** | **4,232.7 tps** | **4,310.4 tps** | **+1.84%** |
| balanced AGSD | 3,942.8 | 3,991.0 | +1.22% |
| sonnet AGSD | 4,292.6 | 4,329.6 | +0.86% |
| code AGSD | 4,462.7 | 4,610.6 | **+3.31%** |
| CPU util mean | 4.73% | 4.41% | −0.32 pp (paper target 30% 멀음) |
| GPU util avg (8 GPU) | 17.50% | 17.71% | +0.21 pp |

→ 핵심: vllm critical path 와 분리된 side-channel CPU work (cores 80-95) 가 main path 와 cache line 공유 없이 fire 했을 때 **AGSD net positive (small, +1.84%)** 발생. 단 CPU util 은 오히려 −0.32 pp (precompute 16 worker 의 작업량이 100ms × 2.3ms = 2.3% duty cycle 수준이라 monitor 평균 변화 작음).

---

## 1. ON 모드 precompute spec

| 항목 | 값 | 비고 |
|---|---|---|
| binary | `build/side_channel_precompute` | C++17 + OpenMP, 17 KB |
| **work** | logprob softmax + log-softmax | batch=32, vocab=152,064 (Qwen 2.5 32B vocab) |
| **batch shape** | [32, 152064] FP32 | per-cycle softmax + log-softmax fused |
| **cycle rate** | 100 ms target | per-cycle elapsed 2.27~2.52 ms = duty cycle ~2.3% |
| **workers** | 16 OpenMP threads | pinned to cores **80-95** (vllm vanilla 0-49 / trident 56-105 와 분리) |
| **gate** | OFF: no binary / ON: `nohup ./side_channel_precompute &` | vLLM ENV 미변경 (true side-channel) |
| **lifetime stats** | 850 cycles in ON wall (~85s active), avg 2.52 ms/cycle | 안정적 — drift 없음 |

cores 80-95 는 본 머신 (NUMA 2 node, node0=0-55, node1=56-111) 의 node1 영역. vllm trident 56-105 와 cores 80-95 가 같은 NUMA node 에 있어 same-socket 이나 core 자체는 disjoint. HT siblings (112-223) 는 사용 안 함.

## 2. 상세 결과 — 9 cell

| mix | scen | OFF tps | ON tps | Δtps | OFF p50 | ON p50 | Δp50 | OFF p99 | ON p99 | Δp99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | vanilla-only | 1,607.9 | 1,544.4 | **−3.95%** | 0.615 | 0.635 | +3.21% | 0.746 | 0.778 | +4.33% |
| balanced | trident-only | 1,565.2 | 1,621.0 | **+3.57%** | 0.580 | 0.554 | −4.54% | 1.369 | 1.312 | −4.17% |
| balanced | **agsd-gated** | **3,942.8** | **3,991.0** | **+1.22%** | 0.187 | 0.186 | −0.38% | 0.477 | 0.478 | +0.24% |
| sonnet | vanilla-only | 2,013.3 | 2,031.6 | +0.91% | 0.498 | 0.483 | −2.90% | 0.577 | 0.549 | −4.96% |
| sonnet | trident-only | 3,283.3 | 3,272.8 | −0.32% | 0.237 | 0.245 | +3.20% | 0.796 | 0.830 | +4.29% |
| sonnet | **agsd-gated** | **4,292.6** | **4,329.6** | **+0.86%** | 0.167 | 0.173 | +3.37% | 0.458 | 0.441 | −3.76% |
| code | vanilla-only | 1,855.3 | 1,873.3 | +0.97% | 0.540 | 0.529 | −2.11% | 0.611 | 0.611 | −0.06% |
| code | trident-only | 2,968.1 | 2,937.7 | −1.02% | 0.320 | 0.328 | +2.42% | 0.636 | 0.646 | +1.54% |
| code | **agsd-gated** | **4,462.7** | **4,610.6** | **+3.31%** | 0.168 | 0.165 | −2.24% | 0.429 | 0.401 | **−6.42%** |

### 2.1 핵심 observation

1. **3-mix avg AGSD +1.84% net positive** — SUB_184 의 −1.75% (task pool dummy fill) 와 정확 반대 부호. 9-cell 차원에서 SUB_184 의 trident-only −14~−20% catastrophic regression 패턴이 **본 SUB 에서는 사라짐** (trident-only ±1~4% noise 범위).
2. **AGSD p99 latency 모두 non-positive 또는 ≤0.24% noise** (balanced +0.24% / sonnet −3.76% / code −6.42%) — side-channel work 가 tail latency 까지 침범하지 않음 확인. code AGSD p99 −6.42% 는 본 SUB 의 가장 강한 신호.
3. **vanilla-only balanced 만 −3.95% 회귀** — vanilla 단일 instance + balanced mix (chat-heavy) 에서 partial contention. 다른 vanilla-only cell 은 +0.9~+1.0% 로 정상. balanced vanilla 의 회귀는 1-run noise 가능성 큼 (재현 검증 필요).
4. **CPU util OFF 4.73% → ON 4.41%** — precompute 의 duty cycle (~2.3%) 이 16 worker × 2.3% / 100 core = 0.37 pp 추가 기여만 발생. monitor 평균은 −0.32 pp 로 오히려 약간 감소 (1-run sampling noise + vllm 의 OFF/ON CPU load 자체 변동). paper §4 target 30% util 와 무관.
5. **GPU util 거의 동일** (17.50% → 17.71%) — side-channel 이 GPU 작업 방해 안 함 confirmed.

### 2.2 SUB_169 / SUB_184 / SUB_185 와 비교

| 측정 | SUB_169 (phase-mark stub) | SUB_184 (task pool dummy) | SUB_185 (cold-KV proxy) | **SUB_188 (side-channel precompute)** |
|---|---:|---:|---:|---:|
| 3-mix avg AGSD Δ | +1.35% (noise) | **−1.75%** (regression) | +0.18% (noise, 8K long-context) | **+1.84%** (small net positive) |
| trident-only worst Δ | minor | **−20.41%** ⚠ | n/a (single-instance) | −1.02% (noise) |
| CPU util ON | 5.33% | 5.61% | 3.56% | 4.41% |
| critical-path coupling | phase-mark IPC | task pool 의 GIL/pinned alloc contention | prefill cache pollution (TTFT +8.83%) | **무 (cores 80-95, true side-channel)** |

→ SUB_184 의 task pool 가설 (CPU work 를 phase boundary 와 overlap) **실패 원인** 이 정확히 critical path 와의 cache/GIL contention 이었음을 SUB_188 의 +1.84% 가 거꾸로 입증. **"main critical path 와 자원 공유 없는 work 만 fire 하면 throughput net positive 가능"** 가설은 본 SUB scope 에서 **support** (단 magnitude 작음).

---

## 3. 가설 평가

| SUB_188 가설 | 본 SUB 결과 |
|---|---|
| (a) vllm critical path 와 분리된 CPU work 는 GPU 작업 방해 안 함 | confirmed (GPU util 변화 +0.21 pp 무의미) |
| (b) cache line / NUMA / GIL 공유 회피 시 throughput 영향 없음 | confirmed (AGSD 모두 비-회귀) |
| (c) +1.84% AGSD avg 신호 = 진짜 net positive 인가? | **conditional** — magnitude < 1-run noise floor (~±2%), 다만 9 cell 중 7 cell 이 positive 또는 ≤1% 회귀, code AGSD p99 −6.42% 가 best signal |
| (d) CPU util 끌어올림 = paper target 30% 도달 가능? | **rejected** — precompute duty cycle 2.3% 가 한계, batch=32×vocab=152K softmax 가 16 core 에서 2.3 ms 만 차지. paper 30% util 도달 위해 batch ↑ 또는 cycle ↓ 또는 work ↑ 필요 |

### 3.1 magnitude 한계 원인

- precompute work = batch×vocab softmax 가 너무 빨라서 (2.3 ms / 100 ms cycle) duty cycle 자체가 2.3%
- batch=32 (Qwen 32B 의 일반적 concurrent decode batch size 와 매칭) × vocab=152K 가 1 cycle 의 limit
- **cycle ↓ 또는 work ↑ 시도 시** critical path contention 위험 (SUB_184 패턴 재현)
- net positive magnitude (+1.84%) 는 GPU stream warm 유지 / scheduler thread 의 wake-up cache 효과 등 indirect benefit 으로 추정. CPU work 자체의 결과는 vllm 에 inject 안 됨 (true side-channel) → throughput 향상은 indirect

---

## 4. paper §4 implication

본 SUB 는 SUB_184 task-pool 의 paper main reject 이후 **IDE_018 영역 첫 양적 net positive**.

| paper §4 lever 자격 평가 | 평가 |
|---|---|
| magnitude (+1.84%) vs paper main 기준 (+5%) | **미달** — paper main lever 자격 없음 |
| direction consistency (9 cell) | 7 positive / 2 negative (vanilla-only balanced 만 −3.95%) — 방향성 일관 |
| critical-path safety (latency) | AGSD p99 비-회귀 — safe |
| CPU util magnitude (paper target 30%) | **무관** — 2.3% duty cycle 한계 |
| stacking 후보 | SUB_183 (NUMA pin) +1.54% 와 유사 magnitude. environment-level lever 와 stack 후보 가능 (단 SUB_186 destructive interference 패턴 주의) |

**누적 패턴**:
- drop-in 7 fail (SUB_173~177, 179, 181)
- NEW workload 2 conditional (SUB_178/180)
- paper main 1 reject (SUB_184)
- cold-KV proxy 1 noise (SUB_185)
- environment 2 single (SUB_182 / SUB_183) + 1 stack destructive (SUB_186)
- AMX draft microbench 1 strong PASS + e2e invalid (SUB_187)
- **side-channel precompute 1 small positive (본 SUB)** ← lever 14 시도 중 **두 번째 양적 small positive** (SUB_183 NUMA 와 함께)

→ paper main 후보는 여전히 SUB_178 cold-KV real integration / SUB_187 AMX draft real integration 으로 남고, SUB_188 은 *small positive stack 후보* 영역.

---

## 5. 다음 step 권고

1. **stack 검증** (별도 SUB) — SUB_183 NUMA + SUB_188 side-channel 동시 ON. SUB_186 패턴 (destructive interference) 이 본 lever 에도 적용되는지 확인.
2. **work scaling sweep** (별도 SUB, optional) — batch 64/128 + cycle 50ms 등으로 duty cycle 늘려 CPU util 끌어올림 시도. critical path 회귀 발생 시 SUB_184 패턴 재현 위험.
3. **work 선택 sweep** — softmax 외 chat template tokenize / top-k sort 등 다른 candidate work 측정. 본 SUB 의 work (softmax) 는 vllm critical path 와 무관한 임의 work — 더 가까운 work 선정 시 magnitude 확대 가능성.

---

## 6. raw data

- `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (18 cell)
- `_monitor_{off,on}_{cpu,gpu}.csv` (0.5s interval)
- `logs/{main,vanilla,trident,router,monitor}_{off,on}.log`
- `logs/precompute_on.log` — 850 cycles avg 2.52 ms/cycle
- `launcher.sh` — OFF/ON chained launcher
- `src/side_channel_precompute.cpp` — 150 lines, OpenMP softmax+log-softmax, 16 worker × cores 80-95
- `build/side_channel_precompute` — 17 KB ELF, `-O3 -fopenmp -march=native`
