# SUB_189 — CPU draft candidate ranker side-channel (NEW workload) canonical 500p e2e

> **parent**: IDE_019 multi-source drafter (follow-up to SUB_188 small +1.84% net positive — same side-channel isolation pattern, different work payload).
> **scope**: 2026-05-27 13:37 ~ 13:44 KST (~7 min wall, OFF + ON chained, max-tokens=32).
> **status**: 완료 — CPU draft candidate ranker side-channel 적용 + canonical 500p × 3 mix × 3 scenario × OFF/ON 측정. **3-mix avg AGSD −0.82% (mild net loss / noise floor)** — SUB_188 패턴 (+1.84%) 재현 실패. 본 SUB 의 work (rank/sort) 가 vllm spec_decode 의 ngram path 와 cache footprint / branchy code 양쪽에서 contention 신호 발생 추정.

---

## 0. 두괄식

| 측정 | OFF | ON (cpu_draft_ranker) | Δ |
|---|---:|---:|---:|
| **3-mix avg AGSD** | **4,295.5 tps** | **4,260.5 tps** | **−0.82%** |
| balanced AGSD | 3,973.9 | 4,009.2 | **+0.89%** |
| sonnet AGSD | 4,337.8 | 4,268.4 | −1.60% |
| code AGSD | 4,574.7 | 4,503.8 | −1.55% |
| CPU util mean | 4.29% | 6.31% | **+2.02 pp** (duty cycle ~3.5% confirmed in monitor) |
| GPU util avg (8 GPU) | 17.67% | 17.48% | −0.19 pp |

→ 핵심: SUB_188 의 same-isolation pattern (cores 80-95, 16 OMP, vllm path 분리) 을 적용했지만, **work 종류** (logprob softmax → draft candidate rank/sort) 가 바뀌면서 +1.84% → **−0.82%** 로 부호 반전. CPU util 은 +2.02 pp 로 실제 측정 가능 — 단 그 CPU work 가 net positive 로 변환되지 않음.

---

## 1. ON 모드 ranker spec

| 항목 | 값 | 비고 |
|---|---|---|
| binary | `build/cpu_draft_ranker` | C++17 + OpenMP, 18 KB |
| **work** | candidate frequency-based reorder | batch=32 seq × K=7 candidate × HIST=64 history tokens |
| **inner replicas** | 192 / cycle | per-cycle 작업량 boost 로 duty cycle 목표 도달 |
| **shape per pass** | 32 × 448 cmp + 32 × O(K²) insertion-sort | branchy code (insertion sort + 비교/swap) |
| **cycle rate** | 10 ms target (100 Hz) | per-cycle elapsed 0.351~0.393 ms = duty cycle **3.51%** ✓ (target 2-5%) |
| **workers** | 16 OpenMP threads | pinned to cores **80-95** (SUB_188 와 동일 격리) |
| **gate** | OFF: no binary / ON: `nohup ./cpu_draft_ranker &` | vLLM ENV 미변경 (true side-channel) |
| **lifetime stats** | **8,537 cycles** in ON wall (~85s active), avg 0.351 ms/cycle | 안정적 — drift 없음 |

work 의 본질적 차이 (SUB_188 대비):
- SUB_188 softmax: vector-heavy, regular memory access, branch-free (max / exp / log)
- SUB_189 rank: branchy (insertion-sort 의 cmp+swap loop), index-permutation, 작은 buffer (Kx HIST = 7×64 = 448 int32 per seq)

이 차이가 cache prefetcher / scheduler 의 micro-pattern 변화로 critical path 와 sub-stress contention 가능성 추정.

## 2. 상세 결과 — 9 cell

| mix | scen | OFF tps | ON tps | Δtps | OFF p50 | ON p50 | Δp50 | OFF p99 | ON p99 | Δp99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | vanilla-only | 1,599.8 | 1,601.3 | +0.09% | 0.626 | 0.611 | **−2.42%** | 0.755 | 0.743 | −1.62% |
| balanced | trident-only | 1,624.1 | 1,644.1 | **+1.23%** | 0.567 | 0.550 | **−2.93%** | 1.311 | 1.289 | −1.70% |
| balanced | **agsd-gated** | **3,973.9** | **4,009.2** | **+0.89%** | 0.187 | 0.171 | **−8.92%** | 0.484 | 0.481 | −0.51% |
| sonnet | vanilla-only | 1,975.7 | 1,920.7 | **−2.78%** | 0.508 | 0.516 | +1.46% | 0.592 | 0.629 | **+6.22%** |
| sonnet | trident-only | 3,336.6 | 3,247.2 | **−2.68%** | 0.238 | 0.247 | +3.76% | 0.802 | 0.804 | +0.22% |
| sonnet | **agsd-gated** | **4,337.8** | **4,268.4** | **−1.60%** | 0.169 | 0.174 | +2.79% | 0.436 | 0.442 | +1.37% |
| code | vanilla-only | 1,820.3 | 1,782.9 | **−2.05%** | 0.542 | 0.557 | +2.81% | 0.638 | 0.644 | +0.86% |
| code | trident-only | 3,035.9 | 2,766.8 | **−8.86%** ⚠ | 0.312 | 0.345 | **+10.37%** | 0.655 | 0.678 | +3.48% |
| code | **agsd-gated** | **4,574.7** | **4,503.8** | **−1.55%** | 0.163 | 0.168 | +3.12% | 0.443 | 0.491 | **+10.98%** |

### 2.1 핵심 observation

1. **3-mix avg AGSD −0.82%** — 1-run noise floor (~±2%) 안에 있어 **noise** 분류. 단 SUB_188 의 +1.84% net positive 패턴 재현 실패.
2. **balanced agsd +0.89% / sonnet agsd −1.60% / code agsd −1.55%** — 부호가 mix 따라 갈림 (mix-dependent). balanced 에서는 SUB_188 같은 양적 효과 부분 재현.
3. **code trident-only −8.86% ⚠ worst cell** — spec decoding trident path (suffix decoding) + code workload (dense token gen) 의 conjunction 에서 ranker work 가 contention 신호 보임. SUB_184 의 −14~−20% catastrophic regression 보단 약하지만 SUB_188 의 [−1.02%, +3.57%] 보단 큼.
4. **AGSD p99 latency** — balanced −0.51% / sonnet +1.37% / code **+10.98%** ⚠. code agsd p99 의 +10.98% 는 tail latency 가 ranker 의 branchy cache footprint 와 spec decode 의 ngram lookup 사이 micro-contention 가능성 시사.
5. **CPU util OFF 4.29% → ON 6.31% = +2.02 pp** — 듀티 사이클 3.5% × 16 worker / 100 phys core = 0.56 pp 예측 대비 +2 pp 측정 → ranker work 가 시스템 전체 cache miss / scheduler 호출 면에서 SUB_188 softmax 보다 무거움. paper §4 target 30% 와는 무관 (정확히 6% 수준).
6. **GPU util 거의 동일** (17.67% → 17.48%, −0.19 pp) — GPU 작업 자체는 방해 안 함. throughput 감소의 root cause 가 GPU stream stalling 이 아닌 CPU side ranking 자체.

### 2.2 SUB_188 / SUB_189 비교

| 측정 | SUB_188 (softmax side-channel) | **SUB_189 (rank side-channel)** |
|---|---:|---:|
| 3-mix avg AGSD Δ | **+1.84%** (small net positive) | **−0.82%** (mild net loss / noise) |
| balanced agsd Δ | +1.22% | +0.89% |
| sonnet agsd Δ | +0.86% | −1.60% |
| code agsd Δ | +3.31% (best) | −1.55% |
| trident-only worst | −1.02% (noise) | **−8.86% (code)** ⚠ |
| AGSD p99 worst | +0.24% (noise) | **+10.98% (code)** ⚠ |
| CPU util ON | 4.41% (−0.32 pp) | **6.31% (+2.02 pp)** |
| cycle rate | 100 ms (10 Hz) | 10 ms (100 Hz) |
| work type | vector regular (softmax) | branchy index-sort |
| **net positive** | ✓ small | ✗ |

→ **isolation 만으로는 충분하지 않음** confirm. 같은 cores 80-95, 같은 16 OMP, 같은 ENV 격리 위에서도 (1) **work 의 micro-pattern** (branchy vs regular), (2) **cycle rate** (10x 더 빠른 fire rate), (3) **per-cycle work magnitude** 의 조합이 critical path 와의 sub-cache / scheduler contention 발생 여부를 결정.

---

## 3. 가설 평가

| SUB_189 가설 | 본 SUB 결과 |
|---|---|
| (a) SUB_188 isolation pattern 재현 시 work 무관 net positive 가능 | **reject** — work 의 micro-pattern (branchy vs regular) 이 contention 발생 여부에 영향. |
| (b) cycle rate 100 Hz (vs SUB_188 10 Hz) 가 시스템 부담 증가 | **conditional support** — duty cycle 자체는 3.5% 로 SUB_188 (2.3%) 보다 약간 큼이나, fire rate 자체가 10× 빨라서 scheduler / cache eviction tick 의 frequency 도 증가. |
| (c) CPU util magnitude 끌어올림 | **partial** — +2.02 pp 는 SUB_188 (−0.32 pp) 보다 실측 진전, 단 paper §4 target 30% 와는 무관. |
| (d) draft candidate ranking 자체가 vllm spec_decode 와 indirect 관련 work | **side-channel only confirm** — ranker 결과는 vllm 에 inject 안 됨, 그러나 *유사 작업의 cache footprint* 가 spec_decode ngram path 와 같은 L3 line / scheduler tick 영역 공유 가능. |

### 3.1 magnitude 한계 원인 (mild net loss)

- **branchy code** — insertion-sort 의 cmp+swap loop 는 branch predictor / OoO speculation 에 부담. SUB_188 의 softmax (FP exp / log 의 deterministic execution path) 와 미세한 차이.
- **100 Hz fire rate** — 10 ms cycle 마다 OMP barrier + thread wake-up 이 발생 → SUB_188 의 100 ms cycle 대비 10× 더 잦은 sched event.
- **K × HIST = 7 × 64 = 448 int32 buffer** — small but non-zero L1 footprint. SUB_188 의 batch × vocab buffer 와 다른 access pattern.
- 그럼에도 **balanced mix 만 +0.89% net positive** — work 자체가 *완전히 무력* 한 것은 아님. mix 별 sensitivity 가 큼.

---

## 4. paper §4 implication

본 SUB 는 SUB_188 의 small net positive (+1.84%) 가 **work-agnostic 하지 않음** 을 실측으로 입증.

| paper §4 lever 자격 평가 | 평가 |
|---|---|
| magnitude (−0.82%) vs paper main 기준 (+5%) | **미달 + 부호 반전** — paper main lever 자격 없음. |
| direction consistency (9 cell) | 3 positive (balanced 전부) / 6 negative — 방향성 split (mix-dependent). |
| critical-path safety (latency) | AGSD p99 code +10.98% — tail latency 회귀 신호. SUB_188 의 "tail latency safe" 패턴 *재현 실패*. |
| CPU util magnitude (paper target 30%) | **무관** — 6.31% 까지 도달했으나 paper target 30% 과는 거리. |
| stacking 후보 | **권고 안 함** — 부호 반전 + p99 회귀로 stack 후보에서 제외. |

**누적 패턴** (SUB_188 까지 14 → 본 SUB 가 15번째):
- drop-in 7 fail (SUB_173~177, 179, 181)
- NEW workload 2 conditional (SUB_178/180)
- paper main 1 reject (SUB_184)
- cold-KV proxy 1 noise (SUB_185)
- environment 2 single (SUB_182 / SUB_183) + 1 stack destructive (SUB_186)
- AMX draft microbench 1 strong PASS + e2e invalid (SUB_187)
- side-channel softmax 1 small positive (SUB_188 +1.84%)
- **side-channel rank 1 mild net loss (본 SUB −0.82%)** ← lever 15 시도 중 net positive 여전 2 (모두 SUB_188 / SUB_183, paper main 기준 미달)

→ **paper §4 §4.x section 권고**: SUB_188 의 +1.84% 가 small isolated positive *였음* 을 보존하되, *side-channel work 의 net positive 가 work 종류에 sensitive* 라는 honest reading. 본 SUB 는 SUB_188 의 generalizability 를 약화하는 negative result.

---

## 5. 다음 step 권고

1. **work 의 micro-pattern 분석** (별도 SUB) — branchy vs regular execution path 가 cache prefetch / branch predictor / OoO 에 미치는 영향을 microbench 로 isolated 측정. SUB_188 (softmax) ↔ SUB_189 (rank) 의 net positive 부호 반전을 mechanistic 으로 설명할 수 있는지.
2. **cycle rate sweep** (별도 SUB) — SUB_189 의 work payload 그대로 cycle 10ms → 50ms → 100ms 로 sweep 해서 fire rate 가 magnitude 에 미치는 effect 분리.
3. **paper §4 stacking** — SUB_188 / SUB_183 만으로 stacking 후보 유지. 본 SUB 는 stack 후보 제외.

---

## 6. raw data

- `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (18 cell)
- `_monitor_{off,on}_{cpu,gpu}.csv` (0.5s interval)
- `logs/{main,vanilla,trident,router,monitor}_{off,on}.log`
- `logs/ranker_on.log` — 8,537 cycles avg 0.351 ms/cycle
- `launcher.sh` — OFF/ON chained launcher
- `src/cpu_draft_ranker.cpp` — 175 lines, OpenMP rank/sort, 16 worker × cores 80-95
- `build/cpu_draft_ranker` — 18 KB ELF, `-O3 -fopenmp -march=native`
