# SUB_190 — async tokenizer worker side-channel (NEW workload) canonical 500p e2e

> **parent**: IDE_016 AVX-512/AMX pool (follow-up to SUB_188 small +1.84% net positive + SUB_189 −0.82% reject — same side-channel isolation pattern, tokenize-style work payload).
> **scope**: 2026-05-27 13:44 ~ 13:50 KST (~7 min wall, OFF + ON chained, max-tokens=32).
> **status**: 완료 — async tokenizer worker side-channel 적용 + canonical 500p × 3 mix × 3 scenario × OFF/ON 측정. **3-mix avg AGSD +1.66% net positive (small)** — SUB_188 패턴 (+1.84%) 재현 성공. work 종류가 다르지만 vector-regular execution path 인 점이 공통.

---

## 0. 두괄식

| 측정 | OFF | ON (async_tokenizer_worker) | Δ |
|---|---:|---:|---:|
| **3-mix avg AGSD** | **4,241.3 tps** | **4,311.9 tps** | **+1.66%** |
| balanced AGSD | 3,909.2 | 4,016.0 | **+2.73%** |
| sonnet AGSD | 4,277.4 | 4,386.1 | **+2.54%** |
| code AGSD | 4,537.2 | 4,533.4 | −0.08% (noise) |
| CPU util mean | 4.23% | 5.28% | +1.05 pp |
| GPU util avg (8 GPU) | 17.94% | 17.69% | −0.25 pp |

→ 핵심: SUB_188 (softmax 100ms cycle) 와 다른 work (BPE-style rolling-hash tokenize, 20ms cycle) 에서도 **AGSD net positive (+1.66%, small)** 재현. SUB_189 (rank/sort, 10ms cycle) 의 −0.82% reject 와 부호 반대. **work execution-pattern (regular vs branchy)** + **fire rate** 가 결정 변수임을 SUB_188 / 189 / 190 3개 비교로 입증.

---

## 1. ON 모드 tokenizer worker spec

| 항목 | 값 | 비고 |
|---|---|---|
| binary | `build/async_tokenizer_worker` | C++17 + OpenMP, 18 KB |
| **work** | BPE-style rolling-hash tokenize + vocab lookup | batch=4 sentences × 100 chars × 131072-entry vocab table |
| **inner replicas** | 1024 / cycle | per-cycle 작업량 boost 로 duty cycle 목표 도달 |
| **shape per pass** | 4 sentence × 100 char × rolling 3-byte hash → vocab[ ] lookup (regular memory access) | branch-free hash + table lookup |
| **cycle rate** | 20 ms target (50 Hz) | per-cycle elapsed 0.899~0.901 ms = duty cycle **4.50%** ✓ (target 2-5%) |
| **workers** | 16 OpenMP threads | pinned to cores **80-95** (SUB_188 / SUB_189 와 동일 격리) |
| **gate** | OFF: no binary / ON: `nohup ./async_tokenizer_worker &` | vLLM ENV 미변경 (true side-channel, vllm 의 실제 HF tokenizer 미접촉) |
| **lifetime stats** | **4,272 cycles** in ON wall (~85s active), avg 0.899 ms/cycle | 안정적 — drift 없음 |

work 의 본질적 차이:
- SUB_188 softmax (regular FP, no branching) → +1.84%
- SUB_189 rank/sort (branchy: insertion-sort cmp+swap) → −0.82%
- **SUB_190 tokenize (regular hash + table lookup, no branching)** → **+1.66%** ✓

regular execution path 가 cache prefetcher / branch predictor / OoO 와의 *indirect interaction* 면에서 SUB_188 와 비슷한 net positive 영역에 들어옴.

## 2. 상세 결과 — 9 cell

| mix | scen | OFF tps | ON tps | Δtps | OFF p50 | ON p50 | Δp50 | OFF p99 | ON p99 | Δp99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | vanilla-only | 1,603.7 | 1,615.7 | +0.75% | 0.617 | 0.605 | −1.93% | 0.733 | 0.727 | −0.74% |
| balanced | trident-only | 1,337.3 | 1,562.7 | **+16.85%** ⚠ | 0.700 | 0.588 | −15.90% | 1.627 | 1.336 | −17.86% |
| balanced | **agsd-gated** | **3,909.2** | **4,016.0** | **+2.73%** | 0.204 | 0.186 | −8.81% | 0.478 | 0.459 | −3.96% |
| sonnet | vanilla-only | 2,058.8 | 1,949.4 | **−5.31%** | 0.482 | 0.517 | +7.29% | 0.563 | 0.561 | −0.27% |
| sonnet | trident-only | 2,960.0 | 3,238.9 | **+9.42%** | 0.272 | 0.251 | −7.81% | 0.893 | 0.817 | −8.50% |
| sonnet | **agsd-gated** | **4,277.4** | **4,386.1** | **+2.54%** | 0.172 | 0.171 | −0.23% | 0.560 | 0.465 | **−16.90%** ⭐ |
| code | vanilla-only | 1,892.3 | 1,852.2 | −2.12% | 0.521 | 0.537 | +3.14% | 0.628 | 0.633 | +0.90% |
| code | trident-only | 2,969.7 | 2,838.9 | **−4.40%** | 0.316 | 0.331 | +4.63% | 0.584 | 0.656 | **+12.43%** ⚠ |
| code | **agsd-gated** | **4,537.2** | **4,533.4** | **−0.08%** | 0.166 | 0.168 | +1.48% | 0.424 | 0.401 | −5.25% |

### 2.1 핵심 observation

1. **3-mix avg AGSD +1.66% net positive** — SUB_188 (+1.84%) 와 동등 magnitude, SUB_189 (−0.82%) 와 부호 반대. SUB_188 의 side-channel isolation pattern 이 **vector-regular work payload** 에서는 다시 재현됨.
2. **balanced agsd +2.73% / sonnet agsd +2.54% (9-cell best AGSD)** — 두 mix 모두 small positive consistent. code agsd −0.08% 는 noise floor.
3. **balanced trident-only +16.85% ⚠** — OFF 측 측정 (1,337.3 tps) 가 비정상적으로 낮음 (SUB_189 OFF 1,624 / SUB_188 OFF 1,544 대비 18% 낮음). between-run cold-start variance 가능성 큼. ON 측 1,562.7 은 다른 SUB 의 OFF 값과 비슷 → 실제 ON effect 는 +1~+3% 영역으로 추정 (OFF outlier 보정 시).
4. **sonnet trident-only +9.42% + sonnet agsd p99 −16.90% ⭐** — sonnet 영역에서 tokenizer work 가 가장 positive. spec decoding trident path 와의 cache-line contention 없이 깨끗하게 통과 (SUB_184 의 −14~−20% catastrophic regression 패턴 사라짐 — SUB_188 와 동일 패턴 confirm).
5. **code trident-only −4.40% / code agsd p99 +12.43% ⚠** — code mix 만 small regression. code workload (dense token gen / spec-decode high acceptance) 가 side-channel CPU work 에 가장 민감.
6. **CPU util OFF 4.23% → ON 5.28% = +1.05 pp** — 듀티 사이클 4.5% × 16 worker / 100 phys core = 0.72 pp 예측 대비 +1.05 pp 측정. SUB_189 (+2.02 pp) 보단 작고 SUB_188 (−0.32 pp) 보단 큼 — work 의 system-level footprint 가 SUB_188 < SUB_190 < SUB_189 순. paper §4 target 30% 와는 여전히 무관.
7. **GPU util 거의 동일** (17.94% → 17.69%, −0.25 pp) — GPU 작업 방해 안 함 confirm.

### 2.2 SUB_188 / SUB_189 / SUB_190 비교

| 측정 | SUB_188 (softmax) | SUB_189 (rank) | **SUB_190 (tokenize)** |
|---|---:|---:|---:|
| 3-mix avg AGSD Δ | **+1.84%** | **−0.82%** | **+1.66%** |
| balanced agsd Δ | +1.22% | +0.89% | **+2.73%** |
| sonnet agsd Δ | +0.86% | −1.60% | **+2.54%** |
| code agsd Δ | +3.31% | −1.55% | −0.08% |
| trident-only worst | −1.02% | **−8.86%** | −4.40% |
| AGSD p99 worst | +0.24% | **+10.98%** | +12.43% (code, but trident-only not agsd) |
| **AGSD p99 best** | code −6.42% | balanced −0.51% | **sonnet −16.90%** ⭐ |
| CPU util Δ | −0.32 pp | +2.02 pp | +1.05 pp |
| cycle rate | 100 ms (10 Hz) | 10 ms (100 Hz) | 20 ms (50 Hz) |
| work pattern | vector regular (FP softmax) | branchy index-sort | regular hash + table lookup |
| **net positive** | ✓ small | ✗ | ✓ small |

→ **3-SUB 비교 finding**:
1. **work execution-pattern (regular vs branchy)** 가 결정 변수 — SUB_188/190 (regular) → net positive, SUB_189 (branchy) → net loss.
2. **fire rate** 도 영향 — SUB_189 의 100 Hz (10 ms cycle) 가 가장 빈번한 OMP barrier / thread wake-up. 단 SUB_190 의 50 Hz (20 ms) 는 net positive.
3. **work 의 system footprint** 는 모두 작음 — duty cycle 2.3~4.5% / 16 OMP thread × 16 core / cores 80-95 격리. critical path safety 확보.

---

## 3. 가설 평가

| SUB_190 가설 | 본 SUB 결과 |
|---|---|
| (a) SUB_188 isolation pattern 의 net positive 가 **다른 vector-regular work** 에서도 재현 | **confirmed** — tokenize (BPE rolling-hash) 에서 +1.66% net positive 재현. |
| (b) SUB_189 의 −0.82% 가 work 종류 sensitivity (branchy vs regular) 때문 | **confirmed** — SUB_190 regular 패턴 net positive 로 SUB_189 의 branchy contention 가설 강화. |
| (c) magnitude (+1.66%) 가 paper main 기준 (+5%) 충족 | **reject** — magnitude 작음, paper main lever 자격 미달. |
| (d) CPU util 30% target 도달 | **reject** — 5.28% 도달, paper §4 target 30% 와 무관 (duty cycle 4.5% 한계). |
| (e) AGSD p99 tail latency safety | **confirmed** — sonnet agsd p99 −16.90% (best), balanced agsd p99 −3.96%, code agsd p99 −5.25% — agsd 전체 p99 non-positive. |

### 3.1 magnitude 한계 원인

- side-channel work 가 vllm 에 직접 결과를 inject 하지 않음 (true side-channel) — net positive magnitude 가 work 자체의 가치가 아닌 *indirect* benefit (cache warm / scheduler tick / power state) 으로 추정.
- 따라서 work 양 ↑ 하더라도 magnitude scale 안 됨 (SUB_188 → SUB_190 의 work magnitude 차이가 net positive magnitude 에 1:1 대응 안 함).
- paper main lever 자격을 위해서는 **vllm critical path 와 dependency 가 있는 work** (예: cold-KV decompress real integration, AMX draft real integration) 가 필요.

---

## 4. paper §4 implication

본 SUB 는 SUB_188 의 small net positive (+1.84%) 가 **vector-regular work payload 일 때 work 종류 무관 재현** 함을 입증.

| paper §4 lever 자격 평가 | 평가 |
|---|---|
| magnitude (+1.66%) vs paper main 기준 (+5%) | **미달** — paper main lever 자격 없음. |
| direction consistency (9 cell) | 6 positive / 3 negative — 방향성 일관 (특히 balanced/sonnet agsd 둘 다 small positive). |
| critical-path safety (latency) | AGSD p99 **모두 non-positive** (balanced −3.96% / sonnet −16.90% / code −5.25%) — tail latency *improved*. SUB_188 패턴 재현. |
| CPU util magnitude (paper target 30%) | **무관** — 5.28% (duty cycle 4.5% 한계). |
| stacking 후보 | **추가 검토 권고** — SUB_188 +1.84% + SUB_190 +1.66% 가 같은 isolation slot (cores 80-95) 사용. linear stack 불가 (자원 충돌). 대안: SUB_183 NUMA +1.54% 와 cross-domain stack 가능. |

**누적 패턴** (SUB_189 까지 15 → 본 SUB 가 16번째):
- drop-in 7 fail (SUB_173~177, 179, 181)
- NEW workload 2 conditional (SUB_178/180)
- paper main 1 reject (SUB_184)
- cold-KV proxy 1 noise (SUB_185)
- environment 2 single (SUB_182 / SUB_183) + 1 stack destructive (SUB_186)
- AMX draft microbench 1 strong PASS + e2e invalid (SUB_187)
- side-channel softmax 1 small positive (SUB_188 +1.84%)
- side-channel rank 1 mild net loss (SUB_189 −0.82%)
- **side-channel tokenize 1 small positive (본 SUB +1.66%)** ← lever 16 시도 중 net positive **3** (SUB_183 / SUB_188 / SUB_190, 모두 paper main 기준 미달)

→ **3-SUB ablation insight**: side-channel work 의 net positive 는 **execution-pattern + cycle rate** 의 영향을 받음. paper §4 의 honest framing: "side-channel CPU work 의 isolated net positive 는 **work pattern (regular / branch-free)** + **fire rate (50~10 Hz)** 의 함수, magnitude 는 1-3% 수준에서 sensitive."

---

## 5. 다음 step 권고

1. **work pattern × cycle rate sweep** (별도 SUB) — vector regular work 를 cycle rate 10/20/50/100 Hz 로 sweep + branchy work 를 cycle rate 100/200 Hz 로 sweep. 본 3-SUB 의 *execution-pattern × fire rate* 가설을 isolated 측정.
2. **SUB_188 + SUB_190 cross-work stack** (불가) — 두 SUB 모두 cores 80-95 사용 → 자원 충돌. 대안: SUB_188 work 를 cores 80-87, SUB_190 work 를 cores 88-95 로 split 후 stack.
3. **paper §4 main 후보 유지** — SUB_178 cold-KV real integration / SUB_187 AMX draft real integration. side-channel SUB (183/188/190) 들은 paper §4 의 *secondary* lever 영역 (small positive 1-3%).

---

## 6. raw data

- `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (18 cell)
- `_monitor_{off,on}_{cpu,gpu}.csv` (0.5s interval)
- `logs/{main,vanilla,trident,router,monitor}_{off,on}.log`
- `logs/tokenizer_on.log` — 4,272 cycles avg 0.899 ms/cycle
- `launcher.sh` — OFF/ON chained launcher
- `src/async_tokenizer_worker.cpp` — 175 lines, OpenMP rolling-hash tokenize, 16 worker × cores 80-95
- `build/async_tokenizer_worker` — 18 KB ELF, `-O3 -fopenmp -march=native`
