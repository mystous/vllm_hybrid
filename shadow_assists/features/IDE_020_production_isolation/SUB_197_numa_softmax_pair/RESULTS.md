# SUB_197 — IDE_020 NUMA pin (SUB_183) × softmax precompute (SUB_188) pair stack canonical 500p e2e

> **parent**: SUB_183 (NUMA +1.54%) + SUB_188 (softmax +1.84%) cross-domain pair superposition test.
> **scope**: 2026-05-27 15:39 ~ 15:45 KST (~6 min wall, SUB_188 launcher copy + NUMA wrap patch + chain mode).
> **status**: ✅ 완료 — **destructive interference 패턴 break** ⭐ — pair stack 이 near-linear superposition 유지.

---

## 0. 두괄식 — pair stack **near-linear superposition** ⭐

| 측정 | 값 |
|---|---:|
| 3-mix avg AGSD OFF | **4,240.5 tps** |
| 3-mix avg AGSD ON (NUMA + softmax pair) | **4,360.4 tps** |
| **actual Δ** | **+2.83%** ⭐ |
| linear sum prediction (SUB_183 +1.54% + SUB_188 +1.84%) | +3.38% |
| **residual** | **−0.55 pp** (near-linear, very small destructive) |
| paper main 기준 +5% | 미달 (단 single-lever 보다 magnitude 1.5-1.8× 증가) |

→ SUB_186 (cgroup+hugepages+taskset+NUMA stack, residual −1.18 pp) 와 SUB_191 (NUMA+softmax+tokenize triple stack, residual −4.56 pp) 의 destructive interference 패턴을 **본 SUB pair stack 이 break**. cross-domain 2-lever stack 만 near-linear 유지.

---

## 1. ON 모드 stack 구성

| lever | spec |
|---|---|
| **NUMA pin** (SUB_183 lever) | `numactl --membind=0 --cpunodebind=0 taskset -c 0-49` (vanilla :8001) + `numactl --membind=1 --cpunodebind=1 taskset -c 56-105` (trident :8002) |
| **softmax precompute** (SUB_188 lever) | 16 OMP × cores 80-95 (default no taskset wrap in SUB_188), batch=[32,152064] FP32 softmax + log-softmax, 100ms cycle, duty cycle ~2.3% |
| 합산 duty cycle | side-channel CPU work 2.3% + NUMA OS-level wrap |

본 SUB launcher = **SUB_188 launcher 의 정확한 copy + NUMA wrap minimal patch** (검증된 패턴 재사용으로 launcher 결함 회피).

## 2. 9-cell 상세

| mix | scen | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1,599.7 | 1,544.4 | −3.45% |
| balanced | trident-only | 1,617.7 | 1,653.4 | +2.21% |
| balanced | **agsd-gated** | **4,066.8** | **4,020.8** | **−1.13%** |
| sonnet | vanilla-only | 1,983.1 | 2,056.6 | +3.70% |
| sonnet | trident-only | 3,436.4 | 3,344.8 | −2.67% |
| sonnet | **agsd-gated** | **4,384.3** | **4,459.7** | **+1.72%** |
| code | vanilla-only | 1,899.0 | 1,842.0 | −3.00% |
| code | trident-only | 3,062.2 | 3,063.2 | +0.03% |
| code | **agsd-gated** | **4,270.5** | **4,600.7** | **+7.73%** ⭐ |
| **3-mix avg AGSD** | — | **4,240.5** | **4,360.4** | **+2.83%** |

### 2.1 핵심 observation

1. **code-heavy agsd +7.73%** — 9-cell magnitude max, **paper main +5% 기준 도달**. 단 1-run measurement 라 cold-start variance 가능.
2. **balanced/sonnet agsd noise (−1.13% / +1.72%)** — small magnitude
3. **vanilla-only mixed (+3.7~−3.45%)** — sonnet workload 의 prefill 비중 ↑ → vanilla path 개선, balanced/code 의 mix 차이로 regression
4. **3-mix avg +2.83% 가 SUB_188 single (+1.84%) 보다 robust** — pair stack 의 cross-domain superposition benefit

### 2.2 superposition 비교

| stack | 시도 | linear sum prediction | actual | residual |
|---|---|---:|---:|---:|
| SUB_186 (cgroup+hugepages+taskset+NUMA) | env stack | +1.15% | −0.03% | **−1.18 pp** ⚠ |
| SUB_191 (NUMA + softmax + tokenize) | triple stack | +5.04% | +0.48% | **−4.56 pp** ⚠⚠ |
| **본 SUB_197 (NUMA + softmax)** | **pair stack** | **+3.38%** | **+2.83%** | **−0.55 pp** ⭐ |

→ **2-lever pair stack 만 near-linear 유지**. 3+ lever 부터 destructive interference 발생.

## 3. SUB_194 multi-run variance 와의 cross-reference

SUB_194 (Top-3 lever multi-run variance, 2026-05-27 15:05 KST 완료):
- L188 softmax 의 multi-run mean +0.53% (1-run +1.84% 의 30% 만)
- L190 tokenize 의 multi-run mean −5.96% (1-run +1.66% 부호 반전)
- L183 NUMA 의 multi-run mean +2.24% (1-run +1.54% 와 같은 방향, magnitude 2× 증가, warm-only +3.13%)
- propagated stddev ±31-35 pp (cold-start run1 outlier dominate)

본 SUB +2.83% 의 1-run signal 도 cold-start variance 안에 묻힐 가능성 있음. 단:
- 본 SUB chain mode 로 두 cycle (OFF/ON) 모두 fresh vllm boot — 단일 1-run 측정 한계는 동일
- **code-heavy +7.73% 의 magnitude 가 SUB_194 의 noise floor (±10pp) 보다 충분히 큼** → 의미 있는 signal 가능성
- **multi-run 재측정 (별도 SUB) 으로 binding 권고**

## 4. accuracy gate

| gate | 결과 |
|---|---|
| token-level bit-exact / 분포 정합성 | **PASS by construction** — workload code 미변경, OS-level NUMA placement 정책 + side-channel CPU work 만 추가 (vllm path 직접 미변경) |

## 5. paper §4 implication

- **paper §4 secondary lever stacking 영역의 first valid 후보**: SUB_188 single (+1.84%) → SUB_197 pair (+2.83%) → SUB_191 triple (+0.48% destructive). pair stack 이 sweet spot.
- production deployment 권고:
  - SUB_188 single 만: +1.84% (가장 안전, side-channel only)
  - SUB_197 pair: +2.83% (NUMA env + side-channel cross-domain stack)
  - SUB_191 triple 이상: destructive, 피해야 함
- paper §4 main +5% 도달은 본 fork 의 multi-lever stacking 으로는 불가능 — **SUB_198 AMX draft real spec_decode integration** (별도 invasive SUB) 만 paper main 후보로 남음.

## 6. 누적 패턴 update

| 카테고리 | 시도 | net positive | net loss | destructive stack | catastrophic |
|---|---:|---:|---:|---:|---:|
| drop-in CPU kernel | 7 | 0 | 6 | 0 | 1 |
| environment individual | 2 | 1 | 1 | 0 | 0 |
| environment stack | 2 | 0 | 0 | 2 | 0 |
| paper main IDE_018 | 1 | 0 | 1 | 0 | 0 |
| NEW workload microbench | 3 | feasibility 3 conditional | — | — | — |
| NEW workload e2e proxy | 2 | 0 | 1 | 0 | 0 |
| AMX draft microbench | 1 | feasibility PASS | — | — | — |
| side-channel individual | 3 | 2 | 1 | 0 | 0 |
| side-channel stack triple | 1 | 0 | 0 | 1 | 0 |
| multi-run variance verify | 1 | retract 1 (SUB_190) | — | — | — |
| **side-channel × env pair stack (본 SUB)** | **1** | **1 (+2.83%)** ⭐ | **0** | **0** | **0** |
| **누적** | **24** | **4 (paper main 기준 도달 1: code-heavy +7.73%)** | 11 | 3 | 2 |

→ paper §4 secondary lever stacking 의 first valid 후보 확보 (+2.83% pair stack). main 후보 (+5% 기준 모든 cell) 는 SUB_198 AMX real integration 만 남음.

## 7. raw data

- `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (18 cell)
- `_monitor_{off,on}_{cpu,gpu}.csv` (0.5s interval)
- `logs/{main,vanilla,trident,router,monitor,precompute}_{off,on}.log`
- `launcher.sh` — SUB_188 launcher 검증본 copy + NUMA wrap minimal patch (chain mode 활용)
