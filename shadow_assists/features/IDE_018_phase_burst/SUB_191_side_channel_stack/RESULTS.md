# SUB_191 — side-channel triple-stack (SUB_183 NUMA + SUB_188 softmax + SUB_190 tokenize) canonical 500p e2e

> **parent**: IDE_018 / IDE_017 stack (SUB_183 + SUB_188 + SUB_190 superposition test).
> **scope**: 2026-05-27 14:00 ~ 14:04 KST (~4 min wall, OFF + ON chained).
> **status**: ✅ 완료 — **destructive interference** (SUB_186 패턴 재현).

---

## 0. 두괄식 — triple-stack linear superposition **REJECT** (destructive interference)

| 측정 | 값 |
|---|---:|
| 3-mix avg AGSD OFF | **4,395.1 tps** |
| 3-mix avg AGSD ON (3 lever stack) | **4,416.0 tps** |
| **actual Δ** | **+0.48%** |
| linear sum prediction (SUB_183 +1.54% + SUB_188 +1.84% + SUB_190 +1.66%) | **+5.04%** |
| **residual** | **−4.56 pp** (massive destructive interference) |
| paper main 기준 | +5% — 미달 |

→ SUB_186 (SUB_182+SUB_183 stack −0.03% vs +1.15% prediction) 와 동일 패턴. **environment/side-channel lever 가 stack 시 destructive interference** — paper §4 의 stack lever 자격 없음.

---

## 1. ON 모드 stack 구성

| lever | spec | duty cycle |
|---|---|---|
| **NUMA pin** (SUB_183) | vanilla node0 / trident node1 numactl membind+cpunodebind | n/a |
| **softmax precompute** (SUB_188) | 16 OMP × cores 80-87 / batch=[32,152064] FP32 / 100ms cycle | ~2.3% |
| **tokenize worker** (SUB_190) | 16 OMP × cores 88-95 / BPE-style hash+lookup / 20ms cycle | ~4.5% |
| **합산 duty cycle** | side-channel CPU work | ~6.8% (cores 80-95 split) |

cores 80-87 (softmax) + 88-95 (tokenize) 분리 = 두 side-channel 간 cache contention 직접 회피.

## 2. 9-cell 상세

| mix | scen | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1,601.3 | 1,642.4 | +2.57% |
| balanced | trident-only | 1,644.2 | 1,611.6 | −1.98% |
| balanced | **agsd-gated** | **4,113.6** | **4,131.7** | **+0.44%** |
| sonnet | vanilla-only | 2,029.5 | 2,088.1 | +2.89% |
| sonnet | trident-only | 3,292.3 | 3,358.0 | +1.99% |
| sonnet | **agsd-gated** | **4,471.0** | **4,448.5** | **−0.50%** |
| code | vanilla-only | 1,932.1 | 1,937.9 | +0.30% |
| code | trident-only | 2,996.8 | 3,024.5 | +0.92% |
| code | **agsd-gated** | **4,600.6** | **4,667.9** | **+1.46%** |
| **3-mix avg AGSD** | — | **4,395.1** | **4,416.0** | **+0.48%** |

### 핵심 observation

1. **vanilla-only mean +1.92%** (SUB_188/190 softmax+tokenize 의 individual effect 합산 ~3.5% 의 절반만 surface)
2. **trident-only mean +0.31%** (SUB_188/190 individual +0.5~+3.5% range 의 가장 낮은 수준 surface)
3. **agsd-gated mean +0.47%** (SUB_188 +1.84% + SUB_190 +1.66% + SUB_183 +1.54% = +5.04% prediction 대비 1/10 만 surface)
4. sonnet agsd 만 **−0.50%** (interference 직접 surface)

## 3. destructive interference root cause 추정

| factor | 영향 |
|---|---|
| 두 side-channel worker (softmax + tokenize) 가 cores 80-95 공유 → L3 cache pressure 합산 | cache pressure ↑ → vllm 의 PCIe DMA stage delay |
| NUMA pin (vanilla node0 / trident node1) + side-channel (cores 80-95 = node1) | trident path 가 side-channel 과 same NUMA node → contention 직접 발생 |
| Linux scheduler tick (1000 Hz) × 32 worker thread (16+16) → migration / IPI frequency ↑ | vllm worker thread 의 wake-up latency 증가 |
| pinned alloc + cudaMemcpy stream — side-channel 의 memory bandwidth share | DMA queue depth 증가 시 GPU verify path 의 launch overhead |

→ **side-channel 개별 lever 의 1-2% gain 은 critical path 의 cache/scheduler/bus 의 spare margin** 활용. stack 시 spare margin 자체가 cumulative하지 않음 (sub-additive saturation).

## 4. paper §4 implication

- SUB_186 environment stack destructive (−0.03% vs +1.15%) + 본 SUB_191 side-channel stack destructive (+0.48% vs +5.04%) → **lever stacking 가설 일관 reject**.
- **secondary lever 영역 max 가 individual lever 의 best (+1.84%, SUB_188)** — stacking 으로 paper main 도달 불가능 확정.
- paper §4 main 후보는 microbench feasibility 단계 (SUB_187 AMX draft head 0.5 ms ⭐) 만 남음. real vllm spec_decode integration (별도 invasive SUB) 으로 +5% 도달 가능 여부 검증 필요.

## 5. 누적 패턴

| 카테고리 | 시도 | net positive | net loss | catastrophic |
|---|---:|---:|---:|---:|
| drop-in CPU kernel | 7 | 0 | 6 | 1 (SUB_181 −94%) |
| environment-level individual | 2 | 1 (SUB_183 +1.54%) | 1 | 0 |
| environment-level stack | 2 | 0 (SUB_186, SUB_191 destructive) | 2 | 0 |
| paper main IDE_018 | 1 | 0 (SUB_184 reject) | 1 | 0 |
| NEW workload microbench | 3 (178, 180, 187) | feasibility 3 conditional | — | — |
| NEW workload e2e proxy | 1 (SUB_185) | 0 (+0.18% noise) | 0 | TTFT regression |
| side-channel | 3 (188, 189, 190) | 2 (+1.84/+1.66%) | 1 (SUB_189 −0.82%) | 0 |
| **본 SUB (triple-stack)** | 1 | 0 (+0.48% destructive) | 0 | 0 |
| **누적** | **17** | **3** | **11** | **2** |

paper main 기준 (+5%) 도달 lever 0 / 17 시도.

## 6. raw data

- `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (18 cell)
- `_monitor_{off,on}_{cpu,gpu}.csv`
- `logs/{main,vanilla,trident,router,monitor,chain}_{off,on}.log`
- `launcher.sh` (NUMA + softmax + tokenize 동시 fire)
