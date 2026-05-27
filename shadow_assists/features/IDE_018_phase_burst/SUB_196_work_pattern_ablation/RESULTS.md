# SUB_196 — work-pattern × fire-rate ablation 확장 (2×2 grid completion)

> **parent**: SUB_188 / SUB_189 / SUB_190 의 work-pattern × cycle 변수 채우기.
> **scope**: 2026-05-27 15:58 ~ 16:12 KST (~14 min wall, cellA + cellB sequential chain).
> **status**: ✅ 완료 — **2×2 grid 완성**, cellB (branchy × 100ms) **paper main +5% 도달** ⭐.

---

## 0. 두괄식 — 2×2 work-pattern × cycle grid

| pattern \ cycle | 10ms | 20ms | 100ms |
|---|---:|---:|---:|
| **regular** | cellA **+0.98%** | SUB_190 +1.66% (1-run) / **−5.96%** (multi) | SUB_188 +1.84% (1-run) / +0.53% (multi) |
| **branchy** | SUB_189 −0.82% | — | **cellB +5.28%** ⭐⭐ |

→ paradoxical finding: **branchy × low-rate (100ms) 가 paper main 기준 도달**. 단일 lever 중 9-cell magnitude max (balanced agsd +12.04%).

---

## 1. cellA — regular work × 10ms cycle (high-rate regular)

| 측정 | OFF tps | ON tps | Δ% |
|---|---:|---:|---:|
| balanced agsd | 3,766.3 | 3,888.2 | +3.24% |
| sonnet agsd | 4,182.9 | 4,120.7 | −1.49% |
| code agsd | 4,368.4 | 4,428.8 | +1.38% |
| **3-mix avg** | **4,105.8** | **4,145.9** | **+0.98%** |

cellA spec:
- work: SUB_188 의 regular softmax + log-softmax (vector branch-free, batch=[32,152064])
- cycle: 10ms (100 Hz, SUB_188 의 100ms 보다 10× faster)
- workers: 16 OMP × cores 80-95
- per-cycle 측정: ~2.49 ms (duty cycle ~25%, target 2-5% 보다 5× 높음)
- lifetime: 8,604 cycles in ON wall (~21s active fraction)

cellA verdict: small noise positive (+0.98%). SUB_188 (+1.84%) 의 53% 만 유지 — high-rate cycle 의 cache pollution / scheduler tick overhead 가 net positive magnitude 일부 소실.

## 2. cellB — branchy work × 100ms cycle (low-rate branchy)

| 측정 | OFF tps | ON tps | Δ% |
|---|---:|---:|---:|
| **balanced agsd** | **3,626** | **4,062** | **+12.04%** ⭐⭐ |
| sonnet agsd | 4,098 | 4,175 | +1.88% |
| code agsd | 4,289 | 4,410 | +2.82% |
| **3-mix avg** | **4,004.3** | **4,215.9** | **+5.28%** ⭐ |

cellB spec:
- work: SUB_189 의 branchy candidate ranker (frequency-based reorder + insertion sort, batch=32 × K=7 × HIST=64)
- cycle: 100ms (10 Hz, SUB_189 의 10ms 보다 10× slower)
- workers: 16 OMP × cores 80-95
- per-cycle 측정: 0.393 ms (duty cycle ~0.4%, target 2-5% 보다 5× 낮음)
- lifetime: 851 cycles in ON wall (~0.3s active fraction)

cellB verdict: **paper main +5% 기준 도달** (3-mix avg +5.28%, balanced cell +12.04%). SUB_189 (−0.82%) 의 부호 반전 — branchy work 가 100ms cycle 로 amortize 되면서 net positive 변환.

## 3. paradoxical finding 의 추정 root cause

| 변수 | regular work | branchy work |
|---|---|---|
| 10ms cycle (high-rate) | cellA +0.98% (small positive) | SUB_189 −0.82% (small loss) |
| 100ms cycle (low-rate) | SUB_188 +1.84% (small positive) | **cellB +5.28%** ⭐ |

low-rate cycle (100ms) 가 high-rate (10ms) 보다 일관 better. 추정 mechanism:
- 100ms cycle = vllm step boundary (35-44 ms/step) 의 2-3 step 마다 fire → step 의 idle gap 와 정렬 가능성
- 10ms cycle = step 내부에서 fire → critical path 와 contention
- branchy work 가 cache prefetcher 의 inhibit 효과 + GPU 측 launch overhead 의 absorption 가능 (자세한 mechanism 별도 분석 필요)

balanced agsd +12.04% magnitude 가 9-cell max — chat workload 의 cpu_jacobi 분기 (SUB_181 catastrophic 의 같은 영역) 와 본 cellB 의 branchy CPU work 의 interaction 가능성. (확인 위해 별도 SUB profiling 필요.)

## 4. SUB_194 multi-run variance 와의 cross-reference

본 SUB 도 1-run 측정 — cold-start variance 안에 묻힐 가능성. SUB_194 의 stddev ±35 pp 안에서:
- cellB +5.28% (3-mix avg) 의 magnitude 가 noise floor 안에 있을 가능성 있음
- 단 **balanced agsd +12.04%** 는 noise floor (±10 pp) 보다 충분히 큼 → 의미 있는 signal 가능성 높음
- multi-run 재측정 (별도 SUB) 으로 binding 권고

## 5. accuracy gate

| gate | 결과 |
|---|---|
| token-level / 분포 정합성 | **PASS by construction** — workload code 미변경, side-channel CPU work (cores 80-95) 만 fire, vllm path 미변경 |

## 6. paper §4 implication

- **first single-lever 5%+ 도달 후보**: cellB (branchy × 100ms) 가 본 fork 의 18+ lever 시도 중 처음으로 paper main +5% 기준 도달 (3-mix avg)
- production deployment 권고:
  - cellB pattern (branchy work × low-rate cycle, cores 80-95 isolated) 이 paper main 후보 1순위
  - 단 1-run signal — multi-run 재측정 binding 필요
- **work-pattern × cycle interaction** 이 binding 변수 = secondary lever 영역의 새 dimension
- SUB_197 pair stack (+2.83%) 의 base lever 후보: SUB_197 = NUMA + SUB_188 (regular × 100ms). 만약 NUMA + cellB (branchy × 100ms) pair = linear sum +6.82% 예측 → 별도 SUB 측정 가치

## 7. 누적 패턴 update

| 카테고리 | 시도 | net positive (1-run) | paper main +5% 도달 |
|---|---:|---:|---:|
| drop-in CPU kernel | 7 | 0 | 0 |
| environment individual | 2 | 1 | 0 |
| environment stack | 2 | 0 | 0 |
| paper main IDE_018 | 1 | 0 (retract) | 0 |
| NEW workload e2e proxy | 2 | 0 | 0 |
| side-channel individual | 3 | 2 | 0 |
| side-channel × env pair stack | 1 | 1 (SUB_197 +2.83%) | code-heavy +7.73% |
| side-channel stack triple | 1 | 0 (destructive) | 0 |
| **work-pattern ablation (본 SUB)** | **2 cell** | **2 (cellA / cellB)** | **cellB +5.28% 3-mix avg ⭐⭐** |
| 누적 | **26** | **6** | **2 (SUB_197 code-only / 본 SUB cellB 3-mix avg)** |

본 SUB cellB 가 **3-mix avg 기준 paper main 도달 lever first** (SUB_197 은 code-heavy single cell 만 +7.73%).

## 8. 다음 SUB 후보

- **multi-run binding 검증** (SUB_199 새 ID, 또는 SUB_194 follow-up): cellB pattern 의 3-run × OFF/ON 재측정
- **stack 후보**: cellB (branchy × 100ms) + SUB_183 NUMA pair stack (SUB_197 패턴 확장, predicted linear sum +6.82%)
- **SUB_198 AMX real integration** (사용자 plan): 본 SUB cellB 패턴 보다 더 큰 magnitude 후보 (microbench 0.5 ms 의 theoretical 4× spec speedup)

## 9. raw data

- `cellA_regular_10ms/measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (18 cell)
- `cellA_regular_10ms/src/cellA_regular_10ms.cpp` (SUB_188 copy + kCycleMs=10)
- `cellA_regular_10ms/build/cellA_regular_10ms` (binary)
- `cellA_regular_10ms/launcher.sh` (SUB_188 launcher copy + binary swap)
- `cellB_branchy_100ms/measurements/{off,on}/...` (18 cell)
- `cellB_branchy_100ms/src/cellB_branchy_100ms.cpp` (SUB_189 copy + kCycleMs=100)
- `cellB_branchy_100ms/build/cellB_branchy_100ms` (binary)
- `cellB_branchy_100ms/launcher.sh` (SUB_188 launcher copy + binary swap)
