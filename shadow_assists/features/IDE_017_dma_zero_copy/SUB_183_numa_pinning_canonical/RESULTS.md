# SUB_183 — IDE_017 NUMA-aware vllm instance pinning canonical 500p e2e

> **parent**: TSK (IDE_017 NUMA-aware data plane).
> **scope**: 2026-05-27 11:51 ~ 11:57 KST (~6 분 wall, OFF + ON chained).
> **status**: 완료 — NUMA dual-node 환경에서 vanilla=node0 / trident=node1 pinning 적용 + canonical 500p × 3 mix × 3 scenario × OFF/ON 측정 완료.

---

## 0. 두괄식 — environment-level lever **noise positive** (3-mix avg AGSD +1.54%)

3-mix avg AGSD: **OFF 4,401 tps → ON 4,469 tps = +1.54%** (noise positive, ON 우세 일관).

| 측정 | OFF (taskset only) | ON (numactl + taskset) | Δ |
|---|---:|---:|---:|
| **3-mix avg AGSD** | **4,401** | **4,469** | **+1.54%** |
| balanced AGSD | 4,104 | 4,159 | +1.34% |
| sonnet-heavy AGSD | 4,443 | 4,586 | +3.22% |
| code-heavy AGSD | 4,657 | 4,663 | +0.12% |
| paper §4 target | — | — | +5-10% (미달) |

→ NUMA-aware pinning = **noise positive** (SUB_182 isolation 의 −0.39% noise floor 와 동일 magnitude, 방향만 반대). magnitude (<2%) 가 paper main lever 자격 기준 (+5%) 보다 작아 fail. drop-in 7 fail + NEW workload 2 conditional + environment 1 noise (SUB_182) + environment 1 small positive (본 SUB) = **lever 9 시도 중 net positive 0**.

---

## 1. NUMA topology dump

| 항목 | 값 |
|---|---|
| NUMA nodes | **2** (node0, node1) |
| node0 phys cores | 0-55 (56 cores) |
| node0 HT siblings | 112-167 (사용 안 함) |
| node0 memory | 1,031,036 MB (free 433,809 MB) |
| node1 phys cores | 56-111 (56 cores) |
| node1 HT siblings | 168-223 (사용 안 함) |
| node1 memory | 1,032,148 MB (free 94,481 MB) |
| node distance 0↔1 | **21** (local 10, remote 2.1×) |

NUMA dual-node 명확. cross-NUMA traffic 의 2.1× latency penalty 가 lever 의 이론적 근거.

---

## 2. 적용 셋업 (OFF/ON 차이)

| 항목 | OFF | ON |
|---|---|---|
| vanilla 코어 | `taskset -c 0-49` (50 phys, node0 영역) | `numactl --membind=0 --cpunodebind=0 taskset -c 0-49` |
| trident 코어 | `taskset -c 56-105` (50 phys, node1 영역) | `numactl --membind=1 --cpunodebind=1 taskset -c 56-105` |
| router/monitor | `taskset -c 0-49,56-105` (union) | `taskset -c 0-49,56-105` (union) |
| 총 phys cores | **100** | **100** |
| HT siblings (112-223) | 미사용 | 미사용 |
| 메모리 정책 | OS default (interleave/touch-first 혼합) | **strict bind to node** |
| GPU 매핑 | vanilla=0-3 / trident=4-7 | vanilla=0-3 / trident=4-7 |

ON 의 lever 효과: vanilla 의 모든 host allocation (KV staging, tokenizer buffer, hf_offline cache, vllm Python heap 일부) 이 node0 메모리에 강제 + 그 위에서 도는 thread 도 node0. 마찬가지로 trident 가 전체 node1. cross-NUMA touch 없음.

---

## 3. throughput delta — 9 cell 상세

| mix | scen | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1,639.0 | 1,607.5 | −1.92% |
| balanced | trident-only | 1,644.9 | 1,633.2 | −0.72% |
| balanced | **agsd-gated** | **4,103.6** | **4,158.7** | **+1.34%** |
| sonnet-heavy | vanilla-only | 2,063.9 | 2,114.9 | +2.47% |
| sonnet-heavy | trident-only | 3,341.4 | 3,434.4 | +2.78% |
| sonnet-heavy | **agsd-gated** | **4,443.3** | **4,586.3** | **+3.22%** |
| code-heavy | vanilla-only | 1,907.0 | 1,934.7 | +1.45% |
| code-heavy | trident-only | 3,101.6 | 3,066.1 | −1.14% |
| code-heavy | **agsd-gated** | **4,657.2** | **4,662.6** | **+0.12%** |
| **3-mix avg vanilla-only** | — | **1,870.0** | **1,885.7** | **+0.84%** |
| **3-mix avg trident-only** | — | **2,696.0** | **2,711.2** | **+0.57%** |
| **3-mix avg agsd-gated** | — | **4,401.4** | **4,469.2** | **+1.54%** |

### 3.1 pattern observation

1. **sonnet-heavy 가 가장 큰 양적 응답** (+2.47 ~ +3.22% 일관): suffix decoding 의 host-side hot path (suffix tree lookup, prompt embedding cache) 가 NUMA-local memory 에서 benefit. AGSD +3.22% 가 9-cell 중 max.
2. **balanced vanilla/trident 둘 다 small regression** (−1.92%, −0.72%): NUMA bind 의 strict 정책이 일부 batched prefill burst 에서 node-local memory 부족 (특히 node1 의 free 94 GB) 으로 marginal 손해. AGSD 는 +1.34% 로 net positive 회복.
3. **code-heavy 가 거의 무반응** (+0.12% AGSD): ngram-style draft 의 GPU-dominant 특성 + 짧은 prompt 가 cross-NUMA traffic 자체가 적어 lever 무력화.
4. **scenario 별 magnitude**: vanilla-only +0.84% < trident-only +0.57% < agsd-gated +1.54%. AGSD 가 가장 큰 양적 효과 — 두 instance 가 동시에 자원 경쟁할 때 NUMA bind 가 가장 의미 있음 (concurrent dual-instance 시나리오에서 cross-NUMA traffic 이 최대).

### 3.2 p50 latency

| mix | OFF p50 (s) | ON p50 (s) | Δ |
|---|---:|---:|---:|
| balanced agsd | 0.183 | 0.177 | −3.25% |
| sonnet-heavy agsd | 0.167 | 0.164 | −1.76% |
| code-heavy agsd | 0.163 | 0.163 | +0.26% |

p50 latency 도 small positive (−1 ~ −3%) 방향성, throughput 결과와 일관.

---

## 4. accuracy gate

| gate | 결과 |
|---|---|
| token-level / 분포 정합성 | **PASS by construction** — workload code 미변경. `numactl --membind` / `--cpunodebind` 은 OS-level memory placement + scheduling 정책만 변경. sampling/logits/output 모두 동일 path |

---

## 5. paper §4 영향

| pattern | 누적 | 본 SUB |
|---|---|---|
| drop-in CPU kernel replacement net positive | 0 / 7 | — |
| NEW workload conditional accept | 2 / 2 | — |
| NEW workload integration net positive | 0 / 1 (SUB_181 fail) | — |
| environment-level (OS isolation) | 0 / 1 (SUB_182 −0.39%) | — |
| **environment-level (NUMA bind)** | — | **0 / 1 (noise positive +1.54%)** |

**paper §4 의 lever 자격**: 없음 (noise positive, +5% 기준 미달). 단 SUB_182 와 SUB_183 둘 다 noise floor 내 이지만 NUMA bind 가 **방향성 일관 positive** (9/9 cell 중 6 cell positive, 3 cell minor negative) 라는 점은 "environment-level lever 가 마이크로 환경에 따라 +1~2% 방향성을 보일 수 있다" 의 supporting datapoint. 후속 SUB 의 default 환경 채택 candidate.

honest report 형태:
> "NUMA-aware dual-instance pinning (vanilla → node0 / trident → node1, strict membind+cpunodebind) 은 본 fork canonical 500p × 3 mix workload 에 small noise positive 영향 (3-mix avg AGSD +1.54%). environment-level lever 자체 paper main lever 자격 없음. 단 방향성 일관 positive 라 후속 SUB baseline default 채택 가능."

---

## 6. 다음 step (별도 SUB)

- **SUB_182 (isolation) + SUB_183 (NUMA bind)** 둘 다 noise floor 내 → 환경 lever 단일 변경으로는 paper main lever 불가 확정.
- **stacking 실험 후보**: SUB_182 ON + SUB_183 ON 동시 적용 (cgroup + hugepages + taskset + numactl). 합산 +1~2% 이 stack 시 보존 또는 cancel 검증.
- **SUB_184 (다음 SUB)**: IDE_019 TSK_036 AMX draft head 또는 IDE_021 신규 NEW workload 후보 등 GPU-relative cost gap 이 더 favorable 한 lever 측정.

---

## 7. raw data

- `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (3 mix × 2 mode = 6 file, 각 3 scenario = 18 cell)
- `aggregate.json` — per-cell + 3-mix avg snapshot
- `_monitor_{off,on}_{cpu,gpu}.csv` (0.5s interval, ~3 분 × 2)
- `logs/{main,vanilla,trident,router,monitor,chain}_{off,on}.log`
- `logs/numa_{off,on}.txt` — 실제 적용된 NUMA wrap + numactl --hardware dump
- `launcher.sh` — OFF/ON chained launcher with `numactl --membind --cpunodebind` (ON) + taskset core split (양 mode 공통)
- `chain.sh` — OFF → cleanup → ON sequential driver
