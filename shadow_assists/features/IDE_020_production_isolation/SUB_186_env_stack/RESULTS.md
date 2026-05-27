# SUB_186 — IDE_020 environment stack (SUB_182 isolation + SUB_183 NUMA pin) canonical 500p e2e

> **parent**: IDE_020 (TSK_038 isolation 본문 흡수 + IDE_017 NUMA backdrop). SUB_182 (isolation) + SUB_183 (NUMA pin) stacking 검증.
> **scope**: 2026-05-27 12:50 ~ 12:56 KST (~6 분 wall, OFF + ON chained).
> **status**: ✅ 완료 — 4 environment levers (cgroup v2 + hugepages + taskset + NUMA bind) 동시 적용 성공 + canonical 500p × 3 mix × 3 scenario × OFF/ON 측정 완료.

---

## 0. 두괄식 — environment stack **destructive interference** (linear superposition 위배)

3-mix avg AGSD: **OFF 4,363.5 tps → ON 4,362.4 tps = −0.03%** (완전한 noise floor, 두 individual lever 모두 무력화).

| 측정 | OFF (default) | ON (full stack) | Δ |
|---|---:|---:|---:|
| **3-mix avg AGSD** | **4,363.5** | **4,362.4** | **−0.03%** |
| balanced AGSD | 4,054.3 | 4,062.1 | +0.19% |
| sonnet AGSD | 4,439.2 | 4,406.0 | −0.75% |
| code AGSD | 4,597.0 | 4,619.0 | +0.48% |

| Lever 별 측정 | agsd-gated Δ% |
|---|---:|
| SUB_182 isolation alone (cgroup + hugepages + taskset) | **−0.39%** |
| SUB_183 NUMA pin alone (numactl --membind+cpunodebind) | **+1.54%** |
| **linear sum 예측** (orthogonal 가정) | **+1.15%** |
| **SUB_186 stacked, measured** | **−0.03%** |
| **residual (measured − predicted)** | **−1.18 pp** |
| verdict | **interference / saturation (sub-linear)** |

→ SUB_183 의 +1.54% 양적 효과가 SUB_182 stack 위에 올렸을 때 **완전 소실**. 두 environment lever 가 orthogonal 이라는 가설 **reject**. NUMA bind 효과의 root cause (concurrent dual-instance 의 cross-NUMA traffic 감소) 가 isolation stack (cgroup cpuset 0-99 + taskset core split) 환경에서 이미 부분적으로 흡수되거나 cancel 되는 것으로 추정.

---

## 1. isolation stack 적용 상세 (ON mode, 4 lever 모두 active)

| Lever | 출처 | ON 적용 상태 | 검증 방법 |
|---|---|---|---|
| cgroup v2 cpuset | SUB_182 | **applied** ✓ | `/sys/fs/cgroup/sub186_on/cpuset.cpus.effective = 0-99` |
| hugepages | SUB_182 | **applied** ✓ | `nr_hugepages = 4` (4 × 2 MB = 8 MB allocated) |
| VLLM_USE_HUGEPAGES env | SUB_182 | **applied** ✓ | env=1 export |
| taskset core split | SUB_183 (+SUB_182 100 core) | **applied** ✓ | vanilla 0-49 / trident 56-105 / mon+router 0-49,56-105 union |
| NUMA membind | SUB_183 | **applied** ✓ | vanilla `--membind=0 --cpunodebind=0` / trident `--membind=1 --cpunodebind=1` |

4 levers 모두 적용 성공. NUMA dual-node (node distance 21, remote 2.1× local), 100-core 제약 (HT 112-223 미사용) 둘 다 enforced.

```
mode=on
cgroup_ok=yes
cgroup_cpuset.cpus.effective=0-99
hugepages_ok=yes (alloc=4)
VLLM_USE_HUGEPAGES=1
numa_nodes=2  numa_applied=yes  taskset_applied=yes
vanilla_wrap='numactl --membind=0 --cpunodebind=0 taskset -c 0-49'
trident_wrap='numactl --membind=1 --cpunodebind=1 taskset -c 56-105'
monitor_wrap='taskset -c 0-49,56-105'
router_wrap='taskset -c 0-49,56-105'
```

OFF mode 는 default (no isolation, no NUMA pin, no taskset, no hugepages).

---

## 2. throughput delta — 9 cell 상세

| mix | scenario | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1,585.7 | 1,580.9 | −0.30% |
| balanced | trident-only | 1,602.5 | 1,625.5 | +1.43% |
| balanced | **agsd-gated** | **4,054.3** | **4,062.1** | **+0.19%** |
| sonnet-heavy | vanilla-only | 2,045.1 | 2,023.7 | −1.05% |
| sonnet-heavy | trident-only | 3,332.0 | 3,351.9 | +0.60% |
| sonnet-heavy | **agsd-gated** | **4,439.2** | **4,406.0** | **−0.75%** |
| code-heavy | vanilla-only | 1,947.8 | 1,871.4 | **−3.92%** |
| code-heavy | trident-only | 2,990.9 | 3,049.0 | +1.94% |
| code-heavy | **agsd-gated** | **4,597.0** | **4,619.0** | **+0.48%** |
| **3-mix avg vanilla-only** | — | **1,859.5** | **1,825.3** | **−1.84%** |
| **3-mix avg trident-only** | — | **2,641.8** | **2,675.4** | **+1.27%** |
| **3-mix avg agsd-gated** | — | **4,363.5** | **4,362.4** | **−0.03%** |

### 2.1 pattern observation

1. **agsd-gated 3-mix avg = −0.03%** (완전 noise floor). 9 cell sign 도 흩어짐 (3 positive / 1 negative / 1 ≈0 amongst AGSD; vanilla 3 negative; trident 3 positive).
2. **vanilla-only 3-mix avg −1.84%**: SUB_182 의 vanilla regression 패턴 (−3.5% 평균) 와 SUB_183 의 vanilla +0.84% 가 합쳐져 중간 regression. code-heavy −3.92% 가 dominant — single-instance code workload 에서 NUMA strict-bind + cgroup limit 동시 적용 시 batched prefill burst 압박.
3. **trident-only 3-mix avg +1.27%**: spec decoding (suffix decoding) 의 GPU-dominant 특성 + NUMA local memory 결합으로 양적 효과. SUB_182 (+1.28% trident avg) + SUB_183 (+0.57% trident avg) 와 비교 시 SUB_182 와 거의 동일 — **SUB_183 의 trident 효과 +0.57% 만큼이 saturation 으로 소실**.
4. **scenario 간 magnitude 가 SUB_183 패턴 (vanilla < trident < agsd) 을 뒤집음** — 본 SUB 는 agsd 가 최저 magnitude. 두 lever 간섭이 가장 큰 영역이 concurrent dual-instance (agsd-gated) 임을 시사.

### 2.2 p50 latency (agsd-gated)

| mix | OFF p50 (s) | ON p50 (s) | Δ |
|---|---:|---:|---:|
| balanced agsd | 0.183 | 0.175 | −4.25% |
| sonnet-heavy agsd | 0.169 | 0.168 | −0.24% |
| code-heavy agsd | 0.165 | 0.163 | −1.18% |

→ p50 latency 는 small positive (1-4% 감소) 일관성 — throughput 0 에 가까운 결과와 모순. p50 만 양적, total throughput 은 무변 = ON mode 가 latency variance 를 줄였지만 throughput 증가로 변환 안 됨 (queue depth saturation 추정).

---

## 3. superposition verdict (핵심 분석)

### 3.1 linear superposition 가설

SUB_182 (isolation alone) Δagsd = **−0.39%** + SUB_183 (NUMA alone) Δagsd = **+1.54%** → 두 lever 가 직각 (orthogonal) 이면 합산 = **+1.15%**.

본 SUB 측정 = **−0.03%**.

**residual = measured − predicted = −0.03 − (+1.15) = −1.18 pp**.

|residual| > 0.5 pp → **linear superposition 위배**. residual sign negative → **destructive interference / saturation**.

### 3.2 saturation mechanism 후보

(a) **NUMA bind 의 본 가치는 OS default scheduler 가 cross-NUMA migration 을 일으킬 때 발생**. SUB_182 의 taskset core split (vanilla 0-99 union, vanilla 0-99 monitor union) 자체가 이미 일부 NUMA locality 를 boost — vanilla 가 0-49 영역 (node0) 에 들 확률 ↑, trident 가 ~56+ (node1) 영역에 들 확률 ↑. 단 SUB_182 는 numactl 없이 *taskset only* 라 memory page 가 여전히 OS default 정책. 본 SUB 가 numactl 추가 후에도 magnitude 미발현 = memory policy 효과보다 cpuset/scheduler 효과가 dominant 였다는 datapoint.

(b) **trident 50 core × 2 instance = 100 core 가 NUMA node 당 50 core × 2 socket 의 절반** — 본 setup 에서 sufficient core utilization 못 채워 NUMA local 의 BW boost 가 한계 utilization. saturation.

(c) **hugepages 4 × 2 MB = 8 MB** 는 vllm 의 1.6 GB W matrix / 32 GB KV cache 대비 trivial — environment lever 자체가 KV/weight memory page 정책을 dominate 못 함.

(d) **balanced agsd −2.40% (SUB_182) → +0.19% (SUB_186)** = balanced 에서는 NUMA 추가가 SUB_182 regression 을 noise floor 로 복구 (+2.59 pp). sonnet/code 는 거의 동일. 즉 **balanced mix 에서는 두 lever 간 약간의 협력**, sonnet/code 에서는 SUB_183 효과 완전 소실 — workload-dependent interference.

### 3.3 verdict

**linear superposition holds**: **REJECT**.

**Direction**: destructive interference / saturation (sub-linear).

**Magnitude**: SUB_183 의 단독 +1.54% positive 가 stack 후 0% 부근으로 흡수 (≈ 100% destructive).

---

## 4. accuracy gate

| gate | 결과 |
|---|---|
| token-level / 분포 정합성 | **PASS by construction** — workload code 미변경. cgroup / hugepages / taskset / numactl 모두 OS-level scheduling+memory policy lever 만 변경. sampling/logits/output 모두 동일 path |

---

## 5. paper §4 영향

| pattern | 누적 (lever 시도 수) | 본 SUB |
|---|---|---|
| drop-in CPU kernel replacement net positive | 0 / 7 | — |
| NEW workload conditional accept | 2 / 2 | — |
| NEW workload integration net positive | 0 / 1 (SUB_181 fail) | — |
| environment-level single lever | 0 / 2 (SUB_182 −0.39% / SUB_183 +1.54%) | — |
| paper main lever (IDE_018 phase-burst) | 0 / 1 reject (SUB_184) | — |
| NEW workload e2e proxy (cold-KV long-ctx) | 0 / 1 (SUB_185 +0.18% noise) | — |
| **environment-level stack (cgroup + hugepages + taskset + NUMA)** | — | **0 / 1 (−0.03% destructive interference)** |

**paper §4 의 lever 자격**: 없음 — environment stack 도 net positive 미발생. 추가 발견: environment lever 간 **linear superposition 가정 false**. drop-in 7 fail + NEW workload 2 conditional + paper main 1 reject + cold-KV proxy 1 noise + environment 2 single + **environment 1 stack (본 SUB)** = **lever 12 시도 중 paper-bound net positive 0**.

### 5.1 §4 one-line implication

> "Environment-level levers (CPU isolation, NUMA pin, hugepages) are **non-additive on H100 dual-instance vLLM** — destructive interference between cgroup/taskset scheduling locality and explicit NUMA membind cancels SUB_183's +1.54% gain on top of SUB_182's −0.39% loss, yielding −0.03% stacked net (vs +1.15% linear prediction). Production isolation cannot be tuned by orthogonal-lever assumption."

### 5.2 후속 SUB 권고

- environment stack 단독 net positive 불가 확정. 별도 SUB 권고 안 함 — environment lever 영역 close.
- **environment lever 가 net positive 0 영역으로 confirm 됨** → paper main lever 후보는 *workload-side 변경* (real cold-KV vllm integration 별도 SUB) 만 남음. SUB_178 cold-KV 의 real integration 가 유일한 conditional 후보.
- saturation root-cause 추가 분석은 별도 study 영역 (cpuset vs numactl 의 scheduler/page policy interaction profile). paper §4 main path 영역 밖.

---

## 6. 비교 표 (SUB_182 / SUB_183 / SUB_186)

| 측정 | SUB_182 (isolation alone) | SUB_183 (NUMA alone) | **SUB_186 (stacked)** |
|---|---:|---:|---:|
| 3-mix avg agsd OFF tps | 4,308 | 4,401 | **4,363** |
| 3-mix avg agsd ON tps | 4,291 | 4,469 | **4,362** |
| 3-mix avg agsd Δ% | −0.39% | +1.54% | **−0.03%** |
| balanced agsd Δ% | −2.40% | +1.34% | +0.19% |
| sonnet agsd Δ% | +0.48% | +3.22% | −0.75% |
| code agsd Δ% | +0.52% | +0.12% | +0.48% |
| vanilla-only avg Δ% | −3.5% (avg) | +0.84% | −1.84% |
| trident-only avg Δ% | +1.3% (avg) | +0.57% | +1.27% |

→ trident-only avg 만 SUB_182 와 거의 동일 (+1.27 vs +1.28) — SUB_183 의 trident 효과 +0.57% 가 stack 위에서는 net contribution 0. agsd avg 가 −0.03% 로 SUB_183 의 +1.54% 가 완전 destruct.

---

## 7. raw data

- `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (3 mix × 2 mode = 6 file, 각 3 scenario = 18 cell)
- `aggregate.json` — per-cell + 3-mix avg + superposition residual snapshot
- `_monitor_{off,on}_{cpu,gpu}.csv` (0.5s interval, ~3 min × 2)
- `logs/{main,vanilla,trident,router,monitor,chain}_{off,on}.log`
- `logs/isolation_{off,on}.txt` — 4 lever 적용 verification dump + `numactl --hardware`
- `launcher.sh` — OFF/ON chained launcher with cgroup v2 + hugepages + taskset + numactl (4 lever stack)
- `chain.sh` — OFF → cleanup → ON sequential driver
- `aggregate.py` — per-cell tps Δ% + 3-mix avg + superposition residual vs SUB_182 −0.39% / SUB_183 +1.54%
