# SUB_182 — IDE_020 / TSK_038 production isolation canonical 500p e2e

> **parent**: TSK_038 (IDE_020).
> **scope**: 2026-05-27 11:36 ~ 11:42 KST (~6 min wall, OFF + ON chained).
> **status**: ✅ 완료 — full isolation stack (cgroup v2 + hugepages + taskset) 모두 적용 성공 + canonical 500p × 3 mix × 3 scenario × OFF/ON 측정 완료.

---

## 0. 두괄식 — environment-level lever **noise floor** (drop-in 8번째 noise/fail)

3-mix avg AGSD: **OFF 4,308 tps → ON 4,291 tps = −0.39%** (noise floor).

| 측정 | OFF (default) | ON (isolated) | Δ |
|---|---:|---:|---:|
| **3-mix avg AGSD** | **4,308** | **4,291** | **−0.39%** ⚠ |
| balanced AGSD | 3,982 | 3,886 | −2.40% |
| sonnet AGSD | 4,390 | 4,411 | +0.48% |
| code AGSD | 4,552 | 4,575 | +0.52% |
| paper §4 target | — | — | +5-10% (실패) |

→ OS-level isolation alone = **noise floor**. Net positive lever 자격 없음. drop-in 7 fail + 본 SUB = **noise/fail 8번째 패턴**.

---

## 1. isolation 적용 상세 (ON mode)

| 항목 | 결과 | 비고 |
|---|---|---|
| cgroup v2 cpuset | **applied** | `/sys/fs/cgroup/sub182_on/cpuset.cpus.effective = 0-99` |
| hugepages | **applied** | `nr_hugepages = 4` (4 × 2 MB = 8 MB allocated) |
| taskset wrapper | **applied** | `taskset -c 0-99` 로 vllm + monitor + router wrap |
| VLLM_USE_HUGEPAGES | =1 | vllm 자체 respect 여부 unknown — best-effort hint |

세 levers 모두 적용 성공. 100-core 제약 + HT 시블링 (112-223) 미사용 정책 enforced.

---

## 2. throughput delta — 9 cell 상세

| mix | scen | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1,614 | 1,536 | **−4.81%** |
| balanced | trident-only | 1,660 | 1,627 | −2.01% |
| balanced | **agsd-gated** | **3,982** | **3,886** | **−2.40%** |
| sonnet | vanilla-only | 2,043 | 1,971 | −3.52% |
| sonnet | trident-only | 3,284 | 3,346 | +1.87% |
| sonnet | **agsd-gated** | **4,390** | **4,411** | **+0.48%** |
| code | vanilla-only | 1,924 | 1,886 | −1.98% |
| code | trident-only | 2,929 | 3,046 | **+3.98%** |
| code | **agsd-gated** | **4,552** | **4,575** | **+0.52%** |
| **3-mix avg AGSD** | — | **4,308** | **4,291** | **−0.39%** |

### 2.1 pattern observation

1. **vanilla-only 일관된 small regression (−2 ~ −5%)**: ON 의 taskset 100-core limit 이 OFF 의 unrestricted vllm (112 physical + 112 HT thread 가능) 보다 약간 손해. CPU-bound work (tokenization / detokenization / sampling) 가 100-core 에 묶이며 throughput marginal loss.
2. **trident-only mixed (±2%)**: spec decoding 의 cudagraph 가 GPU-dominant 라 CPU isolation 영향 minor.
3. **agsd-gated mixed (−2.4 ~ +0.5%)**: balanced 만 regression, sonnet/code 는 noise positive. 3-mix avg −0.39% = **noise floor 내**.

### 2.2 p50 latency

| mix | OFF p50 (s) | ON p50 (s) | Δ |
|---|---:|---:|---:|
| balanced agsd | 0.186 | 0.190 | +2.2% |
| sonnet agsd | 0.168 | 0.168 | 0.0% |
| code agsd | 0.167 | 0.163 | −2.4% |

→ p50 도 noise floor.

---

## 3. accuracy gate

| gate | 결과 |
|---|---|
| token-level / 분포 정합성 | **PASS by construction** — workload code 변경 없음. isolation 은 OS-level lever 만 적용 (스케줄링·메모리 page 정책 변경). sampling/logits/output 모두 동일 path |

---

## 4. paper §4 영향

| pattern | 누적 | 본 SUB |
|---|---|---|
| drop-in CPU kernel replacement net positive | 0 / 7 | — |
| NEW workload conditional accept | 2 / 2 | — |
| NEW workload integration net positive | 0 / 1 (SUB_181 fail) | — |
| **environment-level (OS isolation)** | — | **0 / 1 (noise floor)** |

**paper §4 의 lever 자격**: 없음. cgroup + hugepages + taskset 자체가 workload throughput 에 의미 있는 net positive 를 만들지 못함. 다만 **본 SUB 의 환경 자체는 valid production config** — 후속 SUB 에서 CPU-bound workload 가 actually saturate 되는 경우 (예: SUB_178 cold-KV 의 deep integration) 의 baseline 으로 사용 가능.

honest report 형태:
> "OS-level isolation alone (cgroup v2 cpuset 0-99 + 4× 2 MB hugepages + taskset wrapper) 은 본 fork canonical 500p × 3 mix workload 에 noise floor 영향만 (3-mix avg AGSD −0.39%). 환경 lever 자체 paper main lever 자격 없음."

---

## 5. 다음 step (별도 SUB)

- **isolation 자체는 backdrop**: 본 SUB 의 ON 환경 (cgroup + hugepages + taskset) 을 후속 SUB 들의 default 로 채택해 모든 CPU 경로 측정의 reproducibility 강화 가능.
- **SUB_183 (다음 SUB)**: IDE_017 NUMA-aware data plane 또는 IDE_019 TSK_036 AMX draft head 등 새 lever 측정.

---

## 6. raw data

- `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` (3 mix × 2 mode = 6 file, 각 3 scenario = 18 cell)
- `_monitor_{off,on}_{cpu,gpu}.csv` (0.5s interval, ~3 min × 2)
- `logs/{main,vanilla,trident,router,monitor,chain}_{off,on}.log`
- `logs/isolation_{off,on}.txt` — 실제 적용된 isolation 상태 dump
- `launcher.sh` — OFF/ON chained launcher with cgroup v2 / hugepages / taskset best-effort apply
