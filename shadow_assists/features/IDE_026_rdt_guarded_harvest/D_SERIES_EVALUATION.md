# IDE_026 D-시리즈 (A섹션) 전체 성능 평가 — 2026-06-15

> RDT-guarded CPU harvest 의 간섭 채널 특성화 12 실험 종합. 각 SUB 의 단일 출처는
> `SUB_*/MEASUREMENTS.md`, 본 문서는 통합 평가·정량 종합·논문 스토리.
> **측정 환경**: dgx-b200 호스트, 2× Xeon 8570(EMR, 224T), victim/aggressor CPU-only
> 마이크로벤치 (`src/victim_aggressor.c`) + 70B 실측판(SUB_220). resctrl·turbostat·
> bpftrace·numactl·accel-config 실측.

## 0. 한눈 결과표 (12 SUB)

| SUB | 채널/레버 | 핵심 수치 | 판정 |
|---|---|---|---|
| 220 | 전력/uncore freq | P1 −5.9% = P2(uncore pin) −5.9%, UncMHz~2400 전셀 | ✅ uncore 채널 negligible, pin 무효 |
| 221 | CDP code/data | unrestricted=data_restr=code_restr ~206 ns/load(변화없음) | ✅ CDP 무효(BW-bound) |
| 222 | OS 우선순위 | nice/IDLE ns/load 205.9 = default(무력), cpu.max50 89.2(복구) | ✅ priority 무력 |
| 223 | duty actuator | SIGSTOP 선형; **tpause +33% harvest효율(캐시상주, L2보존)** | ✅ tpause 우위 |
| 224 | SW token-bucket | budget→BW 158→59GB/s@25%, victim p99 49.4→25.8 | ✅ SW-MBA 작동 |
| 227 | IE 지표 | IE 0.78(nt)~2.65(basic8) = 3.4×; cldemote 1.86>basic 1.18>nt 0.78 | ✅ IE 규칙간 ≥2× |
| 228 | useful-work portfolio | AES −3.3%/SHA −3.6%/xz +0.7% vs synth +156% | ✅⭐ 실작업 near-free |
| 229 | constructive prefetch | helper ON p99~124 vs OFF~128 (노이즈 내) | ❌ 개선 없음 |
| 230 | runq latency 신호 | victim runq ~0 (596068/599000 @0µs) while ns/load 90→220 | ❌ 메모리간섭에 blind |
| 234 | TLB/hugepage | 예약 HugeTLB: normal 48.7 vs 2MB 38.8 = **−20%** | ✅ TLB 채널 ~20% |
| 235 | NUMA 드리프트 | local 58.0 vs remote 82.9 ns/load = **+43%** | ✅ 큰 채널 |
| 236 | DSA 셰이핑 | (제출구현+공유WQ 재구성 필요) | ⏸️ 보류 |

집계: **positive/측정완료 9 · 기각 2 · 보류 1** (236 DSA만 = portal EPERM·device-locked).

## 1. Enforcement ladder (보호력 서열) — 정량 확정

harvest 가 serving(victim)을 해치는 것을 막는 수단의 서열 (victim ns/load 회복 기준):

| 수단 | 효과 | harvest 비용 | 출처 |
|---|---|---|---|
| OS 우선순위 (nice/SCHED_IDLE) | **무력** (코어분리시 0) | — | SUB_222 |
| cpu.max (시간 throttle) | 유효 (baseline 복구) | harvest 50% 손실 | SUB_222 |
| **SW-MBA (token-bucket)** | 유효 (budget 비례 보호) | BW 비례 (compute 보존), RDT 불요 | SUB_224 |
| HW-MBA (RDT) | 유효 (BW 타깃) | compute 보존, 정밀 | SUB_220(MBA20 −5.9% vs MBA100 −15.8%) |
| NUMA-local 배치 | +43% 방지 | 없음(배치만) | SUB_235 |
| hugepage (serving WS) | TLB −20% | 없음 | SUB_234 |
| **tpause actuator** | duty 보호 + **캐시상주 harvest +33% 효율** | L2 보존 | SUB_223 |

→ **priority < cpu.max < SW-MBA ≈ HW-MBA (+ NUMA-local)**. priority 의 무력함이 RDT/BW
가드 필요성의 직접 논거. SW-MBA 가 RDT 하드웨어 없는 환경(VM/AMD/ARM)의 보편 대체재.

## 2. 간섭 채널 서열 — 어느 채널이 지배적인가

| 채널 | 크기 | RDT 차단 | 출처 |
|---|---|---|---|
| 메모리 BW (LLC/DRAM) | **지배적** (ns/load 2.3×, +156%) | MBA/SW-MBA ✅ | 220/222/224/227/228 |
| NUMA 드리프트 (UPI) | **큰** (+43%) | 배치/migration | SUB_235 |
| TLB/page-walk | **~20%** (예약 HugeTLB) | hugepage 완화 | SUB_234 |
| uncore freq | negligible (basic harvest) | — (uncore 자동 max) | SUB_220 |
| CPU-시간 경합 | 코어분리시 0 (runq~0) | (코어 분리로 이미 제거) | SUB_230 |

→ 가드 우선순위 = **메모리 BW 격리(MBA/SW-MBA) + NUMA-local 강제**. **L3 CAT/CDP 캐시파티션은 BW-bound harvest에 무효(SUB_221)**. TLB/uncore/runq 는
부차적. runq latency 는 메모리 간섭에 blind 라 SLO 신호로 부적합(메모리-지연 직접측정 필요).

## 3. 헤드라인 ⭐ — harvest 실용성 입증 (SUB_228)

| harvest 작업 | victim degr% |
|---|---|
| openssl AES / SHA256 (연산) | −3.3% / −3.6% (≈0, 약간 빠름) |
| xz 압축 (메모리+연산) | +0.7% (무시) |
| synthetic 순수 메모리 | +156% |

**실세계 유용작업(암호·해시·압축·AMX추론)은 연산-bound + 캐시상주라 serving 간섭 ≈0
→ RDT 가드 없이 유휴 CPU 자유 harvest 가능.** RDT 기계장치는 메모리-스트리밍 harvest
전용 안전망. 합성 worst-case(+156%)는 실제 대비 과도하게 비관적.

= **CLAUDE.md objective("CPU 활용률 극도로 끌어올려 idle 불허")의 실용적 달성 경로 확정.**

## 4. 정중한 harvest 커널 규칙 (SUB_227)

동일 유용작업량에서 간섭을 덜 만드는 작성법 (IE = 유용작업÷degr%):
- **cldemote**(소비후 라인 강등) IE 1.86 > basic 1.18 > **NT-store 0.78**(의외 최악,
  write-BW 포화). **cache-blocking**(WS≤CAT way) 이 간섭 최소.
- 권장 harvest 커널 = cldemote + cache-blocked + NT-store 회피.

## 5. 미완·후속

### DSA 세션 (2026-06-15 컨테이너) 진행
| SUB | 컨테이너 완료분 | 호스트 잔여 (미진) |
|---|---|---|
| 236 DSA | ✅ DSA ENQCMD 제출 성공 + 채널② dose-response(p99 +42%) | ① **MBA 비차단성 입증**(resctrl 미마운트 → 호스트) ② **read_buffers {96,48,24,12} 셰이핑 곡선**(accel-config 부재 → 호스트, 게이트 N=24 p99회복≥70%) ③ MBM 으로 DSA BW 가시화 |
| 239 FERRY | ✅ ns/step −29%, e2e −28% (worker 이득) | ④ **co-located victim 간섭 영향**(node0 iMC 부하 증가 가능 → resctrl 환경 후속) ⑤ host numactl 로 first-touch 배치 교차검증 ⑥ vLLM NEO/prefix 운반 경로 통합 |
| 246 IAA | — | ⑦ **IAA 디바이스 노출**(컨테이너 `--device=/dev/iax` 또는 호스트 IAA WQ user 구성) 후 전 측정 |

### 비-DSA 잔여
*(221 CDP·223 tpause·234 HugeTLB 는 2026-06-15 호스트 완료. D-시리즈 12/12 평가, 236 만 partial)*

### 미진 사항 한눈 (다음 작업자용)
1. **호스트 resctrl 측정** (236-①②③, 239-④): 컨테이너는 `/sys` ro·CAP_SYS_ADMIN 없음·
   accel-config 부재 → MBA/MBM/CAT·read_buffer 셰이핑은 전부 호스트에서. `HOST_RUNBOOK.md`
   절차 + `rdt_ctl.py` 재사용.
2. **IAA 디바이스 노출** (246): 현재 컨테이너 경계에서 차단. 디바이스만 들어오면 DSA
   골격(`dsa_traffic.c`/`ferry.c` ENQCMD) 재사용으로 즉시 진행 가능.
3. **vLLM 통합** (239-⑥): FERRY 를 NEO KV/prefix 운반 경로에 적용 — GPU 가용 시점.
4. **first-touch 한계**(239-⑤): mbind EPERM 우회책이라 THP/마이그레이션 사후 재배치
   가능성 — host numactl 환경 교차검증 권장.

## 6. 논문 기여 요약

1. **간섭 분류학의 채널별 정량** (메모리>NUMA>TLB>uncore>CPU시간) — reviewer 의
   "격리의 물리적 한계" 질문 선제 답변.
2. **enforcement ladder** (priority 무력 → SW-MBA/HW-MBA 유효) — RDT 필요성 + 보편
   대체재(SW-MBA) 동시 입증.
3. **harvest portfolio** (연산-bound 작업 near-free) — "so what do you run?" 직접 답,
   harvest 접근의 실용성.
4. **부산물**: jemalloc 이 70B serving 엔 무효지만(CPU draft 오버랩 천장, SUB_225),
   draft walk 자체는 −17% — CPU-opt 가 의미있는 regime 분리.

---
*근거: 각 `SUB_*/MEASUREMENTS.md`, `id_registry.md` (SUB_220~236). 도구: `src/victim_aggressor.c`
(`--budget-pct`/`--hugepage` 확장), `SUB_*/run_sub*.sh`, `constructive_bench.c`, `duty_ctl.py`.*

---

# A-시리즈 (신규 알고리즘 ②) 종합 — 2026-06-15

비-DSA 7개 호스트 측정완료. **DSA 의존 3개(236·239·246)는 컨테이너에서 진행 (2026-06-15)**:
호스트는 IOMMU `intel_iommu=on`(sm_off)로 SVA 바인딩 차단 → portal mmap EPERM 이었으나,
`--device=/dev/dsa --cap-add=SYS_RAWIO` 컨테이너에서 **shared WQ ENQCMD 제출이 동작**.

| SUB | 알고리즘 | actuator | 결과 | 판정 |
|---|---|---|---|---|
| 236 | DSA 트래픽 셰이핑 (채널②) | DSA ENQCMD | DSA 제출 성공; victim p99 +42%(4스트림), 동시성 비례 | ✅ partial (셰이핑곡선=호스트) |
| 237 | CC-CAT (AIMD CAT) | CAT ways | victim ~92 전 ways 무효 | ❌ 기각 (actuator 무효) |
| 238 | CSMA-MEM (carrier-sense) | BW SIGSTOP | +144%→+30% (−47%) | ✅ positive |
| 239 | FERRY (DSA-운반 NUMA) | DSA ENQCMD | ns/step −29%, e2e −28% (운반=e2e 1.3%) | ✅ positive |
| 241 | CLOSPACK (측정주도 packing) | MBA | +66%→+39% (naive −16%) | ✅ positive |
| 242 | MERCATO (혼잡가격) | MBA(연속) | +137%→+26%, BW→target 수렴 | ✅ positive |
| 243 | CANARY (간섭 센싱) | (센서) | 간섭 +400% 감지 (runq blind) | ✅ positive ⭐ |
| 244 | LULL-SURF (anti-phase) | BW SIGSTOP | carrier-sense=anti-phase, 정량 부정밀 | 🟡 메커니즘 |
| 245 | NUMA-MIRROR (소켓복제) | (배치) | local vs remote +40% | ✅ positive |
| 246 | IAA-SQUEEZE (압축 운반) | IAA ENQCMD | ⏸️ IAA 디바이스 컨테이너 미노출(`/dev/iax` 없음) | ⏸️ 보류 (호스트 IAA WQ 필요) |

## 핵심 종합
1. **actuator가 결정**: BW-레버(SIGSTOP/MBA) 알고리즘(238/241/242)은 전부 작동, **CAT-레버
   (237)는 무효** — D-시리즈 채널서열(BW 지배, CAT 무효)의 알고리즘 차원 재확인.
2. **CANARY(243)가 신호 계층**: 메모리-지연 센싱이 runq blind(SUB_230)를 메움 → 다른
   알고리즘의 SLO 입력.
3. **NUMA-MIRROR(245)는 별개 차원**(데이터 배치) — NUMA 채널(+43%, SUB_235) 실용해.
4. **DSA 트랙(236/239, 컨테이너)**: (a) 채널②(RMID 미태깅 DSA 트래픽)가 victim p99 를
   실측 +42% 악화 — RDT(MBA/MBM)의 *구조적 사각*을 정량 확정, 간섭은 raw BW 아닌
   **descriptor 동시성**에 비례. (b) 반대로 *유용한* DSA 운반(FERRY)은 원격 NUMA 데이터를
   로컬화해 CPU-busy −29% — **같은 DSA 가 해로운 aggressor(236)도 이로운 offloader(239)도
   됨**. 둘 다 RDT 불가시라, DSA 제어는 디바이스측(read_buffers, 호스트)·제출률(duty)에
   의존 → 간섭 분류학의 "RDT-invisible" 칸을 양방향으로 채움.

## ⚠ 환경 의존 capability (중요 기록)
DSA 제출 가능성은 **호스트 ≠ 컨테이너**: 호스트 IOMMU `intel_iommu=on`(sm_off)는 SVA
바인딩을 막아 portal mmap EPERM. `--device=/dev/dsa --cap-add=SYS_RAWIO` 컨테이너에서는
shared WQ ENQCMD 제출이 동작 (dedicated wq0.1 은 ENXIO, shared wq1.x 가능). 단 컨테이너는
resctrl 미마운트(`/sys` ro)·mbind EPERM(seccomp)·accel-config 미설치·IAA 미노출 →
**resctrl/셰이핑/NUMA바인딩 = 호스트, DSA 제출 = 컨테이너** 로 작업 분리됨.

## ⚠ 중요 한계 (성능 관점)
이 알고리즘들은 **harvest 를 안전하게 돌리는 가드 메커니즘**이지 **serving 성능을 올리지
않는다**. 그리고 **SUB_228(헤드라인)**: 실제 유용작업(연산-bound)은 **이 가드 없이도
near-free harvest** → 정교한 가드(CSMA/MERCATO/CANARY)는 **메모리-bound harvest 변두리
케이스용 보험**. 실제 시스템 성능 향상은 spec-decode(SUB_213 +38%)·CPU-offload(①) 트랙.
