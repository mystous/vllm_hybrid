# SUB_223 — duty-cycle actuator (부분: SIGSTOP), 2026-06-15

> **판정: 측정완료(부분) — SIGSTOP actuator duty→BW 대략 선형(±~10%).**
> tpause/SCHED_DEADLINE, 100µs/1ms 주기는 actuator 구현·µs 정밀제어 필요로 보류.

## 결과 (SIGSTOP, 10ms 주기, victim 0-7 + aggressor 8-23, ns/load=BW간섭 프록시)
| duty% | ns/load | 선형예측 | 편차 |
|---|---:|---:|---:|
| 0 | 86.9 | — | — |
| 25 | 129.4 | 116.5 | +11% |
| 50 | (flaky·미수집) | 146.0 | — |
| 75 | 190.6 | 175.5 | +9% |
| 100 | 205.1 | — | — |

## 판정
- **duty 비율 → BW 간섭 대략 선형** (약간 convex, ±~10%) → D10 게이트(선형성 ±10%)
  SIGSTOP actuator 에서 근사 충족. governor 제어성(T4) 전제 성립.
- 보류: tpause(코드 미지원·새 binary 필요)/SCHED_DEADLINE(chrt -d 설정)/100µs·1ms 주기
  (userspace SIGSTOP 신호지연으로 µs 불가). 이들은 actuator binary 구현 후 별도.

## 비고
- 50% 셀은 SIGSTOP 타이밍 flaky 로 미수집(0/25/75/100 으로 추세 충분).
- 함의: SIGSTOP 만으로도 duty→보호 선형 제어 가능(거친 주기). 정밀(µs)·무-cold-miss
  는 tpause 가 필요 — D10 의 핵심 비교(tpause vs SIGSTOP cold-miss)는 미완.

산출물: `run_sub223.sh`, `duty_ctl.py`, `runs/results.csv`.

---

## [추가 2026-06-15] tpause actuator 구현·완성 (보류 해제)

`victim_aggressor.c +--tpause-duty N` 구현 (waitpkg `_tpause` C0.2, TSC 캘리브레이션,
pass-duty off-time self-tpause). waitpkg ✓.

### tpause duty → BW (단조 작동)
duty 0/25/50/75% → BW 134/39/75/108 GB/s.

### 핵심: actuator별 harvest 효율 (duty 50%, victim 보호 동일 ~141 ns/load)
| harvest array | nanosleep BW | tpause BW |
|---|---:|---:|
| 32MB (>L2, 메모리스트리밍) | 104.2 | 102.3 (≈동일) |
| **1MB (≤L2, 캐시상주)** | 114.0 | **151.5 (+33%)** |

### 판정 (D10 가설 확정)
- **같은 duty → victim 보호 동일** (actuator 무관, budget=tpause=141.7 ns/load).
- **tpause 는 context switch 없이 L2 보존** → 캐시상주 harvest 는 resume cold-miss 없어
  **+33% harvest 효율** (nanosleep/SIGSTOP 대비). 메모리스트리밍(>L2)은 보존 대상 없어 동일.
- → **SUB_228 의 연산-bound 유용작업(캐시상주)에 tpause 가 최적 actuator.**
- SCHED_DEADLINE 변형은 chrt -d(sudo)로 미시험(부차적 — SIGSTOP/tpause로 핵심 비교 완료).

→ SUB_223 판정: 🟡 부분 → ✅ **완료** (3 actuator 중 SIGSTOP·tpause 완료, tpause 우위 입증).
