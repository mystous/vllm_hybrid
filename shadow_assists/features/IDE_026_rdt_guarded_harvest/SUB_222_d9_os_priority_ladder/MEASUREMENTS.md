# SUB_222 — OS 우선순위 사다리 (RDT-無 baseline), 2026-06-15

> **판정: 측정완료 — 우선순위 knob 무력 / cpu.max 유효(blunt) → RDT 필요성 직접 입증.**
> CPU-only (victim 0-7 + aggressor 8-23, 동소켓 L3/BW 공유, 코어 분리 = 시간분할 無).

## 결과 (victim p99 latency + 메모리지연 ns/load, aggressor=basic 32MB)
| 셀 | p99(ms) | ns/load | vs baseline |
|---|---:|---:|---|
| C0_baseline | 20.8 | 91.1 | — |
| C1_default | 41.8 | 205.9 | +101% (2배) |
| C2_nice19 | 43.0 | 205.9 | 무력 (=default) |
| C3_sched_idle | 42.1 | 205.9 | 무력 (=default) |
| C4_cpumax50 | 22.3 | 89.2 | ~완전 복구 |

## 판정
1. **우선순위(nice19/SCHED_IDLE) = 완전 무력**: ns/load 205.9 로 default 와 동일.
   코어 분리 → 시간분할 경합 없음 → 스케줄러는 BW 를 모르므로 무효. (가설 정확 입증)
2. **cpu.max=50% (시간 throttle) = 유효**: p99·ns/load baseline 복구. aggressor 가동
   시간 절반 → BW 간섭 절반. **단 harvest useful-work 도 절반 손실** (blunt).
3. **enforcement ladder**: priority(0) < cpu.max(유효·harvest 50%↓) < RDT-MBA
   (BW 타깃·harvest 보존). → "우선순위 무력함" 이 RDT-MBA 채택의 직접 논거.

## 함의
- harvest 가드로 OS 스케줄러 우선순위는 쓸모없음 (코어 분리 설계 전제).
- cpu.max 는 보호되나 harvest 처리량을 비례 희생 → RDT-MBA(BW 만 제한, compute 보존)
  가 우월. SUB_214(D1 MBA)·본 결과가 RDT 트랙의 정당성.

산출물: `run_sub222.sh`, `runs/results.csv`.
