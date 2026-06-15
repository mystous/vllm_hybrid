# SUB_235 — NUMA 배치·드리프트 복구, 2026-06-15

> **판정: positive — NUMA 드리프트 = +43% 지연. 드리프트 복구 가치 정량화.**

## 측정 (victim cpu0=node0, 메모리 배치만 numactl --membind 변경)
| 배치 | ns/load (3반복) | 평균 |
|---|---|---|
| local (node0) | 56.7 54.8 62.4 | 58.0 |
| remote (node1, UPI) | 88.9 79.7 80.1 | 82.9 |

→ **원격 NUMA = +43% 지연** (범위 비중첩으로 명확). UPI 교차소켓 hop.

## 판정
1. **NUMA 드리프트는 실재하는 큰 채널 (+43%)**. serving 스레드의 메모리가 원격
   노드로 드리프트하면 host-path 지연 43% 악화.
2. **드리프트 복구(페이지 local 마이그레이션)는 ~43% 회복 가치** → 운영 레버로 유효.
   harvest 가 victim 페이지를 evict→원격 재배치시키는 시나리오에서 특히 중요.

## 복구 알고리즘 (설계, 구현은 후속)
- 탐지: NUMA hint fault(autonuma) 통계 또는 `/proc/PID/numa_maps` 원격 비율.
- 복구: `move_pages()`/`migrate_pages()` 로 serving working set 을 local 로 재배치.
- 게이트: 복구 후 ns/load ≤ local+5%. (본 측정은 정적 local/remote 로 상한·하한 확정.)

## 함의
- NUMA-aware 배치(이미 gpu_worker.py NEO pinning 에 부분 적용)가 +43% 를 막음.
  harvest 배치도 NUMA-local 강제 + serving 페이지 보호 필요.
- 정적 측정으로 드리프트 비용 상·하한 확정(58 local / 83 remote). 동적 복구는 후속.

산출물: `runs/` (inline). 도구: numactl + victim_aggressor.
