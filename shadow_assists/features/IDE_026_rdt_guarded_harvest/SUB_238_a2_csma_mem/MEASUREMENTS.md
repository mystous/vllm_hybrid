# SUB_238 — CSMA-MEM (분산 carrier-sense 메모리 BW 중재), 2026-06-15

> **판정: positive — 작동.** harvest 가 MBM 으로 메모리 BW 를 carrier-sense 해서
> busy 면 백오프(SIGSTOP)·idle 면 재개(SIGCONT). victim 악화 +144%→+30% 절감.

## 구현
`csma_ctl.py`: resctrl `mon_data/mon_L3_*/mbm_total_bytes` 합산 delta 로 총 BW 센싱
(50ms epoch). BW > ceiling 면 harvest `os.kill(SIGSTOP)`, ≤ ceiling 면 SIGCONT.
(setsid fork 로 PID 어긋나던 버그 수정 → 직접 os.kill.)

## 결과 (victim 0-7, harvest 8-23, ceiling 5000 MB/s, victim 단독 BW≈3910)
| arm | victim ns/load | vs baseline |
|---|---:|---:|
| baseline (무harvest) | 84.4 | — |
| unthrottled harvest | 206.3 | +144% |
| **CSMA harvest** | **109.7** | **+30%** (unthrottled −47%) |

stops=82, conts=82 (8s 동안 82회 백오프 — 능동 제어 확인).

## 판정
1. **CSMA-MEM 작동**: BW 센싱→백오프로 victim 악화 절반 절감(+144%→+30%).
2. **actuator 가 유효(BW 백오프)** — CC-CAT(CAT, 기각)과 결정적 차이. BW-레버는
   작동(SW-MBA/SUB_224 정합), 캐시-파티션 레버는 무효(SUB_221/237).
3. **분산 설계 성립**: 각 harvest 가 공유 MBM(carrier) 만 보고 독립 백오프 → 중앙
   조정 없이 BW 중재. CSMA/CD(ethernet)의 메모리버스 이식 입증.

## 튜닝 여지 (미완)
- ceiling 낮추면 victim 더 보호(↓ harvest). 50ms→짧은 epoch 면 tighter(버스트 누수↓).
- victim 완전보호(baseline) 위해선 ceiling≈victim_BW + exponential backoff (현재 binary).

산출물: `csma_ctl.py`, `runs/results.csv`.
