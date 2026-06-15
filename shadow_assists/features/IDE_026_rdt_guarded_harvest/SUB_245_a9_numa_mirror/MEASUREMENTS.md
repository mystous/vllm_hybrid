# SUB_245 — NUMA-MIRROR (소켓별 복제 + RCU epoch), 2026-06-15

> **판정: positive — read-hot 데이터 소켓복제로 +40% 지연 회피.** (SUB_235 +43% 정합.)

## 결과 (reader on socket0/cpu0, ws 128MB)
| 배치 | reader ns/load |
|---|---:|
| MIRROR (local 복제본) | 115.2 |
| NO-MIRROR (remote 단일사본) | 192.7 |
| **mirror 이득** | **40%** |

write-replication 비용: 단일 write mean 0.38ms → 복제는 ~2× (read-mostly 면 amortize).

## 판정
- **read-hot 데이터를 소켓마다 복제 → 모든 reader 가 local 접근 → remote NUMA(+43%, SUB_235)
  회피 = +40% 지연 개선** 실측.
- 비용: write 시 전 소켓 복제(2×) + RCU-식 epoch 발행으로 일관성(reader 는 local 복제본,
  writer 는 epoch 갱신). read≫write 인 핫데이터(KV index, 라우팅 테이블 등)에 적합.
- **데이터-배치 알고리즘**(간섭제어 아님) — NUMA 채널(SUB_235)의 실용적 해법.

## 비고
- 본 데모는 local/remote 정적 비교로 이득 상한 확정. 동적 복제·epoch 일관성 구현은 후속.

## 부록 — first-touch vs mbind 교차검증 (호스트, 2026-06-15, 미진사항 ④)

컨테이너의 FERRY/NUMA 측정은 `mbind` EPERM 우회로 **first-touch 배치**(cpunodebind +
preferred)에 의존 → THP 자동승격·페이지 마이그레이션으로 사후 재배치되면 측정이
오염될 수 있음. 호스트에서 두 배치법을 동일 ws(128MB, cpu0=node0)로 직접 대조:

| 배치법 | local ns/load | remote ns/load | remote 패널티 |
|---|---:|---:|---:|
| `numactl --membind` (명시적) | 115.7 | 192.6 | +66% |
| first-touch (`--cpunodebind 0 --preferred`) | 115.2 | 192.0 | +67% |

→ **두 방식 일치 (+66% vs +67%)**. first-touch 배치가 실제로 의도한 노드에 고정됨을
확인 = 컨테이너의 FERRY(SUB_239)·NUMA-MIRROR(SUB_245) 수치는 마이그레이션 아티팩트
없이 **신뢰 가능**. 미진사항 ④ 해소.

산출물: `runs/results.csv`, `runs/firsttouch_xcheck.csv`.
