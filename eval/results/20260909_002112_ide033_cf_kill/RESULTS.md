# IDE_033 kill test — callback-free CPU↔GPU 핸드오프, hot-96 N=8 (2026-09-09 00:21~00:25)
빌드: cpuinfer.h 패킷 테이블 + go/done mapped 플래그 + 폴러 스레드, experts_base.py `KT_CALLBACK_FREE=1` 분기. 빌드 오류 0, 바인딩 4개 확인.

| 항목 | CF (KT_CALLBACK_FREE=1) | 기준 (host 콜백 노드, IDE_032 N=8) |
|---|---|---|
| 부팅 | 100s, `[kt-cf] callback-free handoff ready` | — |
| greedy 2문 | 동일 텍스트 (Paris / fibonacci) | 동일 |
| 32 버스트 | 32/32, 2.5s | 32/32 |
| C32 sonnet 512/128 | **530.97 tok/s, TPOT 45.47** | 505.4, 48.3 |
| GSM40 | **97.5% (39/40)** | 97.5% |

판정: **kill 기준 (≥580) 미달, +5.1%**. 정확도 무손실, 메커니즘 동작 정상. 스텝 −2.8ms (48.3→45.5) = 프로파일상 host 노드 유휴 8.7ms 중 1/3 만 회수. 잔여 ~6ms 의 위치 (폴러 반응 지연 / 빈 immediate 작업의 스레드풀 고정비 / 큐 내 done 신호 작업 / def(L-1) 완료 대기) 를 TaskQueue 작업별 계측 (`KT_TQ_TIMING`) + CF 프로파일로 특정 → `…_ide033_cf_diag/`.
