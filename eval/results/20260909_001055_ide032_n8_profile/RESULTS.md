# hot-96 N=8 (cold 전량 deferral) 정상 상태 프로파일 — 유휴 갭 기전 (2026-09-09 00:15)
정상 스텝 (step1·2): span 32.2ms = 커널 25.8 + 복사 1.9 + **유휴 8.7ms**. 갭(>30µs) 121~127개/스텝, p50 42µs, p90 50~64µs — **층당 2개**.

| 갭 위치 (앞 커널 → 뒤 커널) | 개수/스텝 | 합계 | 정체 |
|---|---|---|---|
| `Memcpy DtoH(입력·ids·weights) → triton gate/index 커널` | 61~62 | 2.7~2.8 ms | **submit host 노드 2개** (immediate + deferred `cudaLaunchHostFunc`) 디스패치 |
| `triton combine(mul_sum) → Memcpy HtoD(CPU 출력)` | 60~61 | 3.4~4.4 ms | **sync host 노드** (`sync_` 블로킹 호출; N=8 이라 실제 대기할 CPU 작업은 없음) |
| 그 외 | 3~5 | <0.4 ms | |

결론:
1. **CPU 대기 = 0.** deferral 은 CPU cold 작업을 완전히 숨겼다 (기전은 맞았음). 처리량이 안 오른 이유는 남은 유휴가 CPU 가 아니라 **host 콜백 노드의 고정 디스패치 지연** (사이트당 ~45~55µs, 마이크로벤치 20µs 보다 큰 것은 sync_ 의 큐 검사·mutex 포함) 이기 때문. 이 바닥은 N·H 와 무관 → 관측된 "캡" 의 정체.
2. **IDE_033 (host 노드 제거) 의 이득이 기전적으로 확정**: 사이트 2×62 = 124 노드 × (~50 − memop 9)µs ≈ **5ms/스텝** → hot-96 N=8 스텝 32.2 → ~27 (**+18~19%**), TPOT 48 → ~43 → C32 ≈ 505 → **~600 tok/s**. hot-80 은 별도 프로파일로 (CPU 노출분 확인).
3. 원본: `gaps_TP0.txt`, 트레이스 4 rank.
