# SUB_240 [A4 RELAY-Q 프리미티브 vLLM 적용판] — MEASUREMENTS (확정판, 2026-06-13)

> **판정 요약 (tps 중립 — 게이트 미달)**: EngineCore 의 sched_yield 스핀 (self-time
> 64% 실측) 을 tpause(C0.2) 로 교체해도 serving tps 는 **중립** — W1(2µs) 0.984 /
> W2(8µs) 1.001 (mbpp outlier 제외 시 1.007/1.024 — +3% 게이트 미달, corpus 간
> 비일관). **해석**: sched_yield 폴링도 이미 µs 급이라 wake 지연이 병목이 아니었음.
> tpause 의 실이득은 syscall 제거·C0.2 절전·스케줄러 무부하 (본 측정 범위 밖).
> **중요한 부작용 발견 (설계 노트)**: sched_yield 는 대기 중 코어를 다른 스레드에
> *양보* 하지만 tpause 는 코어를 *점유* 한 채 멈춤 — co-location (harvest) 관점에선
> sched_yield 가 오히려 유리. tpause 모드는 전용-코어 배치에서만 의미.

## 1. 근거 프로파일 (py-spy, 70B suffix 부하 중 EngineCore 120s)

```
self-time: sched_yield 63.7% + libc spin ~11% + shm_broadcast wait/acquire ~13%
경로: _process_engine_step → multiproc_executor._wait_for_response (total 98%)
```
→ EngineCore 는 "8 TP 워커 응답을 shared-memory ring 에서 스핀 대기" 가 지배.
산출물: `../profiling/engine_profile.speedscope.json`, `engine_threads.txt`

## 2. 구현

- `uwait.c` → `libuwait.so`: `u_tpause(cycles)` (C0.2 체인, umwait max 100k TSC 분할)
  + `u_umwait_addr` (주소감시 — 향후 확장 자리)
- `vllm/distributed/utils.py`: `VLLM_SCHED_YIELD_MODE=tpause` env-게이트 (기본 off,
  수치 경로 무접촉). **함정 기록**: 모듈 로드 시점 `logger.info_once` 호출은
  parallel_state lazy import 로 순환 import 유발 (실측·수정).

## 3. 결과 (70B suffix × 7 corpus × 3셀, 셀별 fresh boot)

| corpus | W0 base | W1 tp2µs | W2 tp8µs | W1/W0 | W2/W0 |
|---|---:|---:|---:|---:|---:|
| sharegpt | 4,596 | 4,547 | 4,399 | 0.989 | 0.957 |
| swebench | 5,038 | 5,208 | 5,562 | 1.034 | 1.104 |
| humaneval | 4,346 | 4,381 | 4,690 | 1.008 | 1.079 |
| mbpp | 3,090 | 2,648 | 2,698 | 0.857 | 0.873 |
| wildchat | 4,904 | 5,060 | 5,053 | 1.032 | 1.031 |
| lmsys | 4,216 | 4,281 | 3,890 | 1.016 | 0.923 |
| mix | 7,446 | 7,168 | 7,901 | 0.963 | 1.061 |
| **기하평균** | | | | **0.984** | **1.001** |
| (mbpp 제외) | | | | 1.007 | 1.024 |

mbpp 는 W0 값 (3,090) 이 일중 분포 (2.4~2.9k) 상단 outlier — 비율 왜곡 주의.

## 4. 판정·후속

1. tps 게이트 미달 → tpause 모드는 **기본 off 유지** (env 잔존 — 전용 코어 배치
   + 전력 실험용).
2. **다음 표적은 워커**: 엔진은 대기-지배 (실작업이 아님) — sampler/detok 등
   호스트 실작업은 TP 워커 프로세스에 있음 (SUB_161: TP0 sampler 44%). 워커
   프로파일 후 재조준.
3. umwait 주소감시 확장은 보류 — wake 지연이 병목이 아님이 판명됐으므로
   tps 동기로는 약함. RELAY-Q 본판 (큐 프로토콜) 은 D18 ping-pong 맥락에서 유효.
