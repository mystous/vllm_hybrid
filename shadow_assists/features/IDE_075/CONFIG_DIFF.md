# CONFIG_DIFF — IDE_075

| 항목 | IDE_074 | IDE_075 | 분류 |
|---|---|---|---|
| launch argv / env(KT_CALLBACK_FREE·KT_CF_SKIP_EMPTY_IMM·KT_GPU_EXPERTS_PER_LAYER·KT_AVX_RB) / hotmap / 층별 예산 / pinning / KV / graph | 동일 | 동일 | — |
| kt_kernel_ext.so | v1 adf49e48… (stamp task 2개/층, 1<<20 레코드) | **v2 a4e9045a… 8,319,008 B** (기록 전용 task 0, task 링, SPSC, 단계·분기 기록) | SOFTWARE(계측) — OFF/ON 공통 |
| C++ 소스 | cpuinfer.h, moe-tp.hpp, moe_base.hpp (+kt_evt.h) | cpuinfer.h, task_queue.{h,cpp}, moe-tp.hpp, moe_base.hpp, moe.hpp (+kt_evt.h v2) — `evidence/kt_evt_patch_v2.diff` | SOFTWARE(계측) |
| OFF 상시 차이 (env 무관) | decode printf 게이트 | + TaskQueue seq 카운터(atomic fetch_add 1회/enqueue)·Node.seq·thread_local 2개 write | 무시 가능한 비용 (측정 안 함) |
| site-packages experts_base.py | v1 맵 4행 | v2 맵 4행 (동일 형식) | 계측 |
| 계측 모드 | OFF/P1/P2 (부팅 단위) | OFF / CORE / CORR / RESOURCE (CORE·CORR·RESOURCE 같은 부팅, 세션 경계 전환) | OBSERVER |
| perf | perf stat 누적 | perf stat -I 1000 -x , (interval) | OBSERVER |
| 실행 순서 | — | OFF_A → OFF_B → CORE (하네스 결함으로 OFF_OPEN/CORE 부팅 2회 소진) | 한계 |

복원: CONFIG_DIFF(IDE_074) §4 와 동일 + `python3 /tmp/ide075_kt_evt_patch_v2.py revert`, `.so.orig` 복사. 현재 컨테이너는 v2 적용 상태.
