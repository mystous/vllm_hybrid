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

## 확장 단계 (phase=extended) 추가 변경

| 항목 | 값 | 분류 |
|---|---|---|
| kt_kernel_ext.so v3 | d659ca0b… 8,327,200 B (2026-09-17 16:11). v2 의 SPSC 경쟁으로 fwd 레코드 누락 → Node 가 rec 포인터 보유, 스레드 이름(kt-cf-poll / kt-evt-flush / kt-task-worker), expert 단위 rows 표본 ring(`.experts`). CORE2·CORE3·CORE4·OFF_OPEN2/CLOSE2/X/Y 에 사용. 백업 `/sgl-workspace/ide074_backup/kt_kernel_ext.so.v3` | SOFTWARE(계측) |
| kt_kernel_ext.so v4 (fp8fix) | 12926df2… 8,376,352 B (2026-09-17 18:11). v3 + `operators/amx/moe_base.hpp` 의 if-constexpr dangling-else 3곳 수정 (`eval/ide075/glm_fp8_dispatch_fix.py`; 수정 전 백업 `moe_base.hpp.pre_fp8fix`). INT4 경로: env 미설정 기본 동작 동일 (S29_CORR 로 처리량 회귀 확인). FP8/BF16 경로: 입력 gather·A 양자화 복원 → GLM 정상 출력. GLM_D6/D7·G_OFF/G_CORR·FOCUS2 에 사용 | SOFTWARE(수정) |
| GLM python 패치 | `glm_rsf_patch.py` (site-packages `kt_ep_wrapper.py`: cpu_output × routed_scaling_factor, fast path 포함; `glm4_moe.py`: dual_stream KT 분기 제거). 백업 `*.rsf_orig`. D6 = apply, D7 = revert | SOFTWARE(수정) |
| FOCUS perf | `IDE075_FOCUS_SYSWIDE=1` → `perf record -a -k CLOCK_MONOTONIC -m 4096 -e sched:sched_switch` (S28). 기존 S14/S19/S25 는 `-p` (switch-in 누락) | OBSERVER |
| 클라이언트 | `vllm_bench_wrapper.py` (vllmw): vllm bench serve 본체 + benchmark() 진입 barrier·anchor. probe_client.py 는 비동등 판정 | OBSERVER |

복원 (확장 단계 이후): `python3 /tmp/glm_rsf_patch.py revert` · `python3 /tmp/glm_fp8_dispatch_fix.py revert` (moe_base.hpp) · `python3 /tmp/kt_evt_patch_v3.py revert` → v2 → `kt_evt_patch_v2.py revert` → `.so.orig` 복사 (단, `.so.orig` 자체가 FP8 결함 포함; FP8 정상 .so 는 v4). 현재 컨테이너: v4 + rsf 패치(D7 이후 상태는 D7 실행 결과에 따름, WORK_LOG 참조).
