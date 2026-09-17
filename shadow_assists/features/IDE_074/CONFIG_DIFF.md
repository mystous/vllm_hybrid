# CONFIG_DIFF — IDE_074

## 1. IDE_073 Q-OPT4 (기준) ↔ IDE_074 실행 환경
| 항목 | IDE_073 Q-OPT4 (09-16 23:21 부팅) | IDE_074 | 분류 |
|---|---|---|---|
| launch argv / server_args | 동일 (`evidence/environment.json: reference_launch_cmd`) | 동일 | — |
| env KT_CALLBACK_FREE / KT_CF_SKIP_EMPTY_IMM / KT_GPU_EXPERTS_PER_LAYER / KT_AVX_RB | 1 / 1 / layer_budget_5952.json / "0"(ON) | 동일 | — |
| kt_kernel_ext.so | 8,290,336 B, SHA e29357f7… (09-09 빌드) | 8,306,720 B, SHA adf49e48… (09-17 재빌드: 이벤트 프로브 코드 포함, `KT_EVT` 미설정 시 비활성) | SOFTWARE (계측) |
| `moe_base.hpp::forward_decode` printf | qlen==1 마다 무조건 stdout 출력 (게이트 없음) | `KT_PHASE_PROF` + 64회 1회 게이트 | SOFTWARE (로그 정리; C=1 경로에만 영향) |
| site-packages `kt_kernel/experts_base.py` | IDE_033 패치본 (SHA d800a32a…) | + `KT_EVT` 설정 시 slot→layer 맵 4행 (SHA ecb1d899…), 미설정 시 동일 | SOFTWARE (계측) |
| SGLang tree | 71de97b + 로컬 7파일(M) | 동일 (변경 없음) | — |
| 층별 예산 패치 | 부팅 시 patch_per_layer_experts.sh 적용·종료 후 revert | 동일 | — |
| 비-kt pinning | pin_nonkt (free_cpus 32) | 동일 | — |
| 계측 모드 env | (없음) | OFF: 없음 / P1: KT_EVT=/models/kt/ide074/<boot>/kt_evt.csv, SGLANG_TORCH_PROFILER_DIR / P2: KT_EVT + KT_TQ_TIMING=1 + KT_PHASE_PROF=1 | OBSERVER |
| torch profiler | activities ['CPU','GPU'] (retry2) | 동일 + `start_step=1` (다음 forward 부터) | OBSERVER |

## 2. 계측 OFF ↔ ON 에서 바뀌는 것 (같은 바이너리)
| 모드 | 추가되는 것 |
|---|---|
| OFF | 없음 (공통 경량 모니터: /proc/stat 1 s, nvidia-smi 2 s, free/numastat 5 s — IDE_073 과 동일) |
| P1 | poller/worker 의 CLOCK_REALTIME 스탬프(레코드 1개/층·step, 링 1<<20, 200 ms 플러셔), deferred 작업 전후 stamp 태스크 2개/층, Python arm/rearm 시 맵 파일 append, torch profiler(kineto CPU+CUDA) |
| P2 | P1 의 CPU 이벤트 + `[kt-tq]`(2048 태스크당 1행) + `[kt-wrap]`/`Profiling Results`(64회 1회) stdout + perf stat(-p 스케줄러 프로세스) + pcm-memory 1 s |

## 3. 지시서 제안과 다르게 확정한 것
- OFF/ON 을 같은 부팅에서 전환하지 않음 (KT_EVT 는 정적 getenv). 부팅 단위 모드 → 부팅 간 비교의 한계로 보고.
- `hot_ready` 는 트레이스의 HtoD 직전 커널(moe_sum_reduce) 종료로, 층 번호는 memcpy 순번 패턴으로 잡음 (HEURISTIC_LAYER_ID). 그래프 안에 별도 마커 커널을 넣지 않음 (오버헤드·캡처 변경 회피).

## 4. 계측 OFF(원상) 복원 방법
```
# 컨테이너 sgl-kt
python3 /tmp/ide074_kt_evt_patch.py revert            # cpuinfer.h / moe-tp.hpp / moe_base.hpp / site-packages experts_base.py 를 /sgl-workspace/ide074_backup/*.orig 로 복원, kt_evt.h 삭제
cp /sgl-workspace/ide074_backup/kt_kernel_ext.so.orig /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so   # SHA e29357f7…
```
패치 스크립트 원본: `eval/ide074/kt_evt_patch.py` (저장소). 현재(캠페인 종료 시점) 컨테이너는 패치 적용 + 재빌드 .so 상태로 둠 (env `KT_EVT` 미설정이면 기존 경로와 동일 동작; decode printf 게이트만 상시 차이).
