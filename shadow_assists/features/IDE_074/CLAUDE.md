# IDE_074 — Claude 작업 메모

- 노드 violet-h100-016: 컨테이너 `sgl-kt`(서버) / `vllm-h100`(벤치 클라이언트), `~/bin/docker` = sudo nerdctl. SGLang 로컬 tree `/sgl-workspace/sglang` (71de97b + 로컬 패치), kt_kernel 0.7.0.post1 로컬 빌드 (`kt_kernel_ext…so` SHA e29357f7…). turbo BIOS 잠금(no_turbo=1) — 우회 금지.
- `KT_AVX_RB=0` 은 이 빌드에서 **ON** (getenv != NULL). 계측 OFF 는 새 계측 기능만 끄는 것이며 최적화 env 를 unset 하지 않는다.
- SGLang `/start_profile` activities 키는 `["CPU","GPU"]` (`CUDA` 아님).
- 트레이스 시계: kineto chrome trace `ts`(µs) + `baseTimeNanoseconds`/1e3 = host CLOCK_REALTIME µs. 커널 이벤트 args 에 `graph id`/`graph node id`/`stream`/`correlation`/`device` 있음.
- 실행 중인 스크립트 파일은 편집하지 않는다 (bash 는 실행하며 읽음). pgrep 패턴은 `[p]ython3 …` 형식으로 자기 일치 회피.
- 재현 하네스는 `eval/ide071/common.py` (boot_server, run_bench, pin_nonkt) + `eval/ide073/run_campaign.py` 의 OPT4 cfg 를 재사용. 새 코드는 `eval/ide074/` 에 둔다.
- 예산 원장은 `eval/results/IDE_074_20260917/state/execution_events.jsonl` (session_end/boot_end/session_void/status_correction). 상한 24 세션·부팅 12 회.
- 보고 문서는 평가·권고 없이 원값·유효성·누락만 기록한다. 게시는 명시 파일 stage, force push 금지.
