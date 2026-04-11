# vLLM Hybrid — 남은 작업

작업 이력은 `Task_down.md`, 프로젝트 현재 구성은 `CLAUDE.md`, 설계의 단일 진실 공급원은 `docs/paper/main.tex` 를 참조.

---

## 0. 이번 세션 uncommitted 변경 정리 (우선)

이번 세션에서 `git status` 상 uncommitted 변경이 쌓여 있다. 동작 검증은 대부분 통과했지만 **아직 commit 안 됨**. 원칙: **"Don't commit or push without explicit command"** — 사용자 승인 후 commit.

- [ ] `git diff` 로 전체 변경 훑어보기
- [ ] 논리적 단위로 commit 분할 (아래 그룹 예시):
  - Group A: `_C_utils` standalone extension (정도 fix)
    - `cmake/cpu_utils_extension.cmake` (신규)
    - `csrc/cpu/torch_bindings_utils.cpp` (신규)
    - `csrc/cpu/utils.cpp` (cpu_types.hpp 제거)
    - `CMakeLists.txt` (include 추가)
    - `setup.py` (extension 등록)
    - `vllm/_custom_ops.py` (`HAS_CPU_UTILS` import)
  - Group B: `WorkerBase` heterogeneous 휴리스틱 우회 + CPUWorker device_type coerce
    - `vllm/worker/worker_base.py`
    - `vllm/v1/worker/cpu_worker.py` (해당 hunk)
  - Group C: NUMA 원칙 자동 감지 (num_cpu_engines = num_numa, cpu_max_num_seqs = 1)
    - `vllm/config.py` (`HybridConfig.num_cpu_engines` default)
    - `vllm/engine/arg_utils.py` (CLI default)
    - `vllm/v1/engine/hybrid_core.py` (`_resolve_num_cpu_engines` 신규, `_resolve_cpu_params` 수정, `launch_hybrid_engines` 수정, `_create_cpu_vllm_config` passthrough, `run_cpu_engine_core` vllm_config replace)
    - `vllm/v1/engine/core_client.py` (resolver write-back)
    - `vllm/v1/worker/cpu_worker.py` (`_get_autobind_cpu_ids` numa_bind_node 우선)
  - Group D: Python sched_setaffinity fallback + 진단 로그 + execute_model trace
    - `vllm/v1/worker/cpu_worker.py` (나머지 hunk)
    - `vllm/v1/engine/hybrid_core.py` (진단 로그 hunk)
    - `vllm/v1/engine/core_client.py` (dispatch 로그 hunk)
    - `vllm/v1/attention/backends/cpu_attn.py` (decode path counter)
    - `eval/serve.sh` (TRACE env export)
  - Group E: 문서 + eval env
    - `docs/CUDA13_MIGRATION_STATUS.md` (신규)
    - `Task_down.md` (신규, 본 파일로 이관)
    - `CLAUDE.md` (재정비)
    - `TODO.md` (본 파일, 신규)
    - `eval/envs/dev_rtx3090_hybrid_smoke.env`, `dev_rtx3090_500.env`, `dev_rtx3090_qwen7b_hybrid_verify.env`, `dev_rtx3090_qwen7b_500.env` (신규)

---

## 1. dev 환경 로직 검증 잔여 (H100 이관 전 완료해야 할 것)

성능 수치는 dev 에서 의미가 없지만 **로직 무결성** 은 dev 에서 완전히 검증해야 한다. 아래 항목은 모두 `cpu_max_num_seqs = 1`, `num_cpu_engines = 1` 원칙 고정으로 수행.

- [ ] **1-시퀀스 라이프사이클 반복 검증**
  - 요청을 순차로 N번 (예: 50번) 보내며 매번 `cpu_in_flight = 0→1→0` 이 정확히 반복되는지 `Router stats` / `[HYBRID-CLIENT]` 로그로 확인
  - 누수/영구 점유/데드락 0건 확인
  - 각 요청이 매번 동일하게 16 코어 포화를 얻는지 (PSR 고정 1:1 매핑 재확인)

- [ ] **동시 요청 스트레스 — 데드락/stall 탐지**
  - N개(>> cpu_max_num_seqs) 요청을 한꺼번에 보내 CPU wait queue / running queue 라이프사이클 무결성 확인
  - timeout 이나 hang 발생 여부 검증
  - 현재 5-10 req burst 까지 검증됨. 50+ 로 확장 필요

- [ ] **CPU scheduler 코드 경로 트레이싱**
  - V1 Scheduler 가 `cpu_max_num_seqs=1` 경계에서 preemption / reschedule 하는지 코드로 확인
  - chunked prefill 이 CPU engine 에서 정상 동작하는지 (`enable_chunked_prefill=True` 가 기본값)
  - `_update_from_output` → `_free_request` → `on_request_finished` → router slot decrement 경로 검증

- [ ] **`output.finished` 감지 확실성**
  - `process_engine_outputs` 가 `output.finished` 속성을 못 잡으면 슬롯 영구 점유 → 후속 요청이 모두 GPU 로 몰림
  - 현재 단일 요청에서는 정상 반납 관측됨. 다양한 종료 조건 (`length`, `stop`, `abort`) 에서도 확실히 반납되는지 검증

- [ ] **H100 "capacity 에서 멈춤" 증상 원인 후보 dev 에서 배제**
  - Warmup / profile_run hang 가능성
  - ZMQ IPC 경로의 blocking 가능성
  - `chunked_prefill` + CPU 엔진 조합의 edge case

---

## 2. 논문 ↔ 현재 코드 재정합

이번 세션에서 코드가 원칙에 맞게 수정되면서 논문의 §3.4 (Automatic CPU Configuration) 와 Table 2 (Auto Rules) 가 더 이상 정확하지 않다.

- [ ] **논문 Table 2 `max_seqs` auto rule 수정**
  - 현재 논문: `max(4, ⌊cores / 4⌋)` + "4 threads/sequence" rationale
  - 실제 코드: **`1` per engine (NUMA 노드당)** + "1 sequence saturates whole NUMA node via OMP" rationale
  - Table 2 + §3.4 "Maximum concurrent sequences" paragraph 재작성

- [ ] **논문에 `num_cpu_engines = num_numa` auto 규칙 추가**
  - 현재 논문: `num_cpu_engines` 는 CLI 옵션으로만 언급, auto 감지 설명 없음
  - 실제 코드: `_resolve_num_cpu_engines` 가 `NUMAAllocator.num_nodes` 로 자동 결정
  - §3.4 Table 2 에 추가, Figure 4 (hwloc topology) 캡션 업데이트

- [ ] **§3.3 CapacityAwareRouter Algorithm 1 의 `N` 표기 명확화**
  - 현재 논문 `N = cpu_max_num_seqs` 로 추상화
  - 설계 원칙상 `N = 1` per engine, 총 동시 CPU seq = `num_numa × 1` 임을 명시

- [ ] **§5 Implementation 에 `_C_utils` standalone extension 언급**
  - 현재 논문: `_C_cpu_ops` 만 언급
  - 추가: `_C_utils` 는 `init_cpu_threads_env` 전용, AVX-512/AMX 무관, CUDA/ROCm 빌드에서도 항상 빌드

---

## 3. H100 타겟 환경 검증 (dev 로직 검증 완료 후에만 실행)

**조건**: §1 의 모든 항목이 dev 에서 통과된 후에만 H100 자원 사용. "로직 버그를 H100 에서 디버깅" 은 금지.

- [ ] **H100x4 KVM 재측정** — 이전에 96 logical 중 6-8 코어만 사용했던 결과가 이번 fix 로 52 core 전체 사용으로 개선되는지 확인
- [ ] **H100x8 + Xeon 8480+ 2-socket 실 환경 첫 부팅**
  - `[HYBRID-LAUNCH] num_cpu_engines=2 (numa_aware=True, config=0)` 가 auto 로 나오는지
  - 두 CPU EngineCoreProc 이 각각 다른 NUMA 노드에 1:1 pin 되는지 (`init_cpu_threads_env` 로그에서 socket 0/1 cores 구분)
  - multi-NUMA 에서 `_get_autobind_cpu_ids` 의 `numa_bind_node` 우선 경로가 정확히 동작
- [ ] **Exp 1 — 고부하 end-to-end throughput** (논문 §5)
  - ShareGPT 트래픽 또는 random 고부하 with GPU 포화
  - `T_hybrid = T_GPU + α·T_CPU` 의 α 측정
  - CPU tail 이 GPU wall time 안에 들어가는 부하 레벨 확인
- [ ] **Exp 2 — GPU latency impact (p99 preservation)** — 논문 Corollary 1
- [ ] **Exp 3 — 라우팅 전략 비교** (capacity / round-robin / length-aware / throughput-adaptive)
  - 주의: 벤치에서 random 고정 길이 데이터셋은 length-aware / throughput-adaptive 의 prefill_threshold 가 무의미. ShareGPT 같은 **길이 분포가 있는** 데이터셋으로 측정
- [ ] **Exp 4 — ablation** (NUMA binding / IPEX / auto config)
- [ ] **Exp 5 — 모델 크기 스케일링** (8B → 70B)
- [ ] **Exp 6 — 에너지 효율** (Intel RAPL counters 로 실측, 논문 Corollary 2)

---

## 4. 기타 관찰된 잠재 이슈

- [ ] **`set_num_interop_threads` 타이밍 확인**
  - 첫 op 실행 후에는 `RuntimeError` 를 던져 호출 불가
  - `try/except RuntimeError` 로 감쌌지만, 실제 H100 환경에서 interop thread 가 원하는 값으로 설정되는지 로그 확인 필요

- [ ] **`numa_migrate_pages` 가 2TB DRAM 환경에서 느릴 가능성**
  - 부팅 지연만 영향, 런타임 성능은 무관
  - H100 부팅 시간 측정

- [ ] **AMX tile permission 커널 버전 의존성**
  - Linux kernel 5.16+ 필요
  - `_enable_amx_tiles()` 가 ARCH_REQ_XCOMP_PERM syscall 로 permission 요청
  - H100 서버의 커널 버전 확인 필요

- [ ] **논문 §Limitations 의 "70GB weight 중복 로딩" 실측**
  - CPU engine 은 별도 프로세스이므로 weight 를 독립 로드
  - 7B 실측: GPU 14GB + CPU 14GB = 28GB (지금 확인됨)
  - 70B 환경에서 부팅 가능성 / startup time 측정

---

## 5. 문서화 잔여

- [ ] `docs/CUDA13_MIGRATION_STATUS.md` 는 작성됨 (이번 세션)
- [ ] H100 검증 후 `docs/CUDA13_MIGRATION_STATUS.md` 에 "H100 검증 결과" 섹션 추가
- [ ] 논문 draft 업데이트 (§2 위 "논문 ↔ 코드 재정합" 항목)
- [ ] `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md` 가 현재 코드와 맞는지 재확인 (설계 원칙이 바뀐 auto rule 반영 필요)

---

## v2 — 2026-04-11: 코드/로그 검증 세션 직후 상태 스냅샷

> append-only 정책. 위 v1 섹션 (0~5) 은 유지. 본 섹션은 v2 시점의 현황 업데이트 + 신규 항목만 기록.

### v1 이후 신규 완료 항목 (→ `Task_done.md v2`, `Tech_done.md v1` 참조)
- ✅ 4대 핵심 질문 코드/로그 검증 완료 (dev AVX2+NUMA1 매트릭스, `Tech_done.md v1`)
- ✅ 작업 기록 파일 3종 (TODO/Task_done/Tech_done) append-only 운용 규칙 확립

### v2 시점에서 여전히 남아있는 작업 (v1 의 모든 항목 유효)

**§0 uncommitted 변경 commit 정리** — 여전히 대기 중
- Group A~E 논리 분할은 v1 에 정의된 그대로 유효
- 본 세션 (v2) 은 코드 변경 없음. 다만 **신규 파일 3종** 이 추가됨:
  - `Tech_done.md` (신규)
  - `Task_done.md` v2 섹션 append
  - `TODO.md` v2 섹션 append (본 섹션)
  - → 이 세션 commit 은 "Group F: 작업 기록 / 검증 결론 문서화 (v2)" 로 **앞 세션 Group A~E 와 분리** 하여 단독 commit 하는 것을 권장 (검증 기록이 기존 코드 fix 와 섞이면 bisect 어려움)

**§1 dev 로직 검증 잔여** — 부분 진행
- ✅ 1 시퀀스 OMP 1:1 pinning 재확인 (v2 에서 C++ `init_cpu_threads_env` 경로로 확인)
- ✅ 500 req burst 라우팅 + CPU slot 반납 cycle 확인 (GPU 499 / CPU 2 완료)
- ⬜ 50회 이상 **순차 반복** 라이프사이클 (0→1→0 반복) 누수 확인 — burst 만 했지 순차 반복은 미수행
- ⬜ 종료 조건 다양화 (`length` / `stop` / `abort`) 별 slot 반납 검증
- ⬜ V1 Scheduler `cpu_max_num_seqs=1` 경계에서 preemption/reschedule 코드 경로 추적
- ⬜ H100 "capacity 에서 멈춤" 증상 원인 dev 배제

**§2 논문 ↔ 코드 재정합** — 미수행
- 4건 불일치는 이번 세션에도 그대로. 논문 patch 미작성 상태
- v2 에서 불일치 위치/근거만 재확인: `Tech_done.md v1` Q1 에서 `cpu_max_num_seqs=1` 고정 원칙이 코드·로그 양쪽에 일관되게 구현되어 있음을 증명 → 논문 Table 2 수정 정당성 확보

**§3 H100 타겟 검증** — 미수행
- `Tech_done.md v1` Q4 매트릭스 표에서 H100x4 / H100x8+Xeon 2S 행은 "코드 경로 존재, 실측 pending" 으로 기록됨
- 이 셀을 채우는 것이 §3 의 본질

**§4 기타 잠재 이슈** — 미수행

**§5 문서화 잔여** — 부분 진행
- ✅ `Tech_done.md` 신규 생성 (v2 에서 수행)
- ⬜ 논문 draft 업데이트
- ⬜ `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md` 재확인

### v2 에서 새로 발견된 작업/관찰

- [ ] **`post-init: cpu_affinity=1 cores [1]` 는 의도된 동작임을 문서화**
  - `Tech_done.md v1` Q1 에서 상세 설명. C++ `init_cpu_threads_env` 가 OMP worker pool 에만 1:1 pin 을 적용하고 main thread 는 core 1 에 남김
  - 혼란 방지용으로 `CLAUDE.md` 의 hybrid 진단 섹션에 1 줄 주석 추가 검토
- [ ] **dev 환경에서 CPU 완료 req 가 너무 적음 (500 중 2)**
  - 이는 dev 하드웨어 한계 (2.3 tok/s × 요청당 긴 decode) 이지 로직 버그 아님
  - 로직 검증 목적에는 충분하나, **순차 반복 (§1 의 첫 항목)** 로 slot 반납 cycle 을 여러 번 확실히 관측해야 `output.finished` 감지가 모든 경우에 동작함을 증명할 수 있음
- [ ] **`_C_cpu_ops` (AVX-512) 경로 실측 공백**
  - dev 는 AVX-512 없어 `custom_avx=0`. 이 경로는 H100x4 KVM 이상에서만 실행되므로, H100 검증 시 `_decode_path_counts['custom_avx'] > 0` 로 **반드시** 확인해야 함
  - Tech_done.md 매트릭스 표 에 해당 셀 채우기

