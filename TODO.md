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

---

## v3 — 2026-04-11: 진행 상태 스냅샷 (완료 항목 표시)

> append-only 정책: v1 / v2 섹션은 수정하지 않는다. 본 섹션은 v2 이후 시점의
> 진행 상태 스냅샷이며, v1 / v2 에 나열된 항목 중 **완료된 것을 명시적으로 표시**한다.
> 새 독자는 v1 → v2 → v3 순으로 읽고, "현재 남은 작업" 목록은 본 v3 섹션을 기준으로 삼으면 된다.

### v1 §0 — uncommitted 변경 commit 정리

**상태: ✅ 완료** (단 논리 분할은 포기하고 단일 commit 으로 통합)

- [x] Group A (`_C_utils` standalone extension) — commit `ad70b8f4d`, push 완료
- [x] Group B (WorkerBase 우회 + CPUWorker device_type coerce) — commit `ad70b8f4d`, push 완료
- [x] Group C (NUMA auto rule: num_cpu_engines=num_numa, max_seqs=1) — commit `ad70b8f4d`, push 완료
- [x] Group D (Python sched_setaffinity fallback + 진단 로그 7종 + execute_model trace) — commit `ad70b8f4d`, push 완료
- [x] Group E (문서 + eval env 파일들) — commit `ad70b8f4d` + `84e9c8201`, push 완료

**정정**: 본래 TODO v1 §0 은 Group A~E 를 5 개 논리 commit 으로 분할 권장했으나, 원격의 workaround hotfix 4 commits (`b147cafcc`, `19e7ed64f`, `f0b527824`, `c5349b799`) 와의 merge 복잡성을 해결하는 과정에서 **단일 bundled commit 으로 통합**했다. Bisect 어려움이 있을 수 있으나 전체가 하나의 논리 단위 ("hybrid CPU engine end-to-end fix") 이므로 수용한다. 다음부터는 초기 단계에서부터 원격과 로컬을 자주 sync 해서 이런 divergence 를 피해야 한다.

### v1 §1 — dev 로직 검증 잔여

**상태: 부분 완료**

- [x] **1-시퀀스 라이프사이클 기본 검증** — 500 req burst 에서 CPU 2 req 완료, `in_flight=1/1` slot 반납 cycle 관측. 근거: `Tech_done.md v1 Q2`
- [x] **OMP thread 1:1 pinning 재확인** — `init_cpu_threads_env (C++) returned` 로그에 16 tid ↔ 16 core mapping 명시. 근거: `Tech_done.md v1 Q1`
- [x] **16 core PSR 고정 매핑** — 동일 근거
- [ ] **50+ req 순차 반복** — burst 만 수행, 순차 반복은 미수행
- [ ] **동시 요청 스트레스 50+ burst 확장** — 현재 500 req 까지 확장되었으나 데드락 세부 검증은 미수행
- [ ] **CPU scheduler 코드 경로 트레이싱** — preemption / reschedule / chunked prefill 경계 검증 미수행
- [ ] **`output.finished` 감지 확실성 (length/stop/abort)** — 단일 종료 경로만 관측
- [ ] **H100 capacity 멈춤 증상 dev 배제** — 미수행

### v1 §2 — 논문 ↔ 코드 재정합

**상태: ❌ 미수행** (전체 pending)

- [ ] 논문 Table 2 `max_seqs` auto rule 수정 (`max(4, ⌊cores/4⌋)` → `1 per NUMA engine`)
- [ ] 논문에 `num_cpu_engines = num_numa` auto 규칙 추가
- [ ] §3.3 Algorithm 1 의 `N` 표기 명확화 (`N = 1 per engine × num_numa`)
- [ ] §5 Implementation 에 `_C_utils` standalone extension 언급 추가

**추가 관찰 (v3)**: `Tech_done.md v1` 이 dev 실측으로 `cpu_max_num_seqs=1` 원칙의 타당성을 증명했으므로, 논문 Table 2 수정의 근거 자료는 확보됨.

### v1 §3 — H100 타겟 환경 검증

**상태: ❌ 미수행** (전체 pending)

- [ ] H100x4 KVM 재측정
- [ ] H100x8 + Xeon 8480+ 2-socket 실 환경 첫 부팅
- [ ] Exp 1 — end-to-end throughput
- [ ] Exp 2 — GPU latency impact (p99 preservation)
- [ ] Exp 3 — 라우팅 전략 비교 (ShareGPT 필수)
- [ ] Exp 4 — ablation study
- [ ] Exp 5 — 모델 크기 스케일링
- [ ] Exp 6 — 에너지 효율 (Intel RAPL)

### v1 §4 — 기타 잠재 이슈

**상태: ❌ 미수행** (전체 pending)

- [ ] `set_num_interop_threads` 타이밍 확인
- [ ] `numa_migrate_pages` × 2TB 지연 측정
- [ ] AMX tile permission 커널 버전 의존성 확인
- [ ] 70B weight 중복 로딩 실측

### v1 §5 — 문서화 잔여

**상태: 부분 완료**

- [x] `docs/CUDA13_MIGRATION_STATUS.md` 작성 — 원격 `c5349b799` 로 이미 존재 (crossmachine handoff 용)
- [ ] H100 검증 후 `docs/CUDA13_MIGRATION_STATUS.md` 에 "H100 검증 결과" 섹션 추가
- [ ] 논문 draft 업데이트 (v1 §2 항목)
- [ ] `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md` 가 현재 코드와 맞는지 재확인

### v2 에서 제기된 항목

**상태: 부분 완료**

- [x] **`post-init: cpu_affinity=1 cores [1]` 는 의도된 동작임을 문서화** — `Tech_done.md v1 Q1` 에 "main thread 는 core 1 하나에만 affinity, OMP worker pool 16 개가 16 core 에 pin. matmul 병렬은 정상" 로 상세 설명
- [ ] CLAUDE.md 에도 1 줄 주석 추가 (혼란 방지) — v3 에서도 미추가
- [ ] dev 에서 CPU 완료 req 가 너무 적음 — 순차 반복 (§1) 로 추가 관측 필요
- [ ] `_C_cpu_ops` AVX-512 경로 실측 공백 — H100 에서 확인 필요 (§3 에 종속)

### v3 에서 새로 완료된 항목

- [x] **CLAUDE.md 하드웨어 호환성 매트릭스 제거** — "단일 코드베이스 + 런타임 감지 + graceful fallback" 원칙으로 재기술
- [x] **CLAUDE.md 타겟 하드웨어 섹션 삭제** — 특정 기종 못 박지 않고 "x86_64 + NVIDIA GPU 만 요구, 구체 스펙은 논문 평가 환경 참조" 로 대체
- [x] **CLAUDE.md "dev / H100 동일" 빌드 주석 정리** — "환경 구분 없이 동일" 로 일반화
- [x] **`Tech_done.md v2` append — "실측 완료 환경만 나열" 원칙** + v1 Q4 매트릭스 정정 (실측 환경은 dev 1 행만 유효, 나머지 3 행 제외)
- [x] **메모리 `feedback_work_log_files.md` 에 Tech_done.md 전용 원칙 추가** — "미검증 항목은 TODO.md 로"
- [x] **`Task_done.md` / `Tech_done.md` / `TODO.md` 세 파일 append-only 정책 운용** — 메모리 `feedback_work_log_files.md` 에 규칙화

### v3 에서 새로 추가된 작업

- [ ] **다음 작업 세션 시작 시 stale 여부 재확인** — v3 작성 후 시간이 지나면 일부 "미수행" 항목이 실제로는 해결되었을 수 있음. 다음 세션 첫 단계에서 `git log` / 로그 파일로 크로스체크 후 v4 append

### "현재 남은 작업" 한눈 요약 (v3 기준)

높은 우선순위:
1. v1 §1 의 미완 검증 (순차 반복, output.finished 다양 종료 조건, CPU scheduler 코드 경로) — dev 에서 가능
2. v1 §2 논문 정정 (4 항목) — 로컬에서 가능, H100 불필요
3. v1 §3 H100 실측 — dev 검증 완료 후 실행

낮은 우선순위:
4. v1 §4 잠재 이슈 4 건
5. v1 §5 문서화 잔여

---

## v4 — 2026-04-11: §1 dev 로직 검증 완결 + Abort slot leak 버그 수정

> append-only 정책 유지. v1/v2/v3 섹션은 수정 없음. 본 v4 는 당일 후속 세션의 진행 스냅샷.
> 자세한 실측 결과 / 버그 재현 로그 / 수정 diff 는 `Tech_done.md v3` + `Task_done.md v3` + `experiment_result/20260411_06*/` 참조.

### v1 §1 — dev 로직 검증 잔여

**상태: 실질적 완결** (H100 capacity 멈춤 dev 배제 항목까지 포함)

- [x] **1-시퀀스 라이프사이클 반복 검증** — v3 에서 60 req 순차 완결 (`seq_repeat_test.py`). 평균 1.58s/req ±0.05s, Router `in_flight=0/1`, 누수 zero. → `Tech_done.md v3 F3`
- [x] **동시 요청 스트레스 50+ burst** — v3 의 1.5B/7B 500 req burst 로 이미 커버 (`Tech_done.md v3 F1`)
- [x] **CPU scheduler 코드 경로 트레이싱** — v3 에서 코드 분석 완결. CPU 엔진은 `max_num_seqs=1` + `chunked_prefill=False` 로 표준 경계 메커니즘 사용, 경계 edge case 없음. → `Tech_done.md v3 F6`
- [x] **`output.finished` 감지 확실성 (length/stop/abort)** — v3 에서 length/stop 정상 + **abort 치명적 버그 발견 → 수정 → 검증 완결**. → `Tech_done.md v3 F4, F5`
- [x] **H100 "capacity 에서 멈춤" 증상 원인 dev 배제** — v3 에서 **dev 재현 성공**. 원인 = abort slot 누수 (DP=1 + `include_finished_set=False` 에서 `EngineCoreOutputs.finished_requests` 영원히 empty + aborted req 는 새 토큰 없어 `output.finished` 도 emit 안 됨 → 어느 경로로도 slot 반납 신호 안 옴). 수정 후 dev 에서 재현 안 됨. → `Tech_done.md v3 F5`

**의의**: §1 가 실질적으로 완결됨. H100 운영 전 이 패치 (`vllm/v1/engine/core_client.py::abort_requests*`) 가 반드시 포함돼야 함.

### v1 §2 — 논문 ↔ 코드 재정합

**상태: 여전히 ❌ 미수행** (변화 없음). 4 건 pending.

### v1 §3 — H100 타겟 환경 검증

**상태: 여전히 ❌ 미수행** (변화 없음). 단 중요 전제 변경:

- [ ] **H100 이관 전 필수 전제**: 본 v4 에서 수정한 `abort_requests*` 패치가 코드베이스에 포함된 상태여야 함. 패치 없는 이전 commit 에서 H100 에 올리면 client disconnect 한 번이면 capacity 영구 stuck.
- [ ] 나머지 7 항목 (Exp 1~6, H100x4 재측정, H100x8 2S 첫 부팅) 그대로 pending.

### v1 §4 — 기타 잠재 이슈

**상태: 여전히 ❌ 미수행** (변화 없음). 4 건 pending.

### v1 §5 — 문서화 잔여

**상태: 부분 완료** — v4 에서 새로 완결:

- [x] `experiment_result/20260411_063046_dev_rtx3090_1.5B_7B_hybrid_verify/` — 1.5B/7B 재측정 결과 저장 (README + environment + summary + raw artifacts)
- [x] `experiment_result/20260411_065041_dev_logic_verify_abort_slot_leak_fix/` — §1 로직 검증 + abort 버그 재현/수정/검증 기록 저장 (README + patch diff + 3 test scripts + 3 result JSONs + server logs before/after)

여전히 pending:
- [ ] H100 검증 후 `docs/CUDA13_MIGRATION_STATUS.md` "H100 검증 결과" 섹션 추가
- [ ] 논문 draft 업데이트 (v1 §2 항목)
- [ ] `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md` 재확인

### v2 / v3 제기 항목

**상태: 부분 완료**

- [x] `post-init: cpu_affinity=1 cores [1]` 은 의도된 동작임을 `Tech_done.md v1 Q1` 에 기록됨 (v4 에서 CLAUDE.md 주석 추가는 여전히 미진행)
- [x] **dev 에서 CPU 완료 req 가 너무 적음** — 60 req 순차 반복으로 slot cycle 이 여러 번 관측됨 (v3 F3)
- [ ] `_C_cpu_ops` AVX-512 경로 실측 공백 — H100 에서 확인 필요 (§3 에 종속)
- [ ] 다음 작업 세션 시작 시 stale 여부 재확인 — v4 작성 시점에서 stale 없음 확인 완료

### v4 에서 새로 추가된 작업

- [ ] **H100 smoke test 재시도** — v4 의 abort slot 누수 패치 포함 상태에서 H100x4 에 올려 부팅 + 10 req smoke 우선 확인. 이전에 "capacity 에서 멈춤" 으로 관찰된 증상이 재현 안 되면 이 패치가 그 증상의 직접 원인이었음이 H100 에서도 확정됨.
- [ ] **Long-running production smoke** — H100 smoke 가 통과하면 분 단위로 client abort 가 섞여 들어가는 시나리오 (health check timeout, LB graceful restart 등) 를 반복 시뮬레이션해 slot counter 가 끝까지 0 으로 돌아오는지 확인

### "현재 남은 작업" 한눈 요약 (v4 기준)

높은 우선순위:
1. v1 §2 논문 정정 (4 항목) — 로컬 가능
2. v1 §3 H100 실측 — v4 패치 포함 + dev 로직 검증 완결 상태에서 시작 가능 (준비 완료)
3. v4 의 H100 smoke test 에서 abort 버그 증상 해소 확인

낮은 우선순위:
4. v1 §4 잠재 이슈 4 건
5. v1 §5 문서화 잔여 (논문 + HYBRID_OPTIONS plan)

