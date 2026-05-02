# ID Registry

CLAUDE.md Ground RULE 의 ID Rule 에 따라, 본 저장소에서 사용되는 영문 3글자 prefix 별 **넘버링 규칙** 과 **ID 할당 현황** 을 본 파일에서 단일 출처로 관리한다.

> 본 파일은 ID 의 "할당·상태" 만 다룬다. ID 간 파생 관계(Tree) 는 `shadow_assists/README.md` Part VII Trace Tree, ID 의 본문 정의는 각 feature 디렉토리의 `README.md` 가 담당한다.

## 공통 넘버링 규칙

1. 번호는 **prefix 별로 독립된 카운터** 를 가진다. `IDE_###` 의 번호 공간과 `PLN_###` 의 번호 공간은 분리된다.
2. 번호는 **1 부터 시작하고 1 씩 증가** 한다 (예: `IDE_001`, `IDE_002`, …).
3. 번호는 **3 자리 zero-padding** (`001` ~ `999`).
4. 한 번 부여된 번호는 **재사용하지 않는다**. 기각·재정의된 ID 도 번호는 그대로 보존하고 상태(status)만 갱신한다.
5. 새 ID 부여 시 본 파일의 해당 prefix 섹션에서 "**다음 부여 번호**" 를 가져온 뒤, 같은 섹션의 표에 새 항목을 **즉시** 추가하고 "다음 부여 번호" 를 +1 한다. 본문에서의 사용은 그 이후.
6. 상태(status) 값은 다음 중 하나로 통일한다: `활성` / `대기` / `재정의` / `기각` / `완료`.

## 상태(status) 정의

| 상태 | 의미 |
|---|---|
| `활성` | 현재 진입 조건 충족 또는 작업 중 |
| `대기` | 정의되어 있으나 진입 조건 미충족, 측정·판정 대기 |
| `재정의` | 동일 ID 로 정의 내용이 1 회 이상 교체됨. 본문에 재정의 사유 명시 필요 |
| `기각` | 진입 조건 미달 또는 중복 사유로 폐기. 번호는 재사용 안 함 |
| `완료` | 작업 완료 후 더 이상 변경 예정 없음 |

---

## Prefix: `IDE` — Idea

구현 후보 단계. profile / 측정 결과로 진입·기각 판정 후에야 다음 단계 prefix(`PLN` 등) 로 파생된다.

**다음 부여 번호**: `IDE_009`

| ID | 상태 | 제목 | 비고 |
|---|---|---|---|
| `IDE_001` | 대기 | CPU-assisted 동적 배치 planner | Phase 0 profile 진입 조건 대기 |
| `IDE_002` | 대기 | CPU prefill-assist (medium context) | 512~2K GPU compute-bound 확인 대기 |
| `IDE_003` | 대기 | CPU Background Compiler | quantization 정합성 확인 대기 |
| `IDE_004` | 대기 | GPU-idle-phase CPU Burst | sublayer phase 감지 가능성 확인 대기 |
| `IDE_005` | 대기 | CPU drafter + GPU verifier | 다중화 request 큐 가정 검증 대기 |
| `IDE_006` | 재정의 | Cold-KV CPU Partial Attention | 1 차 정의 (Cold KV staging) 업스트림 중복으로 기각 후 재정의 (commit `8f50eeb2a4`). 2 차 정의 = "GPU reload 없이 CPU partial attention". 3 차 재정의 (2026-04-28) = "Speculative CPU partial + deadline 기반 GPU full FA fallback 으로 throughput 하한 보장". GPU memory 절약은 가치 축에서 제외, 단일 가치 축 = 시스템 throughput 향상 |
| `IDE_007` | 대기 | CPU Speculative Logits Rerank | 정확도 영향 측정 대기 |
| `IDE_008` | 대기 | Constrained Decoding 전담 CPU Worker | constrained workload 시점 대기 |

---

## Prefix: `PLN` — Plan

IDE 의 진입·정확도·throughput 가정을 풀기 위한 PoC / microbench 플랜. PLN 결과에 따라 `FEA_###` 진입 또는 IDE 기각.

**다음 부여 번호**: `PLN_002`

| ID | 상태 | 제목 | 비고 |
|---|---|---|---|
| `PLN_001` | 활성 (Phase 1 dev) | Cold-KV CPU Partial Attention PoC 플랜 | 부모 `IDE_006`. Phase 1 (dev simulation — ≥8K prompt 합성, RTX 3090 + 12900KF) 진행 중. Phase 2 (prod — 운영 (a) 충족 후 Xeon SPR + H100×8) 는 사용자 직접 진행 |

---

## Prefix: `TSK` — Task

FEA 구현을 위한 단계별 작업 단위. CLAUDE.md Method 의 feature 디렉토리 내 `task.md` 항목과 매핑된다.

**다음 부여 번호**: `TSK_019`

| ID | 상태 | 제목 | 비고 |
|---|---|---|---|
| `TSK_001` | 완료 | LSE-반환 CPU partial-attention kernel — dev 부분 | 부모 `PLN_001`. 4.0 KVViewAdapter / 4.1 Python reference / 4.2c portable C++ / 4.3 wrapper. 검증 게이트 = `TST_001` (A·B(i)·C — dev 87 + prod 87 pytest 통과, 2026-04-26). prod SIMD (AVX-512/AMX) 는 `TSK_003` 으로 이관 |
| `TSK_002` | 기각 (2026-05-02 — NEO 4차 재정의 후 hot/cold partition 메커니즘이 TSK_018 NEO pacpu 의 cdec dispatch hook 으로 흡수. Phase 1 dev §4.2~§4.5c + §4.6 stream 분리 본문은 history archive 보존) | scheduler / attention metadata 의 hot/cold partition 통합 | 부모 `PLN_001`. `TSK_001` 후속. **2026-05-02 기각**: 4차 재정의 (TSK_013~TSK_018) 적용 후 *partition path 자체* 가 발화 안 함 (cdec_reqs 가 전체 attention 처리). 본 TSK 의 §4.5c 두 단계 알고리즘 / §4.6 stream 분리 / overlap fix 흐름은 reference 보존 (`PLN_001_TSK_002_02_overlap_fix_log.md`). vLLM 통합 인프라 일부 (scheduler / attention metadata 변경 패턴) 는 NEO 4차 적재의 기반으로 재사용됨. 검증 게이트 = `TST_003` (e2e 통합 정확도) — 동행 기각. ID 보존 (재사용 금지) |
| `TSK_003` | 활성 (Phase 1 dev 구현·컴파일·dispatch 완료, Phase 2 prod 정확도 측정 대기) | prod SIMD kernels — AVX-512 + AMX | 부모 `PLN_001`. 선행 `TSK_001` (portable + algorithm reference). §4.2a AVX-512: `csrc/cpu/partial_attention_avx512.cpp` — head_dim dot product 를 AVX-512F (+ optional AVX512_BF16 vdpbf16ps) intrinsics 로 교체, outer 3-pass 구조는 portable 동일. §4.2b AMX: `csrc/cpu/partial_attention_amx.cpp` — BF16 입력에서 `_tile_dpbf16ps` 로 16 K rows × head_dim chunks (32 BF16 단위) 를 batched matmul. FP16/FP32 는 AMX 무관이라 같은 TU 내 AVX-512 fallback. 두 kernel 모두 `_kernel_source_path` JIT load + cpuid 게이트 (`_has_avx512()` / `_has_amx()`) 로 fuse-off CPU 의 static-init SIGILL 회피. 검증 게이트 = `TST_004` (portable vs AVX-512 / portable vs AMX cross-check, BF16 ~5e-3 / FP16 ~1e-3 tolerance). dev 검증: 양 kernel `g++ -mavx512f -mavx512bf16 -mamx-tile -mamx-bf16` 컴파일 clean, pytest 93 통과 + 80 cross-check 자동 skip (AVX-512 fuse-off + AMX hardware 미지원). prod 측 정확도 cross-check 는 사용자 (Xeon SPR+ + H100×8) 가 `bash eval/run_prod_smoke.sh` 또는 `pytest tests/v1/cpu_partial_attention/test_avx512_cross_check.py test_amx_cross_check.py` 로 직접 |
| `TSK_004` | 활성 (Phase 1 dev 구현 완료, Phase 2 prod 측정 대기) | Cold-KV 경로 NUMA-aware 화 — connector buffer + partial-attention 커널 | 부모 `PLN_001`. (a) `vllm/distributed/kv_transfer/kv_connector/v1/offloading/numa_aware.py` 의 `bind_worker_to_local_numa()` 가 `CpuGpuOffloadingHandlers.__init__` 직전에 호출되어 worker 의 local NUMA node 를 `numa_set_preferred` 로 지정 → 이후 `torch.zeros(..., pin_memory=True)` 가 자기 socket 에 박힘. (b) `pin_threads_to_local_numa()` 가 `forward_partial_with_lse` 첫 진입 시 `os.sched_setaffinity` 로 worker 프로세스의 코어 affinity 를 자기 NUMA node 코어로 제한 → OpenMP / std::thread 가 cross-socket 으로 못 빠짐. 두 hook 모두 idempotent + libnuma 미설치 / 단일 socket 시 silent no-op. local node 결정 우선순위: GPU NUMA node → rank % num_nodes → skip. 검증은 `TST_002` (throughput / overlap profile) 에 흡수. dev 검증: `TST_001` 회귀 없음 (93/93), e2e smoke 회귀 없음. prod 측정은 `TST_002` sweep 단계 (사용자 직접) |
| `TSK_005` | 기각 (2026-04-29 — fundamental Q dependency dilemma) | Cross-layer pipeline parallelism — layer N cold CPU work 와 layer N+1 hot FA work 의 overlap (NEO 식 asymmetric pipeline) | 부모 `PLN_001`. 출처 README §8 risk vii / §9 (g) / §12 Q6 + TSK_002 §10 Change Log 2026-04-27 (deferred). 2026-04-29 사용자 지적 — `Q dependency + GPU 가 진짜 Q 가지면 CPU 결과 무용` dilemma 식별. layer N+1 attention 진입 시점에 GPU 가 진짜 Q 를 가지므로 paged FA full 직접 가능 → CPU 결과 활용 의미 없음. speculative Q 추정 영역도 동일 (correction 자체가 GPU paged FA 와 같은 작업). CPU partial 의 *진짜* 가치 영역 = cold blocks 가 진짜 GPU evict 되는 시점 (`TSK_012`) 의 reload 대체로만. **기각**. ID 보존 (재사용 금지) |
| `TSK_006` | 기각 (2026-05-02 — NEO 4차 재정의 후 cold-KV partition path 자체 발화 안 함, 본 TSK 의 chunk pipelining 영역 dead) | Q chunk pipelining (Q D2H 를 chunk 단위 split + CPU kernel 과 stream-level pipeline) | 부모 `PLN_001`. 출처 README §12 Open Q6, TSK_002 §4 표 §4.6 본문. **2026-05-02 기각**: NEO 4차 재정의 (TSK_013 ~ TSK_018) 적용 시 cold-KV partition path 가 NEO 의 cdec dispatch hook 으로 *흡수* — partition path 자체가 발화 안 함. 본 TSK 의 chunk pipelining 메커니즘은 *NEO pacpu (TSK_018)* 의 batch-level dispatch 에 흡수. ID 보존 (재사용 금지). 본문 reference 만 보존. |
| `TSK_007` | 완료 (2026-04-28 — microbench 9 cell sweep 결과 옵션 A 채택, 코드 변경 없음) | GQA K/V broadcast 옵션 결정 (옵션 A compact+broadcast vs B pre-expand) | 부모 `PLN_001`. 출처 README §4.3, §8 risk iv, §12 Open Q3. **결과** (`eval/run_tsk007_gqa_microbench.py`, Llama-3.3-70B + TP=8 worker per 9 cell): 모든 cell 에서 옵션 A 가 1.36×~3.84× 빠름. 원인 — kernel 안 broadcast 비용 (h/q_per_kv division 한 번) << expanded buffer cache miss 비용 (8× 메모리가 L2/L3 못 fit). 옵션 A (현재 코드) 유지. 검증 게이트 = `TST_007` 완료 |
| `TSK_008` | 기각 (2026-05-02 — NEO 4차 재정의 후 hot/cold 분할 정책 영역 dead) | hot/cold 분할 정책 (layer-별 독립 vs request 전체 균일) | 부모 `PLN_001`. 출처 README §12 Open Q4. **2026-05-02 기각**: NEO 4차 재정의 적용 시 KV ownership 이 *layer 별 hot/cold ratio* 가 아닌 *request 단위 GPU/CPU exclusive* (TSK_015) 로 변경 → 본 TSK 의 분할 정책 영역 자체가 dead. ID 보존 (재사용 금지). |
| `TSK_009` | 활성 (2026-04-29 — fix v4 land. invariant 1 acceptable, invariant 2 → TSK_005 분리) | Cold-path non-blocking dispatch (CPU 가 GPU 작업 방해 없이 best-effort 활용) | 부모 `PLN_001`. 사용자 framing (2026-04-29): CPU 가 GPU 방해 안 함 + cold tier host 도착 순간부터 partial 진행 + 미도착 시 폐기 + GPU 가 스스로 계산. **fix v4 land** (2026-04-29): `hot_cold_attention` 의 hot path 호출을 *done 분기 helper* 로 지연. `future.done()` non-blocking poll → done = hot subset paged FA + LSE merge / not done = `future.cancel()` + paged FA full inplace + return. GPU FA 호출 = 항상 1번. prod 6 회차 검증 (`eval/results/20260429_043734_*_tsk009_validation/`): C/B = 1.078~1.103× (input_heavy/output_heavy/equal × 100 prompts × Llama-70B + TP=8) — invariant 1 acceptable. merged % = 0.00% (fundamental Q dependency) — invariant 2 → `TSK_005` 영역으로 분리. 검증 게이트 = `TST_009` 완료. **이전 정의들** (1차 admission prefetch, race-and-merge, overlap profile, baseline mode) 모두 폐기 — Change Log 보존 |
| `TSK_010` | 기각 (2026-05-02 — NEO pacpu 가 multi-OMP-team 자원 활용을 자체 흡수) | CPU 자원 활용 확장 — multi-OMP-team / sub-batching | 부모 `PLN_001`. 출처: 사용자 허락 후 발행 (2026-04-28). **2026-05-02 기각**: NEO pacpu (TSK_018) 의 ISPC + OpenMP 가 batch-level sub-batching + multi-team CPU 자원 활용을 자체 구현 — 본 TSK 의 별도 메커니즘 불필요. NEO 식 적용 후 *single TSK_018* 가 CPU 자원 영역을 cover. ID 보존 (재사용 금지). |
| `TSK_011` | 기각 (2026-05-02 — NEO 4차 재정의 후 partition path 자체 dead, fallback 안전망 의미 영역 영구 소멸) | Speculative cold + GPU fallback (3차 재정의 핵심 코드화) | 부모 `PLN_001`. 출처: IDE_006 §3.1 3차 재정의 (2026-04-28) 의 코드화. **prod sweep (2026-04-28)**: deadline=100ms / 1000ms 모두 baseline 과 lp ~3.43 / ppl ~0.24 동일 발산 — partition path / fallback path 둘 다 *같은 cold KV source (CPU page cache)* 사용으로 같은 root cause. throughput 하한 보장 + worst-case stall 차단은 입증, D-ii 봉합 불가능 → `TSK_012` 분리. **2026-04-29 의미 재정의**: (1) TSK_009 fix v4 가 본 TSK 의 *blocking deadline* (`future.result(timeout=deadline_s)`) 을 *non-blocking poll* (`future.done()`) 로 대체. fallback 도 *paged FA full inplace* (GPU FA 1회) 로 통합 — 사용자 framing "CPU 가 GPU 작업 방해 안 함" 정합. (2) TSK_005 기각으로 partition path 의 진짜 가치 영역 자체가 TSK_012 영역으로 이전. (3) TSK_012 적용 후 cold blocks 가 GPU paged 에 reload → standard hot FA 가 모든 KV 처리 → IDE_006 partition path 발화 안 함 → 본 TSK 의 deadline / fallback *전부 dead code* (예외 안전망 only). 한때 검토되던 race framing 은 측정 데이터로 가치 영역 작음 입증 후 제거 (2026-04-29). 본 TSK 의 코드는 land 그대로 유지 — 호출 패턴만 fix v4 가 변경 |
| `TSK_012` | 기각 (2026-05-02 — NEO 의 request 단위 exclusive ownership 이 본 TSK 의 cold-blocks 단위 evict + reload 메커니즘을 흡수) | Decode-time cold-blocks GPU reload + 진짜 evict 정책 (IDE_006 의 진짜 가치 영역) | 부모 `PLN_001`. 선행: `TSK_011` (fallback 안전망 + sweep 결과의 발견) / `TSK_009` fix v4 (non-blocking dispatch) / `TSK_005` 기각 (Q dependency dilemma 로 cross-layer 영역 폐기 — 본 TSK 가 IDE_006 의 *유일* 가치 영역). **단일 단계 (D-ii 봉합)**: cold blocks 진짜 evict (mirror → swap, vLLM upstream 변경) + decode admission 시점 reload trigger. attention 은 standard hot FA 가 모든 KV 처리. partition path *발화 안 함*. D-ii 봉합 ✓ + KV pool 압박 완화. invariant 2 (CPU 활용) 는 본 TSK 의 목표 영역 *밖*. per-request wall-time 차원에서는 vanilla 보다 *항상 느림* (reload PCIe 비용) — 단 capacity 차원 (KV pool overflow 영역) 에서 진짜 evict 로 동시 in-flight request 수 확장 가능 (측정으로 입증 필요). 한때 검토되던 Phase 2 (race) 는 측정 데이터로 *vanilla 대비 항상 느림* 입증 (PCIe Gen4 32 KB cold block 0.002 ms vs CPU partial 6.4 ms) → 제거. 본문 sequence diagram 2 개 (현재 fix v4 / 본 TSK 적용 후). 검증 게이트 = `TST_012`. **2026-04-29 4 차 재정의 후 본문 폐기 영역**: NEO 식 architecture 로 전환되면서 본 TSK 의 *cold blocks 단위 evict + reload* 메커니즘이 NEO 의 *request 단위 exclusive ownership* 으로 흡수. NEO 식 적용 시 본 TSK 는 폐기 또는 본문 재작성 (TSK_013 분석 결과 후 결정) |
| `TSK_013` | 활성 (2026-04-29 — NEO 4 차 재정의 후 발급) | NEO repo 분석 + vLLM 위 적용 plan + 신규 TSK 발급 (Phase 0) | 부모 `PLN_001`. 출처: [`NEO_redesign.md`](features/IDE_006/NEO_redesign.md) 의 §4 신규 TSK 영역. NEO repo (`https://github.com/NEO-MLSys25/NEO`) 를 `/workspace/neo_ref/` 에 clone 완료. NEO 는 자체 LLM 엔진 (`swiftllm` + `pacpu`) — vLLM 의 fork 가 아니므로 *port / 모방* 영역. 본 TSK 의 산출물: (1) `PLN_001_TSK_013_neo_arch_survey.md` — NEO 의 핵심 파일 (`swiftllm/server/scheduler.py` / `executor.py` / `engine.py` / `block_manager.py` / `worker/model.py` / `worker/block_swapper.py` / `pacpu/`) 의 역할 + vLLM hook 매핑. (2) 신규 TSK 발급 plan — Request scheduler / KV exclusive ownership / Asymmetric pipelining / Load-aware scheduling / CPU attention kernel port 의 5 개 영역. 검증 게이트 = `TST_013` |
| `TSK_014` | 활성 (2026-04-30 — 1.1~1.7 단계 land 완료) | Request-level scheduler (3 큐 + load-aware mode selection, NEO §3) | 부모 `PLN_001`. NEO `swiftllm/server/scheduler.py` 의 알고리즘 차용 ([`NEO_code_deepdive.md`](features/IDE_006/NEO_code_deepdive.md) §3). 변경 범위 land 영역: `vllm/v1/core/sched/sub_batch.py` (SubBatch + BatchPerfData) / `mode_selector.py` (5 단계 decide_mode + `_get_remains`) / `neo_scheduler.py` (6 단계 batch 결정) / `neo_scheduler_adapter.py` (vLLM `Scheduler` interface 와 NEO sibling 동기 + `SchedulerOutput.neo_*` 첨부) / `SchedulerOutput` attribute 확장 / `--enable-neo-asymmetric` flag passthrough. 기존 plan 의 `TwoSubBatchSchedulerOutput` 별도 class 는 attribute 확장으로 변경 (downstream 호환). 의존: `TSK_017` (PerfPredictor) 선행 — 현재 `ZeroPerfPredictor` 로 sequential 만 선택 (vanilla 동등). 검증 게이트 = `TST_014`. |
| `TSK_015` | 활성 (2026-05-02 — Phase 1+2+3+4(4.1~4.6)+5.1~5.3(부분) land. dispatch 인프라 100% (Step3.2.B + Step3.2.C.1~C.8). **§3.4 B-1 ~ B-4 land**. **B-5 영구 deferred** (dev hook design fault). **B-6 prod-only** (§3.5 P-1~P-5 절차)) | KV cache exclusive ownership (mirror → exclusive, NEO §5) | 부모 `PLN_001`. NEO `swiftllm/server/block_manager.py` 의 알고리즘 차용 ([`NEO_code_deepdive.md`](features/IDE_006/NEO_code_deepdive.md) §5). 변경 범위: vLLM `OffloadingConnector` 의 mirror 정책을 *request 단위 GPU/CPU exclusive* 로 변경 + `kv_cache_manager.py` + `cpu_gpu.py` 의 swap-in/out lifecycle. `_initiate_swap` 의 atomic (source 에서 free + dest 에 alloc) 패턴 port. **B-1**: `_handle_neo_swaps` 의 `running.remove` 제거 (cdec 가 `Scheduler.running` 에 유지). **B-2.a/b**: `Scheduler.schedule()` 의 `SWAPPED_OUT` skip + 정상 schedule (`KVCacheManager.allocate_slots` dummy success). **B-3.a/b/c**: finish-swap mutex / worker `self.requests` SWAPPED_OUT 보존 / `slot_mapping` PADDING_SLOT_ID 센티넬. **B-4**: fork branch contiguous check 재검증. **B-5**: 영구 deferred. **B-6**: prod 측정 (§3.5 P-1~P-5). 의존: 독립 (TSK_014 와 병렬 가능). 검증 게이트 = `TST_015`. |
| `TSK_016` | 활성 (2026-04-30 — Step 5.1~5.5 통과 / Step 5.6 모델 확장 + 5.7 TST_016) | Asymmetric pipelining (sub-batch 동시 실행, NEO §4) | 부모 `PLN_001`. NEO `swiftllm/worker/model.py:_forward_pipeline` 의 layer 단위 ping-pong port ([`NEO_code_deepdive.md`](features/IDE_006/NEO_code_deepdive.md) §4). **vLLM 적재 영역 Step 5.1~5.5 land**: `gpu_model_runner._model_forward` 의 dual forward 분기 + per-sub-batch `ForwardContext` × 2 + 입력 slice + `forward_neo_pipelined(per_subbatch_contexts=...)` + `torch.cat` 머지 / `vllm/v1/worker/sub_batch_executor.py` (LayerPipelineCallbacks + SubBatchPipelineExecutor) / Llama + Qwen2 두 모델의 `neo_preproj` / `neo_attention` / `neo_postproj` + `forward_neo_pipelined` (`per_subbatch_contexts` kwarg + `override_forward_context` push/pop) / `execute_model` 의 NEO sub-batch → ubatch_slices 변환 + `should_ubatch=True` + vLLM `split_attn_metadata` 인프라 reuse / `initialize_metadata_builders` 의 `enable_neo_asymmetric=True` 시 자동 2 builders. dev smoke (RTX 3090 + Qwen-1.5B + `VLLM_NEO_FORCE_PIPELINED=1`) 통과: `[NEO-DEBUG] forward-context fork active: split_point=4 (boundary req=1, sub-batch sizes=1/1)` → `forked dual forward (token_slices=[(0, 4), (4, 8)], attn_metadata layers per sb=[28, 28])` → `merged shape=(8, 1536)` → token-id equality PASS. **NEO 의 layer-offset ping-pong (forward_double 의 batch[1] layer i + batch[0] layer i+1) 은 callback orchestration 영역에 그대로 보존** — TSK_018 CPU pacpu 후 진짜 GPU/CPU dual forward 효과 발현 가능. 의존: TSK_014 ✓ + TSK_015 (KV exclusive 의 swap lifecycle 병렬). 검증 게이트 = `TST_016` (TSK_017 PerfPredictor 실측 후). |
| `TSK_017` | 활성 (2026-04-30 — Step 1.1~1.6 land. PerfPredictor 실측 활성 / Step 1.7 disk cache + 1.8 TST_017 남음) | Load-aware scheduling heuristic (PerfPredictor, NEO §6) | 부모 `PLN_001`. NEO `swiftllm/perfpredictor.py:TablePerfPredictor` 의 4 종류 prediction + 1D/2D linear interpolation port ([`NEO_code_deepdive.md`](features/IDE_006/NEO_code_deepdive.md) §6). **vLLM 적재 영역 1.1~1.4 land**: `vllm/v1/core/sched/perfpredictor.py` (`PerfPredictor` / `ZeroPerfPredictor` / `TablePerfPredictor` + `_get_lb_idx_list` + `_interp_1d` + bilinear `get_cdec_T`) + `vllm/v1/metrics/profiler.py` (`ModelProfiler` 추상 + nwarmup=2 + nrepeat=3 + 4 table 채움 알고리즘). **다음 적재 (1.5~1.6)**: engine-side `measure_fn` callback (`vLLM` `_dummy_run` 인프라 reuse) + LLMEngine startup wiring (model load + KV alloc 후 `ModelProfiler.run()` 호출 + `adapter.predictor` 를 `table_predictor` 로 swap). **caching (1.7)**: `(model, gpu_arch, dtype, tp_size, max_num_seqs, block_size)` hash key 로 disk save/load (60s cold-start 비용 회피). 의존: 독립 (TSK_014 의 mode_selector 입력). 검증 게이트 = `TST_017`. |
| `TSK_018` | 활성 (2026-05-02 — Phase 1+2+3.1~3.2+3.4+3.5 land. **3.3 BF16↔FP16 cast 만 prod-only deferred** (TSK_015 §3.5 P-3 합체). 모델 macro 확장 — Qwen-1.5B/7B/32B/72B + Llama-70B) | CPU attention kernel 통합 (NEO pacpu cherry-pick) | 부모 `PLN_001`. **2026-04-30 전략 변경**: 사용자 "NEO 와 가장 가깝게" → IDE_006 AVX-512/AMX 직접 cherry-pick 대신 **NEO 원 pacpu (ISPC + AVX-512spr-x16) cherry-pick**. `csrc/cpu/pacpu/` 의 NEO 6 file (`CMakeLists.txt` + `build.sh` + `core.h` + `dtype.h` + `pacpu.cpp` + `pacpu.ispc`) 그대로. **모델 macro**: `LLAMA3_3_70B` / `QWEN2_5_1_5B` / `QWEN2_5_7B` / `QWEN2_5_32B` / `QWEN2_5_72B`. g++-12 채택 (g++-11 의 `_Float16` 미지원). `assert_hyper_params_expected` 의 num_layers 완화 (per-layer view 호환). `vllm/v1/attention/ops/neo_pacpu.py` Python wrapper + KV layout adapter (vLLM HND → NEO multi-layer view, zero-copy) + `ensure_loaded(...)` startup auto-build (Phase 3.4.b). **Phase 3.1+3.2** 는 TSK_015.Step3.2.B + Step3.2.C.1~C.8 와 합체해 `unified_attention_with_output` dispatch hook 으로 land. **Phase 3.3 BF16↔FP16 cast** 는 prod-only deferred (dev FP16 모델로 측정 무의미). **TST_018 14 unit PASS**. AMX 는 NEO 미지원 — 별도 phase. 검증 게이트 = `TST_018`. |

---

## Prefix: `TST` — Test

PLN/FEA 의 검증 단위. 정확도·throughput·통합성을 각각의 TST 로 분리해 측정한다. CLAUDE.md Method 의 feature 디렉토리 내 `test.md` 및 test 코드와 매핑된다.

**다음 부여 번호**: `TST_019`

| ID | 상태 | 제목 | 비고 |
|---|---|---|---|
| `TST_001` | 완료 | TSK_001 dev kernel 정확도 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_001` 단독**. 단계 A (KVViewAdapter round-trip) · B(i) (Python ref vs portable) · C (wrapper dispatch). dev 87 (12900KF + RTX 3090, 2026-04-25) + prod 87 (Xeon Platinum 8480+ x2 + H100 x8, 2026-04-26) 통과 — eval/results/20260426_050608_..._prod_smoke. prod SIMD cross-check (B(ii)/B(iii)) 는 `TST_004` 로 이관 |
| `TST_002` | 기각 (2026-05-02 — TSK_002 기각 동행) | Cold-KV CPU Partial Attention throughput / overlap profile | 부모 `PLN_001`. 모든 TSK (001/002/003) 의 perf 통합 검증 — kernel throughput + overlap profile. **2026-05-02 기각**: TSK_002 / TSK_011 / TSK_012 일괄 기각으로 본 TST 의 perf 게이트 영역이 NEO 4차 적재 영역 (TSK_015 §3.5) 으로 이동. NEO 의 throughput / overlap profile 는 prod B-6 회차에서 측정. ID 보존 (재사용 금지) |
| `TST_003` | 대기 | Cold-KV CPU Partial Attention e2e 통합 정확도 검증 | 부모 `PLN_001`. 검증 대상 `TSK_002` 의 vLLM forward 통합 (kernel ISA 무관). **D-i** generated token divergence + **D-ii** logprob/PPL diff. IDE_006 §9 (c) 의 e2e 측 게이트 |
| `TST_004` | 완료 (152 passed @ prod simd_verify 20260427_044407) | TSK_003 prod SIMD cross-check | 부모 `PLN_001`. 검증 대상 **`TSK_003` 단독**. 단계 B(ii) (portable vs AVX-512) · B(iii) (portable vs AMX). IDE_006 §9 (c) 의 prod ISA 측 게이트. prod 결과: `eval/results/20260427_044407_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_simd_verify/tst004_pytest.log` (40 + 40 + skipped) BF16 ~5e-3 / FP16 ~1e-3 tolerance 통과 |
| `TST_005` | 기각 (2026-04-29 — TSK_005 기각 동행) | TSK_005 cross-layer pipeline 동작 검증 | 부모 `PLN_001`. 검증 대상 `TSK_005` 의 기각으로 본 TST 도 검증 대상 부재 → 기각. 본문의 측정 spec (CUDA event timeline / wall time / thread state) 은 reference 용으로 보존. ID 보존 (재사용 금지) |
| `TST_006` | 기각 (2026-05-02 — TSK_006 기각 동행) | TSK_006 Q chunk pipelining 동작 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_006`**. (a) chunk N D2H 가 *완료 전* 에 CPU kernel 이 chunk N 데이터로 시작함을 stream timeline 으로 측정. (b) chunk 수 1 vs N 의 cold path 총 시간 비교 (chunk ↑ → 시간 ↓). (c) chunk 크기 sweep 으로 README §12 Q6 답 산출 |
| `TST_007` | 완료 (2026-04-28 — wall-time microbench 9 cell 통과, perf counter 직접 측정은 미진행이지만 wall-time 결과로 cache miss 영향 명확 입증) | TSK_007 GQA broadcast 옵션 A vs B microbench | 부모 `PLN_001`. 검증 대상 **`TSK_007`**. **결과**: 옵션 A 가 모든 cell 1.36×~3.84× 빠름 — wall-time 으로 cache miss 영향 입증. README §4.3 / §8 iv / §12 Q3 답 산출 |
| `TST_008` | 기각 (2026-05-02 — TSK_008 기각 동행) | TSK_008 hot/cold 분할 정책 측정 | 부모 `PLN_001`. 검증 대상 **`TSK_008`**. layer-별 독립 vs request 균일 두 정책의 e2e throughput / cold path firing 빈도 비교. README §12 Q4 답 산출 |
| `TST_009` | 완료 (2026-04-29 — fix v4 prod 6 회차 검증 적재) | TSK_009 cold-path non-blocking dispatch 동작 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_009`**. 사용자 framing 2 invariant — (1) 속도 (C/B ≤ 1.0), (2) CPU 활용 (merged > 0). prod 6 회차 (input_heavy/output_heavy/equal × B/C, 100 prompts × Llama-70B + TP=8) 결과: invariant 1 부분 위반 (C/B 1.078~1.103×, 잔여 cost 7~10% = cold path setup GPU sync) — 사용자 결정 acceptable. invariant 2 완전 위반 (merged 0% across all scenarios, fundamental Q dependency). 단위 회귀 343 passed / 8 known-issue. **TSK_005 (cross-layer pipeline) 영역으로 invariant 2 이전** |
| `TST_010` | 기각 (2026-05-02 — TSK_010 기각 동행) | TSK_010 CPU 자원 확장 동작 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_010`**. (a) multi-OMP-team 활성 시 *각 core utilization* 측정 (pidstat / `/proc/<pid>/sched`). (b) sub-batch 별 batch latency 비교. (c) NUMA bind (TSK_004) 와 함께 cross-socket 트래픽 0 검증 |
| `TST_011` | 기각 (2026-05-02 — TSK_011 기각 동행) | TSK_011 speculative + fallback 동작 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_011`**. (a) sweep 결과 land — prod sweep `eval/results/20260428_041131_*` (deadline=100ms, fallback 40회) / `..._042424_*` (deadline=1000ms, fallback 0회). (b)/(c) throughput 하한 보장 입증. (d) D-ii 미달성으로 `TSK_012` 영역 이전. **2026-04-29 의미 재정의**: TSK_009 fix v4 의 non-blocking poll 적용으로 §1.1 의 deadline 측정이 *merged/dropped 분포* 로 변경. TSK_012 적용 후 partition path 발화 안 함 → 본 TST 의 §1.2 fallback 빈도 / §1.4 numerical 측정은 *dead* (예외 안전망 외 발화 영역 없음). 한때 검토되던 race 검증 spec 은 race framing 자체 제거 (2026-04-29) 동행. (a)/(b)/(c) 결과는 land 그대로 보존 |
| `TST_012` | 기각 (2026-05-02 — TSK_012 기각 동행) | TSK_012 검증 — D-ii 봉합 + reload timeline + throughput tradeoff (+ 선택 capacity) | 부모 `PLN_001`. 검증 대상 **`TSK_012`**. **단일 단계 spec**: TSK_011 sweep 환경 (max-prompts=30, logprobs=1) 재사용. D-ii 봉합 (worst_max_abs_logprob ≤ 0.5, ppl_relative_diff ≤ 0.1, d_ii_pass ≥ 27/30) + reload event timeline (forward 진입 시 reload 완료 비율 ≥ 90%) + throughput tradeoff (per-request wall-time = vanilla + reload PCIe 비용, 항상 느림). (선택) capacity 회차 — TSK_009 validation 환경 (max-prompts=100, 3 시나리오 × B/C) 재사용. KV pool overflow 영역에서 vanilla mirror 대비 본 TSK 의 진짜 evict 가 동시 in-flight request 수 확장 측정. 단 vLLM mirror 정책의 overflow 동작 확정 후 의미. invariant 1 / invariant 2 / race winner 분포 영역 제거 (Phase 2 동행). **2026-04-29 4 차 재정의 후 영역 변동**: TSK_012 자체가 NEO 식 architecture 로 흡수 / 폐기 결정 영역에 있어 본 TST 도 동행 — TSK_013 분석 결과 후 결정 |
| `TST_013` | 완료 (분석 only) — §2 binding 통과 / §3 (선택) NEO reproduce 영구 deferred (2026-05-02) | TSK_013 검증 — NEO repo 분석 결과의 완결성 + (선택) NEO 자체 reproduce | 부모 `PLN_001`. 검증 대상 **`TSK_013`**. (1) **분석 plan 의 완결성**: NEO 의 핵심 파일 (scheduler / executor / engine / block_manager / model worker / block_swapper / pacpu kernel) 모두가 vLLM hook 위치에 매핑되어 누락 없는지. 신규 TSK 발급 plan 의 5 개 영역 (Request scheduler / KV exclusive / Asymmetric pipelining / Load-aware scheduling / CPU attention kernel port) 이 NEO 의 모든 핵심 메커니즘을 cover 하는지. (2) **(선택) NEO 자체 reproduce** — `evaluation/reproduce-fig6c.py` 또는 `reproduce-fig10a.py` 를 prod (Xeon SPR + H100×8 또는 별도 T4/A10G 환경) 에서 실행하여 NEO 의 throughput gain (T4 7.5× / A10G 26%) 가 IDE_006 환경에서 재현 가능한지 측정 |
| `TST_014` | 대기 (2026-04-29 — NEO 4 차 재정의 후 발급) | TSK_014 검증 — Request scheduler 동작 (3 큐 + mode selection) | 부모 `PLN_001`. 검증 대상 **`TSK_014`**. (1) 단위 — `Scheduler.schedule()` 의 두 SubBatch + swap list 반환 형식 + 6 단계 batch 결정 algorithm 의 invariant (swap-in 과 swap-out 동시 발생 안 함, FCFS prefill 분류). (2) 회귀 — vanilla GPU-only path (CPU running 큐 비어있을 때) 가 vLLM 기존과 동등 동작. (3) heuristic 측정 — pipelined vs sequential rate 결정의 실측 분포. |
| `TST_015` | 대기 (2026-04-29 — NEO 4 차 재정의 후 발급) | TSK_015 검증 — KV cache exclusive ownership + swap | 부모 `PLN_001`. 검증 대상 **`TSK_015`**. (1) 단위 — `_initiate_swap` 의 atomic (source free + dest alloc), exclusive invariant (한 request 의 KV 가 GPU 또는 CPU 한 쪽에만 존재). (2) 회귀 — 기존 mirror 정책 user 의 회귀 없음 (config flag 로 분리). (3) capacity — KV pool overflow 영역에서 vanilla mirror 대비 동시 in-flight request 수 비교. |
| `TST_016` | 대기 (2026-04-29 — NEO 4 차 재정의 후 발급) | TSK_016 검증 — Asymmetric pipelining 시간 매칭 + 정확도 | 부모 `PLN_001`. 검증 대상 **`TSK_016`**. (1) 정확도 — 두 sub-batch 의 attention 결과가 single batch (sequential mode) 와 *분포 동등* (D-i/D-ii tolerance). (2) 시간 매칭 — GPU prefilling time ≈ CPU decoding attention time 의 *실측 매칭률* (≥ 80%). (3) Throughput — sub-batch 동시 실행이 sequential 대비 throughput gain. (4) 회귀 — single sub-batch path 무회귀. |
| `TST_017` | 대기 (2026-04-29 — NEO 4 차 재정의 후 발급) | TSK_017 검증 — PerfPredictor 의 prediction 정확도 | 부모 `PLN_001`. 검증 대상 **`TSK_017`**. (1) 단위 — table-based interpolation 의 1D/2D 정확성. (2) prod 정확도 — predicted layer time vs actual layer time 의 오차 ≤ 10% (prod target Xeon SPR + H100×8 + Llama-70B). (3) startup overhead — profile run 의 시간 측정 (acceptable < 60s). |
| `TST_018` | 대기 (2026-04-29 — NEO 4 차 재정의 후 발급) | TSK_018 검증 — CPU attention kernel 통합 + 회귀 | 부모 `PLN_001`. 검증 대상 **`TSK_018`**. (1) 회귀 — IDE_006 TSK_001/003/004/007/010 의 단위 테스트 통과 (LSE-반환 제거 외 회귀 없음). (2) 정확도 — CPU sub-batch attention 결과 vs GPU full attention reference (D-ii tolerance). (3) Throughput — IDE_006 의 AVX-512/AMX kernel vs NEO 의 ISPC AVX2 kernel 의 throughput 비교 (Xeon SPR + Llama-70B + TP=8). |

---

## Prefix: `FEA` — Feature

PLN/TST 통과 후 본 코드 베이스에 들어가는 단위 기능. CLAUDE.md Method 의 feature 디렉토리 구조를 따른다.

**다음 부여 번호**: `FEA_001`

(등록된 ID 없음)

---

## 향후 추가될 prefix

새 prefix 를 도입할 때는 (a) 본 파일에 섹션 신설 + 카운터 초기화, (b) `shadow_assists/README.md` Part VII Legend 에 한 줄 추가, (c) CLAUDE.md ID Rule 변경이 필요하면 그쪽도 갱신.
