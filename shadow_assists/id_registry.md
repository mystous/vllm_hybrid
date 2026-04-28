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

**다음 부여 번호**: `TSK_013`

| ID | 상태 | 제목 | 비고 |
|---|---|---|---|
| `TSK_001` | 완료 | LSE-반환 CPU partial-attention kernel — dev 부분 | 부모 `PLN_001`. 4.0 KVViewAdapter / 4.1 Python reference / 4.2c portable C++ / 4.3 wrapper. 검증 게이트 = `TST_001` (A·B(i)·C — dev 87 + prod 87 pytest 통과, 2026-04-26). prod SIMD (AVX-512/AMX) 는 `TSK_003` 으로 이관 |
| `TSK_002` | 활성 (Phase 1 dev §4.2~§4.5c + §4.6 stream 분리·race fix·async revert 완료, Phase 2 prod overlap 측정 진행 중) | scheduler / attention metadata 의 hot/cold partition 통합 | 부모 `PLN_001`. `TSK_001` 후속. 검증 게이트 = `TST_003` (e2e 통합 정확도). 회귀 fix 흐름 (libgomp EAGAIN / wrapper 5462 ms timeout / device-mismatch / per-seq 필터링 / sequential→stream) 은 PLN-deliverable `PLN_001_TSK_002_02_overlap_fix_log.md` |
| `TSK_003` | 활성 (Phase 1 dev 구현·컴파일·dispatch 완료, Phase 2 prod 정확도 측정 대기) | prod SIMD kernels — AVX-512 + AMX | 부모 `PLN_001`. 선행 `TSK_001` (portable + algorithm reference). §4.2a AVX-512: `csrc/cpu/partial_attention_avx512.cpp` — head_dim dot product 를 AVX-512F (+ optional AVX512_BF16 vdpbf16ps) intrinsics 로 교체, outer 3-pass 구조는 portable 동일. §4.2b AMX: `csrc/cpu/partial_attention_amx.cpp` — BF16 입력에서 `_tile_dpbf16ps` 로 16 K rows × head_dim chunks (32 BF16 단위) 를 batched matmul. FP16/FP32 는 AMX 무관이라 같은 TU 내 AVX-512 fallback. 두 kernel 모두 `_kernel_source_path` JIT load + cpuid 게이트 (`_has_avx512()` / `_has_amx()`) 로 fuse-off CPU 의 static-init SIGILL 회피. 검증 게이트 = `TST_004` (portable vs AVX-512 / portable vs AMX cross-check, BF16 ~5e-3 / FP16 ~1e-3 tolerance). dev 검증: 양 kernel `g++ -mavx512f -mavx512bf16 -mamx-tile -mamx-bf16` 컴파일 clean, pytest 93 통과 + 80 cross-check 자동 skip (AVX-512 fuse-off + AMX hardware 미지원). prod 측 정확도 cross-check 는 사용자 (Xeon SPR+ + H100×8) 가 `bash eval/run_prod_smoke.sh` 또는 `pytest tests/v1/cpu_partial_attention/test_avx512_cross_check.py test_amx_cross_check.py` 로 직접 |
| `TSK_004` | 활성 (Phase 1 dev 구현 완료, Phase 2 prod 측정 대기) | Cold-KV 경로 NUMA-aware 화 — connector buffer + partial-attention 커널 | 부모 `PLN_001`. (a) `vllm/distributed/kv_transfer/kv_connector/v1/offloading/numa_aware.py` 의 `bind_worker_to_local_numa()` 가 `CpuGpuOffloadingHandlers.__init__` 직전에 호출되어 worker 의 local NUMA node 를 `numa_set_preferred` 로 지정 → 이후 `torch.zeros(..., pin_memory=True)` 가 자기 socket 에 박힘. (b) `pin_threads_to_local_numa()` 가 `forward_partial_with_lse` 첫 진입 시 `os.sched_setaffinity` 로 worker 프로세스의 코어 affinity 를 자기 NUMA node 코어로 제한 → OpenMP / std::thread 가 cross-socket 으로 못 빠짐. 두 hook 모두 idempotent + libnuma 미설치 / 단일 socket 시 silent no-op. local node 결정 우선순위: GPU NUMA node → rank % num_nodes → skip. 검증은 `TST_002` (throughput / overlap profile) 에 흡수. dev 검증: `TST_001` 회귀 없음 (93/93), e2e smoke 회귀 없음. prod 측정은 `TST_002` sweep 단계 (사용자 직접) |
| `TSK_005` | 대기 | Cross-layer pipeline parallelism (`model_runner.forward` layer-async refactor) | 부모 `PLN_001`. 출처 README §8 risk vii / §9 (g) / §12 Q6 + TSK_002 §10 Change Log 2026-04-27 (deferred). layer N 의 cold CPU work 와 layer N+1 의 hot FA work 의 cross-layer overlap 으로 매 layer 직렬 누적 비용을 겹쳐 줄임. 검증 게이트 = `TST_005`. NEO 논문 핵심 의도. 사용자 100 req heavy workload throughput 격차 해소의 결정적 영역 |
| `TSK_006` | 대기 | Q chunk pipelining (Q D2H 를 chunk 단위 split + CPU kernel 과 stream-level pipeline) | 부모 `PLN_001`. 출처 README §12 Open Q6, TSK_002 §4 표 §4.6 본문 ("Q chunk pipelining 은 1차 stream 분리 효과 측정 후 결정"). cold path 의 첫 chunk 도착 즉시 CPU kernel 시작, 동시에 다음 chunk D2H 진행. Q transfer + CPU compute 직렬 누적 줄임. 검증 게이트 = `TST_006`. 의존: `TSK_005` (cross-layer 위에서 chunk pipeline 이 의미) |
| `TSK_007` | 완료 (2026-04-28 — microbench 9 cell sweep 결과 옵션 A 채택, 코드 변경 없음) | GQA K/V broadcast 옵션 결정 (옵션 A compact+broadcast vs B pre-expand) | 부모 `PLN_001`. 출처 README §4.3, §8 risk iv, §12 Open Q3. **결과** (`eval/run_tsk007_gqa_microbench.py`, Llama-3.3-70B + TP=8 worker per 9 cell): 모든 cell 에서 옵션 A 가 1.36×~3.84× 빠름. 원인 — kernel 안 broadcast 비용 (h/q_per_kv division 한 번) << expanded buffer cache miss 비용 (8× 메모리가 L2/L3 못 fit). 옵션 A (현재 코드) 유지. 검증 게이트 = `TST_007` 완료 |
| `TSK_008` | 대기 | hot/cold 분할 정책 (layer-별 독립 vs request 전체 균일) | 부모 `PLN_001`. 출처 README §12 Open Q4. layer 별로 다른 cold ratio 가 throughput 에 의미가 있는지 결정. 검증 게이트 = `TST_008`. 측정 후 결정 |
| `TSK_009` | 재정의 (2026-04-28 — inspection 후 TSK_012 의 일부로 통합 결정, 코드 변경 없음) | Reload prefetch — admission 시점 reload 시작으로 forward 와 overlap | 부모 `PLN_001`. 출처: 본 commit `c35585556c` 의 §4.5c reload sync 가 forward 완료까지 wait → 100 req heavy 의 reload 시간이 forward 에 직접 반영되는 비용 (TSK_002 §4 표 §4.5c row + Change Log 2026-04-28 명시). **inspection 결과** (2026-04-28): 현재 vLLM 의 `transfer_async` 가 *이미 별도 CUDA stream + wait_event 로 prefetch overlap* (자연 발생). 추가 영역 (모든 cold-bearing reload / step-advance prefetch / layer-level prefetch) 은 vLLM core block manager 통합 영역으로 `TSK_012` 와 같은 hook. 단독 진행 가치 작아 **TSK_012 의 일부로 통합** 결정. TSK_009 = inspection-based 완료, 코드 변경 없음 |
| `TSK_010` | 대기 | CPU 자원 활용 확장 — multi-OMP-team / sub-batching | 부모 `PLN_001`. 출처: 사용자 허락 후 발행 (2026-04-28). 현재 cold path 가 단일 worker × 단일 OMP team — prod 의 100+ core 자원 (TSK_004 NUMA bind 적용 후) 이 batch sub-batching + multi-OMP-team 으로 더 많이 동원 가능. TSK_004 의 자연 확장. 검증 게이트 = `TST_010` |
| `TSK_011` | 활성 (코드 land + dev pytest 419 passed + prod 70B fallback 발동 검증 + sweep 결과 land — D-ii 봉합 한계 입증) | Speculative cold + GPU fallback (3차 재정의 핵심 코드화) | 부모 `PLN_001`. 출처: IDE_006 §3.1 3차 재정의 (2026-04-28) 의 코드화. cold partial 결과를 deadline 안에 await 하고, 미도달 시 GPU full FA on (hot+cold) 로 fallback (opt-F: layer-local 임시 H2D + native SDPA, vLLM core lifecycle 무변경). 검증 게이트 = `TST_011`. **prod sweep 결과 (2026-04-28)**: deadline=100ms (fallback 100% 발동) 와 deadline=1000ms (fallback 0%) 모두 baseline 과 lp ~3.43 / ppl ~0.24 의 *동일* 발산 — fallback path (SDPA H2D) 와 partition path (CPU partial-attn + LSE merge) 가 *같은 cold KV source (CPU page cache)* 를 사용하므로 같은 root cause 발산. 즉 TSK_011 은 *throughput 하한 보장* 과 *worst-case stall 차단* 은 입증했으나 D-ii 봉합은 못함. 진짜 D-ii 봉합은 `TSK_012` (decode-time cold→GPU reload) 의 영역으로 분리 |
| `TSK_012` | 대기 | Decode-time cold-blocks GPU reload (D-ii equivalence) | 부모 `PLN_001`. 선행: `TSK_011` (fallback 메커니즘이 reload 정책의 안전망 / 본 TSK 의 발견 출처) 및 `TSK_009` (admission 시점 prefetch — 동일 hook 영역). 출처: TSK_011 §4.5 sweep 결과 (2026-04-28) — fallback path 가 *cold KV source 자체* 가 baseline 과 다르기 때문에 D-ii 봉합 못함을 입증. **decode 시점에 cold blocks 를 GPU paged cache 로 reload 하여 baseline 의 source 와 일치시키는** 것이 진짜 D-ii 봉합 메커니즘. vLLM `OffloadingConnectorScheduler` lifecycle 의 decode 단계 hook 추가. 검증 게이트 = `TST_012` |

---

## Prefix: `TST` — Test

PLN/FEA 의 검증 단위. 정확도·throughput·통합성을 각각의 TST 로 분리해 측정한다. CLAUDE.md Method 의 feature 디렉토리 내 `test.md` 및 test 코드와 매핑된다.

**다음 부여 번호**: `TST_013`

| ID | 상태 | 제목 | 비고 |
|---|---|---|---|
| `TST_001` | 완료 | TSK_001 dev kernel 정확도 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_001` 단독**. 단계 A (KVViewAdapter round-trip) · B(i) (Python ref vs portable) · C (wrapper dispatch). dev 87 (12900KF + RTX 3090, 2026-04-25) + prod 87 (Xeon Platinum 8480+ x2 + H100 x8, 2026-04-26) 통과 — eval/results/20260426_050608_..._prod_smoke. prod SIMD cross-check (B(ii)/B(iii)) 는 `TST_004` 로 이관 |
| `TST_002` | 대기 | Cold-KV CPU Partial Attention throughput / overlap profile | 부모 `PLN_001`. 모든 TSK (001/002/003) 의 perf 통합 검증 — kernel throughput + overlap profile. IDE_006 §9 (b)(g) 충족 게이트 |
| `TST_003` | 대기 | Cold-KV CPU Partial Attention e2e 통합 정확도 검증 | 부모 `PLN_001`. 검증 대상 `TSK_002` 의 vLLM forward 통합 (kernel ISA 무관). **D-i** generated token divergence + **D-ii** logprob/PPL diff. IDE_006 §9 (c) 의 e2e 측 게이트 |
| `TST_004` | 완료 (152 passed @ prod simd_verify 20260427_044407) | TSK_003 prod SIMD cross-check | 부모 `PLN_001`. 검증 대상 **`TSK_003` 단독**. 단계 B(ii) (portable vs AVX-512) · B(iii) (portable vs AMX). IDE_006 §9 (c) 의 prod ISA 측 게이트. prod 결과: `eval/results/20260427_044407_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_simd_verify/tst004_pytest.log` (40 + 40 + skipped) BF16 ~5e-3 / FP16 ~1e-3 tolerance 통과 |
| `TST_005` | 대기 | TSK_005 cross-layer pipeline 동작 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_005`**. (a) layer N+1 forward 호출 시점에 layer N cold work 가 *in-flight 상태* 임을 CUDA event timeline 으로 측정 (단순 리턴값 아님). (b) layer 간 stream timeline 의 *시간적 겹침* 이 baseline (TSK_005 비활성) 대비 입증. (c) e2e throughput 회복 |
| `TST_006` | 대기 | TSK_006 Q chunk pipelining 동작 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_006`**. (a) chunk N D2H 가 *완료 전* 에 CPU kernel 이 chunk N 데이터로 시작함을 stream timeline 으로 측정. (b) chunk 수 1 vs N 의 cold path 총 시간 비교 (chunk ↑ → 시간 ↓). (c) chunk 크기 sweep 으로 README §12 Q6 답 산출 |
| `TST_007` | 완료 (2026-04-28 — wall-time microbench 9 cell 통과, perf counter 직접 측정은 미진행이지만 wall-time 결과로 cache miss 영향 명확 입증) | TSK_007 GQA broadcast 옵션 A vs B microbench | 부모 `PLN_001`. 검증 대상 **`TSK_007`**. **결과**: 옵션 A 가 모든 cell 1.36×~3.84× 빠름 — wall-time 으로 cache miss 영향 입증. README §4.3 / §8 iv / §12 Q3 답 산출 |
| `TST_008` | 대기 | TSK_008 hot/cold 분할 정책 측정 | 부모 `PLN_001`. 검증 대상 **`TSK_008`**. layer-별 독립 vs request 균일 두 정책의 e2e throughput / cold path firing 빈도 비교. README §12 Q4 답 산출 |
| `TST_009` | 재정의 (2026-04-28 — TSK_009 의 TSK_012 통합 결정에 따라 본 TST 측정도 TSK_012 진행 시점에 함께 수행, 별도 회차 없음) | TSK_009 reload prefetch 동작 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_009`** — TSK_012 (decode reload) 와 통합 영역 |
| `TST_010` | 대기 | TSK_010 CPU 자원 확장 동작 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_010`**. (a) multi-OMP-team 활성 시 *각 core utilization* 측정 (pidstat / `/proc/<pid>/sched`). (b) sub-batch 별 batch latency 비교. (c) NUMA bind (TSK_004) 와 함께 cross-socket 트래픽 0 검증 |
| `TST_011` | 부분 완료 (a/b/c 입증, d 미달성 — TSK_012 영역으로 분리) | TSK_011 speculative + fallback 동작 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_011`**. (a) cold deadline 도달 / 미도달 시점 측정 — prod sweep `eval/results/20260428_041131_*` (deadline=100ms, fallback 40회) / `..._042424_*` (deadline=1000ms, fallback 0회). (b) fallback 발동 빈도 — sweep 양 끝점 입증. (c) **throughput 하한 보장** — deadline=100ms 187s vs deadline=1000ms 216s, fallback 이 partition 보다 *오히려 빠름* (PCIe < CPU partial). (d) numerical equivalence — sweep 결과 fallback path / partition path 모두 baseline 과 lp ~3.43 의 동일 발산 → fallback 만으로 D-ii 봉합 *불가* 입증. 단위 reference SDPA equivalence 는 `tests/v1/cpu_partial_attention/test_tsk011_fallback.py` 가 통과 (BF16 ~5e-3). e2e D-ii 봉합은 `TSK_012` (decode reload) 의 영역 |
| `TST_012` | 대기 | TSK_012 decode-time GPU reload 의 D-ii 봉합 정량 | 부모 `PLN_001`. 검증 대상 **`TSK_012`**. (a) decode 단계에서 cold blocks 가 GPU paged cache 로 reload 되는 시점 측정 (event timeline). (b) reload 후 baseline 과 동일 source (GPU paged) 를 사용하는 attention 결과의 D-ii — `worst_max_abs_logprob` 가 TSK_011 sweep 의 ~3.43 에서 baseline tolerance (atol=0.5) 안으로 들어오는지. (c) reload 비용 (PCIe) 과 throughput tradeoff. TST_003 wrapper reuse |

---

## Prefix: `FEA` — Feature

PLN/TST 통과 후 본 코드 베이스에 들어가는 단위 기능. CLAUDE.md Method 의 feature 디렉토리 구조를 따른다.

**다음 부여 번호**: `FEA_001`

(등록된 ID 없음)

---

## 향후 추가될 prefix

새 prefix 를 도입할 때는 (a) 본 파일에 섹션 신설 + 카운터 초기화, (b) `shadow_assists/README.md` Part VII Legend 에 한 줄 추가, (c) CLAUDE.md ID Rule 변경이 필요하면 그쪽도 갱신.
