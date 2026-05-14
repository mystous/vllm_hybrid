# v1.5 성능 분석 / TODO 문서

> **문서 목적**: context 잃어도 v1.5 영역의 full state 를 다시 찾을 수 있도록
> 모든 관련 파일 path + line + 분석 결과 fact + 도구 영역 + TODO 까지 통합 기록.
>
> **분석 시점**: 2026-05-12 KST (09:25 ~ 11:30)
> **대상 영역**: vllm_hybrid IDE_006 / TSK_019 v1.5 lineage
> **현재 working tree**: v1.5.2 fix 영역 적용 + **commit 미수행**
> **branch**: `feat/ide006-tsk019-neo-performance-max`

---

## 0. 빠른 navigation

| 영역 | 섹션 |
|---|---|
| v1.5 lineage commit 영역 | §1 |
| 코드 변경 영역 (fix 위치) | §2 |
| 22 항목 정의 + 발화 fact | §3 |
| 측정 결과 (30min sustain / cdec trace / E 영역) | §4 |
| 비교 (vanilla / v1.4 / v1.5) | §5 |
| 분석 도구 결과 fact 종합 | §6 |
| 후속 plan 후보 | §7 |
| TODO 항목 표 | §8 |
| 관련 파일 / 경로 색인 | §9 |

---

## 1. v1.5 lineage commit 영역

| commit | 일시 (KST) | 영역 | 비고 |
|---|---|---|---|
| `a77ad7e96` | 2026-05-11 02:13 | v1.4 정통 (Option I+K+C+L+M2) | chain firing 98.7%, NEO 발화 reference (try84) |
| `858b6df7a` | 2026-05-11 09:40 | **v1.5** — `.cpu()` sync 제거 commit | commit msg "246→904 회복" = **fact 오류** (cdec silent skip 인공값) |
| `e78d82d7e` | 2026-05-11 | **v1.5.1** — silent SEGV root NameError `_os_th` fix | 후속 NameError=0 |
| (uncommitted) | 2026-05-12 (현 working tree) | **v1.5.2** — cdec dispatch silent skip 영역 fallback chain | commit 권한 대기 |

### 1.1 v1.5 commit msg 의 fact 오류

v1.5 commit (`858b6df7a`) 의 message claim:
> "throughput 246 → 904 tps (3.7×) 회복"

**실제 fact**:
- NEO CDEC CALL log fire = 0 (cdec dispatch path 전체 silent skip)
- py-spy raw stack: `_neo_cdec_compute_cpu` / `ispc_attention_tasks` fire count = 0
- 904 tps 는 NEO 의 CPU offload 가 *실제 발화 안* 한 인공값

**증거 영역**:
- D-cdec-trace v4 (cdec_eager): `total=129800 entered=63859 pre_submit=0 post_submit=0`
- 위치: `vllm/model_executor/layers/attention/attention.py:780~820` (env-gated counter `VLLM_DEBUG_CDEC_PATH=1`)

---

## 2. 코드 변경 영역

### 2.1 v1.5.2 working tree 변경 (현재 uncommitted)

| 파일 | line | 변경 | 영역 |
|---|---|---|---|
| `vllm/model_executor/layers/attention/attention.py` | 873~ (Option L) | `_seq_lens_attr_optL` fallback chain | seq_lens_cpu → _seq_lens_cpu → seq_lens.cpu() |
| `vllm/model_executor/layers/attention/attention.py` | 936~ (main cdec) | `seq_lens_attr` fallback chain + forward_context cache | 80 layer × N sub-batch sync → 1회/forward |
| `vllm/model_executor/layers/attention/attention.py` | 780~820 | D-cdec-trace counter (env-gated `VLLM_DEBUG_CDEC_PATH=1`) | 분석 영역, production 영향 0 |
| `vllm/v1/worker/gpu_worker.py` | 754~ | torch.profiler instrument (env-gated `VLLM_DEBUG_TORCH_PROFILER=1`) | E18 chrome trace 영역 |
| `vllm/v1/executor/multiproc_executor.py` | 826 | faulthandler enable (env-gated `VLLM_DEBUG_FAULTHANDLER=1`) | silent SEGV 영역 trap (v1.5.1 영역) |
| `eval/neo_22_items_monitor.sh` | 41 | OPT_C grep v1/v2 패턴 둘 다 catch | monitor 영역 patch |
| `eval/run_probe_step2_30min.sh` | cleanup 영역 | pgrep → 직접 PID kill (ps -o pid,comm) | zombie cleanup 영역 |

### 2.2 v1.5.1 commit (`e78d82d7e`) 변경

| 파일 | line | 변경 | 영역 |
|---|---|---|---|
| `vllm/v1/core/sched/neo_scheduler_adapter.py` | top + 469 | `import os` top-level + `_os_th` → `os` 영역 통일 | silent SEGV `_os_th not defined` 영역 fix |

### 2.3 silent skip root 영역 분석

**핵심 root**: `FlashAttentionMetadata` (`vllm/v1/attention/backends/flash_attn.py:219`) 는 `seq_lens_cpu` / `_seq_lens_cpu` attribute 를 보유 안 함.

| 메타데이터 class | seq_lens_cpu | _seq_lens_cpu |
|---|---|---|
| `CommonAttentionMetadata` (`vllm/v1/attention/backend.py:431`) | property (deprecated) | field |
| `FlashAttentionMetadata` (`vllm/v1/attention/backends/flash_attn.py:219`) | **부재** | **부재** |
| `TritonAttentionMetadata` (다른 backend) | **미검증** | **미검증** |
| `FlexAttentionMetadata` (다른 backend) | **미검증** | **미검증** |

**v1.5 의 silent skip 발생 mechanism**:
1. `attention.py:936` 의 `getattr(attn_metadata, "seq_lens_cpu", None)` 호출
2. `attn_metadata` 가 `FlashAttentionMetadata` 인 경우 → None 반환
3. fallback `getattr(attn_metadata, "_seq_lens_cpu", None)` 도 None
4. `if seq_lens_attr is not None:` block 의 cdec dispatch 전체 skip

**v1.5.2 fix**:
- 3차 fallback `_fc._neo_cdec_seq_lens_cpu_cache` (forward_context cache)
- 4차 fallback `attn_metadata.seq_lens.cpu()` (GPU→CPU sync, cache 후 1회/forward)

### 2.4 forward_context / metadata 관련 영역

- `vllm/forward_context.py:160-167` — `neo_cdec_token_slice` / `neo_cdec_seq_slice` / `neo_cdec_req_ids` 정의
- `vllm/v1/worker/gpu_model_runner.py:2467-2492` — sub_batch metadata split 영역에 cdec slice 적용
- `vllm/v1/worker/gpu_model_runner.py:3745-3761` — per_subbatch_contexts 에 cdec slice attach
- `vllm/v1/worker/gpu_model_runner.py:4360-4373` — `_neo_cdec_slices_for_step` build 영역

---

## 3. 22 항목 정의 + v1.5 발화 fact

### 3.1 22 항목 정의 영역

문서: `shadow_assists/features/IDE_006/Objective-for-NEO-porting.md`
Monitor: `eval/neo_22_items_monitor.sh`

### 3.2 v1.5 (현 30min sustain) 22 항목 fire 결과

측정: `eval/results/20260512_092529_step2_30min_nameerror_fix/` (2026-05-12 09:25→09:55 KST)

| # | 항목 | fact | 상태 |
|---|---|---|---|
| 1 | KV exclusive ownership | SWAP_OUT_CALL=984 | ✓ |
| 2 | CPU attention 직접 | active=3076/3100 (99.2%) | ✓ |
| 3 | Asymmetric Pipelining | OOM=0 | ✓ |
| 4 | Stage 분할 | OOM=0 | ✓ |
| 5 | 6단계 Scheduler | D15+D16 fire=1 | ✓ |
| 6 | Mode Select | 99.2% | ✓ |
| 7 | 3-way attention dispatch | eligible=3076 active=3076 | ✓ |
| 8 | swap_out/in invariant | shape_mismatch=0 | ✓ |
| 9 | pacpu kernel | CDEC_CALL max=246,700 | ✓ |
| 10 | Q/K/V D2H transfer | pacpu fire 동반 | ✓ |
| 11 | sub_batches attach | eligible=3076 | ✓ |
| 12 | b0/b1 정렬 | reject_split_oob=0 | ✓ |
| 13 | forward_pipeline overlap | 99.2% | ✓ |
| 14 | KV migration LRU | swap_out=984 mismatch=0 | ✓ |
| **15** | **NEO > vanilla** | output_tps=247 vs vanilla 4689 (-94.7%) | **✗** |
| 16 | CPU util HIGH | (미측정) | ⏳ |
| 17 | token correctness | (미측정) | ⏳ |
| 18 | deadlock 회피 | engine_dead=0 | ✓ |
| 19 | silent crash 0 | assert=0 cuda=0 segv=0 | ✓ |
| 20 | Option I (resident queue) | fire=1, mirror_set_size=10 | ✓ |
| 21 | Option L (BUF EXTEND) | 160 fire, FAIL=0 | ✓ |
| 22 | Option M (swap-in sync) | 9,816 attach, mismatch=0 | ✓ |

**verdict**: 20/22 fire ✓, 항목 15 fail (NEO 본질 영역), 항목 16/17 미측정.

---

## 4. 측정 결과 (run 영역)

### 4.1 v1.5 30min sustain (steady state baseline)

- 결과 dir: `eval/results/20260512_092529_step2_30min_nameerror_fix/`
- launch script: `eval/run_probe_step2_30min.sh`
- 시간: 2026-05-12 09:25:29 → 09:55:34 KST

| 지표 | 값 |
|---|---|
| Avg throughput (last 50) | **247.4 tps** |
| FORK active/total | 3076/3100 (99.2%) |
| CDEC_CALL max | 246,700 |
| SWAP_OUT_CALL | 984 |
| SWAP_IN attach | 9,816 |
| BUF_EXTEND / FAIL | 160 / 0 |
| Option I / C (v2) / L / M2 | 모두 활성 |
| mirror_set_size | 10 |
| Crash (assert/cuda/segv/dead) | 0 / 0 / 0 / 0 |
| NameError | 0 |
| Processed prompts | 0/500 (incomplete — 30min 영역) |

### 4.2 cdec trace 시계열 (silent skip 영역 식별)

| run | 시각 (KST) | dir | pre_submit | valid_branch | 비고 |
|---|---|---|---|---|---|
| v4 (eager) | 05-12 어제 | (cdec_eager) | 0 | — | cudagraph 가설 disprove |
| v5 (cudagraph) | 05-12 09:05 | `eval/results/20260512_090511_cdec_trace_short/` | 0 | 0 | seq_lens_attr_none=25959 |
| v6 (fix 적용) | 05-12 09:12 | `eval/results/20260512_091250_cdec_trace_short/` | 8,259 | 8,259 | cdec 발화 회복 |
| v7 (fix + cache) | 05-12 09:19 | `eval/results/20260512_091919_cdec_trace_short/` | 9,259 | 9,259 | cache 적용 |

D-cdec-trace counter 위치: `vllm/model_executor/layers/attention/attention.py:780-820` (env-gated `VLLM_DEBUG_CDEC_PATH=1`)

### 4.3 v1.4 정통 reference (try84)

- 결과 dir: `eval/results/20260511_021304_try84_v4_K_OptIKCLM2/`
- 시간: 2026-05-11 02:13 KST (5min short)

| 지표 | 값 |
|---|---|
| Avg throughput (last 50) | 253.2 tps |
| FORK active/total | 1876/1900 (98.7%) |
| CDEC_CALL max | 150,300 |
| SWAP_OUT_CALL | 600 |
| SWAP_IN attach | 5,976 |
| BUF_EXTEND / FAIL | 72 / 0 |
| Option I/C/L/M2 | 모두 활성 |
| Option C path | v1 (decide_mode) |
| mirror_set_size | 10 |
| Crash | 0 |

**verdict**: v1.4 vs v1.5 → throughput -2.4% (oscillation 영역). **regression 없음**.

### 4.4 vanilla baseline reference

- 결과 dir: `eval/results/20260509_014854_try52_v3_C8_phaseD_vanilla_baseline/`
- 시간: 2026-05-09 01:48 KST

| 지표 | 값 |
|---|---|
| output_tps (steady) | **4,689.8** |
| wall_s | 873s (500p 완료) |
| NEO 적용 | × (baseline) |

---

## 5. 비교 (vanilla / v1.4 / v1.5)

측정 조건 공통: Llama-3.3-70B / TP=8 / H100×8 / max_model_len=16384 / max_num_seqs=256 / target_input_len=8192 / max_tokens=8192 / kv_cache_dtype=fp8 / async_scheduling

| 영역 | vanilla (try52) | v1.4 try84 | v1.5 (현 fix) |
|---|---|---|---|
| NEO 적용 | × | ✓ Option I+K+C+L+M2 | ✓ + C_FULL_MIRROR |
| run 형태 | 500p complete | 5min short | 30min sustain (incomplete) |
| commit | 무관 | a77ad7e96 | working tree |
| output_tps | **4,689.8** | 253.2 (last50) | 247.4 (last50) |
| chain firing | N/A | 98.7% | 99.2% |
| CDEC_CALL max | 0 | 150,300 | 246,700 |
| Crash | 0 | 0 | 0 |

**핵심 fact**:
- vanilla → v1.4 (NEO 발화): **-94.6%** (4689 → 253). NEO 의 CPU offload cost 본질
- v1.4 → v1.5: **-2.4%** (oscillation). regression 없음
- NEO 본질의 throughput cost 자체가 큼 — 항목 15 fail 의 직접 원인

---

## 6. 분석 도구 결과 fact 종합

### 6.1 E 영역 도구 결과

| # | 도구 | 상태 | 결과 dir | 결과 fact |
|---|---|---|---|---|
| 16 | compute-sanitizer (memcheck) | ✓ | `eval/results/20260512_103955_E16_sanitizer_memcheck/` | race/sync/OOB **0 발견**. cudaErrorNoKernelImageForDevice 92회 (probe fallback) + cuMemCreate permission 8회 (sanitizer 부수효과). v1.5 cdec path memory safety 영역 안전 |
| 17 | cuda-gdb attach | ✓ | `eval/results/20260512_104616_E17_cuda_gdb/cuda_gdb_TP0.log` | Thread 105-113 (OMP) 모두 libgomp.so 동일 wait 영역 — pacpu barrier 대기 영역 캡처. GPU active kernel 영역 attach 시점 idle |
| 18 | torch.profiler | ✓ | `eval/results/20260512_110346_E18_torch_profiler/traces/` (8 worker × 350MB) | Top op: `vllm::unified_attention_with_output` 13.7s / `cross_device_reduce_1stage` NCCL 10.7s / `cudaEventSynchronize` 7.5s |
| 19 | ncu | △ | `eval/results/20260512_111117_E19_ncu/ncu_report.ncu-rep` | launch-skip 5000 영역이 model init 단계만 캡처. chain firing 영역 미캡처 — re-sample 별도 plan |
| 20 | gprof (rebuild) | △ | (rebuild 권한 대기) | 대체로 VLLM_NEO_PROFILE + py-spy --native 진행 |
| 20-alt | NEO_PROFILE + py-spy --native | ✓ | `eval/results/20260512_112347_E20_alt_neo_profile_pyspy_native/` | cdec_wait avg 5.6-10.7ms / GPU avg 0.14-0.32ms / **ratio 88.82x**. SWAP_OUT 517 call avg 78.35ms / SWAP_IN 448 call avg 57.24ms |

### 6.2 throughput 247 tps regression 의 직접 root

E18 (torch.profiler) + E20-alt (NEO_PROFILE) 의 결합 fact:

| 영역 | 시간 / step | 비율 |
|---|---|---|
| **cdec_wait** (pacpu compute + future.result wait) × 80 layer | **400-800 ms** | dominant |
| GPU forward (attention) × 80 layer | 11-26 ms | small |
| **NCCL allreduce** | 10.7s / 20 step = **535 ms/step** | dominant |
| cudaEventSynchronize | 7.5s / 20 step = 375 ms/step | large |
| SWAP_OUT avg 78ms × 517 call / run | 분포 영역 | per-call cost 큼 |

→ NEO chain firing 영역의 진짜 bottleneck = **cdec pacpu compute time × 80 layer + NCCL TP=8 collective sync**.

### 6.3 NEO 의 throughput > vanilla 미달성 root (항목 15)

vanilla 4689 → NEO 247 의 -94.7% regression 의 직접 contributor:
- cdec compute 가 GPU forward 의 **88.82배** 시간
- pacpu kernel 의 store_kv / qk_product / softmax / av_product 의 OMP parallel 영역이 GPU 대비 비효율
- 80 layer × 5-10ms cdec_wait = 400-800ms/step → 4 token/s/layer 의 cost

### 6.4 memory safety / race 영역 (E16)

E16 compute-sanitizer memcheck: **race/sync/OOB 0**. v1.5 의 cdec dispatch path 영역 memory safety 안전.

### 6.5 py-spy native frame top (E20-alt)

`eval/results/20260512_112347_E20_alt_neo_profile_pyspy_native/pyspy_native_raw.txt`:

| frame | count | 영역 |
|---|---|---|
| pthread_mutex_unlock (libc.so.6) | 27 | GIL / Python lock |
| pthread_mutex_lock (libc.so.6) | 18 | 동일 |
| torch::FunctionSignature::parse | 11 | Python torch op dispatch |
| c10::impl::OperatorEntry::lookup | 8 | torch op lookup |
| c10::cuda::CUDACachingAllocator::Native::...::BlockComparatorSize | 7 | CUDA allocator |
| at::native::as_strided_tensorimpl | 6 | tensor view |

→ main thread 영역에 pacpu kernel 의 specific 함수 (store_kv 등) 가 leaf 로 나오지 않음. **pacpu 가 별도 thread (executor.submit) 에서 fire — main thread sample 은 future.result wait (pthread_mutex)**.

---

## 7. 후속 분석 plan 후보

| plan | 도구 | 목표 | 시간 cost |
|---|---|---|---|
| ncu re-sample (launch-skip 영역 조정) | ncu | chain firing 영역 attention/swap/NCCL kernel detail | 10min run |
| gprof rebuild (RelWithDebInfo) | gprof | pacpu store_kv / qk_product line-level CPU profile | 50min build + 5min run + 50min Release rebuild = 105min |
| 항목 16 측정 | py-spy 14-thread OMP duty cycle | CPU util HIGH 항목 검증 | 5min run |
| 항목 17 측정 | TST_003 분포 동등성 (per-token logprob max abs diff, sequence PPL) | token correctness 항목 검증 | TST_003 영역 |
| cudagraph graph break 검증 | torch.compile trace + dump_graph | `.cpu()` 호출의 graph break 영향 | 5min run |
| backend metadata 일관성 fix | FlashAttention / Triton / Flex 의 `_seq_lens_cpu` 통일 | A4 영역 해소 | 코드 patch + 회귀 test |

---

## 8. TODO 항목 표

### 8.A. 코드 결함

| # | 문제 | 상태 | 영역 |
|---|---|---|---|
| 4 | backend별 metadata seq_lens_cpu 일관성 부재 | **✗ 미해결** | Flash 만 fallback 동작. Triton/Flex 등 미검증. `vllm/v1/attention/backends/*.py` 영역 patch 필요 |
| 5 | v1.5 commit msg fact 오류 ("246→904 회복") | **✗ 미해결** | 후속 commit msg 정정 필요. commit 권한 대기 |

### 8.B. v1.5.2 fix residual

| # | 문제 | 상태 | 영역 |
|---|---|---|---|
| 6 | `.cpu()` GPU→CPU sync 잔존 (1회/forward) | **△ 부분 해소** | 80x → 1x/forward sync 잔존. 0회 영역은 별도 plan (forward 시작 시 pre-compute 영역) |
| 7 | forward_context cache `try/except: pass` best-effort | **✗ 미해결** | cache 실패 시 fallback 영역 측정 부재. `attention.py:961` 영역 |
| 8 | cudagraph `.cpu()` graph break 가능성 | **△ 부분 검증** | E16 결과 race/sync/OOB 0 (memory safety 영역). graph break 자체는 미검증 |

### 8.C. 22 항목 본질

| # | 문제 | 상태 | 영역 |
|---|---|---|---|
| 9 | 항목 15 (NEO > vanilla) fail | **△ root 식별 + 미해결** | E20-alt 결과: cdec_wait 5.6-10.7ms × 80 layer = 400-800ms/step. throughput 247 tps regression 의 직접 root. 별도 plan 영역 |
| 10 | 항목 16 (CPU util HIGH) 미측정 | **✗ 미실행** | py-spy 14-thread OMP duty cycle plan |
| 11 | 항목 17 (token correctness) 미측정 | **✗ 미실행** | TST_003 분포 동등성 plan |

### 8.D. 분석 도구 / 인프라

| # | 문제 | 상태 | 영역 |
|---|---|---|---|
| 13 | Unknown env warning (VLLM_NEO_OPTION_*) | **✗ 미해결** | `vllm/envs.py` 의 `environment_variables` dict 등록 영역 큰 patch 필요 (line 489) |
| 14 | launcher pgrep cleanup fail (zombie worker) | **△ 부분 해소** | 일부 launcher 직접 PID kill 패턴. 전체 launcher 통일 영역 미적용 |
| 15 | result.json 미생성 (5min run / 30min incomplete) | **✗ 미해결** | run 시간 영역 / num_prompts 조정 정합성 영역 |

### 8.E. 분석 도구 진행 상태

| # | 도구 | 상태 |
|---|---|---|
| 16 | compute-sanitizer | ✓ 완료 (race/sync/OOB 0) |
| 17 | cuda-gdb + core dump | ✓ 완료 (OMP barrier wait 영역 캡처) |
| 18 | torch.profiler Chrome trace | ✓ 완료 (8 worker × 350MB) |
| 19 | ncu (Nsight Compute) | △ 부분 완료 (re-sample 필요) |
| 20 | gprof (RelWithDebInfo rebuild) | △ 권한 대기 (alternative 진행됨) |
| 20-alt | NEO_PROFILE + py-spy --native | ✓ 완료 |

### 8.F. 측정 영역 inconsistency

| # | 문제 | 상태 |
|---|---|---|
| 21 | 5min vs 30min last 50 avg 의미 영역 다름 | **✗ 미해결** |
| 22 | engine init time (~4min) 포함 영역 영향 | **✗ 미해결** |

---

## 9. 관련 파일 / 경로 색인

### 9.1 Core 코드 파일

| 파일 | 영역 |
|---|---|
| `vllm/model_executor/layers/attention/attention.py` | cdec dispatch path (line 700~1200). v1.5.2 fix 영역 line 873~, 936~. D-cdec-trace counter line 780~820. NEO_PROFILE line 1146 |
| `vllm/v1/core/sched/neo_scheduler_adapter.py` | NEO scheduler. Option C path line 789-899. v1.5.1 NameError fix top + line 469 |
| `vllm/v1/worker/gpu_worker.py` | worker execute_model (line 754~). E18 torch.profiler instrument 영역 |
| `vllm/v1/worker/gpu_model_runner.py` | NEO fork branch + cdec slice 영역 (line 2467, 3745, 4360). NEO_PROFILE line 6406 |
| `vllm/v1/executor/multiproc_executor.py` | worker_main (line 800~). faulthandler hook line 826 |
| `vllm/v1/attention/backend.py` | `CommonAttentionMetadata` (line 380~). `seq_lens_cpu` property line 431 |
| `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionMetadata` (line 219~). seq_lens_cpu **부재** |
| `vllm/v1/attention/backends/utils.py` | `_seq_lens_cpu=seq_lens_cpu` set 영역 (line 359, 417) |
| `vllm/forward_context.py` | `neo_cdec_*` fields (line 160-167) |
| `vllm/envs.py` | `environment_variables` dict (line 489). NEO env 등록 영역 부재 → warning spam |
| `csrc/cpu/pacpu/core.h` | pacpu kernel (store_kv / qk_product / softmax / av_product). gprof line-level 분석 영역 |

### 9.2 Analysis launch script

| 영역 | 파일 | 비고 |
|---|---|---|
| cdec trace | `eval/run_probe_cdec_trace.sh` | D-cdec-trace 환경 5min run |
| cdec eager (cudagraph disprove) | `eval/run_probe_cdec_eager.sh` | enforce_eager=true |
| 30min sustain | `eval/run_probe_step2_30min.sh` | v1.5 sustain. cleanup 영역 patch 됨 |
| Phase 6 full | `eval/run_probe_phase6_full.sh` | NEO_PROFILE + py-spy + ncu 영역 |
| NEO PROFILE | `eval/run_probe_neo_profile.sh` | VLLM_NEO_PROFILE=1 활성 |
| Phase 1+2 | `eval/run_probe_phase1_2.sh` | faulthandler / core dump setup |
| OMP idle | `eval/run_probe_omp_idle.sh` | OMP duty cycle 영역 |
| **E16** sanitizer | `eval/run_probe_sanitizer.sh` | compute-sanitizer memcheck 360s |
| **E17** cuda-gdb | `eval/run_probe_cuda_gdb.sh` | cuda-gdb attach + thread bt |
| **E18** torch.profiler | `eval/run_probe_torch_profiler.sh` | Chrome trace 8 worker |
| **E19** ncu | `eval/run_probe_ncu.sh` | Nsight Compute kernel detail |
| **E20-alt** NEO_PROFILE + py-spy --native | `eval/run_probe_e20_alt.sh` | gprof rebuild alternative |
| 22 항목 monitor | `eval/neo_22_items_monitor.sh` | fire fact 자동 추출. OPT_C v1/v2 둘 다 catch 영역 patch 됨 (line 41) |

### 9.3 결과 디렉토리

| 영역 | dir | 시각 (KST) |
|---|---|---|
| v1.4 정통 reference | `eval/results/20260511_021304_try84_v4_K_OptIKCLM2/` | 05-11 02:13 |
| v1.5 commit (try102) | `eval/results/20260511_092250_try102_v5_clean/` | 05-11 09:22 |
| v1.5.1 30min run | `eval/results/20260512_014016_step2_30min_nameerror_fix/` | 05-12 01:40 |
| cdec trace v5 (skip 영역) | `eval/results/20260512_090511_cdec_trace_short/` | 05-12 09:05 |
| cdec trace v6 (fix verify) | `eval/results/20260512_091250_cdec_trace_short/` | 05-12 09:12 |
| cdec trace v7 (cache) | `eval/results/20260512_091919_cdec_trace_short/` | 05-12 09:19 |
| **v1.5 30min sustain (현 baseline)** | `eval/results/20260512_092529_step2_30min_nameerror_fix/` | 05-12 09:25 |
| vanilla baseline | `eval/results/20260509_014854_try52_v3_C8_phaseD_vanilla_baseline/` | 05-09 01:48 |
| **E16 sanitizer** | `eval/results/20260512_103955_E16_sanitizer_memcheck/` | 05-12 10:39 |
| **E17 cuda-gdb** | `eval/results/20260512_104616_E17_cuda_gdb/` | 05-12 10:46 |
| **E18 torch.profiler** | `eval/results/20260512_110346_E18_torch_profiler/` | 05-12 11:03 |
| **E19 ncu** | `eval/results/20260512_111117_E19_ncu/` | 05-12 11:11 |
| **E20-alt** | `eval/results/20260512_112347_E20_alt_neo_profile_pyspy_native/` | 05-12 11:23 |

### 9.4 Plan / 문서

| 영역 | 파일 |
|---|---|
| v1.5 SEGV 분석 plan | `/root/.claude/plans/commit-rustling-gizmo.md` |
| 22 항목 정의 | `shadow_assists/features/IDE_006/Objective-for-NEO-porting.md` |
| NEO baseline 결과 | `shadow_assists/features/IDE_006/PLN_001_neo_baseline_results.md` |
| NEO code deepdive | `shadow_assists/features/IDE_006/NEO_code_deepdive.md` |
| 본 문서 | `shadow_assists/features/IDE_006/Performance_analaysis_v1.5.md` |

### 9.5 환경변수 영역 (NEO 관련)

| 변수 | 값 (v1.5 영역) | 영역 |
|---|---|---|
| `VLLM_NEO_PREDICTOR` | heuristic | scheduler predictor |
| `VLLM_NEO_LOAD_AWARE_MIN_RUNNING` | 32 | D14 load aware |
| `VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP` | 2 | swap_out 영역 |
| `VLLM_NEO_FORCE_SWAP_IN` | 1 | swap_in 영역 |
| `VLLM_NEO_MAX_SWAP_IN_PER_STEP` | 4 | swap_in 영역 |
| `VLLM_NEO_CPU_RESIDENT_REQS` | 64 | mirror size 영역 |
| `VLLM_NEO_SWAP_IN_ORDER` | oldest | swap_in 영역 |
| `VLLM_NEO_MIRROR_MIN_BUFFER` | 8 | mirror buffer 영역 |
| `VLLM_NEO_OPTION_K` | 1 | Option K |
| `VLLM_NEO_OPTION_C` | 1 | Option C (D17C) |
| `VLLM_NEO_OPTION_L` | 1 | Option L (BUF EXTEND) |
| `VLLM_NEO_OPTION_M2` | 1 | Option M2 (swap-in sync) |
| `VLLM_NEO_OPTION_C_FULL_MIRROR` | 1 | Option C v2 영역 (warning spam 발생) |
| `VLLM_NEO_OPTION_O2` | unset | budget guard 영역 (default off) |
| `VLLM_NEO_OPTION_A` | unset | brute force cdec_ids 영역 (default off) |
| **분석 한정 env** | | |
| `VLLM_DEBUG_FAULTHANDLER` | (1 시 활성) | silent SEGV trap |
| `VLLM_DEBUG_CDEC_PATH` | (1 시 활성) | D-cdec-trace counter |
| `VLLM_DEBUG_TORCH_PROFILER` | (1 시 활성) | E18 chrome trace |
| `VLLM_NEO_PROFILE` | (1 시 활성) | NEO component breakdown |
| `VLLM_E18_TRACE_DIR` | (출력 영역) | E18 trace output |

### 9.6 Tool path

| 도구 | 경로 |
|---|---|
| python | `/workspace/vllm_dev_prj/bin/python` |
| py-spy | `/workspace/vllm_dev_prj/bin/py-spy` |
| compute-sanitizer | `/usr/local/cuda/bin/compute-sanitizer` |
| cuda-gdb | `/usr/local/cuda/bin/cuda-gdb` |
| ncu (Nsight Compute) | `/usr/local/cuda/bin/ncu` |
| gprof | `/usr/bin/gprof` |
| cuobjdump | `/usr/local/cuda/bin/cuobjdump` |

### 9.7 Memory feedback 영역 (Claude 영역)

| 영역 | 파일 |
|---|---|
| LD_PRELOAD libcuda 필수 | `/root/.claude/projects/-workspace-vllm-hybrid/memory/project_ld_preload_libcuda.md` |
| commit 권한 명시 후 | `/root/.claude/projects/-workspace-vllm-hybrid/memory/feedback_no_auto_commit.md` |
| 시간 표기 KST | `/root/.claude/projects/-workspace-vllm-hybrid/memory/feedback_time_kst.md` |
| NEO 비활성화 금지 | `/root/.claude/projects/-workspace-vllm-hybrid/memory/feedback_no_neo_disable.md` |
| 사과 수식어 금지 | `/root/.claude/projects/-workspace-vllm-hybrid/memory/feedback_no_apology_phrases.md` |
| 사용자 감정 묘사 금지 | `/root/.claude/projects/-workspace-vllm-hybrid/memory/feedback_no_emotion_phrase.md` |
| build 병렬도 | `/root/.claude/projects/-workspace-vllm-hybrid/memory/feedback_build_parallelism.md` |

---

## 10. context 복원 시 진입 영역

새 세션에서 v1.5 영역 context 복원 시:

1. **현재 코드 상태**: `git status` + `git diff vllm/model_executor/layers/attention/attention.py` 으로 v1.5.2 fix 영역 확인
2. **최신 결과**: `eval/results/20260512_092529_step2_30min_nameerror_fix/` 의 result.json (없으면 engine.log.stdout 의 last 50 avg)
3. **22 항목 fact**: `bash eval/neo_22_items_monitor.sh` 또는 본 문서 §3.2
4. **silent skip root**: 본 문서 §2.3
5. **throughput regression root**: 본 문서 §6.2
6. **TODO**: 본 문서 §8

다음 step 결정:
- commit 영역 → §9.7 의 commit 권한 영역 확인 → 사용자 명시 후 진행
- 후속 분석 → 본 문서 §7 plan 중 선택
- 22 항목 16/17 측정 → §7 plan 영역
