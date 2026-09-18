# CODE_MAP — IDE_076 (현재 실행 소스 기준, 2026-09-17 22:55 KST)

소스 루트: 컨테이너 `sgl-kt:/sgl-workspace/ktransformers/kt-kernel` (upstream 6d460cc + 로컬 수정). 파일 SHA 는 `SOURCE_MANIFEST.json`. 행 번호는 현재 파일 기준(수정 시 갱신).

## 1. Cold job 의 실제 실행 경로 (Qwen OPT4, bs=63 디코드)
| 단계 | 위치 | 내용 |
|---|---|---|
| go 관측·enqueue | `cpu_backend/cpuinfer.h::CPUInfer::poll_loop_` | callback-free 폴러가 GPU go 플래그를 보고 imm/done-setter/deferred task 를 단일 FIFO(`task_queue.cpp::TaskQueue`) 에 enqueue |
| task 실행 | `task_queue.cpp::TaskQueue::worker` (스레드 `kt-task-worker`) | FIFO 순서로 std::function 실행. deferred task = `TP_MOE_Common::forward` (moe-tp.hpp:205) |
| NUMA 분기 | `operators/moe-tp.hpp:221 do_numa_job` | `NumaJobDistributor` 가 서브풀 2개(각 48 워커) 에 같은 job 을 fork → 각 `tps[numa]->forward(qlen,k,...)` → join 후 merge |
| 서브풀 forward | `operators/amx/moe_base.hpp::forward` (L192) | qlen>1 → `forward_prefill` (L210). **bs=63 deferred job 은 qlen=64 이므로 이 경로** |
| S0 gather | `forward_prefill` L327~ `direct_or_pool(qlen, …)` | 토큰→expert 행 복사 (memcpy hidden×2 B) |
| S1 gate_up A 양자화 | L≈367 `direct_or_pool(activated_expert, from_mat)` | expert 당 1 task: BufferA(int8, per-row scale) 생성 |
| S2 gate/up GEMM | L394 `pool->do_work_stealing_job(nth*activated_expert*2, T::config, …)` | nth = ceil(1280/128)=10 → expert 당 20 task; task = `do_gate_up_gemm` (moe.hpp:246 → `vec_mul` → `integer_mat_mul<Int4,false>` (amx_kernels.hpp:2635) → RB `avx_kernel_rb` (:1832)) + `to_mat` (int32→bf16, 스케일 적용) |
| S3 activation | L≈418 `apply_activation` (L788) → `do_work_stealing_job(nth*activated_expert)` | SwiGLU, expert×nth task |
| S4 down A 양자화 | L≈432 `do_work_stealing_job(activated_expert, from_mat)` | expert 당 1 task (KT_QA_PAR 미설정) |
| S5 down GEMM | L457 `do_work_stealing_job(nth*activated_expert, T::config, …)` | nth = ceil(6144/128)=48 → expert 당 48 task; `do_down_gemm` + `to_mat` |
| S6 가중 합 | L475 `do_work_stealing_job(qlen, …)` | 토큰당 1 task, routing weight 가중 합 (AVX-512) |
| 기록 | L≈502 `kt_rec_` stage_us | IDE_075 v3 기록기 (OFF 에서는 rec==nullptr) |

각 `do_work_stealing_job` (`cpu_backend/worker_pool.cpp:143`) = `do_work_stealing_job_async` (워커 min(48, task_num) 개에 mutex lock + `cv.notify_one`; guided block=1: task 당 `curr_.fetch_add`) + 부모(스레드 0)도 task 처리 + `wait()` (워커 status 를 순서대로 spin). 워커는 50 ms 동안 spin 후 cv sleep. → **job 당 서브풀별 장벽 7회(S0~S6), notify 최대 48×7, atomic fetch_add ≈ task 수(≈18 expert × (1+1+20+10+1+48) + 64 ≈ 1,500)**. 이것이 A2 의 조사 대상(§7.1).

## 2. A1 대상: 기존 RB 커널
- 진입: `integer_mat_mul<GemmKernel224Int4, amx_or_avx=false>` (amx_kernels.hpp:2635) → `K::avx_rb_on()` (env `KT_AVX_RB` 존재 여부; OPT4 는 `KT_AVX_RB=0` → ON) → K_BLOCK(3,584)×M_STEP(32)×N_STEP(32) 타일 루프 → `avx_kernel_rb` (:1832): 행 블록을 8/4/2/1 로 나눠 `avx_rb_rows<RB,SPLIT>` (:1777) 호출 (R=1 → `<1,4>`, R=2 → `<2,2>`, R=3 → `<2,2>`+`<1,4>`), 마지막 K 블록 뒤 `apply_scale` (:1541, 행별 as × 열별 bs × int32→fp32).
- `avx_rb_rows<RB,SPLIT>`: acc[RB][2][SPLIT] (RB×2×SPLIT zmm), K_STEP 64 단위로 A(int8, 행당 16×int32 = 64 B) 를 `_mm512_set1_epi32` 브로드캐스트(행·k_i 당 2회), B(int4 packed 2 KB/32열×128k) 를 k_i 당 2 로드 + and/shift 4회로 lo/hi 언팩, `vpdpbssd` RB×4/k_i. 프리페치 `KT_AVX_PF`(기본 0).
- 데이터 배치: BufferA = `BufferAImpl<Int4>` (int8 [max_m×k] + 행 스케일 float), BufferB = `BufferBInt4Impl` (n×k/2 B + 열 스케일; 타일 32×128 → 2,048 B), BufferC = int32/fp32 [max_m×n]. NUMA-local shard: gate/up K=6,144·N=1,280, down K=1,280·N=6,144.
- 실제 shape 분포(IDE_075 expert 표본): expert 당 rows ≤2 가 67.7 %(bs=63), rows 1~5.
- asm 증거: `A1_GAP_ANALYSIS.md` (컴파일 -O3 -march=native 결과의 명령 수).

## 3. Hot/Cold 배치·H2D·결합 (B 관련)
- `site-packages/kt_kernel/experts_base.py::submit_forward` — 슬롯 arm/rearm, go 전 slot→layer 맵 기록(v3), gpu_experts_mask 로 Hot/Cold 분리, cold 결과 H2D(pinned) 후 결합.
- `sglang/srt/layers/moe/kt_ep_wrapper.py` — KTEPWrapperMethod: GPU fused_moe(Hot) + KT(Cold) 합성, `KT_GPU_EXPERTS_PER_LAYER` (층별 budget JSON) 주입 (`eval/ide070/patch_per_layer_experts.sh` 로 부팅 시 패치), `init-expert-location` hotmap (logical→physical).
- H0: `/models/kt/ide070/hotmap_v2.json` (7ce02cad…), `/models/kt/ide070/layer_budget_5952.json` (c8ba4247…).

## 4. 러너·분석기
- `eval/ide076/runner.py` — IDE_075 하네스 재사용(importlib), 캠페인 `eval/results/IDE_076_20260917`, 원장 `state/execution_events.jsonl`, 변형 manifest `variants/<id>/manifest_<boot>.json`, 플랜 R0_REF/R0_PROBE/MAIN3/MAIN3_C1_LONG/CORR1, `--env K=V --hotmap --budget`.
- 재사용 분석기: `eval/ide075/{dependency_v2.py, analyze_costs_v2.py, validate_v2.py, expert_samples.py}` (v2 지표·수명 검사), `eval/ide074/gpu_layer_timeline.py`.
- 비교기 후보: `kt-kernel/examples/test_moe_amx.py` (AMXInt4_MOE, CPUInfer, expert 단위 forward_task) → `eval/ide076/expert_comparator.py` 로 확장 (C04/C06).
