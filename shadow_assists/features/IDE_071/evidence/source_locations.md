# source_locations — IDE_071

| symbol | location | note |
|---|---|---|
| KT_CALLBACK_FREE | kt_kernel/experts_base.py:880,996 | set(비어있지 않은 문자열) 이면 패킷 등록 + GPU memop 트리거 경로; unset 이면 cudaLaunchHostFunc 경로. '0' 도 set 으로 취급 (bool(str)) |
| KT_CF_SKIP_EMPTY_IMM | kt_kernel/experts_base.py:853 | callback-free AND max_deferred >= top-k(8) AND 이전 층 deferred 보류 있음 일 때 immediate 작업 생략. def0/4 에서는 효과 없음 |
| KT_COLD_DEFER | kt_kernel/experts_base.py:553 | set 이면 deferral 기준을 'cold(GPU 미상주) AND score < KT_COLD_TAU' 로 바꿈 (hot 은 항상 immediate). 미설정 시 protected_k(=topk-max_deferred) topk 보호 방식 |
| KT_COLD_TAU | kt_kernel/experts_base.py:555 | float, 기본 1.0. KT_COLD_DEFER 와 함께만 의미 |
| KT_AVX_RB | kt-kernel/operators/amx/la/amx_kernels.hpp:1769 | getenv!=NULL 이면 레지스터 블로킹 AVX-512 커널 (값 무관; '0' 도 ON). 정수 누산이므로 bit 동일 (주석) |
| KT_AVX_PF | kt-kernel/operators/amx/la/amx_kernels.hpp:1773 | atoi → prefetch distance, 기본 0 |
| KT_FUSE_QIN | kt-kernel/operators/amx/moe_base.hpp:511 | getenv!=NULL 이면 입력 quantization 결합 (값 무관) |
| KT_QA_PAR | kt-kernel/operators/amx/moe_base.hpp:515 | getenv!=NULL 이면 quantize-A 병렬 |
| KT_AMX_MIN_ROWS | kt-kernel/operators/amx/moe.hpp:240 | atoi, 기본 1<<30. GEMM 선택: qlen > MIN_QLEN OR m >= MIN_ROWS → amx::mat_mul, 아니면 amx::vec_mul(AVX-512 VNNI) |
| KT_AMX_MIN_QLEN | kt-kernel/operators/amx/moe.hpp:244 | atoi, 기본 4*expert_num/topk = 80. 1000000 이면 qlen 조건이 사실상 거짓 → MIN_ROWS 만으로 AMX 선택 |
| KT_HUGEPAGE | kt-kernel/operators/amx/moe_base.hpp:16 | getenv!=NULL 이면 madvise hugepage |
| KT_AMX_BCACHE | amx_kernels.hpp:1893 | getenv!=NULL |
| KT_AMX_A_LOADD | amx_kernels.hpp:1897 | getenv!=NULL |
| KT_GPU_EXPERTS_PER_LAYER | sglang kt_ep_wrapper.py (IDE_070 패치, 컨테이너 로컬) | json per_layer[layer_idx] → KTConfig.num_gpu_experts. 패치 적용 시에만 |
| KT_FORCE_SYNC_SUBMIT | kt_kernel/experts_base.py:75 | '1' 이면 동기 submit/sync 경로 |
| SGL_INTERLEAVE | kt_kernel/experts_base.py:998 | signal 슬롯 대기 경로 (IDE_030) |
| KT_CF_SKIP_EMPTY_DEF | kt-kernel/cpu_backend/cpuinfer.h:139 | 빈 deferred 생략 |
| KT_PHASE_PROF | cpu_backend | phase 계측 (진단용) |
| KT_TQ_TIMING | cpu_backend | TaskQueue 계측 (진단용) |
| KT_MOE_PHASE_TIMING | cpu_backend | MoE phase 계측 |
| KTEPWrapperMethod.apply / submit / sync | /sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py:300-430 | CPU expert 계산은 tp_rank==0 만 수행 (`if self.tp_rank != 0: return`) |
| create_kt_config_from_server_args | kt_ep_wrapper.py:96-131 | num_gpu_experts 균일 (패치 지점) |
| callback-free handoff (IDE_033) | /sgl-workspace/ktransformers/kt-kernel/cpu_backend/cpuinfer.h:129-290 | mapped go/done flags + poller thread |
| KExpertsCPUBuffer (pinned ring) | kt_kernel/experts_base.py:216-270 | pin_memory=True, slot = layer_idx % buffer_depth |
| do_gate_up_gemm / do_down_gemm | kt-kernel/operators/amx/moe.hpp:252-270 | AMX vs AVX(vec_mul) 선택 조건 |
| hotmap_mixed α | eval/harness/20260909_mechanism/build_hotmap_mixed.py | score(e)=α·p_prefill(e)+(1−α)·p_decode(e), prefill 패스 = >64 tok, decode = ≤64 tok |
