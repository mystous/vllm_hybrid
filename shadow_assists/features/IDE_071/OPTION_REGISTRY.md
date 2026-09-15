# OPTION_REGISTRY — IDE_071 (P0)

생성 2026-09-16T00:44:15.097709+09:00. 설치본: SGLang 0.5.18 @71de97b (+로컬 수정), kt-kernel 0.7.0.post1 (컨테이너 sgl-kt). `installed_cli_supported` 는 `python3 -m sglang.launch_server --help` (evidence/sglang_launch_server_help.txt) 기준. 실제 적용 여부는 셀별 `runtime_proofs.json`/`effective_config.json`.

| name | category | source_kind | installed_cli_supported | source / default | semantics (source) |
|---|---|---|---|---|---|
| `--cuda-graph-max-bs` | graph | LOCAL_CODE | True | /sgl-workspace/sglang/python/sglang/srt/server_args.py |  |
| `--cuda-graph-bs` | graph | LOCAL_CODE | True | /sgl-workspace/sglang/python/sglang/srt/server_args.py |  |
| `--cuda-graph-max-bs-decode` | graph | LOCAL_CODE | True | 1879:    cuda_graph_max_bs_decode: A[ | 별칭/단계별 |
| `--cuda-graph-bs-decode` | graph | LOCAL_CODE | True | 1889:    cuda_graph_bs_decode: A[ |  |
| `--cuda-graph-backend-prefill` | graph | LOCAL_CODE | True | 1871:    cuda_graph_backend_prefill: A[ |  |
| `--disable-cuda-graph` | graph | LOCAL_CODE | True | 1914:    disable_cuda_graph: A[bool, Arg(no_cli=True), NS("exec.graph")] = False |  |
| `--disable-cuda-graph-padding` | graph | LOCAL_CODE | True | 1915:    disable_cuda_graph_padding: A[ |  |
| `--enable-breakable-cuda-graph` | graph | LOCAL_CODE | True | /sgl-workspace/sglang/python/sglang/srt/server_args.py |  |
| `--disable-piecewise-cuda-graph` | graph | LOCAL_CODE | True | /sgl-workspace/sglang/python/sglang/srt/server_args.py |  |
| `--piecewise-cuda-graph-tokens` | graph | LOCAL_CODE | True | /sgl-workspace/sglang/python/sglang/srt/server_args.py |  |
| `--max-total-tokens` | kv | LOCAL_CODE | True | 787:    max_total_tokens: A[ |  |
| `--mem-fraction-static` | kv | LOCAL_CODE | True | 774:    mem_fraction_static: A[ |  |
| `--kv-cache-dtype` | kv | LOCAL_CODE | True | 680:    kv_cache_dtype: A[ |  |
| `--page-size` | kv | LOCAL_CODE | True | 891:    page_size: A[ |  |
| `--chunked-prefill-size` | sched | LOCAL_CODE | True | 801:    chunked_prefill_size: A[ |  |
| `--prefill-max-requests` | sched | LOCAL_CODE | True | 823:    prefill_max_requests: A[ |  |
| `--max-prefill-tokens` | sched | LOCAL_CODE | True | 811:    max_prefill_tokens: A[ |  |
| `--enable-mixed-chunk` | sched | LOCAL_CODE | True | 981:    enable_mixed_chunk: A[ |  |
| `--max-running-requests` | sched | LOCAL_CODE | True | 779:    max_running_requests: A[ |  |
| `--schedule-conservativeness` | sched | LOCAL_CODE | True | 886:    schedule_conservativeness: A[ |  |
| `--schedule-policy` | sched | LOCAL_CODE | True | 828:    schedule_policy: A[ |  |
| `--num-continuous-decode-steps` | sched | LOCAL_CODE | True | 971:    num_continuous_decode_steps: A[ |  |
| `--scheduler-recv-interval` | sched | LOCAL_CODE | True | 976:    scheduler_recv_interval: A[ |  |
| `--disable-overlap-schedule` | sched | LOCAL_CODE | True | 963:    disable_overlap_schedule: A[ |  |
| `--attention-backend` | attn | LOCAL_CODE | True | 1698:    attention_backend: A[ |  |
| `--prefill-attention-backend` | attn | LOCAL_CODE | True | 1716:    prefill_attention_backend: A[ |  |
| `--decode-attention-backend` | attn | LOCAL_CODE | True | 1707:    decode_attention_backend: A[ |  |
| `--triton-attention-num-kv-splits` | attn | LOCAL_CODE | True | 1961:    triton_attention_num_kv_splits: A[ |  |
| `--moe-runner-backend` | moe | LOCAL_CODE | True | 2366:    moe_runner_backend: A[ |  |
| `--disable-custom-all-reduce` | comm | LOCAL_CODE | True | 1994:    disable_custom_all_reduce: A[ |  |
| `--enable-flashinfer-allreduce-fusion` | comm | LOCAL_CODE | True | 2027:    enable_flashinfer_allreduce_fusion: A[bool, Arg(no_cli=True), NS("exec.comm")] =  |  |
| `--tokenizer-worker-num` | api | LOCAL_CODE | True | 520:    tokenizer_worker_num: A[ |  |
| `--detokenizer-worker-num` | api | LOCAL_CODE | True | 523:    detokenizer_worker_num: A[ |  |
| `--enable-tokenizer-batch-encode` | api | LOCAL_CODE | True | 3445:    enable_tokenizer_batch_encode: A[ |  |
| `--kt-weight-path` | kt | LOCAL_CODE | True | 3045:    kt_weight_path: A[ |  |
| `--kt-method` | kt | LOCAL_CODE | True | 3050:    kt_method: A[ |  |
| `--kt-cpuinfer` | kt | LOCAL_CODE | True | 3055:    kt_cpuinfer: A[ |  |
| `--kt-threadpool-count` | kt | LOCAL_CODE | True | 3060:    kt_threadpool_count: A[ |  |
| `--kt-num-gpu-experts` | kt | LOCAL_CODE | True | 3065:    kt_num_gpu_experts: A[ |  |
| `--kt-max-deferred-experts-per-token` | kt | LOCAL_CODE | True | 3070:    kt_max_deferred_experts_per_token: A[ |  |
| `--init-expert-location` | eplb | LOCAL_CODE | True | 2405:    init_expert_location: A[str, "Initial location of EP experts.", NS("exec.moe")] = |  |
| `--ep-dispatch-algorithm` | eplb | LOCAL_CODE | True | 2400:    ep_dispatch_algorithm: A[ |  |
| `--ep-size` | parallel | LOCAL_CODE | True | 2338:    ep_size: A[ |  |
| `--enable-dp-attention` | parallel | LOCAL_CODE | True | 1148:    enable_dp_attention: A[ |  |
| `--dp-size` | parallel | LOCAL_CODE | True | 1042:    dp_size: A[ |  |
| `--speculative-algorithm` | spec | LOCAL_CODE | True | 2074:    speculative_algorithm: A[ |  |
| `--speculative-draft-model-path` | spec | LOCAL_CODE | True | 2079:    speculative_draft_model_path: A[ |  |
| `--speculative-num-steps` | spec | LOCAL_CODE | True | 2100:    speculative_num_steps: A[ |  |
| `--speculative-eagle-topk` | spec | LOCAL_CODE | True | 2105:    speculative_eagle_topk: A[ |  |
| `--speculative-num-draft-tokens` | spec | LOCAL_CODE | True | 2110:    speculative_num_draft_tokens: A[ |  |
| `--enable-hierarchical-cache` | hicache | LOCAL_CODE | True | 2668:    enable_hierarchical_cache: A[bool, "Enable hierarchical cache", NS("memory")] = ( |  |
| `--hicache-size` | hicache | LOCAL_CODE | True | 2676:    hicache_size: A[ |  |
| `KT_CALLBACK_FREE` | env | LOCAL_CODE | None | kt_kernel/experts_base.py:880,996 | set(비어있지 않은 문자열) 이면 패킷 등록 + GPU memop 트리거 경로; unset 이면 cudaLaunchHostFunc 경로. '0' 도 set 으로 취급 (bool(str)) |
| `KT_CF_SKIP_EMPTY_IMM` | env | LOCAL_CODE | None | kt_kernel/experts_base.py:853 | callback-free AND max_deferred >= top-k(8) AND 이전 층 deferred 보류 있음 일 때 immediate 작업 생략. def0/4 에서는 효과 없음 |
| `KT_COLD_DEFER` | env | LOCAL_CODE | None | kt_kernel/experts_base.py:553 | set 이면 deferral 기준을 'cold(GPU 미상주) AND score < KT_COLD_TAU' 로 바꿈 (hot 은 항상 immediate). 미설정 시 protected_k(=topk-max_deferred) topk 보호 방식 |
| `KT_COLD_TAU` | env | LOCAL_CODE | None | kt_kernel/experts_base.py:555 | float, 기본 1.0. KT_COLD_DEFER 와 함께만 의미 |
| `KT_AVX_RB` | env | LOCAL_CODE | None | kt-kernel/operators/amx/la/amx_kernels.hpp:1769 | getenv!=NULL 이면 레지스터 블로킹 AVX-512 커널 (값 무관; '0' 도 ON). 정수 누산이므로 bit 동일 (주석) |
| `KT_AVX_PF` | env | LOCAL_CODE | None | kt-kernel/operators/amx/la/amx_kernels.hpp:1773 | atoi → prefetch distance, 기본 0 |
| `KT_FUSE_QIN` | env | LOCAL_CODE | None | kt-kernel/operators/amx/moe_base.hpp:511 | getenv!=NULL 이면 입력 quantization 결합 (값 무관) |
| `KT_QA_PAR` | env | LOCAL_CODE | None | kt-kernel/operators/amx/moe_base.hpp:515 | getenv!=NULL 이면 quantize-A 병렬 |
| `KT_AMX_MIN_ROWS` | env | LOCAL_CODE | None | kt-kernel/operators/amx/moe.hpp:240 | atoi, 기본 1<<30. GEMM 선택: qlen > MIN_QLEN OR m >= MIN_ROWS → amx::mat_mul, 아니면 amx::vec_mul(AVX-512 VNNI) |
| `KT_AMX_MIN_QLEN` | env | LOCAL_CODE | None | kt-kernel/operators/amx/moe.hpp:244 | atoi, 기본 4*expert_num/topk = 80. 1000000 이면 qlen 조건이 사실상 거짓 → MIN_ROWS 만으로 AMX 선택 |
| `KT_HUGEPAGE` | env | LOCAL_CODE | None | kt-kernel/operators/amx/moe_base.hpp:16 | getenv!=NULL 이면 madvise hugepage |
| `KT_AMX_BCACHE` | env | LOCAL_CODE | None | amx_kernels.hpp:1893 | getenv!=NULL |
| `KT_AMX_A_LOADD` | env | LOCAL_CODE | None | amx_kernels.hpp:1897 | getenv!=NULL |
| `KT_GPU_EXPERTS_PER_LAYER` | env | LOCAL_CODE | None | sglang kt_ep_wrapper.py (IDE_070 패치, 컨테이너 로컬) | json per_layer[layer_idx] → KTConfig.num_gpu_experts. 패치 적용 시에만 |
| `KT_FORCE_SYNC_SUBMIT` | env | LOCAL_CODE | None | kt_kernel/experts_base.py:75 | '1' 이면 동기 submit/sync 경로 |
| `SGL_INTERLEAVE` | env | LOCAL_CODE | None | kt_kernel/experts_base.py:998 | signal 슬롯 대기 경로 (IDE_030) |
| `KT_CF_SKIP_EMPTY_DEF` | env | LOCAL_CODE | None | kt-kernel/cpu_backend/cpuinfer.h:139 | 빈 deferred 생략 |
| `KT_PHASE_PROF` | env | LOCAL_CODE | None | cpu_backend | phase 계측 (진단용) |
| `KT_TQ_TIMING` | env | LOCAL_CODE | None | cpu_backend | TaskQueue 계측 (진단용) |
| `KT_MOE_PHASE_TIMING` | env | LOCAL_CODE | None | cpu_backend | MoE phase 계측 |
| `polling interval / bounded spin` | patch | NEW_PATCH_PROPOSAL | False |  | §7.4 로컬 옵션 없음 (cpuinfer.h poll_loop_ 는 고정 spin) |
| `exact grouped expert execution` | patch | NEW_PATCH_PROPOSAL | False |  | §7.4 |
| `dynamic GPU expert cache/prefetch` | patch | NEW_PATCH_PROPOSAL | False |  | §10.4 |
| `per-layer non-uniform GPU experts` | patch | LOCAL_CODE | False |  | IDE_070 패치 (patch_per_layer_experts.sh) |
| `hot expert replication` | patch | NEW_PATCH_PROPOSAL | False |  | §10.5 |
| `routing-aware request scheduling` | patch | NEW_PATCH_PROPOSAL | False |  | §10.5 |
