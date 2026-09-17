# layer/stage aggregate — qwen (프로브 산출의 기계적 집계)

## CPU expert 작업 (D1, KT_PHASE_PROF 64회당 1회 표본, µs; numa 서브풀별 1행 = 층 작업의 절반)

표본 754 행

| qlen 구간 | n | activated_expert p50 | total p50/p95/p99 | up_gate p50 | down p50 | q_input p50 | cpy_input p50 | weight p50 |
|---|---|---|---|---|---|---|---|---|
| decode(qlen≤80) | 732 | 8 | 652/1224/1451 | 353 | 214 | 18 | 15 | 17 |
| prefill(qlen>80) | 22 | 33 | 5494/6968/7500 | 1544 | 1355 | 135 | 769 | 1197 |

activated_expert 별 total µs 중앙값 (decode): 0:10(n=180), 1:120(n=54), 2:168(n=12), 3:270(n=4), 4:348(n=16), 5:434(n=9), 6:485(n=28), 7:561(n=26), 8:622(n=45), 9:680(n=42), 10:774(n=32), 11:815(n=54), 12:880(n=50), 13:952(n=33), 14:1008(n=50), 15:1078(n=31), 16:1144(n=18), 17:1192(n=20), 18:1286(n=13), 19:1376(n=2), 20:1402(n=4), 21:1475(n=5), 22:1554(n=2), 23:1570(n=2)

[kt-tq] TaskQueue 집계 (마지막 3행):

```
Profiling Results (numa[1]): activated_expert: 1, prepare: 3 us, cpy_input: 1 us, q_input: 4 us, up_gate: 70 us, act:[kt-tq] n=2048 exec_us p50=24 p90=144 p99=283 mean=41 | hist(<5,<30,<100,<300,<600,<1k,<2k,<4k,>=4k)=1016,376,276,368,11,1,0,0,0 | wait_us p50=0 p90=203 p99=223 mean=76 | busy=0.35
[kt-tq] n=2048 exec_us p50=24 p90=143 p99=296 mean=43 | hist(<5,<30,<100,<300,<600,<1k,<2k,<4k,>=4k)=1016,319,305,388,19,1,0,0,0 | wait_us p50=0 p90=207 p99=222 mean=75 | busy=0.37
Profiling Results (numa[1]): activated_expert: 0, prepare: 1 us, cpy_input: 0 us, q_input: 0 us, up_gate: 2 us, act: 0 us, q_down: 1 us, down: 1 us, weight: 5 us, total: 11 us, max_local_n[kt-tq] n=2048 exec_us p50=24 p90=138 p99=201 mean=31 | hist(<5,<30,<100,<300,<600,<1k,<2k,<4k,>=4k)=1016,445,33
```

## GPU 타임라인 (D2, SGLang torch profiler CPU+CUDA activity; rank 별 트레이스)

| rank trace | events | GPU ops | GPU busy s | span s | busy 비율 | 커널 간 유휴 gap: n / p50 / p95 / p99 / max µs / 합 s | MoE 커널 수 | MoE 커널 사이 유휴 p50/p95/p99/max µs |
|---|---|---|---|---|---|---|---|---|
| D2_1789610161-TP-0.trace.json.gz | 1523341 | 825802 | 16.438 | 40.961 | 0.401 | 774666 / 0.5126953125 / 10.462890625 / 354.33203125 / 16411161.338867188 / 24.888 | 114266 | 1.087890625/470.2685546875/848.150390625/408809.0439453125 |
| D2_1789610161-TP-1.trace.json.gz | 1363766 | 604262 | 31.758 | 40.962 | 0.775 | 549947 / 0.4482421875 / 1.1826171875 / 2.431640625 / 16411838.08203125 / 17.526 | 114266 | 1.4072265625/6.4326171875/17.3447265625/411287.5341796875 |
| D2_1789610161-TP-2.trace.json.gz | 1364053 | 604262 | 32.151 | 40.961 | 0.785 | 550522 / 0.4482421875 / 1.15234375 / 2.0791015625 / 16411531.216796875 / 17.261 | 114266 | 1.376953125/6.4326171875/7.392578125/408457.1513671875 |
| D2_1789610161-TP-3.trace.json.gz | 1363992 | 604262 | 32.149 | 40.962 | 0.785 | 549089 / 0.4482421875 / 1.18359375 / 2.1435546875 / 16412551.061523438 / 17.289 | 114266 | 1.4072265625/6.43359375/7.6494140625/411893.7744140625 |

### 상위 GPU 커널 (첫 rank 트레이스, 이름 원문, 호출 수, 합계 µs)

| cat | kernel | n | sum µs |
|---|---|---|---|
| gpu_user_annotation | `scheduler.run_batch` | 397 | 24101270.5 |
| gpu_user_annotation | `step[DECODE bs=63]` | 256 | 15083939.9 |
| kernel | `fused_moe_kernel` | 49228 | 7134431.7 |
| gpu_user_annotation | `step[EXTEND bs=17 toks=8192]` | 5 | 3570243.0 |
| gpu_user_annotation | `step[DECODE bs=2]` | 128 | 2007244.0 |
| gpu_memcpy | `Memcpy DtoH (Device -> Pinned)` | 98853 | 1870112.6 |
| gpu_memcpy | `Memcpy HtoD (Pinned -> Device)` | 24716 | 1774908.8 |
| kernel | `_w8a8_block_fp8_matmul` | 49228 | 1618872.5 |
| kernel | `void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lampor` | 48375 | 846696.5 |
| gpu_user_annotation | `step[EXTEND bs=16 toks=8192]` | 1 | 810709.5 |
| gpu_user_annotation | `step[EXTEND bs=15 toks=7171]` | 1 | 689915.1 |
| gpu_user_annotation | `step[EXTEND bs=11 toks=5620]` | 1 | 579748.4 |
| kernel | `void sglang::per_token_group_quant_flat_kernel<sglang::QuantTrait<__nv_bfloat16,` | 98456 | 555873.0 |
| gpu_user_annotation | `nccl:all_reduce` | 1000 | 476082.4 |
| kernel | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1000 | 476080.4 |
| kernel | `_fwd_grouped_kernel_stage1` | 23870 | 420042.3 |
| gpu_user_annotation | `step[EXTEND bs=4 toks=2013]` | 1 | 362605.4 |
| gpu_user_annotation | `step[EXTEND bs=2 toks=1018]` | 1 | 324876.4 |
| gpu_user_annotation | `step[EXTEND bs=1 toks=510]` | 1 | 266003.5 |
| gpu_user_annotation | `step[EXTEND bs=1 toks=1]` | 1 | 265506.3 |
| kernel | `void moe_sum_reduce_warp_per_token_vec_kernel<8>(c10::BFloat16 const*, c10::BFlo` | 682 | 146664.3 |
| kernel | `_fwd_kernel` | 744 | 134922.4 |
| gpu_user_annotation | `scheduler.get_next_batch_to_run` | 397 | 134557.8 |
| kernel | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKerne` | 1381 | 131617.0 |
| kernel | `_fwd_kernel_stage2` | 23870 | 96552.8 |
| kernel | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind)0, true, ` | 24614 | 92160.1 |
| kernel | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c1` | 24614 | 86285.0 |
| kernel | `nvjet_sm90_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 15872 | 78632.5 |
| kernel | `_router_triton_kernel` | 24614 | 73605.4 |
| kernel | `void sglang::fused_rope_store_kernel<true, 128l, true, __nv_bfloat16, __nv_bfloa` | 24614 | 69225.1 |
| kernel | `triton_poi_fused__to_copy_arange_floor_divide_index_remainder_view_0` | 24614 | 68633.3 |
| kernel | `void moe_sum_reduce_kernel<c10::BFloat16, 8>(c10::BFloat16 const*, c10::BFloat16` | 15872 | 66908.5 |
| kernel | `void sglang::fused_qknorm_warp<128l, true, __nv_bfloat16>(sglang::QKNormParams)` | 24614 | 59994.7 |
| kernel | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda` | 27026 | 57994.2 |
| kernel | `void moe_align_block_size_kernel<int>(int const*, int*, int*, int*, int, int, un` | 16554 | 51146.5 |
| gpu_user_annotation | `scheduler.process_batch_result` | 15 | 45184.6 |
| kernel | `nvjet_sm90_tst_288x64_64x4_2x1_v_bz_TNN` | 256 | 44205.6 |
| kernel | `void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_twoshot_sync<(` | 250 | 37745.4 |
| kernel | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __n` | 23994 | 37587.4 |
| kernel | `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>,` | 24614 | 37485.7 |

의존 대기(cold_exposed_wait 등): 트레이스에 CPU→GPU done 신호(memop)·H2D 복사 완료 이벤트가 커널과 별도 항목으로 식별되지 않아 `PARTIAL_DEPENDENCY_MEASUREMENT` — 위 'MoE 커널 사이 GPU 유휴' 가 결합 전 대기의 상한 관측치이며, hot/cold 어느 경로가 늦었는지는 이 트레이스만으로 분리되지 않음.

## CPU 타임라인 (D3, perf record 99 Hz 20 s, 스케줄러 4 프로세스; 상위 25 심볼)

```
     2.87%     0.00%  sglang::schedul  [unknown]                                                                                      [.] 0000000000000000
     2.68%     0.00%  sglang::schedul  [unknown]                                                                                      [k] 0x8b485355fc894900
     2.68%     0.00%  sglang::schedul  libtorch_python.so                                                                             [.] torch::PyWarningHandle
     2.63%     0.00%  sglang::schedul  libtorch_python.so                                                                             [.] THCPEvent_synchronize(
     2.63%     0.00%  sglang::schedul  libcudart.so.13                                                                                [.] cudaEventSynchronize
     2.63%     0.00%  sglang::schedul  libcuda.so.580.126.20                                                                          [.] cuEventSynchronize
     1.44%     1.27%  sglang::schedul  kt_kernel_ext.cpython-312-x86_64-linux-gnu.so                                                  [.] std::thread::_State_im
     1.15%     0.37%  sglang::schedul  python3.12                                                                                     [.] _PyEval_EvalFrameDefau
     1.06%     0.00%  sglang::schedul  python3.12                                                                                     [.] PyEval_EvalCode
     1.03%     0.00%  sglang::schedul  python3.12                                                                                     [.] _start
     1.03%     0.00%  sglang::schedul  libc.so.6                                                                                      [.] __libc_start_main
     1.03%     0.00%  sglang::schedul  python3.12                                                                                     [.] Py_BytesMain
     1.03%     0.00%  sglang::schedul  python3.12                                                                                     [.] Py_RunMain
     1.03%     0.00%  sglang::schedul  python3.12                                                                                     [.] PyRun_SimpleStringFlag
     1.03%     0.00%  sglang::schedul  python3.12                                                                                     [.] PyRun_StringFlags
     1.01%     0.01%  pt_gloo_runloop  [kernel.kallsyms]                                                                              [k] entry_SYSCALL_64_after
     0.99%     0.01%  pt_gloo_runloop  [kernel.kallsyms]                                                                              [k] do_syscall_64
```

perf stat 20 s:

```
 Performance counter stats for process id '1181181,1181306,1181374,1181489':
         41,652.16 msec task-clock                       #    2.082 CPUs utilized
    81,688,853,588      cycles                           #    1.961 GHz
   152,425,034,373      instructions                     #    1.87  insn per cycle
           433,946      context-switches                 #   10.418 K/sec
             5,870      cpu-migrations                   #  140.929 /sec
      20.004523646 seconds time elapsed
```

pcm-memory (2 s, MB/s 단위는 pcm-memory 헤더 정의 'System Read/Write'): 샘플 10, 읽기 평균 2592 MB/s, 쓰기 평균 1732 MB/s, 읽기 최대 5892 MB/s

