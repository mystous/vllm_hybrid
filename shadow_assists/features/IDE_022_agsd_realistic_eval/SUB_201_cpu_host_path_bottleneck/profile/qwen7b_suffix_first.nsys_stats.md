# Qwen-7B suffix nsys stats (재현용)

## cuda_gpu_kern_sum
```

NOTICE: Existing SQLite export found: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/qwen7b_suffix_first.sqlite
        It is assumed file was previously exported from: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/qwen7b_suffix_first.nsys-rep
        Consider using --force-export=true if needed.

Processing [/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/qwen7b_suffix_first.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

+----------+-----------------+-----------+----------+----------+----------+----------+-------------+------------------------------------------------------------------------------------------------------+
| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) |                                                 Name                                                 |
+----------+-----------------+-----------+----------+----------+----------+----------+-------------+------------------------------------------------------------------------------------------------------+
|     65.8 |     24038586640 |    333648 |  72047.7 |  77376.0 |     7808 |   117536 |     25273.9 | fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16VarSeqQ128Kv128PersistentContext            |
|      9.9 |      3602652081 |    273840 |  13156.0 |  13024.0 |     5344 |    19360 |      2075.3 | fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbFo… |
|      3.6 |      1300739111 |     84000 |  15485.0 |  16320.0 |     7040 |    20288 |      2000.9 | fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen |
|      2.9 |      1055128733 |    360752 |   2924.8 |   2944.0 |     1792 |     5632 |       184.1 | void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0… |
|      2.7 |       979257472 |     12884 |  76005.7 |  63776.0 |    14976 |  1507741 |     47015.7 | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)                       |
|      2.4 |       894615585 |     51520 |  17364.4 |  11104.0 |     3744 |    55232 |     11518.4 | void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::… |
|      1.5 |       558925481 |     51096 |  10938.7 |   7424.0 |     3231 |    39360 |      6410.5 | void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<void … |
|      1.4 |       502018195 |     25760 |  19488.3 |  11328.0 |     2368 |    71648 |     13997.6 | void vllm::apply_repetition_penalties_kernel<float>(T1 *, const bool *, const bool *, const T1 *, i… |
|      1.1 |       392092900 |      8232 |  47630.3 |  47584.0 |    45664 |    56384 |       738.8 | nvjet_tst_320x64_64x4_2x1_2cta_v_bz_TNT                                                              |
|      1.1 |       389416499 |     51516 |   7559.1 |   7040.0 |     1600 |    24032 |      2670.7 | void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)8, void at::native::_cuda_scatte… |
|      1.0 |       379375642 |    113108 |   3354.1 |   1664.0 |      864 |    15392 |      2947.2 | void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<ch… |
|      0.9 |       345790546 |     38636 |   8950.0 |   7232.0 |     2688 |    28672 |      5299.9 | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator… |
|      0.9 |       333508027 |     25760 |  12946.7 |  11200.0 |     7808 |    25280 |      3821.5 | void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<… |
|      0.8 |       294250136 |     51520 |   5711.4 |   4032.0 |     1760 |    21440 |      3468.8 | void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::arr… |
|      0.6 |       227922459 |     12884 |  17690.3 |  18400.0 |     3424 |    31936 |      5342.5 | void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n… |
|      0.5 |       193165670 |      3964 |  48730.0 |  48672.0 |    47424 |    53152 |       495.8 | nvjet_tst_144x128_64x6_4x1_v_bz_TNN                                                                  |
|      0.5 |       169856369 |     77264 |   2198.4 |   2208.0 |     1504 |     4256 |       313.9 | void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void … |
|      0.4 |       145500672 |     45088 |   3227.0 |   2432.0 |     1376 |    10272 |      1596.6 | void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, lon… |
|      0.3 |       100758204 |     51528 |   1955.4 |   1952.0 |     1184 |     3136 |       278.3 | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator… |
|      0.2 |        88744119 |       912 |  97307.1 |  82303.5 |    40160 |   374656 |     60982.0 | void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_twoshot_sync<(flashinfer::trtllm_… |
|      0.2 |        55177971 |     25756 |   2142.3 |   2144.0 |     1152 |     3552 |       202.4 | void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<long, long, bool, … |
|      0.1 |        53733570 |     25756 |   2086.3 |   2080.0 |     1184 |     3232 |       252.0 | void at::native::vectorized_elementwise_kernel<(int)2, at::native::<unnamed>::masked_fill_kernel(at… |
|      0.1 |        41018396 |     12884 |   3183.7 |   3168.0 |     2720 |     4992 |       193.7 | _compute_slot_mapping_kernel                                                                         |
|      0.1 |        39140329 |     12876 |   3039.8 |   3072.0 |     1504 |     4096 |       314.5 | rejection_greedy_sample_kernel                                                                       |
|      0.1 |        31313263 |      6212 |   5040.8 |   5152.0 |     2112 |     6048 |       421.3 | void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void … |
|      0.1 |        29733109 |       640 |  46458.0 |  46304.0 |    45248 |    60192 |      1147.2 | nvjet_tst_384x16_64x4_4x1_v_bz_TNT                                                                   |
|      0.1 |        26536208 |      1568 |  16923.6 |  17952.0 |    13728 |    20256 |      1757.7 | fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForG… |
|      0.1 |        26408408 |     19104 |   1382.3 |   1408.0 |      864 |     2080 |       226.9 | void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<cha… |
```

## cuda_api_sum
```

NOTICE: Existing SQLite export found: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/qwen7b_suffix_first.sqlite
        It is assumed file was previously exported from: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/qwen7b_suffix_first.nsys-rep
        Consider using --force-export=true if needed.

Processing [/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/qwen7b_suffix_first.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

+----------+-----------------+-----------+----------+----------+----------+----------+-------------+--------------------------------------------+
| Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) |                    Name                    |
+----------+-----------------+-----------+----------+----------+----------+----------+-------------+--------------------------------------------+
|     36.3 |      5261402681 |   1116364 |   4713.0 |   3928.0 |     2347 | 63517825 |    125578.2 | cudaLaunchKernel                           |
|     20.2 |      2929771200 |    759524 |   3857.4 |   3501.0 |     2589 |  3904110 |      7256.9 | cuLaunchKernelEx                           |
|     17.8 |      2575381527 |    373172 |   6901.3 |   5948.0 |     4706 |   334013 |      2964.9 | cudaGraphLaunch_v10000                     |
|     13.4 |      1948277048 |    308548 |   6314.3 |   3857.0 |     1977 |  3009017 |     22412.2 | cudaMemcpyAsync                            |
|      2.5 |       359996165 |   3041024 |    118.4 |    103.0 |       85 |    21199 |        78.5 | cuTensorMapEncodeTiled                     |
|      2.2 |       312509713 |    694940 |    449.7 |    412.0 |      199 |   113805 |       295.0 | cudaStreamIsCapturing_v10000               |
|      1.9 |       274617192 |     64946 |   4228.4 |   2673.0 |      952 |   249227 |      2771.4 | cudaEventQuery                             |
|      1.2 |       175430890 |     25760 |   6810.2 |   6734.5 |     3886 |    27564 |      1121.9 | cudaMemsetAsync                            |
|      1.0 |       143811941 |     38976 |   3689.8 |   3823.0 |     1774 |    61884 |      1291.4 | cudaEventRecordWithFlags_v11010            |
|      0.9 |       128305681 |     38660 |   3318.8 |   2479.0 |     1591 |   194214 |      2499.0 | cudaEventRecord                            |
|      0.9 |       123785097 |    340032 |    364.0 |    332.0 |      242 |     9219 |       158.7 | cuOccupancyMaxActiveClusters               |
|      0.4 |        59759050 |     12876 |   4641.1 |   4565.0 |     3610 |    18669 |       796.6 | cudaStreamSynchronize                      |
|      0.4 |        57837137 |     26022 |   2222.6 |    181.0 |      159 | 51129577 |    316956.7 | cudaThreadExchangeStreamCaptureMode_v10010 |
|      0.2 |        29873573 |     25768 |   1159.3 |   1521.5 |      331 |    15366 |       804.0 | cudaStreamWaitEvent                        |
|      0.2 |        27771765 |        80 | 347147.1 |  18351.0 |    10622 |  3027703 |    731960.0 | cudaHostAlloc                              |
|      0.2 |        23872929 |     12915 |   1848.5 |   1577.0 |      349 |   583840 |      9644.7 | cudaEventCreateWithFlags                   |
|      0.1 |        15572706 |     12884 |   1208.7 |    936.0 |      775 |   377613 |      9380.7 | cudaEventDestroy                           |
|      0.1 |        10383613 |     12884 |    805.9 |    794.0 |      604 |     8352 |       179.5 | cudaStreamGetCaptureInfo_v2_v11030         |
```

## nvtx_pushpop_sum
```

NOTICE: Existing SQLite export found: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/qwen7b_suffix_first.sqlite
        It is assumed file was previously exported from: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/qwen7b_suffix_first.nsys-rep
        Consider using --force-export=true if needed.

Processing [/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/qwen7b_suffix_first.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/nvtx_pushpop_sum.py]... 

 ** NVTX Push/Pop Range Summary (nvtx_pushpop_sum):

+----------+-----------------+-----------+----------+----------+----------+----------+-------------+--------------------+
| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) |       Range        |
+----------+-----------------+-----------+----------+----------+----------+----------+-------------+--------------------+
|    100.0 |       448485486 |     12884 |  34809.5 |  33717.5 |    30609 |   108985 |      3532.7 | NCCL:ncclAllGather |
+----------+-----------------+-----------+----------+----------+----------+----------+-------------+--------------------+

```
