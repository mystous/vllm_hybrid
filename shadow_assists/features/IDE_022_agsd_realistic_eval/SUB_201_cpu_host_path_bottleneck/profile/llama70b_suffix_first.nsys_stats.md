# Llama-70B suffix nsys stats (재현용)

## cuda_gpu_kern_sum
```

NOTICE: Existing SQLite export found: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/llama70b_suffix_first.sqlite
        It is assumed file was previously exported from: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/llama70b_suffix_first.nsys-rep
        Consider using --force-export=true if needed.

Processing [/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/llama70b_suffix_first.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

+----------+-----------------+-----------+----------+----------+----------+----------+-------------+------------------------------------------------------------------------------------------------------+
| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) |                                                 Name                                                 |
+----------+-----------------+-----------+----------+----------+----------+----------+-------------+------------------------------------------------------------------------------------------------------+
|     76.9 |     53351148261 |    671360 |  79467.3 |  88640.0 |    20544 |   124159 |     25780.2 | fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16VarSeqQ128Kv128PersistentContext            |
|     10.5 |      7309552741 |    549760 |  13295.9 |  13120.0 |     5408 |    18112 |       831.1 | fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbFo… |
|      2.9 |      2013233300 |    671360 |   2998.7 |   3008.0 |     2240 |     6496 |       214.3 | void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0… |
|      2.3 |      1627559125 |    118400 |  13746.3 |  13600.0 |     7488 |    16256 |       640.4 | fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen |
|      1.3 |       893758039 |      9016 |  99130.2 |  86703.5 |    30336 |  2053752 |    127737.4 | void <unnamed>::multimem_all_reduce_kernel<c10::BFloat16, (int)16>(T1 *, unsigned long, unsigned in… |
|      1.2 |       801382093 |      8392 |  95493.6 |  82816.0 |    32544 |  2820661 |    135993.6 | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)                       |
|      0.9 |       621587111 |     16760 |  37087.5 |  31712.0 |    29664 |    51296 |      7151.7 | void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<… |
|      0.6 |       446382087 |     25136 |  17758.7 |  10624.0 |     4224 |    60288 |     11923.6 | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator… |
|      0.5 |       332836627 |      8384 |  39699.0 |  41184.0 |     6240 |    63328 |      9383.9 | void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n… |
|      0.2 |       149851947 |     25144 |   5959.7 |   4096.0 |     1888 |    18560 |      4198.1 | void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, lon… |
|      0.2 |       144639388 |       640 | 225999.0 | 225504.0 |   223647 |   241984 |      2119.4 | nvjet_tst_256x224_64x4_2x2_2cta_h_bz_TNT                                                             |
|      0.2 |       132399905 |      1280 | 103437.4 | 103152.0 |    38080 |   183103 |     64361.4 | nvjet_tst_256x240_64x4_2x1_2cta_v_bz_TNT                                                             |
|      0.1 |        99689945 |      1944 |  51280.8 |  51200.0 |    50016 |    55968 |       598.6 | nvjet_tst_256x80_64x5_2x2_2cta_v_bz_TNT                                                              |
|      0.1 |        86349186 |      4480 |  19274.4 |  13344.0 |     6784 |    44320 |     12900.9 | triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2                                                   |
|      0.1 |        76216414 |      1280 |  59544.1 |  59775.5 |    30047 |    92095 |     28895.0 | nvjet_tst_192x240_64x5_2x1_2cta_v_bz_TNT                                                             |
|      0.1 |        75590393 |       640 | 118110.0 | 118080.0 |   116096 |   123168 |       814.6 | nvjet_tst_256x256_64x4_2x2_2cta_h_bz_TNT                                                             |
|      0.1 |        69396330 |     33544 |   2068.8 |   2048.0 |     1152 |     3552 |       392.3 | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator… |
|      0.1 |        64673039 |     25160 |   2570.5 |   2688.0 |     1375 |     4768 |       514.8 | void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void … |
|      0.1 |        61155024 |      1344 |  45502.3 |  45375.0 |    44160 |    52960 |       785.2 | nvjet_tst_224x64_64x9_1x2_2cta_h_bz_TNN                                                              |
|      0.1 |        57894043 |       640 |  90459.4 |  90176.0 |    88608 |    99008 |      1329.3 | nvjet_tst_256x256_64x4_2x1_2cta_v_bz_TNT                                                             |
|      0.1 |        55826940 |      4480 |  12461.4 |   9024.0 |     5312 |    27840 |      7279.1 | triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0                                                   |
|      0.1 |        47511755 |       936 |  50760.4 |  50656.0 |    49664 |    59072 |       686.6 | nvjet_tst_128x168_64x5_4x1_v_bz_TNT                                                                  |
|      0.1 |        44400191 |       640 |  69375.3 |  69248.0 |    67872 |    74976 |       942.0 | nvjet_tst_256x192_64x4_2x1_2cta_v_bz_TNT                                                             |
|      0.1 |        41145894 |       856 |  48067.6 |  48047.5 |    46784 |    50208 |       517.2 | nvjet_tst_128x144_64x6_4x1_v_bz_TNT                                                                  |
|      0.1 |        40056510 |       640 |  62588.3 |  62592.0 |    60480 |    66784 |       758.7 | nvjet_tst_256x176_64x4_2x4_2cta_v_bz_TNT                                                             |
|      0.1 |        36490633 |       704 |  51833.3 |  51776.0 |    50688 |    54080 |       495.0 | nvjet_tst_256x96_64x5_2x2_2cta_v_bz_TNT                                                              |
|      0.1 |        35383969 |      1280 |  27643.7 |  27504.0 |    15328 |    42336 |     11874.4 | nvjet_tst_256x240_64x4_2x2_2cta_h_bz_TNT                                                             |
|      0.0 |        33314711 |       640 |  52054.2 |  52032.0 |    50144 |    54944 |       721.1 | nvjet_tst_256x144_64x5_2x4_2cta_v_bz_TNT                                                             |
```

## cuda_api_sum
```

NOTICE: Existing SQLite export found: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/llama70b_suffix_first.sqlite
        It is assumed file was previously exported from: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/llama70b_suffix_first.nsys-rep
        Consider using --force-export=true if needed.

Processing [/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/llama70b_suffix_first.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

+----------+-----------------+-----------+-----------+-----------+----------+----------+-------------+--------------------------------------------+
| Time (%) | Total Time (ns) | Num Calls | Avg (ns)  | Med (ns)  | Min (ns) | Max (ns) | StdDev (ns) |                    Name                    |
+----------+-----------------+-----------+-----------+-----------+----------+----------+-------------+--------------------------------------------+
|     80.4 |     65259414097 |    194192 |  336056.1 |    3522.0 |     1789 | 29367074 |   1665992.4 | cudaMemcpyAsync                            |
|      6.6 |      5367120848 |    675216 |    7948.7 |    6166.0 |     4564 |  3718559 |      7848.4 | cudaGraphLaunch_v10000                     |
|      5.8 |      4697533117 |   1401944 |    3350.7 |    3154.0 |     2375 |  1198274 |      1532.1 | cuLaunchKernelEx                           |
|      4.4 |      3548134785 |    887704 |    3997.0 |    3644.0 |     2595 |   637139 |      1883.4 | cudaLaunchKernel                           |
|      0.9 |       726823928 |       528 | 1376560.5 |   25610.5 |      454 | 41935643 |   5813767.6 | cuKernelSetAttribute                       |
|      0.8 |       666010873 |   5539840 |     120.2 |     107.0 |       85 |    20465 |        68.2 | cuTensorMapEncodeTiled                     |
|      0.4 |       353549821 |    843008 |     419.4 |     403.0 |      202 |   533703 |       619.0 | cudaStreamIsCapturing_v10000               |
|      0.2 |       199778888 |    594560 |     336.0 |     321.0 |      237 |    19965 |       123.3 | cuOccupancyMaxActiveClusters               |
|      0.1 |        78545876 |     25176 |    3119.9 |    2216.0 |     1442 |   103174 |      2293.2 | cudaEventRecord                            |
|      0.1 |        70367648 |     25790 |    2728.5 |    2231.0 |      877 |    38760 |      1772.5 | cudaEventQuery                             |
|      0.1 |        56632118 |     18032 |    3140.6 |    2857.0 |     2283 |    29435 |      1255.3 | cuLaunchKernel                             |
|      0.0 |        37972043 |      8376 |    4533.4 |    4421.0 |     3486 |   332508 |      3622.5 | cudaStreamSynchronize                      |
|      0.0 |        31153140 |        96 |  324511.9 |  231448.5 |    42072 |  1621914 |    364312.0 | cuModuleLoadData                           |
|      0.0 |        20231003 |      8384 |    2413.0 |    2229.0 |     1640 |    12111 |       710.1 | cudaEventRecordWithFlags_v11010            |
|      0.0 |        19962226 |     16784 |    1189.4 |    1590.0 |      348 |    10714 |       718.3 | cudaStreamWaitEvent                        |
|      0.0 |        13724493 |      8392 |    1635.4 |    1570.0 |      923 |   337558 |      3683.4 | cudaEventCreateWithFlags                   |
|      0.0 |         8057959 |      8384 |     961.1 |     949.0 |      780 |    32960 |       392.2 | cudaEventDestroy                           |
|      0.0 |         7292406 |         8 |  911550.8 | 1111324.0 |   231740 |  1190001 |    406696.4 | cudaMalloc                                 |
```

## nvtx_pushpop_sum
```

NOTICE: Existing SQLite export found: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/llama70b_suffix_first.sqlite
        It is assumed file was previously exported from: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/llama70b_suffix_first.nsys-rep
        Consider using --force-export=true if needed.

Processing [/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/llama70b_suffix_first.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/nvtx_pushpop_sum.py]... 

 ** NVTX Push/Pop Range Summary (nvtx_pushpop_sum):

+----------+-----------------+-----------+----------+----------+----------+----------+-------------+--------------------+
| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) |       Range        |
+----------+-----------------+-----------+----------+----------+----------+----------+-------------+--------------------+
|    100.0 |       281774344 |      8392 |  33576.5 |  32687.0 |    29611 |   151701 |      3868.5 | NCCL:ncclAllGather |
+----------+-----------------+-----------+----------+----------+----------+----------+-------------+--------------------+

```
