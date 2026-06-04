# DeepSeek-R1 671B suffix nsys stats (재현용, v2 — --wait=all 성공)

## cuda_gpu_kern_sum (top 30)
```

NOTICE: Existing SQLite export found: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.sqlite
        It is assumed file was previously exported from: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.nsys-rep
        Consider using --force-export=true if needed.

Processing [/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

+----------+-----------------+-----------+----------+----------+----------+----------+-------------+------------------------------------------------------------------------------------------------------+
| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) |                                                 Name                                                 |
+----------+-----------------+-----------+----------+----------+----------+----------+-------------+------------------------------------------------------------------------------------------------------+
|     42.3 |     11596070409 |    469676 |  24689.5 |  11040.0 |     1536 |   146848 |     33959.1 | void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n… |
|     25.4 |      6970246136 |    169011 |  41241.4 |  35264.0 |    17056 |   106784 |     18253.5 | kernel_cutlass_kernel_vllmvllm_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__t… |
|      7.1 |      1947012851 |     50847 |  38291.6 |  29248.0 |     5984 |    83968 |     25084.6 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
|      6.6 |      1810578390 |     48725 |  37159.1 |  26688.0 |     7840 |    74400 |     21856.8 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
|      5.8 |      1581661699 |    169313 |   9341.6 |   6048.0 |     2016 |    27776 |      8202.8 | void per_token_group_quant_8bit_packed_kernel<c10::BFloat16, __nv_fp8_e4m3>(const T1 *, void *, uns… |
|      4.5 |      1228505864 |    131035 |   9375.4 |   7296.0 |     2912 |    20352 |      5355.2 | void vllm::gather_and_maybe_dequant_cache<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0… |
|      1.6 |       437195975 |     26082 |  16762.4 |  16800.0 |    14976 |    18400 |       543.5 | fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta128PagedKvDenseP32MultiCtasKvVarSeqQ16Kv128… |
|      1.4 |       375197283 |    131039 |   2863.2 |   2816.0 |     2272 |     5280 |       261.9 | void vllm::merge_attn_states_kernel<__nv_bfloat16, __nv_bfloat16, (unsigned int)128, (bool)0>(T2 *,… |
|      0.5 |       136332294 |     30683 |   4443.3 |   4448.0 |     4000 |     5120 |       109.5 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
|      0.5 |       135189722 |      6649 |  20332.3 |  20320.0 |    19168 |    23136 |       486.7 | fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta256PagedKvDenseP32MultiCtasKvVarSeqQ16Kv128… |
|      0.5 |       131656611 |     36694 |   3588.0 |   3584.0 |     3168 |     4288 |       129.3 | nvjet_tst_64x8_64x16_2x1_v_bz_NNT                                                                    |
|      0.5 |       125943933 |     36695 |   3432.2 |   3424.0 |     2976 |     4064 |       126.3 | nvjet_tst_64x8_64x16_4x1_v_bz_TNT                                                                    |
|      0.4 |       106505724 |     37976 |   2804.6 |   2752.0 |     2304 |     4320 |       223.4 | void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)2>,… |
|      0.4 |        99987396 |     37973 |   2633.1 |   2624.0 |     2176 |     5408 |       188.8 | void vllm::concat_and_cache_mla_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(c… |
|      0.3 |        93765371 |     10633 |   8818.3 |   8928.0 |     5888 |    12832 |      1826.6 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
|      0.3 |        78378657 |      6726 |  11653.1 |  11328.0 |     6592 |    17889 |      3290.2 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
|      0.3 |        76652072 |       123 | 623187.6 | 659392.0 |    79167 |  5175167 |    448137.6 | void <unnamed>::multimem_all_reduce_kernel<c10::BFloat16, (int)16>(T1 *, unsigned long, unsigned in… |
|      0.2 |        61363745 |      2440 |  25149.1 |  25152.0 |    23232 |    28096 |       642.5 | fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512PagedKvDenseP32MultiCtasKvVarSeqQ16Kv128StaticSwaps… |
|      0.2 |        51052311 |      6466 |   7895.5 |   8768.0 |     4992 |     9440 |      1337.3 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
|      0.2 |        46702618 |       621 |  75205.5 |  68384.0 |    51072 |  1131616 |     58239.0 | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)                       |
|      0.2 |        42648371 |      1242 |  34338.5 |  31744.0 |    30240 |    46560 |      5048.3 | void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<… |
|      0.1 |        27984502 |      5704 |   4906.1 |   4896.0 |     4639 |     5376 |        93.7 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
|      0.1 |        27481086 |      1220 |  22525.5 |  22336.0 |    21216 |    24128 |       632.3 | fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512PagedKvDenseP32MultiCtasKvCgaVarSeqQ16Kv128StaticSw… |
|      0.1 |        26381130 |      1863 |  14160.6 |  10688.0 |     9856 |    36032 |      6450.6 | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator… |
|      0.1 |        21664218 |      3721 |   5822.1 |   6080.0 |     4480 |     6816 |       485.6 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
|      0.1 |        17253344 |       915 |  18856.1 |  18848.0 |    18048 |    19712 |       274.5 | fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta256PagedKvDenseP32MultiCtasKvCgaVarSeqQ16Kv… |
|      0.0 |        13412062 |      1661 |   8074.7 |   8384.0 |     6624 |     9024 |       642.1 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
|      0.0 |        11277727 |        58 | 194443.6 | 197312.0 |   187232 |   200864 |      4825.3 | bmm_E4m3_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_cga1x1x1_16dp256b_rM_BN_transOut_dsFp8_sc… |
|      0.0 |        10368805 |       671 |  15452.8 |  15456.0 |    14912 |    16096 |       193.3 | fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta128PagedKvDenseP32MultiCtasKvCgaVarSeqQ16Kv… |
|      0.0 |         9372477 |      1464 |   6402.0 |   6400.0 |     5760 |     6816 |       154.7 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
|      0.0 |         8058081 |      1864 |   4323.0 |   3712.0 |     2240 |    11296 |      2061.8 | void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, lon… |
|      0.0 |         7529344 |        58 | 129816.3 | 130000.0 |   124352 |   133248 |      2009.4 | bmm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_cga1x1x1_16dp256b_rM_BN_transOut_dsFp… |
|      0.0 |         7139584 |      1647 |   4334.9 |   4352.0 |     3808 |     4864 |       121.9 | void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)… |
```

## cuda_api_sum (top 20)
```

NOTICE: Existing SQLite export found: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.sqlite
        It is assumed file was previously exported from: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.nsys-rep
        Consider using --force-export=true if needed.

Processing [/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

+----------+-----------------+-----------+----------+----------+----------+----------+-------------+--------------------------------------------+
| Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) |                    Name                    |
+----------+-----------------+-----------+----------+----------+----------+----------+-------------+--------------------------------------------+
|     34.9 |      3521583822 |     14529 | 242383.1 |   3588.0 |     1792 | 29650169 |   1273368.0 | cudaMemcpyAsync                            |
|     34.8 |      3517364707 |    992730 |   3543.1 |   3248.0 |     2398 |  5132514 |      6702.0 | cudaLaunchKernel                           |
|     11.8 |      1195165258 |    285904 |   4180.3 |   3977.0 |     2117 |  6062610 |     12241.2 | cuLaunchKernelEx                           |
|      7.6 |       771724994 |     38533 |  20027.6 |  17406.0 |    13649 |  1145716 |     14382.4 | cudaGraphLaunch_v10000                     |
|      7.4 |       742697711 |    169243 |   4388.4 |   4131.0 |     2720 |    29113 |       936.8 | cudaLaunchKernelExC_v11060                 |
|      1.4 |       143928791 |   1034953 |    139.1 |    119.0 |       67 |    64264 |       121.0 | cuTensorMapEncodeTiled                     |
|      0.9 |        90497778 |    169014 |    535.4 |    440.0 |      289 |    20123 |       237.4 | cuKernelGetFunction                        |
|      0.4 |        38575179 |     52817 |    730.4 |    812.0 |      216 |    10816 |       318.9 | cudaStreamIsCapturing_v10000               |
|      0.2 |        17307940 |    153267 |    112.9 |     92.0 |       37 |    11199 |        89.3 | cuKernelGetAttribute                       |
|      0.2 |        16555039 |    169013 |     98.0 |     86.0 |       40 |     7823 |        75.9 | cuFuncGetName                              |
|      0.1 |         9156648 |     11834 |    773.8 |    764.0 |      599 |     7962 |       207.9 | cuOccupancyMaxActiveClusters               |
|      0.1 |         8224315 |      2144 |   3836.0 |   3074.0 |     1116 |    16676 |      2284.0 | cudaEventQuery                             |
|      0.1 |         7193748 |     58631 |    122.7 |    113.0 |       97 |     7226 |        70.9 | cudaGetDevice                              |
|      0.1 |         5993879 |      1863 |   3217.3 |   2136.0 |     1457 |    58327 |      2219.3 | cudaEventRecord                            |
|      0.1 |         5696084 |      1242 |   4586.2 |   4558.0 |     3790 |    11329 |       415.7 | cudaStreamSynchronize                      |
|      0.0 |         2977783 |      1242 |   2397.6 |   2281.5 |     1809 |    11257 |       588.6 | cudaEventRecordWithFlags_v11010            |
|      0.0 |         1686393 |       426 |   3958.7 |   3617.5 |     3006 |    37018 |      1856.7 | cuLaunchKernel                             |
|      0.0 |         1470093 |      1242 |   1183.6 |   1612.0 |      377 |    10774 |       769.7 | cudaStreamWaitEvent                        |
```

## nvtx_pushpop_sum
```

NOTICE: Existing SQLite export found: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.sqlite
        It is assumed file was previously exported from: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.nsys-rep
        Consider using --force-export=true if needed.

Processing [/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/nvtx_pushpop_sum.py]... 

 ** NVTX Push/Pop Range Summary (nvtx_pushpop_sum):

+----------+-----------------+-----------+------------+----------+----------+-------------+-------------+--------------------+
| Time (%) | Total Time (ns) | Instances |  Avg (ns)  | Med (ns) | Min (ns) |  Max (ns)   | StdDev (ns) |       Range        |
+----------+-----------------+-----------+------------+----------+----------+-------------+-------------+--------------------+
|    100.0 |     49650381851 |      3968 | 12512697.0 |  32261.5 |    29474 | 49518207703 | 786101804.5 | NCCL:ncclAllGather |
+----------+-----------------+-----------+------------+----------+----------+-------------+-------------+--------------------+

```

## osrt_sum (host syscall, wait 류 dominant)
```

NOTICE: Existing SQLite export found: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.sqlite
        It is assumed file was previously exported from: /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.nsys-rep
        Consider using --force-export=true if needed.

Processing [/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/r1_671b_suffix_v2.sqlite] with [/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/reports/osrt_sum.py]... 

 ** OS Runtime Summary (osrt_sum):

+----------+-----------------+-----------+---------------+---------------+-------------+-------------+-------------+------------------------+
| Time (%) | Total Time (ns) | Num Calls |   Avg (ns)    |   Med (ns)    |  Min (ns)   |  Max (ns)   | StdDev (ns) |          Name          |
+----------+-----------------+-----------+---------------+---------------+-------------+-------------+-------------+------------------------+
|     31.1 |   6532794715988 |     94192 |    69356152.5 |    10116558.0 |        1001 |  5369722072 | 166589093.9 | poll                   |
|     30.8 |   6471463064411 |     61454 |   105305807.0 |   100086943.5 |        1001 |  1000306590 |  99988662.3 | pthread_cond_timedwait |
|     26.7 |   5618912027020 |    509809 |    11021602.3 |    10078847.0 |        1000 |  5370409373 |  21621999.1 | epoll_wait             |
|     10.1 |   2130023510261 |       213 | 10000110376.8 | 10000094373.0 | 10000020221 | 10000286306 |     51449.4 | sem_clockwait          |
|      0.9 |    179264056678 |      2514 |    71306307.4 |    40484598.5 |        1001 |   773888042 | 140317171.3 | epoll_pwait            |
|      0.3 |     73337676009 |       512 |   143237648.5 |    90078519.5 |    60262229 |  5367995267 | 438570896.9 | sem_wait               |
|      0.0 |       570962466 |      2380 |      239900.2 |       23348.5 |        7280 |     5714467 |    596901.5 | ioctl                  |
|      0.0 |       296389965 |     12782 |       23188.1 |        5111.5 |        1005 |      489918 |     73770.4 | recv                   |
|      0.0 |       187093767 |        17 |    11005515.7 |     2272156.0 |        6229 |    22834861 |  10611533.4 | fgets                  |
|      0.0 |        69333450 |     19061 |        3637.5 |        2202.0 |        1000 |       38916 |      4925.7 | write                  |
|      0.0 |        53643830 |     13024 |        4118.8 |        1860.0 |        1024 |       32161 |      5137.9 | send                   |
|      0.0 |        46489446 |       203 |      229012.0 |      268932.0 |        1225 |     1664340 |    174236.8 | pthread_rwlock_wrlock  |
|      0.0 |        44051058 |      3968 |       11101.6 |       10442.0 |        8013 |     1054811 |     16634.4 | munmap                 |
```
