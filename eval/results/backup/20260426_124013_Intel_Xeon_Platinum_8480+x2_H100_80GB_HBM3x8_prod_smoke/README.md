# 20260426_124013_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_prod_smoke

- timestamp: 20260426_124013
- hw_tag:    Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    1c3ce3f080fb77c12ddc02b84268a47d0baf8b10
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45

## components
- pytest TST_001 (TSK_001 dev kernel: stages A, B(i), C — reproduces 87 dev testcases)
- pytest TST_004 (TSK_003 prod SIMD: B(ii) portable vs AVX-512 + B(iii) portable vs AMX)
    └─ skipped via skipif marker if the TSK_003 §4.2a/§4.2b kernels are not built
- eval/run.sh envs/vllm_original_long_ctx.env (split-off baseline, Llama-3.3-70B + TP=8)
- eval/run.sh envs/ide006_cold_kv_long_ctx.env (OffloadingConnector only)
- eval/run.sh envs/ide006_cold_kv_split_on_long_ctx.env (full Cold-KV CPU partial attention — TSK_002 Phase 4c)

Result subdirs (run.sh): see eval/results/<TS>_<HW_TAG>_<MODEL>/

## exit codes
- pytest:                       0
- scenario baseline:            0
- scenario cold_kv (offload):   0
- scenario cold_kv split-on:    0
