# 20260426_045803_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_prod_smoke

- timestamp: 20260426_045803
- hw_tag:    Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    cd88e96115b96535e70e0e5d7416f93dbbdeb4a6
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45

## components
- pytest TST_001 (TSK_001 dev kernel: stages A, B(i), C — reproduces 87 dev testcases)
- pytest TST_004 (TSK_003 prod SIMD: B(ii) portable vs AVX-512 + B(iii) portable vs AMX)
    └─ skipped via skipif marker if the TSK_003 §4.2a/§4.2b kernels are not built
- eval/run.sh envs/vllm_original_long_ctx.env (split-off long-context baseline)
- eval/run.sh envs/ide006_cold_kv_long_ctx.env (cold-tier KV offload)

Result subdirs (run.sh): see eval/results/<TS>_<HW_TAG>_<MODEL>/

## exit codes
- pytest:               1
- scenario baseline:    0
- scenario cold_kv:     0
