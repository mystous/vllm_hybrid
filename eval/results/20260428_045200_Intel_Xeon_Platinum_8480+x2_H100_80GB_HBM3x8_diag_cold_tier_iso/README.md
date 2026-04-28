# 20260428_045200_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_diag_cold_tier_iso

- timestamp: 20260428_045200
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    ec6cd7de226557c0ca3cbc36d70dabcf26b046b3
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45

## scope
- baseline: vllm_original_long_ctx.env       (cold-tier 비활성)
- split_on: ide006_cold_tier_only_long_ctx.env  (cold-tier 활성, IDE_006 비활성)
- max-prompts: 30
- max-tokens:  16
- logprobs:    1

## 진단 결정 분기
- worst_lp ≈ 3.43 → cold-tier 자체가 발산 source → TSK_012 진행
- worst_lp ≪ 1   → cold-tier 는 OK → TSK_012 design 재검토

## exit code
- e2e RC: 1

## comparison.json (jq)
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
FileNotFoundError: [Errno 2] No such file or directory: '/workspace/vllm_hybrid/eval/results/20260428_045200_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_diag_cold_tier_iso/e2e_artifacts/comparison.json'
