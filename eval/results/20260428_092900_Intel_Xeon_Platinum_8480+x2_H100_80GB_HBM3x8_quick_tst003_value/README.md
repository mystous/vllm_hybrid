# 20260428_092900_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_quick_tst003_value

- timestamp: 20260428_092900
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    666b675bac9fe6b58fc1f0ceb8d367f689e16a93
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45

## scope — IDE_006 가치 영역 (long-context, KV pool 초과)
- baseline: ide006_cold_tier_only_long_ctx.env  (cold-tier ON, IDE_006 OFF — PCIe 비용 동등화)
- split_on: ide006_cold_kv_split_on_long_ctx.env (cold-tier ON, IDE_006 ON)
- max-prompts: 100  (KV pool ~46 GiB worker per 초과 — cold-tier 강제 발화)
- max-tokens:  8   (decode 단축, KV pool 초과는 prefill 에서 보장)
- logprobs:    1
- VLLM_COLD_KV_FALLBACK_DEADLINE_MS: 100

## 의도
- 두 회차 모두 PCIe 비용 동등화 → wall-time 차이가 IDE_006 *순 increment*
- 두 회차 모두 cold-tier 동일 source → lp 차이가 IDE_006 attention kernel *순 발산*

## exit code
- e2e RC: 1

## comparison.json (jq)
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
FileNotFoundError: [Errno 2] No such file or directory: '/workspace/vllm_hybrid/eval/results/20260428_092900_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_quick_tst003_value/e2e_artifacts/comparison.json'
