# 20260428_094320_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_quick_tst003_value

- timestamp: 20260428_094320
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
{
  "verdict_d_i": true,
  "verdict_d_ii": false,
  "verdict_overall": false,
  "worst_diverging_tokens": 6,
  "worst_max_abs_logprob": 3.653306961062299,
  "worst_ppl_relative_diff": 0.40455702458287557,
  "suspicious_no_cold_path": false,
  "tolerances": {
    "max_diverging_tokens": 10,
    "atol_logprob": 0.5,
    "rtol_ppl": 0.1
  },
  "per_prompt": [
    {
      "prompt_index": 0,
      "len_baseline": 8,
      "len_split_on": 8,
      "n_diverging_tokens": 0,
      "max_abs_logprob": 7.152266334742308e-07,
      "ppl_relative_diff": 1.3410310896625743e-07,
      "d_i_pass": true,
      "d_ii_pass": true
    },
    {
      "prompt_index": 1,
      "len_baseline": 8,
      "len_split_on": 8,
      "n_diverging_tokens": 0,
      "max_abs_logprob": 9.77407762547955e-06,
```

## bench wall time
  [baseline] batched generate complete: 100 prompts in 43.5s (avg 0.44s/prompt)
[baseline] done in 184.4s
  [split_on] batched generate complete: 100 prompts in 144.6s (avg 1.45s/prompt)
[split_on] done in 320.4s

## fallback firing
- TSK_011 cold-fallback 발동 횟수: 0
0
- TSK_004 cold-path 발동 횟수:    40
