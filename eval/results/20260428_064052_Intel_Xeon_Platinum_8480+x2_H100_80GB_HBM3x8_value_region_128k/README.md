# 20260428_064052_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_value_region_128k
- timestamp: 20260428_064052
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    8649d114426aa664fbe4b3f5509ec44c4f844810

## scope — IDE_006 극한 가치 영역 (≥128K)
- baseline: ide006_cold_tier_only_value_128k.env  (cold-tier ON, IDE_006 OFF)
- split_on: ide006_cold_kv_split_on_value_128k.env (cold-tier ON, IDE_006 ON)
- max-prompts: 10 / max-tokens: 8 / input-len: 131072
- KV size estimate: ~52 GB/worker (pool 46 GiB 1.13× 초과)

## exit code: 1

## comparison.json
```
{
  "verdict_d_i": true,
  "verdict_d_ii": false,
  "verdict_overall": false,
  "worst_diverging_tokens": 2,
  "worst_max_abs_logprob": 2.050290942195261,
  "worst_ppl_relative_diff": 0.3811181986424047,
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
      "max_abs_logprob": 0.005587197840213776,
      "ppl_relative_diff": 0.0003593735658342579,
      "d_i_pass": true,
      "d_ii_pass": true
    },
    {
      "prompt_index": 1,
      "len_baseline": 8,
      "len_split_on": 8,
      "n_diverging_tokens": 0,
      "max_abs_logprob": 0.04885751008987427,
```

## bench wall time
  [baseline] batched generate complete: 10 prompts in 63.0s (avg 6.30s/prompt)
[baseline] done in 264.3s
  [split_on] batched generate complete: 10 prompts in 164.1s (avg 16.41s/prompt)
[split_on] done in 341.5s

## firing
- TSK_011 cold-fallback fired: 0
0
- TSK_004 cold-path fired:    40

## OOM / 에러 trace
(none)
