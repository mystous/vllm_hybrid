# 20260428_062533_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_value_region_64k

- timestamp: 20260428_062533
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    8649d114426aa664fbe4b3f5509ec44c4f844810

## scope — IDE_006 진정한 가치 영역 (≥64K, KV pool 초과)
- baseline: ide006_cold_tier_only_value_64k.env  (cold-tier ON, IDE_006 OFF)
- split_on: ide006_cold_kv_split_on_value_64k.env (cold-tier ON, IDE_006 ON)
- max-prompts: 20
- max-tokens:  8
- input-len:   65536  (env file 에서 가져옴)
- max-model-len: 65536
- KV size estimate: 20 × 65536 × 320KB / 8 worker ≈ 52 GB/worker (KV pool ~46 GiB 초과)
- VLLM_COLD_KV_FALLBACK_DEADLINE_MS: 100

## exit code
- e2e RC: 1

## comparison.json
```
{
  "verdict_d_i": true,
  "verdict_d_ii": false,
  "verdict_overall": false,
  "worst_diverging_tokens": 6,
  "worst_max_abs_logprob": 2.7505131405196153,
  "worst_ppl_relative_diff": 0.4343961899159595,
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
      "max_abs_logprob": 0.00027428101748228073,
      "ppl_relative_diff": 1.259064206325861e-05,
      "d_i_pass": true,
      "d_ii_pass": true
    },
    {
      "prompt_index": 1,
      "len_baseline": 8,
      "len_split_on": 8,
      "n_diverging_tokens": 0,
      "max_abs_logprob": 0.057068467140197754,
```

## bench wall time
  [baseline] batched generate complete: 20 prompts in 50.7s (avg 2.54s/prompt)
[baseline] done in 257.8s
  [split_on] batched generate complete: 20 prompts in 179.5s (avg 8.97s/prompt)
[split_on] done in 359.4s

## firing
- TSK_011 cold-fallback fired: 0
0
- TSK_004 cold-path fired:    40
