# 20260428_211501_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_tsk009_long_output_B

- timestamp: 20260428_211501
- mode:      B
- split-on env: /workspace/vllm_hybrid/eval/results/20260428_211501_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_tsk009_long_output_B/split_on.env
- input_len: 1024
- output_len: 14336
- num_prompts: 100
- max-prompts: 100
- max-tokens:  14336    ← 16 → 256 으로 decode 길게

## exit code
- e2e RC: 0

## events
- reload: 0
0
- overlap hidden: 0
0
- overlap wait:   0
0

## generate timing
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 320.4s (avg 3.20s/prompt)
[baseline] done in 417.6s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 353.1s (avg 3.53s/prompt)
[split_on] done in 481.6s
```
