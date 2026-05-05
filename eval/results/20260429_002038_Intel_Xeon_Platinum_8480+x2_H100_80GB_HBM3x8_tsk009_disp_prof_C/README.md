# 20260429_002038_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_tsk009_disp_prof_C
- mode: C
- input_len: 15360, output_len: 1024, num_prompts: 100

## generate timing
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 68.1s (avg 0.68s/prompt)
[baseline] done in 164.8s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "enable_cpu_partial_attention": true, "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 83.8s (avg 0.84s/prompt)
[split_on] done in 219.3s
```

## dispatcher profile last summary
```
(Worker_TP2 pid=2897505) [IDE_006/TSK_009 profile pid=2897505 tag=dispatcher] #69/100 disp_calls=5200 disp_total_us_per_call=7854.3 lookup_us=6.9 layout_us=5.4 meta_us=17.8 hcattn_us=7821.2 pfa_calls=8640 pfa_total_us_per_call=90.6
(Worker_TP2 pid=2897505) [IDE_006/TSK_009 profile pid=2897505 tag=pagedfa] #43/100 disp_calls=0 disp_total_us_per_call=0.0 lookup_us=0.0 layout_us=0.0 meta_us=0.0 hcattn_us=0.0 pfa_calls=8600 pfa_total_us_per_call=90.5
```
