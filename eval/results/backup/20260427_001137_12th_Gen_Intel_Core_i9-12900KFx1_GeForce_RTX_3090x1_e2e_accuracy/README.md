# TST_003 e2e accuracy run

- timestamp:           2026-04-27 00:12:20
- baseline env:        eval/envs/vllm_original.env
- split_on env:        eval/envs/ide006_cold_kv_split_on.env
- model:               Qwen/Qwen2.5-7B-Instruct
- tensor_parallel:     1
- max_model_len:       4096
- max_tokens:          64
- logprobs_k:          10
- num_prompts:         8
- baseline duration:   23.8s
- split_on duration:   18.0s

## Tolerance
- MAX_DIVERGING_TOKENS: 10
- ATOL_LOGPROB:         0.5
- RTOL_PPL:             0.1

## Verdict
- D-i  (token divergence):  PASS (worst = 0 tokens) — informational, does not affect overall
- D-ii (logprob / PPL):     PASS (worst max_abs=0.0945, worst ppl_rel=0.0012) — **binding** (verdict_overall = verdict_d_ii)
- overall:                  PASS
- suspicious_no_cold_path: False

Verdict policy: CLAUDE.md Constraint 운영 해석 — token-level bit-exact 가 아니라 분포 수준 유사성. D-ii (logprob/PPL atol/rtol) 이 binding 이고 D-i (token argmax 일치) 는 informational. BF16 산술 비결합성 + greedy argmax cascade 로 D-i 가 깨질 수 있어도 D-ii 가 통과하면 정확도 게이트 PASS. 자세한 근거는 IDE_006 README §8·§9 / PLN_001 §4.1 참조.

## Files
- baseline.json     — per-prompt outputs from baseline (default forward)
- split_on.json     — per-prompt outputs from feature-on (TSK_002 Phase 4c)
- comparison.json   — full per-prompt + aggregate D-i / D-ii verdict
