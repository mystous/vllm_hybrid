# TST_003 e2e accuracy run

- timestamp:           2026-04-28 13:57:55
- baseline env:        /workspace/vllm_hybrid/eval/envs/vllm_original_long_ctx.env
- split_on env:        /workspace/vllm_hybrid/eval/envs/ide006_cold_kv_split_on_long_ctx.env
- model:               meta-llama/Llama-3.3-70B-Instruct
- tensor_parallel:     8
- max_model_len:       16384
- max_tokens:          16
- logprobs_k:          0
- num_prompts:         100
- baseline duration:   142.7s
- split_on duration:   378.1s

## Tolerance
- MAX_DIVERGING_TOKENS: 10
- ATOL_LOGPROB:         0.5
- RTOL_PPL:             0.1

## Verdict
- D-i  (token divergence):  FAIL (worst = 14 tokens) — informational, does not affect overall
- D-ii (logprob / PPL):     PASS (worst max_abs=0.0000, worst ppl_rel=0.0000) — **binding** (verdict_overall = verdict_d_ii)
- overall:                  PASS
- suspicious_no_cold_path: False

Verdict policy: CLAUDE.md Constraint 운영 해석 — token-level bit-exact 가 아니라 분포 수준 유사성. D-ii (logprob/PPL atol/rtol) 이 binding 이고 D-i (token argmax 일치) 는 informational. BF16 산술 비결합성 + greedy argmax cascade 로 D-i 가 깨질 수 있어도 D-ii 가 통과하면 정확도 게이트 PASS. 자세한 근거는 IDE_006 README §8·§9 / PLN_001 §4.1 참조.

## Files
- baseline.json     — per-prompt outputs from baseline (default forward)
- split_on.json     — per-prompt outputs from feature-on (TSK_002 Phase 4c)
- comparison.json   — full per-prompt + aggregate D-i / D-ii verdict
