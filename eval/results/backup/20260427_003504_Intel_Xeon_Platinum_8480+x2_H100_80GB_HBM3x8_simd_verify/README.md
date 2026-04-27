# 20260427_003504_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_simd_verify

- timestamp: 20260427_003504
- hw_tag:    Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    8d7abc914e43a716cd3a6c8f43f964d8f0a5b5a0
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45

## components
- pytest TST_004 cross-check (TSK_003 §4.2a portable vs AVX-512 + §4.2b portable vs AMX)
    └─ 40 + 40 = 80 케이스. dev 에서는 cpuid 게이트로 자동 skipif
- eval/run_e2e_accuracy.py --split-on-only (cold-path dispatcher 발화 + D-ii binding 확인)
    └─ baseline LLM 로딩 생략. 8 prompts × 14336 input × 32 output 의 짧은 verifier

본 결과 디렉토리는 prod 검증 후 dev 에서 분석하기 위해 push 되며,
tst004_junit.xml 의 PASS/FAIL 패턴 + e2e_quick.log 의 [IDE_006 diag ...]
발화 카운터를 통해 SIMD kernel 의 numerical correctness 와 dispatcher
wiring 안정성을 분석한다. 정식 throughput / accuracy 비교는 풀
run_prod_smoke.sh (--push) 가 별도로 수행.
