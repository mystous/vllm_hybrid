# SUB_078 — I006 vLLM Issue #16258 reproduction (small model cross-validation)

> **parent**: TSK_020 / idea I006
> **status**: 활성 (2026-05-24 신설)
> **effort**: 1-1.5 시간 (vLLM 영역 small model 로드 + 4 cell 측정)
> **idea**: [`../idea/IDE_014_issue_16258_repro.md`](../idea/IDE_014_issue_16258_repro.md)

## 1. 진행 절차

1. issue #16258 본문 확인 (`https://github.com/vllm-project/vllm/issues/16258`) — workload / hardware / exact config 재확인.
2. wrapper 확장 또는 새 wrapper `/tmp/run_sub078_small_model.py` (model 인자 + spec config 가변).
3. launcher `/tmp/run_sub078_repro.sh` (4 cell): {opt-125m, starcoder2-3b} × {vanilla, ngram_spec5_lookup2}.
4. 측정 실행 (각 cell ~5-15분).
5. 결과 doc.

## 2. cell matrix (4 cell)

| model | config | spec_config |
|---|---|---|
| opt-125m | vanilla | None |
| opt-125m | ngram | `{"method":"ngram", "num_speculative_tokens":5, "prompt_lookup_max":2}` |
| starcoder2-3b | vanilla | None |
| starcoder2-3b | ngram | 같음 |

issue #16258 의 hardware = 2× L4. 본 fork 환경 = H100×8 (단 본 SUB 는 TP=1 single GPU 로 충분). model loading 시 `tensor_parallel_size=1` 사용.

## 3. workload

issue #16258 의 explicit workload 명시 없음. 추정:
- opt-125m: 일반 chat / wikitext prompt (50~500 prompt, 길이 512~1024)
- starcoder2-3b: HumanEval-like code prompt 또는 본 SUB_071 의 code builder 재사용

→ 본 SUB 는 **본 fork 의 code workload builder (SUB_071 code prompts)** 재사용 — issue 의 정확한 workload reproduction 보다 본 doc 의 "high acceptance ≠ net win" 명제 cross-validation 이 main 목적.

## 4. 가설

- opt-125m + ngram on chat-like workload: acceptance moderate, throughput 회귀 가능성 — issue #16258 의 "Pareto worse" 명제 재현.
- starcoder2-3b + ngram on code workload: 본 fork SUB_071 의 code 회귀 (−23.2%) 와 같은 패턴 예상 (different model, same workload class).

## 5. 산출물

- `eval/results/<TS>_sub078_<model>_<config>/` (4 디렉토리)
- `measurements/sub078_repro_<TS>/RESULTS.md` (4 cell + acceptance + comparison with issue #16258 보고값)
- 분석 doc §11.3 차별점 cross-validation 추가
- idea I006 §7 결과
