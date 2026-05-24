# SUB_079 — SUB_078 의 sonnet/chat 확장 측정 (Qwen 0.5B/1.5B small model)

> **parent**: TSK_020 / idea IDE_014
> **status**: 활성 (2026-05-24 신설)
> **effort**: 8 cell × ~30s-2min = ~10-15 min
> **idea**: [`../idea/IDE_014_issue_16258_repro.md`](../idea/IDE_014_issue_16258_repro.md)

## 1. 동기

SUB_078 가 small model (Qwen 0.5B/1.5B) 의 **code workload 만** 측정. 결과: 두 model 모두 ngram 영역 −59~−62% 회귀.

본 SUB 는 같은 small model + same setup 으로 **sonnet/chat 까지 확장** — small model 영역 ngram 회귀가:
- (a) workload-universal (R ≫ K 가 모든 workload 에서 발생) 인지
- (b) 또는 large model 의 SUB_071 (sonnet +134% / chat +37.5%) 처럼 workload 별로 다른지

검증.

## 2. cell matrix (8 cell)

| model | workload | config |
|---|---|---|
| Qwen2.5-0.5B | sonnet | vanilla |
| Qwen2.5-0.5B | sonnet | ngram spec=5, lookup_max=2 |
| Qwen2.5-0.5B | chat | vanilla |
| Qwen2.5-0.5B | chat | ngram |
| Qwen2.5-1.5B | sonnet | vanilla |
| Qwen2.5-1.5B | sonnet | ngram |
| Qwen2.5-1.5B | chat | vanilla |
| Qwen2.5-1.5B | chat | ngram |

setup: SUB_078 와 동일 (TP=1, 50p × 1024in × 512max, fp8 KV).

## 3. 가설

1. **workload-universal regression**: 모든 workload (sonnet/chat/code) 에서 small model 영역 ngram 회귀. R ≫ K → workload 어떤 것이든 회귀.
2. **workload-specific (large model 동일 패턴)**: sonnet 회귀 가장 작음 또는 net positive, chat 중간, code 가장 큼 (R 영역 model-scale 의존, K 는 workload 의존).

가설 1 더 가능성 큼 (small model 의 R 가 K_sonnet ≈ 3-4 도 넘을 정도로 크다면).

## 4. 산출물

- `eval/results/<TS>_sub079_<model>_<workload>_<config>/` (8 디렉토리)
- `measurements/sub079_small_model_full_<TS>/RESULTS.md`
- idea IDE_014 §결과 보강
- 분석 doc §11.3 추가 cross-validation

## 5. raw 자료 위치 후보

- launcher: `/tmp/run_sub079_qwen_sonnet_chat.sh`
- wrapper: `/tmp/run_sub078_wrap.py` (재사용, --workload sonnet|chat 지원)
