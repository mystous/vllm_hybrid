# SUB_074 — I002 SuffixDecoding 측정 (sonnet/chat/code × method=suffix)

> **parent**: TSK_020 / idea I002
> **status**: 활성 (2026-05-24 신설)
> **effort**: 1.5-2 시간 (패키지 설치 + 3 cell 측정 + 분석)
> **idea**: [`../idea/IDE_010_suffix_decoding_measurement.md`](../idea/IDE_010_suffix_decoding_measurement.md)

## 1. 진행 절차

1. `arctic-inference` 패키지 설치 (`.venv/bin/pip install arctic-inference`).
2. wrapper 확장 — `/tmp/run_workload_gen.py` 에 `--spec-method` (ngram|suffix) 옵션 추가, suffix 시 `method=suffix, num_speculative_tokens=32` 사용.
3. launcher `/tmp/run_sub074_suffix.sh` (3 cell): sonnet/chat/code × suffix_spec32.
4. 측정 (각 cell ~5-15분), background 실행.
5. 결과 doc `measurements/sub074_suffix_<TS>/RESULTS.md` 작성.
6. 분석 doc §10.4 / §11 갱신.

## 2. cell matrix (3 cell, 신규)

| workload | config | env | spec_method |
|---|---|---|---|
| sonnet | suffix_spec32 | (no NGRAM env 필요) | suffix |
| chat | suffix_spec32 | 같음 | suffix |
| code | suffix_spec32 | 같음 | suffix |

기존 vanilla / ngram 측정 (SUB_044/047/071) 과 fair comparison 위해 동일 prompt set + 동일 scale:

```
num_prompts            = 500
target_input_len       = 8192
max_tokens             = 8192
max_num_seqs           = 256
gpu_memory_utilization = 0.85
kv_cache_dtype         = fp8
max_num_batched_tokens = 8192
max_model_len          = 16384
seed                   = 0
```

## 3. 가설

- code workload: ngram −23.2% → suffix 가 prompt+generation 양쪽 pool 활용 → **net positive 또는 회귀 완화** 기대.
- sonnet/chat: ngram 와 비슷 또는 약간 향상 (adaptive num_spec 효과).

## 4. 산출물

- `eval/results/<TS>_sub074_<workload>_suffix/` (3 디렉토리)
- `measurements/sub074_suffix_<TS>/RESULTS.md`
- 분석 doc §10.4 / §11 갱신
- idea I002 §7 결과 채움
