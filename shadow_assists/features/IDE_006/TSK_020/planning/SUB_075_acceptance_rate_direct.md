# SUB_075 — I003 acceptance rate 직접 측정 (R/K 분리)

> **parent**: TSK_020 / idea I003
> **status**: 활성 (2026-05-24 신설)
> **effort**: 30-60 분
> **idea**: [`../idea/IDE_011_acceptance_rate_direct_measure.md`](../idea/IDE_011_acceptance_rate_direct_measure.md)

## 1. 진행 절차

1. wrapper 확장 — `/tmp/run_workload_gen.py` 의 LLM kwargs `disable_log_stats=False`. (또는 `--enable-spec-stats` flag).
2. launcher `/tmp/run_sub075_acceptance.sh` (3 cell): sonnet/chat/code × ngram_spec7_cap8.
3. 측정 실행 — stdout log 영역 `SpecDecoding metrics: Mean acceptance length: ...` 출력 capture.
4. log 파싱: mean_acceptance_length → α = (mean - 1) / 7 → K_linear = 1 + α×7 / K_exact = (1−α^8)/(1−α).
5. R 분리: R_actual = wall_ratio × K_actual.
6. 분석 doc §3.3 표 갱신 (실측 K), §3.4 closed-form 표 갱신 (실측 α).

## 2. cell matrix (3 cell, 신규)

| workload | config | env | 측정값 |
|---|---|---|---|
| sonnet | ngram_spec7_cap8 (SUB_047 best) | CAP=8/DIV=0 | mean_accept_len, α, K, R |
| chat | 같음 | 같음 | 같음 |
| code | 같음 | 같음 | 같음 |

vanilla baseline 은 측정 안 함 (spec OFF 에는 spec metric 없음). 본 SUB 의 wall 측정값은 SUB_047/071 의 ngram 측정과 cross-check 가능.

## 3. 예측 (분석 doc §3.4 의 prediction)

| workload | 예측 α | linear K = 1+7α | Leviathan K_exact | R_actual (= wall_ratio × K) |
|---|---:|---:|---:|---:|
| sonnet | 0.5~0.6 | 4.5~5.2 | 1.99~2.46 | 0.83~1.08 (wall_ratio 0.419) |
| chat | 0.10~0.15 | 1.7~2.05 | 1.10~1.16 | 0.82~0.87 (wall_ratio 0.749) |
| code | 0~0.03 | 1~1.21 | 1.00~1.03 | 1.29~1.33 (wall_ratio 1.292) |

→ 실측 결과로 본 doc 의 R=1.30 가정이 정확한지, workload 별 R 차이가 있는지 확인.

## 4. 산출물

- `eval/results/<TS>_sub075_<workload>_acceptance/` (3 디렉토리, stdout log + result.json)
- `measurements/sub075_acceptance_<TS>/RESULTS.md` (mean_accept_len + α + K + R 표)
- 분석 doc §3.3 / §3.4 / §11 갱신
- idea I003 §7 결과 채움
