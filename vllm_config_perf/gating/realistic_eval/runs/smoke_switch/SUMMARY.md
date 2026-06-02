# TSK_042 (B) 처리량 — 실 trace, conc=32, max_tokens=8192, TP=8(7B만 4)

corpus 별 격리 측정 + mix 별도. 값 = aggregate output_tps.

## humaneval
| model | suffix | vanilla | best | suffix vs van |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 2,360 | 2,186 | suffix | +8% |

## mix
| model | suffix | vanilla | best | suffix vs van |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 1,553 | 2,095 | vanilla | -26% |

## condition-level oracle (model × condition → best method)
| model | sharegpt | swebench | humaneval | mbpp | wildchat | lmsys | mix |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | — | — | suffix | — | — | — | vanilla |


## util (mix 조건, gpu%/cpu%)
| model | suffix | vanilla |
|---|---|---|
| Qwen2.5-7B-Instruct | 55.2/2.7 | — |
