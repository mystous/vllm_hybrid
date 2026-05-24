# SUB_071 — chat/code workload large-scale validation (500p × 8192 × 8192)

> **parent**: TSK_020 / workload generalization (post-plateau verification)
> **status**: 신설 (2026-05-24)
> **effort**: 4 cell × ~6–15 min ≈ 60 min wallclock
> **출처**: 사용자 지시 — Rec 1 medium (200p × 4096 × 4096) 측정에서 chat (+22%) / code (-30%) 확인된 뒤, large workload (SUB_044/047 의 측정 조건과 동일한 500p × 8192 × 8192) 에서도 검증

## 1. 배경 (왜 다시 측정하는가)

Rec 1 medium 측정 (2026-05-24, `eval/results/20260524_073538_rec1v2_*`) 의 결론:

| workload | vanilla | spec7+cap8 | speedup |
|---|---:|---:|---:|
| sonnet | 8,395.2 | 9,370.1 | +12% |
| chat | 2,113.6 | 2,577.1 | +22% |
| **code** | 7,889.1 | 5,505.6 | **−30%** (회귀) |

→ 그러나 SUB_047 best (10,956.5 tps, +134%) 는 **500p × 8192 × 8192** scale 에서 측정. medium (200p × 4096 × 4096) 와 large 의 차이가 워크로드별 speedup 형태에 영향을 미칠 수 있음:
- prompt 가 길어지면 ngram pool 도 커져 chat 의 acceptance 가 향상될 가능성
- 반대로 code 는 어휘 다양성이 커서 large 에서 더 큰 회귀 가능성
- batch 가 커지면 GPU saturate → spec 의 latency-bound 가정이 깨질 수 있음

본 SUB 는 "SUB_047 best 가 large workload 의 어떤 종류에서 generalize 되는가" 의 fact-finding 측정.

## 2. 측정 설계

### 2.1 cell (2 workload × 2 config = 4 cell)

| workload | config | env | spec |
|---|---|---|---:|
| chat | vanilla | CAP=1 / DIV=1 | 0 |
| chat | spec7+cap8 | CAP=8 / DIV=0 | 7 |
| code | vanilla | CAP=1 / DIV=1 | 0 |
| code | spec7+cap8 | CAP=8 / DIV=0 | 7 |

> sonnet 은 SUB_047 t3 canonical 3-run 으로 이미 확정 (avg 10,956.5 tps) — 본 sweep 에서 제외.

### 2.2 공통 parameter (SUB_047 / SUB_044 와 정합)

```
num_prompts             = 500
target_input_len        = 8192
max_tokens              = 8192
max_num_seqs            = 256
max_model_len           = 16384
gpu_memory_utilization  = 0.85
kv_cache_dtype          = fp8
max_num_batched_tokens  = 8192
prompt_lookup_min/max   = 2 / 5
seed                    = 0
```

### 2.3 prompts

`/tmp/run_workload_gen.py` 의 builder 재사용:
- `chat`: `<|system|>…<|user|>{sonnet excerpt}\n{question}<|assistant|>` 형태 (sonnet.txt 에서 derive)
- `code`: HumanEval-style Python function stub + comment padding

→ Rec 1 v2 (medium) 와 동일 builder, parameter 만 large 로 scale.

## 3. 가설

- **chat 가설**: prompt 가 길어진 만큼 ngram pool 도 커져 acceptance ↑ → medium 의 +22% 보다 large 에서 더 큰 speedup 가능 (sonnet 의 +134% 패턴에 근접 가능). 다만 chat 의 vocabulary 가 sonnet 보다 다양 → +50~100% 정도 예상.
- **code 가설**: code workload 는 어휘 패턴이 자연어 sonnet 와 달라 medium 에서 -30% 회귀. large 에서는 batch 가 커져 vanilla forward 가 더 saturate → spec 의 batch overhead 가 더 커질 가능성 → 회귀 폭이 medium 보다 클 수도 있음 (-30 ~ -50% 가능).
- **counter 가설**: 둘 다 large 에서는 vanilla 가 GPU bound 가 되어 spec 가 무력화 → 모두 ~0 ~ +20% 정도 small effect 만 남을 수도 있음.

→ 측정으로 확정.

## 4. 실행

- launcher: `/tmp/run_sub071_workload_large.sh` (run_workload_gen.py wrapper 재사용)
- 각 cell 결과: `eval/results/<TS>_sub071_<workload>_<config>/result.json` + `engine.log` + `engine.log.stdout`
- 측정 후 결과: `shadow_assists/features/IDE_006/TSK_020/measurements/sub071_workload_large_20260524/RESULTS.md`

### 4.1 진행 순서

`vanilla → spec` 교대 (workload 별로) 로 진행, 각 cell 사이에 vLLM worker process 종료 + 5초 sleep (Rec 1 v2 launcher 와 동일 패턴).

## 5. 판정 기준 (post-measurement)

| workload | speedup ≥ +50% | +10~50% | −10 ~ +10% | < −10% |
|---|---|---|---|---|
| 해석 | sonnet 와 유사하게 generalize | partial benefit | noise / inconclusive | regression (workload-aware gating 필요) |

→ 결과에 따라 다음 lever:
- chat positive: workload-aware gating 의 "spec ON" set 에 chat 포함
- code regression 재확인: production 권장 doc 에 "code workload 검출 시 spec OFF" 명시
- 양쪽 다 회귀: SUB_047 best 가 sonnet-shaped workload 한정임을 명문화

## 6. risk / 비고

- chat / code generator 가 sonnet 와 다르게 prompt token count 산정 — `target_input_len 8192` 가 실제 token 8192 가 아닐 수 있음 (chat 은 padding template 있음, code 는 docstring 위주). 단 동일 builder 의 vanilla vs spec 비교는 fair (둘 다 같은 prompt set 사용).
- 시간: vanilla 약 700–900 s, spec 약 350–500 s 예상. init 30 s × 4 + cleanup 5 s × 4 ≈ 총 50–70 분.
- code workload 의 `max_tokens=8192` 는 docstring 만 있는 prompt 에서 stop 안 걸리고 8192 까지 끝까지 생성 가능 → 회귀 폭이 medium 보다 커질 가능성. 이는 의도된 worst-case 측정.
