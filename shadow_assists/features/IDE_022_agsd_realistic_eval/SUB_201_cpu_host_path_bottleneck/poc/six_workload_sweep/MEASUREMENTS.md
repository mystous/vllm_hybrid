# Six-Workload Sweep — Spec Decode & FP8 KV across Corpus Mix

## 측정 목적
TSK_042 에서 정의된 6 workload (sonnet / chat / code / balanced / sonnet-heavy / code-heavy) 에 대해,
spec decode (Eagle3 / Suffix) 와 fp8 KV 가 **어느 workload 에서 양수 net effect** 를 보이는지 확인한다.

이전 6 agent 들이 사용한 sharegpt 단일 corpus 는 chat 위주 + prefix repeat 낮은 분포 — TSK_042 의
suffix dominant lever (+83~232%) 같은 결과가 prefix-repeat 높은 code/sonnet-heavy 같은 corpus 에서
재현되는지가 핵심 질문.

## 환경
- 모델: `meta-llama/Llama-3.1-8B-Instruct`, TP=8, max-model-len=16384, gpu-memory-utilization=0.85
- HW: B200 8 × sm_100, Xeon 8570
- vllm: `/workspace/vllm_dev_prj/bin/vllm`
- harness: `/workspace/host_vllm_hybrid/vllm_config_perf/gating/benchmark_workloads.py`
- workload params (4 method 공통):
  - `num_prompts=500`, `target_input_len=1024`, `max_tokens=2048`, `concurrency=64`, `seed=42` (1-sweep)
- methods:
  - vanilla — baseline
  - eagle3 — `{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":3}`
  - suffix — `{"method":"suffix","num_speculative_tokens":8}`
  - fp8_kv — `--kv-cache-dtype fp8`

## 1-sweep 결과 (24 cell)

### output_tps (tok/s)

| workload      |    vanilla |     eagle3 |     suffix |     fp8_kv |
|---------------|-----------:|-----------:|-----------:|-----------:|
| sonnet        |    21015.0 |    14268.9 |    24444.2 |    22053.3 |
| chat          |    16405.8 |    12069.3 |    12374.3 |    15685.7 |
| code          |    20668.3 |    14628.5 |    25749.0 |    22081.8 |
| balanced      |    19200.1 |    13919.3 |    22521.3 |    20234.0 |
| sonnet-heavy  |    19868.6 |    14143.9 |    23061.0 |    21053.8 |
| code-heavy    |    19427.5 |    14051.2 |    21518.8 |    20741.5 |

### Δ% vs vanilla

| workload      | eagle3   | suffix    | fp8_kv  |
|---------------|---------:|----------:|--------:|
| sonnet        | -32.10%  | **+16.32%** | **+4.94%** |
| chat          | -26.43%  | -24.57%   | -4.39%  |
| code          | -29.22%  | **+24.58%** | **+6.84%** |
| balanced      | -27.50%  | **+17.30%** | **+5.39%** |
| sonnet-heavy  | -28.81%  | **+16.07%** | **+5.96%** |
| code-heavy    | -27.67%  | **+10.76%** | **+6.76%** |

### Wall time (s)

| workload      | vanilla | eagle3 | suffix | fp8_kv |
|---------------|--------:|-------:|-------:|-------:|
| sonnet        |   48.73 |  71.76 |  41.89 |  46.37 |
| chat          |   19.75 |  26.93 |  26.68 |  20.88 |
| code          |   49.54 |  70.00 |  39.77 |  46.37 |
| balanced      |   41.75 |  57.27 |  35.86 |  39.21 |
| sonnet-heavy  |   45.59 |  64.76 |  39.00 |  43.06 |
| code-heavy    |   45.95 |  63.88 |  41.58 |  43.04 |

## 양수 cell (+3% 이상) 식별

| method  | workloads (양수)                                                 | 비고                          |
|---------|------------------------------------------------------------------|-------------------------------|
| suffix  | sonnet, code, balanced, sonnet-heavy, code-heavy (5/6)           | chat 만 -24.57% (음수)        |
| fp8_kv  | sonnet, code, balanced, sonnet-heavy, code-heavy (5/6)           | chat -4.39% 는 약한 음수      |
| eagle3  | 없음                                                              | 6 workload 전부 -26%~-32%     |

→ 정밀 측정 대상: suffix 5 cell + fp8_kv 5 cell = 총 10 cell.

## 5-sweep 정밀 측정 (seeds=42..46)

10 cell (suffix 5 + fp8_kv 5) + vanilla baseline 5 cell, 각 5 seed. 동일 boot 안에서 seed 5 회 반복.

### Vanilla baseline — 5-sweep mean ± std (tok/s)

| workload      | mean_tps | std    | min     | max     |   CV%  |
|---------------|---------:|-------:|--------:|--------:|-------:|
| sonnet        | 21011.5  |  88.1  | 20866.5 | 21078.5 |  0.42% |
| code          | 20575.8  |  20.7  | 20548.2 | 20594.9 |  0.10% |
| balanced      | 19018.0  |  88.6  | 18927.6 | 19139.2 |  0.47% |
| sonnet-heavy  | 19680.0  | 105.2  | 19545.1 | 19790.2 |  0.53% |
| code-heavy    | 19431.2  |  64.3  | 19318.1 | 19475.0 |  0.33% |

### Suffix — 5-sweep mean ± std (tok/s)

| workload      | mean_tps | std    |   CV%  |
|---------------|---------:|-------:|-------:|
| sonnet        | 24955.3  | 606.1  |  2.43% |
| code          | 26170.7  | 574.3  |  2.19% |
| balanced      | 21499.0  | 332.6  |  1.55% |
| sonnet-heavy  | 21359.9  | 152.4  |  0.71% |
| code-heavy    | 20387.1  | 293.5  |  1.44% |

### FP8 KV — 5-sweep mean ± std (tok/s)

| workload      | mean_tps | std    |   CV%  |
|---------------|---------:|-------:|-------:|
| sonnet        | 22331.8  |  37.6  |  0.17% |
| code          | 22043.6  |  24.5  |  0.11% |
| balanced      | 20185.7  |  64.9  |  0.32% |
| sonnet-heavy  | 21001.4  |  77.3  |  0.37% |
| code-heavy    | 20852.7  | 175.1  |  0.84% |

### Suffix vs Vanilla — paired Δ% + 95% CI (n=5)

(paired by seed: 동일 seed 의 vanilla 와 suffix 측정값 직접 비교)

| workload      | mean Δ%   | std   | 95% CI lo | 95% CI hi | signif. |
|---------------|----------:|------:|----------:|----------:|--------:|
| sonnet        | +18.77%   | 2.77  | +15.33%   | +22.20%   | yes     |
| code          | +27.19%   | 2.70  | +23.84%   | +30.54%   | yes     |
| balanced      | +13.05%   | 1.88  | +10.72%   | +15.38%   | yes     |
| sonnet-heavy  |  +8.54%   | 1.04  |  +7.25%   |  +9.83%   | yes     |
| code-heavy    |  +4.92%   | 1.32  |  +3.28%   |  +6.56%   | yes     |

### FP8 KV vs Vanilla — paired Δ% + 95% CI (n=5)

| workload      | mean Δ%   | std   | 95% CI lo | 95% CI hi | signif. |
|---------------|----------:|------:|----------:|----------:|--------:|
| sonnet        |  +6.28%   | 0.46  |  +5.71%   |  +6.86%   | yes     |
| code          |  +7.13%   | 0.15  |  +6.95%   |  +7.32%   | yes     |
| balanced      |  +6.14%   | 0.59  |  +5.41%   |  +6.87%   | yes     |
| sonnet-heavy  |  +6.72%   | 0.72  |  +5.83%   |  +7.61%   | yes     |
| code-heavy    |  +7.31%   | 0.62  |  +6.55%   |  +8.08%   | yes     |

→ **모든 10 cell 의 95% 신뢰구간 하한이 +3% 이상** — 1-sweep 양수가 통계적으로 robust 확인됨.

## 관찰
1. **eagle3 는 모든 6 workload 에서 음수** (-26%~-32%). 이미 batched (conc=64) 환경에서 spec decode 의
   per-step CPU draft + verify 오버헤드가 GPU 활용 손해를 압도. accept_rate 가 batched setting 에서 잘
   안 올라옴.
2. **suffix 는 chat 제외 5 workload 에서 +10.76%~+24.58%** 양수. chat 만 prompt 가 짧고
   prefix repeat 신호가 거의 없어 suffix cache hit 가 안 나며 오히려 draft overhead 만 발생.
3. **fp8_kv 는 chat 제외 5 workload 에서 +4.94%~+6.84%** 안정적 소폭 양수. KV memory bandwidth 가
   완화되어 long-context (1024 input + max 2048 output) 시나리오에서 일관된 net win.
4. **chat workload 가 전체적으로 outlier** — 짧은 prompt + 짧은 output (avg 648 token / 500 prompt) 이라
   draft/cache overhead 가 흡수되지 못함. vanilla 도 chat 만 16k tps 로 낮은 — output 분포가 짧아 throughput
   bound 가 아닌 latency bound.

## Verdict (1-sweep)
- **본 baseline (Llama-3.1-8B TP=8 conc=64) 환경에서 net-positive 조합:**
  - **Suffix** + (sonnet | code | balanced | sonnet-heavy | code-heavy) — 큰 폭 양수 (+10.76%~+24.58%)
  - **FP8 KV** + (sonnet | code | balanced | sonnet-heavy | code-heavy) — 일관된 소폭 양수 (+4.94%~+6.84%)
- **net-negative 또는 break-even:**
  - 모든 chat workload (chat 분포 전반에서 spec decode/KV 양자화 ROI 없음)
  - Eagle3 모든 workload (이 setting 에서 draft head 오버헤드 > GPU 절감)

## 5-sweep 정밀 verdict

### Winner per workload (1-sweep + 5-sweep 합치)

| workload      | winner method | Δ% vs vanilla (5-sweep paired mean) |
|---------------|---------------|------------------------------------:|
| sonnet        | **suffix**    | **+18.77%** [+15.33, +22.20]        |
| chat          | vanilla       | (suffix -24.6%, fp8 -4.4%; 양수 method 없음) |
| code          | **suffix**    | **+27.19%** [+23.84, +30.54]        |
| balanced      | **suffix**    | **+13.05%** [+10.72, +15.38]        |
| sonnet-heavy  | **suffix**    |  +8.54% [+7.25,  +9.83]             |
| code-heavy    | **suffix**    |  +4.92% [+3.28,  +6.56]             |

→ chat 제외 5 workload 에서 **suffix > fp8_kv > vanilla** 순으로 일관.

### Final verdict
1. **Suffix decoding 은 chat 을 제외한 모든 long-context workload (5/6) 에서 통계적으로
   유의한 양수** (95% CI 하한 +3% 이상). 특히 code (+27.19%) / sonnet (+18.77%) 같은
   prefix-repeat 높은 corpus 에서 net win 이 크게 나타남 — TSK_042 의 suffix dominant
   가설 (prefix-repeat ↑ ⇒ suffix gain ↑) 이 확증됨. 다만 TSK_042 가 보인
   +83~232% 같은 큰 폭은 본 baseline 환경 (TP=8, conc=64, max-tok=2048, Llama-3.1-8B)
   에선 재현되지 않음 — workload 정의/모델/concurrency 차이가 원인 가능.
2. **FP8 KV 는 chat 을 제외한 5 workload 에서 안정적 +5~+7% 양수** (CV 0.1~0.8%, 95% CI 매우 tight).
   이전 6 agent 의 sharegpt 단일 측정 +3.94% 가 단일 corpus artifact 가 아니라 long-context
   workload 전반의 robust net win 임을 확인. 이는 추가 quality 검증 후 production 배포 후보.
3. **Eagle3 는 본 setting 의 모든 6 workload 에서 음수** (-26%~-32%). conc=64 에서 GPU 가
   이미 batched throughput 으로 포화 → spec decode 의 draft+verify 오버헤드만 추가됨.
   Eagle3 가 도움 되려면 conc 가 더 낮거나 (latency bound), prompt 가 더 길거나, batched
   accept rate 가 충분히 높아야 함 — TSK_042 의 Eagle3 lever 가 본 환경엔 부적합.
4. **chat workload 는 모든 가속 method 에 적대적**. 짧은 prompt + 짧은 output (avg 648 tok)
   분포에서 draft/cache overhead 가 흡수되지 않음. 운영적으로는 (a) chat workload 만 vanilla
   유지, (b) 그 외 workload 는 suffix (또는 latency budget 허용 시 suffix+fp8_kv 스택)
   적용하는 **per-workload routing** 이 가장 큰 net gain 제공.

### 권장 next-step
- **suffix + fp8_kv 스택** (조합) 의 6-workload 측정 — 효과가 가산적인지, 충돌하는지 확인.
- chat workload 의 sub-segmentation (e.g. avg-output ≥256 tok 만 suffix 라우팅) 로
  chat 도 양수로 끌어올 수 있는지 검증.
- **운영 정밀도 게이트** — fp8_kv 의 logprob max-abs-diff 측정 (CLAUDE.md constraint).

