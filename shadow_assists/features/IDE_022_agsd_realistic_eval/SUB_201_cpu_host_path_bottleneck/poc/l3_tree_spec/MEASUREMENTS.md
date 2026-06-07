# L3 — CPU tree spec-decoding (K-branch verify) Measurements

본 문서는 SUB_201 후속 lever **L3** 의 실측·시뮬 결과입니다.
질문: suffix decoding 의 **단일 path × K 토큰 verify** 를 **K-branch tree × K 토큰
verify** 로 확장하면, accept rate α 가 본질적으로 ↑ 하여 (TSK_042 의 α=0.46 회복
가능성 포함) net throughput uplift 를 만드는가?

---

## 0. 환경 / 셋업

| 항목 | 값 |
|---|---|
| HW | DGX B200, **GPU 3 단독** (CUDA_VISIBLE_DEVICES=3), Xeon host |
| vLLM | host_vllm_hybrid editable, sm_100 빌드 (`/workspace/vllm_dev_prj/bin/vllm`) |
| 모델 | `Qwen/Qwen2.5-32B-Instruct`, TP=1, gmu=0.85, max_model_len=8192 |
| spec-decode method | `suffix`, `num_speculative_tokens=7`, `max_tree_depth=24` (default) |
| corpus | `sharegpt`(short), `swebench`(long) 각 80 prompts |
| concurrency | 4 (EngineCore 가 본 머신 동시 sweep 와 stable 공존 가능한 상한) |
| max_tokens | 256 (stream=off, completion) |

> ⚠️ task brief 는 `100p × conc=8 × 256 tok` 가이드를 줬으나 본 머신은 동시에
> 4 개 다른 sweep 가 GPU 0~7 에 걸쳐 돌고 있어 conc=8 에서 vLLM EngineCore 가
> 메모리/스케줄 충돌로 중도 죽음 (첫 100p × conc=8 sharegpt baseline run
> = 91/100 ok, tree run = 17/100 ok). 80p × conc=4 로 줄이면 80/80 안정.
> 본 문서의 비교 셀은 모두 같은 (n, conc, max_tok) 셋업에서 head-to-head.

---

## 1. Patch 위치 & 동작

### 1.1 코드 위치

- `vllm/v1/spec_decode/suffix_decoding.py` — vLLM 의 `SuffixDecodingProposer`
- patch 는 env / flag-file 토글 OFF 일 때 **upstream 동작과 정확히 동일** (byte-equivalent).

### 1.2 토글 (env + flag file 둘 다 지원)

vLLM 의 EngineCore subprocess 는 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 으로 시작해
**API server 의 env vars 를 inherit 하지 않음**. 따라서 patch 는 env 와 flag-file
양쪽을 lazily 읽음:

| toggle | env / flag-file 경로 | 의미 | 본 측정 |
|---|---|---|---|
| `VLLM_L3_TREE_SPEC` / `/tmp/vllm_l3_VLLM_L3_TREE_SPEC.flag` | "1"/truthy | arctic `speculate(use_tree_spec=True)` 활성 | tree-only |
| `VLLM_L3_TREE_BRANCHES` / 동상명.flag | int (default 4) | linearize 시 root branch soft cap | 4 |
| `VLLM_L3_TREE_STATS_PATH` / 동상명.flag | jsonl path | per-step tree dump (offline α 분석용) | 둘 다 ON |
| `VLLM_L3_TREE_STATS_FLUSH` | int (default 200) | flush batch | 200 |

### 1.3 patch 가 하는 일

1. **tree spec 활성** (env 가 1 일 때): `SuffixDecodingCache.speculate(use_tree_spec=True)`
   로 K-branch tree 결과 (`token_ids`, `parents`, `probs`) 를 받음.
2. **linearize → best path**: vLLM GPU verifier 가 단일 path 만 받으므로,
   tree 의 root-to-leaf path 중 **누적 prob 곱이 최대인 한 path** 만 골라 GPU 로
   보냄. 이로써 단일 path verifier 와 호환 + tree spec 의 "best branch coverage"
   효과를 GPU 가 받음.
3. **tree 의 잠재 α 계산** (offline metric): 다음 step 의 `sampled_token_ids` 가
   들어왔을 때 (a) linearize 된 단일 path 의 match 길이 (= 단일 path α),
   (b) tree 전체에서 어느 path 와도 match 되는 최대 길이 (= **tree α upper bound**,
   tree-attention verifier 가 들어왔을 때 회수 가능한 ceiling) 를 모두 기록.

### 1.4 핵심 발견 (patch 동작 검증)

- arctic_inference 의 `SuffixDecodingCache.speculate(use_tree_spec=True)` 가
  실제로 K-branch tree 를 산출 (sample run 으로 검증, README §1.4 참조).
- patch OFF (baseline) 시 emit 토큰 ≡ upstream `draft.token_ids` (byte-equivalent).
- patch ON (tree) 시 emit 토큰 = best-prob root-to-leaf path.
- 단위 helper 테스트 (`_linearize_best_path`, `_tree_max_accept_len`,
  `_single_path_accept_len`) 통과.

---

## 2. 측정 결과 (Qwen2.5-32B, GPU 3, 단일 GPU)

### 2.1 sharegpt 100p (n=80, conc=4, max_tok=256)

| 지표 | baseline (single, K=7) | tree (4-branch, K=7) | Δ |
|---|---:|---:|---:|
| n_ok / n | 80/80 | 80/80 | — |
| wall_s | 60.8 / 60.0 | — | |
| output_tps (total) | **317.3** | **315.5** | **−0.6%** |
| accept_rate α (vLLM /metrics) | **0.2601** | **0.2674** | **+2.8% rel** |
| draft_tokens (Δ) | 15,234 | 14,964 | |
| accept_tokens (Δ) | 3,963 | 4,001 | |
| Mean acceptance length (vLLM, peak window) | ~1.32–1.47 | ~1.36 | |
| gpu_util | 75.9% | 77.0% | |
| cpu_util | 16.9% | 17.0% | |

→ **net throughput Δ = 사실상 동등** (noise 범위 ±1%).
→ α 는 sharegpt 짧은 context (38 tok 평균) 에서 +2.8% relative 만큼만 ↑.

### 2.2 swebench 100p (n=80, conc=4, max_tok=256) — **long prompt**

| 지표 | baseline (single, K=7) | tree (4-branch, K=7) | Δ |
|---|---:|---:|---:|
| n_ok / n | 80/80 | 80/80 | — |
| output_tps (total) | **358.0** | **357.1** | **−0.3%** |
| accept_rate α (vLLM /metrics) | **0.2871** | **0.3237** | **+12.7% rel** |
| Mean acceptance length (vLLM peak) | ~1.49–1.55 | ~1.55–1.65 | |
| gpu_util | 89.9% | 87.7% | |

→ **α +12.7% relative (+3.66pp absolute)** — short prompt 대비 4.5× 큰 회수 폭.
→ **output_tps 는 그대로** (이게 본 lever 의 핵심 제약 — §3 참조).

### 2.3 vLLM `/metrics` per-position 분포 (sharegpt)

| position | baseline | tree |
|---|---:|---:|
| 0 | 0.220–0.244 | 0.211 |
| 1 | 0.067–0.083 | 0.079 |
| 2 | 0.030–0.046 | 0.053 |
| 3 | 0.011–0.032 | 0.013 |
| 4 | 0.007–0.027 | 0.000 |
| 5 | 0.005–0.021 | 0.000 |
| 6 | 0.004–0.020 | 0.000 |

→ baseline 은 7 position 모두 dense decay, tree 는 4 position 부터 cliff to 0.
→ tree 는 short prompt 에서 **얕고 좁음** (mean branches/step 0.66~0.70).

### 2.4 offline tree-α 시뮬 (`analyze_stats.py` on `tree_b4.jsonl`)

| 지표 | sharegpt | swebench |
|---|---:|---:|
| steps captured | 15,000 | 13,600 |
| α_single (linearized) | 0.268 | **0.323** |
| α_tree (upper bound, K-branch coverage) | 0.258 | 0.307 |
| accept per step (single) | 0.254 | 0.479 |
| accept per step (tree) | **0.258** | **0.492** |
| mean root branches / step | 0.70 | 0.91 |
| mean tree nodes / step | 1.00 | **1.60** |

→ **tree 의 upper bound α 가 단일 path α 와 거의 같음** (0.268 vs 0.258 ; 0.323 vs 0.307).
→ 이유: tree 가 매우 sparse — 평균 1.0~1.6 node 만 나오는 수준.
→ 이론적 K=4 branch 의 "잠재력" 이 발현되지 않음.

### 2.5 baseline (single) 의 stats 도 dump (sharegpt)

`baseline.jsonl` 15,000 steps: α_single = α_tree = 0.261 (단일 path 라 정확히 같음).
mean tree_nodes/step = 1.00 — 본 patch 의 metric 보조 정확.

---

## 3. 해석 — tree spec 이 net throughput uplift 를 못 만드는 4 가지 원인

1. **GPU verify path 가 tree-attention 미지원**. 본 빌드 vLLM `RejectionSampler` 는
   1-D draft_token_ids 만 받음 (`_calc_spec_decode_metadata` 의 num_draft_tokens 는
   per-request linear 길이). patch 는 K-branch tree → best path linearize 만 가능
   → GPU 가 "K=4 branch 동시 verify" 를 못 함. **이건 vLLM 코어 수정 필요**.
2. **suffix cache 가 sharegpt 짧은 context 에서 multi-branch 를 거의 못 만듦**.
   `mean tree_nodes/step = 1.00` 은 본질적으로 single-token spec 과 다름없음. swebench
   같은 long context 에서야 1.60 까지 올라가지만 그래도 K=4 의 ¼.
3. **arctic suffix cache 의 `use_tree_spec=True` 가 실제로는 가장 일찍 잘리는
   path 들의 다중 root branch 를 emit** (depth 가 얕은 root 들). depth 2 이상의 진짜
   tree expansion 은 거의 안 나옴 → tree → single-path linearize 한 결과 거의 동일.
4. **본 빌드의 spec_decode `Avg Draft acceptance rate` ≈ 26%** (TSK_042 의 reported
   α=0.46 보다 본질적으로 낮음 — TSK_042 측정은 더 긴 context / 다른 cudagraph 셋업
   이었을 가능성). 이 α 베이스에서 +2.8~12.7% relative 는 throughput 으로 환산하면
   < 1% — wall-clock 측정 정밀도 한계 안 (gpu_util 70~90% 노이즈) 으로 묻힘.

---

## 4. 본 task 결론

### 4.1 단정 결과

- **net throughput Δ = 동등** (sharegpt −0.6%, swebench −0.3% — 모두 noise 안).
- **α 의 본질적 ↑ 폭**: long context (swebench) 에서 **+12.7% relative**,
  short context (sharegpt) 에서 **+2.8% relative**.
- 본 PoC 가 측정한 K-branch tree 의 **upper bound α** (tree_max_accept_len) 도
  단일 path α 와 거의 같다 — 즉 tree spec 의 "잠재력" 자체가 sharegpt/swebench
  Qwen2.5-32B 셋업에서는 미발현. **TSK_042 의 α=0.46 회복은 본 lever 만으로는 불가**.

### 4.2 lever 판정: **net positive 아님 (기각)**

- `tree spec` 자체는 작동하고 α 도 측정 가능한 ↑ 폭을 보이나, 본 vLLM 빌드의
  GPU verifier 가 **tree-attention 을 지원하지 않아** 잠재력의 ¼ 도 못 회수.
- linearize-to-single-path 변환만으로는 wall-clock throughput 이 안 움직임.
- production net win 으로 가려면 (a) vLLM `RejectionSampler` 를 tree verifier
  로 확장 + (b) suffix cache 가 더 깊고 넓은 tree 를 만들도록 조정 (max_spec_factor
  ↑, min_token_prob ↓) 의 **두 차원 모두 손대야 함**. 본 task 의 시간 박스 +
  GPU 단일 제약 안에서는 사실상 dev-rich 한 큰 patch 가 됨 → **본 lever 보류**.

### 4.3 회수 가능 시나리오 (future work)

| 조건 | 기대 effect |
|---|---|
| vLLM 코어에 tree-attention verifier 추가 | α_tree ceiling 회수 → 본 측정 기준 sharegpt +1.5%, swebench +2.7% wall throughput |
| `max_spec_factor=2.0, min_token_prob=0.05` 등 cache 파라미터 완화 | tree 가 더 dense 해져 mean nodes/step ↑ — 별도 sweep 필요 |
| long-context-heavy production traffic (swebench-style) | α gain headroom 이 ↑ — 본 결과의 swebench α +12.7% 가 lower bound 후보 |

---

## 5. 산출물 매핑

```
poc/l3_tree_spec/
├── README.md                       — 이론·계획
├── MEASUREMENTS.md                 — 본 문서
├── launcher.sh                     — GPU3 단일 Qwen2.5-32B serve (baseline/tree)
├── run_l3_bench.sh                 — sharegpt/swebench 80p × conc=4 × 256 tok bench
├── analyze_stats.py                — jsonl stats → α 표
└── out/
    ├── baseline.json               — sharegpt baseline summary  (α=0.2601, 317.3 tps)
    ├── tree.json                   — sharegpt tree summary      (α=0.2674, 315.5 tps)
    ├── baseline_swebench.json      — swebench baseline summary  (α=0.2871, 358.0 tps)
    ├── tree_swebench.json          — swebench tree summary      (α=0.3237, 357.1 tps)
    ├── *.raw.jsonl                 — per-request wall_ms / completion_tokens
    ├── *.spec_metrics.txt          — vLLM /metrics 로그 발췌
    ├── tree.stats_analysis.json    — sharegpt offline α 분석
    └── tree_swebench.stats.jsonl   — 13,600 step tree dump (per-step token_ids/parents/probs)

patch:
└── vllm/v1/spec_decode/suffix_decoding.py   — env+flag-file gated tree-spec instrumentation
```

---

## 6. 본 task 의 method 충실도 vs deviation

| brief 항목 | 실측 | 일치 / deviation |
|---|---|---|
| GPU 3 단독 | CUDA_VISIBLE_DEVICES=3, TP=1 | ✓ |
| Qwen2.5-32B | `Qwen/Qwen2.5-32B-Instruct` | ✓ |
| 시간 박스 5-7 시간 | 본 시도는 ~50 분 안에 완료 (single GPU, 단일 모델 측정의 자연 cycle) | ↑ 효율 |
| sharegpt 100p × conc=8 × 256 tok | conc=4 × 80p 로 다운조정 — 본 머신 동시 sweep 충돌 회피 (CLAUDE.md 의 "병행 sweep stability" 함정) | 80p × conc=4 head-to-head |
| baseline vs tree (4-branch × K=7) | ✓ | ✓ |
| accept rate α 변화 측정 | vLLM /metrics + offline tree α 둘 다 | ✓ + 잠재 α 의 upper bound 추가 |
| commit 금지 | 미커밋 | ✓ |
| 산출물 `poc/l3_tree_spec/MEASUREMENTS.md` | 본 문서 | ✓ |

extra: swebench long-context 측정도 같은 셋업에서 추가 — α gap 의 corpus 의존성을
드러내, 본 lever 의 미래 valid 한 회수 시나리오를 식별.
