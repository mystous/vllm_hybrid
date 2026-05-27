# SUB_044 — Speculative Decoding (ngram) — ★★★ 첫 net-positive 성과 (2026-05-23 KST)

> **parent**: TSK_020
> **measurement**: HEAD `0d7dc0334`, 500p × 8192, gmu=0.85, vLLM ngram speculative decoding (draft model 불필요)

> **★ SUB_073/I001 framing 정정 (2026-05-24)**: 본 SUB 의 "+130.3% vs vanilla" 는 **vLLM 영역 코드 변경 0 — `speculative_config={"method":"ngram",...}` 활성화만**으로 달성된 효과. 즉 vLLM 의 built-in ngram speculative decoding 을 본 환경에서 켰을 때의 fact 이며, fork patch contribution 이 아님. fork patch (cap=8 + div_tp=0) 의 추가 기여는 SUB_047 에서 +1.65% (10,778.6 → 10,956.5).

---

## 1. 측정 결과

| test | num_spec | tps | wall (s) | CPU% | GPU% | vs vanilla 4679.8 | acceptance ratio (추정) |
|---|---:|---:|---:|---:|---:|---:|---:|
| t1 | 3 | 9573.6 | 421.2 | 3.97 | 45.8 | **+104.6% (2.05×)** | ~50% |
| t2 | 5 | 10166.6 | 396.8 | 2.86 | 27.4 | **+117.3% (2.17×)** | ~55% |
| **★ t3** | **7** | **10778.6** | **372.9** | **2.46** | **21.9** | **★ +130.3% (2.30×) ⭐ WINNER** | ~60% |
| t4 | 10 | NO_RESULT | crash=3 | 1.60 | 4.4 | **OOM** ✗ | — |

## 2. ★★★ 첫 성과 출현 — 본 모든 시도 중 vanilla 보다 빠른 유일한 lever

SUB_032 ~ SUB_043 (12 SUB) 의 모든 시도가 noise / negative 였던 환경에서, **SUB_044 ngram spec decode 가 본 모든 lever 중 처음으로 vanilla 보다 빠름**.

### 2.1 가속 효과

- **vanilla baseline 4680 tps** → **spec best 10778 tps (+130.3%, 2.30× faster)**
- wall 875s → 373s (-57% wall reduction)
- 500p × 8192 = 4.1M token 을 373s 에 처리 (vanilla 가 4.1M 을 875s 에)

### 2.2 monotonic increase pattern

num_spec 증가 → throughput 증가 (sweet spot num_spec=7, num_spec=10 은 OOM crash):

```
num_spec=3 → 9573.6 tps (+104.6%)
num_spec=5 → 10166.6 tps (+117.3%)  +6.2% vs spec=3
num_spec=7 → 10778.6 tps (+130.3%)  +6.0% vs spec=5  ⭐ best
num_spec=10 → crash (OOM, 3 workers died)
```

→ num_spec=7 이 throughput 영역 최적, num_spec=10 은 KV memory 또는 spec accept slot 부족으로 OOM.

## 3. 측정 fact 분해

### 3.1 GPU util 가 vanilla 73% → spec=7 22% — 의미

spec decode 의 본질:
- 한 forward step 에 multiple tokens 동시 verify (num_spec=7 → 1+7=8 tokens / step in best case)
- per-step compute = 1 token forward + 7 spec verification = 약 1.2× compute (vs vanilla 1 token / step)
- 단 effective tokens / step = 5-6 (accept rate 60% 가정) → effective throughput 6× per step
- net: **GPU 가 일하는 step 수가 줄어듦 → GPU util 평균 ↓, 단 throughput ↑**

### 3.2 CPU util 가 vanilla 4.66% → spec=7 2.46% — 더 idle

spec decode 가 model forward 외의 overhead (scheduling, sampling, ZMQ) 도 줄임 → CPU 도 더 idle. **CLAUDE.md 목표 "CPU 활용" 영역은 여전히 미달** (2.46% 만).

### 3.3 ngram 영역 의 적합성

- sonnet (Shakespeare) prompt 영역 의 어휘 반복 (the/and/of/I 등 + couplet rhyme + meter) 이 n-gram lookup hit rate 높임
- prompt_lookup_min=2, max=4/5 영역에서 2-5 token n-gram 매칭
- random text 영역 보다 sonnet 영역 의 spec acceptance rate 자연스럽게 ↑

## 4. vs vanilla baseline

| Approach | tps | wall | CPU% | GPU% | vs vanilla |
|---|---:|---:|---:|---:|---:|
| **vanilla baseline** | 4680 | 875 | 4.66 | 73.5 | — |
| **★ vanilla + spec=7** | **10778** | **373** | **2.46** | **21.9** | **★ +130% (2.30×)** ⭐ |

## 5. CLAUDE.md `# Objective` 재검증

| 목표 | spec=7 결과 | 평가 |
|---|---|---|
| "CPU 활용률 **극도로** 끌어올리기" | CPU 2.46% | ❌ 여전히 미달 (spec decode 가 GPU-only 가속, CPU 더 idle) |
| "CPU **Idle 허락 안 함**" | CPU idle 97.5% | ❌ |
| "GPU 포함 **서버 전체 throughput 향상**" | **+130% inference throughput** | **✓ 충족** |

→ **CPU 활용 목표 는 미달, 단 서버 throughput 목표 는 달성**. 본 lever 가 GPU-side optimization 이라 CPU 영역 활용 안 함.

## 6. 다음 path

### 6.1 spec decode 후 추가 가속 시도

| 후보 | 이유 |
|---|---|
| (a) **num_spec sweep refinement** (4, 6, 8 추가 + 9) | spec=7 < 8 < 9 < 10 영역 중 OOM 직전 sweet spot |
| (b) **prompt_lookup_max sweep** (3, 4, 5, 6, 7) | n-gram lookup window 가 acceptance rate 영향 |
| (c) **prompt_lookup_min sweep** (1, 2, 3) | min=1 영역에서 hit rate 높이고 reject 줄이기 |
| (d) **spec=7 + 다른 model** (Qwen, smaller model) | model 영역 의 ngram pattern 영향 |
| (e) **spec=7 + 워크로드 변경** (decode-heavy, prefill-heavy) | 워크로드 영역에서 spec 영향 |
| (f) **draft model spec decode** (eagle, medusa) | small draft model + verify 가 ngram 보다 더 빠를 가능성 |

### 6.2 CLAUDE.md 목표 (CPU 활용) 와 정합

본 결과는 GPU throughput 목표 ✓, CPU 활용 ✗. CPU 영역 까지 활용하려면:
- **별도 CPU workload (BG embedding / 별도 CPU LLM instance 등) 의 cluster-level co-serving** — spec decode 가 GPU 빨리 끝내고 CPU 가 BG 처리 (SUB_045 / SUB_049)

## 7. raw 자료

| 항목 | 위치 |
|---|---|
| SUMMARY.tsv | `eval/results/20260523_005314_sub044_spec_decode/SUMMARY.tsv` |
| per-test 결과 (3 dirs + t4 fail) | `eval/results/20260523_005314_sub044_spec_decode/t1_spec3 ~ t4_spec10/` |
| launcher | `/tmp/run_sub044_spec_decode.sh` |
| wrapper (vLLM LLM with speculative_config) | `/tmp/run_spec_decode.py` (MODEL_PATH = "meta-llama/Llama-3.3-70B-Instruct") |
| stdout log | `/tmp/sub044_spec_v2.log` |

## 8. 코드 영향

- `/tmp/run_spec_decode.py` — vLLM `LLM(speculative_config={"method":"ngram",...})` wrapper. 본 코드 베이스 의 `vllm/` source 변경 X.
- 본 lever 는 vLLM 의 built-in feature 활용 — **vanilla vLLM 만 으로 적용 가능**.
- production 적용 권고: `speculative_config={"method":"ngram", "num_speculative_tokens":7, "prompt_lookup_max":5, "prompt_lookup_min":2}`.
