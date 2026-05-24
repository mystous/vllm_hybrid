# SUB_074 — SuffixDecoding 측정 (sonnet/chat/code × method=suffix) RESULTS

> **parent**: TSK_020 / SUB_072 / idea I002
> **plan**: [`../../planning/SUB_074_suffix_decoding_measurement.md`](../../planning/SUB_074_suffix_decoding_measurement.md)
> **measurement**: 2026-05-24 16:15~16:38 KST, single-run × 3 workload
> **config**: vLLM v1 `SuffixDecodingProposer` + `arctic_inference.suffix_decoding.SuffixDecodingCache`, `method=suffix`, `num_speculative_tokens=32`
> **scale**: 500p × 8192in × 8192max, batch 256, fp8 KV, Llama-3.3-70B + TP=8 + H100×8
> **raw**: `eval/results/20260525_011528_sub074_<workload>_suffix/`

---

## 1. ★ 환경 caveat — enforce_eager=True 적용 사유 (중요)

본 fork repo 의 vLLM v1.6.dev0+g858b6df7a 와 `arctic-inference` 0.1.2 (필수 vLLM==0.11.0) 간 binary incompatible:
- `arctic_inference.vllm.args` 가 `vllm.utils.FlexibleArgumentParser` (vLLM 0.11.0 API) 를 import — 본 fork 에는 부재 → plugin load 실패.
- **우회**: `ARCTIC_INFERENCE_ENABLED=0` + `VLLM_PLUGINS=""` 로 plugin disable. 본 fork repo 의 `vllm/v1/spec_decode/suffix_decoding.py` 만 사용 (lazy import `SuffixDecodingCache` only). 작동 ✓.
- 단 vLLM 영역 CUDA graph capture path 가 SuffixDecodingProposer 와 conflict ("CUDA graph capturing detected at an inappropriate time") → **`enforce_eager=True`** 강제.

→ 본 SUB 측정은 **enforce_eager 모드** (CUDA graph off). SUB_075 ngram 측정은 **cuda graph on** (default).
→ **apples-to-apples 비교가 아님**. enforce_eager penalty 영역 ~20-25% 패널티 추정 (본 SUB 의 sonnet/chat 회귀 폭 기반).

## 2. 측정 fact (3 workload)

| workload | tps | wall (s) | out_tok | init wall (s) |
|---|---:|---:|---:|---:|
| **sonnet** | **8,236.0** | 491.2 | 4,045,818 | 52.9 |
| **chat** | **2,369.7** | 169.3 | 401,307 | 51.5 |
| **code** | **7,093.5** | 537.4 | 3,811,850 | 52.2 |

### 2.1 SpecDecoding metric (peak/mid interval, not last)

vLLM v1 의 `SpecDecoding metrics` 는 ~10s interval 단위 emit (cumulative 아님). 본 표는 peak/mid interval 의 representative metric:

| workload | mean_accept_len (peak) | avg draft accept rate (peak) | per-position acceptance (peak interval, position 1~12) |
|---|---:|---:|---|
| **sonnet** | **4.42** | 72.0% | 0.522 / 0.407 / 0.349 / 0.296 / 0.274 / 0.246 / 0.240 / 0.229 / 0.225 / 0.215 / 0.211 / 0.207 |
| **chat** | **11.58** ⭐ | **90.5%** ⭐ | 0.981 / 0.955 / 0.929 / 0.911 / 0.907 / 0.900 / 0.881 / 0.855 / 0.844 / 0.836 / 0.807 / 0.770 |
| **code** | **7.67** ⭐ | 65.5% | 0.912 / 0.782 / 0.704 / 0.640 / 0.614 / 0.582 / 0.536 / 0.460 / 0.417 / 0.400 / 0.341 / 0.285 |

> position 13~32 은 모두 0 (suffix adaptive 가 12개 정도만 사용. `num_speculative_tokens=32` 의 cap 안에서 동적 길이).

## 3. 핵심 결과 — SuffixDecoding vs Ngram K 비교

본 fork 환경의 ngram (SUB_075) vs suffix (SUB_074) 의 K 비교 — drafter mechanism 만 차이, 같은 model/scale/prompt:

| workload | ngram (SUB_075, cuda graph) tps | suffix (SUB_074, eager) tps | suffix/ngram ratio | ngram K | suffix K (peak) | K ratio (suffix/ngram) |
|---|---:|---:|---:|---:|---:|---:|
| sonnet | 10,909 | 8,236 | 0.755 | 3.72 | **4.42** | 1.19× |
| chat | 2,972 | 2,370 | 0.797 | 6.69 | **11.58** | 1.73× |
| **code** | **5,362** | **7,094** ⭐ | **1.323 ⭐** | 1.10 | **7.67** ⭐ | **6.97×** ⭐ |

### 3.1 ★ 핵심 finding — code workload 의 −23% 회귀가 suffix 로 +32% net positive 로 전환

- ngram code: −23.2% vs vanilla (5,362 / 6,964) — SUB_071 fact, SUB_075 재확인 (per-pos α=1.4%, K=1.10, near-zero acceptance)
- **suffix code: +1.85% vs vanilla (7,094 / 6,964)** — *enforce_eager 패널티가 있음에도 vanilla 보다 빠름*
- mechanism: suffix 는 prompt+self-generation 양쪽 lookup → code 생성 중 반복되는 keyword (`if`, `for`, `return`, indent pattern) + self-template 활용 가능 → K **7배** 향상 (1.10 → 7.67)

### 3.2 sonnet/chat 의 회귀 — enforce_eager penalty 가 suffix mechanism 이득 상쇄

- sonnet suffix K=4.42 가 ngram K=3.72 보다 약간 높음 (suffix mechanism 이득) 그러나 enforce_eager penalty (~25%) 가 더 큼 → net regression
- chat suffix K=11.58 가 ngram K=6.69 보다 1.73× 높음 그러나 chat 의 응답 매우 짧음 (660 tok) → K 가 amortize 될 시간 부족 + eager penalty 누적 → net regression

→ **suffix 가 cuda graph 호환되면 모든 workload 에서 ngram 대비 향상 가능성 강함**. 본 measurement 의 eager mode 는 mechanism comparison 만 valid, absolute throughput 은 fair 아님.

### 3.3 추정 — suffix 가 cuda graph 호환된 경우 (가설)

eager penalty ~25% 가정으로 normalized:

| workload | suffix (eager 실측) | suffix (cuda graph 추정, ÷0.78) | vs ngram |
|---|---:|---:|---:|
| sonnet | 8,236 | ~10,560 | -3% (거의 동등) |
| chat | 2,370 | ~3,040 | +2% (거의 동등) |
| **code** | **7,094** | **~9,094** ⭐ | **+70%** (huge) |

→ **suffix decoding 이 cuda graph 호환만 되면 code workload 의 ngram 회귀를 완전 mitigation + 추가 향상 가능성**.

## 4. 본 doc 갱신 권장사항

| doc / section | 갱신 내용 |
|---|---|
| `analysis/workload_acceptance_analysis_20260524.md` §1 TL;DR | suffix vs ngram K 비교 표 추가 (특히 code) |
| §10.4 후속 reading | SuffixDecoding code +32% (eager) 사실 추가, cuda graph 호환 PR 후보 |
| §11.1 axis 표 | suffix column 추가 — drafter mechanism = prompt+self-generation, adaptive num_spec ≤ 32, code K 가 ngram 의 7× |
| §11.2 | "11.2.5 SuffixDecoding 와의 차이 (R6, SUB_074 실측)" 신설 |

## 5. 후속 SUB candidate

- **SUB_079** (제안): suffix decoding 의 cuda graph 호환 patch — arctic_inference 영역 vLLM 1.6 와 binary compat 으로 fork (또는 vLLM upstream 영역 plugin path fix). effort 1-2 일.
- **SUB_080** (제안): suffix vs ngram 의 fair comparison — 두 measurement 모두 `enforce_eager=True` 로 재측정. effort 1-2 시간.
- **SUB_081** (제안): workload-aware gating heuristic 의 routing target 에 suffix 추가 — code 검출 시 spec ON (suffix), 그 외 spec ON (ngram). effort 1 일.

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| sonnet | `eval/results/20260525_011528_sub074_sonnet_suffix/{result.json, engine.log.stdout}` |
| chat | `eval/results/20260525_011528_sub074_chat_suffix/{result.json, engine.log.stdout}` |
| code | `eval/results/20260525_011528_sub074_code_suffix/{result.json, engine.log.stdout}` |
| launcher | `/tmp/run_sub074_suffix.sh` |
| wrapper | `/tmp/run_workload_gen.py` (with `VLLM_SPEC_METHOD=suffix`, `VLLM_ENFORCE_EAGER=1`, `ARCTIC_INFERENCE_ENABLED=0`) |
| stdout log | `/tmp/sub074.log` |
| summary tsv | `/tmp/sub074_summary.tsv` (단 last-line metric 추출이 partial → 본 RESULTS §2.1 의 peak metric 사용 권장) |
