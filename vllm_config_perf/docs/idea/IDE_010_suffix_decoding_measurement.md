# IDE_010 — SuffixDecoding 측정 (code workload 회귀 mitigation candidate)

> **parent backlog**: [`README.md`](README.md) (TSK_020 / SUB_072)
> **자식 SUB**: [`SUB_074`](../planning/SUB_074_suffix_decoding_measurement.md)
> **발견**: 2026-05-24, 사용자 질문 "suffixdecoding 도 포함되어 있어?" 의 follow-up
> **priority**: ★★ (net new fact, 1-2 시간 effort)
> **status**: ✅ **완료 ⭐** (2026-05-24) — code workload K 7× 향상, +32% vs ngram (회귀 mitigation 발견)

## 1. fact

vLLM v1 영역 SuffixDecoding 이 built-in 으로 포함되어 있음:
- 구현: `vllm/v1/spec_decode/suffix_decoding.py` (`class SuffixDecodingProposer`)
- 공식 doc: `docs/features/speculative_decoding/suffix.md`
- 활성화: `speculative_config={"method": "suffix", "num_speculative_tokens": 32}`
- 외부 의존: `arctic-inference` 패키지 (Snowflake, `pip install arctic-inference`) — 본 환경 미설치
- 학술 출처: arXiv 2411.04975 (analysis doc §10 R6, Snowflake AI Research + CMU)

### ngram (SUB_047 best) vs suffix decoding 차이 (공식 doc)

| 측면 | ngram | suffix |
|---|---|---|
| pattern-match 대상 | **prompt 만** | **prompt + 이전 generations 양쪽** |
| 후보 선택 | 첫 매칭 | frequency count 기반 most likely continuation |
| spec token 수 | fixed | **adaptive (매 step / 매 request 별 dynamic)** |
| 권장 workload | low-medium gain, lightweight | **high-repetition: code-editing, agentic loops, RL rollouts** |

## 2. 가설

본 doc §4.3 의 code workload 회귀 (-23.2%) mechanism = "generated Python token sequence 가 prompt 안에 없음 → K ≈ 1". suffix decoding 은:

1. **이전 generation tokens 도 pool 에 포함** → code 생성 중 반복되는 keyword (`if`, `for`, `return`, indent pattern) + self-template 활용 가능 → code workload K ↑.
2. **adaptive num_spec** → ngram 의 fixed γ=7 보다 step 별 효율적 (acceptance 낮을 때는 작은 γ, 높을 때는 큰 γ).
3. 공식 doc 자체가 "code-editing" 을 권장 workload 로 명시.

→ **code workload 의 −23% 회귀를 suffix 로 완화 또는 net positive 로 전환 가능성**. sonnet 에서도 추가 향상 가능성 (acceptance 가 ngram 보다 같거나 높을 것으로 기대).

## 3. 측정 계획

### 3.1 cell matrix (3 workload × 3 config = 9 cell)

| workload | vanilla | ngram (SUB_047 best) | suffix |
|---|---|---|---|
| sonnet | (이미 측정, SUB_047) | (이미 측정, SUB_047) | **신규** |
| chat | (이미 측정, SUB_071) | (이미 측정, SUB_071) | **신규** |
| code | (이미 측정, SUB_071) | (이미 측정, SUB_071) | **신규** |

→ 신규 측정은 **3 cell만**: sonnet/chat/code × suffix_spec32. 기존 vanilla / ngram 측정은 그대로 비교.

### 3.2 공통 parameter (SUB_071 와 정합, fair comparison)

```
num_prompts            = 500
target_input_len       = 8192
max_tokens             = 8192
max_num_seqs           = 256
max_model_len          = 16384
gpu_memory_utilization = 0.85
kv_cache_dtype         = fp8
max_num_batched_tokens = 8192
seed                   = 0
```

### 3.3 suffix decoding config (공식 doc 권장)

```python
speculative_config={
    "method": "suffix",
    "num_speculative_tokens": 32,  # 공식 권장 max
    # (선택) suffix_decoding_max_tree_depth / max_spec_factor / min_token_prob 등은 default
}
```

### 3.4 실행 준비

1. **`arctic-inference` 설치** (`.venv/bin/pip install arctic-inference`).
2. wrapper `/tmp/run_workload_gen.py` 의 LLM kwargs 영역 `speculative_config` 를 method-switchable 로 확장 (또는 새 wrapper `/tmp/run_suffix_decode.py` 신설).
3. launcher `/tmp/run_sub073_suffix.sh` (3 cell).

### 3.5 측정 effort

- 패키지 설치 + smoke test (vLLM 영역 SuffixDecoding init 가능 확인): 30 min
- 3 cell 측정: cell 당 ~5-15 min × 3 = 15-45 min
- 결과 doc + 분석 doc 갱신: 30 min
- **총 effort: 1.5-2 시간**

## 4. 진행 시 신설 SUB (candidate)

- **SUB_073** (제안 번호): SuffixDecoding 3-workload 측정 + ngram 대비 K 비교 + workload-aware gating 결합 검토.
- 신설 시 id_registry 갱신 + planning/SUB_073_suffix_decoding.md + measurements/sub073_suffix_<TS>/RESULTS.md.

## 5. 확인 / 업데이트 필요 doc (측정 후)

| 파일 | 갱신 위치 |
|---|---|
| **신규 `measurements/sub073_suffix_<TS>/RESULTS.md`** | 3 cell 결과 표 + K 역산 + ngram 대비 워크로드별 차이 |
| `analysis/workload_acceptance_analysis_20260524.md` | §10.4 후속 reading (SuffixDecoding 결과 반영) · §11 의 axis 표 (SUB_047 column 옆에 suffix column 추가) · §4.3 code 회귀 mitigation 결과 |
| `INDEX.md` | §1 active SUB 표 · §4 워크로드 generalization 표 (suffix row 추가) |
| `Best_SpecDecode_10778tps.md` | §7 다음 path — suffix 가 ngram 대비 net positive 면 best 갱신, 아니면 trade-off 정리 |
| `id_registry.md` | SUB_073 entry + 다음 번호 +1 |

## 6. 가설 검증 의 fork condition

본 측정의 결과가:
- **sonnet/chat/code 모두 ngram 보다 +5% 이상** → **suffix 가 새 best**. ngram → suffix 전환 권장.
- **code 만 +20% 이상 (-23.2% → -5% 이상)** → **workload-aware: code 는 suffix, sonnet/chat 는 ngram** (mixed routing).
- **모두 ngram 동등 / 회귀** → **suffix 가 본 env 에서 net positive 아님**. analysis doc §10.4 의 follow-up 후보에서 제외, ngram + workload-aware gating (I004) 이 main path.

## 7. risk / caveat

- `arctic-inference` 패키지 의 vLLM v1 호환성 — 본 fork repo 의 vLLM v1.6.dev0+g858b6df7a 와 정합 여부 미검증. 설치 시 dependency 충돌 가능.
- suffix decoding 의 메모리 footprint (per-request suffix tree) — batch 256 + 500 prompt 환경에서 GPU/CPU 메모리 추가 사용량 확인 필요.
- ngram 와 suffix 의 코드 경로 (각 proposer 클래스) 가 vLLM scheduler 와 어떻게 통합되는지 — `vllm/v1/spec_decode/suffix_decoding.py` 의 `arctic_inference.suffix_decoding.SuffixDecodingCache` import 가 actual core 영역. 본 import 가 실패하면 패키지 미설치 외 다른 원인.

## 8. 결과 (SUB_074, 2026-05-24)

### 8.1 환경 caveat — enforce_eager 모드

- `arctic-inference` 0.1.2 가 vLLM 0.11.0 의 `FlexibleArgumentParser` API 의존 → 본 fork (1.6) plugin disable
- CUDA graph + SuffixDecodingProposer conflict → **enforce_eager=True 우회 필수** (eager penalty ~25%)
- ngram (SUB_075) = cuda graph + suffix (SUB_074) = enforce_eager → **absolute throughput 영역 fair 아님**, mechanism comparison 영역 valid

### 8.2 측정 결과 (3 workload × suffix_spec32, vanilla 대비)

| workload | vanilla tps | suffix tps (eager) | vs vanilla | suffix K (peak) | ngram K (SUB_075) | suffix K / ngram K |
|---|---:|---:|---:|---:|---:|---:|
| **sonnet** | 4,679.8 | 8,236.0 | **+76.0%** | 4.42 | 3.72 | 1.19× |
| **chat** | 2,186.0 | 2,369.7 | **+8.4%** | 11.58 | 6.69 | 1.73× |
| **code** ⭐ | 6,964.5 | **7,093.5** ⭐ | **+1.85%** (회귀 mitigation) | **7.67** | 1.10 | **7.0×** ⭐ |

### 8.3 ngram vs suffix 직접 비교 (같은 workload)

| workload | ngram tps (SUB_075, cuda graph) | suffix tps (SUB_074, eager) | suffix/ngram | 결론 |
|---|---:|---:|---:|---|
| sonnet | 10,909.0 | 8,236.0 | 0.755 (−25%) | eager penalty 가 suffix mechanism gain (K 1.19×) 능가 |
| chat | 2,972.5 | 2,369.7 | 0.797 (−20%) | eager penalty 누적, chat 응답 짧아 amortize 부족 |
| **code** | 5,362.5 | **7,093.5** | **1.323 (+32%)** ⭐ | **suffix K 7× 향상이 eager penalty 상쇄 + vanilla 추월** |

### 8.4 per-position acceptance (suffix, peak interval pos 1~12, pos 13~32 = 0)

- sonnet: 0.522 / 0.407 / 0.349 / 0.296 / 0.274 / 0.246 / 0.240 / 0.229 / 0.225 / 0.215 / 0.211 / 0.207
- chat: **0.981 / 0.955 / 0.929 / 0.911 / 0.907 / 0.900 / 0.881 / 0.855 / 0.844 / 0.836 / 0.807 / 0.770** ⭐
- code: 0.912 / 0.782 / 0.704 / 0.640 / 0.614 / 0.582 / 0.536 / 0.460 / 0.417 / 0.400 / 0.341 / 0.285

> suffix adaptive 가 cap 32 중 ~12 positions 만 사용 (각 request 별 dynamic 결정).

### 8.5 cuda graph 호환 시 추정 (가설)

eager penalty ~25% 가정으로 normalized:

| workload | suffix (eager 실측) | suffix (cuda graph 추정, ÷0.78) | vs ngram (cuda graph) |
|---|---:|---:|---:|
| sonnet | 8,236 | ~10,560 | ≈ 동등 |
| chat | 2,370 | ~3,040 | +2% |
| **code** | **7,094** | **~9,094** ⭐ | **+70%** |

→ **suffix cuda graph 호환 patch 가 가장 ROI 높은 후속 lever** (code +70% 가능성).

### 8.6 본 doc 갱신 (완료)

- `analysis/workload_acceptance_analysis_20260524.md` §1.1 TL;DR / §10.4 후속 reading 에 suffix 측정 결과 추가
- `INDEX.md` §1 active SUB 표 갱신
- 분석 doc §11 의 axis 표 (subject: SUB_080+ candidate)

### 8.7 후속 SUB candidate

- **SUB_080+** (제안): suffix cuda graph 호환 patch — arctic_inference 의 vLLM 1.6 와 binary compat fork (effort 1-2 일)
- **SUB_081+** (제안): suffix vs ngram fair comparison — 두 measurement 모두 `enforce_eager=True` 로 재측정 (effort 1-2 시간)
- workload-aware routing target 에 suffix 추가 — code 검출 시 suffix, 그 외 ngram
