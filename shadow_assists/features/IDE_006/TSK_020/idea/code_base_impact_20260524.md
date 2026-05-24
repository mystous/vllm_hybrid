# IDE_009~014 vs 본 fork code base — 적용 / 영향 정리 (2026-05-24)

> **parent**: [`README.md`](README.md) (idea backlog) / [`evaluation_summary_20260524.md`](evaluation_summary_20260524.md)
> **목적**: 사용자 지적 "TSK_020 idea 가 각기 적용이 아닌데 파편화 되었다" 의 정정 — 각 IDE 가 본 fork 영역 어떤 code/file 에 어떻게 영향을 미쳤는지 분류 + 본 fork 의 현재 vLLM core 상태 정리.

---

## 1. 본 fork 의 현재 vLLM core 상태 (IDE_009~014 이전 SUB 들이 적재)

본 fork repo (`vllm/v1/spec_decode/ngram_proposer.py`) 에 적재된 env-tunable lever — **모두 default OFF / upstream 동일 동작**:

| env var | source SUB | default | 효과 (활성화 시) | status |
|---|---|---|---|---|
| **VLLM_NGRAM_NUM_THREADS_CAP** | SUB_047 | `1` (= upstream) | `8` 시 batch parallel threading enable | ✅ 사용 가능 (sonnet +1.65%) |
| **VLLM_NGRAM_DIVIDE_BY_TP** | SUB_047 | `1` (= upstream) | `0` 시 TP 분할 무시 | ✅ 사용 가능 (cap=8 와 함께) |
| VLLM_NGRAM_THRESHOLD | SUB_065 | `8192` (= upstream) | small batch 영역 multi-thread enable | ✗ 기각 (모두 −1.69~−0.07%) |
| VLLM_NGRAM_BROADCAST | SUB_066 | `0` | rank 0 만 ngram compute + broadcast | ✗ 기각 (−1.30%) |
| VLLM_NGRAM_PRECOMPUTE | SUB_067 | `0` | background thread 영역 precompute | ✗ 기각 (−3.77%) |
| VLLM_NGRAM_TOP_M | SUB_057 | `1` (= upstream) | top-M chain 영역 numba kernel | ◐ wired but rejection_sampler tree verify TODO |

→ **본 fork 영역 vLLM core 영역 변경된 lever 는 SUB_047 (★), SUB_057 (◐), SUB_065/066/067 (기각, default OFF 영역 inert)**. 모두 IDE_009~014 이전.

---

## 2. IDE_009~014 의 code base impact 분류

**핵심 사실**: **IDE_009~014 (SUB_073~079) 중 본 fork 의 vLLM core 영역 변경한 IDE 는 없음**. 모두 (a) wrapper / 외부 script 추가, (b) doc 만, (c) 외부 PR proposal 형태.

| IDE | code base impact 종류 | 변경 file | 본 fork vLLM 영향 |
|---|---|---|---|
| **IDE_009** (SUB_073) | **D — doc only** | `analysis/`, `Best`, `INDEX`, `RESULTS sub044/sub047`, `id_registry` | 0 (framing 정정만) |
| **IDE_010** (SUB_074) | **B — wrapper 확장** | `/tmp/run_workload_gen.py` (`VLLM_SPEC_METHOD=suffix`, `VLLM_ENFORCE_EAGER`) + 외부 패키지 (`arctic-inference` 설치) | 0 (vLLM 영역 기존 `suffix_decoding.py` 사용, `enforce_eager=True` 우회) |
| **IDE_011** (SUB_075) | **B — wrapper 확장** | `/tmp/run_workload_gen.py` (`VLLM_ENABLE_SPEC_STATS=1` 영역 `disable_log_stats` 토글) | 0 (vLLM 영역 기존 `metrics.py` 사용) |
| **IDE_012** (SUB_076) | **C — 외부 tool 신설** | `/tmp/workload_classifier.py` (regex feature + rule) + `/tmp/run_sub076_classifier.sh` | 0 (vLLM 미연동, prompt set 분류만) |
| **IDE_013** (SUB_077) | **E — 외부 PR proposal** | `measurements/sub077_pr_draft_20260524/PR_DRAFT.md` (vllm-project/vllm 영역 isolated diff 제안) | 0 (본 fork 변경 없음, 단 SUB_047 의 기존 patch 인용) |
| **IDE_014** (SUB_078 + SUB_079) | **B — small model wrapper 신설** | `/tmp/run_sub078_wrap.py` (TP=1 + arbitrary model 지원, run_workload_gen.py 재사용) + launcher 2개 | 0 (Qwen small model 측정, vLLM 영역 변경 없음) |

### 2.1 분류 legend

- **A**: vLLM core 직접 수정 (e.g., SUB_047 같은 patch) — IDE_009~014 중 없음
- **B**: wrapper / launcher / measurement script 영역 확장 (외부 `/tmp/`) — vLLM 영역 기존 feature 활용
- **C**: 독립 tool 신설 (vLLM 미연동) — analysis / classifier 등
- **D**: doc only — measurement / code 변경 없음
- **E**: 외부 PR proposal — 본 fork 변경 없음

---

## 3. 사용자 지적 "파편화" 의 정정 — IDE 들의 실제 grouping

IDE_009~014 가 각각 별도 idea 로 신설되었지만, 본질적으로는 **TSK_020 의 단일 framework 의 sub-aspects**. 본 grouping:

### Group α — vLLM 영역 fork patch 의 정직성 / upstream 환원
- **IDE_009 (정직성 정정)** + **IDE_013 (upstream PR)** = SUB_047 fork patch 의 contribution 정량화 + 환원
- 단일 lever (cap=8 + div_tp=0) 의 (1) honest measurement framing (2) external sharing

### Group β — TSK_020 framework 의 empirical validation
- **IDE_011 (acceptance rate 직접 측정)** = analysis doc §3 의 R/K 모델 검증
- **IDE_014 (small model cross-validation)** = analysis doc §11.3 차별점 corroboration
- 측정 contribution, vLLM core 영역 변경 없음

### Group γ — code workload 회귀 의 해결 path
- **IDE_010 (suffix decoding)** = TSK_020 의 code workload 회귀 (SUB_071 −23.2%) mitigation 후보
- vLLM 영역 기존 `SuffixDecodingProposer` 활용 (env 영역 `method=suffix` 분기)
- 본 fork 영역 patch 없이도 사용 가능 (단 cuda graph 호환 patch 가 별도 필요 — 후속 SUB)

### Group δ — production translation
- **IDE_012 (workload-aware gating classifier)** = analysis doc §6 의 권장사항 의 1차 PoC
- classifier 단독 (vLLM 미연동), routing/serving 통합은 후속 SUB

### 본 grouping 의 결론

| Group | 본질 | IDE | code base impact |
|---|---|---|---|
| α | fork patch 의 정직성 + upstream 환원 | IDE_009, IDE_013 | D + E (doc + 외부 PR) |
| β | framework empirical validation | IDE_011, IDE_014 | B (wrapper 확장) |
| γ | code 회귀 해결 path | IDE_010 | B (wrapper, suffix method 활용) |
| δ | production translation | IDE_012 | C (외부 tool, vLLM 미연동) |

→ **6 IDE 가 4 group, 본질적으로는 단일 framework (TSK_020 의 R/K + workload-shape 의존성) 의 4 axis 측정/적용**.

---

## 4. "지금 code base 와 구별" — 실제 사용 가능 한 상태

### 4.1 현재 본 fork 에서 즉시 사용 가능

| lever | activation | source SUB | 효과 |
|---|---|---|---|
| ngram cap=8 + div_tp=0 | `VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0` | **SUB_047** ⭐ | sonnet +1.65% (vs vLLM default), +134.1% (vs vanilla) |
| acceptance stats 수집 | `VLLM_ENABLE_SPEC_STATS=1` (wrapper) | **SUB_075** (IDE_011) | metric 추출 |
| suffix decoding (단 enforce_eager + arctic_inference plugin off) | wrapper 의 `VLLM_SPEC_METHOD=suffix VLLM_ENFORCE_EAGER=1` + `ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""` | **SUB_074** (IDE_010) | code workload +1.85% vs vanilla, +32% vs ngram |
| workload classifier | `/tmp/workload_classifier.py --input ... --output ...` | **SUB_076** (IDE_012) | macro accuracy 1.000 (본 환경) |

### 4.2 현재 본 fork 에서 사용 불가 (후속 patch 필요)

| 영역 | 필요 patch | source idea |
|---|---|---|
| suffix decoding + cuda graph | arctic_inference 의 vLLM 1.6 binary compat fork (또는 plugin path fix) — effort 1-2 일 | IDE_010 §8.7 |
| workload-aware routing 통합 | vLLM 영역 per-request `speculative_config` override 지원 (upstream PR 또는 외부 router) | IDE_012 §8.4 |
| precise issue #16258 reproduction | opt-125m / starcoder2-3b HF auth + download | IDE_014 §7.7 |

### 4.3 외부 환원 candidate

| 항목 | target | source |
|---|---|---|
| SUB_047 patch upstream PR | vllm-project/vllm | IDE_013 (SUB_077, draft 완료, human submit 대기) |

---

## 5. 종합 — "code base 와의 구별"

| 측면 | 사실 |
|---|---|
| 본 fork 영역 vLLM core 변경 (IDE_009~014 작업 결과) | **0** — IDE 들 모두 wrapper / 외부 tool / doc / PR proposal 만 |
| 본 fork 영역 기존 vLLM patch (IDE 이전) | **5 env-tunable (cap, threshold, broadcast, precompute, top-m), 모두 default OFF** (SUB_047 만 적극 사용) |
| 본 fork 영역 즉시 사용 가능 lever | 4종 (ngram cap=8, acceptance stats, suffix eager, classifier) |
| 본 fork 영역 후속 patch 필요 항목 | 3종 (suffix cuda graph, workload-aware routing, real issue repro) |
| 외부 환원 candidate | 1종 (SUB_047 → upstream PR) |

→ **IDE_009~014 의 작업은 본 fork 의 vLLM core 변경을 추가하지 않았고**, 모두 (a) measurement / wrapper / 외부 script 영역 또는 (b) doc / proposal 영역 contribution. 본 fork code base 의 *상태* 는 IDE 이전 (SUB_047 등) 그대로이며, IDE 작업은 그 위에서 *측정 / 분석 / 외부 환원 / production heuristic* 을 진행한 것.

이러한 분류로 "파편화" 가 해소됨 — IDE 들이 별개 idea 가 아니라 **TSK_020 의 framework / production translation / 외부 환원 의 4 group**.

---

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| 본 fork vLLM patches (git diff) | `git diff origin/main -- vllm/v1/spec_decode/ngram_proposer.py` 등 |
| IDE 별 측정 RESULTS | `measurements/sub074~079_*/RESULTS.md` |
| IDE doc | `idea/IDE_009~014_*.md` |
| evaluation 종합 | [`evaluation_summary_20260524.md`](evaluation_summary_20260524.md) |
| idea backlog index | [`README.md`](README.md) |
