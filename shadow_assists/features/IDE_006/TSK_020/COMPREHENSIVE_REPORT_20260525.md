# TSK_020 — 종합 리포트 (2026-05-25 KST)

> **scope**: 2026-05-24 ~ 2026-05-25 약 13 시간 작업 — SUB_073~089, IDE_009~014, Phase 1~4, 그리고 SUB_085~088 의 Phase 2 unblock + fair baseline
> **parent**: TSK_020 / PLN_001 (vLLM CPU 활용 + spec decode tuning)
> **scope branch**: `feat/spec-decode-tuning`
> **본 fork code 변경**: 14 줄 vLLM core (utils + arg_utils) + wrapper env-tunable
> **★ user-facing top-level guide**: [`/spec_decoding/README.md`](../../../../spec_decoding/README.md) — production-ready Trident core (always-on) / AGSD (Trident core + gating) + 활성화 한 줄 + 6-workload all-fair benchmark (SUB_093)
> **신규 측정**: [`SUB_093 RESULTS`](measurements/sub093_full_matrix_util_20260525/RESULTS.md) — 57 cell × util 캡처 (Llama 70B 18 + 소형 27 + cross-val 12, 2026-05-25 19:01 KST 완료)

---

## Executive Summary (두괄식)

**본 fork 영역 진짜 새 best = SuffixDecoding + cudagraph_mode=PIECEWISE + gpu_memory_utilization=0.80**

| workload | vanilla (SUB_086, fair gmu=0.80) | **현 best (SUB_085 v2 suffix PIECEWISE)** | **fair contribution** |
|---|---:|---:|---:|
| sonnet | 7,709.8 | **11,589.5** | **+50.3%** ⭐ |
| chat | 2,186.9 | **3,582.4** | **+63.8%** ⭐ |
| code | 6,717.8 | **7,990.0** | **+18.9%** ⭐ (ngram −20.7% 회귀 영역 mitigation) |

→ 3 workload 모두 net positive (Llama-70B + TP=8 + H100×8 환경)
→ small model (Qwen 0.5B/1.5B) 영역 모든 spec method 회귀 — vanilla 권장
→ 본 fork code 변경 14 줄 (backward-compat 100%) + wrapper env-tunable

이전 historical claim ("+134.12%") 영역 wrapper-historical noise 영역 정정 (SUB_073/IDE_009 영역 정직 정리, SUB_086 영역 fair baseline 신설).

---

## §1. 본 fork 영역 production-ready Best Configuration

### 1.1 활성화 코드

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=8,
    max_model_len=16384,
    max_num_seqs=256,
    gpu_memory_utilization=0.80,              # ★ cudagraph PIECEWISE + spec 영역 memory headroom
    enforce_eager=False,
    kv_cache_dtype="fp8",
    max_num_batched_tokens=8192,
    disable_log_stats=True,
    seed=0,
    compilation_config={"cudagraph_mode": "PIECEWISE"},   # ★ SUB_085 핵심 patch
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,
    },
)

sp = SamplingParams(temperature=0.0, max_tokens=8192)
outputs = llm.generate(prompts, sp)
```

### 1.2 환경 설정 (필수)

```bash
# arctic_inference plugin disable — 본 fork 영역 binary incompat (vllm.attention.layer 영역 없음)
# 단 SuffixDecodingProposer 영역 SuffixDecodingCache 영역 lazy import 영역 plugin 없이도 작동
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # cudagraph PIECEWISE 영역 memory fragmentation 영역 줄임
```

### 1.3 본 fork 영역 추가된 vLLM core 변경 (14 줄, backward-compat 100%)

#### `vllm/utils/__init__.py` (SUB_081, +5 줄)

```python
# SUB_081 (Phase 2): Re-export FlexibleArgumentParser at vllm.utils for
# backward-compat with arctic_inference 0.1.2 (expects vllm 0.11.0 API path).
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: F401
except ImportError:
    pass
```

#### `vllm/engine/arg_utils.py` (SUB_084, +9 줄)

```python
# SUB_084 (Phase 2 follow-up): no-op stub for arctic_inference 0.1.2 compat.
# vLLM 0.11.0 had `_is_v1_supported_oracle` to gate v0/v1 selection.
# 본 fork 1.6 영역 v1-only 이므로 그 method 영역 자체 영역 의미 없음 (deprecated).
# arctic_inference 의 EngineArgsPatch 가 본 method 영역 wrap 영역 의존
# → 항상 True 반환 stub 영역 추가 (호출되어도 v1 진행).
def _is_v1_supported_oracle(self, *args, **kwargs):
    return True
```

→ 두 패치 모두 **upstream 이 호출하지 않는 deprecated symbol 의 re-export / no-op stub**. default behavior 영향 0.

### 1.4 wrapper 영역 추가 env-tunable

`/tmp/run_workload_gen.py` 영역 추가된 env 영역:

| env var | 기본값 | 효과 |
|---|---|---|
| `VLLM_ENFORCE_EAGER` | `0` (cuda graph ON) | `1` 시 enforce_eager (cudagraph off) |
| `VLLM_ENABLE_SPEC_STATS` | `0` (off) | `1` 시 `disable_log_stats=False` → spec metric 출력 |
| `VLLM_SPEC_METHOD` | `ngram` | `suffix` 시 `speculative_config={"method":"suffix"}` |
| `VLLM_CUDAGRAPH_MODE` | (unset, default `FULL_AND_PIECEWISE`) | `PIECEWISE` 시 `compilation_config` override |

본 fork 의 vllm core 영역 영향 X.

---

## §2. all-fair table (모든 측정 영역 gmu=0.80 + cudagraph PIECEWISE + same wrapper)

### 2.1 Large model (Llama-3.3-70B + TP=8 + H100×8 + 500p × 8192in × 8192max)

| workload | **vanilla** (SUB_086) | **ngram cap=8** (SUB_087) | **suffix PIECEWISE** (SUB_085 v2) | ngram vs vanilla | suffix vs vanilla | suffix vs ngram |
|---|---:|---:|---:|---:|---:|---:|
| sonnet | 7,709.8 | 10,139.2 | **11,589.5** ⭐ | +31.5% | **+50.3%** ⭐ | +14.3% |
| chat | 2,186.9 | 2,846.4 | **3,582.4** ⭐ | +30.2% | **+63.8%** ⭐ | +25.9% |
| code | 6,717.8 | 5,326.6 ✗ | **7,990.0** ⭐ | **−20.7% (회귀)** ✗ | **+18.9%** ⭐ | **+50.0%** ⭐⭐ |

→ **suffix PIECEWISE 영역 3 workload 모두 ngram + vanilla 영역 능가**.
→ code workload 영역 ngram −20.7% 회귀 → suffix +18.9% **net positive 영역 완전 mitigation**.

### 2.2 Small / medium model (Qwen2.5-0.5B/1.5B/7B + TP=1)

본 doc 영역 multiple SUB 의 cross-cut (SUB_079 / SUB_088 / SUB_090):

| model | config | tps | vs vanilla | source |
|---|---|---:|---:|---|
| Qwen 0.5B | vanilla | 11,220.2 | — | SUB_090 |
| Qwen 0.5B | ngram cap=8 PIECEWISE | 7,793.9 | **−30.5%** | SUB_090 (PIECEWISE 영역 SUB_079 영역 -59% 영역 +28pp 향상) |
| Qwen 0.5B | suffix PIECEWISE | 5,376.2 | -52.1% | SUB_088/090 |
| Qwen 1.5B | vanilla | 10,388.5 | — | SUB_090 |
| Qwen 1.5B | ngram PIECEWISE | 5,855.0 | −43.6% | SUB_090 |
| Qwen 1.5B | suffix PIECEWISE | 4,064.4 | −60.9% | SUB_090 |
| **Qwen 7B** | **vanilla** | **5,556.2** | — | **SUB_090** ⭐ |
| **Qwen 7B** | **ngram PIECEWISE** | **4,593.5** | **−17.3%** ⭐ | **SUB_090 (boundary 근접)** |
| Qwen 7B | suffix PIECEWISE | 3,515.5 | −36.7% | SUB_090 |

→ **R/K boundary 영역 7B 영역 70B 사이** (Qwen 7B ngram -17% 영역 가장 boundary 근접, large 70B 영역 suffix +50% net positive). **production 권장: ≤7B → vanilla, ≥70B → suffix PIECEWISE, boundary 7B~70B 영역 후속 측정 필요**.

### 2.3 ★ small model + ngram = universal regression — 5-model cross-validation (SUB_078/079/088/090/091)

| source | hardware | model | code workload regression | 본 doc 영역 위치 |
|---|---|---|---:|---|
| **issue #16258 외부** | 2× L4 | opt-125m | **2.12×** | issue link |
| **SUB_091 본 fork** | **H100×1** | **opt-125m** | **2.13×** ⭐ (정확 reproduction) | (R6, SUB_091 RESULTS) |
| SUB_091 본 fork | H100×1 | starcoder2-3b | 2.30× | (SUB_091 RESULTS) |
| SUB_078 본 fork | H100×1 | Qwen 0.5B | 2.46× | (SUB_078 RESULTS) |
| SUB_078 본 fork | H100×1 | Qwen 1.5B | 2.63× | (SUB_078 RESULTS) |
| SUB_090 본 fork (PIECEWISE) | H100×1 | Qwen 0.5B | **1.44×** ⭐ (PIECEWISE 영역 차이) | (SUB_090 RESULTS) |
| SUB_090 본 fork (PIECEWISE) | H100×1 | Qwen 7B | **1.21×** (boundary 근접) | (SUB_090 RESULTS) |

→ **5 model × 다양한 hardware/cudagraph mode 영역 small model + ngram = 1.2~2.6× regression** (universal). cudagraph PIECEWISE mode 영역 +28pp 영역 향상 발견 (SUB_090, default FULL_AND_PIECEWISE 영역 SUB_079 -59% vs PIECEWISE 영역 SUB_090 -30%).

### 2.3 acceptance metric (suffix vs ngram, all-fair gmu=0.80 + PIECEWISE)

| workload | ngram K / α | suffix K / α | K 비율 | α 비율 |
|---|---:|---:|---:|---:|
| sonnet | 1.66 / 9.5% | 5.11 / 77.0% | 3.08× | 8.1× |
| chat | 5.98 / 71.2% | 10.06 / 92.7% | 1.68× | 1.30× |
| **code** | **1.09 / 1.2%** | **4.08 / 40.1%** | **3.74×** | **33×** |

→ suffix 영역 모든 workload 영역 K, α 영역 큰 향상. code 영역 가장 극적 (ngram α=1.2% 영역 prompt 안 매칭 거의 없음, suffix α=40% 영역 prompt + generated 양쪽 활용).

---

## §3. mechanism analysis — 왜 suffix 가 모든 workload 영역 best 인가

### 3.1 ngram vs suffix drafter 영역 차이

| 측면 | ngram (vLLM built-in) | suffix (arctic_inference, lazy import) |
|---|---|---|
| draft source | **prompt 내 n-gram 만** (KMP match) | **prompt + 이전 generations 양쪽** (suffix tree, frequency-weighted) |
| draft length | fixed γ (e.g., 7) | adaptive (1~num_spec_max, e.g., 32 max) |
| code workload 영역 fit | ✗ (prompt 영역 word salad 영역 generated Python 영역 disjoint, K≈1) | ✓ (self-generated keyword 영역 활용, K≈4) |
| 본 fork 영역 cuda graph 호환 | ✓ (fixed batch shape) | ✓ (단 PIECEWISE mode 필요 — FULL graph capture 영역 dynamic shape 영역 차단) |

### 3.2 본 doc 영역 R/K framework (정량 모델)

```
spec_wall / vanilla_wall ≈ R / K  (where R = spec step overhead, K = mean accepted tokens per draft)

K > R → net positive (가속)
K < R → net regression
```

본 측정 영역 R, K 의 fact (large model):
- **R ≈ 1.3** (large model + cudagraph + 1-token vs 8-token forward overhead, R/K analysis 영역 fit)
- **K (suffix PIECEWISE)**: sonnet 5.11 / chat 10.06 / code 4.08 → 모두 R 능가 → net positive
- **K (ngram cap=8)**: sonnet 1.66 / chat 5.98 / **code 1.09 (< R)** → code 만 net regression

본 측정 영역 R, K 의 fact (small model, Qwen 0.5B/1.5B):
- **R ≈ 5~10** (T_target 영역 small 영역 spec overhead 영역 forward time 영역 큰 비율)
- **K (suffix)**: 1~4 정도 → R 못 넘김 → 모든 workload 회귀

→ **R/K boundary 영역 model-size dependent**. 7B~32B 영역 boundary 영역 SUB_090 영역 확정.

---

## §4. workload generalization fact (SUB_071~088 정리)

### 4.1 large model 영역 workload 영역 차이

| workload | prompt 특성 | response 특성 | ngram fit | suffix fit |
|---|---|---|---|---|
| sonnet | Shakespeare lines (반복 어휘) | sonnet style continuation | ★★★ (prompt 안 매칭 빈도 ↑) | ★★★ + self-quote |
| chat | sonnet excerpt + system/user template | meta-analysis (sonnet 인용 + meta 어휘) | ★★ (sonnet 부분 만 hit) | ★★★ (self-quote 영역 K ↑) |
| code | HumanEval-style stub + word-salad comment | Python implementation | ✗ (vocab disjoint) | ★★ (keyword self-repeat) |

### 4.2 small model 영역 fundamental constraint

본 SUB_078/079/088 영역 6/6 + 6/6 cell 모두 net regression — workload 무관:
- T_target 영역 small (~0.1-1 ms/step)
- Spec step overhead R 영역 forward time 영역 비교 영역 큰 비율 (~5-10×)
- 어떤 K 영역 R 못 넘김

→ **small model 영역 spec method 영역 fundamentally 안 됨**. 단 boundary (1.5B → 7B → 32B → 70B) 영역 어느 시점 영역 net positive 영역 SUB_090 영역 확정.

---

## §5. historical claim 정정 (SUB_073/IDE_009 영역 정직 정리)

### 5.1 이전 "+134%" claim 영역 wrapper-historical noise

| 단계 | config | source wrapper | tps | claim |
|---|---|---|---:|---|
| (1) historical vanilla | `speculative_config=None` | `run_spec_decode.py` (SUB_043) | 4,679.8 | (historical baseline) |
| (2) vLLM built-in spec ON cap=1 | `num_spec=7` | `run_spec_decode.py` (SUB_044) | 10,778.6 | "+130.3%" (historical) |
| (3) SUB_047 fork patch cap=8/div=0 | `+ env` | `run_spec_decode.py` (SUB_047) | 10,956.5 | "+134.12%" (historical, +1.65% over default) |

### 5.2 fair baseline 영역 SUB_086 (new wrapper, gmu=0.80, PIECEWISE)

| 단계 | config | source wrapper | tps |
|---|---|---|---:|
| (1) fair vanilla | `speculative_config=None` | `run_workload_gen.py` (SUB_086) | **7,709.8** ★ |
| (2) ngram cap=8 + PIECEWISE | `num_spec=7, cap=8, div_tp=0, cudagraph PIECEWISE` | `run_workload_gen.py` (SUB_087) | 10,139.2 (+31.5%) |
| (3) suffix PIECEWISE | `method=suffix, num_spec=32, cudagraph PIECEWISE` | `run_workload_gen.py` (SUB_085 v2) | **11,589.5 (+50.3%)** ⭐ |

→ **fair contribution = +50.3% (suffix vs vanilla), +14.3% (suffix vs ngram), +31.5% (ngram vs vanilla)**.
→ 이전 "+134%" 영역 wrapper-historical (다른 prompt builder 영역 vanilla baseline 영역 noise) — fair number 영역 +50.3%.

---

## §6. production 권장 — workload + model size 별 decision tree

### 6.1 Large model (≥ 7B 영역 70B class)

```
prompt 입력
  ↓
[ workload classifier (SUB_076 / IDE_012) 영역 분류 ]
  ↓
  ├─ sonnet-like (long-form, repetitive vocab) → suffix PIECEWISE (best, +50.3%)
  ├─ chat-like (system + user excerpt + meta) → suffix PIECEWISE (best, +63.8%)
  └─ code-like (def/class/comment heavy) → suffix PIECEWISE (best, +18.9% mitigation)
```

→ **모든 workload 영역 suffix PIECEWISE** 권장 (3 workload 모두 fair best).

### 6.2 Small / medium model (≤ 7B) — SUB_090 측정 확정

```
prompt 입력
  ↓
  ├─ 0.5B/1.5B → vanilla (spec OFF) — 모든 spec method -30~-61% 회귀
  └─ 7B (boundary 근접) → vanilla 권장, 또는 ngram PIECEWISE (-17%, acceptable 영역 단 net loss)
```

→ **≤ 7B 영역 모든 workload 영역 vanilla 권장**. ngram 영역 PIECEWISE mode 영역 small model 영역 영역 회귀 폭 영역 ~30% 감소 (SUB_079 영역 -59% → SUB_090 영역 -30.5%), 단 still net negative.

### 6.3 boundary (7B → 70B 사이)

- **SUB_090 영역 7B 영역 ngram -17% 영역 boundary 근접** 확인.
- (참조) Llama-70B 영역 suffix +50.3% net positive 확정 (SUB_085 v2).
- **boundary 영역 7B↔70B 사이 (예: 14B, 32B)** — 후속 측정 candidate (Qwen 32B cached, TP=2/4).

---

## §7. SUB / IDE / commit history (전체 작업)

### 7.1 SUB summary (SUB_073~088 + 진행 중 SUB_089~092)

| SUB | parent IDE | status | 핵심 결과 |
|---|---|---|---|
| SUB_073 | IDE_009 | ✅ | vanilla framing 정정 (fork 단독 +1.65%) |
| SUB_074 | IDE_010 | ✅ | suffix enforce_eager, code +32% vs ngram (eager penalty 영역) |
| SUB_075 | IDE_011 | ✅ | acceptance rate 직접 측정 (chat α=81% surprise) |
| SUB_076 | IDE_012 | ✅ | workload classifier accuracy 1.000 |
| SUB_077 | IDE_013 | ◐ | vLLM upstream PR draft (사용자 지시 영역 제외) |
| SUB_078 | IDE_014 | ✅ | Qwen 0.5B/1.5B + code 영역 ngram -59~-62% |
| SUB_079 | IDE_014 | ✅ | Qwen sonnet/chat 확장 영역 6/6 cell 회귀 |
| SUB_080 | (Phase 1) | ✅ | workload-aware gating analytical (+9.5~+30.3%) |
| SUB_081 | (Phase 2) | ◐ | suffix cuda graph 영역 1 fix (FlexibleArgumentParser) |
| SUB_082 | (Phase 3) | ◐ | dual instance routing analytical viability |
| SUB_083 | (Phase 4) | ◐ | rejection_sampler tree verify design |
| SUB_084 | (Phase 2 후속) | ◐ | _is_v1_supported_oracle stub (2nd blocker) |
| **SUB_085** | (Phase 2 unblock) | ✅⭐⭐ | **suffix + cudagraph PIECEWISE — 3 workload net positive** |
| **SUB_086** | (fair baseline) | ✅ | vanilla gmu=0.80 + PIECEWISE fair baseline |
| **SUB_087** | (all-fair) | ✅ | ngram cap=8 + PIECEWISE + gmu=0.80 baseline |
| **SUB_088** | IDE_014 확장 | ✅ | small model + suffix 영역도 universal regression |
| SUB_089 | (variance) | 🔄 | sonnet × suffix PIECEWISE × 3-run (진행 중) |
| SUB_090 | (R/K sweep) | 대기 | Qwen 0.5B/1.5B/7B/32B × code × 3 config |
| SUB_091 | IDE_014 정확화 | 대기 | opt-125m/starcoder2-3b reproduction (HF auth 필요) |
| SUB_092 | (router PoC) | 대기 | HTTP router actual deploy |

### 7.2 commit history (8 commit, 모두 push)

| commit | 내용 |
|---|---|
| `cd3ecf8d8` | SUB_071 + analysis doc 신설 |
| `3d500135e` | IDE_009~014 + SUB_072~083 (Phase 1~4 plan) |
| `8cee979ef` | SUB_084 (Phase 2 next blocker — incompat 영역 확정 시도) |
| `ec886b240` | **SUB_085 v2 + SUB_086 — Phase 2 unblock 성공** ⭐⭐ |
| `7094f284f` | Best + analysis doc 영역 SUB_085/086 통합 |
| `90f7f823d` | **SUB_087 — all-fair table 확정** |
| `90ee85328` | SUB_088 — small model + suffix universal regression |
| (이번 commit) | 본 종합 리포트 |

---

## §8. 후속 작업 (남은 SUB candidate)

| 우선순위 | SUB | scope | effort | ROI |
|---|---|---|---|---|
| ★★★ | SUB_090 | R/K model-size sweep (Qwen 0.5B~32B × code × 3 config) | 1 시간 | boundary 확정, production model-size threshold |
| ★★ | SUB_092 | router HTTP server PoC (Phase 1 actual deploy) | 0.5 일 | production-applicable, vLLM core 변경 0 |
| ★ | SUB_089 | sonnet 3-run variance (진행 중) | 20 분 | best canonical |
| ★ | SUB_091 | precise issue #16258 reproduction (opt-125m) | 30 분 (HF auth 후) | 외부 cross-validation 정확화 |
| ★ | Phase 3 dual instance | actual init + measurement | 1-2 일 | routing 정식화 (단 본 환경 영역 large effort) |
| ◐ | Phase 4 tree verify | rejection_sampler tree path | 1 주 | sonnet/chat 추가 +40-80 pp 가능성 |

**제외**: IDE_013/SUB_077 (vLLM upstream PR) — 사용자 지시.

### 본 session 영역 추가 완료된 SUB (2026-05-25 KST 11:00~11:45)

| SUB | scope | 결과 |
|---|---|---|
| **SUB_089** | sonnet canonical 3-run variance | avg 11,687.4, var 0.20%, fair +51.6% |
| **SUB_090** | R/K model-size sweep (Qwen 0.5B/1.5B/7B × code) | boundary 7B↔70B 확정, PIECEWISE 영역 small model ngram +28pp 향상 |
| **SUB_091** | precise issue #16258 reproduction (opt-125m + starcoder2-3b) | opt-125m 2.13× regression — issue 영역 2.12× 영역 정확 일치 ⭐⭐ |
| **SUB_092** | router HTTP server PoC (Phase 1 actual deploy) | classifier router 0.26 ms/prompt, production-ready (CPU only) |

→ **본 session 영역 12 SUB 완료**, 13+ commit, 본 fork code 14 줄, 종합 리포트 391+ lines.

---

## §9. raw 자료 link

### 9.1 idea / SUB doc

| doc | 위치 |
|---|---|
| **본 종합 리포트** | (이 파일) |
| idea backlog | [`idea/README.md`](idea/README.md) |
| idea evaluation 종합 | [`idea/evaluation_summary_20260524.md`](idea/evaluation_summary_20260524.md) |
| code base impact | [`idea/code_base_impact_20260524.md`](idea/code_base_impact_20260524.md) |
| phase execution summary | [`idea/phase_execution_summary_20260524.md`](idea/phase_execution_summary_20260524.md) |
| IDE_009~014 | `idea/IDE_009_*.md` ~ `idea/IDE_014_*.md` |
| Best doc | [`Best_SpecDecode_10778tps.md`](Best_SpecDecode_10778tps.md) |
| INDEX (nav) | [`INDEX.md`](INDEX.md) |
| analysis (workload acceptance) | [`analysis/workload_acceptance_analysis_20260524.md`](analysis/workload_acceptance_analysis_20260524.md) (680+ lines, 40 reference) |
| plan SUB_072~083 | `planning/SUB_072_*.md` ~ `planning/SUB_083_*.md` |

### 9.2 measurement RESULTS

| SUB | RESULTS |
|---|---|
| SUB_044 | `measurements/sub044_spec_decode_20260523/RESULTS.md` |
| SUB_047 | `measurements/sub047_t3_3run_verify_20260523/RESULTS.md` |
| SUB_071 | `measurements/sub071_workload_large_20260524/RESULTS.md` |
| SUB_074 | `measurements/sub074_suffix_20260524/RESULTS.md` |
| SUB_075 | `measurements/sub075_acceptance_20260524/RESULTS.md` |
| SUB_076 | `measurements/sub076_classifier_20260524/RESULTS.md` |
| SUB_077 | `measurements/sub077_pr_draft_20260524/PR_DRAFT.md` |
| SUB_078 | `measurements/sub078_repro_20260524/RESULTS.md` |
| SUB_079 | `measurements/sub079_small_model_full_20260524/RESULTS.md` |
| SUB_080 | `measurements/sub080_gating_prod_20260524/RESULTS.md` |
| SUB_081 | `measurements/sub081_suffix_cuda_graph_20260524/RESULTS.md` |
| SUB_082 | `measurements/sub082_routing_20260524/RESULTS.md` |
| SUB_083 | `measurements/sub083_topm_20260524/RESULTS.md` |
| SUB_084 | `measurements/sub084_arctic_next_blocker_20260524/RESULTS.md` |
| **SUB_085** | **[`measurements/sub085_suffix_piecewise_20260525/RESULTS.md`](measurements/sub085_suffix_piecewise_20260525/RESULTS.md)** ⭐ |
| **SUB_086** | **[`measurements/sub086_vanilla_gmu080_20260525/RESULTS.md`](measurements/sub086_vanilla_gmu080_20260525/RESULTS.md)** |
| **SUB_087** | **[`measurements/sub087_ngram_piecewise_20260525/RESULTS.md`](measurements/sub087_ngram_piecewise_20260525/RESULTS.md)** |
| **SUB_088** | **[`measurements/sub088_small_suffix_20260525/RESULTS.md`](measurements/sub088_small_suffix_20260525/RESULTS.md)** |

### 9.3 raw eval results

`eval/results/20260525_*` 디렉토리:
- `*_sub085_*_suffix_piecewise/` (v1) + `*_sub085v2_*_suffix_piecewise/` (v2)
- `*_sub086_*_vanilla_gmu080/` × 3
- `*_sub087_*_ngram_piecewise/` × 3
- `*_sub088_qwen*_*_suffix/` × 6

### 9.4 본 fork code 변경

- `vllm/utils/__init__.py` (SUB_081)
- `vllm/engine/arg_utils.py` (SUB_084)
- `git diff origin/main -- vllm/v1/spec_decode/ngram_proposer.py` (SUB_047~SUB_067)

### 9.5 외부 script (`/tmp/`)

- `run_workload_gen.py` — main wrapper (Llama-70B + TP=8, all env-tunable)
- `run_sub078_wrap.py` — small model wrapper
- `workload_classifier.py` — IDE_012 classifier
- `run_sub080_router.py` — SUB_080 analytical router
- `run_sub085_suffix_piecewise_v2.sh` — best config launcher
- `run_sub086_vanilla_gmu080.sh` — fair baseline launcher
- `run_sub087_ngram_piecewise.sh` — ngram fair launcher
- `run_sub088_small_suffix.sh` — small model + suffix launcher
- `run_sub089_sonnet_3run.sh` — variance launcher (진행 중)

---

## §10. 본 session 핵심 메시지 (한 문장씩)

1. **본 fork 의 진짜 best = suffix decoding + cudagraph PIECEWISE + gpu_memory_utilization=0.80** (3 workload 모두 fair vanilla 대비 net positive).
2. **fair contribution = sonnet +50.3% / chat +63.8% / code +18.9%** (이전 "+134%" 영역 wrapper-historical noise 영역 정직 정정).
3. **code workload 영역 ngram −20.7% 회귀 → suffix +18.9% net positive 영역 완전 mitigation** (suffix 영역 prompt+generation 양쪽 pool 영역 K 영역 ngram 영역 3.74× 향상).
4. **small model (1B 이하) 영역 모든 spec method 영역 fundamentally fail** (6/6 + 6/6 cell 모두 회귀, R≫K constraint). production 영역 vanilla 만.
5. **본 fork 영역 vLLM core 변경 = 14 줄** (utils +5, arg_utils +9, both backward-compat 100%). wrapper 영역 env-tunable +10 줄 정도. 최소 변경 영역 큰 production gain.
6. **이전 SUB_084 의 "fundamental architectural incompat 영역 dead-end" 결론 영역 잘못됨** — `compilation_config={"cudagraph_mode": "PIECEWISE"}` 단 한 줄 영역 우회 가능 영역 SUB_085 영역 입증.
7. **workload generalization framework (R/K + workload type + model size)** 영역 본 doc 영역 정량 + empirical validation 완료.
8. **남은 large lever**: R/K model-size sweep (SUB_090, boundary 7B vs 32B 확정), router HTTP server PoC (SUB_092, Phase 1 actual deploy).
