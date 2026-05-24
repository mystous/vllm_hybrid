# SUB_084 — Phase 2 follow-up: arctic_inference next blocker 해소 시도 (fundamental incompat 확정)

> **parent**: TSK_020 / SUB_081 (Phase 2)
> **measurement**: 2026-05-24 23:18 KST
> **status**: ◐ **부분 (1 fix 완료, fundamental incompat 확정)**

---

## 1. 진행

### Step 2.1 — `EngineArgs._is_v1_supported_oracle` stub 추가 (✅ 완료)

본 fork `vllm/engine/arg_utils.py` 영역 +9 줄 stub method 추가:

```python
# SUB_084 (Phase 2 follow-up): no-op stub for arctic_inference 0.1.2 compat.
# vLLM 0.11.0 had `_is_v1_supported_oracle` to gate v0/v1 selection.
# 본 fork 1.6 영역 v1-only 이므로 그 method 영역 자체 영역 의미 없음 (deprecated).
# arctic_inference 의 EngineArgsPatch 가 본 method 영역 wrap 영역 의존
# → 항상 True 반환 stub 영역 추가 (호출되어도 v1 진행).
def _is_v1_supported_oracle(self, *args, **kwargs):
    return True
```

→ arctic plugin Step 2 의 `EngineArgsPatch` 영역 통과 ✓.

### Step 2.2 — 다음 blocker: `vllm.attention.layer.Attention` 부재 (❌ fundamental incompat)

```
File ".../arctic_inference/vllm/ulysses.py", line 25, in <module>
    from vllm.attention.layer import Attention
ModuleNotFoundError: No module named 'vllm.attention'
```

본 fork 1.6 영역 attention namespace:
- ✓ `vllm/v1/attention/` (v1 전용 — AttentionBackend abstract + per-backend implementation)
- ✓ `vllm/model_executor/layers/attention/` (model-side)
- ✗ `vllm/attention/` (vLLM 0.11.0 영역 high-level Attention nn.Module 영역 부재)

`class Attention` 영역 본 fork 영역 fundamentally **다른 architecture**:
- vLLM 0.11.0: `vllm/attention/layer.py:Attention(nn.Module)` — model-side에서 attention 영역 wrap 영역 single high-level class
- 본 fork 1.6: `vllm/v1/attention/backend.py:AttentionBackend(ABC)` + per-backend Implementation (FlashAttn, FlashInfer 등) — v1 transition 영역 design 변경

→ arctic_inference 의 `ulysses.py` 영역 본 class 영역 monkey-patch 영역 본 fork 영역 single-line stub 영역 해소 불가능. **fundamental architectural incompat 확정**.

## 2. 결론 — Phase 2 영역 single-session 영역 dead-end

| 시도 | 결과 |
|---|---|
| SUB_081 Step 1: FlexibleArgumentParser re-export | ✅ |
| SUB_084 Step 2.1: _is_v1_supported_oracle stub | ✅ |
| SUB_084 Step 2.2: vllm.attention.layer.Attention | ❌ fundamental incompat |
| (예상) ulysses 외 다른 patch (model_runner, config, structured_output 등) | ❌ 다수 fundamental incompat 예상 |

→ **arctic_inference v0.1.2 + 본 fork vLLM 1.6 binary compat 영역 single-session fork-side patches 영역 불가**. fundamental architecture 영역 다름 (v1 transition).

## 3. 본 fork 영역 SuffixDecoding 사용 path — SUB_074 영역 그대로 (변경 없음)

본 fork 영역 SUB_074 영역 이미 검증된 path:
- `ARCTIC_INFERENCE_ENABLED=0 + VLLM_PLUGINS=""` 영역 plugin disable
- 본 fork `vllm/v1/spec_decode/suffix_decoding.py` 의 SuffixDecodingProposer 영역 직접 사용 (lazy import `arctic_inference.suffix_decoding.SuffixDecodingCache` only)
- `VLLM_ENFORCE_EAGER=1` 영역 cuda graph 우회

→ **SUB_074 영역 enforce_eager 결과 영역 본 fork 영역 best-known suffix configuration**:
- code: 7,094 tps (vs vanilla +1.85%, vs ngram +32%)

cuda graph 호환 영역 위 fundamental incompat 영역 다음 path:
- (a) arctic_inference v0.2 (vLLM 1.6 native 지원) release 대기 — 가장 안전, ETA 불명
- (b) 본 fork 영역 minimal suffix decoding 영역 native 구현 — `vllm/v1/spec_decode/suffix_decoding.py` 영역 cuda graph 호환 patch (1-2 일 effort)
- (c) arctic_inference 영역 본 fork 영역 binary compat fork — fundamental 영역 large effort (1 주+, vLLM 0.11.0→1.6 transition 영역 모두 sequential 해소)

## 4. 본 fork 영역 추가 변경 정리

| file | 변경 | 라인 | 용도 |
|---|---|---:|---|
| `vllm/utils/__init__.py` (SUB_081) | FlexibleArgumentParser re-export | +5 | arctic plugin Step 1 unblock |
| **`vllm/engine/arg_utils.py` (SUB_084)** | **_is_v1_supported_oracle stub** | **+9** | **arctic plugin Step 2 unblock** |
| (다음 blocker) `vllm/attention/__init__.py` 신설 | (불가능 — fundamental architecture diff) | — | — |

→ **본 session 영역 추가 변경 = 14 줄 (2 file)**. 모두 backward-compat (default behavior 영역 영향 0).

## 5. raw 자료

| 항목 | 위치 |
|---|---|
| stub patch | `vllm/engine/arg_utils.py` (SUB_084 marker) |
| smoke test stdout | `eval/results/20260525_081809_sub081_smoke_sonnet/engine.log.stdout` (Step 2.2 blocker 확인) |
| 본 SUB plan | (inline — SUB_081 plan 의 후속) |

## 6. 후속 — Phase 2 진정한 unblock path

- 우선 (b) "본 fork minimal suffix decoding native 구현" — effort 1-2 일, single-session 영역 가능, SUB_085+ candidate
- 또는 (a) arctic_inference v0.2 wait — passive
- (c) 영역 1 주+ 영역 effort 영역 본 환경 영역 가치 대비 낮음
