# SUB_081 — Phase 2: suffix cuda graph 호환 patch (partial, blocked)

> **parent**: TSK_020 (성능 향상 plan Phase 2)
> **plan**: [`../../planning/SUB_081_suffix_cuda_graph_compat.md`](../../planning/SUB_081_suffix_cuda_graph_compat.md)
> **measurement**: 2026-05-24 22:41 KST, **partial — blocked by arctic_inference v0.1.2 vs vLLM 1.6 multi-layer incompat**
> **status**: ◐ **부분 진행 (1 fix 완료, 다음 blocker 노출)**

---

## 1. 진행 결과

### Step 1 — FlexibleArgumentParser re-export (✅ 완료)

본 fork `vllm/utils/__init__.py` 영역 SUB_081 patch (5 줄) 추가:

```python
# SUB_081 (Phase 2): Re-export FlexibleArgumentParser at vllm.utils for
# backward-compat with arctic_inference 0.1.2 (expects vllm 0.11.0 API path).
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: F401
except ImportError:
    pass
```

→ `from vllm.utils import FlexibleArgumentParser` 영역 본 fork 1.6 영역 작동 ✅.

### Step 2 — arctic plugin 활성화 시도 (❌ blocked)

`ARCTIC_INFERENCE_ENABLED=1 + ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1` 영역 plugin load 시도. Step 1 patch 영역 첫 번째 blocker (`FlexibleArgumentParser`) 해소되었으나, **다음 blocker 노출**:

```
File "/site-packages/arctic_inference/vllm/args.py", line 53, in EngineArgsPatch
    _orig_is_v1_supported_oracle = EngineArgs._is_v1_supported_oracle
AttributeError: type object 'EngineArgs' has no attribute '_is_v1_supported_oracle'
```

vLLM 0.11.0 영역 `EngineArgs._is_v1_supported_oracle` 메서드 가 본 fork 1.6 영역 부재 (vLLM v1 transition 영역 변경됨).

### Step 3 — 더 많은 incompat 예상

`arctic_inference/vllm/{args,config,model_runner,patches,plugin,structured_output,stats}.py` 영역 vLLM 0.11.0 API 광범위 의존. 본 fork 1.6 호환 영역 각 file 영역 missing API 영역 sequential 해소 필요 → **1-2 일 effort 추정** (본 session 범위 초과).

## 2. blocked path 의 정직 정리

| binary incompat 항목 | 상태 |
|---|---|
| `vllm.utils.FlexibleArgumentParser` | ✅ 해소 (본 fork side re-export) |
| `EngineArgs._is_v1_supported_oracle` | ❌ blocked |
| (예상) `vllm.config.*Patches`, `vllm.executor.*`, `vllm.spec_dec.*`, `vllm.stats.*` 등 | ❌ 다수 예상 |

**본 session 영역 불가**: arctic_inference v0.1.2 영역 vLLM 1.6 binary compat 영역 fork-side patches.

## 3. fallback — SUB_074 eager mode 결과 그대로 사용

본 SUB 영역 cuda graph 호환 patch 실패 → SUB_074 영역 측정값 (enforce_eager 모드) 그대로 best-known suffix 결과:

| workload | suffix (eager, SUB_074) | ngram (cuda graph, SUB_075) | suffix/ngram |
|---|---:|---:|---:|
| sonnet | 8,236 | 10,909 | 0.755 (-25% eager penalty) |
| chat | 2,370 | 2,972 | 0.797 (-20% eager penalty) |
| **code** | **7,094** | **5,362** | **1.323 (+32%)** ⭐ |

→ **code workload 영역 suffix eager 모드도 ngram 대비 +32% 향상 + vanilla 대비 +1.85%**. cuda graph 호환 시 +60-70% 가능했으나 본 session 영역 미실현.

## 4. 후속 — SUB_084+ candidate

| 후속 SUB | effort | scope |
|---|---|---|
| SUB_084 (제안) | 1-2 일 | arctic_inference v0.1.2 영역 본 fork 1.6 binary compat patch fork (모든 missing API sequential 해소) |
| 또는 | wait | arctic_inference v0.2 (vLLM 1.6 native 지원) release 대기 |
| 또는 | 1-2 일 | 본 fork 영역 minimal suffix decoding 구현 (arctic_inference 의 SuffixDecodingCache 만 import, vLLM-side proposer 영역 fork-native) |

## 5. 본 fork side 변경

- `vllm/v1/spec_decode/ngram_proposer.py`: 변경 없음
- **`vllm/utils/__init__.py`**: SUB_081 patch 5 줄 (FlexibleArgumentParser re-export) — **backward-compat 100%** (existing path 영향 0)

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| smoke test stdout | `eval/results/20260525_074129_sub081_smoke_sonnet/engine.log.stdout` |
| launcher | `/tmp/run_sub081_arctic_smoke.sh` |
| patch file | `vllm/utils/__init__.py` (SUB_081 마커 영역 5 줄) |
