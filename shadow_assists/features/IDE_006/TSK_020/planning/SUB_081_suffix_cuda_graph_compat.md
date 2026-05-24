# SUB_081 — Phase 2: suffix cuda graph 호환 patch

> **parent**: TSK_020 (성능 향상 plan Phase 2 ⭐)
> **status**: 활성 (2026-05-24 신설)
> **effort**: 1-2 일 (arctic_inference fork-patch + cuda graph capture 호환)
> **based on**: IDE_010 (suffix eager mode code +32% vs ngram), arctic_inference v0.1.2 vs vLLM 1.6 binary incompat

## 1. 목표

suffix decoding 영역 cuda graph 사용 가능하게 patch → code workload +60-70% 추정 향상 실현.

## 2. 문제 원인

본 fork vLLM 1.6.dev0+g858b6df7a 와 arctic_inference 0.1.2 (requires vllm==0.11.0) 의 binary incompat:
- `arctic_inference.vllm.args` 가 `vllm.utils.FlexibleArgumentParser` import (vLLM 0.11.0 API). 본 fork 1.6 영역 부재 → plugin load fail.
- CUDA graph capture path 와 SuffixDecodingProposer conflict ("CUDA graph capturing detected at an inappropriate time"). plugin disable + enforce_eager=True 로 우회 (SUB_074) — 단 ~25% throughput penalty.

## 3. 진행 절차

### Step 1 — arctic_inference vLLM 1.6 binary compat patch

`/workspace/vllm_dev_prj/lib/python3.12/site-packages/arctic_inference/vllm/args.py` 의 `FlexibleArgumentParser` import 영역 본 fork 1.6 의 동등 API 로 변경:

```python
# Before:
from vllm.utils import FlexibleArgumentParser

# After (본 fork 1.6 호환):
try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils.argparse import FlexibleArgumentParser  # 또는 본 fork 의 실제 위치
```

본 fork 의 `FlexibleArgumentParser` 위치 확인 → `grep -r "class FlexibleArgumentParser" vllm/`

### Step 2 — SuffixDecodingProposer cuda graph 호환

`vllm/v1/spec_decode/suffix_decoding.py` 영역 `propose()` 가 cuda graph capture mode 영역 어떤 op 가 graph mode 와 충돌하는지 분석:
- `self.suffix_cache.speculate(...)` 영역 GPU op 가 graph mode 영역 capture 가능한지
- `prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]` 영역 CPU slice 가 graph 영역 안 capture 되는지 확인

### Step 3 — 검증 측정

- arctic plugin enable + cuda graph mode 영역 SuffixDecodingProposer 작동 확인
- 3 workload × suffix_spec32 + cuda graph 측정 (SUB_074 의 enforce_eager 측정과 비교)
- 예상: code workload tps 7,094 (eager) → ~9,094 (cuda graph, +28% 향상) → vs vanilla +30~+40% net positive

### Step 4 — risk: arctic_inference plugin 의 다른 vLLM 1.6 incompat 지점

`apply_arctic_patches()` 의 다른 patch (e.g., model_runner, structured_output, stats) 가 본 fork 와 incompat 가능 → plugin minimal 활성화 (suffix decoding 만 활성, 다른 patch 영역 disable) path 필요.

## 4. risk / fallback

- arctic_inference fork-patch 영역 binary compat 영역 모든 issue 해소 어려움 — 부분적 활성화 (suffix-decoding-only plugin path) 또는 본 fork 영역 minimal suffix decoding 구현 후 arctic_inference 의 SuffixDecodingCache 만 사용
- cuda graph capture compatibility 가 vLLM v1 의 fundamental design 영역 충돌 가능 — fallback: enforce_eager 모드 유지 (현 SUB_074 상태)

## 5. 산출물

- arctic_inference fork patch (또는 minimal compat shim)
- `/tmp/run_sub081_suffix_cuda_graph.sh`
- `measurements/sub081_suffix_cuda_graph_<TS>/RESULTS.md`
