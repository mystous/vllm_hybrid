# SUB_082 — Phase 3: workload-aware routing 통합 (analytical viability)

> **parent**: TSK_020 (성능 향상 plan Phase 3)
> **plan**: [`../../planning/SUB_082_workload_aware_routing_integrated.md`](../../planning/SUB_082_workload_aware_routing_integrated.md)
> **measurement**: 2026-05-24 22:42 KST, **viability analytical (실제 dual instance init 영역 본 session 범위 초과)**
> **status**: ◐ **analytical viable, PoC 후속**

---

## 1. dual instance viability analytical estimate

### 1.1 hardware budget

- H100 80GB × 8 GPU = **640 GB total VRAM**
- Llama-3.3-70B BF16 weights = ~140 GB
- KV cache (max_num_seqs=256, fp8, max_model_len=16384, 70B GQA) = ~50-80 GB
- activations + buffers = ~10-20 GB

### 1.2 single instance (현 best, TP=8)

- weights / 8 GPU = 17.5 GB/GPU
- KV cache + activations 추가 영역 = ~30-50 GB/GPU (gmu=0.85)
- 실측 (SUB_047): GPU util 54.7% (compute-bound 영역 아닌 memory-bandwidth 영역)
- **VRAM 사용 영역 80 GB × 0.85 ≈ 68 GB/GPU**

### 1.3 dual instance TP=4 × 2 (가설)

- 각 instance: weights / 4 GPU = 35 GB/GPU
- KV cache (각 instance 영역 max_num_seqs=128, gmu=0.42 분할 가정) = ~15-25 GB/GPU
- **VRAM 사용 영역 ~50-60 GB/GPU** ← viable (80 GB 영역 한계 안)

→ **TP=4 × 2 instance 영역 GPU memory viable** (analytical). 단 실제 init 영역 multi-process 영역 large engineering work.

### 1.4 GPU 0/7 의 bentoml services (이미 6 GB / 1 GB 사용)

- GPU 0 영역 추가 6 GB 여유 → ngram instance 영역 GPU 0~3 할당 시 OK
- GPU 7 영역 추가 1 GB 여유 → suffix instance 영역 GPU 4~7 할당 시 OK

## 2. routing 구현 path 비교

### 2.1 Option A — dual vLLM instance (별도 process)

```bash
# Process 1: ngram instance (GPU 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0 \
    python vllm_serve --tp 4 --port 8000 --spec ngram_7

# Process 2: suffix instance (GPU 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
    ARCTIC_INFERENCE_ENABLED=1 \  # Phase 2 patch 완료 후
    python vllm_serve --tp 4 --port 8001 --spec suffix_32

# Router: HTTP server, classifier (IDE_012) 영역 분류 → port 8000 or 8001 forward
```

- **장점**: 즉시 구현 가능 (vLLM core 변경 없음). Phase 1 의 classifier 그대로 사용.
- **단점**:
  - TP=4 영역 single instance (현 SUB_047 TP=8) 대비 성능 -10~-20% 추정 (TP scaling 불완전)
  - Phase 2 (suffix cuda graph 호환) 영역 dependency
  - GPU 메모리 budget tight (gmu 영역 보수적 0.40 영역 KV cache 영역 제한)
- **effort**: 1-2 일 (multi-process orchestration + router HTTP)

### 2.2 Option B — vLLM upstream per-request spec override PR

```python
# vLLM 영역 core 변경 — SamplingParams 또는 generate() 영역 per-request spec override
llm.generate(
    prompts,
    sampling_params=[
        SamplingParams(...spec_config={"method": "suffix"} if is_code(p) else {"method": "ngram"})
        for p in prompts
    ]
)
```

- **장점**: single instance, 최대 throughput (TP=8 유지), 메모리 효율
- **단점**: vLLM core 변경 영역 large PR, review cycle 길음 (수 주)
- **effort**: 2-4 주 (vLLM scheduler + spec_decode + sampler 변경)

## 3. 본 session 진행 한계

| 항목 | 본 session 가능성 | 후속 |
|---|---|---|
| dual instance VRAM viability | ✅ analytical 확인 | SUB_084+ 영역 actual init test |
| router HTTP server 구현 | ◐ Python 단순 PoC 가능 | (Phase 2 완료 후 의미 있음) |
| dual vLLM serve init test | ✗ multi-process orchestration 영역 large effort | SUB_085+ (1-2 일) |
| upstream PR | ✗ vLLM core 변경 영역 large | SUB_086+ (2-4 주) |

## 4. expected gain (analytical, Phase 2 완료 가정)

Phase 1 의 mix scenario 결과 + Phase 2 의 suffix cuda graph 추정 결과 합산:

| mix | Phase 1 only (gating, code→vanilla) | Phase 1+2+3 (gating, code→suffix cuda graph) | 추가 gain |
|---|---:|---:|---:|
| M1 sonnet-heavy | 9,192 | 9,250 (code 5347→9094 가설) | +0.6% |
| M2 balanced | 7,977 | 8,500 | +6.6% |
| M3 code-heavy | 7,091 | **8,500** | **+19.9%** ⭐ |

→ Phase 3 routing 영역 Phase 1 단순 heuristic + Phase 2 suffix patch 결합 시 **code-heavy 영역 추가 +20%** 가능.

## 5. 본 session 산출물

- viability analytical: dual TP=4 × 2 instance 영역 memory budget viable 확인
- routing design doc (Option A / B 비교)
- expected gain 표 (Phase 1 + Phase 2 분리 vs 통합 시 비교)

## 6. 후속 SUB candidate

- SUB_085 (제안): dual TP=4 × 2 instance actual init test (Phase 2 완료 후)
- SUB_086 (제안): vLLM upstream per-request spec override PR draft
- SUB_087 (제안): router HTTP server PoC (classifier IDE_012 + Phase 1 gating)
