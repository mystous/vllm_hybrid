# B1 AVX-512 detok lever — Phase A4 e2e shadow MEASUREMENTS (Llama-3.1-8B)

- **Hardware**: NVIDIA RTX 3090 × 2 (GPU 4,5; 다른 agent 의 70B 측정과 격리)
- **Model**: meta-llama/Llama-3.1-8B-Instruct, TP=2, max-model-len=16384, gpu-mem-util=0.85
- **Method**: vanilla (no speculative decoding)
- **Workload**: sharegpt corpus, 200 prompt × concurrency=16, max_tokens=8192, stream
- **vLLM version**: `v1.7.dev16107+gffe20fb09.d20260601` (HEAD `2ab5233af` — B1 incremental wrapper 통합)
- **Runner**: `vllm_config_perf/gating/realistic_eval/throughput_runner.py`
- **Boot template**:
  ```
  CUDA_VISIBLE_DEVICES=4,5 VLLM_USE_AVX512_DETOK_INC=<0|1> \
    setsid vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 --port 8002 \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --allow-deprecated-quantization
  ```
- **Date**: 2026-06-05

---

## 1. 두 run 비교 (native vs shadow)

| Run | env flag | boot (s, READY) | wall (s) | tokens | **output_tps** | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GPU util (%) | CPU% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **run1 native** | `VLLM_USE_AVX512_DETOK_INC=0` | 63 | 354.5 | 1 503 747 | **4 242.0** | **22.7** | 261.1 | 3.6 | 3.7 | 96.1 | 2.2 |
| **run2 shadow** | `VLLM_USE_AVX512_DETOK_INC=1` | 45 | 353.5 | 1 491 793 | **4 219.8** | **117.7** | 1 751.0 | 3.5 | 3.6 | 83.0 | 4.5 |

### 1.1 Δ 표

| Metric | native | shadow | Δ (shadow − native) | Δ % |
|---|---|---|---|---|
| output_tps | 4 242.0 | 4 219.8 | −22.2 | **−0.52%** |
| wall_total_s | 354.5 | 353.5 | −1.0 | −0.28% |
| TTFT p50 (ms) | 22.7 | 117.7 | **+95.0** | **+418%** |
| TTFT p99 (ms) | 261.1 | 1 751.0 | +1 489.9 | +570% |
| TPOT p50 (ms) | 3.6 | 3.5 | −0.1 | −2.8% |
| TPOT p99 (ms) | 3.7 | 3.6 | −0.1 | −2.7% |
| GPU util (%) | 96.1 | 83.0 | −13.1 pp | — |
| CPU% | 2.2 | 4.5 | +2.3 pp | — |

---

## 2. 핵심 발견: shadow path 가 **실제로 실행되지 않음** (init 실패)

shadow run 의 `boot_shadow.log` 에서 `AVX512Detokenizer init failed` warning 이 **200회** 발생 (200 request 각각 per-request init 실패):

```
WARNING [detokenizer.py:89] SUB_201 B1 PoC (inc): AVX512Detokenizer init failed (
  /workspace/host_vllm_hybrid/shadow_assists/features/IDE_016_avx512_amx_pool/build/
  avx512_tokenizer/libavx512_tokenizer.so: undefined symbol: avx512_batch_detokenize_bytes)
```

원인 — `.so` symbol mismatch:

```
$ nm -D libavx512_tokenizer.so | grep -E "T |U "
0000000000001260 T _ZN15vllm_hybrid_tok29batch_detokenize_bytes_avx512ERKNS_10VocabTableEPKiS4_iPhPiS6_
0000000000001140 T _ZN15vllm_hybrid_tok29batch_detokenize_bytes_scalarERKNS_10VocabTableEPKiS4_iPhPiS6_
0000000000001630 T _ZN15vllm_hybrid_tok27batch_detokenize_byte_totalERKNS_10VocabTableEPKii
0000000000001730 T _ZN15vllm_hybrid_tok25batch_bpe_min_rank_avx512EPKiiS1_iiPi
```

- `.so` 는 **C++ mangled symbol** 만 export (namespace `vllm_hybrid_tok`).
- `vllm/tokenizers/avx512_detokenizer.py:139` 의 `self._lib.avx512_batch_detokenize_bytes` 는 **C-style symbol** 을 기대.
- 따라서 `from_hf_tokenizer` → `__init__` → `_configure_signatures()` 에서 `AttributeError` → wrapper 의 try/except 가 silent fallback → **incremental_append 한 번도 실행 안 됨**.

결과적으로 shadow run 의 실제 동작은:
1. `OutputProcessor.add_request` 매 호출에서 wrapper 인스턴스 생성 시도 (build_vocab_table — Llama-3 vocab 128 k entries 빌드, 매 request) → 실패 → warning emit.
2. `_protected_step` 에서 `_avx512_detok_inc` 가 None → 분기 미진입 → token-loop overhead = 0.

→ **본 측정의 Δ tps (−0.5%), Δ TTFT p50 (+95 ms)** 는 **shadow path 의 진짜 overhead 가 아니라**:
- per-request 200회의 vocab-table build (128 k × convert_ids_to_tokens) → CPU 증가 (2.2 → 4.5%) 와 TTFT 증가 (warm-up phase 의 init burst) 의 합.
- TPOT 가 native 대비 살짝 **더 낮은** 것 (3.6 → 3.5 ms) 은 sample noise 범위.

---

## 3. B1 e2e ROI 1차 판정

| 후보 | 판정 |
|---|---|
| positive | × |
| negligible | × |
| negative | × |
| **blocked — measurement invalid** | **○** |

이유: shadow path 실제 실행 = 0 회 → 본 측정은 B1 의 e2e overhead 를 정량화하지 **못함**. TTFT p50 의 +95 ms 는 의도하지 않은 per-request vocab-table build 의 효과이며, 실제 incremental_append × N_tokens 의 hot-loop overhead 는 별도 측정 필요.

단, **하한선 1 가지 결론**:
- shadow path 가 실제로 호출되지 않는 경로 자체가 native flow 의 hot-path 에 추가한 분기 (`getattr(self, "_avx512_detok_inc", None)` + `if _b1_inc is not None:` = None 분기) 만으로는 **TPOT p50 변화 ≤ 0.1 ms** (sample noise 내). 즉 hook 자체의 cold-cost 는 negligible.

---

## 4. 다음 step (action items)

| 우선순위 | 항목 | 위치 |
|---|---|---|
| **P0 (blocker)** | `.so` 에 `extern "C"` wrapper 추가 + 재빌드 — `avx512_batch_detokenize_bytes` C symbol export. `incremental_append` 는 **Python-side bytearray 누적** 이므로 .so 변경 불필요. | `shadow_assists/features/IDE_016_avx512_amx_pool/build/avx512_tokenizer/` |
| P0 | 재빌드 후 본 sweep (native vs shadow) 재실행 → 진짜 incremental_append × N_tokens overhead 측정. | 본 디렉토리 `run.sh` 재실행. |
| P1 | per-request `from_hf_tokenizer` 호출 (vocab-table 128 k 빌드) 가 TTFT 에 +95 ms 영향 — singleton 캐시 필요 (`_avx512_detok_inc_get_for` 의 `key = id(hf_tok)` 와 별도 cache dict). | `vllm/v1/engine/detokenizer.py:74-91` |
| P2 | TTFT p99 = 1 751 ms 는 init burst 의 첫 1-2 request 가 hold up — singleton 캐시 적용 시 함께 해결. | (위와 동일) |

---

## 5. GPU 4,5 free 확인

native run 종료 후:
```
4, 0, 182632
5, 0, 182632
```
shadow run 종료 후:
```
4, 0, 182632
5, 0, 182632
```
두 run 모두 backend kill 후 pgroup 정리 + orphan VLLM::Worker 정리 → GPU 4,5 mem.used = 0 MiB 확인 완료. **GPU 0-3 / 6-7 미접촉**.

---

## 6. 산출물

- `llama8b_native.json` — run1 (native, INC=0) summary
- `llama8b_native.raw.jsonl` — run1 per-request raw
- `llama8b_shadow.json` — run2 (shadow, INC=1) summary (단, 위 §2 caveat 반드시 동반 인용)
- `llama8b_shadow.raw.jsonl` — run2 per-request raw
- `run.sh` — boot + bench + kill 자동화 스크립트
- `_logs/boot_native.log` — engine boot stderr
- `_logs/boot_shadow.log` — engine boot stderr (warning floods 포함)
- `_logs/bench_native.log` / `_logs/bench_shadow.log` — runner stdout
- `_logs/native.boot_sec` / `_logs/shadow.boot_sec` — READY wall time (63 / 45 s)
- `_logs/native.gpu_after.txt` / `_logs/shadow.gpu_after.txt` — kill 후 GPU 4,5 row
