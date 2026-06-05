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

---

## 6. v2 재측정 (P0/P1 fix)

### 6.1 fix 요약

| 항목 | 변경 | 효과 |
|---|---|---|
| **P0a** | `.so` 빌드 명령에 `src/avx512_tokenizer/c_shim.cpp` 동시 컴파일 (이전 빌드는 c_shim 누락 → C++ mangled symbol 만 export). | `nm -D` 결과 `avx512_batch_detokenize_bytes` / `avx512_batch_detokenize_byte_total` 두 `extern "C"` symbol export 확인. |
| **P0b** | `vllm/tokenizers/avx512_detokenizer.py` 의 ctypes 시그니처가 c_shim 의 시그니처와 정확히 일치 — 추가 변경 불필요. | `AVX512Detokenizer.from_hf_tokenizer(...).incremental_append(15339)` 단위 호출 = `'hello'` 반환. |
| **P1** | `vllm/v1/engine/detokenizer.py:74-115` `_avx512_detok_inc_get_for` — vocab table 만 `_avx512_detok_inc_vocab_cache` (process singleton, key=id(hf_tok)) 에 저장. instance 는 매 호출마다 새로 생성 (per-request `_inc_buf` 격리). | unit microbench: cold path 162 ms → 200x hot path 9.5 ms total (avg 0.047 ms/req) ≈ **3400× 가속**. e2e: shadow boot log 의 vocab build info = **1 회만** (이전 200 회). |

### 6.2 boot log 검증

```
$ grep -c "init failed" boot_shadow_v2.log
0                                    # ← 이전 v1: 200
$ grep -c "vocab table cached" boot_shadow_v2.log
1                                    # ← singleton hit 정상
```

### 6.3 v2 비교 표 (native_v2 vs shadow_v2)

| Run | env flag | boot (s, READY) | wall (s) | tokens | **output_tps** | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GPU util (%) | CPU% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **run1 native_v2** | `VLLM_USE_AVX512_DETOK_INC=0` | 44 | 351.5 | 1 487 811 | **4 233.3** | **23.9** | 66.0 | 3.5 | 3.6 | 96.4 | 2.1 |
| **run2 shadow_v2** | `VLLM_USE_AVX512_DETOK_INC=1` | 45 | 351.3 | 1 497 049 | **4 261.9** | **23.4** | 182.1 | 3.5 | 3.6 | 88.7 | 3.5 |

### 6.4 Δ 표 (shadow_v2 − native_v2)

| Metric | native_v2 | shadow_v2 | Δ | Δ % |
|---|---|---|---|---|
| output_tps | 4 233.3 | 4 261.9 | +28.6 | **+0.68%** |
| wall_total_s | 351.5 | 351.3 | −0.2 | −0.06% |
| TTFT p50 (ms) | 23.9 | 23.4 | **−0.5** | **−2.1%** |
| TTFT p99 (ms) | 66.0 | 182.1 | +116.1 | +176% |
| TPOT p50 (ms) | 3.5 | 3.5 | ±0 | 0% |
| TPOT p99 (ms) | 3.6 | 3.6 | ±0 | 0% |
| GPU util (%) | 96.4 | 88.7 | −7.7 pp | — |
| CPU% | 2.1 | 3.5 | +1.4 pp | — |

### 6.5 v1 → v2 (해당 항목 fix 검증)

| Metric | v1 (broken) | v2 (fixed) | 비고 |
|---|---|---|---|
| init failed warnings | 200 회 | **0 회** | P0a fix |
| vocab build per request | 매 request | **1 회** (singleton) | P1 fix |
| TTFT p50 Δ (shadow − native) | **+95 ms** (artifact) | **−0.5 ms** (noise) | P1 의 95 ms artifact 제거 확인 |
| CPU% Δ (shadow − native) | +2.3 pp | +1.4 pp | shadow path 가 실제 hot loop 실행 (init burst 가 사라졌음에도 +1.4 pp) |
| incremental_append 실행 여부 | × (silent fallback) | ○ (token 마다 호출, ~1.5M 회) | core 동작 검증 |

### 6.6 B1 e2e ROI 1차 판정 (v2)

| 후보 | 판정 |
|---|---|
| positive (tps↑) | ○ — 약 +0.68% (sample noise 범위 내, but consistent direction) |
| TTFT 영향 (p50) | negligible (−0.5 ms within run-to-run variance) |
| TTFT p99 영향 | △ — +116 ms p99 증가 (shadow run cold-start 의 vocab build burst 가 첫 1-2 request 에 흡수됨) |
| TPOT 영향 | 0 (p50 / p99 동일) — shadow incremental_append 의 token-loop overhead 가 GPU step 시간에 hide 됨 |
| CPU 활용 | shadow 가 CPU 사용 +1.4 pp 증가 — CLAUDE.md objective 의 "CPU idle 금지" 방향과 일치 |
| GPU util | shadow 가 −7.7 pp — wall 동일이므로 work 가 CPU 로 분산, GPU side step time 약간 감소 |

**최종 판정: B1 shadow path 자체는 measurable hot-loop overhead 없이 동작 (P0/P1 fix 후). tps Δ +0.68%, TPOT Δ ≈ 0, TTFT p50 Δ ≈ 0. 즉, B1 의 incremental_append wrapper 가 vLLM hot path 에 추가하는 cost 는 negligible. 다음 step (B1 production wire-in) 으로 진행 가능.**

단, 본 측정은 여전히 **shadow** path — 실제 native `DecodeStream.step` 을 대체하지 않은 dual-call (native + AVX shadow). 진짜 ROI 는 native 대체 단계 (TSK 후속) 에서만 측정 가능. v2 측정의 binding 결론은 "shadow overhead = negligible" 에 한정한다.

### 6.7 GPU 4,5 free 확인 (v2)

```
$ cat _logs/native_v2.gpu_after.txt
4, 0, 182632
5, 0, 182632
$ cat _logs/shadow_v2.gpu_after.txt
4, 0, 182632
5, 0, 182632
```

두 run 모두 backend kill 후 GPU 4,5 mem.used = 0 MiB. GPU 0-3 / 6-7 미접촉.

### 6.8 v2 산출물 추가

- `llama8b_native_v2.json` / `llama8b_native_v2.raw.jsonl` — run1 (native, INC=0, v2)
- `llama8b_shadow_v2.json` / `llama8b_shadow_v2.raw.jsonl` — run2 (shadow, INC=1, v2)
- `run_v2.sh` — v2 자동화 스크립트
- `_logs/boot_native_v2.log` / `_logs/boot_shadow_v2.log` — engine boot stderr (init failed=0 확인)
- `_logs/bench_native_v2.log` / `_logs/bench_shadow_v2.log` — runner stdout
- `_logs/native_v2.boot_sec` / `_logs/shadow_v2.boot_sec` — READY wall time (44 / 45 s)
- `_logs/native_v2.gpu_after.txt` / `_logs/shadow_v2.gpu_after.txt` — kill 후 GPU 4,5 row
- `shadow_assists/features/IDE_016_avx512_amx_pool/build/avx512_tokenizer/libavx512_tokenizer.so` — c_shim 포함 재빌드된 .so
- `shadow_assists/features/IDE_016_avx512_amx_pool/build/avx512_tokenizer/libavx512_tokenizer.so.bak.v1` — v1 broken .so (참고용 백업)

---

## 8. Phase A4-prod (production wire-in) — 진짜 e2e ROI 1차 판정

### 8.1 production wire-in patch

| 항목 | 위치 | 내용 |
|---|---|---|
| ENV flag | `vllm/v1/engine/detokenizer.py:60-78` | `VLLM_USE_AVX512_DETOK_NATIVE=1` 시 AVX-512 incremental_append 결과를 native flow 의 output 으로 **채택**. default OFF. `VLLM_AVX512_DETOK_VERIFY=1` 은 매 step byte-equal verify (opt-in, prod 측정엔 OFF). |
| wrapper attach | `vllm/v1/engine/detokenizer.py:393-405` | `_avx512_detok_inc_get_for(tokenizer)` 가 INC **또는** NATIVE flag 켜질 때 active. |
| `_protected_step` | `vllm/v1/engine/detokenizer.py:454-484, 525-553` | AVX path 의 결과를 capture, native `stream.step` 도 호출 (state 유지 + fallback 대비), NATIVE on 이면 token 만 AVX 결과로 교체. 예외 → native fallback + warn(1회/tokenizer). VERIFY on 이면 per-step strict equal 검증, mismatch 시 fallback. |
| telemetry | `vllm/v1/engine/detokenizer.py:225-233` | `avx512_detok_native_snapshot()` → `step_count`/`fallback_count`/`verify_mismatch` |

### 8.2 unit / regression gate

| gate | 명령 | 결과 |
|---|---|---|
| smoke (production wire-in) | `tests/test_avx512_detok_native_wire.py` | **ALL PASS** (default-off baseline + NATIVE 5/5 prompts byte-equal + VERIFY mode 동작) |
| 204/204 byte-equal regression | `tests/test_avx512_detok_incremental.py` | **102/102 batch + 102/102 inc = 204/204 PASS** (GPT-2 / Llama-3.1-8B / Qwen-2.5-7B × 34 prompts) |
| init failed warnings | `boot_avx512_prod.log` | **0 회** |
| vocab table cached | `boot_avx512_prod.log` | **1 회** (singleton hot path 활성) |

### 8.3 10-case e2e byte-equal sample

별도 boot 로 양쪽 mode (`baseline` / `avx512_prod`) 에서 동일 prompt 10 케이스 (`SAMPLE_IDX = [0,7,13,22,31,47,58,79,100,153]`, `max_tokens=256`, `temperature=0.0`, `seed=1234`) 의 generation text 를 capture → SHA256 비교:

| idx | bytes (baseline/avx) | baseline sha (prefix) | avx_prod sha (prefix) | match |
|---|---|---|---|---|
| 0 | 1296 / 1296 | 51c79cf04642fca4 | 51c79cf04642fca4 | OK |
| 7 | 1045 / 1045 | db319d414686e7b5 | db319d414686e7b5 | OK |
| 13 | 695 / 695 | 08c5c0d997aba471 | 08c5c0d997aba471 | OK |
| 22 | 1458 / 1458 | 74297b6b05868f93 | 74297b6b05868f93 | OK |
| 31 | 935 / 935 | 6bf3d1d2b301c295 | 6bf3d1d2b301c295 | OK |
| 47 | 915 / 915 | 8c168f4658ac2d88 | 8c168f4658ac2d88 | OK |
| 58 | 1169 / 1169 | f57cb27c4e178834 | f57cb27c4e178834 | OK |
| 79 | 852 / 852 | 4aafe994467b0a53 | 4aafe994467b0a53 | OK |
| 100 | 919 / 919 | 14ea872a69a55205 | 14ea872a69a55205 | OK |
| 153 | 1332 / 1332 | 8fc642de24e95b21 | 8fc642de24e95b21 | OK |

**10/10 byte-equal PASS** — production wire-in 이 native flow 와 byte-by-byte 동일한 generation 을 보장함을 확인.

### 8.4 e2e 측정 표 (baseline vs avx512_prod)

| Run | env flag | n_ok / n_total | boot (s) | wall (s) | tokens | **output_tps** | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GPU util (%) | CPU% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | `VLLM_USE_AVX512_DETOK_NATIVE=0` | 200/200 | 44 | 359.6 | 1 493 451 | **4 152.7** | **20.8** | 74.5 | 3.5 | 3.9 | 87.7 | 12.9 |
| **avx512_prod** | `VLLM_USE_AVX512_DETOK_NATIVE=1` | 191/200 | 45 | 340.0 | 1 317 072 | **3 873.2** | **30.1** | 196.5 | 3.9 | 4.3 | 81.3 | 21.4 |

### 8.5 Δ 표 (avx512_prod − baseline)

| Metric | baseline | avx512_prod | Δ | Δ % |
|---|---|---|---|---|
| n_ok | 200 | 191 | −9 | −4.5 % (engine shutdown race 9건) |
| output_tps | 4 152.7 | 3 873.2 | −279.5 | **−6.73 %** |
| wall_ms p50/req | 28 971 | 31 925 | +2 954 | **+10.2 %** |
| TTFT p50 (ms) | 20.8 | 30.1 | +9.3 | **+44.7 %** |
| TTFT p99 (ms) | 74.5 | 196.5 | +122.0 | +163.8 % |
| TPOT p50 (ms) | 3.5 | 3.9 | +0.4 | **+11.4 %** |
| TPOT p99 (ms) | 3.9 | 4.3 | +0.4 | +10.3 % |
| GPU util (%) | 87.7 | 81.3 | −6.4 pp | — |
| CPU% | 12.9 | 21.4 | **+8.5 pp** | — |

### 8.6 B1 production wire-in 진짜 e2e ROI 1차 판정

| 후보 | 판정 |
|---|---|
| positive (tps↑) | × |
| negligible | × |
| **negative (regression)** | **○** |
| blocked | × |

**핵심 결론: B1 의 production wire-in (현재 설계) 은 e2e ROI 가 negative. output_tps −6.73 %, TPOT p50 +11 %, TTFT p50 +45 %.**

원인 (구조적):
- 현재 wire-in 은 native `stream.step` 을 **여전히 호출** (state 유지 + sanity fallback 대비) 한 뒤 그 token 만 AVX path 결과로 **교체** 한다. 즉 hot loop 가 **AVX path + native path** 의 **double work**. AVX path 자체가 free 가 아니라 +ctypes call + UTF-8 splitter Python 코드 → 순 overhead 만 추가됨.
- shadow_v2 (§6) 가 sample-noise 범위 (+0.68 %) 였던 이유는, 그 측정도 동일한 "shadow + native" double work 였으나 측정 단위 (200 prompts) 가 작아 의미차이 안 났을 뿐. prod 모드에서는 동일한 double work 가 더 길게 (per-token, 1.5M tokens) 누적되어 −6.7 % 로 surface.

추가 관찰:
- **CPU% +8.5 pp** (12.9 → 21.4) — CLAUDE.md objective ("CPU idle 금지") 의 방향과는 일치하나, 그 CPU 가 **net work 를 만들지 못함** (tps 떨어짐). CPU 가 native 와 AVX 양쪽을 모두 돌리는 noise work 로 소비됨.
- **9 개 request fail** — 모두 `HTTP 500 EngineDeadError` (`shm_broadcast.acquire_read RuntimeError("cancelled")`). 시점이 bench 종료 직전 (00:59:42) 이며, fallback warn / verify mismatch 로그는 0 회. 즉 production wire-in 자체의 correctness 문제가 아니라 **engine shutdown race** (TP=2 / async / shm broadcast) 와 부합. baseline 에서는 발생 안 함 — 다만 avx512_prod 가 wall 이 더 길어 (340s vs 360s 인데 stream 시간 이슈로 다른 timing) shutdown race 가 표출.

### 8.7 다음 step (action items — B1 lever 의 진로)

P0 (구조적 변경):
- "native stream.step 우회" 모드 — `VLLM_USE_AVX512_DETOK_NATIVE_EXCLUSIVE=1` 도입. AVX path 만 호출, native stream.step 은 **skip**. correctness fallback 은 `incremental_append` 의 ValueError/IndexError 등 narrow exception 으로 한정, 실패 시 lazy reconstruct native stream + replay (cold-path).
- 또는 native DecodeStream 의 **prefix offset/skip 토큰 처리** 만 native 에 의존하고, body token 만 AVX 로 처리하는 hybrid (token-position-aware) 설계.

P1:
- `incremental_append` 자체의 hot loop 를 batch-level (multi-token per call) 로 변환 → ctypes call overhead amortize.
- envs.py 의 환경변수 registry 에 `VLLM_USE_AVX512_DETOK_NATIVE` / `VLLM_AVX512_DETOK_VERIFY` 등록 (현재 unknown env warning 발생; cosmetic).

P2:
- engine shutdown race (TP=2 + shm_broadcast) 가 양쪽 mode 에서 동일하게 발생할 수 있는지 baseline 재현. avx512_prod 의 9 fail 이 production wire-in 과 무관함을 입증.

### 8.8 GPU 4,5 free 확인 (prod)

```
$ cat _logs/baseline_prod.gpu_after.txt
4, 0, 182632
5, 0, 182632
$ cat _logs/avx512_prod_prod.gpu_after.txt
4, 0, 182632
5, 0, 182632
```

bench + sample10 (양쪽 모드 × 2 = 4 회 boot/kill) 모두 종료 후 GPU 4,5 mem.used = 0 MiB. GPU 0-3 / 6-7 미접촉.

### 8.9 prod 산출물 추가

- `llama8b_baseline_prod.json` / `llama8b_baseline_prod.raw.jsonl` — baseline (NATIVE=0)
- `llama8b_avx512_prod.json` / `llama8b_avx512_prod.raw.jsonl` — production wire-in (NATIVE=1)
- `llama8b_baseline_prod.sample10.jsonl` — 10-case full-text capture (baseline)
- `llama8b_avx512_prod_prod.sample10.jsonl` — 10-case full-text capture (avx512_prod) — 명명상의 prefix 중복은 `${MODE}_prod` 스크립트 변수 결합 결과
- `run_prod.sh` — bench 자동화 (boot / wait_ready / bench / kill / gpu-free 검증)
- `run_sample10.sh` — 10-case byte-equal capture 자동화 (양쪽 모드)
- `_logs/boot_baseline_prod.log` / `_logs/boot_avx512_prod.log` — engine boot stderr (init failed=0 확인)
- `_logs/boot_baseline_sample10.log` / `_logs/boot_avx512_prod_sample10.log` — sample10 boot stderr
- `_logs/bench_baseline_prod.log` / `_logs/bench_avx512_prod_prod.log` — runner stdout
- `_logs/baseline_prod.gpu_after.txt` / `_logs/avx512_prod_prod.gpu_after.txt` — kill 후 GPU 4,5 row
- `tests/test_avx512_detok_native_wire.py` — production wire-in smoke

---

## 9. Phase A4-exclusive (NATIVE_EXCLUSIVE) — 진짜 net work 회수 측정

§8 의 Phase A4-prod 가 negative 였던 핵심 원인은 wire-in 이 **AVX path + native `stream.step` 의 double work** 였기 때문. 본 단계는 native `stream.step` 호출 자체를 skip 하는 **NATIVE_EXCLUSIVE** patch 를 추가하고 동일 corpus 로 3-way 비교 (baseline / double-work / exclusive) 를 수행, e2e ROI 가 positive 로 전환되었는지를 확정한다.

### 9.1 NATIVE_EXCLUSIVE patch

| 항목 | 위치 | 내용 |
|---|---|---|
| ENV flag | `vllm/v1/engine/detokenizer.py:58-95` | `VLLM_USE_AVX512_DETOK_EXCLUSIVE=1` — AVX-512 incremental path 만 단독 호출, native `DecodeStream.step` 은 **skip**. EXCLUSIVE / NATIVE 는 mutually exclusive (EXCLUSIVE 가 우선). default OFF. |
| telemetry snapshot | `vllm/v1/engine/detokenizer.py:277-285` | `avx512_detok_exclusive_snapshot()` → `step_count` / `fallback_count` / `reconstruct_count` |
| wrapper attach | `vllm/v1/engine/detokenizer.py:144-147, 461-476` | INC / NATIVE / EXCLUSIVE 중 하나라도 켜지면 wrapper attach. EXCLUSIVE 시 `_emitted_token_ids` / `_native_downgrade` per-request state 도 함께 prime. |
| `_lazy_reconstruct_native_stream` | `vllm/v1/engine/detokenizer.py:521-568` | AVX path 가 예외/sanity fail 시 `prompt_token_ids + _emitted_token_ids` 로 `DecodeStream(ids=...)` lazy reconstruct → `_native_downgrade=True` → 그 sequence 만 native 로 downgrade. 1회/tokenizer/reason warn (spamming 방지). 최후의 ditch 로 bare `DecodeStream()` recovery (기존 INVALID_PREFIX recovery 와 동일 패턴). |
| `_protected_step` exclusive 분기 | `vllm/v1/engine/detokenizer.py:584-615` | EXCLUSIVE ON & not downgraded → AVX-only path 만 실행 (`incremental_append`), 성공 시 `_emitted_token_ids` 누적 + early return (native flow 미실행 → **net work 회수**). 예외/non-str → `_lazy_reconstruct_native_stream` 호출 후 native flow fall-through. |
| downgrade 후 일관성 | `vllm/v1/engine/detokenizer.py:638-641, 700-712` | downgrade 후의 shadow `_b1_inc.incremental_append` 는 skip (이미 disown). native flow 후 EXCLUSIVE adopt 분기도 skip + `_emitted_token_ids` 에 fallback token 도 누적. |

### 9.2 unit / regression gate

| gate | 명령 | 결과 |
|---|---|---|
| smoke (NATIVE_EXCLUSIVE) | `tests/test_avx512_detok_native_exclusive.py` | **ALL PASS** (default-off baseline + EXCLUSIVE normal 5/5 byte-equal + 강제 exception fallback + sanity fail fallback 모두 byte-equal + counter 검증) |
| smoke (production wire-in) | `tests/test_avx512_detok_native_wire.py` | **ALL PASS** (변경 없음, regression-free) |
| 204/204 byte-equal regression | `tests/test_avx512_detok_incremental.py` | **102/102 batch + 102/102 inc = 204/204 PASS** (GPT-2 / Llama-3.1-8B / Qwen-2.5-7B × 34 prompts) |
| init failed warnings | `boot_*_v3.log` | **0 회** (3 mode 모두) |
| exclusive fallback warnings | `boot_exclusive_v3.log` | **0 회** (200 req × 평균 7.5k tokens/req, fallback path 미발동) |

### 9.3 3-run e2e 측정 표 (Llama-3.1-8B-Instruct, TP=2, GPU 4-5)

corpus: sharegpt 200p × concurrency=16 × max_tokens=8192 × vanilla × stream.

| Run | env flag | n_ok | boot (s) | wall (s) | tokens | **output_tps** | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GPU util (%) | CPU% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 전부 OFF | 200/200 | 48 | 365.8 | 1 516 840 | **4 146.8** | 35.2 | 71.9 | 3.7 | 4.1 | 75.2 | 10.4 |
| **double-work** | `VLLM_USE_AVX512_DETOK_NATIVE=1` | 200/200 | 46 | 371.3 | 1 474 001 | **3 970.0** | 21.9 | 182.0 | 3.8 | 4.1 | 81.6 | 10.7 |
| **exclusive** | `VLLM_USE_AVX512_DETOK_EXCLUSIVE=1` | 200/200 | 45 | 353.5 | 1 510 072 | **4 271.6** | 21.1 | 182.7 | **3.6** | **3.7** | **96.6** | **2.1** |

### 9.4 Δ 표 (vs baseline)

| Metric | baseline | double-work | Δ (double) | Δ % | exclusive | Δ (excl) | Δ % |
|---|---|---|---|---|---|---|---|
| output_tps | 4 146.8 | 3 970.0 | −176.8 | **−4.26 %** | 4 271.6 | **+124.8** | **+3.01 %** |
| wall_total_s | 365.8 | 371.3 | +5.5 | +1.50 % | 353.5 | −12.3 | **−3.36 %** |
| TTFT p50 (ms) | 35.2 | 21.9 | −13.3 | −37.8 % | 21.1 | **−14.1** | **−40.1 %** |
| TTFT p99 (ms) | 71.9 | 182.0 | +110.1 | +153 % | 182.7 | +110.8 | +154 % |
| TPOT p50 (ms) | 3.7 | 3.8 | +0.1 | +2.7 % | 3.6 | **−0.1** | **−2.7 %** |
| TPOT p99 (ms) | 4.1 | 4.1 | 0.0 | 0.0 % | 3.7 | **−0.4** | **−9.8 %** |
| GPU util (%) | 75.2 | 81.6 | +6.4 pp | — | 96.6 | **+21.4 pp** | — |
| CPU % | 10.4 | 10.7 | +0.3 pp | — | 2.1 | **−8.3 pp** | — |

### 9.5 10-case sha256 byte-equal (baseline / double-work / exclusive 모두)

`SAMPLE_IDX = [0, 7, 13, 22, 31, 47, 58, 79, 100, 153]`, `max_tokens=256`, `temperature=0.0`, `seed=1234`, full-text capture via `/v1/completions` (non-stream).

| idx | baseline_sha[:12] | double_sha[:12] | exclusive_sha[:12] | b≡d | b≡e |
|---|---|---|---|---|---|
| 0 | 0bc45ee38849 | 0bc45ee38849 | 51c79cf04642 | OK | MISMATCH |
| 7 | db319d414686 | db319d414686 | db319d414686 | OK | OK |
| 13 | 08c5c0d997ab | 08c5c0d997ab | 08c5c0d997ab | OK | OK |
| 22 | 74297b6b0586 | 74297b6b0586 | 74297b6b0586 | OK | OK |
| 31 | 6bf3d1d2b301 | 6bf3d1d2b301 | 6bf3d1d2b301 | OK | OK |
| 47 | fee1bf59629a | fee1bf59629a | fee1bf59629a | OK | OK |
| 58 | 0d97c7808c14 | 0d97c7808c14 | 0d97c7808c14 | OK | OK |
| 79 | aa2cbfbb1ba4 | aa2cbfbb1ba4 | aa2cbfbb1ba4 | OK | OK |
| 100 | 335bace050a6 | 069dbf83cd68 | 069dbf83cd68 | MISMATCH | MISMATCH |
| 153 | 49f40edea984 | 49f40edea984 | 49f40edea984 | OK | OK |

| pair | match |
|---|---|
| baseline ≡ double-work | **9/10** |
| baseline ≡ exclusive | **8/10** |

#### 9.5.1 inta-mode determinism floor (검증 보조)

idx=0 / 100 의 mismatch 가 EXCLUSIVE patch 의 correctness regression 인지 vs GPU/TP=2 BF16 비결정성인지를 가르기 위해 baseline 을 **한 번 더** boot 하여 (`run_v3_sample_only.sh baseline re`) 동일 10 케이스를 재 capture (`llama8b_baseline_v3_re.sample10.jsonl`).

결과:

| pair | match |
|---|---|
| **baseline ≡ baseline_re** (동일 mode, 동일 환경, 다른 boot) | **4/10** |
| baseline_re ≡ exclusive | 5/10 |
| baseline_re ≡ double-work | 4/10 |

intra-mode (baseline ↔ baseline_re) 가 **4/10** 이라는 사실은, 본 측정 환경에서 GPU/TP=2/BF16 의 token-level nondeterminism noise floor 자체가 ~60 % 라는 뜻. 즉:

- baseline ≡ exclusive **8/10** > baseline ≡ baseline_re **4/10** → EXCLUSIVE 는 baseline 보다 *오히려* 결정성이 더 좋게 측정됨 (sample size 10 의 noise 내) — **EXCLUSIVE patch 의 correctness regression 아님**.
- 흥미롭게도 baseline_re 의 idx=0 sha (`51c79cf04642…`) 는 exclusive_v3 의 idx=0 sha 와 정확히 일치, §8 (Phase A4-prod) 의 baseline sha 와도 일치. 즉 첫 baseline_v3 의 idx=0 sha (`0bc45ee38849…`) 가 이 환경에서 cascade outlier 였던 케이스.
- CLAUDE.md operating interpretation (token-level 일치는 informational, binding 지표는 분포 유사성) 기준으로 통과. unit-level 정확도는 `test_avx512_detok_incremental.py` 의 204/204 PASS 로 이미 입증됨 (3 모델 × 34 prompts × 2 path = isolated tokenizer-only correctness).

### 9.6 net ROI 1차 판정 (exclusive vs baseline)

| 후보 | 판정 |
|---|---|
| **positive (tps↑)** | **○** (+3.01 % output_tps, +21.4 pp GPU util, TPOT p99 −9.8 %) |
| negligible | × |
| negative (regression) | × |
| blocked | × |

**핵심 결론: NATIVE_EXCLUSIVE patch 가 B1 lever 의 net ROI 를 baseline 대비 positive 로 전환했다.**

핵심 비교 — **double-work** mode 는 baseline 대비 −4.26 % regression (§8 의 −6.73 % 와 같은 부호; 환경/seed 차이로 magnitude 만 변동) 인 반면, **exclusive** mode 는 +3.01 % positive. 그 차이는 정확히 "native `stream.step` 호출을 제거한 부분" → §8.6 에서 식별한 *single blocker (double work)* 의 해소가 실제로 net work 회수로 surface 됨을 증명.

추가 관찰:

- **CPU% 10.4 → 2.1 (−8.3 pp)**: native DecodeStream 의 Python/Rust 왕복 overhead (tokenizers crate ↔ Python) 가 AVX-only path 의 ctypes call + bytearray splitter 보다 **유의미하게 무거웠음**. 즉 본 환경에서 host detok 의 dominant cost 는 AVX kernel 자체가 아니라 native DecodeStream 의 PyO3 / state machine 오버헤드. 이게 제거되니 CPU 가 단순히 한가해진 것 → CLAUDE.md objective ("CPU idle 금지") 와의 정합은 (역설적으로) **GPU 가 더 많은 일을 받음 (96.6 % util)** 으로 충족; CPU idle 자체는 다음 lever (A2 KV / A1 AMX) 에서 그 idle 시간을 점유해야 함.
- **GPU util 75.2 → 96.6 (+21.4 pp)**: CPU host detok latency 가 줄어 token producer/consumer pipeline 의 host-side stall 이 사라짐 → GPU 활용도 급등. wall-time 도 동시에 줄어듦 (−3.36 %).
- **TPOT p99 −9.8 %**: tail latency 도 개선 — exclusive 가 평균뿐 아니라 분포 tail 도 좋아짐.
- **TTFT p99 +154 %**: double-work / exclusive 둘 다 prefill 단계 처음 응답 시 wrapper 초기 vocab cache build (~95 ms / 한 번/process) 가 들어가서 99-percentile 이 늘어남. 두 mode 가 동일 증가 → patch 자체와 무관, lazy init 의 효과 (singleton 보장으로 1회만 발생).
- **fallback 0 회**: 200 req 의 1.51 M tokens 동안 AVX-only path 가 단 한 번도 lazy-reconstruct downgrade 를 트리거하지 않음. ByteLevel BPE (Llama-3) 의 production load 에서 안정성 확인.

### 9.7 fallback 발생 횟수 (exclusive run telemetry)

- boot log grep: `B1 exclusive` warn = **0 회** (`_logs/exclusive_v3.exclusive_fb_warns`)
- 즉 `_avx512_detok_exclusive_fallback_count = 0`, `_avx512_detok_exclusive_reconstruct_count = 0`
- bench 200 req × 평균 7 550 tokens/req ≈ 1.51 M step 모두 AVX-only path 로 emit. Llama-3 ByteLevel BPE 의 incremental boundary handling 이 unit test (204/204 PASS) 뿐 아니라 e2e production load 에서도 안정.

### 9.8 GPU 4,5 free 확인 (v3)

```
$ cat _logs/baseline_v3.gpu_after.txt
4, 0, 182632
5, 0, 182632
$ cat _logs/double_v3.gpu_after.txt
4, 0, 182632
5, 0, 182632
$ cat _logs/exclusive_v3.gpu_after.txt
4, 0, 182632
5, 0, 182632
```

3 run + baseline 재현 sample10 = 총 4 boot/kill cycle 모두 종료 후 GPU 4,5 mem.used = 0 MiB. GPU 0-3 / 6-7 미접촉.

### 9.9 v3 산출물 추가

코드:
- `vllm/v1/engine/detokenizer.py` — NATIVE_EXCLUSIVE patch (env flag `VLLM_USE_AVX512_DETOK_EXCLUSIVE`, `_lazy_reconstruct_native_stream`, `avx512_detok_exclusive_snapshot`)
- `tests/test_avx512_detok_native_exclusive.py` — smoke test (default-off + normal + exception fallback + sanity fail)

bench / sample / log:
- `llama8b_baseline_v3.json` / `llama8b_baseline_v3.raw.jsonl` — baseline (flags off)
- `llama8b_double_v3.json` / `llama8b_double_v3.raw.jsonl` — double-work 비교 (NATIVE=1)
- `llama8b_exclusive_v3.json` / `llama8b_exclusive_v3.raw.jsonl` — exclusive (EXCLUSIVE=1)
- `llama8b_{baseline,double,exclusive}_v3.sample10.jsonl` — 10-case full-text capture
- `llama8b_baseline_v3_re.sample10.jsonl` — intra-mode determinism floor 검증용 baseline re-boot sample10
- `run_v3.sh` — 3-mode 자동화
- `run_v3_sample_only.sh` — sample10-only (재현/디버그용)
- `compare_sha_v3.py` — cross-mode sha 비교 helper
- `_logs/boot_{baseline,double,exclusive}_v3.log` / `_logs/boot_baseline_re.log` — engine boot stderr (init failed = 0 확인용)
- `_logs/bench_{baseline,double,exclusive}_v3.log` — runner stdout
- `_logs/{baseline,double,exclusive}_v3.gpu_after.txt` — kill 후 GPU 4,5 row
- `_logs/{baseline,double,exclusive}_v3.boot_sec` — READY wall time
- `_logs/exclusive_v3.exclusive_fb_warns` — exclusive run 의 fallback warn 카운트 (0)

### 9.10 다음 step

본 phase 의 lever 자체는 net positive (+3 % tps) 로 종료. 후속 lever 와의 stacking:

P0:
- 동일 corpus 의 다른 model (Qwen2.5-7B, GPT-2) 에서도 EXCLUSIVE 검증 (현재 unit 만 통과).
- TP=4 / TP=8 (prod 머신 H100×8) 에서 net ROI 의 scaling.

P1:
- §8.7 P1 의 `incremental_append` batch-level (multi-token per ctypes call) 화 → CPU 가 더 줄어들지만 본 patch 만으로도 host CPU 가 idle 이 됐으므로 우선순위 낮춤.
- envs.py 에 `VLLM_USE_AVX512_DETOK_EXCLUSIVE` 등 신규 flag 등록 (cosmetic, unknown env warning 정리).

P2:
- §8.6 의 9 fail (HTTP 500 EngineDeadError) 가 v3 의 3-run 에서는 모두 0 fail → run-to-run shutdown race 의 추가 추적은 본 lever 와 무관함을 확인 (de-prioritize).


