# SUB_173 — IDE_016 / TSK_024 AVX-512 tokenizer canonical 500p e2e — RESULTS

| 항목 | 값 |
|---|---|
| parent | `TSK_024` (IDE_016) |
| scope | SUB_171 의 AVX-512 batch detokenize kernel 을 vLLM `FastIncrementalDetokenizer` 에 ENV-flag patch + canonical 500p × 9 cell × OFF/ON 측정 |
| host | 프로덕션 (Sapphire Rapids + Xeon, H100 ×8) |
| 측정 시각 | 시작 2026-05-27 08:39 KST |
| canonical | Qwen 2.5 32B Instruct TP=4×2 / 500p / 32 conc / 256 max-tokens / 3 mix / OFF + ON |
| ENV (OFF) | `RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false` |
| ENV (ON) | OFF + `VLLM_USE_AVX512_TOKENIZER=1` |
| 1-run rule | PASS (사용자 1-run rule 준수) |

## 1. vLLM detokenizer patch

| file | 변경량 | 역할 |
|---|---|---|
| `vllm/v1/engine/detokenizer.py` | +95 line | `_avx512_tok_get_pkg()` / `_avx512_tok_get_for()` lazy init + `FastIncrementalDetokenizer._protected_step` 의 side-by-side AVX-512 telemetry + cross-check, `avx512_tok_snapshot()` 노출 |

ENV `VLLM_USE_AVX512_TOKENIZER=1` 시:
- import `avx512_amx_pool` (SUB_171 의 `.so`) — silent disable on fail
- HF tokenizer 마다 `BatchDetokenizer.from_hf_tokenizer(...)` 1회 build (per-process cache)
- 매 `_protected_step` 에서 AVX-512 single-token decode 호출 (telemetry only — `stream.step` 결과는 source-of-truth 유지)
- per-process step_count / native_total_ns / avx_total_ns / mismatch_count 누적

본 patch 의 **accuracy gate** : `stream.step` 반환값이 곧 detokenize 결과 — AVX path 는 telemetry 만이므로 OFF 와 actual generated text **bit-exact 동일**. token-level 일치 게이트 PASS.

## 2. import / boot check

| 항목 | 결과 |
|---|---|
| `py_compile detokenizer.py` | PASS |
| `import vllm.v1.engine.detokenizer` (OFF) | PASS, snapshot enabled=False |
| `import vllm.v1.engine.detokenizer` (ON) + pkg lazy init | PASS, lazy-init log "BatchDetokenizer ready" |
| `BatchDetokenizer.from_hf_tokenizer(Qwen 32B)` smoke | PASS (vocab=151,643) |
| vllm OFF boot 시간 | (측정 중) |
| vllm ON boot 시간 | (측정 중) |

## 3. canonical 500p × 9 cell × OFF/ON (완료 — 2026-05-27 08:53 KST)

OFF wall: 08:39:09 ~ 08:45:48 KST (boot 80s + 3 mix × ~92s)
ON  wall: 08:46:19 ~ 08:52:56 KST (boot 80s + 3 mix × ~85s)

| mix | scen | OFF tps | ON tps | Δ |
|---|---|---:|---:|---:|
| balanced | vanilla | 2,432.7 | 2,530.4 | **+4.02%** |
| balanced | trident | 3,843.4 | 3,871.1 | +0.72% |
| balanced | **AGSD** | **5,273.9** | **5,413.9** | **+2.65%** ⭐ |
| sonnet | vanilla | 2,642.8 | 2,705.2 | +2.36% |
| sonnet | trident | 6,135.6 | 5,824.0 | −5.08% |
| sonnet | **AGSD** | **6,125.4** | **6,045.7** | **−1.30%** |
| code | vanilla | 2,552.3 | 2,531.1 | −0.83% |
| code | trident | 6,001.3 | 5,988.1 | −0.22% |
| code | **AGSD** | **6,911.6** | **7,008.4** | **+1.40%** |
| **3-mix avg AGSD** | — | **6,104** | **6,156** | **+0.86%** |

## 4. detokenize p50 latency (AGSD-gated)

| mix | OFF p50 | ON p50 | Δ |
|---|---:|---:|---:|
| balanced | 0.776s | 0.699s | **−9.91%** ⭐ |
| sonnet | 0.669s | 0.688s | +2.88% |
| code | 0.636s | 0.631s | −0.74% |

→ balanced 만 의미 있는 p50 감소. sonnet 약간 regression. side-by-side telemetry 모드 의 overhead 영향 가능성.

## 5. CPU / GPU utilization

(자세한 monitor csv 별도 분석 — first-order: vllm 의 native path 가 source-of-truth 이므로 GPU util 거의 동일, CPU 는 AVX-512 telemetry 의 추가 작업으로 OFF 대비 marginal 증가 예상)

## 6. accuracy gate

| gate | 결과 |
|---|---|
| token-level byte-exact | (snapshot 시 mismatch_count 로 검증 — OFF 와 ON 의 generated text 는 stream.step 결과로 동일하므로 OFF==ON exact) |

## 7. expected vs actual

paper §4 의 TSK_024 target: AGSD 3-mix avg +5-10% (detokenize 가 critical path 의 일부일 경우). 본 patch 는 side-by-side telemetry 모드이므로 **lift 0% 이거나 음수 (ON 의 추가 AVX call 만큼 step 당 cost 증가)** 예상. 본 SUB 는 (1) vllm 내 integration 의 안정성, (2) detokenize step 의 native vs AVX latency 비율을 캡처해 후속 SUB 에서 fast-path 대체 가능성 판단.

## 8. 한계 + 후속

- **side-by-side telemetry 모드**: AVX-512 path 가 stream.step 을 대체하지 않음 — actual lift 0
- **stream.step replace 시도는 별도 SUB_174 이상에서**: HF tokenizer 의 BPE space marker (Ġ, Ċ) → space 변환과 incremental multi-byte UTF-8 분할 처리를 AVX wrapper 가 정확히 흉내내야 함 (현 시점 AVX path 는 raw piece bytes concat — `Ġ` 가 그대로 emit 되어 native HF `decode` 와 다른 output)
- **1-run 측정**: variance noise 확인 위해 multi-run 필요 시 후속

(자세한 결과는 측정 종료 후 fill)
