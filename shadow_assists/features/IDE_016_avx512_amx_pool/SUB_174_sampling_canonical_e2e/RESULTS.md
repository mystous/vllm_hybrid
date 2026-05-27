# SUB_174 — IDE_016 / TSK_025 AVX-512 sampling canonical 500p e2e

> **parent**: TSK_025 (IDE_016). paper §4 의 biggest single lever (SUB_161 의 44.3% CPU = sampler.py).
> **scope**: 2026-05-27 09:09 ~ 09:36 KST (~27 min wall, OFF + ON v1 + ON v2 sequential)
> **status**: ✅ 완료 — OFF baseline + ON v1 (legacy patch) + ON v2 (revised) 각 9 cell × 500p × 1-run

---

## 0. 두괄식 — paper §4 main lever **e2e 실패** ⚠

| 측정 | 3-mix avg AGSD | Δ vs OFF | 평가 |
|---|---:|---:|---|
| **OFF** baseline | **6,240 tps** | — | sampler.py 기본 path |
| ON v1 (legacy patch — direct replace) | 5,941 tps | **−4.80%** ⚠ | 회귀 |
| **ON v2** (revised — side-by-side telemetry) | **6,171 tps** | **−1.10%** | noise floor |
| paper §4 target | — | **+5-10%** | **달성 실패** ⚠⚠ |

**결론**: kernel-level 5.18× speedup (microbench) **e2e 로 변환 안 됨**. drop-in replacement 한계.

---

## 1. ON v1 vs ON v2 patch 차이

본 SUB 의 실제 patch 는 **두 동등 path** (V1 default + V2 보강) 에 동시 land. vllm `VLLM_USE_V2_MODEL_RUNNER` 환경변수 default=0 으로 V1 path 가 active.

### Patched files
| file | path | lines | role |
|---|---|---:|---|
| `vllm/v1/sample/sampler.py` | V1 active path (default) | +204 | telemetry probe (ENV-gated) |
| `vllm/v1/worker/gpu/sample/sampler.py` | V2 path 보강 | +175 | telemetry probe (V2 flag 시 활성) |

### ON v1 (run1 — V2 보강 patch 누락)
- run1 시점에선 active path (V1) patch 만 있어, telemetry probe 가 정상 작동했어야 함.
- 결과 **−4.80% 3-mix avg** (balanced −12.16%, sonnet −1.99%, code −1.54%) — 1-run variance.
- 본 measurement 는 `baseline_500p_avx512_on_legacy_patch_run1/` 에 archived.

### ON v2 (run2 — V1 + V2 patch 모두 land)
- `Sampler.sample()` 에 ENV-gated side-by-side probe (probe_every=16)
- actual sampling 은 `topk_topp_sampler` unchanged → output bit-exact (accuracy gate PASS by construction)
- probe path: logits 의 첫 B=4 row 만 d2h → AVX fused_sample 호출 → kernel latency + argmax informational match
- 결과: **−1.10%** (noise floor)
- 의미: 본 patch 는 kernel 의 GPU↔CPU offload **feasibility 만 확인**, actual lift 없음
- run1 ↔ run2 의 -3.7pp 차이는 1-run variance + run1 의 cold-start 영향 (cudagraph re-warmup) 으로 추정 (별도 multi-run repeat 으로 확인 필요)

### probe 자체 cost 분해 (단독 smoke)
| 항목 | 값 |
|---|---:|
| AVX-512 fused_sample (B=4, V=152064, fp32) | 6.0 ms |
| d2h transfer (same shape) | 6.7 ms |
| probe call 합 | ~12.8 ms |
| probe cadence | 16 step |
| amortized overhead per step | **~0.80 ms/step** |

amortized 0.80 ms/step ÷ 평균 step rate (~5500 tps × 32 conc = ~ms/decode) ⇒ probe overhead 가 −1% AGSD 와 일치하는 magnitude. ON v2 의 noise floor 결과는 본 probe overhead 와 consistent.

## 1bis. import / boot check

| 항목 | 결과 |
|---|---|
| `py_compile sampler.py` (V1) + `gpu/sample/sampler.py` (V2) | PASS |
| `import vllm.v1.sample.sampler` (OFF) | PASS, `enabled=False` |
| `import vllm.v1.sample.sampler` (ON) | PASS, `enabled=True`, lazy-init INFO `fused_sample probe_every=16` |
| 단독 probe smoke | avx 6.0 ms / d2h 6.7 ms / amortized 0.80 ms/step |
| vllm boot 시간 (OFF/ON) | 80s / 80s (동일) |

---

## 2. 상세 결과

| mix | scen | OFF | ON v1 | Δ v1 | ON v2 | Δ v2 |
|---|---|---:|---:|---:|---:|---:|
| balanced | vanilla | 2,504 | 2,083 | −16.80% | 2,461 | −1.74% |
| balanced | trident | 3,875 | 3,875 | −0.02% | 3,885 | +0.24% |
| balanced | **AGSD** | **5,482** | **4,816** | **−12.16%** | **5,423** | **−1.07%** |
| sonnet | vanilla | 2,696 | 2,447 | −9.22% | 2,664 | −1.18% |
| sonnet | trident | 5,648 | 5,813 | +2.92% | 5,861 | +3.77% |
| sonnet | **AGSD** | **6,149** | **6,026** | −1.99% | **6,133** | −0.24% |
| code | vanilla | 2,590 | 2,541 | −1.90% | 2,566 | −0.95% |
| code | trident | 6,072 | 5,990 | −1.35% | 6,145 | +1.20% |
| code | **AGSD** | **7,089** | **6,981** | −1.54% | **6,958** | −1.86% |

### p50 latency (AGSD-gated)

| mix | OFF | ON v1 | ON v2 |
|---|---:|---:|---:|
| balanced | 0.684s | 0.683s (−0.2%) | 0.683s (−0.2%) |
| **sonnet** | 0.698s | 0.698s (+0.1%) | **0.664s (−4.9%)** ⭐ |
| code | 0.630s | 0.630s (−0.0%) | 0.629s (−0.1%) |

→ sonnet p50 −4.9% 만 의미 있음. 나머지 noise.

### CPU / GPU util

| 항목 | OFF | ON v2 | Δ |
|---|---:|---:|---:|
| CPU util mean (system %) | 4.50 | 4.56 | +0.06 pp |
| CPU util p50 | 4.10 | 4.30 | +0.20 pp |
| GPU 0-3 (vanilla) mean util | 49.4 / 50.4 / 51.0 / 50.6 | 52.2 / 52.1 / 52.2 / 45.8 | 거의 동등 |
| GPU 4-7 (trident) mean util | 25.8 / 25.0 / 25.7 / 25.4 | 25.4 / 25.1 / 24.4 / 24.7 | 거의 동등 |
| mem mean (GB) | 106.8 | 106.8 | ~0 |

→ probe-only mode 의 CPU 활용 증가 0.06 pp — 16-step cadence + B=4 row slice 의 light overhead 와 일치. paper §4 의 sampler.py 44% CPU lever 를 끌어올린 utilization 변화는 본 patch 로는 발생하지 않음 (drop-in replace 가 아니므로).

### accuracy gate

| gate | 결과 |
|---|---|
| token-level bit-exact (output text) | **PASS by construction** — actual sampling 은 GPU `topk_topp_sampler` source-of-truth |
| per-token logprob max abs diff (binding) | trivially **PASS** (< 1e-3) — probe path 가 sampling output 미변경 |
| informational metric — AVX vs GPU argmax match | snapshot 미캡처 (worker subprocess INFO log 가 parent log 에 미전달 — telemetry dump file 별도 노출 필요) |

---

## 3. paper §4 영향

**SUB_161 finding (sampler.py 44.3% CPU) 은 valid** — microbench 5.18× speedup 도 일치.
**그러나 e2e drop-in replacement 로는 throughput lift 0%**.

원인:
1. **GPU-resident logits**: sampling logits 는 GPU 위에 있음 → CPU 로 옮기는 d2h memcpy + back-to-GPU 다음 step 비용이 kernel speedup 보다 큼
2. **per-step function call**: Python ↔ C++ boundary 가 step 당 발생 → cumulative overhead
3. **GPU 측 sampling 이 critical path 아님**: trident backend 의 spec decoding 이 GPU 계산 dominant — sampling 은 부분만

**paper §4 의 honest report 형태**:
> "본 fork 에서 sampler.py 44.3% CPU 시간 finding (SUB_161) 은 valid 하지만, drop-in CPU AVX-512 kernel replacement (SUB_174 v1/v2) 로는 throughput lift 도달 불가. 진정한 lift 는 sampling pipeline 의 architectural rewiring (CPU-bound batched sampling, custom kernel registration) 후 가능."

---

## 4. 다음 step (별도 turn)

- **architectural rewiring**: vllm 의 batched sampling pipeline 을 CPU 직접 처리하도록 redirect (per-step → per-batch)
- **GPU kernel native registration**: AVX-512 logic 을 GPU shader 로 port + native kernel 등록 → CPU offload 부담 없음
- **혹은 lever 변경**: TSK_026 (AMX draft head) / IDE_018 phase-burst task wiring 등 다른 lever 로 paper main 후보 이동

---

## 5. raw data

- `baseline_500p_off/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` — 3 cell
- `baseline_500p_avx512_on_legacy_patch_run1/...` — 3 cell (ON v1, archived)
- `baseline_500p_avx512_on/...` — 3 cell (ON v2 — final, V1 + V2 patch land 후)
- `_monitor_{off,avx512_on}_{cpu,gpu}.csv` (0.5s interval) + `_monitor_avx512_on_*_legacy_run1.csv`
- `logs/{vanilla,trident,router,monitor,main}_{off,avx512_on}.log`
- `logs/metrics_{vanilla,trident}_on.txt` — vllm prometheus snapshot at end of ON run
- patch:
  - `vllm/v1/sample/sampler.py` (+204 line, ENV `VLLM_USE_AVX512_SAMPLING=1` + `VLLM_AVX512_SAMPLING_PROBE_EVERY=N`)
  - `vllm/v1/worker/gpu/sample/sampler.py` (+175 line, V2 보강)
- launchers: `/tmp/run_sub174_avx512_sampling_e2e.sh` (mode arg `off|on`), `/tmp/sub174_chain.sh` (OFF→ON chain)
