# B3 8GPU FULL 단독 강제 검증 — MEASUREMENTS

> 이전 b3_sched (4GPU + Qwen-7B + FlashInfer/FA forced) finding 재검증 + 추가 backend (TRITON_ATTN, FLEX_ATTENTION) 시도.
> 사용자 명시: **B200 8 GPU 활용 강제 검증** — 모든 우회 옵션 시도 + 가능 시 측정 / 불가능 시 정확한 차단 layer 정리.

---

## 0. 환경 & 사전 준비

- **Hardware**: NVIDIA B200 × 8 (GPU 0-7, sm_100, 각 약 183 GB HBM)
- **vllm**: `1.7.dev16107+gffe20fb09.d20260601` (sm_100 재빌드)
- **torch**: `2.11.0+cu128`, CUDA tk 12.8
- **Date**: 2026-06-05 04:48 ~ 05:29 UTC
- **Model**: `meta-llama/Llama-3.1-8B-Instruct` (TP=8 정합 — 32 heads / 8 KV)
  - 주: 원 task 는 `Qwen/Qwen2.5-7B-Instruct` (28 heads) 였으나 28 ÷ 8 = 3.5 → `Total number of attention heads (28) must be divisible by tensor parallel size (8)` ValidationError 로 boot 불가 (`vllm/engine/arg_utils.py:create_engine_config` → `VllmConfig` pydantic validator). 증거: `sweep.QWEN_TP8_FAIL.log`, `runs/R0_FI_P_boot.QWEN_TP8_FAIL.log`. 8 GPU 활용 강제 요구를 충족하기 위해 TP=8 정합 모델로 8B class 의 Llama-3.1-8B 채택.
- **Workload**: sharegpt 200p × conc=32 × max-tokens=8192, stream=True → TTFT/TPOT 분해
- **Spec decode**: 미사용 (vanilla decode) — cudagraph_mode lever 격리
- **`--max-model-len`**: 20480 (IDE_022 CLAUDE.md 함정 회피 — input 8.4k + output 8.2k 안전 마진)
- **Boot CLI 골격**:
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 8 --port 8005 \
    --gpu-memory-utilization 0.85 --max-model-len 20480 \
    --compilation-config '{"cudagraph_mode":<MODE>}' \
    [--attention-config '<JSON>'] \
    --allow-deprecated-quantization
  ```

### 0.1 attention backend 환경 (사전 점검)

| 항목 | 값 |
|---|---|
| `vllm.vllm_flash_attn.is_fa_version_supported(2)` | `True` |
| `vllm.vllm_flash_attn.is_fa_version_supported(3)` | **`False`** |
| `vllm.vllm_flash_attn.is_fa_version_supported(4)` | `True` |
| device capability | `(10, 0)` (sm_100, Blackwell) |
| auto-selected backend log | `Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION']` |

→ FA3 빌드 부재. B200 + FA3 의 ALWAYS cap 경로 활성 불가 (b3_sched 1·2차 finding 재확인). B200 sm_100 에서 vllm 이 인식하는 가용 backend = `[FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION]` (default 우선순위 = FLASHINFER).

### 0.2 가용 attention backend × `_cudagraph_support` cap

| Backend | `_cudagraph_support` | B200 호환 | 비고 (file:line) |
|---|---|---|---|
| `FLASHINFER` (default) | dynamic (TRTLLM 조건 시 `UNIFORM_BATCH`, 아니면 `UNIFORM_SINGLE_TOKEN_DECODE`) | OK | `vllm/v1/attention/backends/flashinfer.py:705-743` |
| `FLASH_ATTN` (FA4 on B200) | `UNIFORM_BATCH` (FA4) / `ALWAYS` (FA3 만) | OK (FA4) | `vllm/v1/attention/backends/flash_attn.py:292-296` |
| `TRITON_ATTN` | **`ALWAYS`** ★ | OK (`supports_compute_capability=True`) | `vllm/v1/attention/backends/triton_attn.py:127`, `triton_attn.py:383-384` |
| `FLEX_ATTENTION` | **`ALWAYS`** ★ | OK (base class default) | `vllm/v1/attention/backends/flex_attention.py:723` |

→ B200 환경에서 **FULL 단독 활성 후보 = TRITON_ATTN, FLEX_ATTENTION** (둘 다 `_cudagraph_support = ALWAYS`).
→ FlashInfer, FLASH_ATTN/FA4 는 `ALWAYS` 가 아니므로 `vllm/config/compilation.py:1286-1310` 가드에 의해 **단독 FULL 요청 시 자동 FaP 다운그레이드**.

---

## 1. 차단 코드 layer (정적 분석)

### 1.1 FA3 빌드/하드웨어 차단 (b3_sched 에서 확정, 본 sweep 재확인)

- `vllm/v1/attention/backends/fa_utils.py:96-101` — Blackwell (`major >= 10`) 에서 `flash_attn_version=3` 요청 시 자동 FA4 fallback + warning.
- `vllm.vllm_flash_attn.is_fa_version_supported(3) == False` (본 빌드).
- `vllm/v1/attention/backends/flash_attn.py:292-296` — `_cudagraph_support = ALWAYS if FA==3 else UNIFORM_BATCH` → B200 의 FA4 는 UNIFORM_BATCH 로 고정.

### 1.2 cudagraph_mode FULL 자동 다운그레이드 가드 (핵심)

- `vllm/config/compilation.py:1286-1310`:
  ```python
  if (
      cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
      and min_cg_support != AttentionCGSupport.ALWAYS
  ):
      msg = (...)
      if self.splitting_ops_contain_attention():
          msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
          cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
      else:
          msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
          cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
      logger.warning(msg)
  ```
- 즉 어떤 backend 든 `_cudagraph_support != ALWAYS` 이면 `--compilation-config '{"cudagraph_mode":"FULL"}'` 요청은 반드시 FaP (또는 FULL_DECODE_ONLY) 로 다운그레이드. 본 sweep 의 R1_FI_F, R4_FA_F 가 이 가드에 의해 다운그레이드된 사례.

### 1.3 FlashInfer 의 동적 cap

- `vllm/v1/attention/backends/flashinfer.py:705-743` — TRTLLM attention 가용 시 `UNIFORM_BATCH`, 아니면 `UNIFORM_SINGLE_TOKEN_DECODE`. 어느 쪽도 `ALWAYS` 가 아니므로 단독 FULL 불가.

### 1.4 Qwen-7B + TP=8 head 부정합 (본 sweep 에서 새로 발견)

- `Total number of attention heads (28) must be divisible by tensor parallel size (8)` (VllmConfig pydantic validator). Qwen2.5-7B (28 heads) 는 TP=8 로 boot 불가. 8 GPU 활용 강제 시 Llama-3.1-8B (32 heads) 같은 대안 모델 필요.

---

## 2. backend × cudagraph_mode 시도 matrix (boot 단계)

| Run | backend (요청) | cudagraph_mode (요청) | 실 적용 backend | 실 적용 cudagraph_mode | FULL 단독 active? | downgrade evidence |
|---|---|---|---|---|---|---|
| **R0_FI_P** | (auto → FlashInfer) | PIECEWISE | FLASHINFER | PIECEWISE (51 cap) | × (요청대로) | n/a |
| **R1_FI_F** | (auto → FlashInfer) | FULL | FLASHINFER | **FULL_AND_PIECEWISE** (다운그레이드) | × | `compilation.py:1310 ... CUDAGraphMode.FULL is not supported with FlashInferBackend backend (support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_AND_PIECEWISE` (8 ranks 모두) |
| **R2_FI_FaP** | (auto → FlashInfer) | FULL_AND_PIECEWISE | FLASHINFER | FaP (PIECEWISE=51, FULL=51) | × (요청 자체가 FaP) | n/a |
| **R4_FA_F** | FLASH_ATTN | FULL | FLASH_ATTN (FA4) | **FULL_AND_PIECEWISE** (다운그레이드) | × | `compilation.py:1310 ... CUDAGraphMode.FULL is not supported with FlashAttentionBackend backend (support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_AND_PIECEWISE` |
| **R5_FA_FaP** | FLASH_ATTN | FULL_AND_PIECEWISE | FLASH_ATTN (FA4) | FaP | × | n/a |
| **R6_TR_P** | TRITON_ATTN | PIECEWISE | TRITON_ATTN | PIECEWISE | × (요청대로) | n/a |
| **R7_TR_F** ★ | TRITON_ATTN | **FULL** | TRITON_ATTN | **FULL 단독** (PIECEWISE 없음, FULL=51) | **○ 성공** | `Profiling CUDA graph memory: FULL=51 (largest=512)` (PIECEWISE line 부재). downgrade warning 없음. |
| **R8_TR_FaP** | TRITON_ATTN | FULL_AND_PIECEWISE | TRITON_ATTN | FaP (PIECEWISE=51, FULL=51) | × | n/a |
| **R10_FX_F** ★ | FLEX_ATTENTION | **FULL** | FLEX_ATTENTION | **FULL 단독** (PIECEWISE 없음, FULL=51, capture 성공) | △ **boot/capture 성공 그러나 inference 첫 호출 시 `CUDA_ERROR_ILLEGAL_ADDRESS` (`flashinfer/cuda_utils.py:53` cuMemFree)** → bench fail (168/200 err) |

★ = ALWAYS cap backend (FULL 단독 활성 후보).

→ **본 B200 8 GPU + Llama-8B + TP=8 환경에서 FULL 단독 cudagraph mode 활성 + 정상 inference 성공 backend = `TRITON_ATTN` 하나** (FLEX_ATTENTION 은 capture 까진 통과하나 런타임 illegal address).

---

## 3. Throughput 측정 (sharegpt 200p × conc=32, stream)

활성 성공 (boot + bench 모두 통과) case 만 측정 결과 기재. R1_FI_F / R4_FA_F 는 boot 후 즉시 stop (boot-only). R10_FX_F 는 boot 성공 / bench 실패.

| Run | effective backend × mode | wall (s) | tokens | **output_tps** | TTFT p50 (ms) | TTFT p99 | TPOT p50 (ms) | TPOT p99 | GPU util (%) | CPU util (%) | n_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R0_FI_P** | FlashInfer × PIECEWISE | 178.2 | 1 526 570 | **8 568.8** | 48.5 | 305.7 | 3.6 | 3.6 | 94.0 | 4.6 | 200/200 |
| **R2_FI_FaP** | FlashInfer × FaP | 126.1 | (~1.52 M) | **12 078.9** | 62.9 | 294.3 | 2.5 | 2.6 | 98.1 | 5.2 | 200/200 |
| **R5_FA_FaP** | FA4 × FaP | 152.6 | (~1.51 M) | **9 901.7** | 47.0 | 6 500.9 | 2.8 | 18.3 | 90.5 | 5.0 | 200/200 |
| **R6_TR_P** | TritonAttn × PIECEWISE | 294.4 | (~1.47 M) | **4 983.0** | 72.5 | 1 067.0 | 5.9 | 6.0 | 92.1 | 4.5 | 200/200 |
| **R7_TR_F** ★ | TritonAttn × **FULL 단독** | 582.4 | (~1.50 M) | **2 574.9** | 77.6 | 296.8 | 11.2 | 15.8 | 99.8 | 4.8 | 200/200 |
| **R8_TR_FaP** | TritonAttn × FaP | 170.0 | (~1.50 M) | **8 826.7** | 70.5 | 333.1 | 3.4 | 3.8 | 97.7 | 5.0 | 200/200 |
| R10_FX_F | FlexAttn × FULL 단독 | 0.6 | 0 | **0.0 (FAIL)** | — | — | — | — | 0.0 | 2.1 | 32/200, 168 HTTP500 (`CUDA_ERROR_ILLEGAL_ADDRESS`) |

### 3.1 백엔드 별 Δ 분석

#### A) backend 통제 (모두 FaP 로 통일 — backend 자체의 raw 효율)

| backend | output_tps | vs FlashInfer FaP |
|---|---|---|
| FlashInfer × FaP (R2) | 12 078.9 | baseline |
| FlashAttn(FA4) × FaP (R5) | 9 901.7 | -18.0 % |
| TritonAttn × FaP (R8) | 8 826.7 | -26.9 % |

→ FlashInfer 가 B200 + Llama-8B + sharegpt 에서 raw 성능 최고. Triton 은 26.9 % 느림.

#### B) Triton backend 내부 cudagraph_mode 비교 (★ FULL 단독 lever 의 net effect)

| mode | output_tps | vs Triton PIECEWISE | vs Triton FaP |
|---|---|---|---|
| TritonAttn × PIECEWISE (R6) | 4 983.0 | baseline | -43.5 % |
| TritonAttn × **FULL 단독** (R7) | 2 574.9 | **-48.3 %** ◆ | -70.8 % |
| TritonAttn × FaP (R8) | 8 826.7 | **+77.1 %** | baseline |

◆ **본 task 의 핵심 결과**: B200 + Llama-8B + Triton 에서 FULL 단독 cudagraph mode 활성에 성공했으나, **PIECEWISE 대비 48 % 느려짐**, FaP 대비 71 % 느려짐. GPU util 은 99.8 %로 saturate 되지만 wall 이 거의 2 배. 원인 추정: mixed prefill-decode batch 에서 FULL graph 는 큰 batch shape 들만 capture, 그 외 shape (특히 prefill 의 다양한 seq_len) 는 cudagraph miss → eager 또는 비효율 path. FaP 는 prefill 만 PIECEWISE 로 (작은 graph 단위), decode 만 FULL 로 자동 dispatch 하기에 두 stream 모두 효율적.

#### C) FlashInfer backend 내부 (참고 — FULL 단독 활성 불가 case)

| mode (요청) | 실 적용 mode | output_tps | vs PIECEWISE |
|---|---|---|---|
| PIECEWISE (R0) | PIECEWISE | 8 568.8 | baseline |
| FULL (R1, boot-only) | FaP (다운그레이드) | — | — |
| FaP (R2) | FaP | 12 078.9 | **+41.0 %** |

→ **B200 + FlashInfer + Llama-8B 8 GPU 에서 PIECEWISE → FaP 로 변경 시 +41 %** (b3_sched 4GPU + Qwen-7B + 100p × conc=16 의 +30.1 % 보다 더 큰 이득). burst·model·scale 가 커질수록 FaP 이득이 커지는 b3_sched §3.1 의 가설 재확인.

---

## 4. 본 task 결론

### 4.1 "B200 8 GPU 에서 FULL 단독 cudagraph mode 활성 가능?"
- **활성 가능** : `--attention-config '{"backend":"TRITON_ATTN"}' --compilation-config '{"cudagraph_mode":"FULL"}'` 조합 → boot/capture/bench 모두 성공 (R7_TR_F).
- **활성 가능 + bench 실패** : `--attention-config '{"backend":"FLEX_ATTENTION"}' --compilation-config '{"cudagraph_mode":"FULL"}'` → boot/capture 성공, 첫 inference 호출 시 `CUDA_ERROR_ILLEGAL_ADDRESS` (flashinfer cuMemFree). 본 빌드의 FLEX_ATTENTION + B200 + TP=8 런타임 결함.
- **활성 불가** : FlashInfer (default), FLASH_ATTN/FA4 → 모두 `_cudagraph_support` 가 `ALWAYS` 가 아니므로 `vllm/config/compilation.py:1310` 가드에 의해 FaP 로 자동 다운그레이드.

### 4.2 "활성 시 net effect"
- TRITON_ATTN × FULL 단독 = 2 574.9 tps.
- 비교 baseline / 비교 대상:
  - 동일 backend PIECEWISE : 4 983 tps → **FULL 단독은 -48 %** (역효과).
  - 동일 backend FaP : 8 826.7 tps → **FULL 단독은 -71 %** (역효과).
  - **전체 최고치 (FlashInfer FaP)** : 12 078.9 tps → **FULL 단독은 -78.7 %** (대규모 역효과).
- 즉 B200 8 GPU 환경에서 **"FULL 단독 활성 → 더 빨라진다" 라는 SUB_201 §5 의 38 % 회수 가설은 본 sweep 데이터로 반증**.
- 이유 (정성적 추정): TRITON_ATTN 자체가 FlashInfer 대비 raw 효율이 낮은 데다 (-27 %), FULL 단독은 mixed prefill-decode batch 의 다양한 shape 를 모두 cudagraph 로 capture 못하기에 prefill 영역에서 큰 miss penalty 발생. FaP 는 prefill/decode 별도 graph stream 으로 dispatch 하므로 이 문제 회피.

### 4.3 권장 (B200 8 GPU + Llama-8B)
- **production 권장 = FlashInfer × FaP** (12 078.9 tps, GPU 98 %, TPOT p50 = 2.5 ms). b3_sched DECISION §3.1 의 권고를 8 GPU + 8B class + 200p × conc=32 데이터에서 재확인.
- FULL 단독 활성은 **기술적으로 가능 (Triton 경유)** 하나 **net throughput 면에서 손해**. SUB_201 §5 의 launch overhead 회수 시나리오는 본 환경 (mixed prefill-decode, conc=32) 에서 net win 으로 이어지지 않음.
- FULL 단독 lever 의 net win 검증은 (a) prod H100 + FA3 native (DECISION §3.2) 또는 (b) decode-only 워크로드 (prefill 비중 ↓, FULL graph hit-rate ↑) 에서 별도 sweep 필요.

### 4.4 GPU 0-7 최종 free 검증
```
$ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
0, 0    1, 0    2, 0    3, 0    4, 0    5, 0    6, 0    7, 0
$ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
(empty)
```
모든 GPU free, orphan process 없음.

---

## 5. 산출물 paths

- sweep entry: `/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_8gpu_full/sweep.sh`
- runs:
  - `runs/R0_FI_P.{json,raw.jsonl,_boot.log,_bench.log}` — FlashInfer × PIECEWISE baseline
  - `runs/R1_FI_F_boot.log` — FlashInfer FULL → FaP 다운그레이드 증거
  - `runs/R2_FI_FaP.{json,raw.jsonl,_boot.log,_bench.log}` — FlashInfer × FaP (전체 최고)
  - `runs/R4_FA_F_boot.log` — FA4 FULL → FaP 다운그레이드 증거
  - `runs/R5_FA_FaP.{json,raw.jsonl,_boot.log,_bench.log}` — FA4 × FaP
  - `runs/R6_TR_P.{...}` — TritonAttn × PIECEWISE
  - `runs/R7_TR_F.{...}` — ★ TritonAttn × FULL 단독 활성 성공
  - `runs/R8_TR_FaP.{...}` — TritonAttn × FaP
  - `runs/R10_FX_F.{...}` — FLEX_ATTENTION FULL 활성 boot 성공 + bench CUDA_ERROR_ILLEGAL_ADDRESS
- prompts: `sharegpt200.parquet` (`runs/tput_t1t3_20260602/sampled_prompts.parquet` 중 sharegpt 500p 의 random 200p, seed=42)
- 실패 산출물 (Qwen-7B TP=8 head 부정합): `sweep.QWEN_TP8_FAIL.log`, `runs/R0_FI_P_boot.QWEN_TP8_FAIL.log`
- sweep 진행 log: `sweep.log`

---

## 6. b3_sched 의 finding 과의 비교

| 비교 항목 | b3_sched (4GPU, Qwen-7B, 100p × conc16) | **b3_8gpu_full (8GPU, Llama-8B, 200p × conc32)** |
|---|---|---|
| FA3 활성 가능? | 불가 (빌드 부재 + 코드 가드) | **불가 (동일)** |
| FlashInfer default FULL 단독? | 불가 (UNIFORM_BATCH cap) | **불가 (동일, R1_FI_F 다운그레이드)** |
| FA4 FULL 단독? | 불가 (UNIFORM_BATCH cap) | **불가 (동일, R4_FA_F 다운그레이드)** |
| **TRITON_ATTN FULL 단독?** | 미시도 | **○ 활성 성공 (R7_TR_F)** |
| **FLEX_ATTENTION FULL 단독?** | 미시도 | △ boot 성공 / 런타임 illegal address (R10_FX_F) |
| FaP net win vs PIECEWISE (default backend) | +30.1 % (100p) | **+41.0 %** (200p, 8GPU, scale ↑) |
| FULL 단독 활성 시 net effect | 측정 불가 | **-71 % vs FaP (역효과 확인)** |

→ b3_sched 의 핵심 결론 ("B200 에서 FULL 단독 자체가 미활성") 은 **default backend (FlashInfer/FA) 기준으로는 여전히 정확**. 본 sweep 의 추가 finding 으로 **"TRITON_ATTN 강제 시 활성 자체는 가능하지만 net effect 가 역효과 (-71 % vs FaP)"** 가 새로 확정. 즉 §3.2 의 "prod H100 + FA3 native 가 필요" 라는 권고는 여전히 유효 — 본 B200 환경의 Triton FULL 단독은 §5 의 38 % 회수 가설을 지지하지 않음.
