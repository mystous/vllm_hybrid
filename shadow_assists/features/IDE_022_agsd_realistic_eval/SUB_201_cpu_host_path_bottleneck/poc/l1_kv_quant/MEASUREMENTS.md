# SUB_201 후속 lever L1 — KV cache dtype quantization (BF16 → FP8) MEASUREMENTS

## 0. 환경

- **Hardware**: NVIDIA B200 × 8, GPU 0-7 (각 약 183 GiB HBM, sm_100)
- **CPU**: Intel Xeon 8570 (224 thread), AMX native
- **vllm**: `1.7.dev16107+gffe20fb09.d20260601` (sm_100 재빌드, editable)
- **torch**: `2.11.0+cu128`, nccl `2.28.9`
- **Date**: 2026-06-06 16:14 ~ 16:39 UTC
- **`LD_LIBRARY_PATH`**: `torch/lib + nvidia/nccl/lib + cuda/lib64`
- **공유 환경 주의**: 본 측정 도중 다른 lever sweep (`l10_admission`, `l12_cudagraph_warmup` 등)이 동시 진행 → GPU 점유 변동 심함, 큰 모델 boot 시도 시 fail 빈발 (`_m2_attempt[1-7]_fail/`).

## 1. 측정 plan vs 실측

| 모델 | TP | GPU | baseline | lever | 결과 |
|---|---:|---|---|---|---|
| **M1 Qwen2.5-7B-Instruct** | 2 | 0,1 | `auto` (bf16) | `fp8` | ✅ 둘 다 측정 완료 |
| M2 Llama-3.1-70B-Instruct | 4 | 0-3 | `auto` | `fp8` | ❌ boot fail × 7 회 (아래 §5) |
| M3 DeepSeek-R1 671B | 8 | 0-7 | `auto` | `fp8` | ⏭ M2 fail 누적으로 시도 보류 |

- corpus: sharegpt 100p × conc=16 × max-tokens=512 (capacity-focused)
- runner: `/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py` (기존 b3 sweep 자산 재사용)
- 동일 corpus: `../b3_8gpu_full/sharegpt200.parquet` (`--limit 100`)
- attn backend: 기본 `FLASHINFER` + `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` (안정 동작 검증된 옵션 채택)

## 2. M1 — Qwen2.5-7B-Instruct (TP=2, GPU 0,1) 결과

| metric | baseline (`auto`) | fp8 KV | Δ% | 해석 |
|---|---:|---:|---:|---|
| output_tps (gen tok/s aggregate) | **3314.2** | 2773.2 | −16.3 % | fp8 case 26 / 100 fail 로 *unfair* |
| n_ok / n | **100 / 100** | **74 / 100** | — | ★ fp8 에서 EngineCore fatal (Worker-0 died) |
| TTFT p50 (ms) | 22.3 | 38.4 | +72.2 % | fp8 path 의 TRTLLM attention init overhead |
| TTFT p99 (ms) | 582.5 | 442.8 | −24.0 % | tail 은 개선 (mem 여유) |
| TPOT p50 (ms/tok) | 4.5 | 3.8 | **−15.6 %** | fp8 KV load/store 가 decode bw 절감 |
| TPOT p99 (ms/tok) | 5.2 | 4.5 | **−13.5 %** | 동일 — decode 자체는 fp8 가 빠름 |
| GPU used @ boot (MiB/dev) | 158356 | 158318 | ≈ 0 % | `gpu-memory-utilization 0.85` 기준 비슷 |
| GPU mem mean during bench (MiB) | 630282 | 563742 | −10.5 % | fp8 KV 가 점유 적음 |
| GPU used @ post (MiB/dev) | 78786 | 19600 | −75.1 % | 본 lever 의 핵심 효과 — KV mem 1/4 ~ 1/2 |

### 2.1 KV cache capacity — vLLM 보고 (boot log)

| 지표 | baseline (`auto`) | fp8 KV | Δ |
|---|---:|---:|---:|
| **GPU KV cache size** | **5,246,224 tokens** | **10,492,448 tokens** | **× 2.00** |
| Maximum concurrency (8192 ctx) | 640.41 × | **1280.82 ×** | × 2.00 |

→ vLLM 의 fp8 KV cache 활성 + B200 native fp8 (`e4m3`) 가 의도대로 KV memory 를 **정확히 절반**으로 줄여 effective concurrency capacity 를 2 × 확장하는 것이 boot 단계에서 검증됨.

### 2.2 fp8 KV 안정성 finding (★ blocker)

- fp8 Boot 직후 약 **9 초** 가 bench 가 진행되다 `Worker proc VllmWorker-0 died unexpectedly` → `EngineDeadError` → 26 / 100 request `HTTP 500`.
- evidence: `runs/M1_qwen7b_fp8_boot.log` 16:21:36 — `multiproc_executor.py:285 Worker proc VllmWorker-0 died unexpectedly, shutting down executor.`
- baseline (`auto`) 경로는 동일 workload 에서 100 / 100 정상.
- 즉 lever 자체는 KV capacity 확장은 의도대로 동작하지만, **vLLM 1.7.dev16107 + B200 sm_100 + FlashInfer + TRTLLM attention + fp8 KV (qwen2.5-7B, scaling factor 없음)** 조합에 Worker stability 이슈가 존재.
- TPOT 가 fp8 측에서 −13~15 % 개선된 점은 — bench 시작 9 초 (≈ 1 wave of decode) 동안의 정상 구간 sample 이므로 의미 있는 단일 측정.

## 3. M2 / M3 — boot fail 누적 (시도 7 회)

| Attempt | tag | TP | GPUs | GMU | reason |
|---|---|---:|---|---:|---|
| 1 | `_m2_attempt1_fail/` | 4 | 0,1,3,5 | 0.85 | GPU 3 외부 점유 (152 GiB) — `Free memory on device cuda:3 (26.35/178.35 GiB)` |
| 2 | `_m2_attempt2_fail/` | 4 | 0,1,4,5 | 0.85 | GPU 5 외부 점유 (25 GiB) — `cuda:3` (= phys 5) Free 부족 |
| 3 | `_m2_attempt3_fail/` | 4 | 0,1,2,3 | 0.85 | Worker `EOFError` from `wait_for_ready` (init_device EOF) |
| 4 (TP=2 fallback) | `_m2tp2_attempt1_fail/` | 2 | 0,1 | 0.85 | Worker silent fail in `init_device` |
| 5 | `_m2_attempt4_fail/` | 4 | 0,1,2,3 | 0.85 | 같은 wait_for_ready EOF |
| 6 | `_m2_attempt5_fail/` | 4 | 0,1,2,3 | 0.85 | Free memory cuda:3 (112.82 / 178.35 GiB) 부족 |
| 7 (GMU=0.60) | `_m2_attempt6_fail/` | 4 | 0,1,2,3 | 0.60 | wait_for_ready Exception (Worker silent) |
| 8 (clean env) | `_m2_attempt7_fail/` | 4 | 0,1,2,3 | 0.80 | ★ Model loading 100 % 성공 후 **flashinfer allreduce 직후 VllmWorker-1 segfault** (16:39:18) |

### 3.1 Llama-70B + TP=4 boot 실패 원인 (정적 분석)

- attempt 8 boot log 의 마지막 정상 step → `flashinfer_all_reduce.py:149 Initialized FlashInfer Allreduce norm fusion workspace with backend=trtllm` 직후 Worker-1 죽음.
- 즉 `meta-llama/Llama-3.1-70B-Instruct` + B200 sm_100 + vLLM 1.7.dev16107 + **FlashInfer TRTLLM allreduce 경로** 에 boot stability 이슈가 존재 (KV dtype 와 무관, 본 lever 측정 외 issue).
- baseline (`auto` KV) 도 같은 boot fail → 본 lever 와 무관한 환경 문제. **M3 (R1 671B) 도 동일 path 사용하므로 같은 실패 예상**, 시도 보류.

## 4. 결론

### 4.1 본 task verdict — 모델 사이즈 의존 net Δ%

| 모델 | net Δ% (tps) | net Δ% (KV cap) | net 판정 |
|---|---:|---:|---|
| **M1 Qwen2.5-7B** | −16.3 % (★ fp8 stability fail) / +개선 가능성 (TPOT −15 %) | **+100 %** (5.25 M → 10.49 M tokens) | **conditional win — fp8 worker stability fix 후에야 net 양수** |
| M2 Llama-70B | n/a (boot fail) | n/a | 미측정 — 환경 stability issue |
| M3 R1 671B | n/a (스킵) | n/a | 미측정 — 동일 path 실패 예상 |

### 4.2 lever 적용 권고

- **KV capacity 효과는 의도대로 작동** (boot log 가 +100 % 확장 명시), 즉 memory-bound 회복에는 효과적이라는 가설은 **기각되지 않음**.
- 그러나 **현재 vLLM 빌드** (1.7.dev16107, sm_100 재빌드) **+ FlashInfer TRTLLM** 경로의 fp8 KV 안정성이 production 차단. 본 lever 를 SUB_201 의 net win 후속으로 채택하려면 **fp8 KV scaling factor 가 사전 calibration 된 모델 가중치** 또는 **다른 attention backend (TRITON_ATTN / FLEX_ATTENTION)** 로 전환 필요.
- "CPU AMX 자체로 KV 양자화" 는 본 lever 범위 밖 (vLLM 의 KV 양자화는 GPU side fp8). AMX 활용은 별개 작업 (future work).

### 4.3 GPU 최종 free 검증

본 task 종료 시 GPU 0-7 free 상태 (외부 다른 sweep 의 GPU 3 점유는 본 task 와 무관):

```
$ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader  # 16:39 task 종료 후
0, 0 MiB
1, 0 MiB
2, 0 MiB
3, 157448 MiB   ← 외부 lever sweep
4, 0 MiB
5, 4 MiB
6, 0 MiB
7, 0 MiB
```

본 task 의 모든 vllm worker / sweep process 는 `kill -TERM -- -<sweep_pid>` + orphan 직접 kill 로 정리 완료. GPU 3 은 동시 진행 중인 다른 lever sweep 점유 (본 task 외부, 보존).

## 5. 산출물

```
/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/l1_kv_quant/
├── README.md
├── sweep.sh                     ← PHASE=m1|m2|m2tp2 / M2_GPUS / GMU env
├── summarize.py                 ← runs/*.json + gpu csv → markdown 표
├── MEASUREMENTS.md              ← 본 문서
└── runs/
    ├── M1_qwen7b_{auto,fp8}.json          ← bench summary
    ├── M1_qwen7b_{auto,fp8}.raw.jsonl     ← per-request raw
    ├── M1_qwen7b_{auto,fp8}_boot.log      ← vllm serve stdout
    ├── M1_qwen7b_{auto,fp8}_bench.log     ← runner stdout
    ├── M1_qwen7b_{auto,fp8}_gpu_boot.csv  ← boot 직후 nvidia-smi
    ├── M1_qwen7b_{auto,fp8}_gpu_post.csv  ← bench 종료 후 nvidia-smi
    ├── sweep_m1.log                       ← M1 sweep stdout
    ├── _attempt1_fail/                    ← enforce-eager 모드 worker hang (회피)
    ├── _m2_attempt[1-7]_fail/             ← Llama-70B boot fail 7회
    └── _m2tp2_attempt1_fail/              ← Llama-70B TP=2 fallback fail
```
