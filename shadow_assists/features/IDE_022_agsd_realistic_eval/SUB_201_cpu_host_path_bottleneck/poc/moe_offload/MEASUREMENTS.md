# SUB_201 PoC — MoE CPU Offload SHORT 측정 (B200 + AMX)

> **목적**: HANDOFF_b200_moe_cpu_offload.md §3.1 1단계 SHORT 측정.
> **측정 일자**: 2026-06-06
> **하드웨어**: DGX B200 8× (각 183 GB HBM) + Intel Xeon Platinum 8570 (112c/224t, AMX native: `amx_bf16, amx_tile, amx_int8`), 2 TB DRAM, 2 NUMA nodes.
> **모델**: `Qwen/Qwen3-30B-A3B-Instruct-2507` (Qwen3 MoE, 48 layers × 128 experts × top-k 8, BF16, 57 GB).
> **엔진**: SGLang-kt 0.6.2.post3 (kvcache-ai 포크) + kt_kernel 0.6.2.post4 (AMX backend).

## 1. 측정 프로토콜 (요약)

- 20 prompt × max_tokens=256 × decode-weighted (짧은 input, 긴 output) × concurrency=8.
- 동일 prompt set (seed=0), 동일 backend chain (attention=triton, moe_runner=triton, cuda-graph off, flashinfer-autotune off).
- 1회 warm-up (16 tokens) 후 본측정.
- `served_model_name` = 모델 디렉토리 절대 경로(HANDOFF 함정 회피: `"model":"model"` 금지).

## 2. 환경/설치 결과

- `lscpu | grep amx` → `amx_bf16, amx_tile, amx_int8` 확인 (Xeon Platinum 8570, Emerald Rapids).
- SGLang(pypi) → vllm sm_100 wheel ABI 손상 → **vllm baseline 포기**, A/B 모두 SGLang-kt 위에서 측정 (동일 backend → fair).
- kt-kernel 설치(`uv pip install kt-kernel sglang-kt`, 별도 venv `/workspace/sglang_kt_prj`), `kt_kernel.cli.main doctor` 11/12 OK (AMX 활성).
- AMX INT8 weight 변환은 시도했으나 `process_weights_after_loading` 에서 expert-shape 인덱싱 OOB(buggy code path in current sglang-kt for non-quantized models) → BF16 inference mode로 전환.

## 3. A — full-GPU baseline

| 항목 | 값 |
|---|---|
| backend | SGLang-kt, attention=triton, moe-runner=triton, cuda-graph off |
| TP | 2 |
| GPU | 2× B200 (GPU0, GPU1) |
| GPU mem after measurement | 162 701 / 162 793 MiB (KV alloc 60.71 GB × 2) |
| decode tps | **157.94** |
| **tps per GPU** | **78.97** |
| req/s | 0.617 |
| p50 (ms) | 10 490 |
| p99 (ms) | 11 450 |
| n_ok / n_err | 20 / 0 |
| wall (s) | 32.42 |
| total output tokens | 5 120 |

## 4. B — SGLang + KTransformers kt-kernel (MoE expert CPU/AMX offload)

| 항목 | 값 |
|---|---|
| backend | SGLang-kt, attention=triton, moe-runner=triton, cuda-graph off |
| **kt_method** | **BF16** (AMX BF16 path on CPU; INT8 변환 unblock 시도 vs. unrelated sglang-kt index bug 회피) |
| kt-cpuinfer | 112 (physical cores) |
| kt-threadpool-count | 2 (NUMA nodes) |
| kt-num-gpu-experts | 32 of 128 per layer (= 25 % hot path on GPU) |
| kt-max-deferred-experts-per-token | 2 |
| TP | 1 |
| GPU | 1× B200 (GPU0) |
| GPU mem after measurement | 157 673 MiB (KV alloc 46.94 GB) |
| decode tps | **186.99** |
| **tps per GPU** | **186.99** |
| req/s | 0.730 |
| p50 (ms) | 8 841 |
| p99 (ms) | 9 721 |
| n_ok / n_err | 20 / 0 |
| wall (s) | 27.38 |
| total output tokens | 5 120 |

`Creating AMX_MOE_TP 0 at numa 0` / `... numa 1` 가 layer-별 로그에 확인됨 → CPU expert pool 가 NUMA 각 노드에 활성.

## 5. 비교 표 + 판정

| metric | A (full-GPU TP=2) | B (kt-kernel BF16 + AMX TP=1) | B/A | 부호 |
|---|---:|---:|---:|---|
| decode tps (per-model) | 157.94 | 186.99 | **1.184** | ↑ +18.4 % |
| **tps per GPU (cluster 효율)** | 78.97 | 186.99 | **2.368** | ⭐ **+136.8 %** |
| req/s | 0.617 | 0.730 | 1.184 | ↑ |
| p50 latency (ms) | 10 490 | 8 841 | 0.843 | ↓ −15.7 % (faster) |
| p99 latency (ms) | 11 450 | 9 721 | 0.849 | ↓ −15.1 % (faster) |
| GPU count | 2 | 1 | 0.5 | −50 % |

### 판정 (HANDOFF §3.1 기준)
- B/A (per-model decode tps) = **1.18 ≥ 1** → "abundant offload 손해" 의 negative 가설(가설 A) **기각**.
- tps-per-GPU = **+136.8 %** → 같은 cluster 의 남은 GPU 로 다른 모델/요청을 추가 서빙 가능. **가설 B 강하게 시사 — 2단계(R1-671B) 진행 후보**.

### 주의/한계 (측정값만)
- A 는 TP=2 가 over-provision 일 수 있음 (Qwen3-30B-A3B 는 1 B200 에 fit). TP=1 baseline 도 별도 측정 시 비교 의미 강화.
- B 는 BF16 path 측정. AMX INT8 path 는 본 sglang-kt 버전의 unquantized loader 버그(`unquant.py:264`) 로 즉시 차단 — INT8 BR 회피 우회 또는 다른 모델로 재시도 필요.
- warm-up 1 회만 — 추가 통계는 미수집(SHORT 프로토콜 준수).
- conc=8, max_tokens=256 → 비교적 짧은 출력 — 더 긴 decode burst 에서 신호가 같은지 별도 측정 필요.
- B 의 GPU0 메모리 158 GB 는 hot 32 expert + KV + activation 합산 — 나머지 96 expert 가 CPU(AMX)에서 수행됨을 의미.

## 6. GPU 최종 상태

`pkill -9 -f sglang` 후 모든 GPU 가 `memory.used = 0 MiB`, `nvidia-smi --query-compute-apps` 비어있음.

## 7. 산출물

- `result_A.json` — A 측정 raw + summary.
- `result_A_triton_kernel.json` — A 초기 backend mismatch 측정 (참고 보관).
- `result_B.json` — B 측정 raw + summary.
- `A_server.log` / `B_server.log` — 서버 stdout 로그.
- `short_client.py` — 측정 클라이언트(20 prompt × 256 token, conc=8).
- `run_A_baseline.sh` / `run_B_kt_offload.sh` / `run_measure.sh` — 재현 스크립트.

## 8. Next step (HANDOFF §3.2)

가설 B 유망 → **2 단계 R1-671B (DeepSeek-R1) 진행 후보**. 별도 신규 TSK_045 으로 등록 (id_registry).
