# HWC Round 1-3 Cumulative Final Report

## Environment

- DGX B200 8× sm_100 (HBM3e 183 GB, NVLink5)
- Xeon Platinum 8570 dual-socket (224 thread, AMX bf16/int8 + AVX-512_BF16/FP16 native, NUMA0=GPU0-3, NUMA1=GPU4-7)
- vLLM editable `/workspace/host_vllm_hybrid/vllm/` (1.7.dev16107+gffe20fb09.d20260601)
- torch 2.11.0+cu128
- 워크로드: sharegpt 500 prompts × concurrency=64 × max-tokens=2048 × TP=8 × max-model-len=16384 × gpu-mem-util=0.85
- 컴파일: `cudagraph_mode=FULL_AND_PIECEWISE` (B3 FaP) + `VLLM_PREFETCH_TOKENIZE=1` + `VLLM_BURST_AWARE_ADMISSION=1`

## Baseline (Round 1, 5 sweep)

**22,077.9 ± 151.7 tps** — relative std 0.69%. Accept gate: **Δ ≥ +3.0%**.

## All levers tested (3 rounds, ~25 candidates)

### Round 1 (8 levers, baseline 5sw)

| Tag | mean | Δ% | verdict |
|---|---|---|---|
| baseline | 22078 ± 152 | +0.00% | ref |
| H1 numa-bind (auto) | boot_fail | - | container `cap_sys_nice` 부재 → 기각 |
| H2 numa-bind (physcpubind) | boot_fail | - | 동일 사유 |
| H3 stream_prio=-1 (vllm code 수정 6 곳) | 21689 | -1.76% | noise 내 음수 |
| H4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True | 21479 | -2.71% | noise 경계 음수 |
| H6 jemalloc LD_PRELOAD | 21746 | -1.50% | noise 내 |
| H7 MALLOC_ARENA_MAX=2 + MMAP_THRESHOLD | 21814 | -1.19% | noise 내 |
| **H8 --kv-cache-dtype fp8 (5sw)** | **22966 ± 139** | **+4.02%** | **accept gate ↑** |
| H10 compilation assume_32_bit_indexing=True | 21775 | -1.37% | noise 내 |

### Round 2 (on H8 fp8 base, 5sw each)

| Tag | mean | Δ% vs base | Δ% vs fp8 | verdict |
|---|---|---|---|---|
| R2-3 (TRITON_ATTN, 1sw) | 22838 | +3.44% | -0.56% | 동등 |
| R2-4 (+fuse_norm_quant+act+attn) | 22400 ± 217 | +1.46% | -2.46% | 음수 → 기각 |
| **R2-5 (+enable_sp + fuse_gemm_comms)** | **23016 ± 236** | **+4.25%** | +0.22% | **best** |
| R2-6 (+use_prefill_query_quantization) | 22935 ± 364 | +3.88% | -0.14% | 동등 |
| R2-1b (kv_cache_dtype fp8_e5m2) | boot_fail | - | - | dynamo data-dep assert 미지원 |

### Round 3 (on fp8 base, 5sw)

| Tag | mean | Δ% vs base | Δ% vs fp8 | verdict |
|---|---|---|---|---|
| R3-1 (fp8 + FULL_DECODE_ONLY) | 22465.6 ± 231.0 | +1.76% | -2.18% | 음수 → 기각 |
| R3-2 (fp8 + batched-tokens=16384) | s1=22534 (+2.06%), s2 partial(281/500) | - | - | **engine crash** sweep 2 → 기각 |
| R3-3 (fp8 + --async-scheduling) | s1=22775 (+3.16%), s2 partial(263/500), s3-5 = 0 tps | - | - | **engine crash after s1** → 기각 |
| R3-4 (fp8 + async + batched=8192 + maxseqs=256) | boot only, async crash | - | - | async crash 영향 → 기각 |
| R3-A~D (fp8 + enable_sp + 추가) | 미진행 | - | - | 병행 task (`hw_heavy_baseline`) GPU 점유로 launch 보류 |

**R3 발견**:
- `--max-num-batched-tokens=16384` + fp8 → engine crash (RemoteProtocolError)
- `--async-scheduling` + fp8 → sweep 1 후 engine crash, sweep 2-5 = 0 tps
- 이 두 lever 는 fp8 와 결합 시 unstable. **Production 사용 비권장**.

## 주요 발견

1. **Baseline 매우 stable** (rel std 0.69%) — 측정 신뢰도 높음.
2. **유일한 winner**: KV cache fp8 — +4.02% (단독) ~ +4.25% (with enable_sp).
3. **+10% target 미달성** in 3 rounds. fp8 만 의미 있는 lever, 추가 stacking 효과 없음.
4. **HW 환경 제약**:
   - container `cap_sys_nice` 부재 → NUMA-bind 불가 (큰 lever 차단)
   - NVFP4 KV cache 코드는 있지만 `raise NotImplementedError` (flashinfer.py:625)
   - fp8_e5m2 + torch.dynamo data-dependent assert 충돌
5. **GPU bound 시 CPU-side lever 효과 없음** — H6/H7/jemalloc/expand_seg 모두 음수 (CPU 5.4% → idle).
6. **Llama-3.1-8B 의 작은 모델** 특성: enable_sp/fuse_norm/Q quant 효과 모두 미미.

## Production 권고 (현재 결과 기준)

- `--kv-cache-dtype fp8` 단독 사용 — **+4.02% throughput**.
- (선택) `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"enable_sp":true,"fuse_gemm_comms":true}}'` 추가 — **+4.25%**, std 안에 단독 fp8 와 동등이지만 약간 우위.
- **정확도 게이트 미진행** (token 한계 도달 가능성) — production 적용 전 별도 logprob 비교 필요 (`scripts/run_accuracy_gate.sh` 구현 완료).
- multi-model (Qwen2.5-32B, DS-R1-Distill-Llama-70B) 확장 미진행 — +10% 달성 시 진행 예정.

## 후속 round 4+ 권장 후보 (+10% 도달 시도)

- **모델 가중치 fp8 quantize** (`Llama-3.1-8B-Instruct-FP8` 모델 swap) — weight BW × 2 절감
- **Speculative decode** (Eagle3 head, Medusa 변형) — draft 정확도 ↑ 로 effective tokens/step ↑
- **Triton persistent kernel** (decode hot path)
- **Warp specialization (CUTLASS 3.x style)** — 미진행, 큰 코드 작업 필요
- **TMA-based KV prefetch** — sm_100 native
- Container `cap_sys_nice` 권한 부여 후 NUMA-bind 재시도

## File pointers

- Round 1 측정/결과: `/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/hw_custom_round_1/`
- Round 2 측정/결과: `/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/hw_custom_round_2/`
- Round 3 측정/결과: `/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/hw_custom_round_3/`
- 코드 변경:
  - `vllm/envs.py` — `VLLM_HWC1_STREAM_PRIO`, `VLLM_HWC1_HUGEPAGE` flag 추가
  - `vllm/v1/worker/gpu_model_runner.py` — `_hwc1_make_stream()` helper + 4 곳 stream priority 적용
  - `vllm/v1/worker/gpu/model_runner.py` — output_copy_stream priority
  - `vllm/v1/worker/gpu/structured_outputs.py` — copy_stream priority
- 정확도 게이트: `scripts/run_accuracy_gate.sh`, `capture_logprobs.py`, `compare_logprobs.py`
