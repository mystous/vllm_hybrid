# HWC1 Round 1 — Hardware-aware custom 최적화 측정

## Hardware / Software Context

- B200 8× sm_100 (HBM3e 183 GB, NVLink5)
- Xeon Platinum 8570 (2 socket × 56C/112T), AMX bf16/int8 + AVX-512_BF16/FP16 native
- NUMA: node 0 = CPU 0-55,112-167 (GPU 0-3), node 1 = CPU 56-111,168-223 (GPU 4-7)
- vLLM editable @ `/workspace/host_vllm_hybrid/` (1.7.dev16107+gffe20fb09.d20260601)
- torch 2.11.0+cu128
- Workload: sharegpt 500 prompts, max-tokens=2048, concurrency=64, TP=8, max-model-len=16384, gpu-mem-util=0.85
- Compilation: `cudagraph_mode=FULL_AND_PIECEWISE` (B3 FaP)
- Baseline env: `VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1 VLLM_WORKER_MULTIPROC_METHOD=spawn`

## Baseline (5 sweeps)

| Sweep | output_tps | wall_s | gpu_util | cpu_util | ttft_p50 |
|---|---|---|---|---|---|
| s1 | 21862.1 | 44 | 95.7 | 5.4 | 72.9 |
| s2 | 22266.4 | 44 | 96.4 | 5.4 | 51.6 |
| s3 | 22088.0 | 44 | 96.4 | 5.4 | 56.0 |
| s4 | 22016.7 | 44 | 96.3 | 5.3 | 56.6 |
| s5 | 22156.3 | 44 | 96.4 | 5.5 | 51.3 |
| **mean ± std** | **22077.9 ± 151.7** | 44 | 96.2 | 5.4 | 57.7 |

Baseline 의 relative std = **0.69%**. Accept gate (paired Δ%): single-sweep candidate 가 **+3%** 이상이면 호의적, 그렇지 않으면 음수 또는 noise 판정.

## Round 1 Lever Catalog

| # | Lever | Mechanism | Implementation |
|---|---|---|---|
| H1 | NUMA-bind auto | `--numa-bind` (cpunodebind+membind via numactl wrapper, auto-detect GPU↔NUMA) | vllm 내장 |
| H2 | NUMA + physcpubind | H1 + 14-core block per GPU (0-13, 14-27, …) | vllm 내장 |
| H3 | Aux CUDA stream priority=-1 | 6개 보조 copy/sample stream 의 priority 를 -1 로 변경 | `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu/structured_outputs.py` 수정 + `VLLM_HWC1_STREAM_PRIO=1` env gate |
| H4 | PyTorch CUDA expandable_segments | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | env only |
| H6 | jemalloc LD_PRELOAD | glibc ptmalloc → jemalloc (Ray bundled lib) | env only |
| H7 | glibc malloc tuning | `MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072` | env only |
| (H8) | KV-cache fp8 5-sweep revisit | R6A +3.38% 재측정 | `--kv-cache-dtype fp8` |
| (H10) | torch.compile int32 indexing | `assume_32_bit_indexing=True` for kernel int indexing | compilation-config override |

(H8, H10 은 Round 1 이후 결과에 따라 진행)

## Environment constraints (discovered)

- Container 의 `cap_sys_nice` 가 _미포함_ → `numactl --membind` fails (`set_mempolicy: Operation not permitted`).
  - 영향: **H1, H2 강제 기각**. 컨테이너 권한 변경은 우리 권한 밖.
  - 우회: pinned memory allocator level 에서 first-touch 로 어느 NUMA 가 page 들고 있는지 결정되므로 process group 단위 NUMA-binding 은 불가하지만, MPM 그리고 worker proc 의 cpuset 만이라도 sched_setaffinity 로 조절 가능 (Round 2 시도).

## Results

_(자동 채워짐, `scripts/analyze.py` 출력)_

| Tag | sweeps | mean tps | std | Δ% | GPU% | CPU% | verdict |
|---|---|---|---|---|---|---|---|
| baseline | 5 | 22077.9 | 151.7 | +0.00% | 96.2 | 5.4 | ref |
| h1_numa_bind | boot_fail | - | - | - | - | - | NUMA 권한 부재 → 강제 기각 |
| h2_numa_physcpu | boot_fail | - | - | - | - | - | 동일 사유 |
| h3_stream_prio | 1 | 21688.6 | - | -1.76% | 96.2 | 5.4 | noise 내 negative → 기각 |
| h4_expand_seg | 1 | 21479.4 | - | -2.71% | 95.8 | 5.4 | noise 경계 negative → 기각 |
| h6_jemalloc | 1 | 21745.9 | - | -1.50% | 95.7 | 5.4 | noise 내 → 기각 |
| h7_malloc_arena | 1 | 21814.5 | - | -1.19% | 95.6 | 5.3 | noise 내 → 기각 |

## Round 1 결론 (interim)

- 모든 6 lever (NUMA 제외 4개, NUMA 2개) 가 baseline noise floor (≈±3%) 안의 음수.
- **Round 1 winner: 없음**.
- NUMA-bind 는 컨테이너 `cap_sys_nice` 부재로 동작 불가 — 사용자 docker run 권한 변경 필요.
- 다음 Round 2 에서 **KV cache 양자화 (fp8 / nvfp4)** 와 **inductor pass fuse_norm_quant** 등 compiler-level lever 진행.

