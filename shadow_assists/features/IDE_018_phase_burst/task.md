# task.md — IDE_018 단계별 구현

## TSK_031 — Phase detection mechanism

### Step 1: CUDA event hook 위치 정의
- attention layer entry / exit (per-decode-step, 16 layers for Qwen 32B)
- linear (matmul) entry / exit
- sampling entry / exit
- TP all-reduce (NCCL) entry / exit

### Step 2: vLLM patch
- vllm/v1/worker/gpu_model_runner.py 의 `_execute_model` 에 cuda event 삽입
- per-layer 또는 per-block granularity (overhead 측정 후 결정)

### Step 3: IPC primitive (phase signal CPU pool 으로 전달)
- shared memory atomic counter (lockless)
- eventfd or futex notify (low latency)
- alternative: VFIO-style poll loop on CPU task pool side

### Step 4: phase signal latency benchmark
- target < 50 μs per signal (cuda event timestamp → CPU pool dispatch)
- microbench: synthetic workload with known phase durations

## TSK_032 — Attention-phase CPU task pool (memory-bound GPU idle)

### Step 1: task A — schedule next batch (AVX-512 metadata)
- depends: TSK_024 AVX-512 scheduler vectorize
- target: per-step 1-3 ms

### Step 2: task B — detokenize previous step output (AVX-512)
- depends: TSK_024 AVX-512 tokenizer
- target: per-batch 1-5 ms

### Step 3: task C — grammar / constraint check (XGrammar offload)
- existing XGrammar 통합
- target: per-token 2-10 ms

### Step 4: task D — request classifier (SUB_076 PoC)
- existing IDE_012/SUB_076 classifier 재사용
- target: per-request 1 ms

## TSK_033 — Linear-phase CPU task pool (compute-bound GPU idle)

### Step 1: task E — KV prefetch via DMA pull (TSK_028 pinned pool)
- depends: IDE_017 TSK_028
- target: per-chunk 60 μs (SUB_166)

### Step 2: task F — speculative draft (AMX draft head)
- depends: IDE_016 TSK_026 + IDE_019 TSK_036
- target: per-batch ≤ 5 ms

### Step 3: task G — cold-KV decompress (TSK_030)
- depends: IDE_017 TSK_030
- target: per-chunk 5-20 ms

## TSK_034 — Integration + measurement ★ paper main result

### Step 1: phase-burst scheduler (CPU task queue + dispatch)
- src/phase_burst/scheduler.cpp
- worker thread pool (20-32 threads pinned to cpu 80-99 / 80-111)
- priority queue (per-phase + per-task-type)

### Step 2: end-to-end Qwen 32B TP=4×2 + phase-burst CPU on canonical 3 mix
- ENV `VLLM_USE_PHASE_BURST=1`
- SUB_098/SUB_160 protocol 으로 measurement
- monitor.py background capture

### Step 3: paper main figure measurement
- CPU util baseline 4.1% → IDE_018 target **30%+**
- throughput delta (vs SUB_098 baseline)
- GPU util delta (vs baseline)
- per-phase task pool occupancy (which tasks ran when)
