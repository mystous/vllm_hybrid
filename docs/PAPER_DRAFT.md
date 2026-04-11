# Harvesting Idle CPU Cycles for LLM Inference: A Dual-Process Parallel-Batch Architecture for CPU-GPU Heterogeneous Serving

> ⚠️ **This is an early draft. The current single source of truth for the design is
> `docs/paper/main.tex`.** Where this draft diverges from main.tex (notably: the
> `cpu_max_num_seqs` auto rule, the `num_cpu_engines` derivation, the `_C_utils`
> build target, and the CUDA 13 / torch 2.9 toolchain), main.tex is authoritative.
>
> **IEEE Transactions on Parallel and Distributed Systems (TPDS) — Draft**
> Status: Draft v2.1 — last refreshed 2026-04-11 to align with current code principles

---

## Authors

[Author 1], [Author 2], ...
[Institution]
{email1, email2}@institution.ac.kr

---

## Abstract

Modern GPU-based large language model (LLM) serving systems leave hundreds of CPU cores and terabytes of DRAM idle on host servers, representing a significant waste of capital expenditure. We present a **Dual-Process Parallel-Batch** architecture that harvests these idle CPU cycles to serve additional LLM requests *in parallel* with the GPU, achieving a total throughput of T_hybrid = T_GPU + α·T_CPU without degrading GPU latency. The key insight is that **process-level isolation**—running GPU and CPU inference engines as separate OS processes—simultaneously eliminates Python GIL contention and guarantees zero interference with the GPU pipeline. We make four contributions: (1) a dual-process engine architecture with a formal throughput model proving that hybrid throughput is additive under process isolation; (2) a **CapacityAwareRouter** with self-regulating properties that maximizes CPU utilization without requiring a priori knowledge of CPU processing speed; (3) a zero-configuration CPU optimization pipeline that automatically detects NUMA topology, Intel ISA extensions (AVX-512 VNNI, AMX), and derives optimal inference parameters; and (4) specialized AVX-512 VNNI C++ kernels for INT8 GEMM, batched attention, and quantization. Implemented atop vLLM, our system targets NVIDIA H100 ×8 servers with Intel Xeon 8480+ (112 cores, 2 TB DDR5). We present a theoretical throughput model validated by roofline analysis and design eight controlled experiments to measure end-to-end throughput, GPU latency impact, router strategy effectiveness, and ablation across NUMA, IPEX, and model-size dimensions. Theoretical analysis predicts 1–5% throughput gain for 70B-parameter models and 5–15% for 7B models—obtained at zero additional hardware cost.

**Keywords**: LLM inference, heterogeneous computing, CPU-GPU parallelism, PagedAttention, AVX-512, NUMA, vLLM, process isolation

---

## I. Introduction

### A. The Idle CPU Problem

GPU datacenters designed for large language model (LLM) inference harbor a hidden inefficiency: the host CPUs sit almost entirely idle. A typical NVIDIA DGX H100 node pairs eight H100 GPUs (640 GB HBM3, 26 PFLOPS aggregate) with two Intel Xeon Platinum 8480+ processors (112 physical cores, 2 TB DDR5-4800 DRAM)—yet during GPU-only LLM serving, CPU utilization rarely exceeds 5% [8]. At a system price exceeding \$300,000, the CPU and memory subsystem alone accounts for roughly \$30,000–50,000 of capital expenditure per node, translating to millions of dollars of idle silicon across a moderately-sized GPU cluster.

This waste is not merely theoretical. DDR5-4800 memory in an 8-channel configuration delivers approximately 307 GB/s of aggregate bandwidth per socket—sufficient to sustain memory-bound autoregressive decode for quantized LLMs. The question that motivates this work is:

> **Can we harvest idle CPU cycles to serve additional LLM requests without degrading GPU throughput or latency?**

### B. Why This Is Not Trivial

Answering this question affirmatively requires overcoming four interrelated challenges:

1. **GIL contention.** Python's Global Interpreter Lock (GIL) prevents truly concurrent execution of GPU and CPU inference loops within a single process. Any attempt to run both engines in one process serializes their Python-level coordination, turning `T_GPU + T_CPU` into `T_GPU + T_CPU + T_overhead`.

2. **Speed asymmetry.** GPU inference throughput exceeds CPU throughput by roughly two orders of magnitude for large models. A naïve round-robin router would either starve the CPU or overwhelm it, requiring careful load-aware routing.

3. **NUMA complexity.** Multi-socket servers exhibit non-uniform memory access (NUMA) topologies. Cross-socket memory access incurs ~2× latency and ~0.5× bandwidth penalties [8, 9], demanding careful thread and memory affinity management.

4. **Interference risk.** Any CPU inference activity that contends for shared resources (PCIe bandwidth, LLC, memory controller queues) with the GPU pipeline risks degrading GPU latency—violating the "do no harm" requirement.

### C. Key Insight: Process-Level Isolation

Our central observation is that **running GPU and CPU engines as separate OS processes** resolves challenges 1, 3, and 4 simultaneously:

- **Independent GILs**: Each process has its own Python interpreter and GIL, enabling truly concurrent busy loops.
- **Independent address spaces**: CPU process memory allocations (model weights, KV cache) are isolated from GPU process memory, eliminating LLC and memory controller contention.
- **Independent scheduling**: The OS schedules each process independently; NUMA affinity, CPU pinning, and CUDA isolation (`CUDA_VISIBLE_DEVICES=""`) are set per-process.

Under process isolation, the system throughput becomes:

$$T_{\text{hybrid}} = T_{\text{GPU}} + \alpha \cdot T_{\text{CPU}}, \quad \alpha \in [0, 1]$$

where α represents the router's efficiency in keeping the CPU fully utilized. Our CapacityAwareRouter ensures α → 1 under sustained load.

### D. Contributions

This paper makes four contributions:

1. **Dual-Process Parallel-Batch Architecture.** We design a system in which GPU and CPU each run a complete, independent EngineCore instance in separate OS processes, communicating via ZeroMQ IPC. We provide a formal throughput model (Section IV-B) proving that hybrid throughput is additive and that GPU tail latency is unaffected.

2. **CapacityAwareRouter with Self-Regulating Properties.** We propose a capacity-based routing algorithm that tracks in-flight CPU requests and routes new requests to the CPU only when slots are available. We prove its self-regulating property: without requiring any knowledge of CPU processing speed, the router automatically maximizes CPU utilization while guaranteeing zero GPU throughput degradation. We further extend it with length-aware and throughput-adaptive variants incorporating EMA-based dynamic slot adjustment.

3. **Zero-Configuration CPU Optimization Pipeline.** We implement automatic detection of NUMA topology, Hyper-Threading, Intel ISA extensions (AVX-512 VNNI, AMX-BF16/INT8), and IPEX availability, deriving optimal CPU inference parameters (thread count, KV cache size, batch size, OpenMP affinity) without manual tuning.

4. **AVX-512 VNNI Specialized Kernels.** We develop five C++ kernels optimized for CPU inference: a 6×16 VNNI INT8 GEMM micro-kernel with Goto-style cache blocking, a 16-sequence batched attention kernel with online softmax, Q8_0 quantization, decode GEMV, and NUMA-aware memory operations with cache-line prefetching.

### E. Results Preview

Theoretical roofline analysis predicts that on an Intel Xeon 8480+ with DDR5-4800, the CPU can decode a Q8_0-quantized LLaMA 3 70B model at approximately 2–5 tok/s, and a 7B model at approximately 15–40 tok/s. Against GPU throughput of ~100 tok/s (70B, TP=8) and ~500 tok/s (7B, single GPU), this corresponds to 1–5% and 3–8% additional throughput respectively—obtained at zero marginal hardware cost. We design eight controlled experiments (Section VI) to validate these predictions and measure GPU latency impact, router strategy effectiveness, and component-level ablation.

---

## II. Background and Motivation

### A. LLM Inference Characteristics

LLM inference proceeds in two distinct phases with fundamentally different computational profiles [22, 24]:

**Prefill phase** (compute-bound). Given an input prompt of *n* tokens, the model processes all tokens in parallel through the full transformer stack, generating the KV cache entries. The computational cost scales as O(n · d²) where d is the model dimension, and the operation is dominated by large matrix multiplications that saturate GPU compute units.

**Decode phase** (memory-bound). Each subsequent token is generated autoregressively: the model reads the full KV cache and model weights to produce one token, then appends the new KV entries. The arithmetic intensity drops to O(1) per weight byte read, making decode throughput fundamentally limited by memory bandwidth:

$$T_{\text{decode}} \leq \frac{B_{\text{mem}}}{S_{\text{model}} / q}$$

where B_mem is memory bandwidth, S_model is model size in bytes, and q is the quantization compression ratio. This memory-bound nature is critical: it means that **any device with sufficient memory bandwidth can contribute to decode throughput**, including CPUs.

**Quantitative analysis for CPU decode.** For an Intel Xeon 8480+ with DDR5-4800 in 8-channel configuration:

$$B_{\text{DDR5}} = 8 \times 4800 \times 10^6 \times 8 \text{ bytes} = 307.2 \text{ GB/s}$$

For LLaMA 3 70B with Q8_0 quantization (q = 2, reducing 16-bit to 8-bit):

$$T_{\text{CPU,70B}} \leq \frac{307.2 \text{ GB/s}}{70 \times 10^9 / 2 \text{ bytes}} \approx 8.8 \text{ tok/s (upper bound)}$$

Accounting for practical overheads (attention computation, memory access patterns, OS overhead), we estimate an achievable range of 2–5 tok/s. For LLaMA 3 8B:

$$T_{\text{CPU,8B}} \leq \frac{307.2 \text{ GB/s}}{8 \times 10^9 / 2 \text{ bytes}} \approx 76.8 \text{ tok/s (upper bound)}$$

with a practical estimate of 15–40 tok/s. These estimates demonstrate that CPU-based decode is feasible and non-trivial, particularly for smaller models.

### B. The Idle CPU Problem: A Quantitative Analysis

Table I quantifies the resource composition and utilization of a representative DGX H100 server during GPU-only LLM serving.

**Table I. Resource utilization during GPU-only LLM serving on DGX H100**

| Resource | Specification | Utilization | Idle Capacity |
|----------|--------------|-------------|---------------|
| GPU Compute | 8× H100 SXM, 26 PFLOPS | 60–90% | — |
| GPU Memory | 8× 80 GB HBM3 | 70–95% | — |
| CPU Compute | 2× Xeon 8480+, 112 cores | < 5% | > 106 cores |
| System Memory | 2 TB DDR5-4800 | < 10% | > 1.8 TB |
| DDR5 Bandwidth | 614 GB/s (2-socket) | < 3% | > 595 GB/s |

The CPU subsystem represents approximately 10–15% of the total system cost (\$30,000–50,000 out of \$300,000+). Across a 100-node cluster, this amounts to \$3–5 million of idle hardware. More importantly, the idle DDR5 bandwidth (595+ GB/s) is sufficient to sustain multiple concurrent decode streams for quantized LLMs.

**Total Cost of Ownership (TCO) perspective.** Over a typical 3-year depreciation cycle with 1.5× operational cost multiplier (power, cooling, space), the wasted CPU capacity costs:

$$C_{\text{waste}} = N_{\text{nodes}} \times C_{\text{CPU+mem}} \times (1 + r_{\text{ops}}) = 100 \times \$40\text{K} \times 2.5 = \$10\text{M}$$

Even a 3% throughput improvement from harvesting idle CPU cycles yields \$300K of equivalent GPU capacity—more than justifying the software engineering investment.

### C. Why Existing Approaches Fall Short

Existing heterogeneous LLM inference systems can be categorized by their CPU-GPU work-splitting granularity:

**Neuron-level splitting.** PowerInfer [3] exploits the power-law distribution of neuron activations, placing "hot" neurons on GPU and computing "cold" neurons on CPU. While achieving up to 11.69× speedup over llama.cpp, this approach executes CPU and GPU computations **within the same inference pipeline**, creating synchronization barriers at every layer.

**Expert-level splitting.** KTransformers [2] targets Mixture-of-Experts (MoE) models, routing inactive experts to CPU while active experts execute on GPU. This achieves 4.62–19.74× prefill speedup for DeepSeek V3/R1 but requires **tight CPU-GPU coordination** within each forward pass.

**Layer/tensor-level splitting.** HeteGen [5] and FlexGen [31] partition model layers or tensor dimensions across CPU and GPU, using asynchronous overlap to hide data transfer latency. These approaches fundamentally create **sequential dependencies** between CPU and GPU computation.

**Table II. Comparison of heterogeneous LLM inference approaches**

| Approach | Granularity | Parallelism | GIL-Free | GPU Interference | CPU Autonomy |
|----------|-------------|-------------|----------|-----------------|--------------|
| PowerInfer [3] | Neuron | Intra-request | No | Yes (sync barriers) | No |
| KTransformers [2] | Expert | Intra-request | No | Yes (data transfer) | No |
| HeteGen [5] | Layer/Tensor | Intra-request | No | Yes (pipeline) | No |
| FlexGen [31] | Layer/Tensor | Intra-request | No | Yes (offloading) | No |
| **This work** | **Request** | **Inter-request** | **Yes** | **No (process isolation)** | **Yes** |

The common limitation of prior work is **intra-request splitting**: CPU and GPU cooperate to serve *the same request*, creating synchronization points that cause GPU stalls. Our approach is fundamentally different—we perform **inter-request splitting**: CPU and GPU each serve *independent requests* in separate processes, eliminating all synchronization and enabling truly additive throughput.

### D. Key Insight: Process-Level Parallelism

The insight that distinguishes our approach is the recognition that **process-level isolation** transforms heterogeneous inference from a coordination problem into an independence problem:

**Single-process model (prior work):**
$$T_{\text{total}} = T_{\text{GPU}} + T_{\text{CPU}} + T_{\text{sync}} \quad \text{(serialized by GIL + synchronization)}$$

**Dual-process model (this work):**
$$T_{\text{total}} = \max(T_{\text{GPU}}, T_{\text{CPU}}) \quad \text{(fully parallel, independent processes)}$$

Since T_GPU >> T_CPU for large models, the GPU is always the bottleneck, and:

$$\text{Throughput}_{\text{hybrid}} = \text{Throughput}_{\text{GPU}} + \text{Throughput}_{\text{CPU}}$$

This additive property holds because:
1. **No shared GIL**: Each process has its own Python interpreter.
2. **No shared compute resources**: CPU process is bound to specific NUMA node; GPU process offloads all computation to CUDA devices.
3. **No shared memory bus contention**: CPU process accesses DDR5 via local NUMA path; GPU process accesses HBM3 via NVLink/PCIe.
4. **No shared control flow**: Each process runs an independent `while True: poll → schedule → execute → push` busy loop.

The only shared resource is the ZMQ IPC channel, which handles lightweight request/response metadata with O(μs) latency—negligible compared to inference latency of O(ms) to O(s).

---

## III. Related Work

### A. LLM Serving Systems

The modern LLM serving stack builds on two foundational innovations. **Continuous batching** (Orca [22]) replaces static request-level batching with iteration-level scheduling, enabling new requests to enter the batch as soon as previous ones complete a generation step, achieving 36.9× throughput improvement for GPT-3 175B. **PagedAttention** (vLLM [1]) manages KV cache as non-contiguous fixed-size blocks inspired by OS virtual memory paging, nearly eliminating memory fragmentation and delivering 2–4× throughput over prior systems. SGLang [6] adds RadixAttention for automatic KV cache reuse across shared prefixes and structured output decoding optimization. SARATHI [23] and Sarathi-Serve [24] introduce chunked prefill to eliminate GPU utilization imbalance between prefill and decode phases, achieving stall-free scheduling.

### B. Heterogeneous LLM Inference

Beyond the systems discussed in Section II-C, the heterogeneous inference landscape includes FlexGen [31], which aggregates GPU/CPU/disk memory for high-throughput offline inference of OPT-175B on a single 16 GB GPU, and PowerInfer-2 [4], which extends neuron-level CPU-NPU splitting to mobile devices. A recent development is Intel's native C++ backend for SGLang [6a], which runs DeepSeek R1 on Intel Xeon 6 processors using AMX-based kernels. All of these operate within a single-process boundary, in contrast to our inter-process parallel approach.

### C. CPU-Optimized LLM Inference

CPU inference optimization spans three axes: **ISA exploitation** (AVX-512 VNNI for 64 INT8 MACs/cycle [12], AMX for 2,048 INT8 ops/cycle via tile-based matrix multiply [13]); **memory hierarchy optimization** (NUMA-aware allocation [9, 11], cache blocking [39]); and **specialized engines** (llama.cpp [32] with 1.5–8 bit quantization, xFasterTransformer [7] with INT8 KV cache and SlimAttention, IPEX [28] with transparent AMX/VNNI utilization). Na et al. [8] provide a systematic characterization of CPU LLM inference at IISWC 2024, concluding that single-NUMA-node binding is optimal for throughput.

### D. Disaggregated Serving

DistServe [20] separates prefill and decode into distinct GPU pools to eliminate inter-phase interference, achieving 7.4× higher request processing. Splitwise [21] (ISCA 2024 Best Paper) extends this to heterogeneous hardware (H100 for prefill, A100 for decode), achieving 1.4× throughput and 20% cost reduction. Our approach is complementary: we add CPU-based decode as a third processing tier alongside GPU prefill and GPU decode.

### E. Speculative Decoding and MoE Offloading

Speculative decoding [14, 15] uses a fast draft model to propose tokens verified in parallel by the target model, achieving 2–3× decode speedup. MoE offloading systems (DeepSpeed-MoE [16], MoE-Infinity [18]) exploit expert sparsity to reduce GPU memory requirements. Both represent future extension points for our dual-process architecture (Section VII).

---

## IV. System Design

### A. Architecture Overview

Figure 1 presents the Dual-Process Parallel-Batch architecture.

```
                    ┌──────────────────────────────────────┐
                    │       HybridAsyncMPClient            │
                    │   ┌────────────────────────────┐     │
                    │   │    CapacityAwareRouter      │     │
  Client            │   │  ┌──────────┐ ┌──────────┐ │     │
  Requests ─────────┤   │  │ slots<N  │ │ else     │ │     │
                    │   │  │  → CPU   │ │  → GPU   │ │     │
                    │   │  └──────────┘ └──────────┘ │     │
                    │   └──────────┬─────────────────┘     │
                    └──────────┬──┼────────────────────────┘
                               │  │
                      ZMQ IPC  │  │  ZMQ IPC
                    (ROUTER)   │  │  (ROUTER)
                               │  │
               ┌───────────────┘  └────────────────┐
               ▼                                   ▼
 ┌──────────────────────────┐    ┌───────────────────────────┐
 │   GPU EngineCoreProc     │    │   CPU EngineCoreProc      │
 │   (Process A, PID α)     │    │   (Process B, PID β)      │
 │                          │    │                           │
 │   EngineCore             │    │   EngineCore              │
 │   ├─ Scheduler           │    │   ├─ Scheduler            │
 │   ├─ MultiprocExecutor   │    │   ├─ UniProcExecutor      │
 │   │   (8× H100, TP=8)   │    │   │   (CPUWorker)         │
 │   └─ KV Cache (HBM3)    │    │   └─ KV Cache (DDR5)      │
 │                          │    │       NUMA-aware           │
 └──────────┬───────────────┘    └──────────┬────────────────┘
            │                               │
            │       ZMQ IPC (PUSH/PULL)     │
            └───────────────┬───────────────┘
                            ▼
                    Output Aggregation
```
**Fig. 1.** Dual-Process Parallel-Batch architecture. GPU and CPU engines run as independent OS processes, each executing a complete EngineCore instance with its own scheduler, executor, and KV cache.

The architecture is guided by four design principles, each targeting a specific challenge:

**Table III. Design principles and the challenges they address**

| Principle | Challenge Addressed | Mechanism |
|-----------|-------------------|-----------|
| P1: Process Isolation | GIL contention (C1), Interference (C4) | Separate PID, GIL, address space |
| P2: Minimal Modification | Maintainability, upstream compatibility | Hybrid code in 2 files only (hybrid_core.py, core_client.py); core.py untouched |
| P3: Capacity-Based Routing | Speed asymmetry (C2), unknown CPU speed | CapacityAwareRouter: slot-based, self-regulating |
| P4: Zero Configuration | NUMA complexity (C3), deployment friction | Auto-detect topology, ISA, derive params |

### B. Throughput Model

We formalize the throughput properties of the dual-process architecture.

**Definition 1** (Hybrid throughput). Let T_GPU and T_CPU denote the steady-state throughput (requests/second) of the GPU and CPU engines respectively when operating independently. The hybrid system throughput is:

$$T_{\text{hybrid}} = T_{\text{GPU}} + \alpha \cdot T_{\text{CPU}}$$

where α ∈ [0,1] is the **CPU utilization factor**, representing the fraction of time the CPU engine is actively processing requests.

**Theorem 1** (Additive throughput under CapacityAwareRouter). Under the CapacityAwareRouter with N_CPU available slots, if the request arrival rate λ satisfies λ > T_GPU (i.e., GPU alone cannot serve all requests), then α → 1 and:

$$T_{\text{hybrid}} \to T_{\text{GPU}} + T_{\text{CPU}}$$

*Proof sketch.* When λ > T_GPU, the GPU cannot absorb all incoming requests. The excess requests fill CPU slots via CapacityAwareRouter. Since the router routes to CPU whenever cpu_in_flight < N_CPU, and request completions free slots, the CPU operates as a loss-free M/G/N_CPU queue. When λ - T_GPU ≥ T_CPU, all CPU slots are perpetually occupied, yielding α = 1. When T_GPU < λ < T_GPU + T_CPU, the CPU utilization is α = (λ - T_GPU)/T_CPU < 1, but the system serves all requests without queueing at the GPU. □

**Corollary 1** (GPU latency preservation). Since GPU and CPU engines run in separate processes with independent schedulers and executors:

$$p_{99}(L_{\text{GPU,hybrid}}) = p_{99}(L_{\text{GPU,standalone}}) + O(\mu s)$$

The only additional latency is the ZMQ routing decision (O(1) integer comparison) and IPC message overhead (O(μs)), both negligible compared to inference latency (O(ms)–O(s)).

**Corollary 2** (Throughput bound). The maximum achievable hybrid throughput is bounded by:

$$T_{\text{hybrid}} \leq T_{\text{GPU}} + \min\left(T_{\text{CPU}}, \frac{B_{\text{DDR5}}}{S_{\text{model}}/q}\right)$$

where the second term is the memory-bandwidth roofline for CPU decode.

**Overhead analysis.** The system introduces three sources of overhead:
1. **Routing decision**: O(1) integer comparison per request (~10 ns).
2. **ZMQ IPC**: Request metadata serialization/deserialization (~10–50 μs per message).
3. **Model weight duplication**: CPU loads its own copy of model weights into DDR5 (e.g., ~35 GB for 70B Q8_0).

Overheads 1 and 2 are negligible relative to per-request inference time. Overhead 3 is a one-time memory cost that is amortized over the server's lifetime and is minor relative to the 2 TB available DDR5.

### C. CapacityAwareRouter

The CapacityAwareRouter is the central component that bridges the GPU and CPU engines. Unlike ratio-based routers that require pre-configured splitting ratios, it operates on a simple capacity principle.

#### C.1 Basic Algorithm

```
Algorithm 1: CapacityAwareRouter — Basic Capacity Routing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
State: cpu_in_flight ← 0, N ← num_cpu_engines × cpu_max_num_seqs
                               = num_numa_nodes × 1
                               (one slot per NUMA-bound CPU engine)

function ROUTE(request r):
    if cpu_in_flight < N then
        cpu_in_flight ← cpu_in_flight + 1
        return CPU
    else
        return GPU

function ON_COMPLETE(request r, target):
    if target = CPU then
        cpu_in_flight ← max(0, cpu_in_flight - 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### C.2 Self-Regulating Property

The router exhibits a **self-regulating** property analogous to a feedback control system:

**Property 1** (Self-regulation). Let μ_CPU denote CPU processing rate (completions/second). The steady-state CPU utilization U_CPU satisfies:

$$U_{\text{CPU}} = \frac{\min(\lambda_{\text{excess}}, N \cdot \mu_{\text{CPU}})}{N \cdot \mu_{\text{CPU}}}$$

where λ_excess = max(0, λ - T_GPU) is the excess arrival rate beyond GPU capacity.

*Interpretation.* When CPU is fast (high μ_CPU), slots free quickly → cpu_in_flight stays low → more requests routed to CPU → CPU stays busy. When CPU is slow (low μ_CPU), slots fill quickly → cpu_in_flight reaches N → new requests go to GPU → GPU is unaffected. The router self-adjusts without any explicit speed measurement.

**Property 2** (GPU non-interference). When cpu_in_flight = N, all new requests are routed to GPU. Therefore, the presence of the CPU engine never *reduces* GPU throughput—it can only *add* to the total.

**Property 3** (Maximum CPU utilization). The CPU is idle only when there are no incoming requests that could be routed to it (i.e., λ < T_GPU and CPU has finished all its work). Under load (λ ≥ T_GPU), CPU operates at maximum capacity.

#### C.3 Extension 1: Length-Aware Routing

For workloads with heterogeneous prompt lengths, short prompts complete faster on CPU while long prompts benefit from GPU's superior compute. The length-aware variant adds a threshold filter:

```
Algorithm 2: Length-Aware Routing Extension
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Parameter: τ ← cpu_prefill_threshold (default: 512 tokens)

function ROUTE(request r with prompt_len p):
    if cpu_in_flight < N AND p ≤ τ then
        cpu_in_flight ← cpu_in_flight + 1
        return CPU
    else
        return GPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The threshold τ prevents routing compute-heavy long prefills to the CPU, where they would occupy a slot for an extended period and reduce CPU's effective decode throughput.

#### C.4 Extension 2: Throughput-Adaptive Routing

The throughput-adaptive variant dynamically adjusts the effective CPU slot count N_eff based on measured CPU and GPU throughput using Exponential Moving Average (EMA):

```
Algorithm 3: Throughput-Adaptive Routing with EMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Parameters: α_ema ← 0.3, N_base ← cpu_max_num_seqs
State: EMA_GPU ← 0, EMA_CPU ← 0, N_eff ← N_base

function ON_COMPLETE(request r, target, tokens, elapsed):
    tps_instant ← tokens / elapsed
    if target = CPU then
        EMA_CPU ← α_ema · tps_instant + (1 - α_ema) · EMA_CPU
    else
        EMA_GPU ← α_ema · tps_instant + (1 - α_ema) · EMA_GPU
    UPDATE_SLOTS()

function UPDATE_SLOTS():
    if EMA_CPU > 0 AND EMA_GPU > 0 then
        ratio ← EMA_CPU / EMA_GPU
        N_eff ← clamp(N_base · (1 + ratio), 2, 2 · N_base)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**EMA convergence analysis.** With smoothing factor α = 0.3, the half-life of the EMA is:

$$t_{1/2} = \frac{-\ln 2}{\ln(1 - \alpha)} = \frac{-0.693}{\ln 0.7} \approx 1.94 \text{ samples}$$

This means the EMA adapts to throughput changes within ~2 request completions, providing responsive adaptation while filtering transient fluctuations.

#### C.5 Warmup Profiling

The throughput-adaptive strategy incorporates a warmup phase to initialize EMA values with meaningful measurements rather than starting from zero:

```
Algorithm 4: Warmup Profiling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Parameter: W ← warmup_requests (default: 10)
State: warmup_complete ← false

function ON_COMPLETE_WARMUP(target, tokens, elapsed):
    Accumulate tokens/elapsed per target
    if GPU_completed ≥ W AND CPU_completed ≥ W then
        EMA_GPU ← GPU_total_tokens / GPU_total_elapsed
        EMA_CPU ← CPU_total_tokens / CPU_total_elapsed
        UPDATE_SLOTS()
        warmup_complete ← true
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

During warmup, the router uses basic capacity routing. After warmup completes, EMA values are initialized with the measured average throughput, and the throughput-adaptive logic activates immediately.

### D. Automatic CPU Parameter Resolution

A key barrier to CPU inference deployment is the complexity of tuning parameters for diverse hardware configurations. We implement a zero-configuration pipeline (`_resolve_cpu_params()`) that automatically derives optimal settings from hardware detection.

**Table IV. Automatic parameter derivation rules (current)**

| Parameter | Auto Value | Detection Source | Rationale |
|-----------|-----------|-----------------|-----------|
| `num_cpu_engines` | `num_numa_nodes` | `NUMAAllocator.num_nodes` | One independent CPU EngineCore process per NUMA node — strict bind eliminates remote memory access |
| `cpu_num_threads` (per engine) | physical cores of the bound NUMA node | `/proc/cpuinfo` ÷ `threads_per_core` | HT provides minimal benefit for memory-bound decode [8] |
| `cpu_max_num_seqs` (per engine) | **fixed at 1** | — | One sequence saturates every physical core of the NUMA node via OMP; batching fractions the OMP pool and loses NUMA/cache locality |
| `cpu_kvcache_space_gb` (per engine) | `clamp(effective_mem × 0.4, 32, 512)` | `psutil.virtual_memory()` + NUMA node memory | Reserve headroom for OS, model weights, PyTorch buffers |
| `cpu_max_batched_tokens` | `cpu_max_num_seqs × 256` = 256 | Derived | Decode batch buffer |

> The earlier draft rule (`cpu_max_num_seqs = physical_cores ÷ 4`) was empirically
> worse than the current fixed-at-1 rule. Keeping one sequence per NUMA engine lets
> every matmul saturate the full OMP pool bound to that node, which maximizes
> wall-clock throughput per NUMA node. Total concurrent CPU sequences across the
> system equal `num_numa_nodes`.

**NUMA topology handling.** Each CPU EngineCore process is bound to its own NUMA
node (engine index → node index) and receives strict `numa_set_membind` plus
per-thread `sched_setaffinity` via the `_C_utils` extension. The binding pipeline is:

1. Detect NUMA topology via `libnuma` / `/sys/devices/system/node/`
2. Identify target node's CPU IDs and memory
3. Divide logical CPU count by `threads_per_core` to get physical cores
4. Set thread affinity: `KMP_AFFINITY=granularity=fine,compact,1,0`
5. Set memory policy: `numa_set_preferred(target_node)`

This automatic detection generalizes across hardware configurations—from single-socket development machines (no NUMA) to 2-socket production servers (NUMA with 2 nodes) and 4/8-socket HPC systems—without requiring any manual tuning.

### E. Intel CPU Optimization Pipeline

The CPU process startup (`_setup_cpu_process_env()`) applies a sequence of optimizations:

**Table V. Environment configuration for CPU inference**

| Variable | Value | Effect |
|----------|-------|--------|
| `CUDA_VISIBLE_DEVICES` | `""` | Prevent GPU memory allocation in CPU process |
| `KMP_AFFINITY` | `granularity=fine,compact,1,0` | Pack OpenMP threads on contiguous cores (maximize L1/L2 sharing) |
| `KMP_BLOCKTIME` | `1` | Minimize idle thread spin-wait (1 ms) |
| `OMP_NUM_THREADS` | NUMA physical cores | Optimal thread count excluding HT |
| `MKL_ENABLE_INSTRUCTIONS` | `AVX512` | Enable AVX-512 in MKL |
| `ONEDNN_MAX_CPU_ISA` | Auto-detected | `AVX512_CORE_AMX` if AMX available, else `AVX512_CORE_VNNI` |

**ISA detection hierarchy.** The system detects CPU capabilities in priority order: AMX-BF16/INT8 > AVX-512 VNNI > AVX-512F > AVX2, selecting the highest available ISA for both oneDNN and custom kernels.

**IPEX integration.** When Intel Extension for PyTorch (IPEX) [28] is installed, the system applies `ipex.optimize(model, dtype=torch.bfloat16)` to enable AMX-accelerated BF16 matrix multiplication and fused attention kernels. When IPEX is unavailable, the system falls back to pure PyTorch with manual AVX-512 optimization—ensuring universal deployability.

**AMX tile permission.** On Linux kernels ≥ 5.16, AMX requires explicit `ARCH_REQ_XCOMP_PERM` syscall to enable XSTATE components for TILECFG and TILEDATA. The optimization pipeline issues this syscall automatically during CPU process initialization.

---

## V. Implementation

This section focuses on non-trivial implementation decisions that impact system correctness and performance.

### A. Dual-Process Engine Lifecycle

**ZMQ IPC communication.** We extend vLLM V1's existing ZMQ infrastructure to support dual-engine routing. The client's ROUTER socket identifies engines by index: GPU uses identity `b'\x00\x00'` (engine_index=0) and CPU uses identity `b'\x01\x00'` (engine_index=1). A single PULL socket aggregates outputs from both engines' PUSH sockets in a fan-in pattern.

**CPU configuration derivation.** The CPU engine's VllmConfig is derived from the GPU config via `copy.deepcopy` with targeted modifications:

**Table VI. Configuration derivation: GPU → CPU**

| Config Field | GPU Value | CPU Value | Reason |
|-------------|-----------|-----------|--------|
| `DeviceConfig.device` | `"cuda"` | `"cpu"` | Target device |
| `ParallelConfig` | TP=8 | TP=1, PP=1, `UniProcExecutor` | CPU uses single-process execution |
| `CacheConfig` | GPU VRAM-based | DDR5-based, auto-sized | Memory pool differs |
| `SchedulerConfig` | GPU batch limits | CPU limits (auto-detected) | CPU handles fewer concurrent seqs |
| `CompilationConfig` | CUDA graph enabled | CUDA graph disabled | No CUDA on CPU process |
| `HybridConfig` | Active | `None` | Prevent recursive hybrid spawning |

**Completion tracking.** The `HybridAsyncMPClient` maintains a `_hybrid_reqs_in_flight` dictionary mapping `request_id → target_device`. When engine output processing detects a completed request from the CPU engine, it invokes `router.on_request_finished(req_id, was_cpu=True)` to reclaim the CPU slot.

### B. AVX-512 Kernel Design

We implement five C++ kernels targeting the CPU inference hot path:

#### B.1 VNNI INT8 GEMM (gemm_vnni.cpp)

We adopt a **6×16 micro-kernel** design: 6 ZMM accumulator registers compute 6 rows of output simultaneously, with `vpdpbusd` performing 384 INT8 MACs per inner-loop iteration (6 rows × 16 columns × 4 elements).

**Register pressure analysis.** AVX-512 provides 32 ZMM registers. Our micro-kernel uses: 6 accumulators + 1 broadcast A + 1 load B = 8 registers, leaving 24 for prefetch addresses and loop control—well within budget.

**Cache blocking** follows Goto's BLAS methodology [37]:
- **L3 level**: Partition N by NC=256; B panel (~256×256 = 64 KB) resides in L3
- **L2 level**: Partition K by KC=256; A panel (~72×256 = 18 KB) resides in L2
- **L1/Register level**: Partition M by MR=6; micro-kernel operates from L1/registers

B matrix is pre-packed into VNNI layout `[K/4][N/16][16][4]` to ensure aligned, sequential memory access.

#### B.2 Batched Attention (batch_attention.cpp)

We exploit AVX-512's 16 SIMD lanes to process **16 sequences simultaneously**: lane[i] holds the Q·K attention score for sequence i. Online softmax [Algorithm: Milakov & Gimelshein, 2018] computes attention weights in a single pass without materializing the full score matrix.

**KV cache prefetch strategy.** Each of the 6 block-processing loops includes `_mm_prefetch` instructions with `_MM_HINT_T1` (L2 prefetch) to preload the next block's K and V cache lines 256 bytes ahead of the current access point. This hides DDR5 access latency (~80 ns) behind computation.

#### B.3 Memory Operations (mem_opt.cpp)

- **Non-temporal memcpy**: `_mm512_stream_si512` for large sequential writes (>64 KB) to avoid polluting cache hierarchy with write-once data.
- **NUMA allocation**: `numa_alloc_onnode` for KV cache blocks, ensuring local memory access.
- **Software prefetch**: Configurable prefetch distance for sequential model weight reads.

### C. Build System

We implement dual-target CMake to produce two shared libraries from a single build:

- `_C.abi3.so` — CUDA operations (compiled with NVCC)
- `_C_cpu_ops.abi3.so` — CPU operations (compiled with GCC, `-mavx512f -mavx512vnni` flags)

CPU kernels are registered under the `_C_cpu_ops` namespace to prevent symbol collisions. On platforms without AVX-512 support, the CPU extension build is automatically skipped, maintaining compatibility with CUDA-only deployments.

---

## VI. Theoretical Analysis and Experimental Design

### A. Throughput Model Predictions

From the throughput model (Section IV-B) and the roofline analysis (Section II-A), we derive quantitative predictions:

**Table VII. Predicted throughput improvement by model size**

| Model | GPU Throughput | CPU Throughput (est.) | Hybrid Improvement | Limiting Factor |
|-------|---------------|----------------------|-------------------|----------------|
| LLaMA 3 70B (TP=8, BF16) | ~100 tok/s | ~2–5 tok/s (Q8_0) | +2–5% | DDR5 bandwidth (307 GB/s ÷ 35 GB = 8.8 tok/s roofline) |
| LLaMA 3 8B (single GPU) | ~500 tok/s | ~15–40 tok/s (Q8_0) | +3–8% | DDR5 bandwidth (307 GB/s ÷ 4 GB = 76.8 tok/s roofline) |
| LLaMA 3 8B (Q4_0 on CPU) | ~500 tok/s | ~30–60 tok/s | +6–12% | Compute (INT4 decompression overhead) |

The roofline model confirms that CPU decode is **memory-bandwidth bound**: the operational intensity (FLOPs/byte) for autoregressive decode is ~1, far below the CPU's compute-memory balance point. Therefore, DDR5 bandwidth is the fundamental throughput limiter, and improvements require either (a) higher memory bandwidth (HBM, CXL) or (b) more aggressive quantization (Q4_0, Q2_K).

### B. Experimental Setup

**Table VIII. Hardware and software configuration**

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA H100 SXM 80 GB × 8 (NVLink 4.0) |
| CPU | Intel Xeon Platinum 8480+ × 2 (56 cores/socket, 112 cores total) |
| Memory | DDR5-4800 2 TB (1 TB/socket, NUMA 2 nodes) |
| ISA | AVX-512F, AVX-512 VNNI, AMX-BF16, AMX-INT8 |
| OS | Ubuntu 22.04, Linux 6.x |
| Framework | vLLM Hybrid (this work) |
| Baseline | vLLM upstream (GPU-only) |
| IPEX | v2.3.0 (when applicable) |

**Models and configurations:**
- LLaMA 3 70B Instruct: TP=8 on GPU, Q8_0 on CPU
- LLaMA 3 8B Instruct: Single GPU (BF16), Q8_0 on CPU

**Workloads:**
- ShareGPT conversation traces (realistic length distribution)
- Synthetic: Fixed input/output lengths (128/128, 512/256, 1024/512)
- Varying request rates: 1, 5, 10, 20, 50, 100 req/s

**Metrics:**
- Throughput: Total tokens generated per second (tok/s)
- TTFT: Time to first token (ms), p50/p95/p99
- TPOT: Time per output token (ms), p50/p95/p99
- GPU tail latency: p99 latency of GPU-routed requests
- CPU utilization: Fraction of time CPU slots are occupied

### C. Planned Experiments

We design eight experiments to validate the throughput model and characterize system behavior:

**Experiment 1: End-to-End Throughput Comparison.**
- Compare GPU-only vLLM vs. Hybrid (capacity routing) under varying request rates.
- Hypothesis: Hybrid throughput = GPU throughput + CPU throughput (within 5% of theoretical prediction).
- Independent variable: Request rate (1–100 req/s).
- Dependent variable: Total throughput (tok/s).

**Experiment 2: GPU Latency Impact Analysis.**
- Measure GPU-routed request latency distributions (p50/p95/p99) with and without CPU engine active.
- Hypothesis: No statistically significant difference (paired t-test, p > 0.05).
- This validates Corollary 1 (GPU latency preservation).

**Experiment 3: Router Strategy Comparison.**
- Compare three routing strategies: capacity, length-aware (τ=512), throughput-adaptive (α=0.3, W=10).
- Workload: ShareGPT traces (heterogeneous lengths).
- Metrics: Total throughput, CPU utilization, CPU-routed request latency.
- Hypothesis: Throughput-adaptive achieves highest throughput under variable workloads; capacity is sufficient for uniform workloads.

**Experiment 4: NUMA Binding Ablation.**
- Compare: (a) single-node binding (our default), (b) cross-node execution, (c) interleaved allocation.
- Hypothesis: Single-node binding achieves ≥1.3× higher CPU throughput than cross-node [8, 9].
- Metrics: CPU decode throughput, memory bandwidth utilization.

**Experiment 5: IPEX Ablation.**
- Compare: (a) IPEX-optimized CPU engine, (b) pure PyTorch fallback.
- Hypothesis: IPEX provides 1.5–3× CPU throughput improvement via AMX utilization [13].
- Metrics: CPU decode throughput, AMX instruction utilization (via `perf stat`).

**Experiment 6: Auto-Detection vs. Manual Tuning.**
- Compare: (a) zero-config (auto-detected parameters), (b) expert-tuned manual configuration.
- Hypothesis: Auto-detection achieves ≥90% of manually-tuned performance.
- This validates the zero-configuration pipeline's robustness.

**Experiment 7: Model Size Scalability.**
- Sweep model sizes: 7B, 13B, 34B, 70B.
- Plot: CPU throughput contribution (%) vs. model size.
- Hypothesis: Contribution decreases with model size (inverse relationship with model memory footprint).

**Experiment 8: Request Rate Sensitivity.**
- Sweep request rates from under-provisioned (λ < T_GPU) to over-provisioned (λ >> T_GPU + T_CPU).
- Plot: CPU utilization factor α vs. λ/T_GPU ratio.
- Hypothesis: α transitions from ~0 (when λ << T_GPU) to ~1 (when λ > T_GPU), validating Theorem 1.

### D. Expected Results and Discussion

Based on the theoretical model, we expect the following results:

**Throughput gain scales inversely with model size.** Larger models require more memory bandwidth per token, reducing CPU throughput and its relative contribution. For 70B models, the gain is modest (1–5%) but non-trivial given zero hardware cost. For 7B models, the gain becomes significant (5–15%).

**GPU latency is unaffected.** Process isolation guarantees that the CPU engine's activity does not interfere with GPU scheduling, memory access, or PCIe traffic. We expect p99 GPU latency to remain within 1% of baseline.

**Throughput-adaptive routing outperforms static capacity routing** under variable workloads because it dynamically adjusts CPU slot allocation based on measured performance, preventing slot starvation or over-allocation.

**NUMA binding is critical.** Based on prior characterization work [8], we expect cross-NUMA access to degrade CPU throughput by 30–50% due to increased memory latency and reduced effective bandwidth.

**Conditions for maximum and minimum CPU contribution:**
- **Maximum**: Small model (7B), high request rate, short prompts, IPEX-optimized, single-NUMA binding.
- **Minimum**: Large model (70B), low request rate (λ < T_GPU), long prompts, no IPEX, cross-NUMA access.

---

## VII. Discussion and Future Work

### A. Practical Implications

**TCO analysis.** Our system delivers additional throughput at zero marginal hardware cost—the CPU and memory already exist in the server. For a 100-node cluster with 3% throughput improvement:

$$\Delta T_{\text{equivalent}} = 100 \times 0.03 = 3 \text{ GPU-equivalent nodes}$$

At \$300,000 per DGX H100 node, this represents \$900,000 of equivalent GPU capacity saved, far exceeding the engineering investment.

**Deployment considerations.** The system requires no hardware modifications, network changes, or additional services. A single CLI flag (`--hybrid-mode parallel-batch`) enables the feature, with all parameters auto-configured. This minimal deployment friction is critical for production adoption.

### B. Extensibility

The dual-process architecture provides a natural foundation for three extensions:

**MoE Expert Offloading.** Process isolation can be reused for MoE models [34, 35, 36]: inactive experts reside in CPU process memory, with asynchronous prefetch loading active experts to GPU. The CapacityAwareRouter naturally extends to expert-level routing decisions.

**CPU-based Speculative Decoding.** The CPU process can run lightweight draft models (or N-gram proposers) to generate candidate tokens, which the GPU process verifies in parallel [14, 15]. The existing ZMQ IPC infrastructure supports the bidirectional token exchange required for speculative coordination.

**Disaggregated Serving.** Our architecture is complementary to prefill-decode disaggregation [20, 21]. CPU-based decode can serve as a third tier: GPU handles prefill, GPU handles high-priority decode, and CPU handles overflow decode—maximizing both compute and memory utilization.

### C. Limitations

1. **Absolute CPU throughput bound.** CPU inference throughput is fundamentally limited by DDR5 bandwidth, providing only 1–5% additional throughput for large (70B+) models. The benefit-to-complexity ratio diminishes for very large models.

2. **Model weight duplication.** CPU and GPU each load full model weights, consuming additional DDR5 memory (e.g., ~35 GB for 70B Q8_0). On memory-constrained servers, this reduces available KV cache space.

3. **CPU quantization requirement.** Running FP16/BF16 models on CPU is impractical due to memory bandwidth constraints; INT8 or lower quantization is effectively mandatory, introducing potential accuracy degradation.

4. **Single-server scope.** The current design targets a single server with local CPU-GPU pairs. Extending to multi-node CPU inference with network-attached CPUs would require additional KV cache transfer mechanisms.

---

## VIII. Conclusion

We have presented a Dual-Process Parallel-Batch architecture that harvests idle CPU cycles in GPU servers for LLM inference. By running GPU and CPU engines as independent OS processes, our system achieves truly additive throughput (T_hybrid = T_GPU + α·T_CPU) while guaranteeing zero interference with the GPU pipeline. The CapacityAwareRouter's self-regulating property maximizes CPU utilization without requiring knowledge of CPU processing speed, and the zero-configuration pipeline eliminates deployment friction by automatically detecting hardware topology and deriving optimal parameters.

Our formal throughput model predicts 1–5% improvement for 70B models and 5–15% for 7B models, obtained at zero additional hardware cost. While the absolute CPU contribution is modest for large models, the architectural principle—**idle resources should not remain idle**—represents a new dimension in LLM serving optimization that is orthogonal to and composable with existing GPU-focused techniques.

The dual-process architecture extends naturally to MoE expert offloading, CPU-based speculative decoding, and disaggregated serving, positioning it as a foundational building block for CPU-GPU heterogeneous LLM serving systems.

---

## Acknowledgment

*(To be completed)*

---

## References

[1] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model serving with PagedAttention," in *Proc. 29th ACM SOSP*, 2023, pp. 611–626.

[2] Y. Chen et al., "KTransformers: Unleashing the full potential of CPU/GPU hybrid inference for MoE models," in *Proc. 31st ACM SOSP*, 2025.

[3] Y. Song, Z. Mi, H. Xie, and H. Chen, "PowerInfer: Fast large language model serving with a consumer-grade GPU," in *Proc. 30th ACM SOSP*, 2024.

[4] Z. Xue, Y. Song et al., "PowerInfer-2: Fast large language model inference on a smartphone," arXiv preprint arXiv:2406.06282, 2024.

[5] X. Zhao, B. Jia, H. Zhou, Z. Liu, S. Cheng, and Y. You, "HeteGen: Heterogeneous parallel inference for large language models on resource-constrained devices," in *Proc. MLSys*, 2024.

[6] L. Zheng, L. Yin, Z. Xie, J. Huang, C. Sun, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y. Sheng, "SGLang: Efficient execution of structured language model programs," in *Proc. NeurIPS*, 2024.

[6a] Intel PyTorch Team / LMSYS, "Cost effective deployment of DeepSeek R1 with Intel Xeon 6 CPU on SGLang," LMSYS Blog, 2025. [Online]. Available: https://lmsys.org/blog/2025-07-14-intel-xeon-optimization/

[7] P. He, S. Zhou, W. Huang, C. Li, D. Wang, B. Guo, and C. Meng, "Inference performance optimization for large language models on CPUs," arXiv preprint arXiv:2407.07304, 2024.

[8] S. Na, G. Jeong, B. H. Ahn, J. Young, T. Krishna, and H. Kim, "Understanding performance implications of LLM inference on CPUs," in *Proc. IEEE IISWC*, 2024.

[9] Various, "Optimization of NUMA aware DNN computing system," Springer, 2024.

[10] Y. Zhang et al., "ParaX: Boosting deep learning for big data analytics on many-core CPUs," *Proc. VLDB Endow.*, vol. 14, no. 5, pp. 864–876.

[11] Intel, "NUMA-Caffe: NUMA-aware deep learning neural networks," Intel Technical Document.

[12] Intel, "Deep learning with Intel AVX-512 and Intel DL Boost," Intel Developer Guide. [Online]. Available: https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html

[13] Intel, "Accelerate PyTorch training and inference using Intel AMX," Intel Technical Article. [Online]. Available: https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-pytorch-training-inference-on-amx.html

[14] Y. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via speculative decoding," in *Proc. 40th ICML*, 2023.

[15] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper, "Accelerating large language model decoding with speculative sampling," arXiv preprint arXiv:2302.01318, 2023.

[16] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale," in *Proc. ICML*, 2022.

[17] A. Eliseev and D. Mazur, "Fast inference of mixture-of-experts language models with offloading," arXiv preprint arXiv:2312.17238, 2023.

[18] Various, "MoE-Infinity: Activation-aware expert offloading for efficient MoE serving," arXiv preprint arXiv:2401.14361, 2024.

[19] A. Q. Jiang, A. Sablayrolles et al., "Mixtral of experts," arXiv preprint arXiv:2401.04088, 2024.

[20] Y. Zhong et al., "DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving," in *Proc. 18th USENIX OSDI*, 2024.

[21] P. Patel, E. Choukse, C. Zhang, A. Shah, I. Goiri, S. Maleki, and R. Bianchini, "Splitwise: Efficient generative LLM inference using phase splitting," in *Proc. 51st IEEE/ACM ISCA*, 2024. (Best Paper Award)

[22] G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, and B.-G. Chun, "Orca: A distributed serving system for transformer-based generative models," in *Proc. 16th USENIX OSDI*, 2022.

[23] A. Agrawal, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, and R. Ramjee, "SARATHI: Efficient LLM inference by piggybacking decodes with chunked prefills," arXiv preprint arXiv:2308.16369, 2023.

[24] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. Gulavani, A. Tumanov, and R. Ramjee, "Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve," in *Proc. USENIX OSDI*, 2024.

[25] Various, "Prompt Cache: Modular attention reuse for low-latency inference," in *Proc. MLSys*, 2024.

[26] Y. Tang et al., "Exploring CXL-based KV cache storage for LLM serving," in *NeurIPS ML for Systems Workshop*, 2024.

[27] Various, "LMCache: An efficient KV cache layer for enterprise-scale LLM inference," Technical Report, lmcache.ai.

[28] Intel, "Intel Extension for PyTorch (IPEX)," Open-source project. [Online]. Available: https://github.com/intel/intel-extension-for-pytorch

[29] Intel, "Optimizing large language model inference on Intel CPUs with IPEX and IPEX-LLM," Intel Technical Paper, 2024.

[30] Intel, "IPEX-LLM: Intel LLM library for PyTorch," Open-source project. [Online]. Available: https://github.com/intel/ipex-llm

[31] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, D. Y. Fu, Z. Xie, B. Chen, C. Barrett, J. E. Gonzalez, P. Liang, C. Re, I. Stoica, and C. Zhang, "FlexGen: High-throughput generative inference of large language models with a single GPU," in *Proc. ICML (Oral)*, 2023.

[32] G. Gerganov et al., "llama.cpp / GGML," Open-source project. [Online]. Available: https://github.com/ggml-org/llama.cpp

[33] Various, "Which quantization should I use? A unified evaluation of llama.cpp quantization on Llama-3.1-8B-Instruct," arXiv preprint arXiv:2601.14277, 2026.

[34] D. Dai, C. Deng, C. Zhao, R. Xu et al., "DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models," in *Proc. ACL*, 2024.

[35] DeepSeek AI, "DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model," arXiv preprint arXiv:2405.04434, 2024.

[36] DeepSeek AI, "DeepSeek-V3 Technical Report," arXiv preprint arXiv:2412.19437, 2024.

[37] K. Goto and R. A. van de Geijn, "Anatomy of high-performance matrix multiplication," *ACM Trans. Math. Softw.*, vol. 34, no. 3, 2008.

---

*Draft v2 — 2026-02-26*
