# Legacy Context — Async Executor Rejection and Memory-Bandwidth Pivot

> Source branch: `origin/archive/pre-v0.20.0-main`
>
> Purpose: preserve the rationale that moved this work away from same-request
> CPU pipeline schemes and toward `shadow_assists` ideas that use CPU DRAM /
> DDR bandwidth as a distinct resource.

## 1. What Was Rejected

The rejected line was `X — Pipelined Async CPU Executor`.

Archive source:

- `super_power/implementation/X_pipelined_async_cpu_executor_rejected/README.md`
- `super_power/implementation/X_pipelined_async_cpu_executor_rejected/01_design_and_plan.md`
- `super_power/implementation/B3_gateway_rejected/README.md`
- `super_power/implementation/newidea/README.md`

The original goal was to replace the CPU-side `UniProcExecutor` with a
pipelined async executor so that the main Python thread and compute thread could
run in parallel. The mechanism was a 1-step lookahead pipeline over consecutive
decode steps.

That approach was rejected because the pipeline was not independent work. It
created same-request duplicate compute across consecutive decode steps. In the
H100 measurement recorded in the archive, the light workload regressed from
382 s to 747 s, and per-engine generation dropped from 2.8 tok/s to 1.3 tok/s.
The practical result was roughly half the synchronous throughput.

The durable lesson is:

> Do not create a CPU "pipeline" by speculating the next decode step of the same
> request unless the dependency and correctness contract proves that the work is
> not duplicate compute.

This is why current `shadow_assists` documents explicitly call out the X failure
pattern as something to avoid.

## 2. Why The Pivot Was Not "CPU Is Useless"

The rejection of X did not invalidate CPU-side work in general. It invalidated a
specific executor-level lookahead design.

The archive's broader system argument was memory-bandwidth based:

- Decode is memory-bandwidth-bound: each token repeatedly reads model weights
  and KV state.
- Host CPUs on H100-class servers bring substantial local DDR5 capacity and
  bandwidth.
- Multi-socket systems require NUMA-local placement; cross-socket reads can
  erase the expected gain.
- CPU work is only interesting when it consumes resources that are genuinely
  separate from the GPU critical path, or when it removes GPU reload / transfer
  work.

Archive sources:

- `docs/PAPER_DRAFT.md` — CPU decode roofline / DDR5 bandwidth argument.
- `docs/DETAIL_INFO_4_3_4_4.md` — decode as memory-bandwidth-bound and NUMA
  binding constraints.
- `NinjaGap_Todo/20_kv_offload.md` — CPU DRAM as a cold KV tier.
- `NinjaGap_Todo/22_neo_asymmetric.md` — PCIe bandwidth risk for repeated
  per-step transfers.
- `NinjaGap_Todo/25_gqa_batched_attention.md` — KV bandwidth reduction through
  grouped-query structure.

So the correct pivot is not "make the CPU run another copy of the same step".
The correct pivot is to find work where CPU DRAM / DDR bandwidth is the resource
being exploited, and where GPU-visible output can be merged under a clear
correctness contract.

## 3. Implication For Current Shadow Assists

Current `shadow_assists` ideas inherit two constraints from this history.

First, they must avoid same-request duplicate compute. Any planner, async
preparation, or pipeline proposal must prove that it is preparing independent
metadata / state, not recomputing a decode step that will be thrown away.

Second, they should prefer resource-separable work:

- CPU DRAM-resident state.
- NUMA-local reads.
- Reduced GPU reload / transfer volume.
- Explicit merge or verification contracts.
- Measurements that separate CPU work, transfer time, and GPU critical-path
  impact.

This is the context for `IDE_006`.

`IDE_006` is not a resurrection of the rejected async executor. Its premise is
different: cold KV blocks already resident in CPU DRAM are consumed by a CPU
partial-attention path, and only `(partial_output, LSE)` is returned to the GPU
for online-softmax merge. The resource being targeted is CPU DRAM / DDR
bandwidth plus avoided GPU reload of cold KV, not same-request lookahead.

The remaining risk is still the old one in a different form: if the layer-internal
CPU path serializes on the GPU critical path, the design becomes pure latency.
That is why the current plans require overlap / throughput measurements rather
than treating CPU utilization as success by itself.

## 4. Operational Rule

When evaluating future CPU-assist ideas:

1. Reject designs whose benefit depends on same-request lookahead without a
   dependency proof.
2. Require a resource argument: which bandwidth, memory tier, or CPU-only
   control path is being used?
3. Require a correctness contract before performance work.
4. Measure GPU critical-path impact separately from CPU utilization.
5. Treat "CPU became busy" as insufficient; the accepted metric is net throughput
   or latency improvement under unchanged GPU-visible results.

