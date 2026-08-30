# [kt-kernel] CPUInfer pins worker threads to absolute core ids, ignoring numactl/cgroup intent (multi-instance collapse up to 115x)

**Version**: kt-kernel 0.7.0.post2, SGLang 0.5.18 integration
**Environment**: 2x Xeon 8480+ (112C), 2-NUMA, running two kt-hybrid instances on one host

**Symptom**: When two SGLang+kt instances run on one machine (each given one socket via `numactl --cpunodebind`), throughput collapses from ~23 tok/s per instance to 1.6 tok/s (-95%). Per-socket busy measurement shows both instances' CPUInfer threads pinned to socket 0 (socket0 89.6% busy, socket1 5.6%).

**Root cause**: CPUInfer pins worker threads starting from absolute core 0 regardless of the parent process affinity mask or numactl policy.

**Workaround we validated**: run each instance in a container with hard cpuset limits (`--cpuset-cpus <socket cores> --cpuset-mems <node>`); the kernel clamps the pin attempts and interference returns to ~0%.

**Suggested fix**: honor the inherited affinity mask (sched_getaffinity) when selecting pin targets, or expose a base-core offset option.
