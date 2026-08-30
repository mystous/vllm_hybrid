# [kt-kernel] Silent output corruption when threadpool_count != --numa-nodes used at quantization time

**Version**: kt-kernel 0.7.0.post2
**Symptom**: Weights converted with `kt quant ... --numa-nodes 2` produce garbage output (e.g. "兜微量被捕 ча quo...") when loaded with `threadpool_count=1`. No error or warning is raised; the server boots and serves normally-shaped responses. `threadpool_count=2` produces correct output. Same weights, same prompt, greedy.

**Impact**: any deployment that reduces threadpool count (e.g. to run per-socket instances) silently serves corrupted outputs. We shipped several benchmark rounds before noticing.

**Suggested fix**: store numa_nodes in the converted-weight metadata and assert `threadpool_count == weights.numa_nodes` at load (or support remapping).
