# IDE_023 13-Lever PoC — SUB_201

- Model: Llama-3.1-8B-Instruct, TP=8, B200 ×8
- Bench: sharegpt 200p × conc=16 × max-tok=512
- Baseline = Optimal Config (vanilla + FaP + L2 + L10)

## Baseline

- output_tps = **6797.3** tps
- gpu_util = 85.1%
- cpu_util = 5.3%
- wall_total_s = 14.8s
- n_ok = 200/200

## Lever results

| lever | description | tps | Δ% vs baseline | status | gpu_util | cpu_util | apply |
|---|---|---:|---:|---|---:|---:|---|
| N1 | AVX-512 BPE encode | 6767.8 | -0.43 % | ok | 90.4 | 5.2 | applied (tiktoken 0.12.0 loaded, BPE AVX-512 hint set) |
| N4 | SoA paged attention layout | 6710.5 | -1.28 % | ok | 91.1 | 5.2 | applied (VLLM_KV_TILE_BYTES=4MiB hint) |
| N5 | SMT-pair pinning scheduler | 6728.9 | -1.01 % | ok | 91.6 | 5.2 | applied (SMT pair affinity, |mask|=224) |
| N6 | Lock-free priority queue | 6705.8 | -1.35 % | ok | 91.9 | 5.3 | applied (VLLM_SCHEDULER_DEQUE_PATH=1 hint) |
| N7 | Huge pages 2MB for KV | 6708.3 | -1.31 % | ok | 91.8 | 5.3 | applied (THP policy = always [madvise] never, libc madvis... |
| N8 | NUMA-local draft state | 6798.5 | 0.02 % | ok | 86.9 | 5.3 | applied (numa_run_on_node(0) rc=0) |
| N9 | DSA memcpy host<->pinned | 6726.2 | -1.05 % | ok | 91.0 | 4.8 | na: no /dev/dsa/* WQ visible (need accel-config + WQ enable) |
| N10 | AVX-512 simdjson request parse | 6709.0 | -1.3 % | ok | 91.8 | 5.3 | applied (simdjson unknown hook installed) |
| N11 | AVX-512 base64 output streaming | 6682.5 | -1.69 % | ok | 91.8 | 4.9 | applied (pybase64 1.4.3 hooked into base64) |
| N14 | Prefetch suffix tree | 6696.8 | -1.48 % | ok | 91.4 | 5.3 | applied (ARCTIC_SUFFIX_PREFETCH=1 hint) |
| N17 | CMT-driven priority (Intel PCM) | 6699.1 | -1.44 % | ok | 91.8 | 5.3 | applied (resctrl visible, entries=0) |
| N19 | AVX-512 SSE writer | 6708.0 | -1.31 % | ok | 91.9 | 5.3 | applied (VLLM_USE_AVX512_SSE_WRITER=1 hint) |
| N20 | LogGP admission (cost-aware) | 6726.8 | -1.04 % | ok | 91.7 | 5.3 | applied (VLLM_LOGGP_ADMISSION_ACTIVE=1 hint) |

## Net-positive (Δ% ≥ +3%, noise floor)

- (none above +3% threshold)

## N/A or missing levers

- (all 13 produced an output_tps measurement)

## Environmental N/A (apply step reported `na:`)

- **N9** (DSA memcpy host<->pinned): na: no /dev/dsa/* WQ visible (need accel-config + WQ enable)

## Production-ready recommendation (top 3-5)

Ranking by Δ% (positive only):

1. **N8** (NUMA-local draft state): Δ = +0.02%

## Artefacts

- per-tag throughput summary: `runs/baseline.json`, `runs/lever_N*.json`
- boot logs: `logs/*_boot.log`
- bench logs: `logs/*_bench.log`
- patch: `vllm/v1/spec_decode/ide023_levers.py` + `vllm/envs.py` (13 env flags)
- harness: `scripts/sweep.sh`

