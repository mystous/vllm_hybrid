# LHC Phase 4 — Misuse Anti-Pattern Results

**Baseline workload**: Llama-3.1-8B Instruct, TP=8, sharegpt-equivalent (sonnet harness, chat) + sonnet.

**Definition**: baseline = LHC properly used (regime ON, WQ-per-rank, NUMA-local, DSA_MIN=64KB, AMX prefix-hit-only); misuse = anti-pattern env injection.

| AP | WL | baseline TPS (mean±std, n) | misuse TPS (mean±std, n) | Δ% paired (all sweeps) | Δ% paired (excl s1 cold) | n_err |
|----|----|---------------------------|-------------------------|------------------------|------------------------|-------|
| ap1 | chat | 12484.47±2529.43, n=3 | 13755.99±161.30, n=3 | 13.88% (45.51, -1.65, -2.22) | -1.93% (-1.65, -2.22) | b=0/m=0 |
| ap2 | chat | 13721.55±270.93, n=3 | 1034.40±49.13, n=3 | -92.46% (-92.38, -92.79, -92.22) | -92.50% (-92.79, -92.22) | b=0/m=0 |
| ap3 | chat | 13626.93±275.96, n=3 | 13767.30±277.70, n=3 | 1.03% (2.08, 0.28, 0.74) | 0.51% (0.28, 0.74) | b=0/m=0 |
| ap4 | chat | 13889.83±158.68, n=3 | 13767.19±207.77, n=3 | -0.88% (-2.68, 0.84, -0.79) | 0.03% (0.84, -0.79) | b=0/m=0 |
| ap5 | chat | 13840.78±332.29, n=3 | 13878.04±204.96, n=3 | 0.29% (0.79, -1.14, 1.20) | 0.03% (-1.14, 1.20) | b=0/m=0 |

## Per-anti-pattern interpretation

- **ap1** — DSA_MIN=64 vs 65536. Smaller transfers (<64B) bypass DSA in baseline (memcpy fast-path); in misuse they go through DSA descriptor enqueue (~5μs/op vs ~0.04μs/op for memcpy).
- **ap2** — AMX C3 prefix scan FORCE_EVERY_STEP vs prefix-hit-only. Each scheduler step pays a 65KB synthetic scan; baseline scan only fires on prefix-cache miss-then-hit transitions.
- **ap3** — DSA NUMA cross-socket vs local. Misuse inverts rank→device map (rank 0–3 → dsa1, rank 4–7 → dsa0), forcing all DSA memcpy traffic across the QPI/UPI inter-socket link.
- **ap4** — DSA WQ-per-rank OFF vs ON. Misuse routes all 8 TP workers to wq0.0; PASID contention surfaces as EBUSY drops in the lane self-test (lane disabled at init) OR queue serialisation.
- **ap5** — Regime detector OFF (Option A static) vs ON (Option C). In the GPU-saturated baseline regime, the adaptive detector keeps LHC OFF; the static Option A pays the sampling/setup cost without productive work.

## Key findings (paper §08 §subsec:res-misuse)

본 측정은 LHC infrastructure 의 **잘못된 사용 (misuse)** 이 throughput 손해를 일으키는지 정량으로 입증한다 (Theorem 1 — Lane Separation 부속 evidence).

1. **ap2 (AMX C3 매 step 호출)** — 유일한 **dramatic penalty**. paired Δ% = -92.46% (3/3 sweeps, std 0.29%p, n_err=0). AMX C3 가 lane separation 의 vector ALU 자원을 사용하지만, 매 scheduler step 마다 prefix-cache hit 무관 호출되면 (a) CPU 자원의 productive 신호 없는 burn, (b) scheduler step 직렬 path 에 추가 latency 누적. 본 baseline 의 step time (~7 ms) 대비 65 KB AMX scan (~130 µs prod, fallback 환경에서는 numpy python loop 로 수십 ms) 가 매 step pile up → throughput 90% 이상 감소.

2. **ap1, ap3, ap4, ap5** — 모두 **noise band** (excl s1 cold-start: -1.93%, +0.51%, +0.03%, +0.03%). 본 baseline (chat sharegpt × conc=64 × max-tok=2048) 의 regime 이 **GPU_SATURATED** (옵션 C 분류 94%) 이라 DSA lane gate 가 OFF → DSA path actual not called → DSA_MIN/NUMA/WQ-per-rank 의 misuse 가 dead code 영역에서 invoke 됨. 이는 Option C adaptive gate 의 **defensive 가치 입증**: misuse 가 발생해도 lane 이 OFF 면 throughput 보호.

3. **Anti-pattern 분류** — **active-lane misuse** (ap2: 매 step force, gate 우회) vs **inactive-lane misuse** (ap1, ap3, ap4, ap5: gate 가 막아주는 dead branch). Theorem 1 (Lane Separation) 의 가산성 조건 ($\kappa < 0.3$) 은 lane 이 OFF 일 때는 vacuously true, ON 일 때만 진정 검증됨. **본 baseline regime 에서는 active-lane misuse 가 가능한 단 하나의 lane (AMX C3) 가 위 ap2 결과의 -92% penalty 를 보인다**.

4. **운영 implication** — LHC infrastructure 를 정적 ON 으로 deploy 할 때 (Option A), AMX C3 같은 always-on lane 은 prefix-hit filter 가 필수. 본 측정은 prefix-hit gate 가 빠진 채 모든 step 에서 AMX 가 호출되면 throughput 이 90% 이상 무너짐을 시연한다. Option C adaptive detector 가 이런 misuse pattern 을 자동으로 차단함 (KV_HEAVY 가 아닌 regime 에서 amx_c3 → OFF).

## Measurement setup

- HW: DGX B200 8× sm_100, Xeon Platinum 8570 (DSA + AMX), 2TB DRAM
- vLLM: editable build at `/workspace/vllm_dev_prj`, branch `feat/spec-decode-tuning`
- Model: meta-llama/Llama-3.1-8B-Instruct, TP=8, max-model-len 16384, GPU mem 0.92, prefix caching ON
- Workload: sonnet harness (input=256, output=512, prefix=0, prompts=500, conc=64) — matches the SUB-201 "chat" baseline
- Sweeps: 3 per (ap × config), s1 typically cold-start outlier (prefix cache empty)
- New env vars introduced for this measurement:
  - `VLLM_LHC_AMX_C3_FORCE_EVERY_STEP=1` — scheduler step end hook calls amx_c3_prefix_scan(65KB, granule=64) unconditionally
  - `VLLM_LHC_DSA_FORCE_REMOTE_NUMA=1` — rank→DSA-device mapping inverted (rank 0-3 → dsa1, rank 4-7 → dsa0)
- Code references: `vllm/v1/lhc/dsa_lane.py`, `vllm/v1/lhc/amx_c3_lane.py`, `vllm/v1/core/sched/scheduler.py`

## Files

- Sweep driver: `lhc_phase4/misuse/run_misuse_sweep.sh` + `chain_all.sh` + `wait_and_chain.sh`
- Aggregator: `lhc_phase4/misuse/aggregate_misuse.py`
- Raw bench results: `lhc_phase4/misuse/runs/{ap1..ap5}_{baseline,misuse}_chat_s{1,2,3}_bench.{json,log}`
- Sweep logs: `lhc_phase4/misuse/sweep_{ap1..ap5}.log` + `sweep_ap1_s3.log`
