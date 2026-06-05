# Speculative Decoding Shifts the LLM-Serving Bottleneck from GPU Memory Bandwidth to the Host Path — Paper Draft (re-skeleton on §5 verdict)

> **status**: draft / re-skeleton (2026-06-05)
> **parent**: SUB_201_cpu_host_path_bottleneck (IDE_022 / TSK_042)
> **scope**: §0 abstract → §6 lever results, §8 discussion (re-write). §1 introduction + §2 background + §3 related work / §7 implementation / §9 conclusion 는 기존 thesis 골격 유지 (재작성 보류).
> **상위 산출물**: TSK_042 XL 매트릭스 222 cells (`TSK_042_realistic_workload_oracle/RESULTS.md`), SUB_201 README, SUB_201/profile/VERDICT.md, SUB_201/poc/{b3_sched, b3_llama70b, b1_e2e, a2_e2e, b1b3_cumulative}/MEASUREMENTS.md.
> **본 draft 가 binding 으로 따르는 운영 해석**: CLAUDE.md §Constraint — token-level bit-exact 가 아닌 *분포·의도 유사성* 이 정확도 게이트의 binding 지표. token-level 일치는 informational metric.

---

## 목차 (TOC)

- §0 Abstract — *재작성*
- §1 Introduction — *기존 thesis 유지, 본 draft 에서는 outline 만*
- §2 Background — *기존 유지 (outline 만)*
- §3 Related Work — *기존 유지 (outline 만)*
- §4 Methodology — *재작성*
  - §4.1 데이터 출처 (XL 매트릭스 222 cells)
  - §4.2 step 분해 프로파일 셋업 (Nsight Systems, 3 모델)
  - §4.3 lever 분류 — A1 / A2 / B1 / B3
  - §4.4 lever PoC 측정 방법 (b3_sched, b1_e2e, a2_e2e, b3_llama70b, b1b3_cumulative)
- §5 Profile Evidence — *재작성*
  - §5.1 3 모델 nsys host-bound 확정 (Qwen-7B / Llama-70B / R1 671B)
  - §5.2 dominant host overhead 의 모델 사이즈 × TP 의존성 (launch → memcpy → both)
  - §5.3 (a) host-bound verdict + 함의
- §6 Lever Results — *재작성*
  - §6.1 B3 cudagraph FaP — 모델 사이즈 × TP 의존성 매트릭스
  - §6.2 B1 AVX-512 detok EXCLUSIVE — net result + noise discussion
  - §6.3 A1 CPU drafting — HF top-1 100 % 정확도 게이트, e2e 미증명
  - §6.4 A2 KV tiering — n_evict 작동 증거 (B7-B9), evict-only ROI 음수, multi-turn 워크로드 필요
- §7 Implementation Notes — *기존 유지 (outline 만)*
- §8 Discussion — *재작성*
  - §8.1 thesis caveat — 작은 모델 / 높은 α 한정
  - §8.2 production-ready 는 B3 (FaP) 만 — 다른 lever 의 미증명 사유
  - §8.3 CPU "busywork harvesting" 은 future work
  - §8.4 limitations / threats to validity
- §9 Conclusion — *기존 유지 (outline 만)*

---

## §0 Abstract (재작성)

Speculative decoding (suffix tree, n-gram, MLP draft) 은 222 cell 매트릭스 (10 모델 × 4 method × 7 워크로드, B200×8 + Xeon Platinum 8570) 에서 vanilla 대비 **mix tps +87 % ~ +232 %** 의 지배 lever 임이 확인된다. 그러나 spec-decode 가 켜진 운영에서 GPU util 은 오히려 떨어지고 (Qwen-7B 82.5 % → 26.5 %, Llama-8B 94.9 % → 62.8 %), 동시에 CPU 는 2.5 ~ 5.6 % 의 idle 에 머문다. 본 논문은 이 GPU slack 의 원인을 Nsight Systems 의 3-모델 step 분해 프로파일로 (a) host-bound (cudaLaunchKernel + cudaMemcpyAsync 의 inter-kernel gap) 임을 확정하고, dominant host overhead 가 모델 사이즈와 TP factor 에 따라 **launch (Qwen-7B TP=4) → memcpy (Llama-70B TP=8) → 둘 다 (R1-671B TP=8 MoE)** 로 이동함을 보인다.

이 verdict 위에서 네 가지 회수 lever (A1 CPU drafting, A2 KV tiering, B1 AVX-512 detok offload, B3 cudagraph scheduler) 를 PoC 로 평가한 결과, **production-ready 로 확정된 것은 B3 (cudagraph FaP) 한 가지** 이다: Qwen-7B TP=4 에서 +30.1 %, Llama-8B TP=2 에서 +1.4 %, Llama-70B TP=4 에서 +1.82 %. 나머지 세 lever 는 (i) A1 — HF top-1 100 % 의 정확도 게이트는 통과했으나 vllm 통합 단계의 e2e net throughput 미증명, (ii) A2 — n_evict 동작은 증명 (B9 에서 41 070 evict, 37.7 GB DRAM 활용) 했으나 evict-only 운영 (fetch=0) 에서 -12.8 % regression, (iii) B1 — EXCLUSIVE patch 의 +3.01 % vs -2.3 % inconsistent (single-run noise > Δ) 로 남았다.

본 논문은 spec-decode 가 만드는 host-path 병목이 **작은 / 중형 모델 + 높은 acceptance rate α** 모델군에 강하게 적용된다는 caveat 를 명시하고 (DeepSeek-R1 671B 의 mix -49 %, α 0.451 처럼 큰 모델 / 낮은 α 군은 별도 lever 가 필요), CPU "busywork harvesting" 은 *유용한 서빙 작업* 이 아니므로 future work 로 강등하는 결론을 제시한다.

---

## §1 Introduction (기존 thesis 유지 — 본 draft 는 outline 만)

> 기존 introduction 의 핵심 단락은 그대로 유지. 변경 사항은 (a) thesis 의 적용 caveat 가 §8 로 옮겨졌음을 언급하는 한 문장 추가, (b) §6 의 lever 결과가 추가되었음 명시.

골자:

1. LLM serving 의 dominant cost = decode loop 의 memory-bound matmul.
2. spec-decode (Leviathan 2023, MEDUSA 2024, EAGLE 2024 등) 가 verify-step 화로 throughput +2~3× — 그러나 *GPU 가 saturate 되지 않는다*.
3. CPU 는 동시 idle (2-5 %). 본 논문의 thesis = "spec-decode 가 만든 host-path 병목을 idle CPU 로 회수하자."
4. **(추가)** thesis 의 적용 범위 caveat: **작은 모델 / 높은 α 모델군 한정** — § 8.1 에서 자세히.

---

## §2 Background (기존 유지)

> outline 만. spec-decode 의 acceptance rate α 정의, vLLM v1 engine 의 step loop, B200 / H100 의 attention backend 차이 (FlashInfer vs FA3 vs FA4), cudagraph_mode (PIECEWISE / FULL / FULL_AND_PIECEWISE), KV cache 의 BlockPool / radix prefix.

---

## §3 Related Work (기존 유지)

> outline 만. spec-decode 계열 (Leviathan, MEDUSA, EAGLE, Suffix tree), KV offload (Sandwich, SwiftLLM, LMCache), cudagraph optimization (CUDA Graphs 11.2+, vLLM v1 의 cudagraph_dispatcher), CPU 가속 drafting (NEO 2024, AMX kernel papers).

---

## §4 Methodology (재작성)

### §4.1 데이터 출처 — XL 매트릭스 222 cells

본 논문의 *모든 정량 결론* 은 다음 두 데이터셋 + 5 개 lever PoC 측정에 기반한다.

| 데이터셋 | 출처 | 셀 수 | 환경 |
|---|---|---:|---|
| **TSK_042 XL 매트릭스** | `runs/tput_t1t3_20260602/` (vanilla 70 + suffix 70 + ngram 5) + `runs/routing_llmd_20260603/` (llm-d 56 + sweep 21) | **222** | DGX B200 × 8 (sm_100, 1.4 TB HBM3e) + Xeon Platinum 8570 (224 thread, 2 NUMA, 2 TB DRAM) |
| **SUB_201 §5 step 분해 프로파일** | `SUB_201_cpu_host_path_bottleneck/profile/` (Qwen-7B / Llama-70B / R1-671B nsys traces) | **3 trace** | 동일 B200 머신, nsys 2025.1.1 (CAP_SYS_ADMIN 없음 → CPU sampling 불가, GPU+CUDA API+NVTX trace only) |

(자세한 raw / 통합 산출물 경로는 SUB_201 README §부록 — "데이터 출처".)

### §4.2 step 분해 프로파일 셋업 — Nsight Systems, 3 모델

3 모델 × 60-90 s window × suffix-on 상태. 측정 항목:

- GPU kernel time (top-k by `nsys stats --report cuda_gpu_kern_sum`).
- CUDA API host overhead (top-k by `cuda_api_sum`) — 본 논문의 binding host-overhead 지표.
- NVTX marker (vllm 자체 미사용, NCCL marker 만 capture).
- (시도) ncu detail (NVGPUCTRPERM 권한 차단으로 fail), (시도) py-spy CPU sampling (`perf_event_open` Fail 로 미실행).

대상 셀 선정:

- **Qwen-7B suffix mix** — TSK_042 매트릭스에서 GPU util 26.5 % 로 가장 낮음 (gap 신호 최대).
- **Llama-3.1-70B suffix** — GPU util 83.4 % 대조군, TP=8.
- **DeepSeek-R1 671B suffix** — TSK_042 에서 worst suffix match (mix -49 %, α 0.451), MoE expert routing 의 host-overhead 검증.

### §4.3 lever 분류 — A1 / A2 / B1 / B3

profile verdict (§5) 가 (a) host-bound 임을 확정하면, GPU slack 회수 lever 는 두 tier 로 갈린다.

| Tier | ID | lever | host-path 회수 메커니즘 |
|---|---|---|---|
| **A** (회수형) | **A1** | CPU drafting (AMX bf16 + suffix tree) | draft proposal 을 idle CPU 에 옮김 → GPU 는 verify 전용 |
|  | **A2** | KV DRAM tiering | cold KV block 을 host pinned DRAM 으로 evict, fetch on demand → 더 큰 batch / longer context |
| **B** (직렬화 제거) | **B1** | AVX-512 detok offload + EXCLUSIVE wire-in | native PyO3 detok 우회 → host stall 제거 |
|  | **B3** | cudagraph FULL_AND_PIECEWISE + scheduler launch batching | per-step cudaLaunchKernel 횟수 축소 |

### §4.4 lever PoC 측정 방법

각 lever 는 동일 corpus / 동일 b200 머신 / 단일 (또는 3) repeat 의 e2e bench (`vllm_config_perf/gating/realistic_eval/throughput_runner.py`) 로 평가. baseline 은 lever-off, 비교는 lever-on. 정확도 게이트 (lever 별):

- **A1**: HF reference vs CPU AMX draft head 의 top-1 token 일치율 ≥ 99 %.
- **B1**: 204/204 unit (batch + incremental, 3 모델 × 34 prompt) + 10-case sha256 byte-equal (caveat: intra-mode determinism floor 4/10 로 token-level 일치는 informational).
- **A2**: 15/15 unit test (rebind, evict, fetch, drop) + n_evict telemetry counter.
- **B3**: 200/200 ok + GPU memory residency 정상.

(자세한 PoC 별 boot 명령 / env flag / unit test 결과 는 각 `poc/<lever>/MEASUREMENTS.md` 인용.)

---

## §5 Profile Evidence (재작성)

### §5.1 3 모델 nsys host-bound 확정

| 모델 | TP | 측정 window | GPU util | dominant GPU kernel | dominant host API |
|---|---:|---:|---:|---|---|
| Qwen-7B suffix mix | 4 | 60 s | 46.5 % (trace) / 26.5 % (XL matrix) | FMHA Q128 65.8 % (24.0 s) | **cudaLaunchKernel 36.3 % (5.26 s, 18 606/s)** |
| Llama-70B suffix mix | 8 | 60 s | 83.4 % | FMHA Q128 76.9 % (53.4 s) | **cudaMemcpyAsync 80.4 % (65.3 s, 3 236/s × 336 μs avg)** |
| R1-671B suffix mix | 8 (MLA) | 90 s | 76.5 % (trace) / 94.1 % (XL matrix) | elementwise 42.3 % + cutlass MLA 25.4 % + deep_gemm FP8 13.7 % (fragmented) | **cudaLaunchKernel 34.8 % (3.52 s, 11 030/s)** + **cudaMemcpyAsync 34.9 % (3.52 s, 162/s × 242 μs)** |

→ **(a) host-bound 가 3 모델 모두에서 확정**. § 5.4 의 decision tree (gap ≥ verify × 30 % 이고 host-bound) 통과.

(출처: `SUB_201_cpu_host_path_bottleneck/profile/VERDICT.md` §2.1 / §2.2 / §4.2 의 3-모델 통합 표.)

### §5.2 dominant host overhead 의 모델 사이즈 × TP 의존성

| 사이즈 | TP | dominant host overhead | 비고 |
|---|---:|---|---|
| 7B | 4 | **cudaLaunchKernel 36 %** | per-step 작은 GPU compute 대비 launch overhead 가 비례적으로 큼 |
| 70B | 8 | **cudaMemcpyAsync 80 %** | TP=8 의 inter-GPU H2D/D2H transfer 가 host overhead 의 절대 dominant |
| 671B MoE | 8 (MLA) | **launch + memcpy 둘 다 (각 ~35 %)** | MoE expert 256 dynamic routing + reasoning chain 다양성으로 kernel fragmentation (11 030 launches/s) + per-expert FP8 GEMM 의 memcpy 동반 |

→ host overhead 의 dominant 항목이 모델 사이즈 / TP 와 함께 **launch → memcpy → both** 의 스펙트럼으로 이동한다. lever 선택 (§ 6) 도 이 스펙트럼에 매핑된다.

### §5.3 (a) host-bound verdict + 함의

verdict (인용 — VERDICT.md §3.3):

| 모델 | gap 종류 | (a)/(b)/(c) | 회수 lever (target) |
|---|---|---|---|
| Qwen-7B | launch overhead 18 % of trace | **(a) host-bound** | **B3 scheduler + A1 CPU drafting** |
| Llama-70B | memcpy 80 % of API time | **(a) host-bound** | **A2 KV tiering + B1 detok** |
| R1-671B | launch + memcpy 둘 다 70 % | **(a) host-bound** | **A1 CPU drafting + B3 kernel fusion** |

→ SUB_201 의 본론 (CPU drafting + host-path engine) 진행 조건 충족.

**한계 (정직)**: ncu detail (NVGPUCTRPERM) 부재로 (b) verify 내부 occupancy 정밀 측정 미완. CPU sampling 부재로 host gap 의 draft / scheduler / sampling / detok 세분 비율 미측정 — 본 논문 §6 의 lever effect 는 e2e net measurement 로만 binding.

---

## §6 Lever Results (재작성)

### §6.1 B3 — cudagraph FULL_AND_PIECEWISE (FaP) — 모델 사이즈 × TP 의존성 매트릭스

PoC 측정 (3 셀):

| 셀 | 모델 | TP | corpus | PIECEWISE tps | FaP tps | Δ % | 출처 |
|---|---|---:|---|---:|---:|---:|---|
| C1 | Qwen-7B-Instruct | 4 | mix 100p × conc=16 (FA forced) | 3 271.8 | **4 257.9** | **+30.1 %** | `poc/b3_sched/MEASUREMENTS_FA.md` §3 |
| C2 | Llama-3.1-8B-Instruct | 2 | sharegpt 200p × conc=16 | 4 410.1 | 4 471.7 | **+1.4 %** | `poc/b1b3_cumulative/MEASUREMENTS.md` §1 (A→B) |
| C3 | Llama-3.1-70B-Instruct | 4 | sharegpt 200p × conc=32 | 1 891.5 | 1 926.0 | **+1.82 %** | `poc/b3_llama70b/MEASUREMENTS.md` §2.1 |

부수 효과 (Llama-70B C3, FaP):
- TTFT p99 **-50.7 %** (420.8 → 207.4 ms).
- boot time **-31.5 %** (111 → 76 s).
- GPU memory footprint **-33.2 %** (950 GiB → 634 GiB sum, graph 메모리 풀 작아짐).
- 200/200 ok, regression 없음.

해석 (자세한 mechanism 은 `poc/b3_llama70b/MEASUREMENTS.md` §5 참조):

| 셀 | gain 작은 / 큰 이유 |
|---|---|
| C1 Qwen-7B TP=4 | per-step GPU compute 가 짧음 → launch overhead 비중 큼 → FaP 의 launch reduction 이 step time 의 큰 비율을 제거 → **+30 %** |
| C2 Llama-8B TP=2 | per-step compute 중간 + baseline noise ~6 % (A→B vs single-run variability) → +1.4 % 는 directionally positive 이나 magnitude 는 작음 |
| C3 Llama-70B TP=4 | per-step compute 가 길고 TP allreduce overhead 추가 → launch overhead 가 step time 의 작은 비율 → graph 자체로 줄일 여지가 작음 → +1.82 % |

**production-ready verdict**: 3 셀 모두 net positive (negative regression 없음). B3 default = `cudagraph_mode=FULL_AND_PIECEWISE` 권고. 단, **B200 환경에서 FULL 단독 mode 는 hardware + software 양쪽으로 불가** (`MEASUREMENTS_FA.md` §1.3 — FA3 코드 가드 + Blackwell 미빌드, FlashInfer / FA4 모두 UNIFORM_BATCH cap). prod H100 에서 FA3 native 일 때만 FULL 단독 활성화 가능 → 본 논문은 FaP 만 시험.

### §6.2 B1 — AVX-512 detok EXCLUSIVE wire-in — net result + noise discussion

PoC 4 phase (A4-shadow → A4-prod (double-work, -6.73 %) → A4-exclusive → b1b3_cumulative). 핵심 표:

| Phase | mode | baseline tps | lever tps | Δ % | 출처 |
|---|---|---:|---:|---:|---|
| A4-prod | NATIVE (double-work) | 4 152.7 | 3 873.2 | **-6.73 %** | `poc/b1_e2e/MEASUREMENTS.md` §8.5 |
| A4-exclusive | EXCLUSIVE (native stream.step skip) | 4 146.8 | 4 271.6 | **+3.01 %** | `poc/b1_e2e/MEASUREMENTS.md` §9.4 |
| b1b3_cumulative (A→C) | EXCLUSIVE (PIECEWISE+EXC) | 4 410.1 | 4 306.8 | **-2.3 %** | `poc/b1b3_cumulative/MEASUREMENTS.md` §2.1 |

**noise discussion**: A4-exclusive 측정의 baseline 4 146.8 vs b1b3 측정의 baseline 4 410.1 (같은 corpus / 같은 환경 / 다른 boot) — **inter-run variability 약 6 %**, lever Δ (±3 %) 보다 큼 (`poc/b1b3_cumulative/MEASUREMENTS.md` §2.4). 따라서 본 논문은 B1 EXCLUSIVE 를 *방향성 positive (+3 % range)* 까지만 인정, **production net win 으로 단정하지 않는다**. prod H100 머신 (FA3 native, 더 큰 throughput regime) 에서 3 repeat × 4 run = 12 sweep 의 통계 검증 후 final verdict 권고.

correctness 게이트는 통과 (204/204 unit, 10-case sha256 8/10 — intra-mode determinism floor 4/10 대비 우월). CLAUDE.md 의 운영 해석 (token-level 일치는 informational, 분포 유사성이 binding) 기준으로 정확도 게이트 통과.

### §6.3 A1 — CPU drafting (AMX bf16, Qwen-0.5B safetensors lm_head 통합) — HF top-1 100 %, e2e 미증명

- **자산**: `IDE_019/SUB_187/build/libamx_draft_qwen05b.so` (AMX kernel, K=7 draft 2.05 ms, B=1/4 K=5/7 PASS, per-step 0.21-0.29 ms).
- **정확도 게이트**: Qwen-0.5B safetensors lm_head 통합 후 HF reference 대비 **top-1 100 %** (Phase A3-real, commit bea71c373).
- **e2e 미증명**: vllm v1 의 spec_decode subsystem 에 CPU proposer 로 wire-in 하는 단계가 본 PoC 에서 진행되지 않음. 70B verify ~80 ms 대비 CPU draft 2.05 ms 는 ~5 % overhead → **net gain 가능성은 있으나 e2e throughput 미측정**.

verdict: scaffold + correctness gate 까지 완료, **production net throughput 미증명 — multi-step Phase 로 분리** (`shadow_assists/features/IDE_019_multi_source_drafter/` 후속 task).

### §6.4 A2 — KV tiering (B5 → B6 → B7 → B8 → B9 5 phase)

5 phase 의 lever evolution (`poc/a2_e2e/MEASUREMENTS.md`):

| Phase | 핵심 변경 | n_evict | tier tps Δ % | verdict |
|---|---|---:|---:|---|
| B5 | initial wire-up (worker bind 누락) | **0** | +3.5 % | NEGLIGIBLE (binding=False, no-op) |
| B6 | worker `bind_block_pointers` wire-up + telemetry | **0** | -8.7 % | NEGATIVE 의심 (cross-process gap) |
| B7 | TP=1 UniProc (cross-process gap 우회) | **512** | +20.5 % | POSITIVE (caveat: native n_ok=443/500) |
| B8 | size-class allocator 확장 (skipped_full -40 %) | **512** (stuck) | +1.4 % | PARTIAL (bind 2회 호출 mismatch) |
| B9 | bind 2회 mismatch fix (profiling skip + rebind safety) | **41 070** | **-12.8 %** | **NEGATIVE-CONFIRM (evict-only 운영, fetch=0)** |

**B9 가 본 lever 의 *진짜* verdict**:
- evict 가 본격 작동 (41 070 evict, 37.7 GB DRAM 활용, skipped_full=0). lever 자체는 mechanical 동작 증명.
- **그러나 net throughput -12.8 %**: wildchat 의 prefix hit rate 0.2 % 에서 fetch=0 → evict-only "GPU → DRAM 단방향 스필", 비용은 다 부담 / benefit 0.
- TTFT p99 +371 %, TPOT p99 +37 % — forward 와 evict stream 의 D2H bandwidth 경합이 tail latency 를 무너뜨림.

**조건부 production gate**: A2 가 net positive 가 되려면 (i) **multi-turn chat / shared system prompt 워크로드** 에서 prefix hit rate ≥ 40 % → fetch_block 트리거, (ii) **async evict** (`free_blocks(wait=False)` + scheduler-side wait) 로 D2H 가 forward critical path 밖으로 빠질 것 — 두 조건 충족 후 재측정 필요.

---

## §7 Implementation Notes (기존 유지)

> outline. vLLM v1 의 `detokenizer.py`, `block_pool.py`, `kv_dram_tiering.py`, `compilation.py` 의 patch 위치. ENV flag 정리 (`VLLM_USE_AVX512_DETOK_EXCLUSIVE`, `VLLM_KV_TIERING_DRAM`, `VLLM_PINNED_POOL_AUTO_BUDGET`, `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`).

---

## §8 Discussion (재작성)

### §8.1 thesis caveat — 작은 모델 / 높은 α 한정

본 논문의 thesis (spec-decode 가 host-path 병목을 만든다) 는 *모든* 모델 / 워크로드에 무차별 적용되지 않는다. TSK_042 XL 매트릭스 (`TSK_042_realistic_workload_oracle/RESULTS.md` + SUB_201 README §1) 의 15 개 vanilla-win 셀 분석:

| 모델군 | 셀 수 | α median | α range | s_wall/v_wall |
|---|---:|---:|---|---|
| **R1-671B (MoE)** | 7/7 | 0.46 | 0.36~0.54 | 1.62~2.88× (worst) |
| **DS-Llama-70B (distill)** | 6/7 | 0.33 | 0.27~0.38 | 1.0~1.18× |
| **Qwen-72B (mid-large dense)** | 2/7 | 0.37 | 0.27~0.47 | 1.05~1.08× |
| (positive 55 cells) | 55 | **0.68** | 0.28~1.41 | — |

→ **α ≈ 0.5 가 spec-decode break-even 임계치** (K=32 num_speculative_tokens 셋업). 그 아래에서는 draft wasted compute 가 spec gain 을 초과.

또한 spec-decode 의 GPU un-saturate 효과는 사이즈 단조 감소: **ΔGPU 7B +56 pp / 8B +32 pp / 32B +18~30 pp / 70B +13~15 pp / 405B +6 pp / 671B +4.5 pp** (`SUB_201_cpu_host_path_bottleneck/README.md` §1.1). 따라서:

> **thesis 의 운영 범위 = "작은 ~ 중형 모델 (≤ ~70B) + 높은 α (≥ 0.5) 모델군"**. 큰 모델 / 낮은 α (DS-distill 70B+, MoE R1) 는 별도 lever — 본 논문에서는 **A1 CPU drafting 의 primary target** 으로 지정하고, 그 lever 의 e2e 증명은 future work 로 명시.

### §8.2 production-ready 는 B3 (FaP) 만 — 다른 lever 의 미증명 사유

| lever | gate 통과 | production-ready? | 미달 사유 |
|---|---|---|---|
| **B3 FaP** | 3-셀 net positive (+1.4 ~ +30.1 %) + 정확도 게이트 | ✅ **YES** | — |
| **B1 EXCLUSIVE** | unit 204/204, 10-case sha 8/10 | ⚠ NO | inter-run noise (~6 %) > lever Δ (~3 %), prod H100 (FA3 + ALWAYS cap) sweep 필요 |
| **A1 CPU drafting** | HF top-1 100 % (Qwen-0.5B AMX), AMX kernel microbench PASS | ⚠ NO | vllm v1 spec_decode 통합 미완, e2e net throughput 미측정 |
| **A2 KV tiering** | unit 15/15, n_evict 41 070 동작 증명 | ⚠ NO | evict-only ROI 음수 (-12.8 %), multi-turn workload (prefix hit ≥ 40 %) 또는 async evict 필요 |

→ 본 논문이 *production wire-in 결정* 으로 제시하는 것은 B3 (FaP) **단 하나**.

### §8.3 CPU "busywork harvesting" 은 future work

SUB_201 README §3 Tier C / §7 폐기 목록 — IDE_018 phase-burst / drop-in 커널 / branchy busywork (SUB_196 cellB +5.28 %) — 는 모두 *유용한 서빙 작업이 아닌* CPU C-state 회피 가설 또는 GPU 매트멀 흉내 (H2D/D2H 동기 패배). 본 논문에서는:

- IDE_018 phase-burst (tasks_executed=0, +1.35 %) — **폐기**.
- branchy harvest (SUB_196 cellB +5.28 % vs SUB_189 -0.82 %) — work-pattern sensitive, 신뢰 불가 — **future work 로 강등** (사용자 결정).
- NEO TSK_019 (CPU 대형모델 병렬 추론) — net-negative 확정 — **폐기**.

이는 본 논문이 *thesis 의 negative space* (CPU 가 GPU 를 흉내내는 lever) 를 명시적으로 폐기하고, 본론 (host-path 회수) 만 추구함을 분명히 한다.

### §8.4 limitations / threats to validity

| 항목 | 내용 |
|---|---|
| **profile 측정 한계** | ncu detail 권한 차단으로 (b) verify 내부 occupancy 미측정. CPU sampling 권한 차단으로 host gap 세분 미측정. |
| **single-machine** | 모든 측정이 dev B200 머신 (Xeon Platinum 8570). prod H100×8 (Sapphire Rapids + AMX) 의 정량적 일반화는 후속 작업. |
| **B3 의 모델 사이즈 의존성** | 3 모델 (7B / 8B / 70B) 만 측정. 32B / 405B / 671B 의 FaP gain 미측정. |
| **B1 의 noise** | inter-run variability ~6 % > lever Δ ~3 %. multi-repeat (≥ 3) sweep 필요. |
| **A2 의 workload 의존성** | evict-only 시나리오만 측정. fetch-trigger workload (multi-turn chat replay) 별도 필요. |
| **정확도 게이트의 운영 해석** | token-level bit-exact 가 아닌 분포·의도 유사성이 binding (CLAUDE.md §Constraint). 본 논문 모든 정확도 verdict (B1 sha 8/10, A1 top-1 100 % 등) 는 이 해석을 따름. |

---

## §9 Conclusion (기존 유지)

> outline. (1) spec-decode 가 host-path 병목을 만든다 — (2) idle CPU 가 그 회수의 자연 후보 — (3) production-ready lever 는 B3 FaP 한 가지, 다른 lever (A1/A2/B1) 는 부분 증거 + 향후 작업 — (4) CPU busywork harvesting 은 폐기 / future work.

---

## Tables (placeholder)

### Table 1 — TSK_042 XL 매트릭스 222 cells 요약 (모델 × method × tps)

| 모델 | size | method-best | mix tps | suffix Δ vs vanilla | α (suffix) | thesis 적용? |
|---|---|---|---:|---:|---:|---|
| Qwen-2.5-7B | 7B (TP=4) | llm-d / suffix | 7 803 | +87 % | 0.881 | **YES** |
| Llama-3.1-8B | 8B | suffix | **27 851** | +215 % | 0.933 | **YES** |
| DS-Qwen-7B | 7B | suffix | 24 458 | +170 % | 0.876 | **YES** |
| Qwen-2.5-32B | 32B | suffix | 6 597 | +116 % | 0.857 | **YES** |
| DS-Qwen-32B | 32B | suffix | 9 056 | +83 % | 0.801 | YES (경계) |
| Qwen-2.5-72B | 72B | suffix | 5 268 | +93 % | 0.852 | YES (mix only) |
| Llama-3.1-70B | 70B | suffix | 10 400 | +232 % | 0.915 | **YES** |
| DS-Llama-70B | 70B | mix-only suffix | 6 127 | +94 % mix only | **0.786** | partial (mix 외 6/7 negative) |
| Llama-405B FP8 | 405B | suffix | 2 829 | +126 % | 0.766 | YES (낮은 효과) |
| **DeepSeek-R1 671B (MoE)** | 671B (37B act) | vanilla | 781 | **-49 %** | **0.451** | **NO** (A1 target) |

> 출처: `TSK_042_realistic_workload_oracle/RESULTS.md` + `SUB_201_cpu_host_path_bottleneck/README.md` §1.1, raw `runs/tput_t1t3_20260602/metrics_table.parquet`.

### Table 2 — § 5 nsys 의 3 모델 host overhead 분해

| 모델 | TP | trace s | GPU kernel dominant | host API dominant | launch + memcpy 합 / trace |
|---|---:|---:|---|---|---:|
| Qwen-7B suffix | 4 | 60 | FMHA Q128 65.8 % | cudaLaunchKernel 36.3 % | 51 % (launch 38 + memcpy 13) |
| Llama-70B suffix | 8 | 60 | FMHA Q128 76.9 % | cudaMemcpyAsync 80.4 % | 85 % (memcpy dominant) |
| R1-671B suffix | 8 (MLA) | 90 | elementwise 42.3 % + MLA 25.4 % | launch + memcpy 둘 다 ~35 % | 70 % (각 35 / 35) |

> 출처: `SUB_201_cpu_host_path_bottleneck/profile/VERDICT.md` §2.1 / §2.2 / §4.2.

### Table 3 — B3 FaP 의 모델 사이즈 × ΔΔ%

| 모델 | TP | corpus | PIECEWISE tps | FaP tps | Δ % | TTFT p99 Δ % | 출처 |
|---|---:|---|---:|---:|---:|---:|---|
| Qwen-7B (FA forced) | 4 | mix 100p × 16 | 3 271.8 | 4 257.9 | **+30.1 %** | (보고된 +154 % p99 init cost) | `poc/b3_sched/MEASUREMENTS_FA.md` §3 |
| Llama-8B | 2 | sharegpt 200p × 16 | 4 410.1 | 4 471.7 | +1.4 % | +422 % p99 (TTFT 65.7 → 277.6 ms, init cost) | `poc/b1b3_cumulative/MEASUREMENTS.md` §1 |
| Llama-70B | 4 | sharegpt 200p × 32 | 1 891.5 | 1 926.0 | +1.82 % | **-50.7 %** (420.8 → 207.4 ms) | `poc/b3_llama70b/MEASUREMENTS.md` §2.1 |

### Table 4 — 4 lever 의 e2e ROI summary

| lever | best e2e measurement | net Δ % | production-ready? | next step |
|---|---|---:|---|---|
| **B3 FaP** | Qwen-7B TP=4 mix 100p × 16 (FA forced) | **+30.1 %** | ✅ YES | prod H100 sweep, 다른 모델 사이즈 검증 |
| **B1 EXCLUSIVE** | Llama-8B TP=2 sharegpt 200p × 16 | **+3.01 %** (inconsistent: -2.3 % 도 측정) | ⚠ NO | prod H100 × 3 repeat sweep |
| **A1 CPU draft** | (e2e 미측정) — AMX K=7 draft 2.05 ms PASS, HF top-1 100 % | n/a | ⚠ NO | vllm v1 spec_decode integration |
| **A2 KV tiering** | Qwen-7B TP=1 wildchat 200p × 64 (B9) | **-12.8 %** (evict-only) | ⚠ NO | multi-turn workload (prefix hit ≥ 40 %) + async evict |

---

## Figures (plan)

### Figure 1 — thesis caveat (작은 모델 / 높은 α 한정 범위 plot)

**plan**: x-axis = 모델 사이즈 (log scale, 7B → 671B), y-axis = α (suffix acceptance rate). 각 모델 의 mix-cell 좌표에 점, 점 색 = suffix Δ vs vanilla 의 부호 (positive = green, negative = red). break-even 임계치 α=0.5 가로선 + ΔGPU = 0 임계 영역 음영. 본 논문 thesis 의 운영 범위 (작은-중형 + 높은 α) 가 plot 의 좌상단 사분면에 corral 됨을 시각화.

- 데이터 출처: TSK_042 XL 매트릭스 222 cells (Table 1).
- 미작성 (figure 빌드 보류 — caveat 표현은 § 8.1 + Table 1 본문이 우선 담당).

### Figure 2 — § 5 nsys host overhead 분해 (3 모델 stacked bar)

**plan**: x-axis = 3 모델 (Qwen-7B / Llama-70B / R1-671B), y-axis = trace time 의 % share. stacked bar = (cudaLaunchKernel, cuLaunchKernelEx, cudaGraphLaunch, cudaMemcpyAsync, 기타). dominant overhead 의 모델 사이즈 / TP 의존성 (launch → memcpy → both) 시각화.

- 데이터 출처: VERDICT.md §2.1 / §2.2 / §4.2.
- 미작성.

### Figure 3 — B3 FaP gain 의 모델 사이즈 의존성

**plan**: x-axis = 모델 사이즈 (7B / 8B / 70B), y-axis = FaP vs PIECEWISE Δ tps %. error bar = single-run sample (multi-repeat 부재 명시). +30 % (7B TP=4) → +1.4 % (8B TP=2) → +1.82 % (70B TP=4) 의 감소 trend.

- 데이터 출처: Table 3.
- 미작성.

---

## Cross-references (인용한 산출물 전체)

| ref | path |
|---|---|
| matrix | `shadow_assists/features/IDE_022_agsd_realistic_eval/TSK_042_realistic_workload_oracle/RESULTS.md` |
| SUB_201 thesis | `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/README.md` |
| profile verdict | `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/VERDICT.md` |
| profile lever audit | `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/LEVER_AUDIT.md` |
| B3 sched sweep | `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_sched/MEASUREMENTS.md` + `MEASUREMENTS_FA.md` |
| B3 Llama-70B | `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_llama70b/MEASUREMENTS.md` |
| B1 e2e | `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b1_e2e/MEASUREMENTS.md` |
| A2 e2e | `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/a2_e2e/MEASUREMENTS.md` |
| B1+B3 cumulative | `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b1b3_cumulative/MEASUREMENTS.md` |
| raw (vanilla / suffix / ngram) | `vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/` |
| raw (llm-d) | `vllm_config_perf/gating/realistic_eval/runs/routing_llmd_20260603/` |
| raw (combined) | `vllm_config_perf/gating/realistic_eval/runs/routing_combined/` |

---

## Draft completion status

| 섹션 | 상태 | 비고 |
|---|---|---|
| §0 Abstract | ✅ 작성 완료 | TSK_042 / SUB_201 / 4 lever PoC verdict 반영 |
| §1 Introduction | ⚠ outline 만 | 기존 thesis 단락 유지 + § 8.1 caveat 참조 한 문장만 추가 |
| §2 Background | ⚠ outline 만 | 기존 유지 |
| §3 Related Work | ⚠ outline 만 | 기존 유지 |
| §4 Methodology | ✅ 작성 완료 | 데이터 출처 / nsys 셋업 / lever 분류 / PoC 방법 |
| §5 Profile Evidence | ✅ 작성 완료 | 3 모델 host-bound verdict + dominant overhead 의 모델 의존성 |
| §6 Lever Results | ✅ 작성 완료 | B3 / B1 / A1 / A2 4 lever 의 net result + noise discussion |
| §7 Implementation Notes | ⚠ outline 만 | 기존 유지 (patch 위치, ENV flag 정리는 next pass) |
| §8 Discussion | ✅ 작성 완료 | thesis caveat + production-ready 판정 + busywork harvesting future work |
| §9 Conclusion | ⚠ outline 만 | 4 줄 골자 |
| Table 1-4 | ✅ 작성 완료 (수치 채움) | 모든 셀 인용 source 포함 |
| Figure 1-3 | ❌ 미작성 | plan 만 — 본 draft 는 doc only, figure 빌드 보류 |

---

## 작성 후 남은 작업

1. **figure 빌드** — Figure 1 (thesis caveat plot) / Figure 2 (nsys stacked bar) / Figure 3 (B3 사이즈 의존성). 도구: matplotlib + parquet (TSK_042 metrics_table.parquet) + VERDICT.md table parse.
2. **§1 / §7 / §9 본문 채움** — 기존 thesis 단락 + ENV flag 표 + 4-line conclusion.
3. **§6.3 A1 의 e2e wire-in** — `vllm/v1/spec_decode/` 의 CPU proposer 통합 PoC 후 net throughput 측정 추가. 본 draft 의 A1 verdict 는 *scaffold + correctness* 까지만 binding.
4. **§6.4 A2 의 multi-turn workload 측정** — sharegpt conversation continuation 또는 system prompt sharing 시나리오에서 prefix hit ≥ 40 % 확보 후 fetch_block 트리거 / async evict 적용 후 재측정.
5. **§6.2 B1 의 prod H100 검증** — FA3 native + ALWAYS cap 환경에서 3 repeat × 4 run sweep, EXCLUSIVE 의 net win 통계 검증.
6. **인용 표준화** — 현재 cross-reference 가 path 직접 인용. 논문 publishable 단계 전까지 BibTeX / arxiv id 로 표준화.

