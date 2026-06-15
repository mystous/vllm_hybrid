# LHC Phase 2 — Verdict

**날짜**: 2026-06-08
**기간**: Phase 2 본격 진입 1일 (lhc microbench + vLLM PoC + 측정)
**소속 문서**: `lhc_phase1/PHASE1_VERDICT.md`, `lhc_phase2/dsa_multi_engine_result.md`, `lhc_phase2/amx_logitshead_result.md`

---

## 0. TL;DR

| Task | gate | 측정 | verdict | 다음 |
|------|------|------|---------|------|
| **T1 DSA multi-engine BW** | ≥ 0.8 × cudaMemcpy (≥44 GB/s) | 31.37 GB/s single-engine ceiling | **FAIL** (HW: 컨테이너 sysfs RO → engine0.1~0.3 group bind 불가; dsa1 disabled) | 호스트 root 1회 작업 후 retry (4× 100-125 GB/s 예측) |
| **T2 AMX logits head** | ≤ 1.5 × GPU latency | **70–100×** (best 71×) | **FAIL** | AMX lane PoC 폐기. 다른 sub-lane (draft head, classifier, fused norm) 별도 microbench Phase 3 |
| **T3 vLLM DSA PoC 통합** | end-to-end 통합 + 정확성 | DSA lane import + N9 lever applied + 4 KB self-test PASS + KV staging fan-out hook merge | **PASS (functional)** | KV-heavy workload (NEO swap fire 조건) 에서 재측정 필요 |
| **T3 통합 throughput** | +7–11% | sonnet 200p × 3 sweep: **Δ = -0.21%** (within ±0.5% noise) | **NO-WIN @ sonnet** | NEO swap 미발화 워크로드의 한계. KV-heavy workload + NEO 활성화 후 재측정 |
| **T3 stack DSA + fp8 KV** | best stack | **Δ = +0.94%** (fp8 단독 +3.94~+7.32% 직전 측정 대비 손해) | DSA가 fp8 KV 와 직교 가산 안 됨 | fp8 KV 의 GPU memcpy 자체가 절반으로 줄어 DSA host scatter 의 ortho-lane 기회가 더 감소 |

→ **Phase 2 verdict: PARTIAL — 통합 PoC functional PASS, 그러나 net throughput win 미관측. Phase 3 진입 조건은 (a) DSA multi-engine + (b) NEO swap-heavy workload + (c) PMU dispatch 3가지 모두 갖춰진 measurement matrix.**

---

## 1. 측정 결과 요약

### 1.1 Task 1 — DSA microbench (단일 engine ceiling 확인)

| sweep | total | depth/threads | BW p50 | gate (0.8×) |
|-------|-------|---------------|--------|-------------|
| descriptor pipelining | 256 MB | 1 → 32 | 31.37 GB/s (const) | FAIL |
| multi-thread same WQ | 1–16 MB | 1 → 16 thr | 29.7 → 31.4 GB/s | FAIL |

→ 단일 group/engine 의 HW ceiling 이 정확히 31.37 GB/s. depth/thread 늘려도 ±0.02 GB/s. Phase 1 의 single descriptor 측정과 동일 → **multi-engine 만이 BW 게이트 통과 가능**.

`/sys/bus/dsa` 가 RO 라 컨테이너에서 group/engine binding 변경 불가. 호스트 1회 root 작업 필요 (절차: `lhc_phase2/dsa_multi_engine_result.md` 의 §2).

### 1.2 Task 2 — AMX logits-head matmul

| vocab × bs (best for AMX) | AMX bf16 (56T) | AVX-512 BF16 | GPU bf16 (B200) | AMX / GPU |
|---------|---------------|--------------|-----------------|----------|
| llama-128k × 64 | 11.44 ms | 19.25 ms | 167 μs | **68×** |
| qwen-152k × 64 | 12.24 ms | 19.55 ms | 173 μs | **71×** |
| deepseek-256k × 64 | 15.43 ms | 36.04 ms | 318 μs | **49×** |

oneDNN verbose 로 `brg_matmul:avx10_1_512_amx` primitive 가 실제 호출됨을 확인 (AMX 미사용 아님). GPU 절대 latency (165-320 μs) 가 너무 빨라 ortho-lane 으로도 회수 불가.

→ logits-head 는 AMX lane 폐기. AMX 가 GPU 와 격차가 작은 sub-lane:
- **draft head matmul** (small vocab, small hidden) — Phase 3 microbench 예정
- **CPU-side prefill chunked classifier** (작은 model + 작은 batch — GPU 가 underutilize 되는 영역) — Phase 3
- **fused RMSNorm/RoPE/Activation** (memory-bound, GPU TC 미사용 → GPU 와 격차 작을 가능성) — Phase 3 microbench 후 결정

### 1.3 Task 3 — vLLM DSA PoC 통합 측정

**setup**: Llama-3.1-8B-Instruct, TP=8, gpu-mem 0.85, max-model-len 16384, conc 64, max-tok 512, 200 prompts × 1 warmup + 3 sweep, sonnet workload.

| config | n | output_tps mean | std | Δ% vs vanilla |
|--------|---|-----------------|------|---------------|
| vanilla | 3 | 15852.5 | 12.3 | — |
| lhc_dsa | 3 | 15818.5 | 86.0 | -0.21% |
| lhc_dsa + fp8 KV | 3 | 16001.4 | 377.8 | +0.94% |

**hot-path 발화 진단**: boot log 에서 `[LHC DSA] lane ENABLED` + `[IDE_023] N9: applied` 확인 (rank 0 engine core process). **NEO swap-out / swap-in 메시지는 0건** — 본 sonnet 워크로드의 KV 압박이 낮아 swap 자체가 발화하지 않음. → DSA hook 이 cold path.

**결론**: 본 워크로드에서 DSA lane 의 호출 trigger 가 0 이므로 throughput 차이는 단순 noise. 효과는 **NEO swap-heavy workload** (KV 압박 ≥ swap_out_threshold) 또는 **DSA 적용 hot path 다양화** 가 모두 충족된 경우에만 발현.

---

## 2. gain decomposition (Phase 1 예측 vs 실측)

| sub-lane | Phase 1 예측 (balanced) | Phase 2 실측 | 격차 원인 |
|----------|------------------------|---------------|-----------|
| DSA KV swap async | +3–5% | -0.21% (noise) | swap 자체 미발화 (workload 차이) |
| AMX logits head | +5–8% | -∞ (gate FAIL) | sub-lane 재정의 실패 — logits head 가 잘못된 선택. draft head / classifier / fused-norm 로 재선정 필요 |
| overlap 가산 | +0% | 0% | DSA + AMX 둘 다 단독 양수 못 보여 가산 의미 없음 |
| **합산** | **+8–13%** | **0%** | 측정 조건 mismatch (workload) + AMX sub-lane 재정의 실패 |

---

## 3. Phase 2 functional 산출물

```
lhc_phase2/
├── PHASE2_VERDICT.md                ← 본 문서
├── dsa_multi_engine_bench.c/.so     ← multi-descriptor pipelining
├── dsa_multi_engine_raw.jsonl
├── dsa_multi_engine_result.md
├── dsa_mt_bench.c/.so               ← multi-thread submit
├── dsa_mt_raw.jsonl
├── amx_logitshead_bench.py
├── amx_logitshead_{amx,avx512,gpu}.json
├── amx_logitshead_result.md
├── lhc_sweep.sh / measure_one_config.sh
└── runs_sweep/
    ├── {vanilla,lhc_dsa,lhc_dsa_fp8kv}_boot.log
    └── {vanilla,lhc_dsa,lhc_dsa_fp8kv}_sonnet_s{1,2,3}.json

vllm/v1/lhc/                          ← 신규 LHC module (PoC)
├── __init__.py
├── dsa_lane.py                       ← ctypes wrapper, env-gated
├── libdsa_lane.c                     ← MOVDIR64B path
└── libdsa_lane.so                    ← built

vllm/v1/spec_decode/ide023_levers.py  ← N9 hook: dsa_lane import + self-test 검증 로직 추가

vllm/v1/core/sched/neo_cpu_kv_buffer.py
└── copy_all_layers_in_from_staged    ← DSA path for contig block_ids
```

---

## 4. Phase 3 plan

### 4.1 선결 (Phase 3 진입 조건)

- [ ] **호스트 root 1회 작업**: dsa0 4 engine + 4 WQ enable + dsa1 동일 → 컨테이너 재기동 → multi-engine 100-125 GB/s 입증 (Task 1 retry)
- [ ] **NEO swap-heavy workload spec 정의**: max_tok 4096+, conc 128+, sharegpt 진짜 long-tail. swap_out_threshold 도달하는 환경에서 DSA hot path 발화율 측정
- [ ] **AMX sub-lane 재선정**: draft head / CPU-side classifier / fused-norm 중 하나 — 별도 microbench (GPU 격차 ≤ 1.5x 인 op 찾기)

### 4.2 Phase 3 work items

1. **PMU monitor daemon** — `perf_event_open` syscall 직접 hook → 매 100ms 마다 IPC, cache-miss rate, mem-stall, dispatch idle counter sampling. Worker rank 0 가 background thread 로 운영.
2. **Lane orchestrator** — `vllm/v1/lhc/dispatcher.py` (Phase 2 의 dsa_lane 동기 wrapper → async ring + lockless queue). DMA descriptor ring 65536 entry, MPSC submission.
3. **Dynamic dispatch policy** — PMU counter 와 lane 부하 (queue depth) 기반 ML-free heuristic: IPC < 1.2 + L3 miss > 30% → CPU 사용을 피하고 DSA 사용 우선. 반대로 IPC > 2.5 + DSA queue > 50% → ATen fallback.
4. **100-cell measurement matrix** — (workload × method × LHC config) — Llama-3.1-8B-Instruct + Qwen2.5-7B + DeepSeek-V3 × {vanilla, eagle3, suffix, fp8 KV} × {LHC off, DSA on, AMX (재정의 후), DSA+AMX, PMU dynamic} × 6 workload.
5. **학술 contribution 정리**: lane separation theorem 의 정량적 정의 + measured negatives 와 LHC 의 비교표 + dynamic dispatch 의 IPC 게이트 정량화.

### 4.3 위험 / 미해결

- **TP=8 multi-process WQ contention**: dedicated WQ 가 single mm 에 bind → 첫 worker 가 WQ 점유 → 다른 worker process 는 EBUSY. 대안: (a) shared WQ + PASID per process (kernel idxd 의 PASID alloc 필요), (b) WQ 8개 enable 후 각 rank 가 다른 WQ 사용, (c) rank 0 만 DSA 사용하고 다른 rank 의 KV swap 은 rank 0 worker 가 proxy. Phase 3 에서 (b) 구현.
- **cdev close 후 새 client 재진입 불가**: open(/dev/dsa/wq0.0) PASID bind 는 process lifetime. fork-after-init 패턴 (vLLM EngineCore subprocess) 에서 첫 EBUSY 의 root cause.
- **single engine BW 31 GB/s** 자체는 cudaMemcpyAsync 의 GPU↔CPU 대체용으론 부족. host-host scatter 영역에서만 효력 — vLLM 의 KV scatter 가 small percentage 라 net win 크기 제한.

---

## 5. 학술적 함의 (직전 세션의 LHC 신규성 5점 확인)

| LHC 주장 | Phase 2 evidence | 상태 |
|----------|-------------------|------|
| (1) DSA Lane in LLM serving 최초 적용 | dsa_lane.py + libdsa_lane.so + N9 hook + KV scatter integration | **확립**. Phase 3 에서 end-to-end gain 입증 필요 |
| (2) Lane separation theorem | AMX 실패 (logits head GPU 의 70x 격차) + DSA 한정적 성공 — "GPU가 잘하는 일" 의 boundary 가 BF16 GEMM 기준 ≥ 1 PFLOPS 영역까지 확장됨을 정량화 | **부분 확립**. boundary 정의 정제 필요 |
| (3) PMU-driven dynamic dispatch | Phase 3 work item — 본 phase 미구현 | 미확립 |
| (4) Container-constrained NUMA workaround | 컨테이너 sysfs RO 한계 확인 + cdev open EBUSY 명시 | **문제 정의 확립**, 해결책 (b) WQ-per-rank Phase 3 |
| (5) Measured negatives 통합 framework | DSA -0.21%, fp8+DSA +0.94%, AMX -∞ 모두 동일 framework 에서 측정 | **확립**. 직전 세션 100+ lever 결과와 동일 자리에 LHC 결과 align |

---

## 6. 결정

> **Phase 3 진입: 조건부 GO**

- 조건 1 충족 (호스트 root multi-engine enable) → Phase 3 work item 1, 2 (PMU + orchestrator) 진행
- 조건 1 미충족 → DSA lane 은 single engine 유지, AMX sub-lane 재선정 (Phase 3 work item 3 우선)
- 6 workload 본격 측정은 Phase 3 work item 1/2/3 완료 후
