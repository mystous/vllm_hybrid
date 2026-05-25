# TSK_020 — INDEX (navigation hub)

> **목적**: TSK_020 의 모든 doc / measurement / plan 의 single entry point.
> **상태**: 활성 — **★ 현 best = Trident core (SUB_093 sonnet 11,676.9 / SUB_089 canonical 11,687.4 / fair +51~+52%)**
> **branch**: `feat/spec-decode-tuning`
> **상위**: [`README.md`](README.md) — Best Configuration index
> **★★ user-facing entry point**: [`/spec_decoding/README.md`](../../../../spec_decoding/README.md) — top-level production guide (Trident core + AGSD framework)
> **종합 리포트**: [`COMPREHENSIVE_REPORT_20260525.md`](COMPREHENSIVE_REPORT_20260525.md) / [`OUTSTANDING_CONTRIBUTIONS_20260525.md`](OUTSTANDING_CONTRIBUTIONS_20260525.md) / **[`★ _ALL_RESULTS_20260526.md`](measurements/_ALL_RESULTS_20260526.md)** (전체 129 cell 단일 doc) / [`SUB_093`](measurements/sub093_full_matrix_util_20260525/RESULTS.md) / [`SUB_094`](measurements/sub094_agsd_e2e_20260525/RESULTS.md) / [`SUB_095`](measurements/sub095_agsd_e2e_multi_model_20260525/RESULTS.md) / [`SUB_096`](measurements/sub096_large_models_20260525/RESULTS.md)

---

## ★★★ 0. 현 best — Trident core / AGSD framework (SUB_093, 2026-05-25)

**용어**: **Trident core** = spec config 자체 (suffix+PIECEWISE+gmu=0.80, always-on) / **AGSD** = Trident core + workload/model-size gating. Llama 70B 단독 영역 모든 workload 영역 Trident core 가 best 이므로 **AGSD = Trident core** (gating = 항상 suffix).

### Trident core × 6 workload (SUB_093)

| workload | vanilla | **Trident core** | fair contribution | CPU% | GPU% |
|---|---:|---:|---:|---:|---:|
| **sonnet** | 7,678.7 | **11,676.9** | **+52.1%** ⭐ | 5.3 | 73.3 |
| **chat** | 2,266.8 | **3,830.4** | **+68.9%** ⭐ | (config-wide) | (config-wide) |
| **code** | 6,717.7 | **7,981.4** | **+18.8%** ⭐ | — | — |
| **mix-sh** (M1 60:20:20) | 6,325.9 | **10,297.7** | **+62.8%** ⭐ | — | — |
| **mix-bal** (M2 34:33:33) | 6,053.9 | **9,514.3** | **+57.2%** ⭐ | — | — |
| **mix-ch** (M3 10:20:70) | 6,494.9 | **9,457.3** | **+45.6%** ⭐ | — | — |

→ vanilla 영역 CPU 5.6% / GPU 93.8% / Trident core 영역 CPU 5.3% / GPU 73.3% — wall 31% 단축.

**활성화 (한 줄, Trident core)**:
```python
LLM(speculative_config={"method": "suffix", "num_speculative_tokens": 32},
    compilation_config={"cudagraph_mode": "PIECEWISE"},
    gpu_memory_utilization=0.80, ...)
```

상세 = [`/spec_decoding/README.md`](../../../../spec_decoding/README.md) (production guide) / [`Best_SpecDecode_10778tps.md`](Best_SpecDecode_10778tps.md) §0.

---

## ★★ 0-historical. 이전 best (2026-05-23) — SUB_047 t3 canonical 3-run

| 항목 | 값 |
|---|---:|
| 3-run avg / min / max | **10,956.5** / 10,931.7 / 10,981.4 |
| variance (range/avg) | **0.454%** |
| wall (500p × 8192) | 366.83 s avg |
| CPU busy / GPU util | 5.557 % / 54.70 % avg |
| vs vanilla 4,679.8 | **+134.12% (2.341×)** ⭐ |

**★ Contribution breakdown** (SUB_073/I001 정정, 2026-05-24): vanilla 4,680 → vLLM built-in spec ON (cap=1) 10,778.6 = **+130.3% (vLLM 영역 코드 변경 0)** → SUB_047 fork patch (cap=8/div_tp=0) 10,956.5 = **+1.65% over default (본 fork ~6 줄 patch)**. **본 작업 fork contribution = +1.65%**, 나머지 130 pp 는 vLLM built-in feature 활성화 효과.

상세 3-run 표 / 동작 원리 / 코드 패치 = [`Best_SpecDecode_10778tps.md`](Best_SpecDecode_10778tps.md)

### 활성화

```python
LLM(speculative_config={
    "method": "ngram",
    "num_speculative_tokens": 7,
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 2,
})
```

```bash
export VLLM_NGRAM_NUM_THREADS_CAP=8     # vLLM 기본 1 → 8
export VLLM_NGRAM_DIVIDE_BY_TP=0        # tp_size 로 나누지 않음 → 8 thread/rank
```

### CLAUDE.md Objective 평가

| 목표 | 평가 |
|---|---|
| GPU 포함 서버 throughput 향상 | ✓ **+134.1%** |
| CPU 활용률 극도로 끌어올리기 | ✗ CPU 5.51 % (vanilla 4.66 % 거의 동일) |
| CPU Idle 허락 안 함 | ✗ idle 94.5 % |

→ throughput ✓, CPU 미달. 다음 path = SUB_045/049 (spec + CPU 결합).

---

## ★ 1. Active SUB (시간순)

| 날짜 | SUB | 영역 | 파일 | 상태 |
|---|---|---|---|---|
| 2026-05-23 | SUB_044 | measurement | [`measurements/sub044_spec_decode_20260523/RESULTS.md`](measurements/sub044_spec_decode_20260523/RESULTS.md) | 완료 (★ 첫 net-positive 10,778 tps) |
| 2026-05-23 | SUB_045 | (CPU + spec multi-workload) | (eval/results 출력 예정) | background 측정 중 |
| 2026-05-23 | SUB_047 (5-way sweep) | measurement | (eval/results/20260523_081619_sub047_ngram_threads/) | 완료 (ngram cap 1→8 patch, 10,949.8 tps) |
| 2026-05-23 | SUB_049 | (CPU LLM + GPU spec) | (eval/results 출력 예정) | 완료 (3-scenario: t1 solo / t2 +Qwen0.5B / t3 +Qwen1.5B) |
| **2026-05-23** | **SUB_047 canonical 3-run** | **measurement** | **[`measurements/sub047_t3_3run_verify_20260523/RESULTS.md`](measurements/sub047_t3_3run_verify_20260523/RESULTS.md)** | **★★★ 현 best 확정 (3-run avg 10,956.5, variance 0.454%)** |
| 2026-05-24 | SUB_065 (B-4 threshold sweep) | measurement | `eval/results/20260524_093106_sub065_thr*/` | **기각** (5-way 모두 baseline 동등/-1.7%) |
| 2026-05-24 | SUB_069 (F1 prompt sorting) | measurement | `eval/results/20260524_102409_sub069_*/` | **완료** (none 10,213 / desc -5.6% / asc +1.1%) |
| 2026-05-24 | SUB_068 (D2/D4 stop+tokenizer parallel) | measurement | `eval/results/20260524_110017_sub068_rayon8/` | **기각** (rayon=8 10,985 vs baseline 10,982 = +0.03% noise) |
| 2026-05-24 | SUB_069 3-run interleaved 재측정 | measurement | `eval/results/20260524_112326_sub069v_*/` | **기각** (asc avg 10,300 vs none avg 10,324 = -0.23%, n=3) |
| 2026-05-24 | SUB_066 (B-2 ngram broadcast) | measurement | `eval/results/20260524_121836_sub066_*/` | **기각** (broadcast 10,832 vs baseline 10,975 = -1.30% / pickle+broadcast overhead 가 duplicate 절감 보다 큼) |
| 2026-05-24 | SUB_067 (C1 speculative precompute) | measurement | `eval/results/20260524_*_sub067_*/` | **기각** (precompute 10,573 vs baseline 10,987 = -3.77% / 최대 회귀, 16MB copy + low hit rate) |
| 2026-05-24 | **SUB_071 (chat/code large 500p × 8192)** | **measurement** | **[`measurements/sub071_workload_large_20260524/RESULTS.md`](measurements/sub071_workload_large_20260524/RESULTS.md)** | **완료 — chat +37.5% / code −23.2% (workload-aware gating 필수성 확정)** |
| 2026-05-24 | **SUB_075 (I003 acceptance rate 직접 측정)** | **measurement** | **[`measurements/sub075_acceptance_20260524/RESULTS.md`](measurements/sub075_acceptance_20260524/RESULTS.md)** | **완료 ★ — sonnet K=3.72/α=38.8% / chat K=6.69/α=81.2% (surprise) / code K=1.10/α=1.4% (예측 일치). 본 doc R/K framework 의 first empirical validation** |
| 2026-05-24 | **SUB_076 (I004 workload classifier PoC)** | **measurement** | **[`measurements/sub076_classifier_20260524/RESULTS.md`](measurements/sub076_classifier_20260524/RESULTS.md)** | **완료 — macro accuracy 1.000 (3-workload × 500 prompt perfect, 본 환경 builder set 의 trivial classification capability)** |
| 2026-05-24 | SUB_077 (I005 vLLM upstream PR) | doc / WebFetch | [`measurements/sub077_pr_draft_20260524/PR_DRAFT.md`](measurements/sub077_pr_draft_20260524/PR_DRAFT.md) | draft 완료 (duplicate check ✓, human review 후 submit) |
| 2026-05-24 | **SUB_074 (I002 SuffixDecoding 측정)** | **measurement** | **[`measurements/sub074_suffix_20260524/RESULTS.md`](measurements/sub074_suffix_20260524/RESULTS.md)** | **완료 ⭐ — code workload K=1.10→7.67 (7×) / tps 5362→7094 (+32%, enforce_eager 모드도). sonnet/chat 는 eager penalty 로 회귀 (~-25%). cuda graph 호환 시 모든 workload 향상 가능성 강함** |
| 2026-05-24 | **SUB_078 (IDE_014 Issue #16258 reproduction — code only)** | **measurement** | **[`measurements/sub078_repro_20260524/RESULTS.md`](measurements/sub078_repro_20260524/RESULTS.md)** | **완료 ⭐ — Qwen2.5-0.5B/1.5B + code → ngram 모두 2.5× 회귀 (-59~-62%). issue #16258 패턴 재현** |
| 2026-05-24 | **SUB_079 (IDE_014 small model sonnet/chat 확장)** | **measurement** | **[`measurements/sub079_small_model_full_20260524/RESULTS.md`](measurements/sub079_small_model_full_20260524/RESULTS.md)** | **완료 ⭐ — Qwen 0.5B/1.5B × {sonnet, chat} 4 cell. sonnet -48~-60%, chat -61~-65%. 6/6 cell 회귀 = small model 영역 workload-universal regression 확정** |
| 2026-05-25 | **SUB_085 ⭐⭐ (Phase 2 unblock — suffix + PIECEWISE)** | **measurement** | **[`measurements/sub085_suffix_piecewise_20260525/RESULTS.md`](measurements/sub085_suffix_piecewise_20260525/RESULTS.md)** | **완료 ⭐⭐ — fundamental incompat 아니었음, cudagraph_mode=PIECEWISE 영역 우회. v2 (gmu=0.80): sonnet 11,590 / chat 3,582 / code 7,990. fair vs SUB_086: +50.3% / +63.8% / +18.9% (code 회귀 완전 mitigation)** |
| 2026-05-25 | **SUB_086 (fair vanilla gmu=0.80 baseline)** | **measurement** | **[`measurements/sub086_vanilla_gmu080_20260525/RESULTS.md`](measurements/sub086_vanilla_gmu080_20260525/RESULTS.md)** | **완료 — sonnet 7,710 (vs historical 4,680 +64.7% — wrapper 영역 noise 추정), chat 2,187 stable, code 6,718 (-3.5%)** |
| 2026-05-25 | **SUB_087 (ngram + PIECEWISE + gmu=0.80 fair baseline)** | **measurement** | **[`measurements/sub087_ngram_piecewise_20260525/RESULTS.md`](measurements/sub087_ngram_piecewise_20260525/RESULTS.md)** | **완료 — all-fair table 확정: sonnet 10,139 (+31.5%) / chat 2,846 (+30.2%) / code 5,327 (−20.7%, 회귀)** |
| 2026-05-25 | **SUB_088 (small model + suffix universal regression)** | **measurement** | **[`measurements/sub088_small_suffix_20260525/RESULTS.md`](measurements/sub088_small_suffix_20260525/RESULTS.md)** | **완료 — Qwen 0.5B/1.5B × {sonnet, chat, code} × suffix = 6/6 cell 영역 -51~-73% 회귀 (suffix 도 small model 영역 못 구원)** |
| 2026-05-25 | **SUB_089 ⭐ (sonnet canonical 3-run variance)** | **measurement** | **[`measurements/sub089_sonnet_3run_20260525/RESULTS.md`](measurements/sub089_sonnet_3run_20260525/RESULTS.md)** | **완료 — avg 11,687.4 / min 11,672 / max 11,695, variance 0.20% (매우 stable). fair canonical +51.6% vs SUB_086 vanilla** |
| 2026-05-25 | **SUB_090 ⭐ (R/K model-size sweep)** | **measurement** | **[`measurements/sub090_model_size_sweep_20260525/RESULTS.md`](measurements/sub090_model_size_sweep_20260525/RESULTS.md)** | **완료 — Qwen 0.5B/1.5B/7B × code × 3 config. boundary 7B↔70B (Qwen 7B ngram -17% 영역 boundary 근접). PIECEWISE 영역 small model ngram 영역 SUB_079 영역 -59% → SUB_090 영역 -30.5% 영역 +28pp 향상** |
| 2026-05-25 | **SUB_091 ⭐⭐ (issue #16258 precise reproduction)** | **measurement** | **[`measurements/sub091_issue16258_precise_20260525/RESULTS.md`](measurements/sub091_issue16258_precise_20260525/RESULTS.md)** | **완료 — opt-125m 2.13× regression (issue 영역 2.12× 정확 일치), starcoder2-3b 2.30×. 5-model cross-validation 영역 hardware-independent 영역 확정** |
| 2026-05-25 | **SUB_092 (router HTTP server PoC)** | **CPU only PoC** | **[`measurements/sub092_router_poc_20260525/RESULTS.md`](measurements/sub092_router_poc_20260525/RESULTS.md)** | **완료 — classifier router 0.26 ms/prompt (150 prompts × 2 model_size). production deploy-ready (vLLM core 변경 없음)** |

---

## 📊 2. Measurements (sonnet workload 기준)

### 2.1 SUB_044~049 (초기 active SUB)

| 날짜 | 디렉토리 | 측정 | 결과 |
|---|---|---|---:|
| 2026-05-23 | [`measurements/sub044_spec_decode_20260523/`](measurements/sub044_spec_decode_20260523/) | ngram spec=3/5/7/10 sweep | **t3 spec=7 = 10,778.6 tps (+130%)** ⭐ 첫 net-positive |
| 2026-05-23 | [`measurements/sub047_t3_3run_verify_20260523/`](measurements/sub047_t3_3run_verify_20260523/) | SUB_047 t3 (cap=8 + div_tp=0) canonical 3-run | **★★★ avg 10,956.5 / min 10,931.7 / max 10,981.4 (var 0.454%) — 현 best** |
| 2026-05-23 | `eval/results/20260523_073053_sub045_spec_multiworkload/` | SUB_045 3-scenario (spec solo / +CPU BG / vanilla+BG) | t1=10,749 / t2=10,562 (CPU 29.4%) / t3=4,680 |
| 2026-05-23 | `eval/results/20260523_102915_sub049_cpu_llm_combo/` | SUB_049 3-scenario (solo / +Qwen0.5B / +Qwen1.5B NUMA1) | t1=10,973 / t2=10,580 (CPU 27.6%) / t3=10,745 (CPU 26.4%) |

### 2.2 SUB_050~064 단독 측정 (2026-05-23)

| SUB | 디렉토리 | main tps | vs SUB_047 | CPU% | GPU% | 비고 |
|---|---|---:|---:|---:|---:|---|
| SUB_054 (BGE emb b=32) | `eval/results/20260523_182152_sub054_cpu_embedder/` | 10,834.2 | -1.1% | 19.70 | 41.0 | embedder ~36 sps |
| **SUB_054 (b=64)** | `eval/results/20260523_194409_phase2_sub054_batch64/` | **10,848.1** | **-1.0%** | **21.21** | 41.1 | **production sweet spot, embedder 36.7 sps** |
| SUB_054 (b=128) | `eval/results/20260523_194409_phase2_sub054_batch128/` | 10,616.3 | -3.1% | 21.35 | 41.8 | embedder 42.2 sps (best emb throughput) |
| SUB_055 (BGE reranker) | `eval/results/20260523_183915_sub055_cpu_reranker/` | 10,555.5 | -3.7% | 21.23 | 43.2 | reranker 44 pps |
| SUB_060 (NUMA + KMP affinity) | `eval/results/20260523_180427_sub060_numa_hugepages/` | 10,268.0 | **-6.3%** | 24.92 | 49.9 | 회귀 — KMP_AFFINITY 영역 vLLM conflict 추정 |
| SUB_061 (isolcpus + cgroup) | — | infeasible | — | — | — | container 영역 host cgroup partition root 필요 |

### 2.3 Phase 1/3 combo 측정 — 모두 회귀

| Phase | 디렉토리 | combo | main tps | vs SUB_047 | CPU% |
|---|---|---|---:|---:|---:|
| Phase 1 (3-way) | `eval/results/20260523_192729_phase1_combo/` | Qwen 16t + emb 20t + rerank 20t (NUMA1 split) | 9,635.4 | **-12.1%** | 23.85 |
| Phase 3 combo A | `eval/results/20260523_220228_phase3_combo_A_qwen_emb/` | Qwen 28t + BGE emb b=64 28t (NUMA1) | 10,268.7 | -6.3% | 23.68 |
| Phase 3 combo B | `eval/results/20260523_220228_phase3_combo_B_emb_rerank/` | BGE emb b=64 28t + BGE rerank 28t (NUMA1) | 9,598.2 | **-12.4%** | 24.01 |

**결론**: 모든 multi-SUB combo 회귀 — NUMA bandwidth contention. **단일 SUB 영역 best Objective trade-off** (SUB_054 b=64 = -1.0% / CPU +15.7pp).

---

## 📚 3. Planning

| 파일 | 의미 |
|---|---|
| [`planning/SUB_046_to_049_cpu_spec_plans.md`](planning/SUB_046_to_049_cpu_spec_plans.md) | Tier 1 A/B/C + Tier 3 E CPU+spec 결합 plan (SUB_046~049 의 원래 plan) |
| **[`planning/SUB_050_to_064_objective_levers.md`](planning/SUB_050_to_064_objective_levers.md)** | **★ Objective 정합 lever 탐색 (SUB_050~064, 15 SUB, 5 카테고리)** |

### SUB_065~071 개별 plan (2026-05-24 신설)

| SUB | Lever | bottleneck / 목적 | 상세 plan |
|---|---|---|---|
| **SUB_065** | num_tokens_threshold 영역 (B-4) | small batch self-imposed barrier | [`planning/SUB_065_ngram_threshold_lower.md`](planning/SUB_065_ngram_threshold_lower.md) |
| SUB_066 | ngram broadcast from rank 0 (B-2) | TP rank 7x duplicate | [`planning/SUB_066_ngram_broadcast.md`](planning/SUB_066_ngram_broadcast.md) |
| **SUB_067** | speculative ngram precompute (C1) | inter-step sequential barrier | [`planning/SUB_067_speculative_ngram_precompute.md`](planning/SUB_067_speculative_ngram_precompute.md) |
| SUB_068 | stop-string + tokenizer parallel (D2/D4) | output processing idle | [`planning/SUB_068_stop_tokenizer_parallel.md`](planning/SUB_068_stop_tokenizer_parallel.md) |
| SUB_069 | prompt sorting by length (F1) | batch density / cache hit | [`planning/SUB_069_prompt_sorting.md`](planning/SUB_069_prompt_sorting.md) |
| **SUB_071** | **chat/code workload large-scale validation (500p × 8192)** | workload generalization (post-plateau) | [`planning/SUB_071_workload_large_chatcode.md`](planning/SUB_071_workload_large_chatcode.md) |

### SUB_050~064 개별 plan (2026-05-23 신설)

| 카테고리 | SUB | Lever | 상세 plan |
|---|---|---|---|
| A. Advanced spec decode | SUB_050 | Eagle/Eagle2 CPU draft head | [`planning/SUB_050_eagle_cpu_draft.md`](planning/SUB_050_eagle_cpu_draft.md) |
| A. | SUB_051 | Medusa multiple draft heads (CPU) | [`planning/SUB_051_medusa_cpu.md`](planning/SUB_051_medusa_cpu.md) |
| A. | SUB_052 | Lookahead Decoding (CPU Jacobi) | [`planning/SUB_052_lookahead_decoding.md`](planning/SUB_052_lookahead_decoding.md) |
| A. | SUB_053 | SpecInfer tree spec decode | [`planning/SUB_053_specinfer_tree.md`](planning/SUB_053_specinfer_tree.md) |
| B. Multi-instance CPU pipeline | SUB_054 | CPU embedding model preprocessor | [`planning/SUB_054_cpu_embedding_preprocessor.md`](planning/SUB_054_cpu_embedding_preprocessor.md) |
| B. | SUB_055 | CPU re-ranker / safety filter | [`planning/SUB_055_cpu_reranker_safety.md`](planning/SUB_055_cpu_reranker_safety.md) |
| B. | SUB_056 | CPU prefill offload for long prompts | [`planning/SUB_056_cpu_prefill_offload.md`](planning/SUB_056_cpu_prefill_offload.md) |
| C. vLLM 내부 CPU lever | SUB_057 | ngram tree expansion (multi-chain) | [`planning/SUB_057_ngram_tree_expansion.md`](planning/SUB_057_ngram_tree_expansion.md) |
| C. | SUB_058 | CPU radix-tree prefix KV cache | [`planning/SUB_058_cpu_radix_prefix_cache.md`](planning/SUB_058_cpu_radix_prefix_cache.md) |
| C. | SUB_059 | CPU tokenizer / stop-string parallel | [`planning/SUB_059_cpu_tokenizer_parallel.md`](planning/SUB_059_cpu_tokenizer_parallel.md) |
| D. HPC classic | SUB_060 | NUMA + hugepages + cache prefetch | [`planning/SUB_060_numa_hugepages_tuning.md`](planning/SUB_060_numa_hugepages_tuning.md) |
| D. | **SUB_061** | **Isolcpus + cgroup v2 cpuset (★ CPU 70-90% target)** | [`planning/SUB_061_isolcpus_cgroup.md`](planning/SUB_061_isolcpus_cgroup.md) |
| D. | SUB_062 | GPU Direct + lockfree queue | [`planning/SUB_062_gpu_direct_lockfree.md`](planning/SUB_062_gpu_direct_lockfree.md) |
| E. Scheduling | SUB_063 | CPU-load aware request scheduler | [`planning/SUB_063_cpu_load_scheduler.md`](planning/SUB_063_cpu_load_scheduler.md) |
| E. | SUB_064 | Dynamic CPU/GPU workload migration | [`planning/SUB_064_dynamic_cpu_gpu_migration.md`](planning/SUB_064_dynamic_cpu_gpu_migration.md) |

---

## 🔧 4. 작업 진행 (next steps, 2026-05-24 기준 갱신)

### 4.0 Rec 1/2/3 측정 완료 (2026-05-24)

#### Rec 1: workload generalization (medium 200p × 4096 × 4096)

| workload | vanilla | spec7+cap8 | speedup |
|---|---:|---:|---:|
| sonnet | 8,395.2 | 9,370.1 | **1.12x (+12%)** |
| chat | 2,113.6 | 2,577.1 | **1.22x (+22%)** |
| **code** | **7,889.1** | **5,505.6** | **0.70x (-30% 회귀!)** |

→ **SUB_047 +134% 는 large workload (500p × 8192) 특화**. medium scale 에서 sonnet/chat 은 +12~22%, **code workload 는 -30% 회귀** (ngram acceptance 매우 낮음).
→ small workload (200p × 1024) 에서는 sonnet/chat/code 모두 spec 가 0.67~0.73x (회귀).

#### Rec 1 + SUB_071: workload generalization (large 500p × 8192 × 8192, 2026-05-24)

| workload | vanilla | spec7+cap8 | speedup | source |
|---|---:|---:|---:|---|
| sonnet | 4,679.8 | **10,956.5** | **+134.1%** ⭐ | SUB_047 canonical 3-run |
| **chat** | **2,186.0** | **3,006.6** | **+37.5%** | **SUB_071** |
| **code** | **6,964.5** | **5,346.8** | **−23.2% 회귀** | **SUB_071** |

→ large scale 에서도 ranking 동일: sonnet ≫ chat > 0 > code. chat 은 medium (+22%) → large (+37.5%) 로 개선 (긴 prompt 의 ngram pool 효과). code 는 large 에서도 net regression (out_tok 3.9M / 500 = ~7,830 tok/prompt, EOS 없이 max 까지 생성 → spec overhead 누적).

#### Rec 2: Eagle GPU smoke (200p × 4096 × 4096, sonnet)

| config | tps | vs ngram (9,370) |
|---|---:|---:|
| Eagle (Llama-3 head + 3.3 base, num_spec=5) | 3,209.5 | **-65.7%** |

→ Eagle Llama-3 head 가 Llama-3.3 base 와 호환 낮음. acceptance rate 매우 낮음 추정. SUB_050 결론: **같은 model 의 전용 Eagle head 없으면 효과 없음**. yuhuili/EAGLE-LLaMA3.3 ckpt 미존재 확인.

#### Rec 3: SUB_055 deeper (Llama-Guard-3-1B + BGE-reranker, NUMA1 split 28+28)

| 항목 | 결과 |
|---|---|
| main_tps | 10,844.5 (-1.0% vs SUB_047 best) |
| LlamaGuard | **FAILED** — gated repo, HF 인증 없음 (meta-llama/Llama-Guard-3-1B access restricted) |
| BGE-reranker | 36.4 pps sustained (28-core NUMA1) |

→ LlamaGuard 영역 access 영역 — Meta gated model. 실효: BGE-reranker 단독 (28-core split). 결과 SUB_055 단독 (56-core, -3.7%) 과 비교 시 28-core split 가 main throughput 영역 덜 영향 (-1.0%).

### 4.1 종합 결론 (Rec 1/2/3 + SUB_071 통합)

1. **SUB_047 best 는 workload-shape 의존** — large+repetitive sonnet 만 +134%, chat large +37.5%, **code large −23.2% 회귀** (SUB_071 fact). production 적용 시 workload-aware 활성화 필수 (예: code workload 검출 시 spec OFF).
2. **Eagle 경로는 model-matched ckpt 필수** — Llama-3.3 전용 head 없으면 의미 없음. self-train 1-2주 effort 외 path 없음.
3. **CPU activation 단독 instance pattern 이 best** — multi-process combo 는 NUMA bandwidth contention. SUB_054 batch=64 (단독, -1.0% / CPU +15.7pp) 가 현 best dual-axis.
4. **LlamaGuard 경로 차단** — Meta gated, HF 인증 필요. alternative (BGE-reranker 만 또는 다른 open safety model) 사용 가능.

### 4.2 새 후보 (Rec 1/2/3 분석 후)

| 우선순위 | 작업 | 사유 |
|---|---|---|
| ★★ | workload-aware spec decode gating | code workload 자동 검출 + spec OFF 영역 라우팅 |
| ★ | SUB_057 chain 0 정렬 fix + 측정 | top-M tie-break (latest position) refinement |
| ⚪ | 종결 + production 권장 doc | 현 결과 충분, 추가 lever returns diminishing |

### 4.3 ★ Bottleneck-driven SUB_065~069 (2026-05-24 신설, 진행 중)

SUB_047 step pipeline 의 sequential barrier + idle CPU time 활용 lever. 모두 500p × 8192 × 8192 1-run 검증, positive 확정 시에만 3-run 재측정 정책.

| 우선순위 | SUB | 작업 | bottleneck | effort | 상태 | 결과 (vs baseline) |
|---|---|---|---|:-:|---|---|
| ★★★ | **SUB_065** | num_tokens_threshold sweep (8192/4096/2048/1024/0) | B-4 small-batch self-barrier | 1 시간 | **✅ 기각** (2026-05-24) | thr=8192 10,982 / 4096 -0.17% / 2048 -1.69% / 1024 -0.07% / 0 -0.13% — 가설 기각 |
| ★★★ | **SUB_067** | speculative ngram precompute (ThreadPoolExecutor + per-request suffix cache + chain[0][0] 가정) | B-1 inter-step barrier | 2-3 일 → PoC 1h | **✅ 기각** (2026-05-24) | baseline 10,987 / precompute 10,573 = **-3.77%** (최대 회귀) — 16MB token_ids copy + low hit rate + background numba overhead |
| ★★ | **SUB_066** | ngram broadcast from rank 0 (`broadcast_object` cpu_group) | B-2 TP duplicate | 1-2 일 | **✅ 기각** (2026-05-24) | baseline 10,975 / broadcast 10,832 = **-1.30%** — CPU duplicate -1.21pp 절감했으나 pickle+broadcast overhead 가 더 큼 |
| ★★ | **SUB_068** | stop-string + tokenizer parallel (RAYON=8 + TOKENIZERS_PARALLELISM=true) | D2/D4 output idle | 1 일 | **✅ 기각** (2026-05-24) | rayon=8 10,985 vs baseline 10,982 = **+0.03% noise** — output processing critical path 아님 |
| ★ | **SUB_069** | prompt sorting by length (none/desc/asc) | F1 batch density | 0.5 일 | **✅ 기각** (2026-05-24, 3-run 재측정) | 1-run: none 10,213 / desc -5.56% / asc +1.10% → 3-run interleaved: none avg 10,323.9 / asc avg 10,299.7 = **-0.23%** (가설 기각) |

**최종 결론** (2026-05-24): 5 SUB **모두 기각**.

| SUB | bottleneck | Δ vs baseline |
|---|---|---:|
| SUB_065 | B-4 small-batch barrier (5-way threshold) | -0.07 ~ -1.69% |
| SUB_066 | B-2 ngram broadcast | -1.30% |
| SUB_067 | C1 speculative precompute | **-3.77%** (최대 회귀) |
| SUB_068 | D2/D4 stop+tokenizer parallel | +0.03% noise |
| SUB_069 | F1 prompt sorting (3-run) | -0.23% |

**SUB_047 (10,956.5 tps, +134.1%) 가 현 ngram-spec lever 의 plateau 확정**. 추가 throughput 영역 ngram-spec 외부 lever (Eagle CPU draft, CPU pipeline 결합, workload-aware gating 등) 에서 찾아야 함.

### 4.3 폐기 / 보류 (현 시점)

| SUB | 사유 |
|---|---|
| SUB_061 | container env 영역 infeasible |
| SUB_060 | 측정 결과 회귀 (-6.3%) — KMP_AFFINITY 영역 영역 영향 |
| Phase 1/3 multi-SUB combo | 모두 contention 영역 회귀 (-6~-12%) — 단일 SUB 사용 권장 |
| SUB_051, 053, 056, 058, 062, 064 | large effort 또는 large risk — 본 critical gate 후 재평가 |
| SUB_052 GPU Jacobi integration | gpu_model_runner.py 영역 deep change (2-3 일) — 본 critical gate 후 |

### 4.4 권장 진입 sequence

1. **workload generalization 검증** (1-2 일) — ★ critical gate
2. SUB_050 Eagle CPU (ckpt 가용 확인 후, 가용 시) — 3-5 일
3. SUB_055 + LlamaGuard (Eagle 불가 시 alternative) — 1-2 일
2. **SUB_060** (NUMA/hugepages) — 1-2 시간, +5-10% CPU
3. **SUB_054** (CPU embedding preprocessor) — 반나절, CPU 30-50%
4. 위 3 lever 결합 측정 → CPU 60-80% target

→ 후속: SUB_050 (Eagle CPU) 본격 진입 결정


---

## 📂 5. 디렉토리 구조

```
TSK_020/
├── README.md                              ← Best Configuration index (현 best + history + 동작 원리 + 다음 path)
├── INDEX.md                               ← 본 파일 (navigation hub)
├── Best_SpecDecode_10778tps.md            ← Best 상세 fact + 설정 + mechanism + 코드 변경
├── measurements/
│   ├── sub044_spec_decode_20260523/       ← 첫 net-positive (spec=3/5/7/10 sweep)
│   └── sub047_t3_3run_verify_20260523/    ← ★★★ 현 best 확정 (3-run avg 10,956.5)
├── planning/
│   ├── SUB_046_to_049_cpu_spec_plans.md   ← Tier 1/3 CPU+spec 결합 plan
│   ├── SUB_050_to_064_objective_levers.md ← ★ Objective 정합 lever master plan (5 카테고리 × 15 SUB)
│   ├── SUB_050~SUB_064 *.md (15 개)        ← 각 SUB 개별 상세 plan
│   └── SUB_065~SUB_069 *.md (5 개, 2026-05-24) ← ★ bottleneck-driven SUB plan
└── analysis/
│   └── workload_acceptance_analysis_20260524.md  ← ★ sonnet/chat/code spec decode 정량 분석 (K 역산 + prompt 구조)
└── idea/
    ├── README.md                                  ← ★ SUB_072 idea backlog (6 idea: I001~I006)
    ├── I001_vanilla_contribution_framing.md       ← vanilla +134% framing 의 vLLM built-in / fork patch 분리 (★★★)
    ├── I002_suffix_decoding_measurement.md        ← SuffixDecoding 측정 candidate (★★)
    ├── I003_acceptance_rate_direct_measure.md     ← acceptance rate 직접 측정 (★★)
    ├── I004_workload_aware_gating_poc.md          ← workload-aware predictive gating PoC (★)
    ├── I005_vllm_upstream_pr.md                   ← SUB_047 patch 의 vLLM upstream PR (★)
    └── I006_issue_16258_repro.md                  ← vLLM Issue #16258 reproduction (◐)
```

---

## 📜 6. 관련 doc

| 파일 | 의미 |
|---|---|
| [`../TSK_020.md`](../TSK_020.md) | 본 task 의 정식 doc (sub-task 영역 + 상세 history) |
| [`../../../id_registry.md`](../../../id_registry.md) | TSK_020 + SUB_044~071 entry |
| **[`analysis/workload_acceptance_analysis_20260524.md`](analysis/workload_acceptance_analysis_20260524.md)** | **★ sonnet/chat/code 의 spec decode 효과 정량 분석 (R/K 모델 + Leviathan closed-form alignment + prompt 구조 + workload-aware gating heuristic). §10 = 40 reference (Leviathan/Chen/PLD/EAGLE/Medusa/REST/Lookahead/SuffixDecoding + vLLM PR #24986/#12193/#15151 + issue #16258/#19254 + Spec-Bench/Cascade/Nightjar/DSDE/AdaSpec 등 핵심 산학 자료). §11 = SUB_047 구현이 literature 8개 (PLD/Leviathan/Chen/Medusa/EAGLE/REST/SpecInfer/Lookahead) 와 어떤 axis 에서 다른지 정리 + 본 framework 차별점 + 정직한 contribution 위치** |
