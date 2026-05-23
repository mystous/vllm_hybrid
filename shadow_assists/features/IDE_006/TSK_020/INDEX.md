# TSK_020 — INDEX (navigation hub)

> **목적**: TSK_020 의 모든 doc / measurement / plan 의 single entry point.
> **상태**: 활성 — current best 10,956.5 tps (+134.1% vs vanilla)
> **branch**: `feat/spec-decode-tuning`
> **상위**: [`README.md`](README.md) — Best Configuration index

---

## ★★★ 0. 현 absolute best (2026-05-23) — SUB_047 t3 canonical 3-run

| 항목 | 값 |
|---|---:|
| 3-run avg / min / max | **10,956.5** / 10,931.7 / 10,981.4 |
| variance (range/avg) | **0.454%** |
| wall (500p × 8192) | 366.83 s avg |
| CPU busy / GPU util | 5.557 % / 54.70 % avg |
| vs vanilla 4,679.8 | **+134.12% (2.341×)** ⭐ |

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

### 4.1 종합 결론 (Rec 1/2/3 통합)

1. **SUB_047 best 는 workload-specific** — large+repetitive workload 외에서는 효과 작음 또는 회귀. production 적용 시 workload-aware 활성화 필요 (예: code workload 검출 시 spec OFF).
2. **Eagle 경로는 model-matched ckpt 필수** — Llama-3.3 전용 head 없으면 의미 없음. self-train 1-2주 effort 외 path 없음.
3. **CPU activation 단독 instance pattern 이 best** — multi-process combo 는 NUMA bandwidth contention. SUB_054 batch=64 (단독, -1.0% / CPU +15.7pp) 가 현 best dual-axis.
4. **LlamaGuard 경로 차단** — Meta gated, HF 인증 필요. alternative (BGE-reranker 만 또는 다른 open safety model) 사용 가능.

### 4.2 새 후보 (Rec 1/2/3 분석 후)

| 우선순위 | 작업 | 사유 |
|---|---|---|
| ★★ | workload-aware spec decode gating | code workload 자동 검출 + spec OFF 영역 라우팅 |
| ★ | SUB_057 chain 0 정렬 fix + 측정 | top-M tie-break (latest position) refinement |
| ⚪ | 종결 + production 권장 doc | 현 결과 충분, 추가 lever returns diminishing |

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
│   └── SUB_050~SUB_064 *.md (15 개)        ← 각 SUB 개별 상세 plan
└── analysis/                              ← (현재 비어 있음, 향후 spec workload 분석 등)
```

---

## 📜 6. 관련 doc

| 파일 | 의미 |
|---|---|
| [`../TSK_020.md`](../TSK_020.md) | 본 task 의 정식 doc (sub-task 영역 + 상세 history) |
| [`../../../id_registry.md`](../../../id_registry.md) | TSK_020 + SUB_044~049 entry |
