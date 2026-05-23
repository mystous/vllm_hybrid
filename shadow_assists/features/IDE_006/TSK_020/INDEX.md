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

## 📊 2. Measurements

| 날짜 | 디렉토리 | 측정 | 결과 |
|---|---|---|---:|
| 2026-05-23 | [`measurements/sub044_spec_decode_20260523/`](measurements/sub044_spec_decode_20260523/) | ngram spec=3/5/7/10 sweep | **t3 spec=7 = 10,778.6 tps (+130%)** ⭐ 첫 net-positive |
| 2026-05-23 | [`measurements/sub047_t3_3run_verify_20260523/`](measurements/sub047_t3_3run_verify_20260523/) | SUB_047 t3 (cap=8 + div_tp=0) canonical 3-run | **★★★ avg 10,956.5 / min 10,931.7 / max 10,981.4 (var 0.454%) — 현 best** |

추가 raw 결과 (RESULTS.md 미작성, eval/results 만):
- `eval/results/20260523_005314_sub044_spec_decode/` (SUB_044 raw)
- `eval/results/20260523_081619_sub047_ngram_threads/` (SUB_047 5-way sweep raw)
- `eval/results/20260523_133929_sub047_t3_verify/` (SUB_047 3-run raw)
- `eval/results/<TS>_sub045_spec_multiworkload/` (SUB_045, 진행 중)
- `eval/results/<TS>_sub049_cpu_llm_combo/` (SUB_049, 진행 중)

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

## 🔧 4. 작업 진행 (next steps) — SUB_050~064 우선순위

상세 plan: [`planning/SUB_050_to_064_objective_levers.md`](planning/SUB_050_to_064_objective_levers.md)

| 우선순위 | SUB | 작업 | 카테고리 | effort | CPU% target |
|---|---|---|---|:-:|---:|
| **★★★** | **SUB_061** | Isolcpus + cgroup v2 cpuset 분리 | D HPC | 1 일 | **70-90% saturate** |
| **★★★** | **SUB_060** | NUMA + hugepages + cache prefetch | D HPC | 1-2 일 | 30-40% |
| **★★★** | **SUB_054** | CPU embedding model preprocessor | B Multi-inst | 1-2 일 | 30-50% |
| ★★ | SUB_050 | Eagle/Eagle2 CPU draft head | A Adv spec | 3-5 일 | 40-60% |
| ★★ | SUB_059 | CPU tokenizer / stop-string parallel | C vLLM 내부 | 1 일 | 5-15% |
| ★★ | SUB_052 | Lookahead Decoding (CPU Jacobi) | A Adv spec | 2-3 일 | 40-60% |
| ★ | SUB_055 | CPU re-ranker / safety filter | B Multi-inst | 1-2 일 | 25-45% |
| ★ | SUB_057 | ngram tree expansion | C vLLM 내부 | 2-3 일 | 15-25% |
| ★ | SUB_063 | CPU-load aware scheduler | E Scheduling | 2-3 일 | 30-50% (결합) |
| ⚪ | SUB_051 | Medusa CPU | A | 3-5 일 | 30-50% |
| ⚪ | SUB_056 | CPU prefill offload | B | 1-2 주 | 30-50% (위험) |
| ⚪ | SUB_058 | CPU radix prefix cache | C | 1-2 주 | 20-40% |
| ⚪ | SUB_053 | SpecInfer tree | A | 1-2 주 | 35-50% |
| ⚪ | SUB_062 | GPU Direct + lockfree | D | 3-5 일 | marginal |
| ⚪ | SUB_064 | Dynamic CPU/GPU migration | E | 1-2 주 | 결합 |

### 권장 진입 sequence (다음 turn)

1. **SUB_061** (isolcpus + cgroup) — 1-2 시간, CPU LLM 영역 dedicated 70%+
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
│   └── SUB_046_to_049_cpu_spec_plans.md   ← Tier 1/3 CPU+spec 결합 plan
└── analysis/                              ← (현재 비어 있음, 향후 spec workload 분석 등)
```

---

## 📜 6. 관련 doc

| 파일 | 의미 |
|---|---|
| [`../TSK_020.md`](../TSK_020.md) | 본 task 의 정식 doc (sub-task 영역 + 상세 history) |
| [`../../../id_registry.md`](../../../id_registry.md) | TSK_020 + SUB_044~049 entry |
