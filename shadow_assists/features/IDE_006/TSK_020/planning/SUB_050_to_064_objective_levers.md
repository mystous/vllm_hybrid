# SUB_050~064 — Objective 정합 lever 탐색 (2026-05-23 신설)

> **parent**: TSK_020 — Spec decode tuning + CPU+spec 결합
> **motivation**: 현 Best (SUB_047 canonical 3-run avg 10,956.5 tps, +134.12%) 가 throughput 목표는 충족하지만 CPU 활용 5.51% (idle 94.49%) — CLAUDE.md `# Objective` 의 "CPU 활용률 극도로 끌어올리기" 조건 미달. 본 plan 은 SUB_050~SUB_064 (15 SUB) 를 5 카테고리로 정리하여 throughput 유지 + CPU 50%+ saturate 영역 lever 후보를 등록한다.
> **각 SUB 상세 plan**: 본 doc § 카테고리 표 의 plan link 참조 (개별 doc 별도 적재).

---

## 0. 현 상태 + Objective gap

| 항목 | SUB_047 (Best) | SUB_049 t3 (CPU lever base) | Objective 요구 |
|---|---:|---:|---|
| throughput vs vanilla | **+134.12%** ✓ | +129.6% | 향상 |
| CPU util | **5.51%** ❌ | 26.41% | 극대화 (idle 금지) |
| GPU util | 54.70% | 55.0% | (불요건) |

→ Best 의 +134% throughput 유지하면서 CPU 60~90% saturate 영역으로 끌어올리는 lever 군 필요.

---

## 1. 카테고리 A — Advanced speculative decoding (CPU draft 기반)

본 카테고리의 hypothesis: draft model 의 inference 자체를 CPU 에서 돌리면 GPU 는 verify 만, CPU 는 draft 로 active.

| SUB | Lever | 출처 | Effort | CPU% target | 상세 plan |
|---|---|---|:-:|---:|---|
| **SUB_050** | Eagle/Eagle2 CPU draft head | arXiv [2401.15077](https://arxiv.org/abs/2401.15077) (Eagle1), [2406.16858](https://arxiv.org/abs/2406.16858) (Eagle2), GitHub `SafeAILab/EAGLE` | 3-5 일 | 40-60% | [SUB_050_eagle_cpu_draft.md](SUB_050_eagle_cpu_draft.md) |
| **SUB_051** | Medusa multiple draft heads (CPU) | arXiv [2401.10774](https://arxiv.org/abs/2401.10774), GitHub `FasterDecoding/Medusa` | 3-5 일 | 30-50% | [SUB_051_medusa_cpu.md](SUB_051_medusa_cpu.md) |
| **SUB_052** | Lookahead Decoding (CPU Jacobi) | arXiv [2402.02057](https://arxiv.org/abs/2402.02057), GitHub `hao-ai-lab/LookaheadDecoding` | 2-3 일 | 40-60% | [SUB_052_lookahead_decoding.md](SUB_052_lookahead_decoding.md) |
| **SUB_053** | SpecInfer tree spec decode | arXiv [2305.09781](https://arxiv.org/abs/2305.09781), GitHub `flexflow/FlexFlow` | 1-2 주 | 35-50% | [SUB_053_specinfer_tree.md](SUB_053_specinfer_tree.md) |

---

## 2. 카테고리 B — Multi-instance CPU pipeline (SUB_049 확장)

본 카테고리의 hypothesis: 별도 CPU process 가 embedding / re-rank / preprocess 등 LLM-인접 작업을 항시 수행 → CPU saturate + 시스템 throughput ↑.

| SUB | Lever | 출처 | Effort | CPU% target | 상세 plan |
|---|---|---|:-:|---:|---|
| **SUB_054** | CPU embedding model preprocessor (BGE/E5/MiniLM) | BGE arXiv [2309.07597](https://arxiv.org/abs/2309.07597), E5 arXiv [2212.03533](https://arxiv.org/abs/2212.03533), GitHub `FlagOpen/FlagEmbedding` | 1-2 일 | 30-50% | [SUB_054_cpu_embedding_preprocessor.md](SUB_054_cpu_embedding_preprocessor.md) |
| **SUB_055** | CPU re-ranker (BGE-reranker) + safety filter (LlamaGuard) | BGE-reranker arXiv [2402.03216](https://arxiv.org/abs/2402.03216), LlamaGuard arXiv [2312.06674](https://arxiv.org/abs/2312.06674) | 1-2 일 | 25-45% | [SUB_055_cpu_reranker_safety.md](SUB_055_cpu_reranker_safety.md) |
| **SUB_056** | CPU prefill offload for long prompts | PowerInfer arXiv [2312.12456](https://arxiv.org/abs/2312.12456), LLM-in-a-Flash arXiv [2312.11514](https://arxiv.org/abs/2312.11514), Splitwise arXiv [2311.18677](https://arxiv.org/abs/2311.18677) | 1-2 주 | 30-50% | [SUB_056_cpu_prefill_offload.md](SUB_056_cpu_prefill_offload.md) |

---

## 3. 카테고리 C — vLLM 내부 CPU lever (SUB_047 확장)

본 카테고리의 hypothesis: ngram lookup / prefix cache / tokenizer 등 vLLM 내부 CPU path 를 확장하여 CPU 활용 추가 + spec decode throughput 추가 향상.

| SUB | Lever | 출처 | Effort | CPU% target | 상세 plan |
|---|---|---|:-:|---:|---|
| **SUB_057** | ngram tree expansion (multi-chain candidates) | SpecInfer tree idea + Eagle2 dynamic tree | 2-3 일 | 15-25% | [SUB_057_ngram_tree_expansion.md](SUB_057_ngram_tree_expansion.md) |
| **SUB_058** | CPU radix-tree prefix KV cache (SGLang style) | SGLang arXiv [2312.07104](https://arxiv.org/abs/2312.07104), GitHub `sgl-project/sglang` | 1-2 주 | 20-40% | [SUB_058_cpu_radix_prefix_cache.md](SUB_058_cpu_radix_prefix_cache.md) |
| **SUB_059** | CPU tokenizer / stop-string parallel | GitHub `huggingface/tokenizers` (Rust multi-thread) | 1 일 | 5-15% | [SUB_059_cpu_tokenizer_parallel.md](SUB_059_cpu_tokenizer_parallel.md) |

---

## 4. 카테고리 D — HPC classic 기법

본 카테고리의 hypothesis: 기존 운영 환경 (kernel, libc, NUMA, hugepages, CPU pinning) 의 classic HPC 튜닝으로 SUB_049 기반 CPU 활용 추가 끌어올림.

| SUB | Lever | 출처 | Effort | CPU% target | 상세 plan |
|---|---|---|:-:|---:|---|
| **SUB_060** | NUMA + hugepages + cache prefetch 튜닝 | Linux numactl(8), THP docs, Intel TBB pinning | 1-2 일 | 30-40% | [SUB_060_numa_hugepages_tuning.md](SUB_060_numa_hugepages_tuning.md) |
| **SUB_061** | Isolcpus + cgroup v2 cpuset 분리 | Linux kernel isolcpus, cgroup v2 cpuset.cpus.partition=isolated | 1 일 | **70-90% saturate** | [SUB_061_isolcpus_cgroup.md](SUB_061_isolcpus_cgroup.md) |
| **SUB_062** | GPU Direct + lockfree queue CPU/GPU | NVIDIA GPU Direct, Boost.Lockfree, folly ProducerConsumerQueue | 3-5 일 | latency 영역 | [SUB_062_gpu_direct_lockfree.md](SUB_062_gpu_direct_lockfree.md) |

---

## 5. 카테고리 E — Scheduling / workload-side

본 카테고리의 hypothesis: scheduler / workload routing 에서 CPU 자원 인지로 dynamic 라우팅 → CPU idle 시 work steal.

| SUB | Lever | 출처 | Effort | CPU% target | 상세 plan |
|---|---|---|:-:|---:|---|
| **SUB_063** | CPU-load aware request scheduler | Sarathi-Serve arXiv [2403.02310](https://arxiv.org/abs/2403.02310), DistServe arXiv [2401.09670](https://arxiv.org/abs/2401.09670) | 2-3 일 | 30-50% | [SUB_063_cpu_load_scheduler.md](SUB_063_cpu_load_scheduler.md) |
| **SUB_064** | Dynamic CPU/GPU workload migration | LoongServe arXiv [2404.09526](https://arxiv.org/abs/2404.09526), Splitwise | 1-2 주 | (다른 lever 결합) | [SUB_064_dynamic_cpu_gpu_migration.md](SUB_064_dynamic_cpu_gpu_migration.md) |

---

## 6. 우선순위 sequencing (effort × Objective 정합)

| 순위 | SUB | 카테고리 | 이유 |
|---|---|---|---|
| **★★★** | **SUB_061** | D | small effort (1일) + saturate 가능 (CPU LLM 70-90% target). 즉시 검증 |
| **★★★** | **SUB_060** | D | small effort (1-2일) + SUB_049 직접 확장 (26%→35%+) |
| **★★★** | **SUB_054** | B | small effort (1-2일) + 별도 instance 위험 없음 + CPU 30-50% |
| ★★ | SUB_050 | A | medium-large 지만 spec decode + CPU 양 축 모두 충족 가능성 |
| ★★ | SUB_059 | C | small (1일) + 즉시 측정 가능 |
| ★★ | SUB_052 | A | medium (2-3일) + 새 lever (Lookahead) |
| ★ | SUB_055 | B | small (1-2일) + complementary |
| ★ | SUB_057 | C | medium (2-3일) + SUB_047 의 자연 확장 |
| ★ | SUB_063 | E | medium (2-3일) |
| ⚪ | SUB_051 / SUB_058 / SUB_056 / SUB_062 / SUB_053 / SUB_064 | A/B/C/D/E | large effort 또는 negative ROI 가능성 — 위 high-priority 결과 본 후 재평가 |

---

## 7. 권장 진입 sequence (다음 turn)

| step | SUB | 작업 | 예상 시간 | target |
|---|---|---|:-:|---|
| 1 | **SUB_061** | isolcpus + cgroup 으로 CPU LLM dedicated 56 cores | 1-2 시간 | CPU LLM 영역 70%+ |
| 2 | **SUB_060** | SUB_049 위에 NUMA / hugepages 환경 변수 + numactl | 1-2 시간 | +5-10% CPU |
| 3 | **SUB_054** | BGE-large-en-v1.5 wrapper script + 항시 실행 | 반나절 | CPU 30-50% |
| 4 | (측정) | 위 3 lever 결합 측정 (Best + SUB_049 base + step 1~3) | 30 분 | **CPU 60-80% target** |

→ 위 step 결과 후 SUB_050 (Eagle CPU) 본격 진입 결정.

---

## 8. 다음 path (장기)

- SUB_050 Eagle CPU draft head — sonnet 외 일반 chat / code workload 정합 검증 동시
- SUB_058 SGLang radix prefix cache port — repeated prompt workload 효과 큼
- SUB_063 CPU-load aware scheduler — 위 모든 lever 의 dynamic 결합 영역
