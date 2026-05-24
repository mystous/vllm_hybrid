# TSK_020 — Best Configuration index (Spec decode tuning + CPU+spec 결합)

> **부모**: [`IDE_006/README.md`](../README.md) · [`shadow_assists/README.md`](../../../README.md)
> **상태**: 활성 — current best 10,956.5 tps (+134.1% vs vanilla)
> **branch**: `feat/spec-decode-tuning`
> **navigation**: [`INDEX.md`](INDEX.md) — 전체 SUB / measurement / plan single entry
> **task doc**: [`../TSK_020.md`](../TSK_020.md) — task 전체 개요 + sub-task 영역

---

## ★★★ 현 absolute best — SUB_047 t3 (cap=8 + div_tp=0, spec=7)

**canonical 3-run avg 10,956.5 tps (variance 0.454%)**

| run | tps | wall (s) | CPU% | GPU% |
|---|---:|---:|---:|---:|
| 1 | 10,981.4 | 366.0 | 5.51 | 54.6 |
| 2 | 10,931.7 | 367.7 | 5.57 | 54.7 |
| 3 | 10,956.3 | 366.8 | 5.59 | 54.8 |
| **avg** | **10,956.5** | **366.83** | **5.557** | **54.70** |
| min / max | 10,931.7 / 10,981.4 | 366.0 / 367.7 | 5.51 / 5.59 | 54.6 / 54.8 |
| vs vanilla 4,679.8 | **+134.12%** | — | — | — |

variance 0.454% (range/avg). **10,956.5 ± 25 tps** 신뢰 가능.

상세 fact / 설정 / pipeline 도식 / SUB_047 패치 = [`Best_SpecDecode_10778tps.md`](Best_SpecDecode_10778tps.md)

---

## SUB history (시간순)

| 시각 (KST) | SUB | 결과 | 의미 |
|---|---|---:|---|
| 2026-05-23 00:53 | SUB_044 | **10,778.6 tps (+130.3%)** ⭐ | **첫 net-positive** — vanilla + ngram spec=7 (vLLM built-in feature) |
| 2026-05-23 ~02:00 | SUB_045 | 완료 (3-scenario) | t1 spec=10,749 / t2 spec+BG=10,562 (CPU 29%) / t3 vanilla+BG=4,680 |
| 2026-05-23 08:16 | SUB_047 (5-way sweep) | 10,949.8 tps (+134%) ⭐ | ngram numba thread cap env-tunable patch (1→8) |
| 2026-05-23 ~10:30 | SUB_049 | 완료 (3-scenario) | t1 solo=10,973 / t2 +Qwen0.5B=10,580 (CPU 28%) / t3 +Qwen1.5B=10,745 (CPU 26%) |
| **2026-05-23 16:35** | **SUB_047 canonical 3-run** | **avg 10,956.5 / min 10,931.7 / max 10,981.4 (var 0.454%)** ⭐ | **★ 현 best 확정** |
| 2026-05-23 17:04~21:42 | SUB_054/055/060 단독 + Phase 1/3 combo | CPU activation 19~24% / 모든 multi-SUB combo 회귀 | 단독 lever 가 best, NUMA contention 확인 |
| 2026-05-23 22:18 | SUB_050~064 plan 적재 (15 SUB) | 5 카테고리 × 15 plan | Objective 정합 lever 후보 |
| 2026-05-24 07:17 | Rec 1: workload generalization | sonnet +12% / chat +22% / **code -30% 회귀** | SUB_047 +134% 는 large+repetitive 특화 |
| 2026-05-24 08:17 | Rec 2: Eagle GPU smoke | -65.7% (Llama-3 head + 3.3 base 호환 낮음) | SUB_050 dead-end (model-matched ckpt 필요) |
| 2026-05-24 08:35 | Rec 3: BGE-reranker NUMA1 split | -1.0% (LlamaGuard 영역 gated 영역 차단) | HF 인증 필요 |
| **2026-05-24 ~09:30** | **Bottleneck-driven SUB_065~069 신설** | **5 SUB** — B-4 threshold / B-2 broadcast / C1 precompute / D2/D4 parallel / F1 sorting | bottleneck 분석 — sequential barrier 제거 + idle CPU time 활용 lever |
| 2026-05-24 09:31 | SUB_065 (B-4 threshold sweep) | **기각** — thr=8192 10,982 / 4096 -0.17% / 2048 -1.69% / 1024 -0.07% / 0 -0.13% | small-batch self-imposed barrier 가설 기각, 모두 baseline 동등/미세 회귀 |
| 2026-05-24 10:24 | SUB_069 (F1 prompt sorting) | sort=none 10,213 / desc -5.56% / asc +1.10% | desc 회귀 (long prompts KV pool 점유), asc small positive — 3-run 재측정 후보 |
| 2026-05-24 11:00 | SUB_068 (D2/D4 stop+tokenizer parallel) | **기각** — rayon=8 10,985 vs baseline 10,982 = +0.03% noise | output processing 이 critical path 아님 확정 (env-only, 코드 변경 0) |
| 2026-05-24 13:00 | SUB_069 3-run interleaved 재측정 | **기각** — none avg 10,324 / asc avg 10,300 = **-0.23%** (n=3) | 1-run +1.10% 는 baseline noise (SUB_065/068 baseline 10,982 vs SUB_069 baseline 10,213 의 7% drift) — F1 prompt sorting 가설 최종 기각 |
| 2026-05-24 14:00 | SUB_066 (B-2 ngram broadcast, `broadcast_object`) | **기각** — broadcast 10,832 vs baseline 10,975 = **-1.30%** | CPU duplicate -1.21pp 절감되었으나 pickle+cpu_group broadcast overhead 가 더 큼. 5 SUB 중 4 기각 (SUB_067 만 남음, 가장 복잡한 implementation) |

---

## 동작 원리 (요약)

```
prompt → [ngram lookup, CPU 8 thread/rank, sub-ms]  ← SUB_047 cap=8 patch
       → [1 + 7 = 8 token batch] GPU forward 1 회 (TP=8)
       → [rejection sampler accept ~4-5 token/step]
       → output (vanilla 의 ~5× step 압축)
```

| 측면 | 값 |
|---|---|
| 출처 | vLLM built-in `speculative_config` + SUB_047 한 줄짜리 thread cap 패치 (6 줄 코드 변경) |
| draft model | 불필요 (pure prompt-based) |
| 적합 workload | sonnet (어휘 반복 ↑) — 일반 chat/code 영역 acceptance 변화 가능, 검증 필요 |
| KV memory 제약 | num_speculative_tokens=10 은 OOM → 7 이 sweet spot |
| GPU 활용 | 54.7 % (vanilla 73 % 대비 ↓ — forward 횟수 자체 감소) |
| CPU 활용 | 5.5 % (vanilla 4.7 % 거의 동일) |
| throughput | +134.1 % vs vanilla ⭐ |

상세 분석 = [`Best_SpecDecode_10778tps.md §4 동작 원리`](Best_SpecDecode_10778tps.md).

---

## CLAUDE.md `# Objective` 평가

| 목표 | 평가 |
|---|---|
| GPU 포함 서버 전체 throughput 향상 | ✓ **+134.1%** 달성 |
| CPU 활용률 극도로 끌어올리기 | ✗ CPU 5.51 % (vanilla 4.66 % 거의 동일) |
| CPU Idle 허락 안 함 | ✗ idle 94.5 % |

→ throughput 목표 ✓, **CPU 활용 목표 미달**. 다음 path = SUB_045 (spec + CPU BG workload), SUB_049 (CPU LLM 별도 instance 동시).

---

## vs vanilla baseline

| Approach | tps | wall (s) | vs vanilla |
|---|---:|---:|---:|
| vanilla baseline | 4,680 | 875 | — |
| vanilla + ngram spec=7 (SUB_044) | 10,778 | 373 | +130.3% (2.30×) |
| **★ vanilla + ngram spec=7 + cap=8 (SUB_047)** | **10,957** (3-run avg) | **367** | **★ +134.1% (2.341×)** ⭐ |

---

## 다음 path — bottleneck-driven (SUB_065~069)

현 pipeline 분석 결과 GPU forward (70-90ms/step) 가 critical path, CPU 영역 5.51% idle. 진짜 lever 는 idle CPU time 영역 useful work 영역 채우는 것 + sequential barrier 영역 제거.

| 우선순위 | SUB | 작업 | bottleneck | effort |
|---|---|---|---|:-:|
| ★★★ | **SUB_065** | num_tokens_threshold sweep (8192/4096/2048/1024/0) | small batch self-imposed barrier | 1 시간 (★ 진행 중) |
| ★★★ | **SUB_067** | speculative ngram precompute (background thread) | inter-step sequential barrier | 2-3 일 |
| ★★ | **SUB_066** | ngram broadcast from rank 0 | TP rank 7x duplicate work | 1-2 일 |
| ★★ | **SUB_068** | stop-string + tokenizer parallel | output processing idle | 1 일 |
| ★ | **SUB_069** | prompt sorting by length | batch density / ngram cache hit | 0.5 일 |

상세 plan = [`planning/SUB_065_ngram_threshold_lower.md`](planning/SUB_065_ngram_threshold_lower.md) ~ [`SUB_069_prompt_sorting.md`](planning/SUB_069_prompt_sorting.md)

### skip 된 후보 (의미 작음)

- ngram cap > 8 추가 sweep — SUB_047 t5 cap=56 이미 -14% 회귀 확인
- Aho-Corasick / Suffix Array — ngram time 자체 작아서 효과 작음
- 별도 CPU process pool 영역 ngram offload — inter-process latency 영역 step time 회귀 가능

### SUB_050~064 후보 (Rec 1/2/3 검증 후)

| 후보 | 상태 |
|---|---|
| SUB_050 (Eagle CPU) | Llama-3.3 영역 ckpt 미존재 — dead-end (self-train 1-2주 외 path 없음) |
| SUB_055 + LlamaGuard | Meta gated repo, HF 인증 필요 (사용자 token 영역 대기) |
| SUB_058 radix prefix cache | sonnet 영역 cache hit 작음 — workload generalization 영역 |
| SUB_054 batch=64 | dual-axis best (-1.0% / CPU 21%) — production alternative |

상세 = [`Best_SpecDecode_10778tps.md §7`](Best_SpecDecode_10778tps.md), [`INDEX.md §4`](INDEX.md)

---

## raw 자료

| 자료 | 위치 |
|---|---|
| **현 best (SUB_047 3-run)** | **[`measurements/sub047_t3_3run_verify_20260523/RESULTS.md`](measurements/sub047_t3_3run_verify_20260523/RESULTS.md)** |
| SUB_044 base (spec sweep) | [`measurements/sub044_spec_decode_20260523/RESULTS.md`](measurements/sub044_spec_decode_20260523/RESULTS.md) |
| Best doc (mechanism 상세) | [`Best_SpecDecode_10778tps.md`](Best_SpecDecode_10778tps.md) |
| Planning (Tier 1/3 plans) | [`planning/SUB_046_to_049_cpu_spec_plans.md`](planning/SUB_046_to_049_cpu_spec_plans.md) |
| Navigation hub | [`INDEX.md`](INDEX.md) |
| Code change | `vllm/v1/spec_decode/ngram_proposer.py:48` (SUB_047 patch, env-gated) |
| run1 result.json | `eval/results/20260523_081619_sub047_ngram_threads/t3_cap8_div0/result.json` |
| run2 result.json | `eval/results/20260523_133929_sub047_t3_verify/run2_cap8_div0/result.json` |
| run3 result.json | `eval/results/20260523_133929_sub047_t3_verify/run3_cap8_div0/result.json` |
| launcher (5-way sweep) | `/tmp/run_sub047_ngram_threads.sh` |
| launcher (3-run verify) | `/tmp/run_sub047_t3_verify_2runs.sh` |
| wrapper python | `/tmp/run_spec_decode.py` |
