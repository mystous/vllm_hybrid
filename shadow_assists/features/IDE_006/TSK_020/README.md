# TSK_020 — Best Configuration index (Spec decode tuning + CPU+spec 결합)

> **부모**: [`IDE_006/README.md`](../README.md) · [`shadow_assists/README.md`](../../../README.md)
> **상태**: 활성 — current best 10,956.6 tps (+134.1% vs vanilla)
> **branch**: `feat/spec-decode-tuning`
> **navigation**: [`INDEX.md`](INDEX.md) — 전체 SUB / measurement / plan single entry
> **task doc**: [`../TSK_020.md`](../TSK_020.md) — task 전체 개요 + sub-task 영역
> **자매 task**: [`../TSK_019/`](../TSK_019/) — NEO 작업 (net-negative 확정, dead path)

---

## ★★★ 현 absolute best — SUB_047 t3 (cap=8 + div_tp=0, spec=7)

**3-run avg 10,956.6 tps (variance 0.125%)**

| 항목 | run1 | run2 | run3 | **avg** | min | max |
|---|---:|---:|---:|---:|---:|---:|
| output_tps | 10,949.8 | 10,963.5 | 10,956.5 | **10,956.6** | 10,949.8 | 10,963.5 |
| wall (s) | 367.1 | 366.6 | 366.8 | 366.83 | 366.6 | 367.1 |
| CPU busy avg (%) | 5.52 | 5.47 | 5.55 | 5.51 | 5.47 | 5.55 |
| GPU util avg (%) | 54.6 | 54.7 | 54.8 | 54.70 | 54.6 | 54.8 |
| vs vanilla 4,679.8 | +134.0% | +134.3% | +134.1% | **+134.1%** | +133.9% | +134.3% |

상세 fact / 설정 / pipeline 도식 / SUB_047 패치 = [`Best_SpecDecode_10778tps.md`](Best_SpecDecode_10778tps.md)

---

## SUB history (시간순)

| 시각 (KST) | SUB | 결과 | 의미 |
|---|---|---:|---|
| 2026-05-23 00:53 | SUB_044 | **10,778.6 tps (+130.3%)** ⭐ | **첫 net-positive** — vanilla + ngram spec=7 (vLLM built-in feature) |
| 2026-05-23 ~02:00 | SUB_045 | (진행 중) | spec=7 + CPU BG multi-workload (CLAUDE.md CPU 활용 목표 검증) |
| 2026-05-23 ~04:00 | SUB_046 | 기각 | NEO + spec=7 결합 시도 → schedule path conflict crash |
| 2026-05-23 08:16 | SUB_047 (5-way sweep) | 10,949.8 tps (+134%) ⭐ | ngram numba thread cap env-tunable patch (1→8) |
| 2026-05-23 ~09:00 | SUB_048 | 중단 | spec sampling CPU offload → 본질이 GPU lever 라 사용자 중단 |
| 2026-05-23 ~10:30 | SUB_049 | (진행 중) | 메인 vLLM (spec=7+cap=8) + 별도 CPU LLM (Qwen 0.5B/1.5B NUMA1) 동시 |
| **2026-05-23 13:39** | **SUB_047 3-run 검증** | **avg 10,956.6 (min 10,949.8 / max 10,963.5, var 0.125%)** ⭐ | **현 best 확정** |

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
| NEO 코드 사용 | **0 %** (vanilla path) |
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

## NEO (TSK_019) vs spec decode (TSK_020) 비교

| Approach | tps | wall (s) | vs vanilla | task |
|---|---:|---:|---:|---|
| vanilla baseline | 4,680 | 875 | — | (baseline) |
| NEO best (S1-S9, gmu=0.92) | 2,238 | 1,819 | **-52.2%** ⚠️ | TSK_019 (dead path 확정) |
| NEO multi-workload (NEO+BG) | 1,652 | — | **-13% (vs vanilla+BG)** | TSK_019 SUB_041 |
| **★ vanilla + ngram spec=7 + cap=8** | **10,957** (3-run avg) | **367** | **★ +134.1%** ⭐ | **TSK_020 (현 active)** |
| spec / NEO best ratio | — | — | **spec 가 NEO 의 4.90×** | — |

---

## 다음 path (CPU 활용 추가)

| 우선순위 | SUB | 작업 | Tier | 상태 |
|---|---|---|---|---|
| ★★★ | SUB_045 | spec=7 solo / +CPU BG / vanilla+BG 비교 | Tier 3 F | background 측정 중 |
| ★★★ | SUB_049 | 메인 vLLM (spec=7+cap=8) + 별도 CPU LLM (Qwen 0.5B/1.5B NUMA1) | Tier 3 E (redefined) | background 측정 중 |
| ★ | (workload generalization) | sonnet 외 일반 chat/code workload 영역 acceptance rate 측정 | — | 미시작 |

상세 plan = [`planning/SUB_046_to_049_cpu_spec_plans.md`](planning/SUB_046_to_049_cpu_spec_plans.md).

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
