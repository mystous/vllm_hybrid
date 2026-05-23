# TSK_020 — INDEX (navigation hub)

> **목적**: TSK_020 의 모든 doc / measurement / plan 의 single entry point.
> **상태**: 활성 — current best 10,956.6 tps (+134.1% vs vanilla)
> **branch**: `feat/spec-decode-tuning`
> **상위**: [`README.md`](README.md) — Best Configuration index

---

## ★★★ 0. 현 absolute best (2026-05-23) — SUB_047 t3 3-run verified

| 항목 | 값 |
|---|---:|
| 3-run avg / min / max | **10,956.6** / 10,949.8 / 10,963.5 |
| variance (CV) | **0.125%** |
| wall (500p × 8192) | 366.83 s |
| CPU busy / GPU util | 5.51 % / 54.70 % |
| vs vanilla 4,679.8 | **+134.1% (2.341×)** ⭐ |

상세: [`Best_SpecDecode_10778tps.md`](Best_SpecDecode_10778tps.md)

### 동작 원리 (요약)

| Path | 활성 조건 | 상태 |
|---|---|---|
| A. NEO/AMX (TSK_019) | `LLM(enable_neo_asymmetric=True)` + `VLLM_NEO_*` env | **dead path** — net-negative (-13~62%) |
| **B. vanilla + ngram spec (TSK_020)** | `LLM(speculative_config={"method":"ngram",num_speculative_tokens=7,...})` + `VLLM_NGRAM_NUM_THREADS_CAP=8` + `VLLM_NGRAM_DIVIDE_BY_TP=0` | **★ 현 best (+134.1%)** |

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
| 2026-05-23 | SUB_046 | (NEO+spec 결합 시도) | (eval/results) | 기각 (schedule path conflict) |
| 2026-05-23 | SUB_047 (5-way sweep) | measurement | (eval/results/20260523_081619_sub047_ngram_threads/) | 완료 (ngram cap 1→8 patch, 10,949.8 tps) |
| 2026-05-23 | SUB_048 | (spec sampling CPU) | — | 사용자 중단 |
| 2026-05-23 | SUB_049 | (CPU LLM + GPU spec) | (eval/results 출력 예정) | background 측정 중 |
| **2026-05-23** | **SUB_047 3-run verify** | **measurement** | **[`measurements/sub047_t3_3run_verify_20260523/RESULTS.md`](measurements/sub047_t3_3run_verify_20260523/RESULTS.md)** | **★★★ 현 best 확정 (avg 10,956.6, variance 0.125%)** |

---

## 📊 2. Measurements

| 날짜 | 디렉토리 | 측정 | 결과 |
|---|---|---|---:|
| 2026-05-23 | [`measurements/sub044_spec_decode_20260523/`](measurements/sub044_spec_decode_20260523/) | ngram spec=3/5/7/10 sweep | **t3 spec=7 = 10,778.6 tps (+130%)** ⭐ 첫 net-positive |
| 2026-05-23 | [`measurements/sub047_t3_3run_verify_20260523/`](measurements/sub047_t3_3run_verify_20260523/) | SUB_047 t3 (cap=8 + div_tp=0) 3-run | **★★★ avg 10,956.6 / min 10,949.8 / max 10,963.5 (var 0.125%) — 현 best** |

추가 raw 결과 (RESULTS.md 미작성, eval/results 만):
- `eval/results/20260523_005314_sub044_spec_decode/` (SUB_044 raw)
- `eval/results/20260523_081619_sub047_ngram_threads/` (SUB_047 5-way sweep raw)
- `eval/results/20260523_091142_sub046_neo_spec_combo/` (SUB_046 기각 raw)
- `eval/results/20260523_133929_sub047_t3_verify/` (SUB_047 3-run raw)
- `eval/results/<TS>_sub045_spec_multiworkload/` (SUB_045, 진행 중)
- `eval/results/<TS>_sub049_cpu_llm_combo/` (SUB_049, 진행 중)

---

## 📚 3. Planning

| 파일 | 의미 |
|---|---|
| [`planning/SUB_046_to_049_cpu_spec_plans.md`](planning/SUB_046_to_049_cpu_spec_plans.md) | Tier 1 A/B/C + Tier 3 E CPU+spec 결합 plan (SUB_046~049 의 원래 plan) |

---

## 🔧 4. 작업 진행 영역 (next steps)

| 우선순위 | SUB | 작업 | Tier | effort | 가설 / 비고 |
|---|---|---|---|:-:|---|
| ★★★ (진행 중) | SUB_045 | spec=7 solo / +CPU BG / vanilla+BG 비교 (3-scenario) | Tier 3 F | 측정 중 | spec + CPU BG 동시 부하 → CPU 활용 ↑ 가능성 |
| ★★★ (진행 중) | SUB_049 | 메인 vLLM (spec=7+cap=8) + 별도 CPU LLM (Qwen 0.5B/1.5B NUMA1 56-thread) | Tier 3 E (redefined) | 측정 중 | 진정한 CPU 활용 (CPU LLM active inference) |
| ★ | workload generalization | sonnet 외 일반 chat/code workload 영역 acceptance rate 측정 | — | 1-2 일 | 본 best 가 sonnet 특화 임 검증. 모든 workload 적용 전 필수 |
| ★ | SUB_046 retry | CPU draft model (Llama-3.2-1B CPU 추론 + GPU verify) | Tier 1 A 원래 plan | 3-5 일 | vLLM 내부 `SpeculativeConfig.draft_device_type=cpu` 지원 추가 필요. very large effort |

폐기: SUB_048 (spec sampling CPU offload — GPU lever 본질, CPU 활용 거의 없음 — 사용자 중단)

---

## 📂 5. 디렉토리 구조

```
TSK_020/
├── README.md                              ← Best Configuration index (현 best + history + 동작 원리 + 다음 path)
├── INDEX.md                               ← 본 파일 (navigation hub)
├── Best_SpecDecode_10778tps.md            ← Best 상세 fact + 설정 + mechanism + 코드 변경
├── measurements/
│   ├── sub044_spec_decode_20260523/       ← 첫 net-positive (spec=3/5/7/10 sweep)
│   └── sub047_t3_3run_verify_20260523/    ← ★★★ 현 best 확정 (3-run avg 10,956.6)
├── planning/
│   └── SUB_046_to_049_cpu_spec_plans.md   ← Tier 1/3 CPU+spec 결합 plan
└── analysis/                              ← (현재 비어 있음, 향후 spec workload 분석 등)
```

---

## 📜 6. 관련 task

| Task | 관계 |
|---|---|
| [`../TSK_019/`](../TSK_019/) | NEO architecture 정합 — net-negative 확정, **dead path**. SUB_001~SUB_043 (1~3차 frame). 본 task (TSK_020) 가 후속 |
| [`../TSK_020.md`](../TSK_020.md) | 본 task 의 정식 doc (sub-task 영역 + 상세 history) |
