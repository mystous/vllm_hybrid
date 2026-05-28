# TSK_020 — idea backlog (IDE_009 ~ IDE_014)

> **parent**: TSK_020
> **owner SUB**: [`SUB_072`](../planning/SUB_072_idea_backlog.md)
> **★ user-facing entry point**: [`/spec_decoding/README.md`](../../../../../spec_decoding/README.md) — Trident core / AGSD production guide
> **신규 SUB**: SUB_093 (57 cell + util) / SUB_094 (AGSD e2e Qwen 7B) / SUB_095 (AGSD e2e × 4 model, 12/12 positive) / SUB_096 (Phi-3 14B + Qwen 72B, **R/K boundary < 14B + cross-vendor 차이**)
> **prefix 정정 (2026-05-24)**: 기존 임시 `I001~I006` → CLAUDE.md prefix rule 정합 위해 **`IDE_009~IDE_014`** 로 재명명. parent prefix `IDE` 는 id_registry 의 "Idea" prefix 와 동일 카운터 사용 (IDE_001~IDE_008 = 큰 architectural idea, IDE_009~ = 본 TSK_020 의 measurement/doc idea).

## 사용 방법

1. 새 idea 가 생기면 다음 IDE 번호 (`grep "다음 부여 번호.*IDE" ../../../id_registry.md` 로 확인) 으로 `IDE_###_<slug>.md` 신규 생성. 본 README 표에 한 줄 추가.
2. idea 가 실제 진행되면:
   - 측정·코드 변경이 필요한 경우 → 새 `SUB_###` 신설 (id_registry +1), 해당 idea md 의 "관련 SUB" 에 링크.
   - 단순 doc 정정만 필요한 경우 → SUB 신설 없이 idea md 안의 procedure 따라 진행 후 idea status `완료` 로 갱신.
3. idea 의 진행 결과는 idea md 의 §7 결과 에 기록. 본 README 표의 status 도 동기화.

## ★ vanilla 대비 workload별 현 best (2026-05-24 기준)

**Llama-3.3-70B + TP=8 + H100×8 + 500p × 8192in × 8192max**:

| workload | vanilla tps | **현 best config** | best tps | **vs vanilla** |
|---|---:|---|---:|---:|
| **sonnet** | 4,679.8 | ngram cap=8/div=0 (SUB_047) | 10,956.5 | **+134.1% (2.341×)** ⭐ |
| **chat** | 2,186.0 | ngram cap=8 (SUB_071/075) | 3,006.6 | **+37.5% (1.376×)** |
| **code** | 6,964.5 | suffix_spec32 (SUB_074, eager) | 7,094 | **+1.85% (1.018×)** ⭐ mitigation |

**Qwen2.5 small model (TP=1) — code workload**:

| model | vanilla tps | ngram tps | vs vanilla |
|---|---:|---:|---:|
| Qwen 0.5B | 11,056.2 | 4,485.9 | **−59.4%** ✗ (ngram OFF 권장) |
| Qwen 1.5B | 11,015.5 | 4,195.1 | **−62.0%** ✗ (ngram OFF 권장) |

**Mixed traffic (workload-aware gating, SUB_080 analytical)**:

| mix scenario | always-on tps | gating tps | gating 향상 |
|---|---:|---:|---:|
| M1 sonnet-heavy (60:20:20) | 8,393 | 9,192 | +9.52% |
| **M2 balanced (34:33:33)** | 6,871 | **7,977** | **+16.09%** |
| **M3 code-heavy (10:20:70)** | 5,616 | **7,091** | **+26.26%** ⭐ |

## ★ 평가 종합 (2026-05-24)

- **[`evaluation_summary_20260524.md`](evaluation_summary_20260524.md)** — IDE_009~014 각각의 측정 결과 + vanilla 대비 + 성과 평가 (★★ / ★ / ◐) + production 의사결정 권장 + 후속 SUB candidate 종합. **본 6 시간 작업의 net contribution 한눈 보기**.
- **[`code_base_impact_20260524.md`](code_base_impact_20260524.md)** — 각 IDE 가 본 fork 영역 어떤 file 에 영향을 미쳤는지 분류 (A=vLLM core / B=wrapper / C=외부 tool / D=doc / E=외부 PR). **IDE_009~014 모두 vLLM core 영역 변경 없음** — 측정/wrapper/doc/PR 만. 본 fork 영역 vLLM patch 는 IDE 이전 SUB_047 등에서 이미 적재. "파편화" 정정용 grouping (Group α/β/γ/δ) 도 포함.
- **[`phase_execution_summary_20260524.md`](phase_execution_summary_20260524.md)** — 성능 향상 plan Phase 1~4 (SUB_080~083) 실행 결과 종합. **Phase 1 analytical +9.5~+30.3%**, Phase 2 부분 (1 blocker 해소, 다음 blocker 노출), Phase 3 viability analytical, Phase 4 design. 본 session 영역 vLLM core 추가 변경 = 5 줄 (`vllm/utils/__init__.py` re-export).

### ★ 2026-05-25 추가 진행 (SUB_085~092)

Phase 2 영역 **unblock 성공** ⭐⭐ + 모든 잔여 SUB 영역 진행 (제외 SUB_077 upstream PR). 종합: [`../COMPREHENSIVE_REPORT_20260525.md`](../COMPREHENSIVE_REPORT_20260525.md) (416 lines).

| SUB | 핵심 결과 |
|---|---|
| SUB_084 (Phase 2 follow-up) | `_is_v1_supported_oracle` stub 추가 — 2nd blocker, fundamental incompat 영역 결론 영역 잘못됨 영역 SUB_085 영역 입증 |
| **SUB_085 ⭐⭐ (Phase 2 unblock)** | `compilation_config={cudagraph_mode: PIECEWISE}` 영역 우회 — suffix + PIECEWISE 영역 3 workload 모두 net positive |
| SUB_086 (fair baseline) | vanilla gmu=0.80 + PIECEWISE — sonnet 7,710 (wrapper-historical noise 발견) |
| SUB_087 (all-fair) | ngram + PIECEWISE + gmu=0.80 — all-fair table 확정 |
| SUB_088 (small + suffix) | Qwen 0.5B/1.5B × suffix = universal regression 확장 |
| **SUB_089 ⭐ (canonical)** | sonnet 3-run avg **11,687.4** variance **0.20%** |
| **SUB_090 ⭐ (model-size sweep)** | Qwen 0.5B/1.5B/7B × code × 3 config — boundary 7B↔70B 확정 |
| **SUB_091 ⭐⭐ (issue #16258 정확 reproduction)** | opt-125m 영역 **2.13× regression — issue 영역 2.12× 영역 정확 일치** |
| SUB_092 (router HTTP PoC) | classifier router 0.26 ms/prompt — production deploy-ready |

→ 본 fork code 변경 **14 줄** (utils +5, arg_utils +9, 모두 backward-compat 100%) + wrapper env-tunable.

## 본 backlog 의 idea — 측정 결과 통합 (sonnet/chat/code workload 별)

### vanilla baseline (모든 비율의 기준)

**Llama-3.3-70B + TP=8 + 500p × 8192in × 8192max**:
- sonnet vanilla: **4,679.8 tps** (wall 875s)
- chat vanilla: **2,186.0 tps** (wall 151.1s)
- code vanilla: **6,964.5 tps** (wall 562.1s)

**Qwen2.5 + TP=1 + 50p × 1024in × 512max**:
- Qwen 0.5B vanilla: **11,056.2 tps**
- Qwen 1.5B vanilla: **11,015.5 tps**

### idea 별 측정 결과 (vanilla 대비)

| IDE ID | 자식 SUB | status | sonnet | chat | code | 주요 fact |
|---|---|---|---|---|---|---|
| [IDE_009](IDE_009_vanilla_contribution_framing.md) | SUB_073 | ✅ 완료 (doc only) | — | — | — | vanilla → vLLM built-in +130.3% (SUB_044) → fork patch +1.65% (SUB_047) 3-단계 breakdown. fork 단독 contribution = **+1.65%** |
| [IDE_010](IDE_010_suffix_decoding_measurement.md) ⭐ | SUB_074 | ✅ 완료 | 8,236 (**+76.0%**, eager) | 2,370 (+8.4%, eager) | **7,094 (+1.85%, eager)** ⭐ | suffix vs ngram K: sonnet 4.42/3.72 (1.19×) / chat 11.58/6.69 (1.73×) / **code 7.67/1.10 (★ 7×)**. enforce_eager penalty ~25%. cuda graph 호환 시 모든 workload 향상 가능 |
| [IDE_011](IDE_011_acceptance_rate_direct_measure.md) ⭐ | SUB_075 | ✅ 완료 | 10,909 (+133.1%, ngram) | 2,972 (+36.0%) | 5,362 (−23.0%) | acceptance 실측 — sonnet K=3.72/α=38.8%, **chat K=6.69/α=81.2% ⭐ (surprise)**, code K=1.10/α=1.4%. linear `1+7α` 가 mean_accept_len 와 정합 (Leviathan i.i.d. 가정 위반) |
| [IDE_012](IDE_012_workload_aware_gating_poc.md) | SUB_076 | ✅ 완료 (classifier only) | classified ✓ (500/500) | classified ✓ (500/500) | classified ✓ (500/500) | macro accuracy **1.000**. routing/serving 통합은 후속 (GPU 제약) |
| [IDE_013](IDE_013_vllm_upstream_pr.md) | SUB_077 | ◐ draft only (human submit 대기) | — | — | — | duplicate 0 (vllm-project/vllm), PR description + isolated diff 준비 |
| [IDE_014](IDE_014_issue_16258_repro.md) | SUB_078 (code) + SUB_079 (sonnet/chat 확장) | ✅ 완료 (code) / 활성 (sonnet/chat) | (SUB_079 진행 중) | (SUB_079 진행 중) | Qwen 0.5B: 4,486 (**−59.4%**), Qwen 1.5B: 4,195 (**−62.0%**) | issue #16258 의 small model + ngram = severe regression 패턴 재현. sonnet/chat 도 확정 시 small model 영역 R≫K universal regression 가설 검증 |
| [IDE_022](IDE_022_agsd_realistic_eval.md) | TBD | 활성 (계획) | (P1 oracle 측정 후) | (P1 oracle 측정 후) | (P1 oracle 측정 후) | 선행 IDE_012 의 in-distribution 1.000 accuracy 한계 정정. **합성 sonnet/chat/code 500p × 3 → 실 trace 5 corpus (LMSYS-Chat-1M / WildChat-1M / ShareGPT / LiveCodeBench / SWE-Bench Lite)** + accuracy → **decision regret (vs oracle method)** metric. 4 분류기 (C0 regex / C1 ext regex / C2 TF-IDF+LR / C3 MiniLM head) sweep |
| [IDE_023](IDE_023_cpu_parallelism_beyond_avx512.md) | TBD | 활성 (계획) | (P3 e2e 후) | (P3 e2e 후) | (P3 e2e 후) | 선행 IDE_015~020. paper §4 가설 (sustained CPU util ≥ 30%) 충족 + IDE_018 stub 답습 방지. **HW target = DGX B200 (8× B200 192GB HBM3e + 2× EMR Platinum 8570 112 코어), dev+prod 단일 환경**. **HPC 이론 model-driven + timing instrumentation 의무 + HW lever 전면 (AVX-512 / AMX / DSA / QAT / CMT / SMT / NUMA / CXL / NVLink5 / PCIe Gen5)**. 5-axis × anchor: **A1 Temporal — BSP + Amdahl + LogGP** / **A2 Compute SIMD — Roofline + AVX freq license + AMX TMUL** / **A3 Data plane — McCalpin STREAM + Shannon + LogGP** / **A4 Memory hierarchy — Mattson + SLIT + CXL 1.1 + Smith 2-level + FP8 (B200 HBM 1.5TB → multi-tenant + ≥128k context narrative)** / **A5 SMT — Tullsen + Snavely**. Timing L0-L7 (RDTSC / clock_gettime / perf PMU / Intel PCM / eBPF / CUDA event / shared-mem / Nsight Systems). P0 이론 게이트 ρ ≥ 1.5, P1 fit ≤ 20%. 10 자식 SUB. EMR 의 AMX/DSA/QAT/CMT/NUMA cross-socket 모두 must-deliver 격상 (이전 prod-only stretch). critical path: must-only 1 주 / 전체 2 주 |

### priority 설명

- ★★★ — 정직성 / accuracy 영역, 즉시 처리 권장
- ★★ — net new fact 확보 가능, 1-2 시간 effort
- ★ — net new fact 확보, 수 일 effort
- ◐ — exploratory, 후순위

### 진행 결과 종합

| 영역 | 결과 |
|---|---|
| ngram-spec 내부 lever (SUB_065~069) | 모두 기각 (plateau 확정) |
| workload-shape 의존성 | **확정** — sonnet +134% / chat +37.5% / code −23.2% (large model), small model 영역 모든 workload 회귀 |
| **code 회귀 mitigation** | **SuffixDecoding (IDE_010) 발견** — code K 7× 향상, +32% vs ngram (eager penalty 있어도 vanilla 보다 빠름) |
| acceptance framework | **R/K empirical validation** (IDE_011) — chat α=81% surprise, R≈1.30 가정 정합 |
| workload-aware gating | classifier 1.000 (IDE_012). routing 통합은 후속 |
| upstream contribution | PR draft 준비 (IDE_013), human submit 대기 |
| external cross-validation | small model 영역 R≫K 명제 corroboration (IDE_014/SUB_078, SUB_079 확장 진행 중) |
