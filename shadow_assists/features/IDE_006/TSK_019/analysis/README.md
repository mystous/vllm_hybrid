# TSK_019/analysis/ — status index

> **Navigation**: [`../INDEX.md`](../INDEX.md) (전체 hub) ← 본 영역 의 entry point
> **정렬 규칙**: 각 카테고리 내 **시간순 (오래된 → 최신)** + 날짜 표기.

---

## ★ active (현재 작업 영역, 시간순)

| 날짜 | 파일 | size | 의미 |
|---|---|---:|---|
| 2026-05-20 | [`M_sub015_phase3_hpc_optimization.md`](M_sub015_phase3_hpc_optimization.md) | 599 lines | SUB_015-Phase 3 HPC 측면 최적화 분석 + 외부 1차 출처 backing (Intel ISA / OpenBLAS / FlashDecoding++) |
| 2026-05-21 | [`N_cdec_leftover_elimination_ideas.md`](N_cdec_leftover_elimination_ideas.md) | 568 lines | cdec leftover 제거 외부 idea 22 개 (7 영역: pipelining / CPU kernel / OMP barrier / async / cross-layer / GPU-only / NEO-like) — turn 10 |
| **2026-05-21** | [`O_stage1_stage3_root_cause.md`](O_stage1_stage3_root_cause.md) | — | **★ Stage 1+3 (24 tests, A1~A5) 영역 영역 영역** — 코드/라이브러리 레벨 분석 + B/C-tier 영역 영역 영역 — turn 17 |

---

## 📚 reference/ (SUB_015 단계별 분석 — M doc backing, 시간순)

| 날짜 | 파일 | 의미 | M doc 내 영역 |
|---|---|---|---|
| 2026-05-17 | [`reference/H_dynamic_analysis.md`](reference/H_dynamic_analysis.md) | perf record 60s — libgomp 43.75% / libpacpu 26.38% / libtorch 10.24% / python 1.84% | §1.2 |
| 2026-05-17 | [`reference/H_static_analysis.md`](reference/H_static_analysis.md) | 정적 분석 | §2 |
| 2026-05-17 | [`reference/H_phase1_final_levers.md`](reference/H_phase1_final_levers.md) | Phase 1 lever ranking | §3.1 |
| 2026-05-18 | [`reference/H_phase2_results.md`](reference/H_phase2_results.md) | Phase 2 결과 | §3.2 |
| 2026-05-18 | [`reference/I_amx_proper_design.md`](reference/I_amx_proper_design.md) | AMX Strategy A~H ranking + Step 1~6 plan | §2.1 |
| 2026-05-19 | [`reference/J_sub015_root_cause_analysis.md`](reference/J_sub015_root_cause_analysis.md) | Tier A/B/C backing + Amdahl 한계 | §1.3 |
| 2026-05-19 | [`reference/K_sub015_improvement_roadmap.md`](reference/K_sub015_improvement_roadmap.md) | F1~F6 lever roadmap | §4 |
| 2026-05-19 | [`reference/L_sub015_evidence_based_priority.md`](reference/L_sub015_evidence_based_priority.md) | P1~P6 ranking (Tier 영역 backing) | §4 |

---

## 📚 archive/ (초기 5-phase 분석 — NEO rewrite 완료, 시간순 + phase 순)

| 날짜 | Phase | 파일 | 의미 |
|---|---|---|---|
| 2026-05-15 | A | [`archive/A_kernel_signature_map.md`](archive/A_kernel_signature_map.md) | kernel signature |
| 2026-05-15 | A | [`archive/A_neo_upstream_audit.md`](archive/A_neo_upstream_audit.md) | NEO upstream audit |
| 2026-05-15 | B | [`archive/B_paper_section_notes.md`](archive/B_paper_section_notes.md) | paper section notes |
| 2026-05-15 | B | [`archive/B_paper_vs_our_measure.md`](archive/B_paper_vs_our_measure.md) | paper vs 측정 비교 |
| 2026-05-15 | C | [`archive/C_existing_paths_inventory.md`](archive/C_existing_paths_inventory.md) | existing paths inventory |
| 2026-05-15 | C | [`archive/C_pacpu_vs_cpu_attn_amx_gap.md`](archive/C_pacpu_vs_cpu_attn_amx_gap.md) | pacpu vs cpu_attn AMX gap |
| 2026-05-15 | D | [`archive/D_bottleneck_table.md`](archive/D_bottleneck_table.md) | bottleneck table |
| 2026-05-15 | D | [`archive/D_candidate_long_list.md`](archive/D_candidate_long_list.md) | 후보 long list |
| 2026-05-15 | D | [`archive/D_roofline_notes.md`](archive/D_roofline_notes.md) | roofline notes |
| 2026-05-15 | E | [`archive/E_amx_avx_applicability.md`](archive/E_amx_avx_applicability.md) | AMX/AVX applicability |
| 2026-05-15 | E | [`archive/E_bottleneck_map.md`](archive/E_bottleneck_map.md) | bottleneck map |
| 2026-05-15 | E | [`archive/E_open_questions.md`](archive/E_open_questions.md) | open questions |
| 2026-05-15 | F | [`archive/F_hardware_acceleration_candidates.md`](archive/F_hardware_acceleration_candidates.md) | hardware acceleration 후보 |
| 2026-05-17 | G | [`archive/G_neo_rewrite_plan.md`](archive/G_neo_rewrite_plan.md) | NEO rewrite plan (S1-S9 base) |

---

## 작업 흐름 영역 (시간순)

```
2026-05-15  [archive A-F: 초기 5-phase 분석]
                          ↓
2026-05-17  [archive G: NEO rewrite plan] → [reference H: perf record 60s + 정적/Phase1 분석]
                                              ↓
2026-05-18                                   [reference H phase2 + I: AMX Step 1~6]
                                              ↓
2026-05-19                                   [reference J: root cause + K: F1~F6 roadmap + L: P1~P6 ranking]
                                              ↓
2026-05-20                                   [★ active M: HPC 최적화 통합 분석]
                                              ↓
2026-05-21                                   [★ active N: cdec leftover 제거 외부 idea]
                                              ↓
2026-05-21                                   [Stage 1 (A1-A4 12 tests) + Stage 3 (A5+ 12 tests) 측정]
                                              ↓
2026-05-21                                   [★ active O: Stage 1+3 영역 영역 분석, turn 17]
                                              ↓
2026-05-20  [../planning/AMX_OPTIMIZATION_PLAN.md: 실행 plan, turn 8 작성]
                          ↓
2026-05-21  [turn 11: 영역 정리 (archive/reference/active 분리)]
            [turn 12: 카테고리 내 시간순 정렬]
            [turn 17: Stage 1+3 결과 + O 영역 영역 영역 ★ 본 turn]
```

---

## 정리 기준 (turn 11)

- **active** (analysis/ root) = 지금도 자주 참조 + 작성 중 (M, N)
- **reference** (reference/) = 1회성이 아닌 backing 영역. M doc 의 cross-reference 영역 의 source.
- **archive** (archive/) = NEO rewrite plan (G doc) 완료된 영역. 재참조 영역 작음, 단 history 보존.

## 정렬 규칙 (turn 12)

- 각 카테고리 내 **시간순 (오래된 → 최신)** 정렬
- 각 항목 옆에 날짜 (YYYY-MM-DD) 표기 — git log `--diff-filter=A --follow` 기준 첫 commit 날짜
- 동일 날짜 내 phase 순서 (A → B → C → D → E → F → G) 또는 sub-phase 순서 (H_static → H_phase1 → H_phase2 → I → J → K → L)
