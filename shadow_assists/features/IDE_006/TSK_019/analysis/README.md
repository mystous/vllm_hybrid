# TSK_019/analysis/ — status index

> **Navigation**: [`../INDEX.md`](../INDEX.md) (전체 hub) ← 본 영역 의 entry point

## ★ active (현재 작업 영역 — 자주 참조)

| 파일 | size | 의미 |
|---|---:|---|
| [`M_sub015_phase3_hpc_optimization.md`](M_sub015_phase3_hpc_optimization.md) | 600 lines | SUB_015-Phase 3 HPC 측면 최적화 분석 + 외부 1차 출처 backing (Intel ISA / OpenBLAS / FlashDecoding++ 등) |
| [`N_cdec_leftover_elimination_ideas.md`](N_cdec_leftover_elimination_ideas.md) | 568 lines | **cdec leftover 제거 외부 idea 22 개** (7 영역: pipelining / CPU kernel / OMP barrier / async / cross-layer / GPU-only / NEO-like) — turn 10 |

## 📚 reference/ (SUB_015 단계별 분석 — M doc 이 통합 references)

| 파일 | 의미 | M doc 에서 영역 |
|---|---|---|
| [`reference/H_dynamic_analysis.md`](reference/H_dynamic_analysis.md) | perf record 60s — libgomp 43.75% / libpacpu 26.38% / libtorch 10.24% / python 1.84% | §1.2 |
| [`reference/H_static_analysis.md`](reference/H_static_analysis.md) | 정적 분석 | §2 |
| [`reference/H_phase1_final_levers.md`](reference/H_phase1_final_levers.md) | Phase 1 lever ranking | §3.1 |
| [`reference/H_phase2_results.md`](reference/H_phase2_results.md) | Phase 2 결과 | §3.2 |
| [`reference/I_amx_proper_design.md`](reference/I_amx_proper_design.md) | AMX Strategy A~H ranking + Step 1~6 plan | §2.1 |
| [`reference/J_sub015_root_cause_analysis.md`](reference/J_sub015_root_cause_analysis.md) | Tier A/B/C backing + Amdahl 한계 | §1.3 |
| [`reference/K_sub015_improvement_roadmap.md`](reference/K_sub015_improvement_roadmap.md) | F1~F6 lever roadmap | §4 |
| [`reference/L_sub015_evidence_based_priority.md`](reference/L_sub015_evidence_based_priority.md) | P1~P6 ranking (Tier 영역 backing) | §4 |

## 📚 archive/ (초기 5-phase 분석 — NEO rewrite plan 완료)

| Phase | 파일 | 의미 |
|---|---|---|
| A | [`archive/A_kernel_signature_map.md`](archive/A_kernel_signature_map.md), [`archive/A_neo_upstream_audit.md`](archive/A_neo_upstream_audit.md) | kernel signature + NEO upstream audit |
| B | [`archive/B_paper_section_notes.md`](archive/B_paper_section_notes.md), [`archive/B_paper_vs_our_measure.md`](archive/B_paper_vs_our_measure.md) | paper section + paper vs 측정 |
| C | [`archive/C_existing_paths_inventory.md`](archive/C_existing_paths_inventory.md), [`archive/C_pacpu_vs_cpu_attn_amx_gap.md`](archive/C_pacpu_vs_cpu_attn_amx_gap.md) | existing paths + pacpu vs cpu attn gap |
| D | [`archive/D_bottleneck_table.md`](archive/D_bottleneck_table.md), [`archive/D_candidate_long_list.md`](archive/D_candidate_long_list.md), [`archive/D_roofline_notes.md`](archive/D_roofline_notes.md) | bottleneck table + 후보 long list + roofline |
| E | [`archive/E_amx_avx_applicability.md`](archive/E_amx_avx_applicability.md), [`archive/E_bottleneck_map.md`](archive/E_bottleneck_map.md), [`archive/E_open_questions.md`](archive/E_open_questions.md) | AMX/AVX applicability + bottleneck map + open questions |
| F | [`archive/F_hardware_acceleration_candidates.md`](archive/F_hardware_acceleration_candidates.md) | hardware acceleration 후보 |
| G | [`archive/G_neo_rewrite_plan.md`](archive/G_neo_rewrite_plan.md) | NEO rewrite plan (S1-S9 base) |

## 작업 흐름 영역

```
[archive A-G: 초기 5-phase]
        ↓
[reference H-L: SUB_015 단계별 분석]
        ↓
[★ active M: HPC 최적화 통합 분석]
        ↓
[★ active N: cdec leftover 제거 외부 idea]
        ↓
[../planning/AMX_OPTIMIZATION_PLAN.md: 실행 plan]
```

## 정리 기준 (turn 11)

- **active** (analysis/ root) = 지금도 자주 참조 + 작성 중. M doc / N doc 만.
- **reference** (reference/) = 1회성이 아닌 backing 영역. M doc 의 cross-reference 영역 의 source.
- **archive** (archive/) = NEO rewrite plan (G doc) 완료된 영역. 재참조 영역 작음, 단 history 보존.
