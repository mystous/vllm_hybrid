# TSK_019 — INDEX (navigation hub)

> **목적**: TSK_019 영역 의 모든 doc / measurement 영역 의 single entry point.
> **organization scheme** (turn 11, 2026-05-21 KST 정리): active / reference / archive 분리.

## ★ 현재 active 영역 (지금 작업 중인 영역)

| 영역 | 파일 | size | 의미 |
|---|---|---:|---|
| ★ AMX 최적화 + 전체 최적화 통합 plan | [`planning/AMX_OPTIMIZATION_PLAN.md`](planning/AMX_OPTIMIZATION_PLAN.md) | (planning) | Phase α/β/γ/δ 영역 7 sub-task (A1~A7), cumulative 영역 예상 |
| ★ cdec leftover 제거 외부 idea 22 개 | [`analysis/N_cdec_leftover_elimination_ideas.md`](analysis/N_cdec_leftover_elimination_ideas.md) | 568 lines | 7 영역 × 22 idea (논문 + GitHub) + 우선순위 A/B/C |
| ★ SUB_015-Phase 3 HPC 최적화 분석 | [`analysis/M_sub015_phase3_hpc_optimization.md`](analysis/M_sub015_phase3_hpc_optimization.md) | (M doc) | HPC 측면 분석 + 외부 1차 출처 backing |
| ★ 현재 timeline 분석 + workflow | [`measurements/timeline_neo_amx_apply_20260520/README.md`](measurements/timeline_neo_amx_apply_20260520/README.md) | 1,559 lines | HEAD `0776086f5` timeline + workflow breakdown + env-OFF/ON measurement |
| ★ 현재 timeline SVG | [`measurements/timeline_neo_amx_apply_20260520/timeline_schematic.svg`](measurements/timeline_neo_amx_apply_20260520/timeline_schematic.svg) | 449 lines | 동적 분석 기반 도식 |

## 📊 측정 산출물 (Best config snapshots — production reference)

| 영역 | 파일 | env | 측정 | tps |
|---|---|---|---|---:|
| **★ Best S1-S9** | [`Best_S1_S9_2238tps.md`](Best_S1_S9_2238tps.md) | OFF | gmu=0.92, 500p × 8192, 3-run | **2,238.6** |
| Best v1.6 | [`Best_v1.6_2157tps.md`](Best_v1.6_2157tps.md) | OFF | gmu=0.92, 500p × 8192, 3-run | 2,197.4 |
| Best Phase 3.1 + KMP=50 | [`Best_Phase3_1_kmp50.md`](Best_Phase3_1_kmp50.md) | OFF | gmu=0.92, 400p × 8192, 1-run | 2,038.7 |

## 📂 디렉토리 구조 (turn 11 정리 후)

```
TSK_019/
├── INDEX.md                              ← 본 파일 (navigation hub)
├── README.md                             ← Best history + Reference measurements 영역 표
├── Best_S1_S9_2238tps.md                 ← ★ current best @ gmu=0.92
├── Best_v1.6_2157tps.md                  ← v1.6 best
├── Best_Phase3_1_kmp50.md                ← Phase 3.1 best (400p)
├── Performance_analaysis_v1.6.md         ← v1.6 perf analysis (historical)
├── After_NEO_implementation_plan.md      ← old implementation plan (historical)
│
├── analysis/                             ← 의미 별 분리
│   ├── M_sub015_phase3_hpc_optimization.md   ← ★ active: HPC 최적화 분석
│   ├── N_cdec_leftover_elimination_ideas.md  ← ★ active: cdec 제거 idea (turn 10)
│   │
│   ├── reference/                        ← SUB_015 단계별 분석 (M doc 이 통합 references)
│   │   ├── H_dynamic_analysis.md             ← perf record 60s fact
│   │   ├── H_phase1_final_levers.md
│   │   ├── H_phase2_results.md
│   │   ├── H_static_analysis.md
│   │   ├── I_amx_proper_design.md            ← AMX Step 1~6 strategy
│   │   ├── J_sub015_root_cause_analysis.md   ← Tier A/B/C backing
│   │   ├── K_sub015_improvement_roadmap.md   ← F1~F6 roadmap
│   │   └── L_sub015_evidence_based_priority.md  ← P1~P6 ranking
│   │
│   └── archive/                          ← 초기 5-phase 분석 (NEO rewrite plan 완료)
│       ├── A_kernel_signature_map.md
│       ├── A_neo_upstream_audit.md
│       ├── B_paper_section_notes.md
│       ├── B_paper_vs_our_measure.md
│       ├── C_existing_paths_inventory.md
│       ├── C_pacpu_vs_cpu_attn_amx_gap.md
│       ├── D_bottleneck_table.md
│       ├── D_candidate_long_list.md
│       ├── D_roofline_notes.md
│       ├── E_amx_avx_applicability.md
│       ├── E_bottleneck_map.md
│       ├── E_open_questions.md
│       ├── F_hardware_acceleration_candidates.md
│       └── G_neo_rewrite_plan.md
│
├── planning/                             ← Active plan documents (turn 11 신규)
│   └── AMX_OPTIMIZATION_PLAN.md          ← ★ active: 7 sub-task + Phase α/β/γ/δ
│
└── measurements/                         ← 측정 산출물 (각 dir = 1 회 측정)
    ├── timeline_neo_amx_apply_20260520/  ← ★ current timeline (1,559 lines + SVG)
    ├── p3_compare_3run_085_20260520/     ← ★ env-OFF baseline (3-run avg)
    ├── p4_p5_lever_20260520/             ← P4/P5 sweep
    ├── combo_sweep_20260520/             ← A×D combo
    ├── oob_root_fix_20260520/            ← OOB root fix v1
    ├── timeline_v16_s1_s9_20260517/      ← S1-S9 historical timeline (현재 timeline 의 base)
    ├── timeline_v16_optionA_20260516/    ← Option A historical
    ├── timeline_v16_20260516/            ← v1.6 historical
    ├── neo_s1_s9_500p_3run_20260517/     ← S1-S9 3-run measurement
    ├── neo_v1_6_500p_3run_20260516/      ← v1.6 3-run
    ├── neo_phase3_1_kmp200_500p_3run_*   ← Phase 3.1 sweep
    ├── neo_phase3_1_3_kmp200_500p_3run_* ← Phase 3.1+3.3 sweep
    ├── vanilla_3run_20260510/            ← vanilla baseline (gmu=0.85)
    ├── sub015_p3_amx_500p_3run_20260518/ ← AMX qk variant 3-run
    ├── sub015_p3_amx_steps_500p_1run_*   ← Step 1~6 sweep
    ├── sub015_p3_step5_amx_bav_*         ← Step 5 3-run 정식 검증
    └── sub015_p3_measurement_timeline_20260518.md  ← standalone
```

## 📚 reference (SUB_015 단계별 분석 — historical context 영역)

| 영역 | 파일 | 의미 | 통합 영역 |
|---|---|---|---|
| H_dynamic_analysis | [`analysis/reference/H_dynamic_analysis.md`](analysis/reference/H_dynamic_analysis.md) | perf record 60s (libgomp 43.75% / libpacpu 26.38% / libtorch 10.24% / python 1.84%) | M doc §1.2 |
| H_static_analysis | [`analysis/reference/H_static_analysis.md`](analysis/reference/H_static_analysis.md) | 정적 분석 | M doc §2 |
| H_phase1_final_levers | [`analysis/reference/H_phase1_final_levers.md`](analysis/reference/H_phase1_final_levers.md) | Phase 1 lever ranking | M doc §3.1 |
| H_phase2_results | [`analysis/reference/H_phase2_results.md`](analysis/reference/H_phase2_results.md) | Phase 2 결과 | M doc §3.2 |
| I_amx_proper_design | [`analysis/reference/I_amx_proper_design.md`](analysis/reference/I_amx_proper_design.md) | AMX Strategy A~H ranking + Step 1~6 plan | M doc §2.1 |
| J_sub015_root_cause_analysis | [`analysis/reference/J_sub015_root_cause_analysis.md`](analysis/reference/J_sub015_root_cause_analysis.md) | Tier A/B/C backing + Amdahl 한계 | M doc §1.3 |
| K_sub015_improvement_roadmap | [`analysis/reference/K_sub015_improvement_roadmap.md`](analysis/reference/K_sub015_improvement_roadmap.md) | F1~F6 lever roadmap | M doc §4 |
| L_sub015_evidence_based_priority | [`analysis/reference/L_sub015_evidence_based_priority.md`](analysis/reference/L_sub015_evidence_based_priority.md) | P1~P6 ranking (Tier 영역 backing) | M doc §4 |

## 📚 archive (초기 5-phase 분석 — NEO rewrite 완료)

| 영역 | 파일 | 의미 |
|---|---|---|
| A | [`analysis/archive/A_kernel_signature_map.md`](analysis/archive/A_kernel_signature_map.md), [`A_neo_upstream_audit.md`](analysis/archive/A_neo_upstream_audit.md) | kernel signature + NEO upstream audit |
| B | [`analysis/archive/B_paper_section_notes.md`](analysis/archive/B_paper_section_notes.md), [`B_paper_vs_our_measure.md`](analysis/archive/B_paper_vs_our_measure.md) | paper section 정리 + paper vs 측정 |
| C | [`analysis/archive/C_existing_paths_inventory.md`](analysis/archive/C_existing_paths_inventory.md), [`C_pacpu_vs_cpu_attn_amx_gap.md`](analysis/archive/C_pacpu_vs_cpu_attn_amx_gap.md) | existing paths + pacpu vs cpu attn gap |
| D | [`analysis/archive/D_bottleneck_table.md`](analysis/archive/D_bottleneck_table.md), [`D_candidate_long_list.md`](analysis/archive/D_candidate_long_list.md), [`D_roofline_notes.md`](analysis/archive/D_roofline_notes.md) | bottleneck table + 후보 long list + roofline |
| E | [`analysis/archive/E_amx_avx_applicability.md`](analysis/archive/E_amx_avx_applicability.md), [`E_bottleneck_map.md`](analysis/archive/E_bottleneck_map.md), [`E_open_questions.md`](analysis/archive/E_open_questions.md) | AMX/AVX applicability + bottleneck map + open questions |
| F | [`analysis/archive/F_hardware_acceleration_candidates.md`](analysis/archive/F_hardware_acceleration_candidates.md) | hardware acceleration 후보 |
| G | [`analysis/archive/G_neo_rewrite_plan.md`](analysis/archive/G_neo_rewrite_plan.md) | NEO rewrite plan (S1-S9 base) |

## 📜 시간 순 history (turn 1 ~ 11)

| Turn | 작업 영역 |
|---|---|
| 1-4 | timeline_v16/optionA/S1-S9 측정 + 도식 |
| 5 | 현재 HEAD 동적 측정 + py-spy 9-process 60s |
| 6 | b0/b1 sub-batch 구별 명확화 |
| 7 | env-gated default ON 측정 (100p) |
| 8 | env-ON 500p × 1-run validation + instrumentation 적재 + AMX plan 최상단 |
| 9 | 전체 workflow 영역 빠짐없이 적재 (py-spy full stack) |
| 10 | cdec leftover 제거 외부 idea 22 개 (N doc) |
| **11** | **★ 본 turn — 문서 영역 정리 (analysis/archive/, reference/, planning/ 분리)** |

## 🔧 작업 진행 영역 (next steps)

→ [`planning/AMX_OPTIMIZATION_PLAN.md`](planning/AMX_OPTIMIZATION_PLAN.md) §A.6 우선순위 영역 참조

| 우선순위 | 작업 | effort | 영역 |
|---|---|:-:|---|
| **★★★** | OMP barrier instrumentation 활성 + pacpu rebuild → barrier wait time 정량 | 30 min | [`planning/AMX_OPTIMIZATION_PLAN.md`](planning/AMX_OPTIMIZATION_PLAN.md) Phase α-1 |
| ★★★ | Tournament barrier env var sweep (KMP_FORCE_REDUCTION_BARRIER_PATTERN) | 2-3 시간 | [`analysis/N_cdec_leftover_elimination_ideas.md`](analysis/N_cdec_leftover_elimination_ideas.md) A1 |
| ★★ | KMP_BLOCKTIME=INF + KMP_AFFINITY 명시 sweep | 1 시간 | A2 |
| ★★ | env-ON 3-run avg (statistical confidence) | 90 min | [`measurements/timeline_neo_amx_apply_20260520/README.md`](measurements/timeline_neo_amx_apply_20260520/README.md) §18.8 |
| ★★ | FlashDecoding++ unified-max softmax | 3-5 일 | N doc B3 |
| ★ | OmniServe LSE async pattern (P4 race-safe) | 1-2 주 | N doc B1 |
