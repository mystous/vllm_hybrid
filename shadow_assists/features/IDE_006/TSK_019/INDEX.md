# TSK_019 — INDEX (navigation hub)

> **목적**: TSK_019 영역 의 모든 doc / measurement 영역 의 single entry point.
> **organization scheme** (turn 11): active / reference / archive / planning / measurements 카테고리 별 분리, 카테고리 내 시간순 정렬.
> **★ 정렬 규칙** (turn 12): 각 카테고리 내 **시간순 (오래된 → 최신)** + 각 항목 옆에 날짜 (YYYY-MM-DD) 표기.

---

## ★ 1. Active — 현재 작업 영역 (시간순)

| 날짜 | 영역 | 파일 | size | 의미 |
|---|---|---|---:|---|
| 2026-05-20 | timeline | [`measurements/timeline_neo_amx_apply_20260520/README.md`](measurements/timeline_neo_amx_apply_20260520/README.md) | 1,360 lines | HEAD `0776086f5` timeline + workflow breakdown + env-OFF/ON measurement (turn 5~9) |
| 2026-05-20 | timeline | [`measurements/timeline_neo_amx_apply_20260520/timeline_schematic.svg`](measurements/timeline_neo_amx_apply_20260520/timeline_schematic.svg) | 449 lines | 동적 분석 기반 도식 (b0/b1 + workflow phases + ③ sub-breakdown) |
| 2026-05-20 | analysis | [`analysis/M_sub015_phase3_hpc_optimization.md`](analysis/M_sub015_phase3_hpc_optimization.md) | 599 lines | SUB_015-Phase 3 HPC 측면 최적화 분석 + 외부 1차 출처 backing |
| 2026-05-20 | planning | [`planning/AMX_OPTIMIZATION_PLAN.md`](planning/AMX_OPTIMIZATION_PLAN.md) | 165 lines | AMX 최적화 + 전체 최적화 통합 plan (7 sub-task A1~A7 + Phase α/β/γ/δ) — turn 8 작성, turn 11 별도 doc 추출 |
| 2026-05-21 | analysis | [`analysis/N_cdec_leftover_elimination_ideas.md`](analysis/N_cdec_leftover_elimination_ideas.md) | 568 lines | cdec leftover 제거 외부 idea 22 개 (7 영역) — turn 10 |
| 2026-05-21 | measurements | [`measurements/stage1_a1a2a3a4_matrix_100p_20260521/RESULTS.md`](measurements/stage1_a1a2a3a4_matrix_100p_20260521/RESULTS.md) | — | Stage 1 A1~A4 조합 12 tests (100p × 8192, env-ON) — A4 단독 +1.0% ⭐ winner |
| 2026-05-21 | measurements | [`measurements/stage3_a5_matrix_100p_20260521/RESULTS.md`](measurements/stage3_a5_matrix_100p_20260521/RESULTS.md) | — | Stage 3 A5 + 조합 12 tests — A3+A5 +0.6% (Stage 3 best), A1+A5 -1.7% (anti-synergy) |
| **2026-05-21** | analysis | [`analysis/O_stage1_stage3_root_cause.md`](analysis/O_stage1_stage3_root_cause.md) | — | **★ Stage 1+3 (24 tests) 영역 영역 영역 — 코드/라이브러리 레벨 분석 + B/C-tier 영역 영역 영역 (turn 17)** |

---

## 📊 2. Best config snapshots (시간순)

| 날짜 | 파일 | env | tps (3-run avg) | 의미 |
|---|---|---|---:|---|
| 2026-05-14 | [`Best_v1.6_2157tps.md`](Best_v1.6_2157tps.md) | OFF | 2,197.4 | v1.6 best (commit `64f9e0c48`) |
| 2026-05-15 | [`Best_Phase3_1_kmp50.md`](Best_Phase3_1_kmp50.md) | OFF | 2,038.7 (1-run) | Phase 3.1 + KMP_BLOCKTIME=50 (400p) |
| 2026-05-17 | [`Best_S1_S9_2238tps.md`](Best_S1_S9_2238tps.md) | OFF | 2,238.6 | NEO S1-S9 best (deprecated by SpecDecode) |
| **2026-05-23** | **[`Best_SpecDecode_10778tps.md`](Best_SpecDecode_10778tps.md)** | **vanilla + spec=7 + ngram cap=8** | **10,956.6** | **★★★ current absolute best — SUB_047 t3 (cap=8 + div_tp=0) 3-run avg (min 10,949.8 / max 10,963.5, variance 0.125%), +134.1% vs vanilla** ⭐ |

---

## 📊 3. Measurements — 측정 산출물 (시간순)

### 3.1 vanilla / Pre-NEO baseline

| 날짜 | 디렉토리 | 측정 | 결과 |
|---|---|---|---:|
| 2026-05-10 | [`measurements/vanilla_3run_20260510/`](measurements/vanilla_3run_20260510/) | vanilla 분모 (gmu=0.85, 500p × 8192, 3-run) | 4,680.2 tps |

### 3.2 NEO baseline + Phase 3.1 sweep

| 날짜 | 디렉토리 | 측정 | 결과 |
|---|---|---|---:|
| 2026-05-16 | [`measurements/neo_v1_6_500p_3run_20260516/`](measurements/neo_v1_6_500p_3run_20260516/) | v1.6 best @ gmu=0.92, 3-run | 2,197.4 tps |
| 2026-05-16 | [`measurements/neo_phase3_1_kmp200_500p_3run_20260516/`](measurements/neo_phase3_1_kmp200_500p_3run_20260516/) | Phase 3.1 (Persistent OMP) KMP=200 | 2,134.9 tps |
| 2026-05-16 | [`measurements/neo_phase3_1_3_kmp200_500p_3run_20260516/`](measurements/neo_phase3_1_3_kmp200_500p_3run_20260516/) | Phase 3.1+3.3 cherry-pick | 2,083.3 tps |
| 2026-05-16 | [`measurements/timeline_v16_20260516/`](measurements/timeline_v16_20260516/) | v1.6 timeline 도식 (sync 첫 측정) | — |
| 2026-05-16 | [`measurements/timeline_v16_optionA_20260516/`](measurements/timeline_v16_optionA_20260516/) | Option A timeline 도식 (sync 재측정) | — |

### 3.3 S1-S9 (NEO 원본 정합)

| 날짜 | 디렉토리 | 측정 | 결과 |
|---|---|---|---:|
| 2026-05-17 | [`measurements/neo_s1_s9_500p_3run_20260517/`](measurements/neo_s1_s9_500p_3run_20260517/) | S1-S9 @ gmu=0.92, 3-run | **★ 2,238.6 tps** |
| 2026-05-17 | [`measurements/timeline_v16_s1_s9_20260517/`](measurements/timeline_v16_s1_s9_20260517/) | S1-S9 timeline 도식 (현재 timeline 의 base) | — |

### 3.4 SUB_015-Phase 3 AMX 측정

| 날짜 | 디렉토리 | 측정 | 결과 |
|---|---|---|---:|
| 2026-05-18 | [`measurements/sub015_p3_amx_500p_3run_20260518/`](measurements/sub015_p3_amx_500p_3run_20260518/) | AMX qk variant 3-run | 2,142.5 tps (-4.3% vs S1-S9) |
| 2026-05-18 | [`measurements/sub015_p3_amx_steps_500p_1run_20260518/`](measurements/sub015_p3_amx_steps_500p_1run_20260518/) | Step 1~6 sweep | Step 5 = 2,284.0 (1-run) |
| 2026-05-18 | [`measurements/sub015_p3_step5_amx_bav_500p_3run_20260518/`](measurements/sub015_p3_step5_amx_bav_500p_3run_20260518/) | Step 5 (B+A+vec K) 3-run 정식 | 2,186.1 tps (-2.35% vs S1-S9) |
| 2026-05-18 | [`measurements/sub015_p3_measurement_timeline_20260518.md`](measurements/sub015_p3_measurement_timeline_20260518.md) | AMX 측정 timeline 영역 standalone doc | — |

### 3.5 gmu=0.85 cross-env + P3/P4/D/OOB (2026-05-19~20)

| 날짜 | 디렉토리 | 측정 | 결과 |
|---|---|---|---:|
| 2026-05-19~20 | [`measurements/p3_compare_3run_085_20260520/`](measurements/p3_compare_3run_085_20260520/) | gmu=0.85 cross-env 5-case 3-run (vanilla / v1.6 / S1-S9 / P3 / P1) | v1.6 best 1,833.0 / S1-S9 1,800.1 / P3 1,787.9 |
| 2026-05-20 | [`measurements/p4_p5_lever_20260520/`](measurements/p4_p5_lever_20260520/) | P4 (async cdec) + P5 (MIRROR sweep) | unstable / 80=best |
| 2026-05-20 | [`measurements/combo_sweep_20260520/`](measurements/combo_sweep_20260520/) | A (TP=4) × D (OOB silent) 4-combo | TP=4 NO_RESULT / D +0.1% noise |
| 2026-05-20 | [`measurements/oob_root_fix_20260520/`](measurements/oob_root_fix_20260520/) | G/H log rate-limit fix v1 | log 22,755× ↓, NO_RESULT 동일 |

### 3.6 ★ 현재 timeline (HEAD `0776086f5`)

| 날짜 | 디렉토리 | 측정 | 결과 |
|---|---|---|---:|
| 2026-05-20 | [`measurements/timeline_neo_amx_apply_20260520/`](measurements/timeline_neo_amx_apply_20260520/) (★ active) | env-OFF 100p + env-ON 100p + ★ env-ON 500p × 1-run + workflow breakdown | env-ON 500p = 1,833.95 (+0.05% noise vs v1.6 best) |

### 3.7 ★ Stage 1+3 A1~A5 조합 매트릭스 + SUB_032 검증 (HEAD `0d7dc0334`, 100p × 8192, env-ON baseline)

| 날짜 | 디렉토리 | 측정 | 결과 |
|---|---|---|---:|
| 2026-05-21 | [`measurements/stage1_a1a2a3a4_matrix_100p_20260521/`](measurements/stage1_a1a2a3a4_matrix_100p_20260521/) | Stage 1 — A1~A4 조합 12 tests | t04 A4 단독 = 941.1 tps (1-run) → **noise** (SUB_032) |
| 2026-05-21 | [`measurements/stage3_a5_matrix_100p_20260521/`](measurements/stage3_a5_matrix_100p_20260521/) | Stage 3 — A5 + 조합 12 tests | 모두 noise band 안 |
| 2026-05-21 | [`analysis/O_stage1_stage3_root_cause.md`](analysis/O_stage1_stage3_root_cause.md) | Stage 1+3 root cause (코드/라이브러리 레벨) | A-tier 전체 무효 → B/C-tier 권고 |
| **2026-05-21** | **[`measurements/sub032_a4_3run_20260521/`](measurements/sub032_a4_3run_20260521/)** | **SUB_032 — A4 단독 3-run avg** | **avg 930.2 tps (±0.18%), Δ=-0.21% — Stage 1 의 +1.0% 가 1-run artifact 확정** |
| **2026-05-21** | **[`measurements/sub033_b3_softmax_20260521/`](measurements/sub033_b3_softmax_20260521/)** | **SUB_033 — B3 FlashDecoding++ online softmax (3-way)** | **OFF avg 922.65 / ON 915.1 / Δ=-0.82% → 기각 (default OFF 유지)** |
| 2026-05-22 | [`measurements/sub034_b1_async_depth_20260522/`](measurements/sub034_b1_async_depth_20260522/) | SUB_034 — B1 OmniServe async cdec depth (5-way 1/2/3/1/2) | depth=1 avg 921.95 / depth=2 avg 922.95 / Δ=+0.1% → noise, default depth=1 유지 |
| **2026-05-22** | **[`measurements/sub035_c1a_omp_launch_20260522/`](measurements/sub035_c1a_omp_launch_20260522/)** | **★ SUB_035 C1a — OMP team launch overhead instrumentation** | **launch=1.22% / step1 ISPC=70% / b2_wait=0-30% → C-tier 폐기, step1 algorithmic 변경이 진짜 lever** |
| **2026-05-22** | **[`measurements/sub036_pathA_500p_baseline_20260522/`](measurements/sub036_pathA_500p_baseline_20260522/)** | **★ SUB_036 Path A — 500p × 8192 NEO + vanilla baseline** | **NEO 1779 / vanilla 4681 → vanilla 2.63× 빠름. NEO 본 워크로드에서 net negative → vanilla OOM 영역 점검 필요** |
| **2026-05-22** | **[`measurements/sub040_util_baseline_20260522/`](measurements/sub040_util_baseline_20260522/)** | **★★★ SUB_040 CPU/GPU util baseline — 본 프로젝트 목표 미달 확정** | **NEO CPU 11.93% (목표 "극대화"와 큰 거리) / GPU 66% (vanilla 73.4% 대비 idle 발생) / power -31% → net trade-off 음수** |
| **2026-05-22** | **[`measurements/sub041_multi_workload_20260522/`](measurements/sub041_multi_workload_20260522/)** | **★★★ SUB_041 Multi-workload 서버 throughput — NEO net-negative 최종 확정** | **vanilla+BG 4679 (-0.04%) vs NEO+BG 1652 (-13%) → vanilla 가 BG 와 자원 분담, NEO 는 contention 으로 손실. 본 환경에서 NEO 의 raison d'être 무효** |
| **2026-05-22** | **[`measurements/sub042_prefill_decode_20260522/`](measurements/sub042_prefill_decode_20260522/)** | **★★★ SUB_042 prefill/decode 분리 — NEO 의 raison d'être 가설 깨짐** | **모든 phase 에서 vanilla 3.26-4.11× faster. NEO 가설 영역 decode-heavy 에서도 vanilla 4.00× 압승. 본 환경 batch+paging 으로 충분, NEO offload 가 항상 overhead** |
| **2026-05-23** | **[`measurements/sub044_spec_decode_20260523/`](measurements/sub044_spec_decode_20260523/)** | **★★★ SUB_044 ngram spec decode — 첫 net-positive 성과!** | **vanilla 4680 → spec=7 10778 tps (+130%, 2.30× faster) ⭐ — SUB_032-043 12 SUB 모두 noise 후 처음으로 vanilla 보다 빠른 lever** |
| **2026-05-23** | **[`measurements/sub047_t3_3run_verify_20260523/`](measurements/sub047_t3_3run_verify_20260523/)** | **★★★ SUB_047 t3 (cap=8 + div_tp=0) 3-run verify** | **avg 10,956.6 / min 10,949.8 / max 10,963.5 tps (variance 0.125%) — +134.1% vs vanilla ⭐ 현재 best 확정** |

---

## 📚 4. Reference — SUB_015 단계별 분석 (M doc backing, 시간순)

| 날짜 | 파일 | 의미 | M doc 내 영역 |
|---|---|---|---|
| 2026-05-17 | [`analysis/reference/H_dynamic_analysis.md`](analysis/reference/H_dynamic_analysis.md) | perf record 60s — libgomp 43.75% / libpacpu 26.38% / libtorch 10.24% / python 1.84% | §1.2 |
| 2026-05-17 | [`analysis/reference/H_static_analysis.md`](analysis/reference/H_static_analysis.md) | 정적 분석 | §2 |
| 2026-05-17 | [`analysis/reference/H_phase1_final_levers.md`](analysis/reference/H_phase1_final_levers.md) | Phase 1 lever ranking | §3.1 |
| 2026-05-18 | [`analysis/reference/H_phase2_results.md`](analysis/reference/H_phase2_results.md) | Phase 2 결과 | §3.2 |
| 2026-05-18 | [`analysis/reference/I_amx_proper_design.md`](analysis/reference/I_amx_proper_design.md) | AMX Strategy A~H ranking + Step 1~6 plan | §2.1 |
| 2026-05-19 | [`analysis/reference/J_sub015_root_cause_analysis.md`](analysis/reference/J_sub015_root_cause_analysis.md) | Tier A/B/C backing + Amdahl 한계 | §1.3 |
| 2026-05-19 | [`analysis/reference/K_sub015_improvement_roadmap.md`](analysis/reference/K_sub015_improvement_roadmap.md) | F1~F6 lever roadmap | §4 |
| 2026-05-19 | [`analysis/reference/L_sub015_evidence_based_priority.md`](analysis/reference/L_sub015_evidence_based_priority.md) | P1~P6 ranking (Tier 영역 backing) | §4 |

---

## 📚 5. Archive — 초기 5-phase 분석 (NEO rewrite 완료, 시간순)

| 날짜 | Phase | 파일 | 의미 |
|---|---|---|---|
| 2026-05-15 | A | [`analysis/archive/A_kernel_signature_map.md`](analysis/archive/A_kernel_signature_map.md) | kernel signature |
| 2026-05-15 | A | [`analysis/archive/A_neo_upstream_audit.md`](analysis/archive/A_neo_upstream_audit.md) | NEO upstream audit |
| 2026-05-15 | B | [`analysis/archive/B_paper_section_notes.md`](analysis/archive/B_paper_section_notes.md) | paper section notes |
| 2026-05-15 | B | [`analysis/archive/B_paper_vs_our_measure.md`](analysis/archive/B_paper_vs_our_measure.md) | paper vs 측정 비교 |
| 2026-05-15 | C | [`analysis/archive/C_existing_paths_inventory.md`](analysis/archive/C_existing_paths_inventory.md) | existing paths inventory |
| 2026-05-15 | C | [`analysis/archive/C_pacpu_vs_cpu_attn_amx_gap.md`](analysis/archive/C_pacpu_vs_cpu_attn_amx_gap.md) | pacpu vs cpu_attn AMX gap |
| 2026-05-15 | D | [`analysis/archive/D_bottleneck_table.md`](analysis/archive/D_bottleneck_table.md) | bottleneck table |
| 2026-05-15 | D | [`analysis/archive/D_candidate_long_list.md`](analysis/archive/D_candidate_long_list.md) | 후보 long list |
| 2026-05-15 | D | [`analysis/archive/D_roofline_notes.md`](analysis/archive/D_roofline_notes.md) | roofline notes |
| 2026-05-15 | E | [`analysis/archive/E_amx_avx_applicability.md`](analysis/archive/E_amx_avx_applicability.md) | AMX/AVX applicability |
| 2026-05-15 | E | [`analysis/archive/E_bottleneck_map.md`](analysis/archive/E_bottleneck_map.md) | bottleneck map |
| 2026-05-15 | E | [`analysis/archive/E_open_questions.md`](analysis/archive/E_open_questions.md) | open questions |
| 2026-05-15 | F | [`analysis/archive/F_hardware_acceleration_candidates.md`](analysis/archive/F_hardware_acceleration_candidates.md) | hardware acceleration 후보 |
| 2026-05-17 | G | [`analysis/archive/G_neo_rewrite_plan.md`](analysis/archive/G_neo_rewrite_plan.md) | NEO rewrite plan (S1-S9 base) |

---

## 📜 6. Historical — 이전 작업 (시간순)

| 날짜 | 파일 | 의미 |
|---|---|---|
| 2026-05-15 | [`After_NEO_implementation_plan.md`](After_NEO_implementation_plan.md) | NEO implementation plan (S1-S9 이전, after completion) |
| 2026-05-15 | [`Performance_analaysis_v1.6.md`](Performance_analaysis_v1.6.md) | v1.6 시점 성능 분석 |

---

## 📜 7. 시간순 history (turn 1 ~ 12)

| Turn | 날짜 | 작업 영역 |
|---|---|---|
| 1-4 | 2026-05-16~17 | timeline_v16/optionA/S1-S9 측정 + 도식 |
| 5 | 2026-05-20 | 현재 HEAD 동적 측정 + py-spy 9-process 60s (turn 5 = timeline_neo_amx_apply 신설) |
| 6 | 2026-05-20 | b0/b1 sub-batch 구별 명확화 |
| 7 | 2026-05-21 | env-gated default ON 측정 (100p) |
| 8 | 2026-05-21 | env-ON 500p × 1-run validation + instrumentation 적재 + AMX plan 통합 plan 작성 |
| 9 | 2026-05-21 | 전체 workflow 영역 빠짐없이 적재 (py-spy full stack) |
| 10 | 2026-05-21 | cdec leftover 제거 외부 idea 22 개 (N doc) |
| 11 | 2026-05-21 | 문서 영역 정리 (방안 B: active/reference/archive/planning 분리) |
| **12** | **2026-05-21** | **★ 본 turn — 카테고리 내 시간순 정렬 + 날짜 표기** |

---

## 🔧 8. 작업 진행 영역 (next steps, 우선순위 순)

→ 상세: [`planning/AMX_OPTIMIZATION_PLAN.md`](planning/AMX_OPTIMIZATION_PLAN.md) §A.6

| 우선순위 | 작업 | effort | 영역 |
|---|---|:-:|---|
| **★★★** | OMP barrier instrumentation 활성 + pacpu rebuild → barrier wait time 정량 | 30 min | AMX plan Phase α-1 |
| **★★★** | Tournament barrier env var sweep (KMP_FORCE_REDUCTION_BARRIER_PATTERN) | 2-3 시간 | [N doc A1](analysis/N_cdec_leftover_elimination_ideas.md) |
| ★★ | KMP_BLOCKTIME=INF + KMP_AFFINITY 명시 sweep | 1 시간 | [N doc A2](analysis/N_cdec_leftover_elimination_ideas.md) |
| ★★ | env-ON 3-run avg (statistical confidence) | 90 min | [timeline §18.8](measurements/timeline_neo_amx_apply_20260520/README.md) |
| ★★ | FlashDecoding++ unified-max softmax | 3-5 일 | [N doc B3](analysis/N_cdec_leftover_elimination_ideas.md) |
| ★ | OmniServe LSE async pattern (P4 race-safe) | 1-2 주 | [N doc B1](analysis/N_cdec_leftover_elimination_ideas.md) |

---

## 📂 9. 디렉토리 구조 (참조)

```
TSK_019/
├── INDEX.md                              ← 본 파일 (navigation hub)
├── README.md                             ← Best history index + Reference measurements 표
│
├── (root 영역 docs)                     ← Best config + historical
│   ├── Best_v1.6_2157tps.md              (2026-05-14)
│   ├── Best_Phase3_1_kmp50.md            (2026-05-15)
│   ├── Best_S1_S9_2238tps.md             (2026-05-17) ★ current best
│   ├── After_NEO_implementation_plan.md  (2026-05-15) historical
│   └── Performance_analaysis_v1.6.md     (2026-05-15) historical
│
├── analysis/
│   ├── README.md                         ← status doc
│   ├── M_sub015_phase3_hpc_optimization.md  (2026-05-20) ★ active
│   ├── N_cdec_leftover_elimination_ideas.md (2026-05-21) ★ active
│   ├── reference/                        (8 files, 2026-05-17~19) ← M doc backing
│   └── archive/                          (14 files, 2026-05-15~17) ← 초기 5-phase
│
├── planning/
│   └── AMX_OPTIMIZATION_PLAN.md          (2026-05-20) ★ active
│
└── measurements/                         (17 dirs + 1 standalone, 2026-05-10 ~ 05-20)
    ├── 3.1 vanilla baseline             (2026-05-10)
    ├── 3.2 NEO baseline + Phase 3.1      (2026-05-16)
    ├── 3.3 S1-S9                         (2026-05-17)
    ├── 3.4 SUB_015-Phase 3 AMX           (2026-05-18)
    ├── 3.5 gmu=0.85 + P3/P4/D/OOB        (2026-05-19~20)
    └── 3.6 ★ 현재 timeline               (2026-05-20)
```
