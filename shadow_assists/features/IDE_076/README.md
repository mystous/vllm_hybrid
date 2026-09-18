# IDE_076 — CPU MoE 최적화 A1·A2·B 구현·검증

- 지시서: `PLAN.md` (사용자 업로드 `CPU_MoE_A1_A2_B_implementation_instructions.md`, SHA256 bc2a126a…). 실행 원칙: A1 → A2 적용성·구현 → 비용 재측정 → B → 개별·통합 검증 → 채택 또는 원복. 횟수 상한 없음.
- 부모: `IDE_075` (근거 S29, expert 표본, .so v4 12926df2…). 브랜치 `feat/cpu-moe-a1a2b-20260917`.
- 대상: Qwen3-Coder-480B FP8 OPT4 (H100×4, Xeon 2소켓, KT worker 96/NUMA 2, deferred 8, callback-free/skip-empty/RB ON, hotmap_v2 + layer_budget_5952).
- A1: 기존 RB(`avx_rb_rows<RB,SPLIT>`) 안의 실제 R=1/2·shard shape 경로의 잔여 비용 특수화. A2: 같은 Cold job 안의 descriptor 묶음 실행(상위 FIFO·job 순서 유지). B: 생산자 비용 기반 정적 Hot 배치(5,952 슬롯 고정, 승격/강등 쌍).
- 단계 O0~O10 / 작업 C00~C16 (PLAN.md §14, 부록 C). 상태값은 PLAN.md §15.1 의 계약을 쓴다.
- 산출물 위치: 이 디렉터리 (문서) + `eval/ide076/` (러너·분석기) + `eval/results/IDE_076_20260917/` (원자료: variants/·calibration/·placements/·state/).
