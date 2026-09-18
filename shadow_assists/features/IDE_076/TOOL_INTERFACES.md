# TOOL_INTERFACES — IDE_076 (PLAN.md 부록 B 도구 역할 ↔ 실제 구현)

| 도구 역할 | 구현 | 입력 | 출력 · 실패 |
|---|---|---|---|
| baseline/preflight | `eval/ide076/runner.py` (부팅 시 .so SHA·hotmap/budget SHA·env·server_args 를 `variants/<v>/manifest_<boot>.json` 과 `<boot>/variant_manifest.json` 에 기록; IDE_075 하네스의 HEALTH_OK·smoke 4·warmup 32·runtime_proofs 계승) | `<PLAN> --variant --env K=V --hotmap --budget --tag` | 원장 `state/execution_events.jsonl`, 세션 `metrics.json`; 부팅 실패 → plan_abort, exit 2 |
| descriptor export / tile comparator / job replay (expert 단계) | `eval/ide076/expert_comparator.py` (컨테이너) | out_dir, `--scenarios hot_r1,hot_r2,hot_r3,hot_r1_e18,hot_r2_e18,hot_r1_n<N>,hot_r2_n<N>,seq_mixed --iters --threads --pools --seed`; env KT_* 로 변형 선택 | `expert_results.csv`(반복별 경과 ns), `outputs.pt`(bitwise 비교용), `env.json`(.so SHA·KT_ env); routing 계약 위반 시 assert 실패 |
| feature proof | `kt_kernel_ext.kt_opt_status()` / `kt_opt_validate()` (kt_opt.h; `KT_OPT_PROOF=1` 에서만 카운터 증가) | — | a1_eligible/r1/r2/r4/r8/fallback, pool_* (dispatch·tail), 잘못된 플래그 값 → RuntimeError |
| A2 plan verifier | (구현 시) descriptor coverage validator — 미구현 (A2 적용성 판정 후) | | |
| placement cost builder | `eval/ide076/b_cost_model.py` (컨테이너: torch) | `--freq <recorder .pt> --costs <producer_layer_costs.csv> --hotmap --budget --slope --out --k` | `expert_cost_table.csv`, `layer_summary.csv`, `candidate_swaps.jsonl`, `cost_model_provenance.json` |
| Hot swap generator / Hotmap validator | `eval/ide076/hotmap_tools.py validate|diff|swap` | hotmap.json + layer_budget.json (+ `--promote l:e --demote l:e --budget-moves lf:lt:k`) | `validate` exit 1 on error (순열·합 5,952·범위); `swap` → `<out>/hotmap.json, layer_budget.json, candidate.json` |
| variant runner | `runner.py` (위) + 계획 `R0_REF, MAIN3, MAIN3_C1_LONG, R0_PROBE, CORR1, CALIB(recorder start/dump/stop), SELECT` | | |
| comparison analyzer | `eval/ide076/compare_variants.py` | (없음: 캠페인 디렉터리 스캔) | `e2e_comparison.csv`(부록 A.5 필드), `e2e_stats.json`(variant×workload 통계·짝 비교 이득) |
| finalizer | `eval/ide071/publish_results.py` (IDE071_FEAT/IDE071_BR) + 렌더 (작성 예정 `render_report.py`) | | PUBLISH_RECEIPT.md, manifests |

exit code: 모든 러너·분석기는 실패 시 0 이 아닌 코드; 체인 스크립트는 단계별 `exit` 를 chain.log 에 기록.
