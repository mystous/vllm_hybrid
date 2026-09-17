# COMPLETION_STATUS — IDE_073 (2026-09-17T11:07:44.390082+09:00)

- 캠페인 상태: COMPLETED_WITH_FAILURES
- 실행량: PERF 25/30 · DIAG 6/8 · 재시도·조건부 2/6 · 세션 33/44 · 부팅 14/20 · 품질 120/120

| 항목 | 상태 | 증거 |
|---|---|---|
| Q-GPU8 | COMPLETED | eval/results/IDE_073_20260917/Q-GPU8/ |
| Q-KT-BASIC4 | COMPLETED | eval/results/IDE_073_20260917/Q-KT-BASIC4/ |
| Q-OPT4-b1 | COMPLETED | eval/results/IDE_073_20260917/Q-OPT4-b1/ |
| Q-OPT4-b2 | COMPLETED | eval/results/IDE_073_20260917/Q-OPT4-b2/ |
| G-GPU8 | COMPLETED | eval/results/IDE_073_20260917/G-GPU8/ |
| G-KT-BASIC4 | TIMED_OUT_DURING_QUALITY | eval/results/IDE_073_20260917/G-KT-BASIC4/ |
| G-OPT4-b1 | NOT_RUN_DEPENDENCY | eval/results/IDE_073_20260917/G-OPT4-b1/ |
| G-OPT4-b2 | NOT_RUN_DEPENDENCY | eval/results/IDE_073_20260917/G-OPT4-b2/ |
| qwen 프로브 D0~D3 | RUN {'D0': 'valid', 'D2': 'valid', 'D3': 'valid', 'D1': 'valid'} | eval/results/IDE_073_20260917/qwen-PROBES/ |
| glm 프로브 D0~D3 | NOT_RUN | eval/results/IDE_073_20260917/glm-PROBES/ |

## 최종 체크 (§13)

| 항목 | 값 | 증거 |
|---|---|---|
| 두 모델 revision·GPU8·GPU4 자격 기록 | true | FULL_REPORT §1 |
| 공식/정규화/로컬 분리 | true | BASELINE_PROVENANCE.md, CONFIG_DIFF.md |
| offload 구성 GPU expert ≠ 0 | true (Qwen 96/층별 75~142, GLM 80) | config/ |
| 최고 설정 feature·hotmap·예산·RB 파싱 검증 | true | config/qwen/OPT4/*/effective_feature_paths.json, runtime_proofs |
| GLM 이식 상태 | OPT_TRANSFER_BLOCKED_FORMAT | state/glm_prepare_state.json |
| 성능 30·진단 8 각 항목 상태 | 위 표 | RUN_MANIFEST.json |
| 일반 성능/profiler 수치 분리 | true | §4 vs §5 |
| GPU 실행시간/CPU enqueue 구분 | torch profiler CUDA activity (D2) 범위에서 | profiles/ |
| 미계측 구간·overhead 공개 | true | FULL_REPORT §8 |
| 동일 문항 출력 보존 | true | quality/ |
| 30분 보고 이력 | true | FULL_REPORT §11 |
| 평가·추측·권고 없음 | true | — |
| raw·hash 보존 | true | ARTIFACT_INDEX.csv |
| Git push·원격 확인 | PUBLISH_RECEIPT.md | — |
| 다운로드 제공 | 최종 응답 | — |
