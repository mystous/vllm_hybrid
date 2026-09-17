# PLAN_RESOLVED — IDE_073 (지시서 cpu_offload_no_04_two_models_baseline_opt_probe)

## 식별
- campaign_id `IDE_073_20260917` · 브랜치 `feat/cpu-offload-two-models-20260917` (base 43e557fc2) · 노드 violet-h100-016 · T0(준비 시작) 2026-09-17 07:45 KST (UTC+9)
- 결과 루트 `eval/results/IDE_073_20260917/<cell>/<attempt>/…` · 원장 `state/execution_events.jsonl` · 상태 `state/RUN_STATE.json` · manifest `state/RUN_MANIFEST.json`
- 하네스: `eval/ide073/{run_campaign.py, quality20.py, report_loop.py, probes.py}` + `eval/ide071/{run_cell.py, common.py, configs/workloads.json}` 재사용
- 지시서 사본 `PLAN.md` (SHA256 6b742fb4…). 입력 자료: EXPERIMENT_SUMMARY_20260916 (IDE_072), IDE_071 FULL_REPORT, IDE_070 RESULT, IDE_069 FULL_REPORT (evidence/software/software_manifest.json files.*)

## 모델·자격 (§2)
| 모델 | revision | 파일 | tensor 합 | GPU8 | GPU4 GPU-only 자격 |
|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B-Instruct-FP8 | 003f183a… | 49 shard | 482.1 GB (index metadata) | IDE_069·IDE_071 R07 실행 기록 있음 → 이번에 재실행 | 482 GB > 4×80 GB → 산정상 불가 (기동 확인 생략, IDE_068 TP4 OOM 기록) |
| GLM-4.7-FP8 | 7b3b5f81… | 93 shard, 362.1 GB (HF 파일 합; index metadata total_size 0.2 GB 는 오기) | config: 92층(dense 0–2, routed 3–91, MTP 92), 160 routed + 1 shared, top-k 8, hidden 5120, moe_inter 1536, per-channel FP8 (compressed-tensors) | 이번에 확인 | 362 GB > 320 GB → 산정상 불가 |

## 실행 계획 (부팅 단위)
| 셀 | 구성 | 세션 | 품질 |
|---|---|---|---|
| Q-GPU8 | tp8 ep8 triton, KV 40,960 bf16, mem 0.90, graph ≤64, running 64, chunk 8192 | MAIN×3, LOW×1, LONG×1 | 20문항 |
| Q-KT-BASIC4 | BASELINE_PROVENANCE 표, 공식 checkout worktree (PYTHONPATH), KT_* unset, cpuinfer 112, uniform 96, deferred 2 | MAIN×3, LOW, LONG | 20 |
| Q-OPT4-b1 / b2 | IDE_072 V1 복원 (로컬 tree, CF·skip·RB·hotmap_v2·5,952·deferred 8·pin) | b1: MAIN×1, LOW, LONG, REPLAY(32) · b2: MAIN×2 | b1 20 |
| G-GPU8 / G-KT-BASIC4 / G-OPT4-b1·b2 | CONFIG_DIFF 표; OPT4 는 INT4 변환·calibration 성공 시 | 동일 | 동일 |
| 프로브 D0~D3 × (Q-OPT4, G-OPT4) | §8 | 각 1세션 (PROBE128) | — |

워크로드 = `configs/workloads.json` 의 `M073_{QWEN,GLM}_{MAIN_SHORT,LOW_CONCURRENCY,LONGER_PREFILL,PROBE128}` (custom jsonl, 같은 텍스트, 모델별 token id; `inputs/hashes.json`). ignore_eos, temperature 0, top_p 1, request-rate inf, 매 세션 전 flush(+3 s), 워밍업(C16·32, 출력 128 — 지시서 '출력 32 이하' 대신 기존 하네스 값; 기록) 후 flush. `/v1/completions` raw 경로.

## 예산 (§5.2)
PERF 30 · DIAG 8 · 재시도·조건부 6 · 세션 44 · 부팅 20 · 품질 120 문항 · GLM 변환 1(+재시도 1) · calibration 1 (128 요청·출력 64·C16). 계수는 원장 재계산.

## 동시 진행 조건 (기록)
- Qwen 셀 실행 중 GLM-4.7-FP8 다운로드(~80 MB/s, 네트워크·디스크 쓰기) 병행. 벤치와 같은 CPU 소켓을 쓰는 사용자 공간 프로세스는 아님(hf 다운로더 1 프로세스). 각 셀의 cpu_timeseries.csv 로 영향 범위 확인 가능.

## 미지원·차단 예정 항목
- `--kt-enable-dynamic-expert-update`, `--kt-gpu-prefill-token-threshold`, `--kt-expert-placement-strategy`: 설치본 미지원 (BASELINE_PROVENANCE).
- ktransformers.net 문서 snapshot: 응답 없음 → `SOURCE_SNAPSHOT_UNAVAILABLE` (W3 GitHub raw 는 확보).
- kt_kernel 바이너리: 공식 무패치 재빌드 없음 → `SOFTWARE_BASE_DIFFERS(kt_kernel)`; 기본군은 env 미설정으로 upstream 분기.
