# PLAN_RESOLVED — IDE_071 (지시서 cpu_offload_no_02 의 저장소 대응)

## 식별
- campaign_id: `IDE_071_20260916` · 브랜치 `feat/cpu-offload-ide071` (base 448aad450, feat/cpu-offload-diag) · 노드 violet-h100-016
- 결과 루트: `eval/results/IDE_071_20260916/<cell_id>/<attempt_id>/<rep_id>/` (지시서 §18.1 의 `eval/results/<campaign_id>/…` 그대로)
- 상태: `eval/results/IDE_071_20260916/state/{state.json, journal.jsonl}` → 게시 시 `shadow_assists/features/IDE_071/state/` 로 복사

## 지시서 경로 ↔ 실제 경로
| 지시서 | 실제 |
|---|---|
| `eval/ide071/run_all.py` | 동일 (캠페인 드라이버: 고정 seed 무작위 순서, 6셀/90분 앵커 재측정, 재시도 ≤2) |
| `run_cell.py` | 동일 (§16.1 상태 머신, 셀 단위 예외 경계) |
| `report_progress.py` | 동일 (1,800 s 경계 보고 파일; 사용자 전달은 세션 cron 이 파일을 읽어 수행, delivery_log 에 기록) |
| `verify_environment.py` | 동일 → `evidence/software_manifest.json`, `evidence/environment.before.txt`, `evidence/source_changes.patch` |
| `verify_effective_config.py` | `run_cell.py` 내부 VERIFY_EFFECTIVE_CONFIG 단계 (`/get_server_info` + 서버 로그 증거 → `effective_config.json`, `runtime_proofs.json`) |
| `build_manifest.py` | 동일 (`manifests/cells_<phase>.jsonl`; P3~P7 은 앵커 셀 spec 입력) |
| `build_hotmap.py` | P6 에서 작성 (CALIBRATION 트레이스 → PL03/04/05, α map) |
| `collect_metrics.py`, `render_data_report.py`, `validate_artifacts.py`, `publish_results.py` | P11 후 작성·실행 |
| `configs/` | `configs/r0.json` (R0), `configs/workloads.json` (§4.2 워크로드) |
| `eval/results/<cell>/<attempt>/<rep>/` 파일 | launch_cmd.sh, bench_cmd.sh, effective_config.json(셀), requested_config.json, status.json, timestamps.json, bench.stdout/stderr.log, server.full.log.gz(셀), benchmark.raw.json, requests.jsonl, metrics.json, cpu_timeseries.csv, gpu_timeseries.csv, memory_timeseries.csv, thread_affinity_start/end.json, errors.jsonl(셀). layer_step_metrics.jsonl / expert_metrics.jsonl / stream_events.jsonl 은 DIAGNOSTIC 셀에서만 (not_applicable 사유: PERF_CLEAN) |

## 워크로드 (configs/workloads.json)
| 계열 | 구현 | 비고 |
|---|---|---|
| LEGACY_070 | sonnet 512/128 prefix100 seed42, 요청 4×C, ignore_eos 없음, 반복 간 flush 없음 | IDE_070 `lib.sh` 와 동일 |
| SHORT_COLD | sonnet 512/128 prefix0 seed20260916, ignore_eos, 매 반복 전 `/flush_cache` + 3 s | |
| PREFIX_WARM | sonnet 512/128 prefix100 seed20260917, flush 후 prefix 만 1요청 준비 | |
| DECODE_HEAVY | sonnet 512/1024 | ignore_eos 로 1024 강제 |
| PREFILL_HEAVY | sonnet 4096/128 | |
| LONG_CONTEXT | random 16384/256 | sonnet 으로 16k 불가 → random 토큰 (기록) |
| CODE_MIX | custom jsonl (HumanEval 164) output 256 | |
| MIXED_SERVICE | custom jsonl 300 요청, output 128 고정 | custom dataset 행별 output_len 미지원 → DECODE_HEAVY 행도 128 (제약) |

## 측정 규율 적용
- PERF_CLEAN 수집기: /proc/stat 1 s (코어별), nvidia-smi 2 s, free/numastat 5 s, TID affinity 시작·종료 스냅샷. profiler 없음.
- 반복 통계: sample sd (ddof=1). p95 는 반복별 값(`mean_of_rep_p95`) 과 요청 풀 p95(`pooled_request_p95`, requests.jsonl) 구분.
- 셀 timeout: 부팅 1,200 s (첫 JIT 포함 셀 1,800 s), 벤치 1,800 s, 셀 4 h.
- turbo: BIOS 잠금 (root 쓰기 거부, IDE_030/IDE_070 기록). P7 에서 1회 시도 기록 후 `BLOCKED_PERMISSION`.
- 실행 지속: `setsid nohup` + 상태 파일; 30 분 보고 루프 별도 프로세스; 세션 cron 이 사용자 전달.

## 등록된 단계와 셀 수 (생성 시점 갱신)
| 단계 | 셀 | manifest |
|---|---|---|
| P1 | R00~R07 (8) | manifests/cells_P1.jsonl |
| P2 | 18 + 교차 2 (20) | manifests/cells_P2.jsonl |
| P3~P7 | 앵커 확정 후 생성 | — |
| P8~P11 | 조건부·최종 | — |
