# IDE_075 — IDE_074 후속 추가 측정 (측정 오류 정정 · Cold 실행 지연 분해 · 최적화 판단 근거)

지시서 `PLAN.md` (사용자 업로드 `IDE074_additional_measurement_plan.md`, SHA 8a3225f4…3971). 부모 `IDE_074` (게시 커밋 87dee8e55). 브랜치 `feat/cpu-moe-bottleneck-followup-20260917`.

## 목적
CPU 가 오래 실행된다는 사실과 GPU 가 실제로 기다렸다는 사실을 구분하고, 그 기다림을 생산 작업 기준(선행 FIFO 점유 / 직접 서비스 / 게시·전송 / 미분리)으로 분해해 다음 성능 개선 후보(최대 2개)의 근거를 확보한다. 성능 설정은 바꾸지 않는다.

## 산출물
`PLAN_RESOLVED.md`, `SOURCE_INDEX.md`, `CONFIG_DIFF.md`, `PROBE_MAP_V2.md`, `METRIC_DEFINITIONS_V2.yaml`, `WINDOW_RECONSTRUCTION.md`, `VALIDATION_REPORT.md`, `FULL_REPORT.md`, `BOTTLENECK_EVIDENCE.md`, `MEASUREMENT_COMPLETION_STATUS.md`, `READINESS.md`, `OPTIMIZATION_HANDOFF.md`, `WORK_LOG.md`, `FULL_RAW_DATA.md`(+raw_md), `ARTIFACT_INDEX.csv`, `SHA256SUMS.txt`, `PUBLISH_RECEIPT.md`, `profiles/{legacy_vs_corrected.csv, observer_overhead_v2.csv, missing_coverage_v2.csv, clock_alignment_summary.csv}`, `evidence/`. 결과 원자료 `eval/results/IDE_075_20260917/`, 하네스·분석기 `eval/ide075/`.

## 핵심 결과 (READINESS.md)
C64 디코드에서 GPU 가 기다린 315~334 µs/층 은 직전 층 deferred 작업의 직렬 실행(650~663 µs 대기) + 그 작업의 서비스 902~912 µs (up_gate 55 %, down 34 %, 전 expert AVX vec_mul, unique cold expert 수에 비례) 가 예산 1,248~1,270 µs 를 넘긴 결과다. FIFO 전달·dispatch·게시·복사 경로는 지연에 기여하지 않는다. 후보 A(Cold expert 서비스 시간)·B(관측 비용 기반 Hot 예산 재배치) 가 READY_WITH_LIMITED_SCOPE.
