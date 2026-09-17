# IDE_074 — CPU MoE Hot/Cold 부분 상주 병목 측정

지시서: `PLAN.md` (사용자 업로드 `CPU_MoE_bottleneck_measurement_todo.md`, 2026-09-17, SHA256 374c8f6f…c520f7). 부모 `IDE_073` (기준 커밋 `67d24dd84`).

## 목적
Qwen3-Coder-480B OPT4 구성(hot expert GPU 상주 + cold expert CPU AMXINT4, TP4, callback-free·skip-empty·RB·층별 예산 유지)을 바꾸지 않고, 결합 시점에 Hot / Cold / other 중 어느 입력이 늦게 준비되는지와, Cold 가 늦다면 입력 전송·CPU 대기·CPU 계산·결과 반환 중 어디서 늦는지를 **같은 layer·step·결합 단위의 이벤트로 직접 측정**한다. CPU 시간·GPU 커널 시간의 크기만으로 병목을 판정하지 않는다.

## 범위
- 포함: M0 환경 고정 → M1 GPU union/gap 분석기 수정·재집계 → M2 시간창 정렬 → M3 상관 식별자·이벤트 → M4 구간별 수집 → M5 계측 정상성 → M6 OFF/ON 반복 측정 → M7 지연 분해 → M8 GLM 조건부 → M9 보존·게시.
- 제외: 성능 최적화(워커 수·hot 배치·deferred·backend·정밀도·chunk 변경), 전량 CPU 비교군, GLM INT4 변환.
- 상한: 24 세션·부팅 12 회 (지시서 §9.2). 예산 소진을 목표로 실행하지 않는다.

## 이론적 배경 (측정 정의)
- GPU busy 는 stream 별 duration 합이 아니라 **interval union** (§4.2). annotation 은 busy 에 더하지 않는다. 장치별 집계, TP rank 합산 금지.
- 결합 지연: `cold_ready_lateness = max(0, t_cold_ready − max(t_hot_ready, t_other_ready))` 등 (§10.1). 필수 이벤트·clock 정렬이 없으면 계산하지 않고 `PARTIAL_DEPENDENCY_MEASUREMENT` / `UNMEASURED_CLOCK_ALIGNMENT` 로 남긴다.
- 계측 간섭: 같은 PROBE128 의 OFF(P0) 대비 ON(P1/P2) 비율만 보고. 허용 기준은 사전에 정하거나 비율·한계만 보고.

## 산출물
`PLAN_RESOLVED.md`, `SOURCE_INDEX.md`, `CONFIG_DIFF.md`, `WORK_LOG.md`, `profiles/{probe_map.md, corrected_metrics.csv, observer_windows.csv, observer_overhead.csv, dependency_metrics.csv, missing_coverage.csv, validation_results.json}`, `BOTTLENECK_EVIDENCE.md`, `FULL_REPORT.md`, `FULL_RAW_DATA.md`, `COMPLETION_STATUS.md`, `PUBLISH_RECEIPT.md`. 결과 원자료: `eval/results/IDE_074_20260917/`. 하네스·분석기: `eval/ide074/`.
