# g0_00_32b_tp4 — TP=4 과거 snapshot (비교군 아님)

본 디렉토리는 **§06 이전 시점 (2026-04-15 ~ 04-17) 의 baseline 측정** 이며 `TENSOR_PARALLEL_SIZE=4` 로 찍혔다. env 파일 (`g0_h100x8_qwen32b.env`) 에 `TENSOR_PARALLEL_SIZE=4` 박제되어 있어 재현 가능. 2026-04-20 에 `g0_00_32b` → `g0_00_32b_tp4` 로 rename (TP=8 표준 baseline 과 이름 충돌 회피).

## 앞으로의 G1/G2/G3 비교에는 사용하지 않는다

TP=4 라 §06 이후의 TP=8 측정과 wall ratio 직접 비교가 성립하지 않는다. 현재 표준 baseline 은 아래로 이동:

- **TP=8 표준 baseline (§06 off, 모든 flag off)**: `../g0_00_32b_base/`
- **§06 on 실측 sweep** (TP=8): `../g0_06/`
- 앞으로의 모든 기법 비교는 `g0_00_32b_base` 를 기준으로 한다

## 유지 목적

Ninja Gap 초기 분석의 역사적 snapshot. `analysis_g0.ipynb` 와 4 PNG 는 당시 분석 그대로 보존. 향후 기법 측정 결과와 직접 교차 비교하지 말 것.

## 관련 기록

- §06 완료 이력 및 TP 변경 배경: `Task_done.md` v6 / `Tech_done.md` v6
- §06 기법 문서: `NinjaGap_Todo/06_hot_path_wiring.md`
