# g0_00_qwen2.5_32b_base — TP=8 표준 baseline (§06 off)

## 정체성

본 디렉토리는 **현 시점 표준 baseline** 이다. Qwen2.5-32B × H100x8 × TP=8 × 500 req × 128/128 workload 에서 **모든 Ninja Gap feature flag 를 off** 로 한 상태의 hybrid 측정 + gpu_only 대조군. 앞으로의 모든 기법 비교는 이 디렉토리를 기준으로 한다.

- 측정일: 2026-04-20
- Git: main 브랜치 (§06 코드 merge 된 상태, `HYBRID_VNNI_HOT_PATH=0`)
- env: `g0_h100x8_qwen32b_00_tp8.env`
- 주의: `§06` 코드는 main 에 존재하지만 flag off → `§06 disabled` 로그, functional 동작은 §06 이전과 동일 (patch 경로 조기 return)

## 구조

```
g0_00_qwen2.5_32b_base/
├── analysis_g0.ipynb            # g0_06_qwen2.5_32b 노트북 재사용, ROOT 자동 감지
├── analysis_bench.png
├── analysis_cpu_heatmap.png
├── analysis_gpu_power_mem.png
├── analysis_util_timeseries.png
├── g0_h100x8_qwen32b_00_tp8.env # 측정 env snapshot
├── gpu_only_baseline/           # TP=8 gpu_only 대조군
└── seqs{1,2,4,8,16,32,64}/      # hybrid sweep (§06 off)
```

## 용도

- `g0_06_qwen2.5_32b/` (§06 on, TP=8) 과 단일 flag (`HYBRID_VNNI_HOT_PATH`) 차이로 **§06 단독 이득 직접 측정**
- 앞으로의 모든 비교 baseline 으로 사용. 단, **2026-04-20 기준 `§11`은 Phase 1 기각, `§18`은 우선순위 강등** 상태이므로, 직접 비교 대상은 최신 `TODO.md` / `NinjaGap_Todo/README.md` 의 Tier 1 후보 우선순위를 따른다
- wall ratio, batch scaling ratio 모두 여기 수치를 기준으로 계산

## 관련 디렉토리

- `../g0_00_qwen2.5_32b_tp4/` — TP=4 과거 snapshot (비교군 아님, 역사적 기록)
- `../g0_06_qwen2.5_32b/` — §06 Q8_0 hot path on 측정 (같은 TP=8, 단일 flag 차이)

## 관련 문서

- §06 이력: `Task_done.md v6` / `Tech_done.md v6` (단, 이후 `v7/v8` 에서 정정 및 SSOT 갱신)
- §06 기법 문서: `NinjaGap_Todo/06_hot_path_wiring.md`
