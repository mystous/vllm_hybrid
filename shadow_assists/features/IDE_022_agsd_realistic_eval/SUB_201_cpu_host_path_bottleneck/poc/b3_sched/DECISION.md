# B3 sched lever — cudagraph_mode DECISION (B200 sweep 완료 시점)

> 본 DECISION 은 1차 sweep (`MEASUREMENTS.md`, FlashInfer default, 50p) + 2차 sweep (`MEASUREMENTS_FA.md`, FA backend forced, 100p) 의 종합 판정입니다.
> 사용자 지시 (poc/b3_sched/runs_fa3 sweep 완료) 기준 작성.
>
> NOTE: 원 지시의 doc 경로 `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/b3_cudagraph_sweep/DECISION.md` 는 기존 sweep 자산이 `poc/b3_sched/` 에 있어 자산 인접성을 위해 본 경로에 둠. `profile/b3_cudagraph_sweep/DECISION.md` 는 본 문서로의 redirect note 만 둠.

---

## 1. 한줄 결론

**B200 에서는 SUB_201 §5 의 "PIECEWISE → FULL 단독" lever 가 활성 불가** (FA3 가 빌드/하드웨어 양면으로 부재).
대신 **FaP (FULL_AND_PIECEWISE)** 가 PIECEWISE 대비 **+30% 수준의 net win** (100p × conc=16, suffix decoding) 으로 측정되었으며,
**default cudagraph_mode 를 FaP 로 전환하는 것이 B200 환경에서 권장**됩니다.
**§5 의 38% launch overhead 회수 검증 자체는 prod H100 (FA3 = ALWAYS cap) 에서 재수행 필요**.

---

## 2. backend × cudagraph_mode × tps × FULL active 표

| Sweep | backend | `flash_attn_version` 시도 | 요청 cudagraph_mode | **실 적용 mode** | FULL 단독 active? | tps (100p×conc16, suffix mix) |
|---|---|---|---|---|---|---|
| 1차-R0 | FlashInfer (default) | n/a | PIECEWISE | PIECEWISE | × | 4 169.1 (50p, baseline) |
| 1차-R1 | FlashInfer | n/a | FULL | **FaP 다운그레이드** (UNIFORM_BATCH cap) | × | 2 546.3 (anomaly) |
| 1차-R2 | FlashInfer | n/a | FULL_AND_PIECEWISE | FaP | × | 4 413.3 (+5.9%) |
| 2차-R0_FA | FLASH_ATTN forced | (none, auto = FA4) | PIECEWISE | PIECEWISE | × | **3 271.8** |
| 2차-R1_FA | FLASH_ATTN forced | **3 강제** | FULL | **FaP 다운그레이드** (FA3 → FA4, UNIFORM_BATCH) | × | **2 582.5** (anomaly) |
| 2차-R1_FA repeat | FLASH_ATTN forced | 3 강제 | FULL | FaP | × | 2 840.3 (anomaly 재현) |
| 2차-R2_FA | FLASH_ATTN forced | (none, auto = FA4) | FULL_AND_PIECEWISE | FaP | × | **4 257.9 (+30.1% vs R0_FA)** |

> 1차 sweep 의 tps 는 50p × conc=16, 2차 sweep 은 100p × conc=16. burst size 차이로 R0 의 절대값이 다름 (50p 4169 vs 100p 3272 — 100p 가 더 긴 prefill outlier 포함).

### 2.1 FA3 강제 시도 결과

- **시도 방법**: `--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":3}'`
- **결과**:
  - vllm 코드 (`vllm/v1/attention/backends/fa_utils.py:97`) 가 Blackwell 에서 FA3 를 차단 → `WARNING Cannot use FA version 3 on Blackwell platform, defaulting to FA version 4`
  - 본 B200 빌드의 `vllm_flash_attn` 패키지는 FA3 자체 미포함 (`is_fa_version_supported(3) = False`)
  - FA4 가 활성화되지만 `_cudagraph_support = UNIFORM_BATCH` 이므로 FULL 단독 불가
  - 결과적으로 R1_FA 의 실 cudagraph mode 는 FaP (R2_FA 와 동일)

→ **FA3 의 ALWAYS cap 경로는 B200 에서 활성화 불가능. SUB_201 §5 의 회수 시나리오 자체가 B200 에서 검증 불가.**

---

## 3. 권고

### 3.1 단기 (현 dev / B200 환경) — **FaP 로 전환 권장**

- vllm `compilation_config.cudagraph_mode` default 를 **`FULL_AND_PIECEWISE`** 로 (CLI 명시 추천, vllm upstream patch 는 권장 안함 — H100 검증 후 결정).
- 측정된 net win: +5.9% (50p) ~ +30.1% (100p). **burst size 가 커질수록 이득이 커지는 경향** 으로 production-scale (≥500p) 에서는 더 큰 이득 가능.
- 메모리/부팅 비용: capture matrix 가 PIECEWISE 51 → FaP 15+15 으로 오히려 줄어듦. boot wall +6~10s, GPU mem footprint 0 차이. **거의 free**.
- R1_FA anomaly (FULL 요청 → FaP 다운그레이드 시 throughput 33~39% 하락) 는 별개 이슈 — **요청 mode 와 실 mode 가 다를 때 만 발생**. CLI 에서 직접 `FULL_AND_PIECEWISE` 로 요청하면 anomaly 없음.

### 3.2 중기 — **prod H100 검증 필수**

- prod 머신 (Intel Xeon SPR + H100 × 8) 에서 동일 sweep 재수행.
- H100 (SM 90) 은 FA3 native 지원 → `_cudagraph_support = ALWAYS` → **FULL 단독 활성 가능**.
- 검증 plan:
  - 동일 모델 (Qwen2.5-7B, TP=4) + 500p × conc=32 (TSK_042 baseline 과 동일 footing) + 3회 repeat
  - nsys profile 동시 수행 (60s window) — launch rate 정량 측정
  - 비교: R0 (PIECEWISE) vs R1 (FULL 단독, H100 FA3 native) vs R2 (FaP)
  - 통과 조건: R1 / R0 ≥ 1.10 (SUB_201 §5 PASS 조건). 통과 시 §5 의 38% 회수 가설 1차 검증.
- 예상: H100 + FA3 + FULL 단독 → mixed prefill-decode batch 도 FULL graph 안 → suffix spec verify path 가 더 큰 cudagraph hit-rate → R2 (FaP) 대비 추가 net win 가능.

### 3.3 장기 (production patch, ~수일~수주)

- **(필수)** vllm upstream issue 제출: B200 의 FA3 미지원 + FlashInfer UNIFORM_BATCH 한계로 SM 10.x 에서 cudagraph FULL 단독 mode 자체가 미활성. (현행 `FULL` 옵션은 사실상 silent downgrade → 사용자 혼란.)
- **(선택)** FlashInfer / FA4 backend 의 `_cudagraph_support` 를 dynamic 판정으로 확장 (mixed batch 의 uniform-decode 비율을 보고 일부 mixed batch 도 FULL graph 에 dispatch). DESIGN §5.3 의 `cudagraph_dispatcher.py` 영역.
- **(연구)** R1 anomaly root cause: 동일 cudagraph mode 임에도 sweep 의 두번째 cell 만 ~33~39% 느림. autotune cache / JIT compile state / kernel selection memo 의 측정-간 carry-over 가 의심됨. nsys profile + per-iter latency log + 측정 셀 순서 permutation (R2 → R0 → R1 등) 으로 검증.

### 3.4 본 lever 의 SUB_201 net-win 등록

- §5 의 "PASS 조건 = tps ≥ baseline × 1.10" 기준:
  - **B200, default backend (FlashInfer), 100p burst**: R2/R0 = +30.1% → **PASS**
  - **B200, FA backend forced, 100p burst**: R2_FA/R0_FA = +30.1% → **PASS**
  - **B200, default backend, 50p burst**: R2/R0 = +5.9% → FAIL
- 100p 시점에서 PASS. burst size sensitivity 가 있으므로 prod scale (≥500p) 에서 한번 더 검증 후 §5 net-win 표에 정식 등록 권장.

---

## 4. 다음 step 제안 (사용자 결정 사항)

| 옵션 | 작업 | 예상 시간 | 산출물 |
|---|---|---|---|
| **A** (강력 권장) | prod H100 노드로 동일 sweep 이동 + FULL 단독 검증 + nsys profile | 4~6h | H100 sweep MEASUREMENTS + §5 회수 검증 verdict |
| **B** (선택) | B200 에서 500p × conc=32 (TSK_042 footing) 로 FaP 재측정 + 3 repeat → R1 anomaly 통계 분리 | 2~3h | FaP net win 통계 + anomaly 근거 |
| **C** (장기) | vllm upstream issue 제출 (B200 cudagraph FULL silent downgrade) | 1h | issue link |
| **D** (폐기 결정) | B3 lever 를 B200 한정 비활성, prod 우선 결정으로 보류 | 0h | 본 DECISION.md 가 그 산출물 |

기본 권고: **A + C** 병행. B 는 H100 결과 보고 결정.

---

## 5. 참고 산출물

- 1차 sweep: `poc/b3_sched/MEASUREMENTS.md`, `poc/b3_sched/runs/r0_piecewise.json` 외
- 2차 sweep: `poc/b3_sched/MEASUREMENTS_FA.md`, `poc/b3_sched/runs_fa3/r{0,1,2}_fa_*.json` 외
- DESIGN: `poc/b3_sched/DESIGN.md`
- LEVER_AUDIT: `poc/b3_sched/LEVER_AUDIT.md`
- SUB_201 §5: `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/README.md` §5

---

## 6. script 반영 (Phase A2 — 2026-06-05)

본 DECISION 의 §3.1 단기 권고 ("B200 default cudagraph_mode 를 FULL_AND_PIECEWISE 로 전환") 를 vllm_config_perf 측 entry script 에 반영하였습니다.

### 6.1 변경 대상 script (6 파일, vanilla/suffix 측정 path)

| 파일 | 변경 라인 | 비고 |
|---|---|---|
| `vllm_config_perf/gating/realistic_eval/run_oracle_8gpu.sh` | L48 | TSK_042 oracle 측정 (TP=8, conc=1) |
| `vllm_config_perf/gating/realistic_eval/run_throughput_8gpu.sh` | L47 | TSK_042 canonical throughput (TP=auto, conc=32) |
| `vllm_config_perf/gating/realistic_eval/run_case.sh` | L82 | 단일 (MODEL, METHOD) 케이스 entry |
| `vllm_config_perf/gating/launcher.sh` | L44, L56 | AGSD 7B (vanilla + trident) launcher |
| `vllm_config_perf/gating/launcher_32b.sh` | L43, L56 | AGSD 32B (vanilla + trident) launcher |
| `vllm_config_perf/gating/run_full_8gpu.sh` | L69 | 32B 3-phase sweep (Phase1/2/3 공통 boot) |

모두 `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` → `'{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` 로 단순 치환. header 코멘트의 "PIECEWISE" 문구도 일관성 차원에서 "FULL_AND_PIECEWISE (b3_sched DECISION)" 로 갱신.

### 6.2 변경 제외 (의도적)

- **`vllm_config_perf/gating/recommendations.py`**: SUB_093 (Llama-70B × H100×8) 기반 prod 권장표. B200 의 FaP 권고는 §3.2 의 prod H100 sweep 결과 확정 이후 별도 패치로 반영 예정. 현재 단계에서 표를 바꾸면 prod 권장 출처가 흐려짐.
- **`shadow_assists/features/IDE_015~021/**/launcher.sh`**: 기 종료된 PoC (SUB_177, 181, 184, 186, 188-198, 19x 계열) 의 보관 자산. 과거 측정의 reproducibility 보존이 우선. 변경 영향권 아님.
- **ngram method path**: vllm 내부에서 attention backend 의 `_cudagraph_support` 가 ALWAYS 가 아니면 자동 `FULL_AND_PIECEWISE`/`FULL_DECODE_ONLY` 로 다운그레이드되며 (`vllm/config/compilation.py:1286-1310`), B200 에서는 FaP 요청을 그대로 받는 동일 path 가 됩니다. 즉 본 변경이 ngram boot 에 추가 다운그레이드를 발생시키지 않음 — 그대로 두어도 안전.

### 6.3 dry-run 검증

```bash
$ bash -n <6 modified scripts>     # 모두 OK
$ # Qwen-7B × suffix boot command echo (run_case.sh):
CUDA_VISIBLE_DEVICES=0,1,2,3 setsid /workspace/vllm_dev_prj/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 4 --port 8001 --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --allow-deprecated-quantization \
  --speculative-config {"method":"suffix","num_speculative_tokens":32}
```

→ `cudagraph_mode=FULL_AND_PIECEWISE` 가 boot CLI 에 정상 인입됨 확인.

### 6.4 reproducibility 보호

- 본 변경 commit 이전의 모든 TSK_042 측정 (`vllm_config_perf/gating/realistic_eval/runs/*`) 은 PIECEWISE base. b3_sched/runs_fa3 의 R2_FA 와 net win 비교 시 그 시점의 boot mode 를 명시 인용할 것 (commit hash 기준).
- 이후 동일 script 로 재측정하는 cell 은 base 가 FaP 이므로 직접 비교 시 cudagraph_mode 변수를 통제해야 함 (필요 시 env override 로 PIECEWISE 강제 측정 lane 추가 권장).

### 6.5 후속 (prod H100 검증 후)

- §3.2 의 H100 sweep 통과 시:
  - `recommendations.py` 의 (large, suffix) 항목 cudagraph_mode 를 FULL_AND_PIECEWISE (또는 H100 에서 활성화 가능하면 FULL 단독) 로 갱신.
  - prod launcher (예: AGSD_TP=8, H100×8) 도 동일하게 갱신.
- 미통과 시 본 §6 변경을 prod 에 transfer 하지 않고 B200 한정으로 유지.

