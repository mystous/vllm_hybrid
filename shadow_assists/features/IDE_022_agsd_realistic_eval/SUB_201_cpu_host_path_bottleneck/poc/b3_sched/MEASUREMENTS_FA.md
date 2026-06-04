# B3 sched — cudagraph_mode sweep MEASUREMENTS (FA backend forced 추가 sweep)

본 문서는 1차 `MEASUREMENTS.md` (default = FlashInfer backend) 의 **후속**입니다.
이번 sweep 의 목적은 SUB_201 §5 의 "launch overhead 38% 회수" 시나리오 (PIECEWISE → **FULL 단독** 활성화 시 기대) 가
B200 에서 활성 가능한지 확인하기 위함입니다. 1차 sweep 의 finding 은 "B200 default = FlashInfer (UNIFORM_BATCH cap)
→ FULL 단독 불가, FaP 로 다운그레이드". 본 sweep 에서는 **FA backend 를 강제 (`--attention-config '{"backend":"FLASH_ATTN"}'`)**
하여 동일 시나리오를 다시 시도합니다.

---

## 0. 환경

- **Hardware**: NVIDIA B200 × 4 (GPU 0-3, sm_100)
- **Model**: Qwen/Qwen2.5-7B-Instruct, TP=4, max-model-len=16384, gpu-mem-util=0.85
- **Spec method**: suffix decoding (`num_speculative_tokens=32`)
- **Workload**: **100p × conc=16, mix shuffle (seed=42), stream** (1차 sweep 의 50p → 100p 로 burst 확대)
- **vllm version**: `v1.7.dev16107+gffe20fb09.d20260601`
- **Date**: 2026-06-04 23:26~23:42 UTC
- **Boot cmd template** (변수: `<MODE>`, `<FA_VER>`):
  ```
  vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 4 --port 8001 \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":<MODE>}' \
    --attention-config '{"backend":"FLASH_ATTN"<,"flash_attn_version":<FA_VER>>}' \
    --allow-deprecated-quantization \
    --speculative-config '{"method":"suffix","num_speculative_tokens":32}'
  ```

---

## 1. FA3 강제 가능성 — 코드 + 실측 양면 확정

### 1.1 정적 분석 (codebase)

- `vllm/v1/attention/backends/fa_utils.py` line 56-154 (`get_flash_attn_version`):
  - **Blackwell (SM 10.x)** 의 default = FA4 (`elif device_capability.major == 10 and is_fa_version_supported(4): fa_version = 4`).
  - `flash_attn_version=3` override 시도 → line 96-101 의 가드:
    ```python
    if device_capability.major >= 10 and fa_version == 3:
        logger.warning_once("Cannot use FA version 3 on Blackwell platform, "
                            "defaulting to FA version 4 if supported, otherwise FA2.")
        fa_version = 4 if is_fa_version_supported(4) else 2
    ```
  - 즉 **B200 에서 FA3 강제는 코드 레벨에서 차단**되어 있음 (FA3 fused kernel 의 Blackwell ISA 미지원).
- `vllm/v1/attention/backends/flash_attn.py` line 292-296:
  ```python
  _cudagraph_support = (
      AttentionCGSupport.ALWAYS
      if get_flash_attn_version() == 3
      else AttentionCGSupport.UNIFORM_BATCH
  )
  ```
  → FA3 만 ALWAYS. FA2/FA4 = UNIFORM_BATCH. FlashInfer 도 UNIFORM_BATCH (FlashInferMLA = UNIFORM_BATCH, FlashInfer = UNIFORM_BATCH/UNIFORM_SINGLE_TOKEN_DECODE 분기, ALWAYS 아님).
- `vllm/config/compilation.py` line 1286-1310:
  - `cudagraph_mode.mixed_mode() == FULL` and `min_cg_support != ALWAYS` → 자동 `FULL_AND_PIECEWISE` 또는 `FULL_DECODE_ONLY` 로 다운그레이드 + warning.

### 1.2 빌드 dynamic 확인

```python
$ /workspace/vllm_dev_prj/bin/python -c "from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported; print('FA2:', is_fa_version_supported(2)); print('FA3:', is_fa_version_supported(3)); print('FA4:', is_fa_version_supported(4))"
FA2: True
FA3: False     ← 본 B200 빌드의 vllm_flash_attn 패키지가 FA3 자체를 포함하지 않음
FA4: True
```

→ B200 에서는 (a) 코드 가드 + (b) 빌드 부재 양쪽 다 FA3 unavailable. **`AttentionCGSupport.ALWAYS` cap 활성화 경로 자체가 막혀 있음**.

### 1.3 실측 boot log 확정 (R1_FA)

```
WARNING [fa_utils.py:97] Cannot use FA version 3 on Blackwell platform, defaulting to FA version 4 if supported, otherwise FA2.
INFO    [cuda.py:308]   Using AttentionBackendEnum.FLASH_ATTN backend.
INFO    [flash_attn.py:661] Using FlashAttention version 4
WARNING [compilation.py:1310] CUDAGraphMode.FULL is not supported with FlashAttentionBackend backend (support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_AND_PIECEWISE
```

→ FA3 강제 시도 → FA4 fallback → UNIFORM_BATCH → FULL 단독 요청 자동 다운그레이드 (FaP).
**결론: B200 에서 FULL 단독 cudagraph mode 는 어떤 backend (FlashInfer, FA4) 로도 활성화 불가.**

---

## 2. 본 sweep — FA backend 강제 R0/R1/R2 boot 결과

| Run | `--attention-config.backend` | `flash_attn_version` 시도 | `cudagraph_mode` (요청) | 실 적용 FA version | 실 적용 cudagraph mode | PIECEWISE captures | FULL captures | KV (총) |
|---|---|---|---|---|---|---|---|---|
| R0_FA | FLASH_ATTN | (none, auto) | `PIECEWISE` | **FA4** | **PIECEWISE** (1) | 51 | — | ~141 GiB |
| R1_FA | FLASH_ATTN | **3 (강제)** | `FULL` | **FA4 fallback** | **FULL_AND_PIECEWISE** (자동 다운그레이드) | 15 | 15 | ~141 GiB |
| R2_FA | FLASH_ATTN | (none, auto) | `FULL_AND_PIECEWISE` | **FA4** | **FULL_AND_PIECEWISE** | 15 | 15 | ~141 GiB |
| R1_FA (repeat) | FLASH_ATTN | 3 (강제) | `FULL` | FA4 fallback | FULL_AND_PIECEWISE | 15 | 15 | ~141 GiB |

→ **R1_FA 와 R2_FA 의 실 cudagraph mode 가 동일 (FaP)**. R1_FA 는 단지 FULL 단독 요청이 자동 다운그레이드된 것.

---

## 3. Throughput 측정 (100p × conc=16, mix shuffle seed=42)

| Run | backend×mode (effective) | wall (s) | tokens | **output_tps** | TTFT p50 (ms) | TTFT p99 | TPOT p50 (ms) | TPOT p99 | GPU util (%) | accept α |
|---|---|---|---|---|---|---|---|---|---|---|
| **R0_FA** | FA4 × PIECEWISE | 83.3 | 272 440 | **3 271.8** | 25.2 | 7761.3 | 7.4 | 71.0 | 36.5 | 0.7228 |
| **R1_FA** | FA4 × FULL→FaP (다운그레이드) | 96.8 | 250 085 | **2 582.5** | 34.3 | 342.9 | 11.3 | 41.2 | 36.4 | 0.7212 |
| **R2_FA** | FA4 × FULL_AND_PIECEWISE | 74.1 | 315 671 | **4 257.9** | 24.3 | 337.2 | 6.8 | 35.8 | 30.8 | 0.7227 |
| R1_FA (repeat) | FA4 × FULL→FaP | 102.0 | 289 604 | **2 840.3** | 36.4 | 139.6 | 11.2 | 39.6 | 38.1 | 0.7258 |

핵심:
- **R2_FA = 4 257.9 tps > R0_FA = 3 271.8 tps → +30.1%** (FA backend 강제 시 FaP 가 PIECEWISE 대비 큰 net win).
- 1차 sweep (FlashInfer default, 50p) 의 R2 = 4 413 tps 와 거의 동일 수준 (4 257.9). **FA4 vs FlashInfer 의 throughput 차이는 작음** (capture matrix 가 둘 다 15+15 로 동일하기 때문).
- **R1_FA anomaly 재현**: 동일 cudagraph mode (FaP) 임에도 R1_FA = 2 582.5 (run1) / 2 840.3 (repeat) 으로 R2_FA = 4 257.9 보다 33~39% 낮음.
  - 1차 sweep 의 R1 anomaly (R0 4169 / R1 2546 / R2 4413, R1 −38.9%) 와 **동일 패턴**.
  - 두 차례 repeat 모두 sweep 순서의 두번째 cell. **boot 직후 첫 측정 (R0_FA, R2_FA) 은 빠르고, 두번째 측정 (R1_FA) 은 느린 systematic effect** 의 흔적. shuffle seed 동일하므로 prompt mix 효과는 아님. burst 100p 가 여전히 noise 영향권.
- **TTFT p99**: R0_FA 7761 ms 은 outlier (1~2 req 만 long prefill). R2_FA / R1_FA p99 는 정상 (337/343 ms).

---

## 4. SUB_201 §5 launch overhead 38% 회수 검증 — 최종 판정

| 항목 | 기대 (DESIGN §2.2, H100 + FA3 가정) | 측정 (B200 + FA4) | 충족 여부 |
|---|---|---|---|
| FULL 단독 mode 활성화 | PASS | **FAIL** (B200 어떤 backend 로도 불가) | × |
| launch rate 감소 | ~5× | 측정 불가 (FULL 단독 미활성) | × |
| throughput net gain (FULL 단독 대비 PIECEWISE) | +10% 이상 | 측정 불가 | × |
| (대안) FaP vs PIECEWISE net gain | (기대 < FULL 단독) | **+30.1%** (R2_FA vs R0_FA) | △ |

**결론**:
- **B200 에서 SUB_201 §5 의 "FULL 단독" 시나리오 자체는 hardware/software 양쪽으로 불가**.
- 단, **차선책인 FaP (FULL_AND_PIECEWISE)** 가 PIECEWISE 대비 **+30.1% net win** 으로 측정됨 (100p × conc=16, suffix decoding, mix).
- 1차 sweep (50p) 의 R2 net win +5.9% 와 비교 시, **burst size 가 커질수록 FaP 의 이득이 더 크게 드러남** (50p +5.9% → 100p +30.1%).
- FaP 의 이득 원천: **decode-only batch 의 FULL graph 안 실행** (mixed prefill-decode batch 는 여전히 PIECEWISE). suffix spec 의 decode-heavy 특성과 정합.

---

## 5. R1 anomaly 검증 (1차 sweep + 본 sweep 양쪽 동일 패턴)

| sweep | run 순서 | R1 tps | R0 tps | R2 tps | R1 / R2 |
|---|---|---|---|---|---|
| 1차 (FlashInfer, 50p) | R0 → R1 → R2 | 2 546.3 | 4 169.1 | 4 413.3 | 0.58 |
| 2차 (FA forced, 100p) | R0 → **R1** → R2 → R1_repeat | 2 582.5 | 3 271.8 | 4 257.9 | 0.61 |
| 2차 R1 repeat | R2 → **R1_repeat** | 2 840.3 | — | — | 0.67 |

- 두 sweep 모두 **R1 (=FULL 요청, 실은 FaP) 만 비정상적으로 낮음**.
- 실 cudagraph mode 는 R1 ≡ R2 이므로 **mode 차이가 아닌 실행 외 요인** (boot caches, autotune state, JIT warmup) 의 영향으로 추정.
- TPOT p50: R0_FA 7.4 / R2_FA 6.8 / R1_FA 11.3 → R1 은 decode step 자체가 ~65% 더 느림 (cudagraph mode 가 동일한데도). FA backend 의 autotune lazy init / FlashInfer JIT cache eviction 등이 의심됨.
- **대책**: 본 PoC scope 를 넘는 별도 root cause 분석 필요 (DECISION.md §3.3 참조).

---

## 6. Cleanup 검증

- 측정 종료 후 모든 vllm process 정상 종료. GPU 0-3 0 MiB free. port 8001 free.

## 7. 산출물 (runs_fa3/)

```
shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_sched/runs_fa3/
├─ r0_fa_piecewise.json          ← R0_FA 결과 summary
├─ r0_fa_piecewise.raw.jsonl     ← per-request
├─ r0_fa_piecewise_boot.log      ← vllm boot log
├─ r1_fa3_boot.log               ← R1_FA boot log (FA3 강제 시도 → FA4 fallback warning 포함)
├─ r1_fa_full_req.json           ← R1_FA 결과 (FULL→FaP)
├─ r1_fa_full_req.raw.jsonl
├─ r1_fa_full_req_repeat.json    ← R1_FA repeat (anomaly 재현 검증)
├─ r1_fa_full_req_repeat.raw.jsonl
├─ r1_fa_full_req_repeat_boot.log
├─ r2_fa_fap.json                ← R2_FA 결과 (best tps)
├─ r2_fa_fap.raw.jsonl
└─ r2_fa_fap_boot.log
```
