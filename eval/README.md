# `eval/` — 검증 / 측정 wrapper 모음

본 디렉토리는 IDE_006 / TSK_*** 회차의 prod·dev 검증 / 측정용 wrapper 들을 묶는다. 각 wrapper 는 목적·소요 시간·수집 데이터·통과 기준이 다르므로 단계에 맞는 것을 골라 사용한다.

> **권한**: 모든 wrapper 는 `bash eval/<wrapper>.sh` 형태로 실행. 결과는 `eval/results/<TS>_<HW_TAG>_<wrapper-tag>/` 에 저장. `--push` 옵션이 있는 wrapper 는 자동 commit + push.

---

## Wrapper 매트릭스

| wrapper | 목적 | 소요 시간 | cold path 발화 | 자동 push | 주요 산출물 |
|---|---|---|---|---|---|
| [`run.sh`](run.sh) | 단일 시나리오 (env 1 개) 의 server + monitor + bench | ~2-5 분 | env 에 따라 (split_on env 면 발화 가능) | ❌ | `bench.json`, `monitor_cpu.csv`, `monitor_gpu.csv`, `server.log` |
| [`run_prod_simd_verify.sh`](run_prod_simd_verify.sh) | TSK_003 SIMD numerical correctness + e2e_quick (--max-prompts 8) | ~8-10 분 | ❌ (8 prompt 는 GPU KV pool 의 ~10% 만 차지, eviction 미트리거) | ✅ (`--push`) | `tst004_pytest.log`, `tst004_junit.xml`, `e2e_quick.log` |
| [`run_prod_cold_verify.sh`](run_prod_cold_verify.sh) | IDE_006 cold path 실 발화 검증 (NUM_PROMPTS=100, OUTPUT_LEN=16, monitor.py) | ~3-5 분 | ✅ (KV pool 한계 위 워크로드) | ✅ (`--push`) | `e2e.log`, `monitor_cpu.csv`, `monitor_gpu.csv`, README 의 발화 카운트 + CPU/GPU avg util |
| [`run_prod_smoke.sh`](run_prod_smoke.sh) | TST_001 + TST_004 + baseline + cold + split + **D-i / D-ii** (TST_003 정합성 게이트) | ~15-20 분 | ✅ | ✅ (`--push`) | 위 모두 + `comparison.json` (D-i / D-ii verdict) |

## 단계별 권장 흐름

| 진입 시점 | 권장 wrapper | 의도 |
|---|---|---|
| TSK_003 SIMD kernel 작성 직후 / fix 후 numerical 회귀 의심 | `run_prod_simd_verify.sh --push` | 80 case cross-check 빠르게 회전 |
| TSK_002 §4.6 / TSK_004 회귀 fix 후 cold path 동작 / 발화 확인 | `run_prod_cold_verify.sh --push` | KV pool 한계 위 워크로드로 cold path 실 발화 + monitor 시계열 |
| 코드 변경 일단락 후 정합성 본 검증 (D-i / D-ii) | `run_prod_smoke.sh --push` | baseline vs split_on 의 token 분포 + logprob 비교, IDE_006 의 **GPU 동등성** 게이트 |
| 단일 시나리오 단발 측정 (예: 새 env 로 throughput 가늠) | `run.sh <env>` | 빠른 단발 server-mode 측정. `eval/envs/` 의 env 파일 1 개 인자 |

## 환경변수 매트릭스

| env | 효과 | default | 권장 |
|---|---|---|---|
| `VLLM_AMX_TRACE` | Python `_call_amx` per-call entry/exit/elapsed 트레이스 + permission grant 결과 | OFF | 디버그 회차에서만 ON |
| `VLLM_PARTIAL_ATTN_PROFILE` | `_call_compiled_kernel` / `hot_cold_attention` 단면 wall-time 출력 (첫 128 회 / per-process cap) | OFF | overlap 측정 회차에서만 ON. `torch.cuda.synchronize` 가 들어가 정상 운영 측정에 부적합 |
| `VLLM_PARTIAL_ATTN_THREADS` | C++ helper `vllm_partial_attn_thread_count()` 의 thread 수 명시 override (자동으로 `min(env, sched_getaffinity_count)` clamp) | unset (affinity 만큼) | 보통 unset 유지. NUMA partition 디버그 시에만 |
| `VLLM_PARTIAL_ATTN_MAX_QLEN` | `hot_cold_attention` 의 q_len cap (배치의 `max_query_len > N` 이면 cold path 우회 — staged 검증용 escape hatch) | `-1` (OFF, 정합성 보존 모드) | 평상시 OFF. 켜면 mixed batch 에서 cold-with-decode 의 cold prefix 머지가 빠짐 (CLAUDE.md 위반 가능성) |
| `VLLM_COLD_KV_DISABLE_OVERLAP` | TSK_002 §4.6 의 cold-path GPU 작업용 dedicated CUDA stream 비활성. 단일 stream sequential 동작 강제 | OFF (overlap ON) | A/B 비교나 overlap 회귀 의심 시만 ON |

---

## wrapper 별 사용법

### `run.sh`

```bash
# IDE_006 split_on env 로 단일 회차 (server + bench + monitor)
bash eval/run.sh eval/envs/ide006_cold_kv_split_on_long_ctx.env

# baseline (IDE_006 OFF)
bash eval/run.sh eval/envs/vllm_original_long_ctx.env

# 워크로드 override
NUM_PROMPTS=200 OUTPUT_LEN=32 bash eval/run.sh eval/envs/ide006_cold_kv_split_on_long_ctx.env
```

산출물 위치: `eval/results/<TS>_<HW_TAG>_<MODEL>/`

### `run_prod_simd_verify.sh`

TSK_003 의 SIMD kernel (AVX-512 / AMX) numerical correctness 검증 + dispatcher 발화 정상 여부.

```bash
# 풀 (pytest TST_004 + e2e_quick) — ~8-10 분
bash eval/run_prod_simd_verify.sh --push

# pytest 생략, e2e_quick 만 — ~2-3 분
bash eval/run_prod_simd_verify.sh --skip-tst --push
```

`--max-prompts 8` 이 하드코딩되어 있어 cold path 실 발화는 트리거되지 않는다 — 그건 `run_prod_cold_verify.sh` 가 담당.

### `run_prod_cold_verify.sh`

IDE_006 의 cold path 가 실제로 fire 되는지 + per-seq 필터의 reduced row 수가 의도대로 나오는지 + monitor.py 시계열로 CPU/GPU util 변화 측정.

```bash
# default (NUM_PROMPTS=100, OUTPUT_LEN=16)
bash eval/run_prod_cold_verify.sh --push

# KV pool 경계 직전
NUM_PROMPTS=80 bash eval/run_prod_cold_verify.sh --push

# 더 짧게
OUTPUT_LEN=8 bash eval/run_prod_cold_verify.sh --push
```

산출물의 `README.md` 끝부분에 자동 요약:
- `IDE_006 dispatcher 라인: N 회` — cold path 발화 카운트 (per-process 첫 5 회 × 8 worker = 40 이 정상 ceiling)
- `bench 완료 라인` — `N prompts in M.M s`
- `CPU 사용률 평균 / GPU 사용률 평균` — monitor.csv 시계열의 mean

### `run_prod_smoke.sh`

풀 정합성 검증 — baseline vs split_on 의 D-i (token divergence) + D-ii (per-position logprob max abs diff + sequence PPL relative diff). IDE_006 의 **GPU 동등성** (CLAUDE.md "결과 값이 달라져서는 안됨") 최종 게이트.

```bash
bash eval/run_prod_smoke.sh --push
```

5 step 순서:
1. pytest TST_001 + TST_004
2. baseline scenario (`vllm_original_long_ctx.env`)
3. cold-tier scenario (`ide006_cold_kv_long_ctx.env`)
4. split_on scenario (`ide006_cold_kv_split_on_long_ctx.env`)
5. e2e accuracy — D-i + D-ii 비교

산출 디렉토리에 `comparison.json` 으로 D-ii verdict.

---

## monitor.py — CPU/GPU 사용률 시계열

[`eval/monitor.py`](monitor.py) 는 `psutil.cpu_percent` + `nvidia-smi` 로 1 초 (default) 간격 사용률을 CSV 로 기록한다. `run.sh` / `run_prod_cold_verify.sh` 가 background 로 spawn 하고 wrapper 종료 시 자동 정리.

직접 사용:
```bash
python eval/monitor.py /tmp/run1/monitor --interval 0.5 &
# ... bench / load 진행 ...
kill %1
```

산출:
- `<prefix>_cpu.csv` — `timestamp, cpu_util_pct, mem_used_gb, mem_total_gb`
- `<prefix>_gpu.csv` — `timestamp, gpu_index, gpu_util_pct, gpu_mem_util_pct, gpu_mem_used_mib, gpu_mem_total_mib, gpu_temp_c, gpu_power_w`

분석 시 awk 평균:
```bash
awk -F, 'NR>1 && $2!="" {sum+=$2; n++} END{printf "avg cpu util: %.1f%%\n", sum/n}' monitor_cpu.csv
awk -F, 'NR>1 && $3!="" {sum+=$3; n++} END{printf "avg gpu util: %.1f%%\n", sum/n}' monitor_gpu.csv
```

---

## envs/

`eval/envs/<scenario>.env` 들이 시나리오별 env 파일. `run.sh` / `run_prod_*.sh` 가 source 한다.

| env | IDE_006 | 용도 |
|---|---|---|
| `vllm_original_long_ctx.env` | OFF (`EXTRA_SERVE_ARGS=""`) | 14336 input × 128 output, 100 prompts, GPU only baseline |
| `ide006_cold_kv_long_ctx.env` | OffloadingConnector 만 (cold KV evicts to CPU but never read in place) | 중간 단계 — TSK_002 미적용 |
| `ide006_cold_kv_split_on_long_ctx.env` | OffloadingConnector + `enable_cpu_partial_attention=True` (full feature) | IDE_006 ON 풀 동작 |

baseline / IDE_006 비교는 `vllm_original_long_ctx.env` vs `ide006_cold_kv_split_on_long_ctx.env`. 두 env 의 실제 차이는 `GPU_MEMORY_UTIL` (0.9 vs 0.85) + `EXTRA_SERVE_ARGS` (kv-transfer-config) 두 줄. 다른 모든 워크로드 변수 (NUM_PROMPTS / INPUT_LEN / OUTPUT_LEN / MAX_MODEL_LEN / TP / REQUEST_RATE) 는 동일.

---

## 참고

- 회귀 fix 흐름 / 진단 인프라 / opt-out env 의 통합 기록: [`shadow_assists/features/IDE_006/PLN_001_TSK_002_02_overlap_fix_log.md`](../shadow_assists/features/IDE_006/PLN_001_TSK_002_02_overlap_fix_log.md)
- 단계별 ID 명세: [`shadow_assists/features/IDE_006/`](../shadow_assists/features/IDE_006/) 의 `README.md` / `PLN_001.md` / `TSK_*.md` / `TST_*.md`
- ID 할당 / 상태: [`shadow_assists/id_registry.md`](../shadow_assists/id_registry.md)
