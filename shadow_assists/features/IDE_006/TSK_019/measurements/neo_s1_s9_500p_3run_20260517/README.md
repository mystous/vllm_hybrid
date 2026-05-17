# NEO S1-S9 정통 (NEO 원본 source 10/10 함수 정합) 500p × 8192 3-run sequential (2026-05-17)

> S1-S9 = NEO 원본 정통 rewrite 9 단계 적용 후 측정.
> v1.6 commit `64f9e0c48` base + feat/neo-option-b branch.
> NEO 원본 source code (`swiftllm/worker/layers/transformer_layer.py` + `model.py`) 의 10/10 함수 정합 implement. paper §4.4 batch interleave 영역도 S8 forward_double ordering 으로 정합.

## 결과 (3-run 완료)

| Run | output_tps | wall (s) | cdec_wait final | b1_avg | shape_mismatch | crash |
|---|---:|---:|---:|---:|:-:|:-:|
| **run 1** | **2,303.4** | **1,763** | 0.00 ms (직접 호출) | 7 | 0 ✓ | 0 ✓ |
| run 2 | 2,153.6 | 1,889 | 0.00 ms | 6 | 0 ✓ | 0 ✓ |
| run 3 | 2,258.9 | 1,806 | 0.00 ms | 7 | 0 ✓ | 0 ✓ |
| **avg** | **2,238.6** | **1,819** | 0.00 ms | 6.7 | 0 | 0 |
| **min** | 2,153.6 | 1,763 | — | 6 | 0 | 0 |
| **max** | 2,303.4 | 1,889 | — | 7 | 0 | 0 |
| std | 76.9 | 63.5 | — | — | — | — |
| **CV** | **3.44%** | 3.49% | — | — | — | — |

vs v1.6 best 3-run avg (2,197.4): **+1.9%**.
vs vanilla 3-run avg (4,690.7): **47.7%**.

## NEO 원본 100% 정합 구현 (S1-S9)

| step | 영역 | 적용 결과 |
|---|---|:-:|
| S1 | `_neo_comm_wait_compute` / `_neo_compute_wait_comm` helper 추가 (NEO `_comm_wait_compute` / `_compute_wait_comm` 동등) | ✓ |
| S2 | `_get_neo_communication_stream` 이미 있음 | ✓ |
| S3 | Option B (async deque) path 제거 — Option A sync path 만 유지 | ✓ |
| S4 | `forward_double` 의 `with cuda.stream(s0/s1):` 제거 — NEO 원본 main stream 정합 | ✓ |
| S5 | `ThreadPoolExecutor.submit` → `_neo_cdec_compute_cpu(...)` 직접 호출 + `_NeoDirectFuture` wrap. NEO 원본 `paged_attention_cpu` 직접 호출 정합 | ✓ |
| S6 | `qkvtr_e` event 영역 — 이미 `_xfer_stream.record_event()` 사용 중 | ✓ |
| S7 | `_get_batch_streams()` (s0/s1 stream pair) dead code 제거 | ✓ |
| S8 | `forward_double` ordering NEO `_forward_pipeline_stage(cur_stage)` 정합 — batches[cur_stage] postproj+preproj 가 batches[other] attention 보다 *먼저* 실행 | ✓ |
| S9 | result D2D copy 를 `cpu_communication_stream` 위 async + `_neo_compute_wait_comm()` 호출 정합 | ✓ |

→ NEO 원본 `swiftllm/worker/layers/transformer_layer.py` 의 **10/10 함수 정합** ✓.

## 다른 측정과의 비교

| 측정 (500p × 8192) | 3-run avg | 3-run CV | min | max |
|---|---:|---:|---:|---:|
| vanilla (NEO OFF) | 4,690.7 | 0.006% | 4,690.4 | 4,691.0 |
| **NEO S1-S9 (commit `feat/neo-option-b`)** | **2,238.6** | **3.44%** | 2,153.6 | 2,303.4 |
| NEO v1.6 (commit `64f9e0c48`) | 2,197.4 | 1.62% | 2,156.9 | 2,223.8 |
| NEO Phase 3.1 only (KMP=200) | 2,134.9 | 5.68% | 2,013.2 | 2,255.7 |
| NEO Phase 3.1+3.3 (KMP=200) | 2,083.3 | 3.13% | 2,015.4 | 2,145.4 |

→ **S1-S9 = 모든 NEO 측정 중 best avg** (+1.9% vs v1.6 best). CV 3.44% (v1.6 1.62% 보다 큼).

## fact

| 항목 | 값 |
|---|---|
| commit | `feat/neo-option-b` (v1.6 base `64f9e0c48` + S1-S9 변경) |
| 측정 시각 (KST) | 2026-05-17 14:25:33 → 16:04:48 |
| 3-run summary | `eval/results/20260517_142533_S1S9_500p_3run/SUMMARY.txt` |
| 측정 dir | `eval/results/20260517_142533_neo_standard/` (run1), `20260517_145648_neo_standard/` (run2), `20260517_153134_neo_standard/` (run3) |
| launch script | `/tmp/S1S9_500p_3run.sh` → `run_neo_standard.sh` × 3 |
| KMP_BLOCKTIME | unset → 200 default |
| pacpu .so | v1.6 시점 빌드 (S1-S9 모두 Python only) |

### workload (3 run 동일)

| | |
|---|---|
| num_prompts | 500 |
| target_input_len / max_tokens | 8,192 / 8,192 |
| model | meta-llama/Llama-3.3-70B-Instruct |
| TP | 8 (H100 80GB × 8) |
| gpu_memory_utilization | 0.92 |
| kv_cache_dtype | fp8 |

## 의미

1. **NEO 원본 source code 10/10 함수 정합 implement 완성** ✓ (paper §4.4 batch interleave 영역도 S8 으로 정합)
2. v1.6 best (Option A 영역) 대비 throughput **+1.9% avg, +3.6% best individual**
3. **paper claim H100 +14% 도달 X** — 원인:
   - 우리 vanilla 가 paper baseline 보다 빠름 (vllm 최적화 누적)
   - 500p × 8192 long context 가 NEO 의 sweet spot (작은 batch / 짧은 context) 와 다름
   - H100 80GB × 8 = 640 GB HBM 영역에서 NEO 의 KV offload 의 *batch 확장* 의도 영향 작음
4. 22 strict 일관 (shape_mismatch 0, crash 0) ✓

## 파일

| file | 내용 |
|---|---|
| `SUMMARY.txt` | 3-run wrapper 의 timeline + 평균/min/max/CV |
| `run{1,2,3}/result.json` | run 별 throughput + wall |
| `run{1,2,3}/metrics.log` | PROFILE/FORK/SWAP/CDEC/mismatch grep |
| `run{1,2,3}/engine.log.stdout.gz` | full stdout gzipped (4-7 MB) |
