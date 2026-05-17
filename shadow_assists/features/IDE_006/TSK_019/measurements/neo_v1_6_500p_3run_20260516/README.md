# NEO v1.6 best 500p 3-run sequential (2026-05-14 + 2026-05-16)

> v1.6 commit `64f9e0c48` 의 3-run avg/min/max — 본 plan 의 최고 throughput 측정.
> Run 1 (기존, 2026-05-14) + Run 2/3 (2026-05-16 회귀 측정) 합산.

## 결과 (3-run 완료)

| Run | output_tps | wall (s) | cdec_wait final | shape_mismatch | 측정 dir |
|---|---:|---:|---:|:-:|---|
| run 1 (기존) | 2,156.9 | 1,882 | 2.37 ms | 0 ✓ | `eval/results/20260514_233540_neo_standard` |
| **run 2** | **2,223.8** | **1,817** | 2.17 ms (mid) | 0 ✓ | `20260516_115407_neo_standard` |
| run 3 | 2,211.6 | 1,833 | 2.75 ms | 0 ✓ | `20260516_122618_neo_standard` |
| **avg** | **2,197.4** | **1,844** | — | 0 | — |
| **min** | 2,156.9 | 1,817 | — | 0 | — |
| **max** | 2,223.8 | 1,833 | — | 0 | — |
| std | 35.6 | 33.1 | — | — | — |
| **CV** | **1.62%** | 1.80% | — | — | — |

vs vanilla 3-run avg (4,690.7): **46.8%**.

## 다른 측정과의 비교

| 측정 (500p × 8192) | 3-run avg | 3-run CV | min | max |
|---|---:|---:|---:|---:|
| vanilla (NEO OFF) | 4,690.7 | **0.006%** | 4,690.4 | 4,691.0 |
| **v1.6 (commit `64f9e0c48`)** | **2,197.4** | **1.62%** | 2,156.9 | 2,223.8 |
| Phase 3.1 only (KMP=200) | 2,134.9 | 5.68% | 2,013.2 | 2,255.7 |
| Phase 3.1+3.3 (KMP=200) | 2,083.3 | 3.13% | 2,015.4 | 2,145.4 |

→ **v1.6 가 모든 NEO 측정 중 가장 높은 avg + 가장 낮은 CV**. Phase 3.1/3.3 코드 변경이 throughput 하락 + variance 증가 (예상 외 결과).

## fact

| 항목 | 값 |
|---|---|
| commit | `64f9e0c48` (feat IDE_006/TSK_019 v1.6: shape mismatch fix + 22 strict 19/19) |
| 측정 시각 (KST) | run 1: 2026-05-14 23:35 → 2026-05-15 00:07 |
| | run 2: 2026-05-16 11:54 → 12:24 |
| | run 3: 2026-05-16 12:26 → 12:59 |
| run 2/3 launch script | `/tmp/v16_2run_wrapper.sh` → `eval/run_neo_standard.sh` × 2 |
| pacpu .so | v1.6 시점 (`omp_set_dynamic` 없음, build 2026-05-16 02:53 UTC) |
| KMP_BLOCKTIME | unset → 200ms default |
| MIRROR_MAX | 코드 default 80 (export 없음) |
| async_swap_buffers | 3 |
| cpu_resident_reqs | 128 |

### workload (3 run 동일)

| | |
|---|---|
| num_prompts | 500 |
| target_input_len / max_tokens | 8,192 / 8,192 |
| model | meta-llama/Llama-3.3-70B-Instruct |
| TP | 8 (H100 80GB × 8) |
| gpu_memory_utilization | 0.92 |
| kv_cache_dtype | fp8 |

## 22 strict (run 1 fact 인용 — `Best_v1.6_2157tps.md`)

| # | 항목 | 상태 | fact |
|---|---|:-:|---|
| 1 | KV exclusive ownership | ✅ | SWAP_OUT_CALL = 14,168 |
| 2 | CPU attention 직접 (chain) | ✅ | active = 1,582/39,600 (4.0%) |
| 3 | Asymmetric Pipelining | ✅ | OOM=0 |
| 4-7, 9-13 | (NEO 본질) | ✅ | (Best_v1.6_2157tps.md 참고) |
| **8** | **swap_out/in invariant** | **✅** | **mismatch=0** ★ |
| **12** | **b0/b1 정렬** | **✅** | reject_split_oob=0, mismatch=0 |
| **14** | **KV migration LRU + capacity** | **✅** | swap_out=14,168, mismatch=0 |
| 15 | NEO > vanilla throughput | ❌ | 2,156.9 vs 4,886 (44.1%) |
| 18 | deadlock 회피 | ✅ | engine_dead=0 |
| 19 | silent worker crash 0 | ✅ | assert=0 cuda=0 segv=0 |
| 20-22 | (Option I, L, M2) | ✅ | mirror mode=10 (2,897회), fire=272, mismatch=0 |

run 2/3 의 shape_mismatch = 0 확인 ✓ (run 1 fact 와 일관).

## 해석

- **Phase 3.1 (Persistent OMP) + Phase 3.3 (CUDA Stream Priority) 가 v1.6 baseline 보다 throughput 하락** (avg −3% ~ −5%) + variance 증가 (CV 1.62% → 3.13% / 5.68%)
- 가설:
  - Phase 3.1 의 `omp_set_dynamic(0)` 가 OMP team 재초기화 비용 제거하지만, 본 환경에서 그 비용이 작아 측정 net loss
  - Phase 3.3 의 stream priority 가 cdec backpressure 균등화 → chain fire 증가하나 wall 효과 X (Phase D 의 swap path overhead 가 wall 의 진짜 driver)
  - CV 증가의 원인 = predictor / decide_mode 의 wall-clock 의존 trigger 가 새 코드 path 에서 더 sensitive

## 파일

| file | 내용 | size |
|---|---|---:|
| `SUMMARY.txt` | run 2/3 wrapper timeline + 3-run 평균 산출 | — |
| `run1/result.json` + `metrics.log` + `engine.log.stdout.gz` | 기존 1-run (6.9 MB) | — |
| `run2/...` | 동일 (6.5 MB) | — |
| `run3/...` | 동일 (6.8 MB) | — |

총 32 MB.
