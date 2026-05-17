# NEO Phase 3.1 KMP=200 default 500p 3-run sequential (2026-05-16)

> Phase 3.1 (persistent OMP) + KMP_BLOCKTIME=200 default 의 500p × 8192 3-run.
> 직전 1-run 측정 (Phase 3.1+KMP=50 500p, #129, 2,103.4 tps) 과 v1.6 best (2,157 tps) 의 직접 비교 + variance 확인.

## 결과 (3-run 완료)

| Run | output_tps | wall (s) | cdec_wait final | b1_avg | shape_mismatch |
|---|---:|---:|---:|---:|:-:|
| run 1 | 2,135.7 | 1,908 | 2.67 ms | 8 | 0 ✓ |
| run 2 | 2,255.7 | 1,813 | 2.50 ms | 6 | 0 ✓ |
| run 3 | 2,013.2 | 2,020 | 2.97 ms | 7 | 0 ✓ |
| **avg** | **2,134.9** | **1,914** | 2.71 ms | 7.0 | 0 |
| **min** | 2,013.2 | 1,813 | 2.50 ms | 6 | 0 |
| **max** | 2,255.7 | 2,020 | 2.97 ms | 8 | 0 |
| std | 121.3 | 103.5 | — | — | — |
| **CV** | **5.68%** | 5.41% | — | — | — |

vs v1.6 best (2,156.9, single-trial): avg **−1.0%**, best Run 2 **+4.6%**, worst Run 3 **−6.7%**.
vs vanilla 3-run avg (4,690.7): avg **45.5%**, best Run 2 **48.1%**, worst Run 3 **42.9%**.

## fact

| 항목 | 값 |
|---|---|
| source dir 들 | `eval/results/20260516_072121_phase3_1_omp_500p/` (run1), `20260516_075501_phase3_1_omp_500p/` (run2), `20260516_082841_phase3_1_omp_500p/` (run3) |
| launch script | `eval/run_neo_phase3_1_kmp200_3run.sh` → `run_neo_phase3_1_omp_500p.sh` × 3 |
| 측정 시각 (KST) | 2026-05-16 07:21:21 → ~09:00 |
| base commit | `64e4e9973` (Phase 3.1 merged HEAD) |
| KMP_BLOCKTIME | unset → **200ms default** |
| Phase 3.1 코드 | `omp_set_dynamic(0) + omp_set_max_active_levels(1)` in `csrc/cpu/pacpu/core.h` |

### workload (3 run 동일)

| | |
|---|---|
| num_prompts | 500 |
| target_input_len / max_tokens | 8,192 / 8,192 |
| total_input_tokens | 2,709,037 |
| total_output_tokens | ~4.08M |
| model | meta-llama/Llama-3.3-70B-Instruct |
| TP | 8 |
| gpu_memory_utilization | 0.92 |
| kv_cache_dtype | fp8 |

## variance 분석 (Run 1 vs Run 2)

| 항목 | Run 1 | Run 2 | Δ |
|---|---:|---:|---:|
| output_tps | 2,135.7 | 2,255.7 | **+5.6%** |
| wall | 1,908s | 1,813s | −5.0% |
| init | 80.06s | 80.84s | ≈0 |
| first PROFILE (cdec=50) cdec_wait | 3.44 ms | 3.07 ms | −10.8% |
| stabilized cdec_wait (cdec≥400) | 2.67 ms | 2.50 ms | **−6.4%** |
| b1_avg | 8 | 6 | lower backpressure |

**핵심 관측**:
- init time 동일 → vllm/CUDA init bottleneck 아님
- cdec_wait 가 첫 50 cdec 부터 이미 다름 → cdec runtime 자체의 영역
- vanilla 3-run (CV 0.006%) 대비 NEO 의 variance 가 매우 큼 → NEO path 의 stochastic source 존재

## 원인 후보 (vanilla 대조 후)

1. **OMP thread placement variance** — process fork 마다 OMP team (10 thread × 8 worker) 의 core 매핑 비결정성
2. **mirror cache fill ordering / scheduler bistability** — chain fire 가 active 평형 진입에 sensitive
3. **Linux page cache warm** — Run 2 가 Run 1 직후 시작 → libpacpu.so + libtorch + model weight 의 page cache 잔존
4. cf. vanilla 3-run = perfectly deterministic (CV 0.006%) — 위 후보 중 (1)(2) 는 NEO specific, (3) 도 vanilla 받음에도 noise 0 → (1)(2) 가 주 원인 추정

## 파일

| file | 내용 | size |
|---|---|---:|
| `SUMMARY.txt` | 3-run wrapper 의 timeline + 평균 산출 출력 | — |
| `run1/result.json` | run 1 metrics | ~600B |
| `run1/metrics.log` | PROFILE/FORK/SWAP_OUT/CDEC CALL/shape mismatch grep | ~3.6 MB |
| `run1/engine.log.stdout.gz` | full stdout gzipped | ~13 MB |
| `run2/...` | 동일 (gzip 8.7 MB) | — |
| `run3/...` | 동일 (gzip ~12 MB) | — |
