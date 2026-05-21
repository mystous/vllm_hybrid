# NEO Phase 3.1+3.3 KMP=200 default 500p 3-run sequential (2026-05-16)

> Phase 3.1 (persistent OMP) + Phase 3.3 (CUDA Stream Priority gdec=-1/cdec=0) combined.
> Phase 3.3 cherry-pick commit `0717f4b8c` 후 측정.
> Phase 3.1 only 3-run (`neo_phase3_1_kmp200_500p_3run_20260516/`) 과 동일 환경, 코드만 Phase 3.3 추가.

## 결과 (3-run 완료)

| Run | output_tps | wall (s) | cdec_wait final | b1_avg | shape_mismatch |
|---|---:|---:|---:|---:|:-:|
| run 1 | 2,089.2 | 1,947 | 2.05 ms | 4 | 0 ✓ |
| run 2 | 2,015.4 | 2,021 | 1.96 ms | 4 | 0 ✓ |
| run 3 | 2,145.4 | 1,903 | 2.90 ms | 6 | 0 ✓ |
| **avg** | **2,083.3** | **1,957** | 2.30 ms | 4.7 | 0 |
| **min** | 2,015.4 | 1,903 | 1.96 ms | 4 | 0 |
| **max** | 2,145.4 | 2,021 | 2.90 ms | 6 | 0 |
| std | 65.2 | 60.5 | — | — | — |
| **CV** | **3.13%** | 3.09% | — | — | — |

vs Phase 3.1 only avg (2,134.9): avg **−2.4%**, CV **−45%** (5.68% → 3.13%, variance 절반 감소).
vs vanilla 3-run avg (4,690.7): avg **44.4%**.

## fact

| 항목 | 값 |
|---|---|
| source dir | `eval/results/20260516_095651_phase3_1_omp_500p/` (run1), `20260516_103131_*` (run2), `20260516_110810_*` (run3) |
| 3-run summary | `eval/results/20260516_095651_phase3_1_3_kmp200_500p_3run/SUMMARY.txt` |
| launch script | `eval/run_neo_phase3_1_3_kmp200_3run.sh` → `run_neo_phase3_1_omp_500p.sh` × 3 |
| 측정 시각 (KST) | 2026-05-16 09:56:51 → 11:42:50 |
| base commit | `0717f4b8c` (Phase 3.3 cherry-pick into feat/neo-amx-apply) |
| KMP_BLOCKTIME | unset → 200ms default |
| Phase 3.1 코드 | `omp_set_dynamic(0) + omp_set_max_active_levels(1)` in `csrc/cpu/pacpu/core.h` |
| Phase 3.3 코드 | `torch.cuda.Stream(priority=-1/0)` in `vllm/v1/worker/sub_batch_executor.py:_get_batch_streams` |

### workload (3 run 동일)

| | |
|---|---|
| num_prompts | 500 |
| target_input_len / max_tokens | 8,192 / 8,192 |
| model | meta-llama/Llama-3.3-70B-Instruct |
| TP | 8 |
| gpu_memory_utilization | 0.92 |
| kv_cache_dtype | fp8 |

## 관측 (Phase 3.1 only 와 차이)

| 항목 | Phase 3.1 only avg | **Phase 3.1+3.3 avg** | Δ |
|---|---:|---:|---:|
| output_tps | 2,134.9 | **2,083.3** | **−2.4%** |
| wall (s) | 1,914 | 1,957 | +2.2% |
| min tps | 2,013.2 | 2,015.4 | +0.1% |
| max tps | 2,255.7 | 2,145.4 | **−4.9%** |
| std | 121.3 | 65.2 | **−46%** |
| **CV** | **5.68%** | **3.13%** | **−45%** |
| chain fire (Run 1 final) | 61% | 79.4% | +18p |
| cdec_wait avg (final) | 2.71 ms | 2.30 ms | −15% |
| b1_avg (avg of run finals) | 7.0 | 4.7 | −33% |

**Phase 3.3 (CUDA Stream Priority gdec=-1) 의 effect**:
- cdec_wait −15%, chain fire +18p, b1_avg −33% (sub-batch backpressure 완화)
- 그러나 **throughput avg −2.4%, peak −4.9%** — chain fire 증가가 wall 단축으로 연결 X
- **variance −45%** — Phase 3.3 가 sweet spot peak (Phase 3.1 only Run 2 의 2,255.7) 도, worst (Run 3 2,013) 도 막아 평탄화

## 해석

cdec_wait 가 빨라지고 chain fire 가 늘었음에도 throughput 이 lower → **NEO 의 wall critical path 는 cdec 가 아님**.

→ `analysis/archive/D_bottleneck_table.md` 의 NEO swap path Python+ATen overhead 가 wall 의 진짜 driver. Phase 3.3 의 stream priority 가 cdec backpressure 만 균등화 → swap path overhead 자체는 줄지 않음.

가속 우선순위는 `After_NEO_implementation_plan.md` 의 **★ Top Priority — Swap KV manipulation Python+ATen overhead 제거**.

## 파일

| file | 내용 | size |
|---|---|---:|
| `SUMMARY.txt` | 3-run wrapper timeline + 평균 산출 | — |
| `run1/result.json` | run 1 metrics | ~600B |
| `run1/metrics.log` | PROFILE/FORK/SWAP/CDEC/mismatch grep | ~4 MB |
| `run1/engine.log.stdout.gz` | full stdout gzipped | 5.6 MB |
| `run2/...` | 동일 (gzip 8.8 MB) | — |
| `run3/...` | 동일 (gzip 6.7 MB) | — |
