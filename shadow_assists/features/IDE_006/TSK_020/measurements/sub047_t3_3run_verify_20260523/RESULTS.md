# SUB_047 t3 — canonical 3-run 검증 (2026-05-23 KST)

> **parent**: TSK_020 / SUB_047 (Tier 1 B ngram numba thread cap)
> **목적**: SUB_047 t3 (cap=8 + div_tp=0) winner config 의 canonical 3-run avg/min/max 확정.
> **base config**: HEAD `de85efff1`, Llama-3.3-70B, TP=8, gmu=0.85, fp8 KV, 500p × 8192, spec=7

---

## 1. 3-run 결과

| run | tps | wall (s) | CPU busy avg (%) | GPU util avg (%) | crash |
|---|---:|---:|---:|---:|---:|
| 1 | 10,981.4 | 366.0 | 5.51 | 54.6 | 0 |
| 2 | 10,931.7 | 367.7 | 5.57 | 54.7 | 0 |
| 3 | 10,956.3 | 366.8 | 5.59 | 54.8 | 0 |

## 2. 통계 요약

| 항목 | avg | min | max | range | range/avg |
|---|---:|---:|---:|---:|---:|
| **tps** | **10,956.5** | 10,931.7 | 10,981.4 | 49.7 | **0.454%** |
| wall (s) | 366.83 | 366.0 | 367.7 | 1.7 | 0.46% |
| CPU busy (%) | 5.557 | 5.51 | 5.59 | 0.08 | 1.43% |
| GPU util (%) | 54.70 | 54.6 | 54.8 | 0.2 | 0.366% |

→ tps variance **0.454%** = measurement noise 범위 안. configuration 안정.

## 3. vs vanilla baseline (4,679.8 tps)

| 항목 | avg | min | max |
|---|---:|---:|---:|
| vs vanilla | **+134.12%** | +133.59% | +134.65% |
| speedup | **2.341×** | 2.336× | 2.347× |

## 3.1 Historical reference (canonical 3-run 외)

다음 측정들은 canonical 3-run 영역 외 historical reference (동일 config 의 추가 데이터 포인트):
- SUB_047 5-way sweep t3 (2026-05-23 08:16): 10,949.8 tps
- SUB_047 1차 verify batch (2026-05-23 13:39): 10,963.5 / 10,956.5 (2-run)

모두 canonical 3-run avg 10,956.5 의 noise band (±25 tps) 안.

## 4. 설정

### 4.1 vLLM constructor

```python
LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=8,
    max_model_len=16384,
    max_num_seqs=256,
    gpu_memory_utilization=0.85,
    enforce_eager=False,
    kv_cache_dtype="fp8",
    max_num_batched_tokens=8192,
    disable_log_stats=True,
    seed=0,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 7,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    },
)
```

### 4.2 env

```bash
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1
export VLLM_NGRAM_NUM_THREADS_CAP=8
export VLLM_NGRAM_DIVIDE_BY_TP=0
```

### 4.3 sampling

```python
SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8192, seed=0)
```

## 5. 측정 환경 비고

- HEAD: `de85efff1` 이상 (SUB_047 patch + .gitignore + eval/results 산출물 commit 후)
- 측정 시각 (KST): run 1 = 19:04, run 2 = 16:24, run 3 = 16:35 — 2026-05-23
- sub041 leftover `cpu_bg_workload` proc 112 개가 idle 상태로 잔존 (May 22 launch). 모든 run 에서 idle 상태로 영향 작음 (variance 0.454% 안).

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| run 1 result.json | `eval/results/20260523_100441_sub048_ngram_refinement/t1_baseline/result.json` |
| run 2 result.json | `eval/results/20260523_162456_sub047_t3_verify/run2_cap8_div0/result.json` |
| run 3 result.json | `eval/results/20260523_162456_sub047_t3_verify/run3_cap8_div0/result.json` |
| run 2/3 SUMMARY | `eval/results/20260523_162456_sub047_t3_verify/SUMMARY.tsv` |
| launcher (verify) | `/tmp/run_sub047_t3_verify_2runs.sh` |
| wrapper | `/tmp/run_spec_decode.py` |

## 7. 결론

SUB_047 t3 (cap=8 + div_tp=0, spec=7) = **10,956.5 ± 25 tps (3-run avg, 0.454% variance)** 로 본 환경 (H100×8 + SPR dual, Llama-70B FP8 KV, 500p × 8192) 의 현재 throughput WINNER 확정. vs vanilla 4,679.8 tps: **+134.12% (2.341×)** ⭐
