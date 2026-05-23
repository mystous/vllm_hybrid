# SUB_047 t3 — 3-run avg/min/max 검증 (2026-05-23 KST)

> **parent**: TSK_019 / SUB_047 (Tier 1 B ngram numba thread cap)
> **목적**: SUB_047 t3 (cap=8 + div_tp=0) 의 1-run best 10,949.8 tps 를 추가 2-run 으로 검증 → 3-run avg/min/max 확정
> **base config**: HEAD `de85efff1`, Llama-3.3-70B, TP=8, gmu=0.85, fp8 KV, 500p × 8192, spec=7

---

## 1. 3-run 결과

| run | tps | wall (s) | CPU busy avg (%) | GPU util avg (%) | crash |
|---|---:|---:|---:|---:|---:|
| run1 (SUB_047 t3 원본) | 10,949.8 | 367.1 | 5.52 | 54.6 | 0 |
| run2 (verify) | 10,963.5 | 366.6 | 5.47 | 54.7 | 0 |
| run3 (verify) | 10,956.5 | 366.8 | 5.55 | 54.8 | 0 |

## 2. 통계 요약

| 항목 | avg | min | max | range | range/avg |
|---|---:|---:|---:|---:|---:|
| **tps** | **10,956.6** | 10,949.8 | 10,963.5 | 13.7 | **0.125%** |
| wall (s) | 366.83 | 366.6 | 367.1 | 0.5 | 0.136% |
| CPU busy (%) | 5.51 | 5.47 | 5.55 | 0.08 | 1.45% |
| GPU util (%) | 54.70 | 54.6 | 54.8 | 0.2 | 0.366% |

→ tps variance **0.125%** = measurement noise 범위 안. configuration 매우 안정.

## 3. vs vanilla baseline (4,679.8 tps)

| 항목 | avg | min | max |
|---|---:|---:|---:|
| vs vanilla | **+134.1%** | +133.9% | +134.3% |
| speedup | **2.341×** | 2.339× | 2.343× |

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

- run2/run3 측정 중 sub041 leftover `cpu_bg_workload` proc 112 개가 idle 상태로 잔존 (May 22 launch). run2 tps 가 run1 보다 +13.7 ↑ 라서 영향 없음 확인됨.
- HEAD: `de85efff1` (3-run 검증 직전, SUB_047 patch + .gitignore + eval/results 산출물 commit 직후)
- 측정 시각: run1 = 23:36 (UTC), run2 = 04:39 (UTC), run3 = 04:49 (UTC), 2026-05-23

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| run1 result.json | `eval/results/20260523_081619_sub047_ngram_threads/t3_cap8_div0/result.json` |
| run2 result.json | `eval/results/20260523_133929_sub047_t3_verify/run2_cap8_div0/result.json` |
| run3 result.json | `eval/results/20260523_133929_sub047_t3_verify/run3_cap8_div0/result.json` |
| run2/3 SUMMARY | `eval/results/20260523_133929_sub047_t3_verify/SUMMARY.tsv` |
| launcher | `/tmp/run_sub047_t3_verify_2runs.sh` |
| wrapper | `/tmp/run_spec_decode.py` |

## 7. 결론

SUB_047 t3 (cap=8 + div_tp=0, spec=7) = **10,956.6 ± 7 tps (3-run avg, 0.125% variance)** 로 본 환경 (H100×8 + SPR dual, Llama-70B FP8 KV, 500p × 8192) 의 현재 throughput WINNER 확정. vs vanilla 4,679.8 tps: **+134.1% (2.341×)** ⭐
