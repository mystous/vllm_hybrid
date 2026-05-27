# SUB_085 — Phase 2 unblock: suffix decoding + cudagraph_mode=PIECEWISE (⭐⭐ best)

> **parent**: TSK_020 / SUB_081 (Phase 2) 의 후속
> **measurement**: 2026-05-25 KST 08:50~09:31 (smoke + v1 OOM 부분 + v2 3-cell 완료)
> **status**: ✅ **완료 ⭐⭐ — fundamental incompat 영역 아니었음, 모든 workload net positive**

---

## 1. Phase 2 unblock 의 핵심

| 시도 | 결과 |
|---|---|
| SUB_081 / SUB_084 영역 결론 | "arctic_inference v0.1.2 vs vLLM 1.6 영역 fundamental architectural incompat" — **잘못된 결론** |
| SUB_085 영역 fix | `compilation_config={"cudagraph_mode": "PIECEWISE"}` 영역 단 한 줄 영역 우회. FULL graph capture 영역 skip → suffix 영역 dynamic batch shape 영역 적응 가능 |
| wrapper 영역 추가 | `/tmp/run_workload_gen.py` 영역 `VLLM_CUDAGRAPH_MODE` env-tunable (compilation_config override) |
| 본 fork 영역 vLLM core 변경 | **0 (wrapper 영역만 변경)** |

→ fundamental incompat 영역 아니었음. cudagraph mode 영역 단순 downgrade 영역 우회 가능.

## 2. 측정 결과 (3 workload × suffix_spec32 + PIECEWISE)

### 2.1 v1 (gmu=0.85) — OOM 영역 sonnet/code fail

| workload | tps | wall | K | α | status |
|---|---:|---:|---:|---:|---|
| sonnet | (OOM) | — | **9.68** | **93.3%** | fail (raw_target_logits fp32 alloc 1.28 GiB) |
| chat | **3,565.9** | 103.6 | 10.05 | 92.6% | ✓ |
| code | (OOM) | — | **7.96** | **79.0%** | fail |

→ OOM 원인: cudagraph PIECEWISE 영역 graph memory 추가 + bentoml services shared GPU (6 GiB) → gmu=0.85 영역 너무 큼.

### 2.2 v2 (gmu=0.80) ⭐ — 3 workload 모두 SUCCESS

| workload | tps | wall (s) | out_tok | **K (mean_accept_len)** | **α (per-pos)** |
|---|---:|---:|---:|---:|---:|
| **sonnet** | **11,589.5** | 348.7 | 4,040,828 | **5.11** | **77.0%** |
| **chat** | **3,582.4** | 103.1 | 369,368 | **10.06** | **92.7%** |
| **code** | **7,990.0** | 494.2 | 3,948,854 | **4.08** | **40.1%** |

## 3. ★ Fair comparison — SUB_086 (vanilla gmu=0.80 baseline 영역 신설)

이전 vanilla baseline (SUB_044/047/071) 영역 모두 gmu=0.85 + different wrapper (sonnet 영역). SUB_085 v2 영역 fair vs vanilla 위해 **SUB_086 영역 same wrapper / same gmu (0.80) vanilla baseline 영역 새로 측정**.

### 3.1 SUB_085 v2 vs SUB_086 fair comparison (모두 gmu=0.80, same wrapper)

| workload | SUB_086 vanilla | SUB_085 v2 (suffix PIECEWISE) | **fair contribution** |
|---|---:|---:|---:|
| **sonnet** | 7,709.8 | 11,589.5 | **+50.3% (1.503×)** ⭐ |
| **chat** | 2,186.9 | 3,582.4 | **+63.8% (1.638×)** ⭐ |
| **code** ⭐ | 6,717.8 | 7,990.0 | **+18.9% (1.189×)** ⭐ |

→ **본 fork 영역 진짜 새 contribution = suffix + PIECEWISE 영역 3 workload 모두 net positive**.

### 3.2 SUB_085 v2 vs SUB_047/071 ngram (caveat: gmu 다름)

| workload | SUB_047/071 ngram cap=8 (gmu=0.85) | SUB_085 v2 suffix PIECEWISE (gmu=0.80) | 차이 |
|---|---:|---:|---:|
| sonnet | 10,956.5 | 11,589.5 | +5.78% (단 gmu 차이) |
| chat | 3,006.6 | 3,582.4 | +19.2% |
| **code** ⭐ | **5,346.8 (-23.2% 회귀)** | **7,990.0** | **+49.4%** ⭐⭐ (회귀 완전 mitigation + 추가) |

### 3.3 SUB_085 v2 vs SUB_074 suffix eager mode (cuda graph 효과)

| workload | SUB_074 suffix eager (gmu=0.85) | SUB_085 v2 suffix PIECEWISE (gmu=0.80) | cuda graph 효과 |
|---|---:|---:|---:|
| sonnet | 8,236.0 | 11,589.5 | **+40.7%** ⭐ |
| chat | 2,369.7 | 3,582.4 | **+51.2%** ⭐ |
| code | 7,093.5 | 7,990.0 | **+12.6%** |

## 4. K / α 비교 (suffix vs ngram, 모두 같은 mean_accept_len 정의)

| workload | ngram K (SUB_075) | ngram α | suffix K (SUB_085 v2) | suffix α | K 비율 | α 비율 |
|---|---:|---:|---:|---:|---:|---:|
| sonnet | 3.72 | 38.8% | **5.11** | **77.0%** | 1.37× | 1.99× |
| chat | 6.69 | 81.2% | **10.06** | **92.7%** | 1.50× | 1.14× |
| **code** | 1.10 | 1.4% | **4.08** | **40.1%** | **★ 3.71×** | **★ 28.6×** |

→ suffix 영역 모든 workload 영역 K, α 영역 큰 향상. code 영역 가장 극적 — ngram 영역 prompt only lookup 영역 code 영역 매칭 거의 0% (α=1.4%), suffix 영역 prompt + generation 양쪽 영역 lookup 영역 K 7×, α 29× 향상.

## 5. 본 fork 영역 변경 (SUB_085 추가)

| file | 변경 종류 | 라인 |
|---|---|---:|
| `vllm/utils/__init__.py` (SUB_081) | FlexibleArgumentParser re-export | +5 |
| `vllm/engine/arg_utils.py` (SUB_084) | _is_v1_supported_oracle stub | +9 |
| **`/tmp/run_workload_gen.py` (SUB_085)** | **`VLLM_CUDAGRAPH_MODE` env-tunable** (compilation_config override) | **+5** |
| 본 fork vLLM core 누적 | (그대로) | **+14** |

→ SUB_085 영역 본 fork vLLM core 변경 없음. wrapper env 만 (+5 줄). 

## 6. production 적용 method

```python
LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=8,
    max_model_len=16384,
    max_num_seqs=256,
    gpu_memory_utilization=0.80,  # ★ cuda graph PIECEWISE + spec 영역 memory headroom
    kv_cache_dtype="fp8",
    max_num_batched_tokens=8192,
    seed=0,
    compilation_config={"cudagraph_mode": "PIECEWISE"},  # ★ SUB_085 핵심 patch
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,
    },
)
```

**필수 env**:
```bash
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""
# arctic plugin 영역 disable, lazy import SuffixDecodingCache 만 사용
```

## 7. 후속 candidate

- ★★★ **SUB_086 영역 새 vanilla baseline 영역 historical 영역 갱신** — analysis doc / Best doc 영역 sonnet vanilla 영역 7,709.8 영역 baseline 영역 갱신 (4,679.8 영역 다른 wrapper 영역 historical noise)
- ★★ ngram cap=8 영역 cudagraph_mode=PIECEWISE + gmu=0.80 영역 재측정 — gmu 차이 영역 fair comparison
- ★ canonical 3-run for SUB_085 v2 (variance 측정)
- ★ workload-aware routing 영역 ngram (sonnet/chat 영역 +37%/+134%) vs suffix (code 영역 +19%) 영역 mix 영역 적용

## 8. raw 자료

| 항목 | 위치 |
|---|---|
| v1 smoke (sonnet) | `eval/results/20260525_085000_sub085_smoke_sonnet_piecewise/` |
| v1 full | `eval/results/20260525_085511_sub085_{sonnet,chat,code}_suffix_piecewise/` |
| v2 full ⭐ | `eval/results/20260525_091016_sub085v2_{sonnet,chat,code}_suffix_piecewise/` |
| launcher v1 | `/tmp/run_sub085_suffix_piecewise_full.sh` |
| launcher v2 (gmu=0.80) | `/tmp/run_sub085_suffix_piecewise_v2.sh` |
| smoke launcher | `/tmp/run_sub085_suffix_piecewise.sh` |
| stdout v1 | `/tmp/sub085_full.log` |
| stdout v2 | `/tmp/sub085v2.log` |
| summary | `/tmp/sub085_summary.tsv`, `/tmp/sub085v2_summary.tsv` |
| SUB_086 fair baseline | [`../sub086_vanilla_gmu080_20260525/RESULTS.md`](../sub086_vanilla_gmu080_20260525/RESULTS.md) |
