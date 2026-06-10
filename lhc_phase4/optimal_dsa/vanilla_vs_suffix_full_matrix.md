# Vanilla vs Suffix Speculative Decoding — Full 10 Models × 7 Corpora Matrix (DGX B200)

> **Source**: TSK_042 canonical 222-cell measurement (`runs/tput_t1t3_20260602`), 2026-06-02
> **Coverage**: 10 모델 × 7 corpus × {vanilla, suffix} = **140 cells** (이 문서가 다루는 영역)
> **Single completion guarantee**: 본 문서 하나로 재현 + 해석 + 분석 가능 (외부 의존 없음)

---

## 1. Hardware Environment

| 항목 | 값 |
|---|---|
| GPU | **NVIDIA B200 × 8** (sm_100, 183 GiB HBM3e each, NVLink5) |
| CPU | **Intel Xeon Platinum 8570** dual-socket (Emerald Rapids), 224 thread |
| ISA features | AVX-512 + AMX (BF16/INT8) + DSA 8 SWQ (dsa0/dsa1 × 4 engines) |
| NUMA | 2 nodes — NUMA0 = CPU 0-55,112-167, NUMA1 = CPU 56-111,168-223 |
| DRAM | 2 TB system memory |
| Host | `dgx-b200` |
| Container | sm_100 build (CUDA 12.8, driver 580) |

---

## 2. Software / vLLM Configuration

| 항목 | 값 |
|---|---|
| vLLM version | `1.7.dev16107+gffe20fb09.d20260601` (sm_100 rebuild) |
| Venv | `/workspace/vllm_dev_prj/bin/python` (CPython 3.12) |
| vLLM binary | `/workspace/vllm_dev_prj/bin/vllm` |
| Editable source | `/workspace/host_vllm_hybrid/vllm/` |
| Compilation | `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` (FaP) |
| Backend | OpenAI-compatible serve, streaming completions |

### 2.1 Environment variables (all measurements)
```bash
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8
export VLLM_NGRAM_DIVIDE_BY_TP=0
```

### 2.2 vLLM serve command (per model, TP picked by head%8)
```bash
CUDA_VISIBLE_DEVICES=$GPUS \
  /workspace/vllm_dev_prj/bin/vllm serve $MODEL \
  --tensor-parallel-size $TP \
  --port 8001 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  $SPEC_ARGS  # vanilla: 빈 문자열 / suffix: speculative-config
```

### 2.3 Suffix speculative config (suffix method only)
```
--speculative-config '{"method":"suffix","num_speculative_tokens":32}'
```
- `method=suffix` → arctic_inference SuffixDecodingCache (`vllm/v1/spec_decode/suffix_decoding.py`)
- `num_speculative_tokens=K=32` → draft tree depth
- 기본 파라미터 유지: `suffix_decoding_max_tree_depth=24`, `max_spec_factor=1.0`, `min_token_prob=0.1`

---

## 3. Benchmark Configuration

| 항목 | 값 |
|---|---|
| Harness | `vllm_config_perf/gating/realistic_eval/throughput_runner.py` |
| Prompts source | `runs/tput_t1t3_20260602/sampled_prompts.parquet` (실 trace 샘플) |
| Concurrency | **32** (asyncio.Semaphore) |
| max_tokens | **8,192** |
| Streaming | True (TTFT/TPOT 분리 측정) |
| max-model-len | 16,384 (input ≤ 4,096 + output 8,192 ≤ 16,384) |
| Per-corpus limit | 자연 trace 입력 길이, 7 condition 격리 |
| `mix` 셀 | 500-prompt shuffle (seed=0) |

### 3.1 Throughput runner CLI
```bash
PYTHONPATH=. python vllm_config_perf/gating/realistic_eval/throughput_runner.py \
  --in $SAMPLED \
  --method $METHOD          # 'vanilla' 또는 'suffix'
  --model $MODEL \
  --model-tag $TAG \
  --port 8001 \
  --max-tokens 8192 \
  --concurrency 32 \
  --corpus $CORPUS          # mix 의 경우 --limit 500 --shuffle 추가
  --out summ_${TAG}_${METHOD}_${CORPUS}.json \
  --raw per_request_raw.jsonl
```

### 3.2 Metric 정의

| 메트릭 | 정의 |
|---|---|
| `output_tps` | `total_completion_tokens / wall_total_s` (시스템 전체 처리량) |
| `ttft_ms_p50` | First token time p50 (스트리밍 첫 토큰 도착) |
| `tpot_ms_p50` | Time-per-output-token p50 (이후 토큰) |
| `accept_rate` (α) | `num_accepted_tokens / num_draft_tokens` (vLLM /metrics 누적 차이) |

---

## 4. Corpus Information (7 conditions)

| Corpus tag | Source dataset | 특성 | 평균 입력 길이 |
|---|---|---|---|
| `sharegpt` | ShareGPT (Anthropic LMSYS 정제본) | 대화 (chat), multi-turn | 중간 (300~600 tok) |
| `wildchat` | WildChat-1M | 자연 대화 (real-world chat) | 중간 |
| `lmsys` | LMSYS-Chat-1M | 대화 (Arena 입력) | 중간 |
| `humaneval` | LiveCodeBench (HumanEval subset) | 짧은 코드 함수 | 짧음 (~200 tok) |
| `mbpp` | LiveCodeBench (MBPP subset) | 짧은 코드 (Python) | 짧음 |
| `swebench` | SWE-Bench Lite | 코드 + repo context | 김 (~1000 tok) |
| `mix` | 위 6 corpus 의 500-prompt shuffle | 운영 mix proxy | 다양 |

**샘플링 절차** (재현):
- `sampled_prompts.parquet` 의 row 마다 `corpus` 컬럼 — single-corpus 셀은 그 corpus 만 격리 측정
- `mix` 셀은 전체 row 를 seed=0 로 shuffle 후 첫 500개 사용
- TSK_042 측정 시 LIMIT=500 적용 (per_corpus 자연 길이 분포 보존)

---

## 5. Model Information (10 models)

| TAG (재현 시 그대로 사용) | HF model_id | TP | num_heads | head%8 | 모델 특성 |
|---|---|---:|---:|:---:|---|
| `Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | 4 | 28 | ❌ | 7B dense, GQA |
| `DeepSeek-R1-Distill-Qwen-7B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 4 | 28 | ❌ | 7B reasoning distill |
| `Llama-3.1-8B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` | 8 | 32 | ✅ | 8B dense |
| `Qwen2.5-32B-Instruct` | `Qwen/Qwen2.5-32B-Instruct` | 8 | 40 | ✅ | 32B dense |
| `DeepSeek-R1-Distill-Qwen-32B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 8 | 40 | ✅ | 32B reasoning distill |
| `Qwen2.5-72B-Instruct` | `Qwen/Qwen2.5-72B-Instruct` | 8 | 64 | ✅ | 72B dense |
| `Llama-3.1-70B-Instruct` | `meta-llama/Llama-3.1-70B-Instruct` | 8 | 64 | ✅ | 70B dense |
| `DeepSeek-R1-Distill-Llama-70B` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 8 | 64 | ✅ | 70B reasoning distill |
| `Llama-3.1-405B-Instruct-FP8` | `meta-llama/Llama-3.1-405B-Instruct-FP8` | 8 | 128 | ✅ | 405B dense FP8 (`--allow-deprecated-quantization` 필요) |
| `DeepSeek-R1` | `deepseek-ai/DeepSeek-R1` | 8 | 128 | ✅ | 671B MoE (37B active, 256 expert × top-k 8); 2×TP4 + gmu 0.95 적재 |

**TP 결정 규칙**: `num_attention_heads % 8 == 0` → TP=8 (GPU 0-7); 아니면 TP=4 (GPU 0-3).

---

## 6. Main Comparison Table — 70 cells (10 models × 7 corpora)

> 셀 포맷: `output_tps` (vanilla → suffix), `Δ% vs vanilla`, `α (acceptance rate)`, `TTFT p50 ms` (van→suf), `TPOT p50 ms` (van→suf)
> Model TAG 는 첫 행에만 표시. Markdown 표 그대로 복사 가능.

| model | corpus | vanilla tps | suffix tps | Δ% | α | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-7B-Instruct` | sharegpt | 4,189 | 6,167 | +47.2% | 1.413 | 28.1→39.9 | 7.1→11.3 |
| `` | swebench | 4,120 | 5,416 | +31.5% | 1.106 | 27.8→40.5 | 6.4→9.5 |
| `` | humaneval | 3,754 | 5,213 | +38.9% | 0.621 | 20.2→37.3 | 4.4→7.0 |
| `` | mbpp | 3,814 | 5,506 | +44.4% | 0.485 | 19.7→25.5 | 4.2→6.8 |
| `` | wildchat | 4,184 | 6,285 | +50.2% | 0.686 | 28.3→39.2 | 7.1→10.8 |
| `` | lmsys | 4,090 | 5,956 | +45.6% | 0.684 | 27.7→44.8 | 7.2→11.2 |
| `` | mix | 4,169 | 7,803 | **+87.2%** | 0.881 | 26.2→69.3 | 6.9→3.1 |
| `DeepSeek-R1-Distill-Qwen-7B` | sharegpt | 8,724 | 11,961 | +37.1% | 0.621 | 17.0→18.5 | 3.3→5.6 |
| `` | swebench | 8,835 | 15,422 | +74.5% | 0.698 | 21.6→24.1 | 3.3→2.2 |
| `` | humaneval | 8,159 | 11,459 | +40.4% | 0.548 | 19.0→19.5 | 3.3→4.1 |
| `` | mbpp | 8,440 | 12,398 | +46.9% | 0.580 | 16.8→18.4 | 3.3→4.4 |
| `` | wildchat | 8,925 | 11,717 | +31.3% | 0.609 | 16.9→19.3 | 3.3→5.3 |
| `` | lmsys | 8,811 | 11,360 | +28.9% | 0.576 | 16.8→18.6 | 3.3→5.4 |
| `` | mix | 9,058 | 24,458 | **+170.0%** | 0.876 | 16.8→22.3 | 3.3→0.9 |
| `Llama-3.1-8B-Instruct` | sharegpt | 8,868 | 19,054 | +114.8% | 0.851 | 25.2→21.2 | 3.5→1.4 |
| `` | swebench | 8,348 | 21,353 | +155.8% | 0.889 | 49.7→29.8 | 3.5→1.2 |
| `` | humaneval | 9,048 | 15,126 | +67.2% | 0.765 | 48.4→22.9 | 3.5→1.3 |
| `` | mbpp | 8,730 | 17,825 | +104.2% | 0.790 | 32.1→22.1 | 3.5→1.4 |
| `` | wildchat | 9,002 | 19,856 | +120.6% | 0.857 | 41.1→22.0 | 3.5→1.3 |
| `` | lmsys | 9,074 | 19,862 | +118.9% | 0.849 | 35.3→21.6 | 3.5→1.3 |
| `` | mix | 8,850 | 27,851 | **+214.7%** | 0.933 | 22.8→24.7 | 3.5→1.0 |
| `Qwen2.5-32B-Instruct` | sharegpt | 3,079 | 4,662 | +51.4% | 0.654 | 31.4→41.1 | 9.0→13.9 |
| `` | swebench | 2,892 | 5,002 | +73.0% | 0.599 | 33.7→46.0 | 8.4→11.1 |
| `` | humaneval | 2,571 | 4,859 | +89.0% | 0.683 | 30.0→57.8 | 7.8→8.4 |
| `` | mbpp | 2,915 | 5,138 | +76.3% | 0.644 | 30.1→47.4 | 8.3→11.4 |
| `` | wildchat | 3,128 | 4,884 | +56.2% | 0.619 | 29.8→38.8 | 9.1→13.2 |
| `` | lmsys | 3,053 | 4,478 | +46.7% | 0.625 | 30.0→42.1 | 9.2→13.7 |
| `` | mix | 3,056 | 6,597 | **+115.9%** | 0.857 | 29.6→65.4 | 9.3→3.1 |
| `DeepSeek-R1-Distill-Qwen-32B` | sharegpt | 4,803 | 4,996 | +4.0% | 0.551 | 24.1→31.9 | 5.9→10.6 |
| `` | swebench | 4,409 | 5,241 | +18.9% | 0.588 | 27.1→41.8 | 6.0→10.1 |
| `` | humaneval | 3,462 | 3,771 | +8.9% | 0.381 | 24.8→29.8 | 5.7→8.4 |
| `` | mbpp | 4,690 | 5,690 | +21.3% | 0.487 | 23.4→34.7 | 5.8→8.0 |
| `` | wildchat | 4,891 | 5,729 | +17.1% | 0.573 | 24.2→34.2 | 5.9→10.7 |
| `` | lmsys | 4,898 | 5,356 | +9.3% | 0.537 | 24.2→33.2 | 5.9→10.1 |
| `` | mix | 4,938 | 9,056 | **+83.4%** | 0.801 | 24.1→37.0 | 5.9→1.6 |
| `Qwen2.5-72B-Instruct` | sharegpt | 2,688 | 3,219 | +19.7% | 0.530 | 32.3→43.6 | 9.6→15.3 |
| `` | swebench | 2,361 | 2,647 | +12.1% | 0.382 | 40.6→48.8 | 9.3→12.2 |
| `` | humaneval | 806 | 2,489 | **+208.6%** ⚠ | 0.277 | 34.8→34.0 | 8.9→9.8 |
| `` | mbpp | 3,395 | 3,234 | **−4.8%** | 0.266 | 26.8→31.5 | 8.6→9.2 |
| `` | wildchat | 2,803 | 2,621 | **−6.5%** | 0.465 | 30.8→44.4 | 9.3→15.1 |
| `` | lmsys | 2,807 | 3,429 | +22.2% | 0.556 | 32.9→47.5 | 9.8→15.2 |
| `` | mix | 2,735 | 5,268 | **+92.6%** | 0.852 | 31.2→57.4 | 9.3→2.6 |
| `Llama-3.1-70B-Instruct` | sharegpt | 3,091 | 4,864 | +57.4% | 0.735 | 28.9→45.0 | 9.1→15.4 |
| `` | swebench | 2,878 | 6,026 | +109.4% | 0.814 | 39.6→60.1 | 9.3→14.0 |
| `` | humaneval | 3,391 | 4,728 | +39.4% | 0.694 | 37.9→54.4 | 9.1→7.0 |
| `` | mbpp | 1,773 | 3,266 | +84.2% | 0.443 | 26.6→38.2 | 8.4→9.7 |
| `` | wildchat | 3,172 | 5,261 | +65.9% | 0.753 | 29.0→46.8 | 9.2→14.9 |
| `` | lmsys | 3,040 | 3,958 | +30.2% | 0.653 | 28.8→46.1 | 9.1→14.5 |
| `` | mix | 3,129 | 10,400 | **+232.4%** ⭐ | 0.915 | 28.4→56.3 | 9.2→2.5 |
| `DeepSeek-R1-Distill-Llama-70B` | sharegpt | 3,033 | 2,660 | **−12.3%** | 0.346 | 28.2→38.5 | 8.9→13.8 |
| `` | swebench | 3,236 | 2,739 | **−15.4%** | 0.323 | 35.3→52.9 | 9.1→12.9 |
| `` | humaneval | 2,852 | 2,788 | −2.2% | 0.381 | 35.2→46.7 | 8.8→11.8 |
| `` | mbpp | 2,777 | 2,426 | **−12.6%** | 0.265 | 27.9→43.7 | 8.8→11.7 |
| `` | wildchat | 3,127 | 2,658 | **−15.0%** | 0.354 | 28.6→43.1 | 9.0→14.7 |
| `` | lmsys | 2,992 | 2,848 | −4.8% | 0.374 | 28.6→42.3 | 9.0→13.8 |
| `` | mix | 3,164 | 6,127 | **+93.7%** | 0.786 | 28.4→50.1 | 9.0→2.1 |
| `Llama-3.1-405B-Instruct-FP8` | sharegpt | 1,217 | 2,061 | +69.3% | 0.687 | 70.9→97.8 | 23.3→33.4 |
| `` | swebench | 1,204 | 2,639 | +119.1% | 0.825 | 93.7→149.2 | 23.9→29.9 |
| `` | humaneval | 1,253 | 2,112 | +68.5% | 0.703 | 93.1→137.8 | 23.4→14.7 |
| `` | mbpp | 916 | 1,725 | +88.3% | 0.489 | 66.9→87.8 | 22.1→23.1 |
| `` | wildchat | 1,280 | 2,290 | +78.9% | 0.687 | 71.5→104.3 | 23.5→32.5 |
| `` | lmsys | 1,220 | 2,243 | +84.0% | 0.673 | 71.0→102.3 | 23.4→31.5 |
| `` | mix | 1,252 | 2,829 | **+125.9%** | 0.766 | 71.1→122.2 | 23.4→17.3 |
| `DeepSeek-R1` | sharegpt | 1,475 | 797 | **−46.0%** ⚠ | 0.504 | 64.5→173.2 | 19.6→60.7 |
| `` | swebench | 1,474 | 538 | **−63.5%** ⚠ | 0.361 | 80.6→269.6 | 19.7→66.4 |
| `` | humaneval | 1,004 | 606 | **−39.6%** | 0.538 | 66.1→96.4 | 18.1→29.5 |
| `` | mbpp | 1,437 | 677 | **−52.9%** ⚠ | 0.366 | 62.3→186.8 | 18.8→43.3 |
| `` | wildchat | 1,556 | 858 | **−44.8%** | 0.528 | 65.3→179.4 | 19.5→61.2 |
| `` | lmsys | 1,533 | 811 | **−47.1%** | 0.453 | 64.5→174.8 | 19.4→57.1 |
| `` | mix | 1,538 | 781 | **−49.2%** | 0.451 | 65.6→196.8 | 19.7→51.4 |

---

## 7. Aggregate statistics

### 7.1 통계 요약 (70 셀)
| 지표 | 값 |
|---|---:|
| Suffix net-positive (Δ ≥ +1%) | **55 / 70 (78.6%)** |
| Suffix net-negative (Δ ≤ −1%) | **15 / 70 (21.4%)** |
| 최대 gain | `Llama-3.1-70B-Instruct` / `mix` **+232.4%** ⭐ |
| 최악 손실 | `DeepSeek-R1` / `swebench` **−63.5%** |
| 평균 Δ% (전 70 셀) | +52.8% |
| 평균 Δ% (성공 55 셀) | +83.2% |
| 평균 Δ% (실패 15 셀) | −28.4% |

### 7.2 α (acceptance rate) 분포

| set | 셀 수 | α median | α range |
|---|--:|--:|--:|
| Suffix net-positive | 55 | **0.684** | 0.28 ~ 1.41 |
| Suffix net-negative | 15 | **0.374** | 0.27 ~ 0.54 |

→ **α ≈ 0.5 가 break-even 임계치** (K=32 셋업). 그 아래에선 wasted compute 가 gain 을 초과.

### 7.3 모델별 평균 Δ% (7 corpus)

| TAG | 평균 Δ% | 패턴 |
|---|--:|---|
| `Llama-3.1-8B-Instruct` | **+127.9%** | standard dense, α 0.77~0.93 |
| `Llama-3.1-70B-Instruct` | **+88.4%** | standard dense, α 0.44~0.92 |
| `Llama-3.1-405B-Instruct-FP8` | **+90.6%** | XL dense FP8 |
| `DeepSeek-R1-Distill-Qwen-7B` | +61.3% | distill, α 0.55~0.88 |
| `Qwen2.5-32B-Instruct` | +72.6% | standard dense |
| `Qwen2.5-72B-Instruct` | +49.1% | larger dense, α 일부 낮음 (0.27~0.85) |
| `Qwen2.5-7B-Instruct` | +49.3% | small dense |
| `DeepSeek-R1-Distill-Qwen-32B` | +23.3% | distill, α 0.38~0.80 (낮은 α 경계) |
| `DeepSeek-R1-Distill-Llama-70B` | **+4.5%** | distill reasoning, 6/7 corpus 음수 |
| `DeepSeek-R1` (671B MoE) | **−49.0%** | MoE expert routing, 7/7 corpus 음수 |

### 7.4 Vanilla-win 15 셀 분포

| 모델 | 실패 corpus 수 | α 범위 | 원인 |
|---|--:|---|---|
| `DeepSeek-R1` | 7/7 (전부) | 0.36~0.54 | MoE expert dynamic routing → suffix tree prefix-match 율 무력화 + verify forward cost 큼 |
| `DeepSeek-R1-Distill-Llama-70B` | 6/7 (mix 만 성공) | 0.27~0.38 | distill reasoning chain 다양성 → α 0.3 근처, spec gain < overhead |
| `Qwen2.5-72B-Instruct` | 2/7 (mbpp/wildchat) | 0.27~0.47 | marginal break-even, 일부 corpus 에서 wall_ratio 1.05~1.08× |

---

## 8. 재현 가이드 — Step by Step

### 8.1 환경 준비
```bash
# 1. vLLM editable install (sm_100 build for B200)
cd /workspace/vllm_dev_prj
uv venv --python 3.12
source .venv/bin/activate

# 2. Source 디렉토리
cd /workspace/host_vllm_hybrid
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

### 8.2 Corpus 샘플 준비
TSK_042 의 `sampled_prompts.parquet` 재사용 또는 직접 빌드:
```bash
PYTHONPATH=. python vllm_config_perf/gating/realistic_eval/prompt_sampler.py \
  --corpora open \
  --n 500 \
  --out sampled_prompts.parquet
```

### 8.3 단일 셀 측정 (예: Llama-3.1-8B-Instruct × suffix × mix)
```bash
# Boot vLLM serve
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid /workspace/vllm_dev_prj/bin/vllm serve \
  meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 8 --port 8001 \
  --gpu-memory-utilization 0.85 --max-model-len 16384 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method":"suffix","num_speculative_tokens":32}' &

# Wait for ready
until curl -sf "http://127.0.0.1:8001/v1/models" >/dev/null; do sleep 5; done

# Run benchmark
PYTHONPATH=. /workspace/vllm_dev_prj/bin/python \
  vllm_config_perf/gating/realistic_eval/throughput_runner.py \
  --in sampled_prompts.parquet \
  --method suffix \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-tag Llama-3.1-8B-Instruct \
  --port 8001 --max-tokens 8192 --concurrency 32 \
  --limit 500 --shuffle \
  --out summ_Llama-3.1-8B-Instruct_suffix_mix.json
```

### 8.4 전 70 셀 sweep
TSK_042 가 사용한 wrapper:
```bash
MODELS="meta-llama/Llama-3.1-8B-Instruct ..." \
METHODS="vanilla suffix" \
CORPORA="sharegpt swebench humaneval mbpp wildchat lmsys" \
SAMPLED=sampled_prompts.parquet \
OUTDIR=runs/$(date +%Y%m%d) \
bash vllm_config_perf/gating/realistic_eval/run_throughput_8gpu.sh
```

---

## 9. 해석 / Production Implications

### 9.1 모델 유형 분류 (Suffix 적합성)
- ✅ **Suffix 권장**: `Llama-3.1-{8B,70B}-Instruct`, `Llama-3.1-405B-Instruct-FP8`, `Qwen2.5-{32B,72B}-Instruct`, `Qwen2.5-7B-Instruct`, `DeepSeek-R1-Distill-Qwen-7B` — standard dense + 7B distill
- 🟡 **Suffix 조건부**: `DeepSeek-R1-Distill-Qwen-32B` (α 경계), `Qwen2.5-72B-Instruct` 일부 corpus (mbpp/wildchat)
- ❌ **Vanilla 권장**: `DeepSeek-R1-Distill-Llama-70B` (mix 외), `DeepSeek-R1` 671B (전 corpus)

### 9.2 운영 게이트 권장
Per-(model, corpus) oracle 매트릭스 기반으로:
1. **Static gate**: 모델 배포 시 위 표 참조 → vanilla / suffix 정적 선택 (per_request 분류기 무관, TSK_044 기각 근거)
2. **Per-workload gate**: 모델 + corpus 결정 시 표의 α 값 참고 → α<0.5 인 cell 은 vanilla 사용

### 9.3 Open Questions / TBD
- 본 측정은 conc=32 single-tenant. **multi-tenant + concurrent serving** 환경에서 α 분포 변화 미확인
- `Llama-3.1-405B-Instruct-FP8` 의 `--allow-deprecated-quantization` 의존 (vllm 향후 제거 시 재측정 필요)
- `DeepSeek-R1` 671B 의 vanilla 우세는 MoE expert routing 특성 — **TSK_045 MoE CPU offload** 가 진정한 회수 lever (현재 진행 중)

---

## 10. Raw Data Pointers

| 산출물 | 경로 |
|---|---|
| Per-cell JSON | `vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/summ_<TAG>_<METHOD>_<CORPUS>.json` |
| Per-request raw | 동일 dir / `per_request_raw.jsonl` |
| Aggregate parquet | 동일 dir / `metrics_table.parquet` |
| Sampled corpus | 동일 dir / `sampled_prompts.parquet` |
| Boot logs | 동일 dir / `_logs/<TAG>_<METHOD>.log` |
| 생성 wrapper | `vllm_config_perf/gating/realistic_eval/run_throughput_8gpu.sh` |
| Runner source | `vllm_config_perf/gating/realistic_eval/throughput_runner.py` |
| Aggregator | `vllm_config_perf/gating/realistic_eval/build_throughput_table.py` |

---

## 11. Citation / Tracing

- Parent IDE: **IDE_022** (AGSD Realistic-Workload + Decision-Regret Evaluation)
- Parent TSK: **TSK_042** (워크로드 활용 실험), status **완료** (2026-06-01)
- Doc: `shadow_assists/features/IDE_022_agsd_realistic_eval/TSK_042_realistic_workload_oracle/RESULTS.md`
- 222 cells full (vanilla 70 + suffix 70 + ngram 5 + llm-d 56 + llm-d sweep 21) 중 본 문서는 **140 cells** (vanilla + suffix) 만 다룸
