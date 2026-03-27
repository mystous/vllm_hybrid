# vLLM Hybrid Benchmark Results

> 환경: i9-12900KF (16코어/24스레드) + RTX 3090 24GB + DDR5 64GB
> 모델: Qwen/Qwen2.5-1.5B
> 날짜: 2026-03-27

---

## 1. 서버 실행 명령어

### GPU Only

```bash
vllm serve Qwen/Qwen2.5-1.5B \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --port 8000
```

### Hybrid (GPU + CPU)

```bash
vllm serve Qwen/Qwen2.5-1.5B \
  --tensor-parallel-size 1 \
  --hybrid-mode parallel-batch \
  --hybrid-cpu-max-seqs 4 \
  --hybrid-cpu-max-batched-tokens 512 \
  --hybrid-stats-log-interval 10 \
  --enforce-eager \
  --port 8000
```

---

## 2. 벤치마크 실행 명령어

### GPU Only

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate inf \
  --save-result \
  --result-filename gpu_only_qwen_inf.json
```

### Hybrid (GPU + CPU)

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate inf \
  --save-result \
  --result-filename hybrid_qwen_inf.json
```

---

## 3. 결과 비교

| 지표 | GPU Only | Hybrid (GPU+CPU) | 차이 |
|------|----------|-------------------|------|
| **Request throughput (req/s)** | 55.39 | **63.93** | **+15.4%** |
| **Output token throughput (tok/s)** | 6,923 | **7,989** | **+15.4%** |
| **Total token throughput (tok/s)** | 14,000 | **16,157** | **+15.4%** |
| Benchmark duration (s) | 3.61 | **3.13** | **-13.3%** |
| Mean TTFT (ms) | 773.80 | **407.03** | **-47.4%** |
| Median TTFT (ms) | 778.28 | **261.81** | **-66.4%** |
| P99 TTFT (ms) | 1,359.38 | **865.86** | **-36.3%** |
| Mean TPOT (ms) | 23.41 | **22.27** | -4.9% |
| Median TPOT (ms) | 21.73 | 22.02 | +1.3% |
| P99 TPOT (ms) | 89.41 | 93.49 | +4.6% |
| Mean ITL (ms) | 21.78 | **20.95** | -3.8% |
| Median ITL (ms) | 18.15 | 18.09 | -0.3% |
| P99 ITL (ms) | 98.49 | 97.77 | -0.7% |

### 핵심 결론

- **Throughput**: 하이브리드 모드가 **15.4% 향상** (55.39 -> 63.93 req/s)
- **TTFT**: 하이브리드 모드가 **47-66% 개선** (CPU가 overflow 요청을 분담)
- **TPOT/ITL**: 거의 동등 (CPU 요청의 P99만 소폭 증가)
- **참고**: 현재 환경(DDR5 ~70GB/s)은 CPU 메모리 대역폭 제한으로 효과가 제한적. 타겟 환경(Xeon 8480+ 614GB/s)에서 더 큰 차이 예상

---

## 4. 재현 절차

```bash
# Step 1: GPU Only 테스트
vllm serve Qwen/Qwen2.5-1.5B --tensor-parallel-size 1 --enforce-eager --port 8000
# (별도 터미널)
python benchmarks/benchmark_serving.py --backend vllm --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B --dataset-name random --random-input-len 128 \
  --random-output-len 128 --num-prompts 200 --request-rate inf \
  --save-result --result-filename gpu_only_qwen_inf.json

# Step 2: 서버 종료
pkill -f "vllm serve"; sleep 5

# Step 3: Hybrid 테스트
vllm serve Qwen/Qwen2.5-1.5B --tensor-parallel-size 1 --hybrid-mode parallel-batch \
  --hybrid-cpu-max-seqs 4 --hybrid-cpu-max-batched-tokens 512 \
  --hybrid-stats-log-interval 10 --enforce-eager --port 8000
# (별도 터미널)
python benchmarks/benchmark_serving.py --backend vllm --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B --dataset-name random --random-input-len 128 \
  --random-output-len 128 --num-prompts 200 --request-rate inf \
  --save-result --result-filename hybrid_qwen_inf.json

# Step 4: 결과 비교
python -c "
import json
h = json.load(open('hybrid_qwen_inf.json'))
g = json.load(open('gpu_only_qwen_inf.json'))
print(f\"GPU-only:  {g['request_throughput']:.2f} req/s, {g['output_throughput']:.0f} tok/s\")
print(f\"Hybrid:    {h['request_throughput']:.2f} req/s, {h['output_throughput']:.0f} tok/s\")
print(f\"Speedup:   {h['request_throughput']/g['request_throughput']:.1%}\")
print(f\"TTFT gain: {(1-h['mean_ttft_ms']/g['mean_ttft_ms']):.1%}\")
"
```

---

## 5. 원본 벤치마크 출력

### GPU Only (raw output)

```
============ Serving Benchmark Result ============
Successful requests:                     200
Request rate configured (RPS):           inf
Benchmark duration (s):                  3.61
Total input tokens:                      25553
Total generated tokens:                  24995
Request throughput (req/s):              55.39
Output token throughput (tok/s):         6922.87
Total Token throughput (tok/s):          14000.29
---------------Time to First Token----------------
Mean TTFT (ms):                          773.80
Median TTFT (ms):                        778.28
P99 TTFT (ms):                           1359.38
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          23.41
Median TPOT (ms):                        21.73
P99 TPOT (ms):                           89.41
---------------Inter-token Latency----------------
Mean ITL (ms):                           21.78
Median ITL (ms):                         18.15
P99 ITL (ms):                            98.49
==================================================
```

### Hybrid GPU+CPU (raw output)

```
============ Serving Benchmark Result ============
Successful requests:                     200
Request rate configured (RPS):           inf
Benchmark duration (s):                  3.13
Total input tokens:                      25553
Total generated tokens:                  24993
Request throughput (req/s):              63.93
Output token throughput (tok/s):         7989.14
Total Token throughput (tok/s):          16157.29
---------------Time to First Token----------------
Mean TTFT (ms):                          407.03
Median TTFT (ms):                        261.81
P99 TTFT (ms):                           865.86
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          22.27
Median TPOT (ms):                        22.02
P99 TPOT (ms):                           93.49
---------------Inter-token Latency----------------
Mean ITL (ms):                           20.95
Median ITL (ms):                         18.09
P99 ITL (ms):                            97.77
==================================================
```
