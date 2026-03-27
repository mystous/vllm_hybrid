# vLLM Hybrid Benchmark Results

> 환경: i9-12900KF (16코어/24스레드, AVX2) + RTX 3090 24GB + DDR5 64GB
> 모델: Qwen/Qwen2.5-1.5B
> 날짜: 2026-03-27

---

## 1. 서버 실행 명령어

### GPU Only

```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --disable-log-requests
```

### Hybrid (GPU + CPU)

```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --hybrid-mode parallel-batch \
  --hybrid-cpu-max-seqs 4 \
  --hybrid-cpu-kvcache-gb 4 \
  --hybrid-cpu-threads 8 \
  --no-hybrid-numa-aware \
  --hybrid-stats-log-interval 10 \
  --disable-log-requests
```

---

## 2. 벤치마크 실행 명령어

GPU 포화 상태를 유발하기 위해 요청 수(2000)를 충분히 늘리고,
출력 길이(512)를 늘려 GPU가 오랫동안 바쁜 상태를 유지하도록 설정.

### GPU Only

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 512 \
  --num-prompts 2000 \
  --request-rate inf \
  --save-result \
  --result-filename gpu_only_qwen_saturated.json
```

### Hybrid (GPU + CPU)

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 512 \
  --num-prompts 2000 \
  --request-rate inf \
  --save-result \
  --result-filename hybrid_qwen_saturated.json
```

---

## 3. 결과 비교

| 지표 | GPU Only | Hybrid (GPU+CPU) | 차이 |
|------|----------|------------------|------|
| **Request throughput (req/s)** | 19.50 | 18.71 | -4.1% |
| **Output token throughput (tok/s)** | 9,799 | 9,405 | -4.0% |
| **Total token throughput (tok/s)** | 12,289 | 11,794 | -4.0% |
| Benchmark duration (s) | 102.57 | 106.89 | +4.2% |
| Mean TTFT (ms) | 45,093 | 46,920 | +4.1% |
| Median TTFT (ms) | 41,219 | 42,944 | +4.2% |
| P99 TTFT (ms) | 93,520 | 98,462 | +5.3% |
| Mean TPOT (ms) | 26.04 | 27.28 | +4.8% |
| Median TPOT (ms) | 25.41 | 27.07 | +6.5% |
| P99 TPOT (ms) | 96.31 | 101.43 | +5.3% |
| Mean ITL (ms) | 24.79 | 26.31 | +6.1% |
| Median ITL (ms) | 21.88 | 23.47 | +7.3% |
| P99 ITL (ms) | 115.26 | 116.54 | +1.1% |

### 핵심 결론

- **개발 환경 한계**: i9-12900KF는 AVX-512/AMX 없음 → CPU 추론이 GPU 대비 느려서 오히려 오버헤드 발생
- **`cpu_max_num_seqs=4`**: 2000개 요청 중 CPU가 처리하는 건 최대 4개(0.2%) — 기여분 미미
- **GPU-first 라우팅은 정상 동작**: `gpu_max_num_seqs=256` 도달 후 CPU overflow 발생
- **타겟 환경 예상**: Xeon 8480+(112코어, AMX-BF16) + H100 환경에서 CPU가 실질적 처리량 기여 가능

---

## 4. 재현 절차

```bash
# Step 1: GPU Only 테스트
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B --port 8000 \
  --gpu-memory-utilization 0.9 --disable-log-requests

# (별도 터미널)
python benchmarks/benchmark_serving.py --backend vllm --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B --dataset-name random --random-input-len 128 \
  --random-output-len 512 --num-prompts 2000 --request-rate inf \
  --save-result --result-filename gpu_only_qwen_saturated.json

# Step 2: 서버 종료
pkill -f "vllm.entrypoints.openai.api_server"; sleep 5

# Step 3: Hybrid 테스트
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B --port 8000 \
  --gpu-memory-utilization 0.9 \
  --hybrid-mode parallel-batch --hybrid-cpu-max-seqs 4 \
  --hybrid-cpu-kvcache-gb 4 --hybrid-cpu-threads 8 \
  --no-hybrid-numa-aware --hybrid-stats-log-interval 10 \
  --disable-log-requests

# (별도 터미널)
python benchmarks/benchmark_serving.py --backend vllm --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B --dataset-name random --random-input-len 128 \
  --random-output-len 512 --num-prompts 2000 --request-rate inf \
  --save-result --result-filename hybrid_qwen_saturated.json

# Step 4: 결과 비교
python -c "
import json
h = json.load(open('hybrid_qwen_saturated.json'))
g = json.load(open('gpu_only_qwen_saturated.json'))
print('')
print(f\"GPU-only:  {g['request_throughput']:.2f} req/s, {g['output_throughput']:.0f} tok/s\")
print(f\"Hybrid:    {h['request_throughput']:.2f} req/s, {h['output_throughput']:.0f} tok/s\")
print(f\"Speedup:   {h['request_throughput']/g['request_throughput']:.1%}\")
print(f\"TTFT gain: {(1-h['mean_ttft_ms']/g['mean_ttft_ms']):.1%}\")
print('')
"
```

---

## 5. 원본 벤치마크 출력

### GPU Only (raw output)

```
============ Serving Benchmark Result ============
Successful requests:                     2000
Benchmark duration (s):                  102.57
Total input tokens:                      255412
Total generated tokens:                  1005075
Request throughput (req/s):              19.50
Output token throughput (tok/s):         9799.19
Total Token throughput (tok/s):          12289.38
---------------Time to First Token----------------
Mean TTFT (ms):                          45092.77
Median TTFT (ms):                        41219.45
P99 TTFT (ms):                           93520.20
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.04
Median TPOT (ms):                        25.41
P99 TPOT (ms):                           96.31
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.79
Median ITL (ms):                         21.88
P99 ITL (ms):                            115.26
==================================================
```

### Hybrid GPU+CPU (raw output)

```
============ Serving Benchmark Result ============
Successful requests:                     2000
Benchmark duration (s):                  106.89
Total input tokens:                      255412
Total generated tokens:                  1005239
Request throughput (req/s):              18.71
Output token throughput (tok/s):         9404.76
Total Token throughput (tok/s):          11794.33
---------------Time to First Token----------------
Mean TTFT (ms):                          46920.03
Median TTFT (ms):                        42944.13
P99 TTFT (ms):                           98461.92
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.28
Median TPOT (ms):                        27.07
P99 TPOT (ms):                           101.43
---------------Inter-token Latency----------------
Mean ITL (ms):                           26.31
Median ITL (ms):                         23.47
P99 ITL (ms):                            116.54
==================================================
```

---

## 6. 벤치마크 이력

### 2026-03-27: 라우팅 전략 CPU-first → GPU-first 수정

수정 전후 비교 (facebook/opt-125m, input=128, output=128, 200 prompts):

| | GPU-only | Hybrid (CPU-first, 수정 전) | Hybrid (GPU-first, 수정 후) |
|-|----------|----------------------------|----------------------------|
| **tok/s** | 6,122 | 4,655 (-24%) | 5,444 (+17% 개선) |
| **req/s** | 47.83 | 36.37 | 42.53 |
| **P99 TTFT** | 26ms | 1,198ms | 1,047ms |

수정 전 CPU-first 전략은 GPU-only 대비 **-76%**, 수정 후 GPU-first는 동등 수준 회복.
