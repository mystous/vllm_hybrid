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
|------|----------|-------------------|------|
| **Request throughput (req/s)** | 19.21 | 19.17 | -0.2% |
| **Output token throughput (tok/s)** | 9,633 | 9,629 | -0.0% |
| **Speedup** | - | - | **99.8%** |
| **TTFT gain** | - | - | **+0.1%** |

### 핵심 결론

- **개발 환경 한계**: i9-12900KF는 AVX-512/AMX 없음 → CPU 추론 속도가 GPU 대비 너무 낮아 기여분이 미미
- **`cpu_max_num_seqs` 자동 감지**: 16코어 / 4 = **4개** → 2000개 요청 중 CPU가 받는 건 4개(0.2%)
- **라우팅 전략 수정 완료**: GPU-first로 변경 (이전 CPU-first 대비 큰 개선, 아래 이력 참조)
- **타겟 환경 예상**: Xeon 8480+(112코어, AMX-BF16) + H100 환경에서 `cpu_max_num_seqs=28~112` 설정 시 실질적 throughput 기여 예상

---

## 4. 재현 절차

```bash
# Step 1: GPU Only 테스트
vllm serve Qwen/Qwen2.5-1.5B --tensor-parallel-size 1 --enforce-eager --port 8000
# (별도 터미널)
python benchmarks/benchmark_serving.py --backend vllm --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B --dataset-name random --random-input-len 128 \
  --random-output-len 512 --num-prompts 2000 --request-rate inf \
  --save-result --result-filename gpu_only_qwen_saturated.json

# Step 2: 서버 종료
pkill -f "vllm serve"; sleep 5

# Step 3: Hybrid 테스트
vllm serve Qwen/Qwen2.5-1.5B --tensor-parallel-size 1 --hybrid-mode parallel-batch \
  --hybrid-cpu-max-seqs 4 --hybrid-cpu-max-batched-tokens 512 \
  --hybrid-stats-log-interval 10 --enforce-eager --port 8000
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

## 5. 벤치마크 이력

### 2026-03-27: 라우팅 전략 CPU-first → GPU-first 수정

라우팅 전략 수정 전후 비교 (opt-125m, input=128, output=128, 200 prompts):

| | GPU-only | Hybrid (CPU-first, 수정 전) | Hybrid (GPU-first, 수정 후) |
|-|----------|----------------------------|----------------------------|
| **tok/s** | 6,122 | 4,655 (-24%) | 5,444 (+17% 개선) |
| **req/s** | 47.83 | 36.37 | 42.53 |
| **P99 TTFT** | 26ms | 1,198ms | 1,047ms |

수정 전 CPU-first 전략은 GPU-only 대비 **-76% (4.2배 느림)**,
수정 후 GPU-first 전략은 opt-125m 기준 **GPU-only와 동등 수준** 회복.

### 구버전 결과 (참고용, 방법론 오류 포함)

> **주의**: 아래 결과는 `--num-prompts 200 --random-output-len 128` 설정으로
> GPU가 포화되지 않은 상태에서 측정된 것으로 신뢰도 낮음.

| 지표 | GPU Only | Hybrid (GPU+CPU) | 차이 |
|------|----------|-------------------|------|
| Request throughput (req/s) | 55.39 | 63.93 | +15.4% |
| Output token throughput (tok/s) | 6,923 | 7,989 | +15.4% |
| Mean TTFT (ms) | 773.80 | 407.03 | -47.4% |
