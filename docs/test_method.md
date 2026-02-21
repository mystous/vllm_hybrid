# vLLM Hybrid 테스트 가이드

> 최종 업데이트: 2026-02-21

---

## 1. 실행 모드

### 모드 A: Parallel-Batch (Dual-Process, 권장)

GPU와 CPU가 별도 프로세스에서 완전 병렬 실행:

```
GPU EngineCoreProc (PID A) ← TP=8, 8x H100
CPU EngineCoreProc (PID B) ← UniProcExecutor, CPUWorker
```

```bash
# 자동 감지 (권장)
vllm serve meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code

# 수동 설정
vllm serve meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch \
  --hybrid-cpu-max-seqs 28 \
  --hybrid-cpu-kvcache-gb 800 \
  --hybrid-cpu-threads 112 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
```

### 모드 B: Heterogeneous Pipeline (레거시)

GPU와 CPU가 같은 텐서 병렬 그룹에서 파이프라인 실행:

```bash
VLLM_HETEROGENEOUS_PLATFORM=1 vllm serve meta-llama/Llama-3-70B-Instruct \
  --device heterogeneous \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
```

---

## 2. 클라이언트 테스트

### 2.1 curl

```bash
# Health check
curl http://localhost:8000/health

# Chat Completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-70B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "한국의 수도는 어디인가요?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'

# 스트리밍
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-70B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

### 2.2 Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "CPU와 GPU의 차이점을 설명해주세요."},
    ],
    max_tokens=512,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

---

## 3. 벤치마크

### 3.1 서빙 벤치마크 (서버 실행 후)

```bash
python benchmarks/benchmark_serving.py \
  --backend openai \
  --base-url http://localhost:8000 \
  --model meta-llama/Llama-3-70B-Instruct \
  --dataset-name random \
  --num-prompts 500 \
  --random-input-len 128 \
  --random-output-len 128 \
  --request-rate 10
```

### 3.2 GPU-only vs Hybrid 비교

```bash
#!/bin/bash
MODEL="meta-llama/Llama-3-70B-Instruct"
BENCH="python benchmarks/benchmark_serving.py --backend openai --model $MODEL --dataset-name random --num-prompts 500 --random-input-len 128 --random-output-len 128"

echo "===== GPU-only (8x H100) ====="
# 먼저 GPU-only 서버 시작 후
$BENCH --base-url http://localhost:8000

echo ""
echo "===== Hybrid (GPU + CPU) ====="
# Hybrid 서버 시작 후
$BENCH --base-url http://localhost:8000
```

### 3.3 소규모 테스트 (개발 머신)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-1.3b \
    --dtype float16 \
    --port 8000 \
    --hybrid-mode parallel-batch

python3 benchmarks/benchmark_serving.py \
    --backend openai \
    --base-url http://localhost:8000 \
    --model facebook/opt-1.3b \
    --dataset-name random \
    --num-prompts 100 \
    --request-rate 10 \
    --random-input-len 128 \
    --random-output-len 64
```

---

## 4. 상태 확인

```bash
# 프로세스 확인
ps aux | grep -E "GPU_EngineCore|CPU_EngineCore"

# GPU 모니터링
watch -n 1 nvidia-smi

# 서버 모델 목록
curl http://localhost:8000/v1/models

# NUMA 메모리 확인 (CPU 프로세스 PID 필요)
numastat -p <PID>
```

---

## 5. 모듈 단위 테스트

```bash
# CPU 기능 감지
python -c "
from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features
f = detect_intel_cpu_features()
print(f'{f.model_name}: {f.num_sockets}S x {f.cores_per_socket}C x {f.threads_per_core}T')
print(f'AVX-512={f.avx512f}, VNNI={f.avx512_vnni}, AMX={f.amx_bf16}')
"

# Hybrid 모듈 import 확인
python -c "
from vllm.v1.engine.hybrid_core import is_hybrid_mode, CapacityAwareRouter
from vllm.v1.engine.core_client import HybridAsyncMPClient
print('All imports OK')
"

# CPU ops 빌드 확인
python -c "
try:
    import vllm._C_cpu_ops
    print('_C_cpu_ops: available')
except ImportError:
    print('_C_cpu_ops: not built')
"
```
