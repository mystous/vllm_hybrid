# vLLM Hybrid 테스트 가이드

> 마지막 업데이트: 2026-04-11

---

## 1. 실행 모드

### Parallel-Batch (Dual-Process, 현재 유일한 공식 경로)

GPU 와 CPU 가 별도 OS 프로세스에서 완전 병렬 실행:

```
GPU EngineCoreProc (PID A) ← MultiprocExecutor, N × GPUWorker
CPU EngineCoreProc (PID B, num_numa_nodes 개) ← UniProcExecutor, CPUWorker
```

```bash
# 자동 감지 (권장) — 모든 CPU 파라미터 auto, 1 sequence / NUMA engine, thread binding 자동
vllm serve meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code

# 디버깅용 override (대부분 불필요)
vllm serve meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch \
  --hybrid-routing-strategy capacity \
  --hybrid-routing-priority cpu-first
```

> 주의: `VLLM_HETEROGENEOUS_PLATFORM=1` / `--device heterogeneous` / `--pipeline-parallel-size 2`
> 조합은 초기 구현의 유물이며 현재 코드 경로가 아니다.

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

# CPU ops / utils 빌드 확인
python -c "
import vllm._custom_ops as ops, torch
print('HAS_CPU_OPS:', ops.HAS_CPU_OPS)     # AVX-512F 있으면 True
print('HAS_CPU_UTILS:', ops.HAS_CPU_UTILS) # 어떤 x86_64 에서도 True
print('init_cpu_threads_env:', torch.ops._C_utils.init_cpu_threads_env)
"
```

---

## 6. 진단 로그

하이브리드 엔진 진입점 / CPU 워커 / 라우팅 / attention path 등을 추적하려면:

```bash
# 모든 마커 매 호출 로깅 (디버그)
VLLM_HYBRID_TRACE=1 vllm serve ...

# 기본 (N 호출마다)
VLLM_HYBRID_TRACE_EVERY=200 vllm serve ...
```

주요 마커 (8 종):

| marker | 의미 |
|--------|------|
| `[HYBRID-RESOLVE]` | `_resolve_cpu_params` 결과 |
| `[HYBRID-LAUNCH]` | `launch_hybrid_engines` 프로세스 런칭 |
| `[HYBRID-CLIENT]` | 라우팅 dispatch (request_id → engine identity) |
| `[HYBRID-CPU-ENV]` | CPU 프로세스 OS 환경 변수 |
| `[HYBRID-CPU-PROC]` | CPU EngineCore 프로세스 초기화 |
| `[HYBRID-CPU-WORKER]` | thread binding (C++ vs Python fallback) |
| `[HYBRID-CPU-EXEC]` | CPU worker `execute_model` per-step trace |
| `[HYBRID-CPU-ATTN]` | decode path counter (custom_avx / ipex / sdpa_batched / sdpa_loop) |
