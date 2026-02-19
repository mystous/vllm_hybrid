● 8 GPU + CPU 하이브리드 서버 실행 가이드

  ---
  동작 원리

  world_size = tensor_parallel_size (예: 10)

  rank 0~7  → GPUWorker (H100 × 8)
  rank 8~9  → CPUWorker (Xeon 112코어)

  판단 기준: rank >= torch.cuda.device_count() 이면 자동으로 CPUWorker 할당

  vllm/worker/worker_base.py에서 자동 판별:
  if self.rpc_rank >= num_gpus:  # rank >= 8이면
      worker_cls = "vllm.v1.worker.cpu_worker.CPUWorker"

  ---
  1. 서버 실행

  환경 변수 설정
```
  # 필수
  export VLLM_HETEROGENEOUS_PLATFORM=1

  # Intel CPU 최적화
  export OMP_NUM_THREADS=112
  export KMP_AFFINITY=granularity=fine,compact,1,0
  export KMP_BLOCKTIME=1
  export MKL_ENABLE_INSTRUCTIONS=AVX512
  export VLLM_CPU_KVCACHE_SPACE=512
  export VLLM_CPU_OMP_THREADS_BIND=auto

  #서버 시작 (8 GPU + 2 CPU Worker)

  VLLM_HETEROGENEOUS_PLATFORM=1 vllm serve meta-llama/Llama-3-70B-Instruct \
    --device heterogeneous \
    --tensor-parallel-size 10 \
    --dtype bfloat16 \
    --hybrid-mode parallel-batch \
    --hybrid-cpu-threads 112 \
    --hybrid-numa-aware \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code

  ┌────────────────────────┬────────────────┬───────────────────────────┐
  │          인자          │       값       │           설명            │
  ├────────────────────────┼────────────────┼───────────────────────────┤
  │ --device               │ heterogeneous  │ CPU+GPU 이기종 모드       │
  ├────────────────────────┼────────────────┼───────────────────────────┤
  │ --tensor-parallel-size │ 10             │ 8 GPU + 2 CPU 워커        │
  ├────────────────────────┼────────────────┼───────────────────────────┤
  │ --dtype                │ bfloat16       │ AVX-512 BF16 최적화       │
  ├────────────────────────┼────────────────┼───────────────────────────┤
  │ --hybrid-mode          │ parallel-batch │ CPU/GPU 배치 분리         │
  ├────────────────────────┼────────────────┼───────────────────────────┤
  │ --hybrid-cpu-threads   │ 112            │ Xeon 전체 코어 활용       │
  ├────────────────────────┼────────────────┼───────────────────────────┤
  │ --hybrid-numa-aware    │ -              │ NUMA 메모리/스레드 최적화 │
  └────────────────────────┴────────────────┴───────────────────────────┘
```

  Pipeline Parallel 방식 (대안)

  # GPU 8개로 텐서 병렬, 2 파이프라인 스테이지
  VLLM_HETEROGENEOUS_PLATFORM=1 vllm serve meta-llama/Llama-3-70B-Instruct \
    --device heterogeneous \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000

  ---
  2. 클라이언트 실행

  2-1. curl (가장 간단)

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

  # 스트리밍 모드
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3-70B-Instruct",
      "messages": [{"role": "user", "content": "Hello!"}],
      "max_tokens": 100,
      "stream": true
    }'

  # Completions API
  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3-70B-Instruct",
      "prompt": "The capital of France is",
      "max_tokens": 50
    }'

  2-2. Python (OpenAI SDK)

  pip install openai

  from openai import OpenAI

  client = OpenAI(
      api_key="EMPTY",
      base_url="http://localhost:8000/v1",
  )

  # 모델 목록 확인
  models = client.models.list()
  print("Available models:", [m.id for m in models.data])

  # Chat Completion
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

  # 스트리밍
  stream = client.chat.completions.create(
      model="meta-llama/Llama-3-70B-Instruct",
      messages=[{"role": "user", "content": "안녕하세요!"}],
      max_tokens=256,
      stream=True,
  )

  for chunk in stream:
      if chunk.choices[0].delta.content:
          print(chunk.choices[0].delta.content, end="", flush=True)
  print()

  2-3. 벤치마크 클라이언트

  # 처리량 벤치마크 (서버 불필요 - 직접 엔진 실행)
  VLLM_HETEROGENEOUS_PLATFORM=1 python benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 10 \
    --num-prompts 200 \
    --input-len 256 \
    --output-len 128 \
    --dtype bfloat16

  # 서빙 벤치마크 (서버 실행 후)
  python benchmarks/benchmark_serving.py \
    --backend openai \
    --base-url http://localhost:8000 \
    --model meta-llama/Llama-3-70B-Instruct \
    --dataset-name random \
    --num-prompts 500 \
    --random-input-len 256 \
    --random-output-len 128 \
    --request-rate inf

  # 지연시간 벤치마크
  VLLM_HETEROGENEOUS_PLATFORM=1 python benchmarks/benchmark_latency.py \
    --model meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 10 \
    --input-len 256 \
    --output-len 128 \
    --num-iters 20

  ---
  3. GPU-only vs 하이브리드 비교 테스트

  #!/bin/bash
  MODEL="meta-llama/Llama-3-70B-Instruct"

  echo "===== GPU-only (8x H100) ====="
  python benchmarks/benchmark_throughput.py \
    --model $MODEL \
    --tensor-parallel-size 8 \
    --num-prompts 200 \
    --input-len 256 \
    --output-len 128 \
    --dtype bfloat16

  echo ""
  echo "===== Hybrid (8 GPU + 2 CPU) ====="
  VLLM_HETEROGENEOUS_PLATFORM=1 python benchmarks/benchmark_throughput.py \
    --model $MODEL \
    --tensor-parallel-size 10 \
    --num-prompts 200 \
    --input-len 256 \
    --output-len 128 \
    --dtype bfloat16

  ---
  4. 워커 할당 구조
```
  ┌──────────────────────────────────────────────┐
  │              vLLM Server                      │
  │  tensor_parallel_size = 10                    │
  ├──────────────────────────────────────────────┤
  │                                               │
  │  rank 0 ──→ GPUWorker (H100 #0)             │
  │  rank 1 ──→ GPUWorker (H100 #1)             │
  │  rank 2 ──→ GPUWorker (H100 #2)             │
  │  rank 3 ──→ GPUWorker (H100 #3)             │
  │  rank 4 ──→ GPUWorker (H100 #4)             │
  │  rank 5 ──→ GPUWorker (H100 #5)             │
  │  rank 6 ──→ GPUWorker (H100 #6)             │
  │  rank 7 ──→ GPUWorker (H100 #7)             │
  │  rank 8 ──→ CPUWorker (Xeon, NUMA node 0)   │
  │  rank 9 ──→ CPUWorker (Xeon, NUMA node 1)   │
  │                                               │
  │  통신: Gloo backend (CPU↔GPU)                │
  └──────────────────────────────────────────────┘
```

  ---
  5. 서버 상태 확인

  # 서버 health check
  curl http://localhost:8000/health

  # 모델 목록
  curl http://localhost:8000/v1/models

  # GPU 상태 모니터링 (별도 터미널)
  watch -n 1 nvidia-smi
