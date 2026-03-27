# vLLM Hybrid Evaluation Scripts

GPU-only vs Hybrid(GPU+CPU) 서빙 성능을 자동으로 비교하는 평가 스크립트 모음.
GPU/CPU 활용률을 실시간으로 측정(CSV)하고, 벤치마크 결과를 비교 리포트로 출력한다.

---

## 파일 구조

```
eval/
├── envs/                    # 환경별 설정 파일
│   ├── dev_rtx3090.env     # 개발 머신 (i9-12900KF + RTX 3090)
│   ├── h100x1.env          # H100 x1
│   ├── h100x8.env          # H100 x8 + Xeon 8480+ 듀얼소켓
│   └── a100x1.env          # A100 x1
├── serve.sh        # vLLM 서버 시작 (gpu_only / hybrid 모드)
├── benchmark.sh    # benchmark_serving.py 실행
├── monitor.py      # GPU/CPU 활용률 백그라운드 모니터 (CSV 저장)
├── compare.py      # GPU-only vs Hybrid 비교 리포트 생성
├── run_eval.sh     # 전체 파이프라인 오케스트레이터
└── results/        # 결과 파일 저장 디렉토리
```

---

## 빠른 시작

```bash
cd /workspace/vllm_hybrid/eval

# 개발 머신 (RTX 3090)
./run_eval.sh envs/dev_rtx3090.env

# H100 x8 서버
./run_eval.sh envs/h100x8.env

# 결과 확인 (latest 심볼릭 링크 사용)
cat results/latest/comparison.txt
```

---

## run_eval.sh 사용법

```bash
./run_eval.sh <env_file> [mode]

# mode: all (기본) | gpu_only | hybrid | compare
```

### 실행 예시

```bash
# 전체 평가 (GPU-only → Hybrid → 비교 리포트)
./run_eval.sh envs/h100x8.env

# GPU-only만
./run_eval.sh envs/h100x8.env gpu_only

# Hybrid만
./run_eval.sh envs/h100x8.env hybrid

# 기존 결과로 비교 리포트만 재생성
RUN_TS=20260327_120000 ./run_eval.sh envs/h100x8.env compare
```

실행 순서 (`all` 모드):
```
1. GPU-only 서버 시작
2. 모니터 시작 (GPU/CPU 활용률 1초 간격 CSV 기록)
3. 서버 헬스체크 대기 (/health 응답까지)
4. GPU-only 벤치마크 실행
5. 모니터 종료, 서버 종료
6. Hybrid 서버 시작 (GPU + CPU NUMA 엔진들)
7. 모니터 시작
8. 서버 헬스체크 대기
9. Hybrid 벤치마크 실행
10. 모니터 종료, 서버 종료
11. 비교 리포트 생성
```

---

## serve.sh / benchmark.sh 단독 실행

env 파일을 환경변수 또는 인자로 전달:

```bash
# serve.sh: ./serve.sh <mode> [env_file]
./serve.sh gpu_only envs/h100x1.env
./serve.sh hybrid   envs/h100x8.env

# benchmark.sh: ./benchmark.sh <label> [env_file]
./benchmark.sh gpu_only envs/h100x1.env
./benchmark.sh hybrid   envs/h100x8.env

# 또는 환경변수로
EVAL_ENV_FILE=envs/h100x8.env ./serve.sh hybrid
EVAL_ENV_FILE=envs/h100x8.env ./benchmark.sh gpu_only
```

---

## env 파일 설정 가이드

### 환경별 주요 차이

| 파일 | 모델 | TP | CPU 엔진 | NUMA |
|------|------|-----|----------|------|
| `dev_rtx3090.env` | Qwen2.5-1.5B | 1 | 1 | false |
| `h100x1.env` | Qwen2.5-7B | 1 | 1 | true |
| `h100x8.env` | Qwen2.5-72B | 8 | 2 | true |
| `a100x1.env` | Qwen2.5-7B | 1 | 1 | true |

### 전체 파라미터 설명

```bash
# ── 모델 ──────────────────────────────────────────────────────────
MODEL=Qwen/Qwen2.5-1.5B   # HuggingFace 모델 ID
PORT=8000                  # 서버 포트

# ── 서버 공통 ─────────────────────────────────────────────────────
GPU_MEMORY_UTIL=0.9        # GPU 메모리 사용률 (0.0~1.0)
TENSOR_PARALLEL_SIZE=1     # GPU 수 (H100 x8이면 8)
TRANSFORMERS_OFFLINE=1     # HuggingFace 오프라인 모드
HF_DATASETS_OFFLINE=1

# ── 벤치마크 ──────────────────────────────────────────────────────
NUM_PROMPTS=2000           # 총 요청 수
INPUT_LEN=128              # 입력 토큰 길이
OUTPUT_LEN=512             # 출력 토큰 길이
REQUEST_RATE=inf           # 요청 속도 (inf = 최대)

# ── Hybrid 전용 설정 ───────────────────────────────────────────────
# 0이면 _resolve_cpu_params()가 하드웨어에 맞게 자동 계산 (권장)
HYBRID_CPU_MAX_SEQS=0      # CPU 엔진당 최대 동시 시퀀스 수
HYBRID_CPU_KVCACHE_GB=0    # CPU KV 캐시 메모리(GB)
HYBRID_CPU_THREADS=0       # CPU 스레드 수
HYBRID_NUMA_AWARE=true     # NUMA 최적화 (멀티소켓: true)
HYBRID_NUM_CPU_ENGINES=1   # CPU 엔진 프로세스 수 (듀얼소켓: 2)
HYBRID_STATS_LOG_INTERVAL=50
HYBRID_ROUTING_STRATEGY=capacity  # capacity / length-aware / throughput-adaptive

# ── 모니터링 ──────────────────────────────────────────────────────
MONITOR_INTERVAL=1         # GPU/CPU 샘플링 간격 (초)

# ── 결과 저장 ─────────────────────────────────────────────────────
RESULTS_DIR=results

# ── 서버 준비 대기 ────────────────────────────────────────────────
SERVER_READY_TIMEOUT=300   # 서버 시작 최대 대기 시간(초)
SERVER_READY_POLL=3        # 준비 확인 폴링 간격(초)
```

### 라우팅 전략 (`HYBRID_ROUTING_STRATEGY`)

| 전략 | 동작 | 권장 상황 |
|------|------|----------|
| `capacity` | GPU 포화 시 CPU로 overflow | **기본값, 대부분 상황** |
| `length-aware` | GPU 포화 + 짧은 프롬프트만 CPU | 긴 프롬프트가 섞인 workload |
| `throughput-adaptive` | EMA 처리량 기반 동적 CPU 슬롯 조정 | 처리량 편차가 큰 workload |

---

## 결과 파일 구조

`run_eval.sh` 실행 시 `results/<YYYYMMDD_HHMMSS KST>/` 하위 디렉토리에 저장.
`results/latest` 심볼릭 링크가 가장 최근 실행을 가리킨다.

```
results/
├── latest -> 20260327_120000/
├── 20260327_120000/
│   ├── gpu_only.json              # GPU-only 벤치마크 결과
│   ├── hybrid.json                # Hybrid 벤치마크 결과
│   ├── gpu_only_monitor_gpu.csv   # GPU-only GPU 활용률 시계열
│   ├── gpu_only_monitor_cpu.csv   # GPU-only CPU 활용률 시계열
│   ├── hybrid_monitor_gpu.csv     # Hybrid GPU 활용률 시계열
│   ├── hybrid_monitor_cpu.csv     # Hybrid CPU 활용률 시계열
│   ├── gpu_only_serve.log
│   ├── hybrid_serve.log
│   ├── gpu_only_bench.log
│   ├── hybrid_bench.log
│   ├── comparison.txt             # 텍스트 비교 리포트
│   └── comparison.json            # JSON 비교 리포트
└── 20260327_150000/
    └── ...
```

---

## H100 x8 + 듀얼소켓 10-PE 구성

```
HybridAsyncMPClient (CapacityAwareRouter)
│
├── GPU EngineCoreProc [PID A]  engine_index=0
│   └── MultiprocExecutor (TP=8, H100 x8)
│
├── CPU EngineCoreProc [PID B]  engine_index=1, NUMA node=0
│   └── UniProcExecutor (CPUWorker, Xeon 소켓 0, 56코어)
│
└── CPU EngineCoreProc [PID C]  engine_index=2, NUMA node=1
    └── UniProcExecutor (CPUWorker, Xeon 소켓 1, 56코어)

라우팅 (capacity 전략):
  GPU 슬롯 여유  → gpu
  GPU 포화, CPU0 여유 → cpu:0
  GPU 포화, CPU1 여유 → cpu:1
```

```bash
# 실행 중인 엔진 프로세스 확인
ps aux | grep -E "GPU_EngineCore|CPU_EngineCore"
```

---

## 트러블슈팅

### 서버 시작 타임아웃
```bash
SERVER_READY_TIMEOUT=600   # .env에서 늘리기
```

### GPU 메모리 부족
```bash
pkill -f "vllm.entrypoints.openai.api_server"
sleep 5
# .env에서 GPU_MEMORY_UTIL=0.7
```

### Hybrid 서버에서 CPU 엔진 시작 실패
```bash
tail -100 results/latest/hybrid_serve.log | grep -E "ERROR|CPU_EngineCore"
# IPEX 설치로 성능 향상:
pip install intel-extension-for-pytorch==2.7.0
```

### 벤치마크 결과가 GPU-only보다 낮은 경우
```bash
grep "=== Stats ===" results/latest/hybrid_serve.log
# cpu_ratio가 0%에 가까우면 CPU 처리 불가 상태
# → AVX-512 지원 CPU(Xeon 8480+)에서 재실행 필요
```

---

## 의존성

```
psutil       # CPU 활용률 모니터링
nvidia-smi   # GPU 활용률 (NVIDIA 드라이버 포함)
curl         # 서버 헬스체크
```

선택:
```
intel-extension-for-pytorch==2.7.0   # CPU 추론 최적화 (IPEX)
```
