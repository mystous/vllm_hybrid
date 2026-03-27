# vLLM Hybrid Evaluation Scripts

GPU-only vs Hybrid(GPU+CPU) 서빙 성능을 자동으로 비교하는 평가 스크립트 모음.
GPU/CPU 활용률을 실시간으로 측정(CSV)하고, 벤치마크 결과를 비교 리포트로 출력한다.

---

## 파일 구조

```
eval/
├── .env            # 모든 파라미터 설정 (여기만 수정하면 됨)
├── serve.sh        # vLLM 서버 시작 (gpu / hybrid 모드)
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

# 1. 환경 설정 (.env 편집)
vi .env

# 2. 전체 평가 실행 (GPU-only → Hybrid → 비교)
./run_eval.sh
# 결과는 results/20260327_120000/ 형태의 타임스탬프 디렉토리에 저장됨

# 결과 확인 (latest 심볼릭 링크 사용)
cat results/latest/comparison.txt
```

---

## .env 설정 가이드

모든 파라미터는 `.env` 파일에서 관리한다.

### 필수 변경 항목

| 변수 | 설명 | 개발머신 예 | H100 서버 예 |
|------|------|------------|-------------|
| `MODEL` | HuggingFace 모델 ID | `Qwen/Qwen2.5-1.5B` | `meta-llama/Llama-3-70B` |
| `TENSOR_PARALLEL_SIZE` | GPU 수 (TP) | `1` | `8` |
| `HYBRID_NUM_CPU_ENGINES` | CPU 엔진 수 | `1` | `2` |
| `HYBRID_NUMA_AWARE` | NUMA 최적화 | `false` | `true` |

### 전체 파라미터 설명

```bash
# ── 모델 ──────────────────────────────────────────────────────────
MODEL=Qwen/Qwen2.5-1.5B   # HuggingFace 모델 ID
PORT=8000                  # 서버 포트

# ── 서버 공통 ─────────────────────────────────────────────────────
GPU_MEMORY_UTIL=0.9        # GPU 메모리 사용률 (0.0~1.0)
TENSOR_PARALLEL_SIZE=1     # GPU 수 (H100 x8이면 8)
TRANSFORMERS_OFFLINE=1     # HuggingFace 오프라인 모드 (네트워크 차단 환경)
HF_DATASETS_OFFLINE=1      # HuggingFace Datasets 오프라인 모드

# ── 벤치마크 ──────────────────────────────────────────────────────
NUM_PROMPTS=2000           # 총 요청 수 (많을수록 GPU 포화 확률 ↑)
INPUT_LEN=128              # 입력 토큰 길이 (random dataset)
OUTPUT_LEN=512             # 출력 토큰 길이 (길수록 GPU 더 오래 점유)
REQUEST_RATE=inf           # 요청 속도 (inf = 최대 속도)

# ── Hybrid 전용 설정 ───────────────────────────────────────────────
# 0이면 _resolve_cpu_params()가 하드웨어에 맞게 자동 계산 (권장)
HYBRID_CPU_MAX_SEQS=0      # CPU 엔진당 최대 동시 시퀀스 수
HYBRID_CPU_KVCACHE_GB=0    # CPU KV 캐시 메모리(GB)
HYBRID_CPU_THREADS=0       # CPU 스레드 수
HYBRID_NUMA_AWARE=true     # NUMA 최적화 (멀티소켓 서버는 true)
HYBRID_NUM_CPU_ENGINES=1   # CPU 엔진 프로세스 수 (듀얼소켓은 2)
HYBRID_STATS_LOG_INTERVAL=10   # 통계 로깅 간격 (완료 요청 수)
HYBRID_ROUTING_STRATEGY=capacity  # 라우팅 전략 (아래 참조)

# ── 모니터링 ──────────────────────────────────────────────────────
MONITOR_INTERVAL=1         # GPU/CPU 샘플링 간격 (초)

# ── 결과 저장 ─────────────────────────────────────────────────────
RESULTS_DIR=./results      # 결과 저장 경로

# ── 서버 준비 대기 ────────────────────────────────────────────────
SERVER_READY_TIMEOUT=300   # 서버 시작 최대 대기 시간(초)
SERVER_READY_POLL=3        # 준비 확인 폴링 간격(초)
```

### 라우팅 전략 (`HYBRID_ROUTING_STRATEGY`)

| 전략 | 동작 | 권장 상황 |
|------|------|----------|
| `capacity` | GPU 포화 시 CPU로 overflow, 여유 슬롯 많은 CPU 엔진 선택 | **기본값, 대부분 상황** |
| `length-aware` | GPU 포화 + 짧은 프롬프트(`≤ prefill_threshold`)만 CPU | 긴 프롬프트가 섞인 workload |
| `throughput-adaptive` | EMA 처리량 기반 동적 CPU 슬롯 조정 | 처리량 편차가 큰 workload |

---

## 환경별 .env 설정 예시

### 개발 머신 (i9-12900KF + RTX 3090)

```bash
MODEL=Qwen/Qwen2.5-1.5B
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTIL=0.9
HYBRID_NUM_CPU_ENGINES=1
HYBRID_NUMA_AWARE=false
HYBRID_CPU_MAX_SEQS=4      # 수동 지정 (auto는 너무 작게 잡힐 수 있음)
HYBRID_CPU_KVCACHE_GB=4
HYBRID_CPU_THREADS=8
NUM_PROMPTS=2000
OUTPUT_LEN=512
```

### H100 서버 (H100 x8 + Xeon 8480+ 듀얼소켓)

```bash
MODEL=meta-llama/Llama-3-70B   # 또는 원하는 모델
TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTIL=0.9
HYBRID_NUM_CPU_ENGINES=2       # NUMA 노드 0 + NUMA 노드 1
HYBRID_NUMA_AWARE=true
HYBRID_CPU_MAX_SEQS=0          # auto: Xeon 56코어/4 = 14 seqs/엔진
HYBRID_CPU_KVCACHE_GB=0        # auto: NUMA 노드 메모리 * 0.4
HYBRID_CPU_THREADS=0           # auto: NUMA 노드 물리 코어 수
NUM_PROMPTS=5000
OUTPUT_LEN=512
```

---

## 스크립트 사용법

### run_eval.sh — 전체 파이프라인

```bash
# GPU-only + Hybrid 모두 실행 후 비교 (기본)
./run_eval.sh

# GPU-only만 실행
./run_eval.sh gpu_only

# Hybrid만 실행
./run_eval.sh hybrid

# 이미 실행된 결과로 비교만 생성
./run_eval.sh compare
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

### serve.sh — 서버 단독 시작

```bash
# GPU-only 서버
./serve.sh gpu_only

# Hybrid 서버 (HYBRID_NUM_CPU_ENGINES=2이면 CPU 엔진 2개 스폰)
./serve.sh hybrid
```

서버 로그는 `results/gpu_serve.log`, `results/hybrid_serve.log`에 저장된다.

---

### benchmark.sh — 벤치마크 단독 실행

```bash
# 서버가 이미 실행 중일 때 사용
# 단독 실행 시 results/<YYYYMMDD_HHMMSS>/ 에 자동 저장
./benchmark.sh gpu_only    # results/20260327_120000/gpu_only.json
./benchmark.sh hybrid      # results/20260327_120000/hybrid.json

# run_eval.sh 가 호출할 때는 EVAL_RUN_DIR 환경변수로 디렉토리 공유
# (별도 조작 불필요)
```

---

### monitor.py — 활용률 모니터 단독 실행

```bash
# 백그라운드 실행
python monitor.py results/my_run --interval 1 &

# 종료
kill $!
```

출력 파일:
- `results/my_run_gpu.csv` — GPU 카드별 + 종합 평균
- `results/my_run_cpu.csv` — 물리 코어별 + 종합 평균

---

### compare.py — 비교 리포트 단독 생성

```bash
# 기본 (results/ 폴더에서 gpu_only.json, hybrid.json 읽기)
python compare.py

# 경로/label 지정
python compare.py --results-dir /path/to/results \
                  --gpu-label gpu_only \
                  --hybrid-label hybrid
```

---

## 결과 파일 구조

`run_eval.sh` 실행 시 `results/<YYYYMMDD_HHMMSS>/` 하위 디렉토리에 모든 결과를 저장한다.
`results/latest` 심볼릭 링크가 가장 최근 실행을 가리킨다.

```
results/
├── latest -> 20260327_120000/      # 가장 최근 실행 심볼릭 링크
├── 20260327_120000/                # 1차 실행
│   ├── gpu_only.json
│   ├── hybrid.json
│   ├── gpu_only_monitor_gpu.csv
│   ├── gpu_only_monitor_cpu.csv
│   ├── hybrid_monitor_gpu.csv
│   ├── hybrid_monitor_cpu.csv
│   ├── gpu_only_serve.log
│   ├── hybrid_serve.log
│   ├── gpu_only_bench.log
│   ├── hybrid_bench.log
│   ├── comparison.txt
│   └── comparison.json
└── 20260327_150000/                # 2차 실행 (파라미터 변경 후 재실행 등)
    └── ...
```

최신 결과 확인:
```bash
cat results/latest/comparison.txt
ls results/latest/
```

특정 실행 비교:
```bash
# RUN_TS를 직접 지정하여 재실행 없이 비교만
RUN_TS=20260327_120000 ./run_eval.sh compare
```

### GPU CSV 컬럼 (gpu_only_monitor_gpu.csv)

| 컬럼 | 설명 |
|------|------|
| `timestamp` | 측정 시각 (YYYY-MM-DD HH:MM:SS.mmm) |
| `elapsed_s` | 시작으로부터 경과 시간(초) |
| `gpu0_util_pct` | GPU 0 연산 활용률 (%) |
| `gpu0_mem_util_pct` | GPU 0 메모리 대역폭 활용률 (%) |
| `gpu0_mem_used_mb` | GPU 0 사용 메모리 (MB) |
| `gpu0_power_w` | GPU 0 소비 전력 (W) |
| `gpu0_temp_c` | GPU 0 온도 (°C) |
| `gpu1_*` ~ `gpu7_*` | GPU 1~7 동일 (H100 x8이면 8개 카드) |
| `gpu_avg_util_pct` | 전체 GPU 평균 활용률 |
| `gpu_avg_mem_util_pct` | 전체 GPU 평균 메모리 활용률 |
| `gpu_avg_power_w` | 전체 GPU 합산 전력 (W) |

### CPU CSV 컬럼 (gpu_only_monitor_cpu.csv)

| 컬럼 | 설명 |
|------|------|
| `timestamp` | 측정 시각 |
| `elapsed_s` | 경과 시간(초) |
| `core0_util_pct` ~ `coreN_util_pct` | 물리 코어별 활용률 (%) |
| `cpu_avg_util_pct` | 전체 물리 코어 평균 활용률 |
| `cpu_mem_used_mb` | 시스템 사용 메모리 (MB) |
| `cpu_mem_avail_mb` | 시스템 가용 메모리 (MB) |

> HT(하이퍼스레딩) 활성화 환경에서는 논리 코어 쌍을 평균내어 물리 코어 활용률로 보고.
> i9-12900KF처럼 P+E 혼합 아키텍처는 논리 코어 수를 그대로 사용.

### comparison.json 구조

```json
{
  "generated_at": "2026-03-27T12:00:00",
  "gpu_only": { "request_throughput": 19.5, "output_throughput": 9799, ... },
  "hybrid":   { "request_throughput": 18.7, "output_throughput": 9405, ... },
  "comparison": {
    "request_throughput": { "diff_pct": -4.1, "direction": "higher_better" },
    "request_throughput_speedup": 0.959,
    "ttft_gain_pct": -4.1
  },
  "gpu_utilization": {
    "gpu_only": { "gpu0_util_pct": { "mean": 95.2, "max": 100.0, "min": 78.0 }, ... },
    "hybrid":   { ... }
  },
  "cpu_utilization": {
    "gpu_only": { "cpu_avg_util_pct": { "mean": 5.1, ... }, ... },
    "hybrid":   { "cpu_avg_util_pct": { "mean": 42.3, ... }, ... }
  }
}
```

---

## 그래프 생성 예시 (pandas + matplotlib)

```python
import pandas as pd
import matplotlib.pyplot as plt

RUN = "results/latest"   # 또는 "results/20260327_120000"

# GPU 활용률 시계열 비교
gpu_df = pd.read_csv(f"{RUN}/gpu_only_monitor_gpu.csv")
hyb_df = pd.read_csv(f"{RUN}/hybrid_monitor_gpu.csv")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 1. GPU 활용률
axes[0].plot(gpu_df["elapsed_s"], gpu_df["gpu_avg_util_pct"], label="GPU-only")
axes[0].plot(hyb_df["elapsed_s"], hyb_df["gpu_avg_util_pct"], label="Hybrid")
axes[0].set_ylabel("GPU Util (%)")
axes[0].legend()

# 2. GPU 전력
axes[1].plot(gpu_df["elapsed_s"], gpu_df["gpu_avg_power_w"], label="GPU-only")
axes[1].plot(hyb_df["elapsed_s"], hyb_df["gpu_avg_power_w"], label="Hybrid")
axes[1].set_ylabel("GPU Power (W)")
axes[1].legend()

# 3. CPU 활용률 비교 (Hybrid에서 CPU가 얼마나 쓰이는지)
cpu_g = pd.read_csv(f"{RUN}/gpu_only_monitor_cpu.csv")
cpu_h = pd.read_csv(f"{RUN}/hybrid_monitor_cpu.csv")
axes[2].plot(cpu_g["elapsed_s"], cpu_g["cpu_avg_util_pct"], label="GPU-only")
axes[2].plot(cpu_h["elapsed_s"], cpu_h["cpu_avg_util_pct"], label="Hybrid")
axes[2].set_ylabel("CPU Util (%)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{RUN}/utilization_comparison.png", dpi=150)
```

---

## H100 서버 10-PE 구성 상세

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
  GPU 포화, CPU0 여유 → cpu:0  (NUMA 0, 소켓 0)
  GPU 포화, CPU1 여유 → cpu:1  (NUMA 1, 소켓 1)
  CPU0 포화 → CPU1 / GPU 로 분산
```

프로세스 확인:
```bash
ps aux | grep -E "GPU_EngineCore|CPU_EngineCore"
# GPU_EngineCore_0
# CPU_EngineCore_1  (NUMA 0)
# CPU_EngineCore_2  (NUMA 1)
```

---

## 트러블슈팅

### 서버 시작 타임아웃

```bash
# SERVER_READY_TIMEOUT 늘리기 (기본 300초)
# .env에서:
SERVER_READY_TIMEOUT=600
```

모델이 크거나 CPU KV 캐시 할당이 오래 걸리면 타임아웃 발생.

### GPU 메모리 부족

```bash
# 이전 서버 프로세스 정리
pkill -f "vllm.entrypoints.openai.api_server"
sleep 5

# GPU 메모리 사용률 낮추기
GPU_MEMORY_UTIL=0.7
```

### Hybrid 서버에서 CPU 엔진 시작 실패

서버 로그 확인:
```bash
tail -100 results/hybrid_serve.log | grep -E "ERROR|CPU_EngineCore"
```

IPEX 없이도 동작하지만 CPU 성능은 낮아진다. `intel-extension-for-pytorch` 설치:
```bash
pip install intel-extension-for-pytorch==2.7.0
```

### NUMA 바인딩 실패 (HYBRID_NUM_CPU_ENGINES=2)

`libnuma`가 없거나 NUMA 노드가 1개인 환경에서는 자동으로 NUMA 바인딩 없이 동작한다.
서버 로그에 `"disabling per-engine NUMA binding"` 메시지 확인.

### 벤치마크 결과가 GPU-only보다 Hybrid가 낮은 경우

AVX2-only CPU(예: i9-12900KF)에서는 CPU 추론 속도가 GPU보다 느려 오히려 오버헤드 발생.
`--hybrid-stats-log-interval`로 CPU 실제 기여도 확인:
```bash
grep "=== Stats ===" results/hybrid_serve.log
```
`cpu_ratio`가 0%에 가까우면 CPU가 처리를 못 하고 있는 것.
AVX-512/AMX 지원 CPU(Xeon 8480+)에서 재실행 필요.

---

## 의존성

```
# Python
psutil          # CPU 활용률 모니터링
# 시스템
nvidia-smi      # GPU 활용률 모니터링 (NVIDIA 드라이버 포함)
curl            # 서버 헬스체크
```

선택적 의존성 (성능 향상):
```
intel-extension-for-pytorch==2.7.0   # CPU GEMM/Attention 최적화 (IPEX)
```
