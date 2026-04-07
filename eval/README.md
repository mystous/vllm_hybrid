# vLLM Hybrid Evaluation Scripts

GPU-only vs Hybrid(GPU+CPU) 서빙 성능을 비교하는 평가 스크립트 모음.

---

## 파일 구조

```
eval/
├── envs/                    # 환경별 설정 파일
│   ├── dev_rtx3090.env
│   ├── dev_rtx3090_cpu_first.env
│   ├── dev_rtx3090_gpu_first.env
│   ├── dev_rtx3090_rr.env
│   ├── h100x4.env
│   ├── h100x4_cpu_first.env
│   ├── h100x4_gpu_first.env
│   ├── h100x4_rr.env
│   └── ...
├── serve.sh        # 1단계: vLLM 서버 실행 (foreground, 로그 실시간 출력)
├── bench.sh        # 2단계: 벤치마크 실행 + 모니터링 + 결과 저장
├── compare.sh      # 3단계: N개 결과 비교 리포트 생성
├── monitor.py      # GPU/CPU 활용률 백그라운드 모니터 (CSV)
├── compare.py      # compare.sh에서 호출하는 Python 스크립트
├── run_eval.sh     # (레거시) 전체 파이프라인 자동 실행
└── results/        # 결과 저장 디렉토리
```

---

## 빠른 시작

3개 스크립트를 순서대로 사용합니다.

### 1단계: 서버 실행 (`serve.sh`)

```bash
./serve.sh <mode> <env_file>
# mode: gpu_only | hybrid
```

vLLM 서버를 foreground로 실행합니다. 로그가 터미널에 그대로 출력되므로 모델 로딩, CPU 엔진 초기화, 서비스 준비 상태를 직접 확인할 수 있습니다.

```bash
# Terminal 1
./serve.sh hybrid envs/h100x4_cpu_first.env
```

`INFO: Application startup complete` 또는 `Uvicorn running on http://0.0.0.0:8000` 메시지가 나오면 서버 준비 완료.

### 2단계: 벤치마크 실행 (`bench.sh`)

```bash
./bench.sh <mode> <env_file>
# mode: gpu_only | hybrid (serve.sh와 동일하게)
```

서버가 실행 중인 상태에서 별도 터미널에서 실행합니다.

```bash
# Terminal 2
./bench.sh hybrid envs/h100x4_cpu_first.env
```

bench.sh가 하는 일:
1. 모델 캐시 확인 (없으면 다운로드 안내 후 종료)
2. 서버 health 확인 (서버가 안 떠있으면 에러 + 안내 후 종료)
3. 타임스탬프+HW 태그 결과 디렉토리 생성 (예: `results/20260407_180242_H100_80GB_HBM3_x4_Qwen2.5-7B-Instruct/`)
4. system_info.json 수집 (HW, SW, hybrid_config 기록)
5. GPU/CPU 모니터 시작 (1초 간격 CSV)
6. 벤치마크 실행 (`benchmark_serving.py`)
7. 모니터 종료, 결과 저장

### 3단계: 비교 리포트 (`compare.sh`)

```bash
./compare.sh <result_dir1> <result_dir2> [result_dir3 ...] [-o output_dir]
```

N개 결과 디렉토리를 받아 비교 리포트를 생성합니다.

```bash
# 2개 비교
./compare.sh results/20260407_175848_* results/20260407_180242_*

# glob으로 여러 개 비교
./compare.sh results/20260407_*

# 출력 디렉토리 지정
./compare.sh results/20260407_* -o results/comparison_0407
```

`[0]` 런이 기준이 되어 나머지와의 차이(`vs [0]`)를 표시합니다.

---

## 전체 워크플로우 예시

```bash
cd /workspace/vllm_hybrid/eval

# === GPU-only 벤치마크 ===
# Terminal 1: 서버
./serve.sh gpu_only envs/h100x4.env
# Terminal 2: 벤치마크 (서버 ready 확인 후)
./bench.sh gpu_only envs/h100x4.env
# Terminal 1: Ctrl+C로 서버 종료

# === Hybrid cpu-first 벤치마크 ===
# Terminal 1: 서버
./serve.sh hybrid envs/h100x4_cpu_first.env
# Terminal 2: 벤치마크
./bench.sh hybrid envs/h100x4_cpu_first.env
# Terminal 1: Ctrl+C

# === Hybrid gpu-first 벤치마크 ===
# Terminal 1: 서버
./serve.sh hybrid envs/h100x4_gpu_first.env
# Terminal 2: 벤치마크
./bench.sh hybrid envs/h100x4_gpu_first.env
# Terminal 1: Ctrl+C

# === 3개 결과 비교 ===
./compare.sh results/20260407_*
```

---

## env 파일 설정

### 환경별 변형 패턴

각 하드웨어별로 4가지 변형이 존재합니다:

| 파일 | 전략 | 우선순위 | 설명 |
|------|------|---------|------|
| `h100x4.env` | throughput-adaptive | gpu-first | 기본 (원본) |
| `h100x4_cpu_first.env` | capacity | cpu-first | CPU 우선 |
| `h100x4_gpu_first.env` | capacity | gpu-first | GPU 우선 |
| `h100x4_rr.env` | round-robin | (무시) | 교대 분배 |

### 주요 파라미터

```bash
# ── 모델 ──
MODEL=Qwen/Qwen2.5-7B-Instruct
PORT=8000

# ── 서버 ──
GPU_MEMORY_UTIL=0.9
TENSOR_PARALLEL_SIZE=4

# ── 벤치마크 ──
NUM_PROMPTS=500
INPUT_LEN=128
OUTPUT_LEN=512
REQUEST_RATE=inf

# ── Hybrid 라우팅 ──
HYBRID_ROUTING_STRATEGY=capacity       # capacity | length-aware | throughput-adaptive | round-robin
HYBRID_ROUTING_PRIORITY=cpu-first      # cpu-first | gpu-first (round-robin시 무시)
HYBRID_CPU_MAX_SEQS=1                  # 전체 코어가 1개 요청에 집중

# ── Hybrid CPU ──
HYBRID_CPU_KVCACHE_GB=0               # 0=auto
HYBRID_CPU_THREADS=0                  # 0=auto
HYBRID_NUMA_AWARE=false
HYBRID_NUM_CPU_ENGINES=1

# ── 모니터링 ──
MONITOR_INTERVAL=1
```

### 라우팅 전략

| 전략 | 동작 |
|------|------|
| `capacity` + `cpu-first` | CPU 슬롯 먼저 채움, 가득차면 GPU |
| `capacity` + `gpu-first` | GPU 먼저 채움, 포화 시 CPU |
| `round-robin` | GPU/CPU 교대 분배 (priority 무시) |
| `throughput-adaptive` | EMA 처리량 기반 동적 CPU 슬롯 조정 |
| `length-aware` | 짧은 프롬프트만 CPU 허용 |

### `HYBRID_CPU_MAX_SEQS`

| 값 | 동작 |
|----|------|
| `1` | 전체 코어가 1개 요청에 집중 (권장) |
| `0` | 자동 감지 (`cores/4`) — 요청당 코어 분산 |

---

## 결과 파일 구조

```
results/20260407_180242_H100_80GB_HBM3_x4_Qwen2.5-7B-Instruct/
├── hybrid.json                # 벤치마크 결과 (또는 gpu_only.json)
├── hybrid_bench.log           # 벤치마크 콘솔 로그
├── hybrid_monitor_gpu.csv     # GPU 활용률 시계열
├── hybrid_monitor_cpu.csv     # CPU 활용률 시계열
├── monitor_hybrid.log         # 모니터 로그
└── system_info.json           # HW/SW 정보 + hybrid_config
```

compare.sh 실행 시 추가:
```
├── comparison.txt             # 텍스트 비교 리포트
└── comparison.json            # JSON 비교 데이터
```

---

## 트러블슈팅

### 서버가 멈춘 것 같을 때
serve.sh는 foreground 실행이므로 로그를 직접 확인합니다.
- `Loading model weights` → 모델 로딩 중 (대형 모델은 수 분 소요)
- `CPU_EngineCore` 로그 → CPU 엔진 초기화 중
- `Application startup complete` → 서버 준비 완료

### bench.sh에서 서버가 없다고 나올 때
```
[ERROR] Server not running on port 8000
```
→ Terminal 1에서 serve.sh가 실행 중인지, 서버가 준비 완료되었는지 확인

### GPU 메모리 부족
```bash
pkill -f "vllm.entrypoints.openai.api_server"
# .env에서 GPU_MEMORY_UTIL=0.7로 낮추기
```

---

## 의존성

```
psutil       # CPU 활용률 모니터링
nvidia-smi   # GPU 활용률 (NVIDIA 드라이버)
curl         # 서버 헬스체크
```

선택:
```
intel-extension-for-pytorch   # CPU 추론 최적화 (IPEX)
```
