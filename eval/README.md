# vLLM Hybrid Evaluation Scripts

GPU-only vs Hybrid(GPU+CPU) 서빙 성능 측정 + CPU 커널 마이크로벤치 + G0 Ninja Gap 측정을 위한 스크립트 모음.

## 디렉토리 구조

```
eval/
├── README.md                       # 본 파일
│
├── serve.sh                        # vLLM 서버 기동 (foreground)
├── bench.sh                        # 벤치마크 + 모니터 + 결과 수집
├── monitor.py                      # 1Hz per-logical-CPU / per-GPU 샘플러 (CSV)
│
├── cpu_profile.sh                  # CPU-only 마이크로벤치 (thread sweep, kernel 직접 호출)
├── cpu_profile_dev.sh              # dev 머신 축소판
│
├── hw_inspect.sh / hw_inspect.py   # 단일 run 디렉토리 HW 이용률 요약
├── g0_analyze.py                   # G0 sweep 디렉토리 batch scaling 분석
├── compare.sh / compare.py         # N개 run 비교 리포트
│
├── run_eval.sh                     # (레거시) 전체 파이프라인 자동 실행
├── benchmark.sh                    # 내부 호출용 bench wrapper
│
├── analysis*.ipynb                 # Jupyter 탐색 노트북 (분석 스크립트와 동일 계산의 interactive 버전)
│
├── envs/                           # env 파일 (model, GPU, hybrid flag 등)
│   ├── dev_rtx3090_qwen*.env       # dev (RTX3090 + i9-12900KF)
│   ├── h100x1/4/8_qwen*.env        # H100 다양한 구성
│   ├── g0_dev_rtx3090_qwen1.5b.env # G0 측정 template (dev)
│   ├── g0_h100x8_qwen7b.env        # G0 측정 template (H100x8)
│   └── backup/                     # 구버전 env
│
├── results/                        # bench.sh 기본 출력: <ts>_TAG_GPU_MODEL/
├── serve_logs/                     # serve.sh tee log
├── analysis_log/                   # cpu_profile*.sh 출력
│
├── basic/                          # 논문/commit 된 baseline 실측 데이터 (H100x8, RTX3090)
├── RTX3090/                        # dev 분석 노트북 + 이전 결과
└── h100x8/                         # H100 분석 노트북 + 이전 결과
```

---

## 핵심 워크플로 3종

### A. 일반 벤치 (GPU-only / Hybrid 비교)

```bash
# 터미널 1 — 서버
./serve.sh hybrid envs/h100x4_qwen7b_hybrid_wave.env

# 터미널 2 — 벤치 (서버 ready 후)
./bench.sh hybrid envs/h100x4_qwen7b_hybrid_wave.env
# → results/<ts>_H_C_H100_80GB_HBM3_x4_Qwen2.5-7B-Instruct/ 에 저장

# 하드웨어 이용률 요약
./hw_inspect.sh results/<ts>_.../
```

### B. G0 Ninja Gap 측정 (sublayer breakdown + batch scaling)

```bash
# 1. G0 template 복사 + 편집
cp envs/g0_h100x8_qwen7b.env /tmp/run.env
#    편집: HYBRID_TODO_NN, HYBRID_CPU_MAX_SEQS (sweep 값)

# 2. serve + bench (기본과 동일) — PROFILE=1 이라 부가 파일 자동 포함
./serve.sh hybrid /tmp/run.env
./bench.sh hybrid /tmp/run.env

# 3. HYBRID_CPU_MAX_SEQS 값 바꿔가며 반복 (1, 2, 4, 8, 16)

# 4. 수동으로 sweep 단위 정리
mv results/<ts1>_... ../measurement_results/H100x8/g0_00/seqs1
mv results/<ts2>_... ../measurement_results/H100x8/g0_00/seqs2
# ...

# 5. Post-processing
python3 g0_analyze.py ../measurement_results/H100x8/g0_00/
# → <sweep>/analysis_summary.png / _sublayer_scaling.png / _bench.png / _summary.md
```

상세: `../NinjaGap_Todo/01_G0_measurement.md`

### C. CPU-only 마이크로벤치 (thread sweep, kernel 성능)

```bash
./cpu_profile.sh          # H100/SPR 용 풀 sweep (thread 32/56/76/96/112/... × kernel 시험)
./cpu_profile_dev.sh      # dev 용 축소판 (thread 1/2/4/8/12/16/24 × 간단 kernel)
# → analysis_log/<ts>_cpu_profile[_dev]/ 에 저장
```

서버 없이 CPU kernel 만 직접 호출하는 low-level 측정. G0 측정 (서빙 흐름) 과 **독립**.

---

## 스크립트 상세

### serve.sh

vLLM OpenAI API 서버 기동. `MODE` = `gpu_only` / `hybrid`.

- env 파일의 `MODEL`, `TENSOR_PARALLEL_SIZE`, `HYBRID_*`, `VLLM_HYBRID_*` 읽어 CLI 인자 구성
- stdout/stderr 을 `serve_logs/server_<ts>_<MODE>.log` 로 **tee** 복제 (bench.sh 가 slice 해감)
- `VLLM_HYBRID_PROFILE=1` 시 `serve_logs/profile_latest/` 에 manifest (env_snapshot, git_sha, applied_features.json) staging

### bench.sh

`benchmark_serving.py` 를 호출해 벤치 수행 + 서버 로그 수집 + 모니터 결과 병합.

- `results/<ts>_<MODE>_<PRIORITY>_<GPU>_x<N>_<MODEL>/` 디렉토리 생성 (항상)
- 생성물:
  - `hybrid.json` (또는 `gpu_only.json`) — benchmark_serving.py 결과 (wall, TPOT, TTFT)
  - `hybrid_bench.log` — bench 진행 로그
  - `hybrid_monitor_cpu.csv` / `_gpu.csv` — monitor.py 샘플링
  - `hybrid_server_boot.log` — serve_logs 에서 boot marker grep
  - `hybrid_server_run.log` — bench 기간의 서버 log byte-slice (PROFILE=1 이면 `[HYBRID-CPU-PROFILE]` sublayer 라인 포함)
  - `system_info.json` — lscpu / numactl / nvidia-smi 스냅샷
  - `inspect.txt` — hw_inspect 자동 호출 결과
  - (PROFILE=1) `applied_features.json`, `env_snapshot.txt`, `git_sha.txt` — serve 의 staging 에서 복사

### monitor.py

`bench.sh` 가 백그라운드로 1Hz 샘플. 자체 실행도 가능:

```bash
python3 monitor.py /path/to/output_prefix --interval 1.0
# → <prefix>_cpu.csv, <prefix>_gpu.csv
```

CPU CSV 는 **per-logical-CPU** (0..N-1) 시계열. NUMA/HT 분류는 downstream (hw_inspect.py) 에서 `system_info.json` 으로 계산.

### hw_inspect.sh / hw_inspect.py

**단일 run 디렉토리** 의 HW 이용 상황 요약.

```bash
./hw_inspect.sh results/<ts>_H_C_H100_80GB_HBM3_x8_Qwen2.5-7B-Instruct/
# → inspect.txt (stdout + 디렉토리 안에도 저장)
```

출력:
- 시스템 스펙 (CPU 모델, NUMA 토폴로지, GPU 개수)
- CPU pinning (어느 engine 이 어느 CPUs 에 bind)
- CPU util: NUMA 0/1 physical / HT sibling 분리 mean/max
- GPU util / 전력 / 온도

### g0_analyze.py

**G0 sweep 디렉토리** (내부에 `seqs1/`, `seqs2/`, ..., `seqs16/`) 를 읽어 batch scaling 분석.

```bash
python3 g0_analyze.py ../measurement_results/<HW>/g0_<NN>/
```

생성물 (sweep 루트에 저장):
- `analysis_summary.png` — sublayer stacked bar + scaling curve
- `analysis_sublayer_scaling.png` — per-sublayer scaling ratio
- `analysis_bench.png` — wall + TPOT vs num_seqs
- `analysis_summary.md` — 요약 표 + Gate 판정

각 seqs<N>/ 의 `hybrid_server_run.log` 에서 `[HYBRID-CPU-PROFILE]` sublayer 시간 파싱, `hybrid.json` 에서 wall/TPOT 집계.

### cpu_profile.sh / cpu_profile_dev.sh

CPU-only 마이크로벤치 — 서버 없이 thread count 와 kernel 성능 직접 측정.

- `cpu_profile.sh` — H100/SPR 풀 sweep (thread counts 32/56/76/96/112, 다양한 GEMM/GEMV shape)
- `cpu_profile_dev.sh` — dev 축소판 (thread 1/2/4/8/12/16/24)
- 출력: `analysis_log/<ts>_cpu_profile[_dev]/` — 각 thread count 별 tps, MFLOPS

G0 측정 (§B) 과 목적이 다름:
- `cpu_profile*` — CPU kernel 자체 성능 (서버 무관)
- G0 — 실제 서빙 흐름에서 sublayer 시간

### compare.sh / compare.py

N개 run 결과를 표로 정리:

```bash
./compare.sh results/<ts1>_... results/<ts2>_... results/<ts3>_...
```

wall, TPOT mean/p99, TTFT mean/p99 를 나란히 보여줌. G0 전용 아님 (일반 벤치 비교).

### run_eval.sh

레거시 — `serve.sh` + `bench.sh` 자동 연결. 대부분 사용자는 두 터미널 수동 실행 선호. 특정 env 에만 제약적으로 사용.

### analysis*.ipynb

Jupyter notebook — g0_analyze.py / hw_inspect 와 같은 분석을 interactive 로 수행. 여러 버전 (v1/v2/v3/v4) 은 시간순 반복 개발 흔적. 새 측정은 `g0_analyze.py` 사용 권장.

---

## env 파일 분류

| 패턴 | 용도 |
|---|---|
| `dev_rtx3090_*` | dev (RTX3090 + i9-12900KF, 1 NUMA, AVX2) |
| `h100x1_*` | 단일 H100 |
| `h100x4_*` | H100 × 4 (TP=4 권장) |
| `h100x8_*` | H100 × 8 (TP=8 or TP=4, 2 NUMA Xeon SPR) |
| `g0_*` | G0 측정 template — `HYBRID_TODO_NN`, `HYBRID_CPU_MAX_SEQS` + 21개 기법 flag |
| `*_wave.env` | wave-batch routing |
| `*_cpu_first.env` / `*_gpu_first.env` | priority 분기 |
| `*_smoke.env` | 빠른 동작 확인 (작은 NUM_PROMPTS) |
| `backup/` | 구버전 보존 |

---

## 결과 디렉토리 보존 정책

| 디렉토리 | 생성 주체 | 보존 정책 |
|---|---|---|
| `results/` | `bench.sh` — 매 run | 임시. 분석 후 사용자가 정리 |
| `serve_logs/` | `serve.sh` — tee log + profile staging | 디스크 차면 오래된 것부터 삭제 |
| `analysis_log/` | `cpu_profile*.sh` | 필요 시 유지 |
| `basic/` | 사용자가 명시적 mv — **commit 된 baseline 실측 데이터** | 영구 |
| `../measurement_results/<HW>/g0_<NN>/` | 사용자가 results/ 에서 수동 mv — **G0 sweep 결과** | 영구 |

---

## 참고

- 전체 프로젝트 구성: `../CLAUDE.md`
- G0 측정 상세: `../NinjaGap_Todo/01_G0_measurement.md`
- Ninja Gap Applied Features Log: `../README.md` §"Ninja Gap 성능 개선 추적"
