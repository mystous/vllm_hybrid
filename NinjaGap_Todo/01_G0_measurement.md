# 01. G0 — 계측 재정의 (모든 후속 기법의 전제)

**Tier**: -1 (계측)
**상태**: ⭕ 미구현
**우선순위**: 최우선 — **다른 모든 기법의 선행 조건**

---

## 왜 필요한가

현재 `T_hybrid` 가 `T_gpu_only` 대비 26–143× 느린 구체적 원인이 계측으로 분해되어 있지 않다. 가설은:
- **실패 1**: batch scaling 제로 (batch=1 3079ms → batch=16 16390ms, 5.3×. 선형 기대 16× 대비 3× 실패)
- **실패 2**: ISA 경직 (AMX 고정 시 batch=1 에서 AVX-512 대비 2.22× 손해)
- **실패 3**: Dataflow 미설계 (sublayer 8개 체인 = 독립 kernel = DDR 왕복 8회 × batch)

이 중 **어느 것이 실측에서 주된 원인인지** 확정 안 되면 kernel 투자 방향이 틀릴 수 있다. Codex 규율: "계측 전엔 어떤 gain 도 % 로 단정하지 않는다".

---

## 기술적 배경

### 측정해야 할 지표

**CPU-side scaling**:
- `step_ms(batch=1)`, `step_ms(batch=N)` — full forward 시간
- `batch_scaling_ratio = step_ms(batch=N) / step_ms(batch=1)` — 1.0 에 가까울수록 좋음 (완벽 amortization)
- `per_req_cost = step_ms / active_reqs`

**Sublayer breakdown** (매 layer 당):
- QKV projection (fused vs split)
- Attention (prefill QKV · decode single-query KV cached)
- O projection
- RMSNorm (pre-attn, post-attn)
- Gate / Up projection (SwiGLU)
- SiLU activation
- Down projection
- Residual add

**OMP 동작**:
- barrier entry/exit time per parallel region
- thread team create/destroy count (persistent region 이면 0)
- chunk schedule distribution

**Memory**:
- weight read bandwidth (`perf stat -e uncore_imc/mem_bw`)
- packing/repacking count per step (amx tile ↔ zmm)
- L2/L3 cache miss ratio (`perf stat -e l2_rqsts.miss,llc_misses`)

### Profiler 계층

Intel 엔지니어가 권장하는 3단 접근:

1. **Python-level coarse**: `cpu_worker.py` forward hook — 현재 구현됨, 세분화 필요
2. **Intel VTune Profiler**: microarchitectural top-down (frontend / backend / memory bound). [Intel VTune docs](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2024-2/)
3. **Linux perf**: hardware counter 저수준 (cache miss, dTLB load miss, 포트 이용률)

---

## 관련 참고 문헌

- **Codex playbook §6 Tier -1 계측 재정의**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md`
- **Claude 3겹 실패 모델**: `/vllm_hybrid/ideation/20260415_094130_claude_ninja_gap_comprehensive_plan.md` §1-1
- **Intel VTune Top-Down Analysis**: Yasin, A. (2014). "A top-down method for performance analysis and counters architecture." *ISPASS*. https://ieeexplore.ieee.org/document/6844459
- **perf Tutorial (Brendan Gregg)**: https://www.brendangregg.com/perf.html
- **KTransformers micro-analysis**: [SOSP'25 paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) — CPU compute limit 원인 분해 방식 참조

---

## 구체 작업

- [ ] `eval/cpu_profile*.sh` 에 `num_seqs=1/2/4/8/16` sweep 고정 (dev + H100x8 동일)
- [ ] CPU-only 와 hybrid CPU engine 동일 shape 비교 harness 구축
- [ ] `cpu_worker.py` 의 `attn/mlp` coarse hook 을 sublayer 수준으로 세분화:
  - QKV (split or fused)
  - O projection
  - RMSNorm × 2
  - Gate projection
  - Up projection
  - SiLU
  - Down projection
  - Residual add
- [ ] per-step barrier/sync time 측정 marker (`omp_get_wtime()` 기반)
- [ ] memory wait 측정 (packing/repacking count)
- [ ] H100x8 + dev (RTX3090) 동일 CSV schema 로 저장
- [ ] Intel VTune run 1회 (top-down metric: frontend bound / backend memory bound / backend core bound / retiring) — 결과는 `<run_dir>/vtune/<YYYYMMDD_HHMMSS>/` 에 저장 (여러 번 돌릴 경우 타임스탬프로 구분)
- [ ] Linux perf run 1회 (L2/L3 miss, dTLB miss, uncore BW) — 결과는 `<run_dir>/perf/<YYYYMMDD_HHMMSS>/`

---

## 성공 조건

산출물:
1. `batch_scaling_ratio` 가 `num_seqs = 1/2/4/8/16` 각각에 대해 측정됨
2. `per_req_cost` 그래프 — batch 증가에 따른 개선 여부 확인
3. Sublayer breakdown 으로 **top-2 bottleneck sublayer** 식별
4. `num_seqs` 증가 시 어느 sublayer 가 **선형적으로 폭증** 하는지 특정
5. OMP barrier/sync overhead 가 step 시간 중 차지하는 비율

이 5개가 나오기 전에는 §06 이후 어떤 kernel 수정도 시작 금지.

---

## 의존성

- **선행**: 없음 (G0)
- **후속**: 모든 기법 (§02~§22) 이 본 계측 결과에 근거해 우선순위 조정

---

## 리스크

- **계측 overhead 자체가 측정 왜곡**: hook/marker 비용을 빼고 기록하거나, enable/disable 비교로 bias 측정
- **VTune / perf 컨테이너 권한**: `perf_event_paranoid` + `CAP_SYS_ADMIN` 필요
- **sublayer 분해가 안 됨** (IPEX 가 layer 를 하나로 묶어서 hook point 없음): Intel VTune "Hotspots" 로 자동 분해 시도, 또는 IPEX optimize 우회 path 에서 측정

---

## 실행 방법

측정은 기존 `serve.sh` + `bench.sh` 를 그대로 쓰되, **env 파일에 측정 flag 만 추가**. sweep wrapper 스크립트 불필요 — 사용자가 `num_seqs` 값만 바꿔가며 평소대로 실행.

### 1. 측정 template env 복사

**Template 2종** (준비됨, PROFILE flag + 21개 기법 flag 모두 주석과 함께 명시):
- `eval/envs/g0_dev_rtx3090_qwen1.5b.env` — dev (RTX3090 + i9-12900KF)
- `eval/envs/g0_h100x8_qwen7b.env` — H100x8 + Xeon SPR

```bash
cp eval/envs/g0_h100x8_qwen7b.env /tmp/run.env
# 편집: 이번 측정에 맞게 2개만
#   HYBRID_TODO_NN=00         # 00=baseline / 05=§05 적용 후 / ...
#   HYBRID_CPU_MAX_SEQS=1     # sweep: 1, 2, 4, 8, 16 각각 재실행
# 기법 적용 후 해당 flag 를 on (예: TODO NN=06 → HYBRID_VNNI_HOT_PATH=1 까지 수정)
```

Template 안의 각 flag 옆 `§NN` 주석이 어느 TODO 에 대응하는지 보여줌.

### 2. serve + bench 실행 (두 터미널 수동)

```bash
# 터미널 1
./eval/serve.sh hybrid /tmp/run.env

# 터미널 2 (ready 후)
./eval/bench.sh hybrid /tmp/run.env
```

**자동 배치** — PROFILE=1 이므로 serve.sh 와 bench.sh 가 같은 디렉토리로:
- `measurement_results/<HW>/g0_<NN>/seqs<N>/hybrid_server_run.log` — `[HYBRID-APPLIED-FEATURES]` (boot) + `[HYBRID-CPU-PROFILE]` (per step)
- `measurement_results/<HW>/g0_<NN>/seqs<N>/applied_features.json` — 활성 flag snapshot
- `measurement_results/<HW>/g0_<NN>/seqs<N>/env_snapshot.txt` — env 변수 덤프
- `measurement_results/<HW>/g0_<NN>/seqs<N>/hybrid.json` — bench 수치
- `measurement_results/<HW>/g0_<NN>/seqs<N>/git_sha.txt`
- monitor csv, system_info 등 기존 파일

HW 이름 (`RTX3090`, `H100x8`) 은 `nvidia-smi` 에서 자동 감지.

### 3. 필요한 만큼 반복 (최소 2점, 여유 시 5점)

env 의 `HYBRID_CPU_MAX_SEQS` 만 바꿔 재실행. 2-5번 반복.

**최소 구성** (G0 통과 충분): `seqs=1, seqs=16` 2점
**권장**: `seqs=1, 4, 16` 3점 (knee 대략 보임)
**완전**: `seqs=1, 2, 4, 8, 16` 5점

### 4. 결과 디렉토리 이동 (사용자가 정리 명령 시)

```bash
# 사용자 지시 시에만
mv eval/results/<ts1>  measurement_results/H100x8/g0_00/seqs1
mv eval/results/<ts2>  measurement_results/H100x8/g0_00/seqs16
# ...
```

**디렉토리 명명 규칙**: `measurement_results/<HW>/g0_<NN>/seqs<N>/`

- `<NN>` = **해당 측정 시점에 최신 적용된 TODO 문서 번호** (`NinjaGap_Todo/` 파일 번호, 00-22)
- `00` = 아무 Ninja Gap 기법 적용 없는 **baseline** (§01 G0 측정 infra 만 활성)
- 각 기법 적용 후 해당 번호로 저장:

| 디렉토리 | 의미 |
|---|---|
| `g0_00/` | Baseline (pre-Ninja-Gap 상태) |
| `g0_03/` | §03 Huge Pages 1GB 적용 후 |
| `g0_04/` | §04 IPEX WoQ INT8 적용 후 |
| `g0_05/` | §05 KMP_BLOCKTIME 적용 후 |
| `g0_06/` | §06 Hot Path Wiring 적용 후 |
| `g0_08/` | §08 Kernel Fusion 적용 후 |
| `g0_13/` | §13 T-MAC LUT GEMV 적용 후 |

- **누적 적층**: 기법은 on/off 가 아니라 **지금까지 적용된 모든 것 + 이번 §NN**. 즉 `g0_08/` 는 §03 + §05 + §06 + §08 모두 켜진 상태 (README 의 Applied Features Log 테이블에 누적 기록)
- **동일 NN 재측정 시**: 디렉토리 덮어쓰지 말고 재측정 전에 기존 결과 백업하거나 `g0_08_v2/` 등 suffix. 기본 정책은 "최신 적용 시점 기준 재측정은 덮어쓰기, 과거 버전은 git history 로 추적".
- **Attribution**: 각 `g0_<NN>` vs `g0_<이전 NN>` diff 가 §NN 기법의 이득을 보여줌 (`applied_features.json` 의 flag 차이 + 성능 delta)

### 5. VTune 미시 분석 (원하는 run 1개 대상, 타임스탬프 서브디렉토리)

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=measurement_results/H100x8/g0_00/seqs16/vtune/${TS}
mkdir -p "$OUT"

# serve 가 이미 돌아가는 중, 다른 터미널에서
vtune -collect hotspots -result-dir "$OUT" \
    -target-pid $(pgrep -f "CPU_EngineCore_1") -- sleep 120
```

### 6. perf 미시 분석 (타임스탬프 서브디렉토리)

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=measurement_results/H100x8/g0_00/seqs16/perf/${TS}
mkdir -p "$OUT"

perf stat -e l2_rqsts.miss,dTLB-load-misses,uops_dispatched_port.port_5 \
    -p $(pgrep -f "CPU_EngineCore") \
    -o "$OUT/perf_stat.txt" sleep 120

perf record -F 99 -g -p $(pgrep -f "CPU_EngineCore") \
    -o "$OUT/perf.data" -- sleep 60
```

여러 번 돌려도 타임스탬프로 구분되어 overwrite 없음.

### 7. 집계 + 분석 (hw_inspect 스타일)

```bash
# 단일 run 분석 (기존 hw_inspect 확장)
./eval/g0_inspect.sh measurement_results/H100x8/g0_00/seqs1

# 여러 run 비교 (신규)
./eval/g0_compare.sh \
    measurement_results/H100x8/g0_00/seqs1 \
    measurement_results/H100x8/g0_00/seqs16
# → batch_scaling_ratio, sublayer breakdown diff 자동 출력

# 주간 비교 (Feature delta + 성능 delta + attribution)
./eval/g0_compare.sh \
    measurement_results/H100x8/g0_00 \
    measurement_results/H100x8/g0_06
```

---

## 실전 주의사항 (수동 실행 시 함정)

dev sweep 실측 중 확인된 issue 들. 자동 스크립트 아닌 **수동 실행에서도 동일하게 발생** 하므로 사전 체크 필수.

### 1. 이전 서버 좀비로 `curl /health` 가 OLD 서버 응답
- **증상**: 새 env 로 `serve.sh` 시작했는데 curl `/v1/models` 가 즉시 응답 → 실제로는 이전 run 의 서버
- **원인**: `pkill -f "vllm"` 이 **`VLLM::EngineCor` comm (프로세스 이름)** 을 못 잡음. `-f` 는 cmdline 매치인데 engine core 는 `prctl(PR_SET_NAME)` 로 이름만 바뀜
- **해결**:
  ```bash
  ps -ef | grep -E "vllm|VLLM|EngineCor|api_server" | grep -v grep \
      | awk '{print $2}' | xargs -r kill -9
  sleep 5
  # 그래도 남으면 cgroup reparent 의심. PID 로 개별 kill
  ```

### 2. Bench 가 너무 짧아 CPU 가 PROFILE_EVERY 주기 못 닿음
- **증상**: bench 완료, `profile_lines=0`
- **원인**: cpu-first + small bench → CPU 가 첫 req prefill 끝나기도 전에 GPU 가 전체 완료 → CPU forward decode step=0~4 에서 abort. `PROFILE_EVERY=5` 면 step 5 에 못 닿음
- **해결**:
  1. `VLLM_HYBRID_PROFILE_EVERY=1` (매 step 출력)
  2. bench 실행 **전에 synchronous curl 로 warmup** 보내기 — 최소 1 req CPU 완료 보장:
  ```bash
  curl -s --max-time 120 -X POST http://localhost:8765/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"Qwen/...","prompt":"Hello","max_tokens":8,"temperature":0}'
  # 이게 블로킹이므로 CPU decode step 전부 찍힘
  ```

### 3. `set -u` 미정의 env 변수로 serve.sh 즉시 죽음
- **증상**: serve.sh 기동 직후 `HYBRID_STATS_LOG_INTERVAL: unbound variable` 같은 에러
- **원인**: `serve.sh` 는 다수의 `HYBRID_*` / `VLLM_HYBRID_*` 변수를 require (no default)
- **해결**: **기존 env template 그대로 복사** 후 sweep 변수만 수정
  ```bash
  cp eval/envs/dev_rtx3090_qwen1.5b_hybrid_wave.env /tmp/my_g0.env
  # 그 다음 HYBRID_CPU_MAX_SEQS, VLLM_HYBRID_PROFILE 등만 편집
  ```
  minimal env 작성 말고.

### 4. 서버 ready 까지 30s+ 대기 필요 (작은 모델도)
- **증상**: serve 시작 후 너무 일찍 bench 돌리면 connection refused
- **원인**: 모델 weight load + CUDA graph capture 시간. 1.5B 도 CUDA graph 67개 shape × ~150ms = 10s+
- **해결**: ready poll 을 30~90s 여유롭게
  ```bash
  for i in {1..60}; do
      curl -s --max-time 2 http://localhost:8765/v1/models 2>/dev/null \
          | grep -q "Qwen" && break
      sleep 2
  done
  ```

### 5. bench.sh 와 serve.sh 이 같은 env 로 실행되는지 확인
- **이유**: env 불일치 (예: `HYBRID_CPU_MAX_SEQS` 를 serve 전에 바꾸고 bench 는 원본 env 로 실행) 시 dispatch 분석 틀어짐
- **해결**: 양쪽에 **동일 env 파일 path** 전달 고정

### 6. PROFILE 모드 자체 overhead
- **증상**: PROFILE=1 로 측정한 per-step ms 가 PROFILE=0 baseline 보다 약간 큼
- **원인**: sublayer hook 의 `time.perf_counter()` 호출 × 196 hooks per step
- **해결**: 절대값 비교엔 **PROFILE=0 reference run 1개 추가**. scaling ratio 비교에는 무관 (비율이 중심)

### 7. 결과 디렉토리 자동 이동 (bench.sh 기본 동작)
- `bench.sh` 는 `eval/results/<timestamp>_...` 로 결과 저장
- sweep 수동 실행 시 각 run 후 즉시 **의미 있는 이름으로 rename**:
  ```bash
  mv eval/results/20260415_*  measurement_results/RTX3090/g0_<NN>/seqs${SEQS}
  ```
  안 하면 다음 run 과 섞임.

### 8. warmup 및 스윕 순서
- **권장 순서**: seqs=1 먼저 → 2 → 4 → 8 → 16. baseline 부터.
- **이유**: seqs=16 먼저 돌리면 L2/L3 state 오염, 첫 run 이 warmup 효과 없는 깨끗한 baseline

---

## 디렉토리 구조

```
eval/basic/H100x8/
├── g0_seqs1_qwen7b/
│   ├── hybrid_server_run.log
│   ├── hybrid_server_boot.log
│   ├── hybrid_monitor_cpu.csv
│   ├── hybrid_monitor_gpu.csv
│   ├── hybrid_bench.log
│   ├── hybrid.json
│   ├── system_info.json
│   ├── inspect.txt
│   ├── vtune/
│   │   └── 20260416_091200/     # 타임스탬프별 VTune result
│   │       ├── r000hs/           # VTune 원본
│   │       └── vtune_report.txt
│   └── perf/
│       └── 20260416_093000/     # 타임스탬프별 perf result
│           ├── perf_stat.txt
│           └── perf.data
├── g0_seqs2_qwen7b/
├── g0_seqs4_qwen7b/
├── g0_seqs8_qwen7b/
├── g0_seqs16_qwen7b/
├── analysis_g0.ipynb
└── g0_report.md
```

---

## 관련 코드 위치

- `vllm/v1/worker/cpu_worker.py` — forward hook (세분화 대상)
- `eval/g0_measure.sh` — (신규) num_seqs sweep harness
- `eval/cpu_profile.sh`, `eval/cpu_profile_dev.sh` — 기존 sweep (참조)
- `eval/cpu_profile_summary.py` — (신규) 집계 스크립트
- `eval/monitor.py` — 1Hz CSV (이미 raw per-logical-CPU schema)
- `eval/basic/H100x8/analysis_g0.ipynb` — (신규) 분석 노트북
- `eval/basic/H100x8/analysis_h100.ipynb` — 참조 템플릿
