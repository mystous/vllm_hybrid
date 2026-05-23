# SUB_040 — CPU/GPU util metric 통합 baseline (★ 본 프로젝트 목표 직접 검증)

> **parent**: TSK_019 / CLAUDE.md `# Objective` 정합 framing 도입
> **출처**: 사용자 turn — "프로젝트 목표 재확인" 후 (c) Path 선택
> **measurement**: HEAD `0d7dc0334`, 500p × 8192, gmu=0.85, CPU(/proc/stat) + GPU(nvidia-smi) 1Hz sampling

---

## 1. 측정 결과 (★ 본 프로젝트 목표 metric)

| metric | NEO 500p | vanilla 500p | Δ |
|---|---:|---:|---:|
| output_tps | 1685.7 | 4680.9 | vanilla 2.78× faster |
| wall (s) | 2421.3 | 875.0 | NEO +1546s |
| crash | 0 ✓ | 0 ✓ | — |
| **CPU busy avg %** | **11.93** | **4.67** | NEO **+7.26%p** ↑ |
| CPU busy max % | 40.96 | 18.66 | NEO burst 더 큼 |
| CPU busy p50 % | 5.79 | 5.36 | median 거의 동일 (NEO 의 11.93% 는 burst 가 평균을 끌어올림) |
| CPU idle avg % | 88.07 | 95.33 | NEO -7.26%p (mirror) |
| **GPU util avg %** | **66.0** | **73.4** | NEO **-7.4%p** ⚠️ idle 발생 |
| GPU util max % (per-GPU) | 100% (모든 GPU) | 100% (모든 GPU) | peak 동일 |
| GPU mem avg (GPU0) | 71,181 MiB | 54,435 MiB | NEO +16.7 GiB (KV mirror 영역) |
| **GPU power avg** | **327-364 W** | **477-509 W** | **vanilla 가 35-40% 더 강하게 활용** ⚠️ |

## 2. ★ 본 프로젝트 목표 미달 확정

### 2.1 CLAUDE.md `# Objective` 원문 vs 측정

| 목표 | 측정 결과 | 평가 |
|---|---|---|
| "CPU의 활용률을 **극도로** 끌어 올려" | **NEO 11.93% avg** | ❌ 미달 — 극대화 (50-90%+) 와 큰 거리 |
| "CPU의 활용률이 **Idle 또는 낮은 Utilization을 허락하지 않는다**" | **NEO CPU idle 88.07%** | ❌ 미달 — 80%+ idle 발생 |
| "GPU가 포함된 서버 또는 Cluster **전체의 성능을 향상**" | NEO inf_tps 1685 vs vanilla 4681 = **NEO 의 36%** | ❌ 미달 — 단일 job 성능 저하 |

→ **본 NEO 구현은 본 프로젝트 목표의 3 영역 모두 미달**.

### 2.2 부수적 발견 — NEO 가 GPU 도 idle 시킴

| 영역 | vanilla | NEO | Δ |
|---|---:|---:|---:|
| GPU util avg | 73.4% | 66.0% | -7.4%p (NEO idle ↑) |
| GPU power avg | 477-509 W | 327-364 W | **-31% (NEO 가 power-wise GPU 비활용)** |

→ CLAUDE.md "CPU idle 허락 안 함" 목표의 mirror 문제 — NEO 가 CPU 활용 시도하면서 **GPU 까지 idle 시킴**. **net trade-off 가 음수** (GPU 손실 > CPU 이득).

### 2.3 CPU 활용 효율 비교

| metric | NEO | vanilla |
|---|---:|---:|
| output_tps / CPU% | 141 | **1002** |
| output_tps / GPU% | 25.5 | **63.8** |
| output_tps / GPU power (W) | **4.93** | 9.42 |

→ **vanilla 가 모든 효율 지표에서 NEO 압도**. 본 NEO 구현은 **단일 inference job 관점에서 명백한 net-negative**.

## 3. CPU 활용 11.93% 의 분해 추정

| CPU 영역 | 추정 비중 | 근거 |
|---|---:|---|
| NEO worker process Python overhead | ~3-5% | run_neo_baseline 의 prepare_inputs / scheduler / Python ops |
| pacpu OMP team (ISPC compute, libgomp spin) | ~5-7% | SUB_035 C1a step1 ISPC 0.31 ms × 80 layer × 8 worker × 10 thread |
| swap_in / swap_out ATen ops | ~1-2% | SUB_025/026 async path |
| 기타 (Ray, ZMQ, monitoring) | ~1% | — |

→ NEO 의 11.93% 중 ~5-7% 가 pacpu (CPU attention) — **실질적 CPU compute 활용은 5-7%만**.

## 4. 다음 path (SUB_041 + 목표 재정의)

### 4.1 SUB_041 진입 — Multi-workload 서버 throughput 검증

SUB_040 결과가 NEO 의 단일 job 가치를 부정. 단 본 프로젝트 목표 = **서버 전체 throughput**.

→ **SUB_041 가 결정적**:
- vanilla solo (CPU 5% idle) + CPU BG = BG 가 CPU 95% 자유 사용 → 합산 throughput **높음** 가설
- NEO solo (CPU 12% idle) + CPU BG = NEO 와 BG 가 CPU contention → 합산 throughput **낮음** 가설
- → vanilla + BG 가 NEO + BG 보다 합산 throughput 높으면 **NEO 의 raison d'être 자체가 깨짐**

### 4.2 목표 재정의 후보 (SUB_041 결과 본 후 결정)

| 후보 | 의미 |
|---|---|
| (a) NEO 의 CPU 활용을 극대화하는 lever 추가 (현 12% → 50%+) | 본 목표 직접 추구. 단 NEO의 step1 ISPC compute 가 이미 CPU 의 진정한 bottleneck (SUB_035) → 더 짜낼 CPU 작업이 없을 수 있음 |
| (b) NEO 의 적용 영역을 vanilla OOM 영역으로 좁히기 | 본 코드 베이스 의 NEO 는 vanilla 가 OOM 인 영역에서만 가치 |
| (c) NEO 의 CPU offload + GPU full util 동시 달성 lever | NEO 가 GPU 도 idle 시키는 문제 (66 vs 73.4%) 해소 — 가장 어려움 |
| (d) TSK_019 기각 + 새 IDE 시작 — 본 코드 베이스 의 NEO 가 목표 미달 | 본 NEO 구현 자체가 target architecture 부적합 결론 |

## 5. raw 자료

| 항목 | 위치 |
|---|---|
| SUMMARY.tsv | `eval/results/20260522_104624_sub040_util_baseline/SUMMARY.tsv` |
| t1 NEO 500p util_summary | `eval/results/20260522_104624_sub040_util_baseline/t1_neo_500p_util/util_summary.txt` |
| t1 cpu_util.csv (2,636 samples) | `eval/results/20260522_104624_sub040_util_baseline/t1_neo_500p_util/util/cpu_util.csv` |
| t1 gpu_util.csv (2,608 × 8 GPU) | `eval/results/20260522_104624_sub040_util_baseline/t1_neo_500p_util/util/gpu_util.csv` |
| t2 vanilla 500p util_summary | `eval/results/20260522_104624_sub040_util_baseline/t2_vanilla_500p_util/util_summary.txt` |
| launcher | `/tmp/run_sub040_util_baseline.sh` |
| util sampler | `/tmp/util_sampler.sh` |
| stdout log | `/tmp/sub040_util.log` |
