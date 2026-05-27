# SUB_172 — SUB_112 pinned bisect 500p reproducibility

- **목적**: SUB_112 (cores 80-111 pinned CPU fill) 의 200p 결과가 canonical 500p 위에서도 재현되는지 검증
- **측정 시각**: 2026-05-27 08:01-08:27 KST (wall **26분 46초**)
- **머신**: 프로덕션 (Sapphire Rapids + Xeon, H100 ×8). 사용자 100 core max 제약 안에서 cores 80-111 (NUMA1 절반, 32 phys core) 만 사용. HT (112-223) 금지
- **canonical**: Qwen 2.5 32B Instruct, TP=4×2 (vanilla GPU 0-3 :8001 / trident GPU 4-7 :8002 / AGSD router :8000), 500p × 256 max-tokens × 32 concurrency
- **N sweep**: 0 / 4 / 8 / 16 / 32, 각 N 별 balanced / sonnet-heavy / code-heavy 3 mix, 1-run
- **ENV (pthread EAGAIN 회피)**: `RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false`
- **CPU fill kernel**: `/tmp/sub112_cpu_fill_pinned.py --shape qwen32b --batch 128 --dtype bf16 --cpu-base 80` (worker `i` 가 core `80+i` 에 sched_setaffinity pin)

## 1. Per (N, mix) AGSD tps + per-mix Δ vs N=0

| N | balanced AGSD | sonnet AGSD | code AGSD | 3-mix avg | Δ vs N=0 |
|---:|---:|---:|---:|---:|---:|
| 0  | 5,390.1 | 6,067.2 | 7,051.3 | **6,169.5** | +0.00% |
| 4  | 5,510.5 (+2.23%) | 6,214.4 (+2.43%) | 6,847.8 (−2.89%) | 6,190.9 | **+0.35%** |
| 8  | 5,346.8 (−0.80%) | 4,867.7 (−19.77%) | 6,628.2 (−6.00%) | 5,614.2 | **−9.00%** |
| 16 | 5,150.4 (−4.45%) | 6,061.6 (−0.09%) | 6,362.8 (−9.76%) | 5,858.3 | **−5.05%** |
| 32 | 4,309.4 (−20.05%) | 5,473.8 (−9.78%) | 6,084.7 (−13.71%) | 5,289.3 | **−14.27%** |

**N=0 baseline 6,169.5 tps 는 SUB_160 500p OFF baseline (3-mix avg ≈ 6,169) 와 ±0.01% 일치 — 측정 환경 재현 검증 PASS.**

## 2. 200p (SUB_112 원본) ↔ 500p (SUB_172) 3-mix avg AGSD Δ 비교

| N | 200p Δ (SUB_112) | 500p Δ (SUB_172) | 차이 |
|---:|---:|---:|---:|
| 0  | +0.00% (5,456 tps) | +0.00% (6,170 tps) | baseline |
| 4  | **+3.22%** | **+0.35%** | −2.87 pp |
| 8  | **+3.36%** | **−9.00%** | −12.36 pp |
| 16 | **−1.54%** ⚠ | **−5.05%** ⚠ | −3.51 pp |
| 32 | **+3.54%** ⭐ | **−14.27%** ⚠⚠ | −17.81 pp |

- **N=32 winner reversal**: 200p +3.54% → 500p −14.27% (Δ −17.81 pp). 200p 의 lift 는 500p 에서 **재현되지 않음**
- **N=16 valley**: 200p 에서도 약한 valley (−1.54%) 였는데 500p 에서도 valley (−5.05%) 유지. 다만 깊이는 N=32 (−14.27%) 가 더 큼 — valley 위치가 **N=16 → N=8/32** 로 이동
- **N=4 만 양수**: +0.35% — 200p (+3.22%) 보다 크게 약화. lever 의 안전 영역은 N=4 일 가능성

## 3. N curve 비단조 확인 (500p)

```
N=0  → +0.00% ────────
N=4  → +0.35%  ▲
N=8  → −9.00%  ▼▼  ← valley 1
N=16 → −5.05%  ▲ (회복)
N=32 → −14.27% ▼▼▼ ← valley 2 (더 깊음)
```

- 500p N curve 는 **비단조 (non-monotonic) U 형이 아닌 2-valley** 형상: N=8 와 N=32 둘 다 큰 회귀, N=16 은 중간 회복
- 200p (N=16 단일 valley) 와 다른 패턴 — **prompt 길이가 valley 위치와 깊이를 바꿈**
- N=8 sonnet-heavy 만 단독으로 −19.77% — 1-run 변동성 / sonnet-heavy 의 thermal/cache jitter 가 섞여 있을 가능성. multi-run 재측정으로 분리 필요

## 4. CPU/GPU utilization sustained (per-N mean)

| N | CPU% | GPU 0-3 vanilla | GPU 4-7 trident |
|---:|---:|---:|---:|
| 0  |  5.0% | 67.3% | 32.7% |
| 4  |  6.9% | 68.5% | 28.9% |
| 8  |  8.5% | 70.1% | 28.5% |
| 16 | 12.0% | 69.7% | 25.1% |
| 32 | 19.1% | 67.7% | 26.9% |

- CPU 5% → 19.1% (+14 pp) sustained. fill 이 의도대로 cores 80-111 점유
- GPU vanilla(0-3) ~68% 일정 — CPU contention 의 영향 없음 (NUMA 분리)
- GPU trident(4-7) ~25-33% — N 증가 시 미세 감소 (~7 pp). trident 가 더 민감하나 결정적 회귀 신호는 아님
- 즉 **AGSD 회귀의 원인은 CPU/GPU util 자체가 아니라 다른 경로** (router classifier latency 증가 / trident speculator interaction / memory bandwidth contention 등 후속 분석 필요)

## 5. 한계 + 후속

- **1-run 측정**: 사용자 1-run rule 준수. N=8 sonnet-heavy −19.77% / N=32 balanced −20.05% 의 큰 음수는 multi-run 검증 필요
- **CPU fill = qwen32b shape BF16 batch 128 만**: 다른 shape (Llama 70B / batch 64 / FP16) 로의 일반화 미검증
- **router classifier latency 미계측**: AGSD 회귀 원인의 분해 (router vs vllm vs spec-decode) 필요 → 후속 SUB 에서 per-request breakdown profile
- **500p 의 결정적 차이**: 500p 는 200p 대비 2.5× wall, CPU fill 의 cumulative thermal/cache pressure 가 더 길게 가해짐. fill duration / thermal headroom 별도 측정 필요
- **lever 안전 영역**: 500p 에서 net-win 을 보이는 영역 없음. N=4 +0.35% 는 noise 수준 — **SUB_112 lever 는 500p 캐논 위에서 paper 인용 불가**

## 산출물

- `aggregated.json` — per (N, mix, scenario) tps + AGSD Δ 표
- `util_summary.json` — per-N CPU/GPU mean util
- `_monitor_cpu.csv` / `_monitor_gpu.csv` — 0.5s interval raw
- `workers_{0,4,8,16,32}/benchmark_{balanced,sonnet-heavy,code-heavy}.json` — bench raw
- `workers_{4,8,16,32}/cpu_workers/` — pinned worker logs
- `logs/main.log` — launcher timeline
- `logs/{vanilla,trident,router}.log` — server logs

## Verdict

- **SUB_112 lever 의 500p 재현 실패**: 200p winner cell (N=32 +3.54%) → 500p (−14.27%) 로 완전히 역전
- **paper §4 의 500p baseline 위에서 lift 정량**: 5 N 값 중 net-win 없음, N=4 가 noise 한도 (+0.35%) — paper 본문에 sweet spot 으로 인용 불가
- N curve 의 비단조성은 500p 에서도 확인되나 valley 위치가 (200p N=16) → (500p N=8, N=32 2-valley) 로 이동 — **prompt 길이가 worker contention dynamics 를 질적으로 바꿈**
- 후속 SUB 후보: (a) multi-run 분산 검증, (b) router classifier latency per-N 분해, (c) 500p × N=2/6/12 미세 sweep 으로 sweet spot 재탐색, (d) fill shape variation (다른 dtype/batch) 으로 일반화 여부
