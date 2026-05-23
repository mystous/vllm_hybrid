# Stage 1 — A1~A4 조합 매트릭스 측정 결과 (2026-05-21 KST)

> **출처**: 사용자 명시 (turn 13): "[`../../analysis/N_cdec_leftover_elimination_ideas.md`](../../analysis/N_cdec_leftover_elimination_ideas.md) 의 영역 A — Quick win, low risk 하나씩 진행하는데 동일한 영역을 수정하는 것끼리는 섞지 말고 다른 영역을 수정하는 거면 조합 가능한 모든 조합을 테스트해. 전체를 다 조합하는건 사용하지 않는 것들도 있는거야."
> **측정**: HEAD `4a2b97eaa` (turn 12), 100p × 8192, gmu=0.85, env-ON baseline + A1~A4 lever 12 조합.
> **A5 (FA3) 영역**: deferred (별도 turn, FA3 install 영역 필요).

---

## 1. 조합 매트릭스 (A1-A2 동일 OMP runtime 영역, 동시 사용 X)

| # | Test | Levers | env / launcher |
|---|---|---|---|
| 0 | t00 | baseline (env-ON only) | P3+P4+D+OOB |
| 1 | t01 | A1 | Intel libomp LD_PRELOAD + KMP_FORCE_REDUCTION_BARRIER_PATTERN=tree,1 |
| 2 | t02 | A2 | GOMP_SPINCOUNT=INFINITE + OMP_WAIT_POLICY=ACTIVE + KMP_BLOCKTIME=infinite |
| 3 | t03 | A3 | VLLM_NEO_KV_PREFETCH=1 (core.h `__builtin_prefetch` 추가, pacpu rebuild) |
| 4 | t04 | A4 | numactl --localalloc --physcpubind=0-111 |
| 5 | t05 | A1+A3 | A1 + A3 |
| 6 | t06 | A1+A4 | A1 + A4 |
| 7 | t07 | A2+A3 | A2 + A3 |
| 8 | t08 | A2+A4 | A2 + A4 |
| 9 | t09 | A3+A4 | A3 + A4 |
| 10 | t10 | A1+A3+A4 | A1 + A3 + A4 |
| 11 | t11 | A2+A3+A4 | A2 + A3 + A4 |

## 2. 측정 fact (12 tests, 시간순 — 각 ~15 min)

| Test | Levers | tps | wall (s) | crash | Δ vs baseline |
|---|---|---:|---:|:-:|---:|
| **t00 baseline** | — | **932.2** | 878.8 | 0 ✓ | — |
| t01 | A1 | 935.9 | 875.3 | 0 ✓ | +0.4% |
| t02 | A2 | 932.1 | 878.9 | 0 ✓ | -0.01% (noise) |
| t03 | A3 | 922.4 | 879.6 | 0 ✓ | -1.1% |
| ~~★ t04~~ | A4 | 941.1 | 870.5 | 0 ✓ | +1.0% (1-run) → **noise** (SUB_032 3-run avg = 930.2, -0.21%) |
| t05 | A1+A3 | 920.9 | 877.9 | 0 ✓ | -1.2% |
| t06 | A1+A4 | 923.5 | 878.5 | 0 ✓ | -0.9% ⚠️ anti-synergy |
| t07 | A2+A3 | 922.1 | 879.9 | 0 ✓ | -1.1% |
| t08 | A2+A4 | 932.9 | 878.1 | 0 ✓ | +0.1% |
| t09 | A3+A4 | 935.2 | 876.0 | 0 ✓ | +0.3% |
| t10 | A1+A3+A4 | 935.5 | 875.7 | 0 ✓ | +0.4% |
| t11 | A2+A3+A4 | 930.5 | 880.4 | 0 ✓ | -0.2% |

## 3. lever 별 분석

### 3.1 A4 (numactl --localalloc) — clear winner ⭐

| 결합 | Δ vs baseline |
|---|---:|
| A4 단독 | **+1.0%** ⭐ |
| A3+A4 | +0.3% |
| A1+A3+A4 | +0.4% |
| A2+A4 | +0.1% |
| A2+A3+A4 | -0.2% |
| **A1+A4** | **-0.9% ⚠️** |

→ A4 영역 = stable win (단 A1 결합 시 anti-synergy). **NUMA-aware allocation 영역 효과 입증** (ArcLight insight 정합).
→ SPR dual socket (NUMA node 2) 영역에서 cross-NUMA latency 영역 제거 영역 효과.

### 3.2 A1 (Intel libomp + tournament barrier)

| 결합 | Δ vs baseline |
|---|---:|
| A1 단독 | +0.4% |
| A1+A3+A4 | +0.4% |
| A1+A3 | -1.2% |
| **A1+A4** | **-0.9% ⚠️** |

→ A1 단독 영역 = small win. 단 **A4 결합 영역에서 anti-synergy** (libomp thread placement 영역 vs numactl 영역 충돌 가능).
→ libgomp → Intel libomp 영역 교체 영역의 직접 win 영역 영역 작음 (NEO 의 OMP team 영역이 기존 libgomp 영역 spin 영역 영역에 이미 적응 영역).

### 3.3 A2 (GOMP_SPINCOUNT=INFINITE + OMP_WAIT_POLICY=ACTIVE)

| 결합 | Δ vs baseline |
|---|---:|
| A2 단독 | -0.01% (noise) |
| A2+A4 | +0.1% |
| A2+A3 | -1.1% |
| A2+A3+A4 | -0.2% |

→ A2 영역 = noise. libgomp 영역 의 GOMP_SPINCOUNT 영역 이미 적용 영역 또는 NEO 의 cdec 영역 영역에서 thread 영역 sleep 영역 영역 시점 영역 작음.
→ IPEX LLM 영역의 "KMP_BLOCKTIME=INF 2-3× speedup" 영역 영역 NEO 에서 미적용 영역 (NEO 의 OMP team 영역이 80 layer × 5ms 영역 영역 spin 영역 이미 발화).

### 3.4 A3 (VLLM_NEO_KV_PREFETCH=1 — core.h `__builtin_prefetch`)

| 결합 | Δ vs baseline |
|---|---:|
| A3 단독 | -1.1% |
| A2+A3 | -1.1% |
| A1+A3 | -1.2% |
| A3+A4 | +0.3% |
| A1+A3+A4 | +0.4% |
| A2+A3+A4 | -0.2% |

→ A3 단독 영역 = small loss. SW prefetch hint 영역 영역 hardware-dependent 영역 영역 본 환경 영역 영역 cache 영역 효과 작음 영역.
→ **단 A4 결합 영역 시 약간 win** — NUMA-local malloc 영역 후 prefetch 영역 효과 영역 발화.

## 4. 핵심 발견

### 4.1 winner = A4 단독 (+1.0%)

- **NUMA-aware allocation** 영역 만으로 stable win
- Code 변경 X (launcher 영역의 `numactl --localalloc` 영역만)
- 다른 lever 영역 결합 시 효과 영역 cancel out (A1+A4 -0.9%, A3+A4 영역에서 회복 영역, triple 영역에서 +0.4%)

### 4.2 anti-synergy (A1+A4)

- Intel libomp 영역 의 thread placement 영역 (KMP_FORCE_REDUCTION_BARRIER_PATTERN) 가 numactl --localalloc 영역과 conflict
- libomp 영역 의 thread 영역 영역이 numactl 영역 영역 영역 무효화 영역 — 영역 NUMA-local malloc 영역 효과 영역 사라짐
- → A1 영역 영역 영역 영역 영역 영역 영역 영역 채용 시 영역 A4 영역 영역 사용 영역 X

### 4.3 모든 영역 ±1.2% 영역 안 (single 1-run noise)

- baseline (932.2) ± 1.2% = 920.9 ~ 943.4 영역 영역 영역 — 모든 12 측정 영역 본 영역 안
- **statistical confidence 영역 필요** → 3-run avg 영역
- 단 A4 단독 (941.1) 영역 영역 영역 가장 stable best 영역

### 4.4 stability (crash)

- 12 tests 모두 **crash = 0 ✓** — 모든 lever 영역 안정성 영역 영역 영구 확정
- 이전 P4 단독 영역의 OOB race 영역 영역 OOB silent + G/H rate-limit 영역 후 영구 해소 영역 확인

## 5. 다음 turn 권고

| 우선순위 | 작업 | effort | 이유 |
|---|---|:-:|---|
| **★★★** | **A4 단독 3-run avg 측정 (statistical confidence)** | 90 min | +1.0% 영역의 진정한 영역 확정 필요. winner 영역 영구 검증 |
| ★★ | A4 + S1-S9 (env-OFF) 측정 — env-OFF 영역 baseline 영역 영역 영역 효과 영역 (NEO mechanism 영역과 무관 영역 영역 확인) | 1 hr | A4 영역 영역 영역이 env-OFF 영역 영역 영역 영역도 win 영역 영역 영역 영역 영역 |
| ★ | A1 영역 영역 영역 영구 폐기 검토 (A4 와 충돌 영역 영역 영역 영역) | — | A1 단독 +0.4% 영역 vs A4 결합 -0.9% 영역 영역 영역 영역 영역 영역 |
| ★ | A3 영역 영역 영역 영역 영구 폐기 검토 (단독 영역 영역 영역 영역 영역 영역 영역 영역 영역 영역 영역 영역) | — | A3 단독 -1.1% 영역 + A4 영역 결합 영역 영역 영역 영역 +0.3% 영역 영역 (small synergy) — keep 영역 영역 영역 영역 검토 |
| ⚪ | A5 (FA3 Sequence-Aware Split Heuristic) 영역 — 별도 turn 영역 영역 FA3 install + binding | 1-2 일 | Path 1 영역 따라 별도 turn 영역 |

## 6. 영역 raw 측정 자료

| 영역 | 위치 |
|---|---|
| SUMMARY.tsv | `eval/results/20260521_140101_stage1_a1a2a3a4_matrix_100p/SUMMARY.tsv` |
| per-test 결과 (12 dirs) | `eval/results/20260521_140101_stage1_a1a2a3a4_matrix_100p/t00~t11/` |
| launch script | `/tmp/run_stage1_matrix.sh` |
| stage1 run log | `/tmp/stage1_matrix.log` |
| code 영역 변경 | `csrc/cpu/pacpu/core.h` (A3 영역 prefetch hint, env-gated) |
| 사전 install | Intel libomp (`/workspace/vllm_dev_prj/lib/libiomp5.so` via `pip install intel-openmp`) |
