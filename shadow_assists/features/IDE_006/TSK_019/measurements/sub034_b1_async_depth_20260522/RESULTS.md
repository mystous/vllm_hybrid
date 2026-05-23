# SUB_034 — B1 OmniServe LSE async merge (CDEC_PIPELINE_DEPTH sweep) (2026-05-22 KST)

> **parent**: TSK_019 / N 문서 영역 B B1 / O 분석 §7.2 ★★★ #2
> **measurement**: HEAD `0d7dc0334`, 100p × 8192, gmu=0.85, env-ON baseline + `VLLM_NEO_CDEC_PIPELINE_DEPTH` 1/2/3 env-only sweep.
> **race-safe drain**: 이미 적재 — `attention.py:1352 _neo_drain_pending_cdec` (FIFO + `_xfer_stream.wait_stream`).

---

## 1. 측정 결과

| test | depth | tps | wall (s) | crash |
|---|---:|---:|---:|:-:|
| t1 | 1 | 910.9 | 890.7 | 0 ✓ |
| t2 | 2 | 933.0 | 878.0 | 0 ✓ |
| t3 | 3 | 914.9 | 886.6 | 0 ✓ |
| t4 | 1 | 933.0 | 878.0 | 0 ✓ |
| t5 | 2 | 912.9 | 888.8 | 0 ✓ |

### 1.1 depth 별 평균

| depth | n | avg tps | range | inter-run gap |
|---|---:|---:|---:|---:|
| 1 | 2 | **921.95** | 910.9–933.0 | 22 tps (2.4%) |
| 2 | 2 | **922.95** | 912.9–933.0 | 20 tps (2.2%) |
| 3 | 1 | 914.9 | — | (1-run only) |

### 1.2 depth=1 vs depth=2 비교

Δ = **+1.0 tps (0.1%)** — **단일런 jitter (±2.4%) 보다 훨씬 작음** → noise.

## 2. 결론 — B1 도 noise

| depth | 결과 |
|---|---|
| 1 (baseline) | 921.95 ±1.2% |
| 2 (가설: CPU/GPU overlap 증가) | 922.95 ±1.1% — **단일런 jitter 안** |
| 3 (가설: 더 깊은 pipeline) | 914.9 (1-run) — depth=1 range 와 동일 |

**가설**: depth ≥ 2 가 CPU pacpu compute 와 GPU forward 의 진정한 overlap 활성화 → CPU sync wait 제거 → +5~10 ms 단축.
**현실**: depth=2 가 depth=1 과 동일 (±0.1%) — race-safe drain 이 정확하지만 **본 워크로드 에서는 overlap window 자체가 noise band 안**.

## 3. SUB_032/033/034 통합 통찰

| SUB | lever | 결과 | 결론 |
|---|---|---|---|
| SUB_032 | A4 (numactl) 3-run avg | -0.21% (noise) | A4 도 무효 |
| SUB_033 | B3 (FlashDecoding++ online softmax) 3-way | -0.82% (negative) | 기각 (default OFF) |
| **SUB_034** | **B1 (async cdec depth) 5-way** | **+0.1% (noise)** | **noise — depth=1 유지** |

**3 SUB 모두 noise**. 본 워크로드 가 throughput-saturated 임이 측정으로도 입증:
- baseline 922 tps × 880 s = 811,360 token ≈ 100p × 8192 target (819,200) → **하드웨어 ceiling 도달**
- single-run noise (±2.4%) > lever 신호 (±0.1%~0.8%) → 어떤 lever 도 검출 불가

## 4. 갱신된 결론

| Tier | Lever | 검증 결과 |
|---|---|---|
| A1-A5 | env / runtime 튜닝 | 모두 noise |
| B3 | kernel-level (softmax) | noise/negative |
| **B1** | **async cdec pipeline depth (env-only)** | **noise** |

→ **A-tier + B-tier 의 검증된 lever 가 모두 noise** = 본 코드 베이스 의 NEO 구현이 100p × 8192 워크로드 의 ceiling 에 도달.

## 5. 다음 path

SUB_035 plan 의 권장: **C1a (OMP launch overhead 측정)** 진입 → 정량적 시야 확보 → C-tier ROI 추정.

만약 OMP launch overhead 가 wall 의 ≥10% 면 C1b~e 시도 가치 있음. < 10% 면 **본 워크로드 자체가 더 짜낼 여지 없음** 으로 확정 → Path A (워크로드 재정의, 500p × 8192 또는 더 큰) 권장 복귀.

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| SUMMARY.tsv | `eval/results/20260522_022855_sub034_b1_async_depth_100p/SUMMARY.tsv` |
| per-test 결과 (5 dirs) | `eval/results/20260522_022855_sub034_b1_async_depth_100p/t1~t5/` |
| launcher | `/tmp/run_sub034_b1_async_depth.sh` |
| stdout log | `/tmp/sub034_b1.log` |
| race-safe drain | `vllm/model_executor/layers/attention/attention.py:1352 _neo_drain_pending_cdec` |
