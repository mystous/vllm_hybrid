# SUB_036 — Path A 500p × 8192 baseline + NEO net benefit 비교 (2026-05-22 KST)

> **parent**: TSK_019 / O 분석 §7.2 ★★★ Path A 권고
> **measurement**: HEAD `0d7dc0334`, 500p × 8192, gmu=0.85
>   - t1: NEO env-ON (P3+P4+D+OOB, OMP=10, NUMA bind ON)
>   - t2: vanilla GPU-only (모든 NEO env unset)
> **결정적 발견**: vanilla 가 NEO 보다 **2.63× 빠름**. NEO 의 net benefit 가 **negative** in this workload.

---

## 1. 측정 결과

| test | levers | tps | wall (s) | crash | token target | 처리 token |
|---|---|---:|---:|:-:|---:|---:|
| t1 | NEO env-ON | **1778.8** | 2293.2 | 0 ✓ | 4,096,000 | 4,078,000 |
| t2 | **vanilla** | **4680.8** | **875.1** | 0 ✓ | 4,096,000 | 4,095,000 |

**Δ (vanilla vs NEO)**: +163% (vanilla 가 2.63× 빠름), wall -1418s (-62%).

## 2. NEO 100p vs 500p 비교 — batching gain

| 워크로드 | NEO tps | wall 시간 | tps/p ratio |
|---|---:|---:|---:|
| 100p × 8192 (SUB_032 avg) | 930.2 | 875s | 9.30 |
| **500p × 8192 (SUB_036)** | **1778.8** | 2293s | 3.56 |
| Δ | +91% | +162% | -62% |

→ 워크로드 5× 증가 시 throughput 91% 증가 — **batching gain** (큰 batch 가 GPU 활용 향상).
단 5× 증가 시 wall 2.62× 만 증가 — sublinear (좋음).

## 3. vanilla 비교 — net benefit 의 진실

| 워크로드 | NEO tps | vanilla tps | ratio (NEO/vanilla) | NEO net |
|---|---:|---:|---:|---|
| 100p × 8192 | 930 (SUB_032) | 측정 안 함 | ? | ? |
| **500p × 8192** | **1779** | **4681** | **0.38** | **-62% (NEO 가 vanilla 의 38%)** |

**vanilla 4680 tps × 875s ≈ 4.1M token = target 정확히 도달** — vanilla 가 throughput-bound, GPU 가 fully utilized.
**NEO 가 vanilla 의 38% throughput 만 산출** — CPU offload overhead 가 vanilla 의 단순 GPU 만 사용보다 훨씬 비싸다.

## 4. NEO 의 raison d'être 검증

NEO 의 목적: **vanilla 가 OOM 인 영역** (KV cache 가 GPU 메모리 초과 시) 에서 CPU KV offload 로 작동.

| 측정 | vanilla | NEO |
|---|---|---|
| 100p × 8192, gmu=0.85 | (측정 안 함, 작동 추정) | 930 tps |
| **500p × 8192, gmu=0.85** | **4681 tps (정상 작동)** | 1779 tps |
| vanilla OOM 영역 (1000p+? max-tokens 16384?) | (측정 필요) | NEO 만 작동 가능 |

→ **본 워크로드는 NEO 의 raison d'être 가 충족 안 되는 영역**. vanilla 가 정상 작동하므로 NEO 의 overhead (CPU offload + sync) 가 손해.

## 5. 갱신된 결론

### 5.1 본 워크로드 (500p × 8192) 에서의 NEO

- NEO 가 vanilla 의 38% throughput → **본 워크로드에서 NEO 사용은 합리적이지 않음**
- 모든 lever (A1-A5, B1, B3) 가 noise — saturated 워크로드의 ceiling 안에서 노이즈만 측정
- Path A 권고의 "더 큰 워크로드에서 lever 신호 검출 가능" 가설 = **부분적 확인** (NEO 100p 930 → 500p 1779 = throughput 향상), 단 vanilla 는 여전히 훨씬 빠름

### 5.2 NEO 의 진정한 가치 영역 = vanilla OOM 영역

- vanilla 가 500p × 8192 까지 정상 → 그 위 (1000p+, max-tokens 16384, max_num_seqs ↑) 에서 OOM 여부 확인 필요
- vanilla OOM 영역에서만 NEO 의 net benefit measurable
- 그 영역에서 lever (B4, av_amx) 가 의미 있는 신호 산출 가능성

## 6. ★★★ 갱신된 path 권고

| 우선순위 | 작업 | 이유 |
|---|---|---|
| **★★★** | **vanilla OOM threshold 점검** — 1000p / 2000p / max-tokens=16384 / max_num_seqs=512 등 sweep | NEO 의 raison d'être 영역 확정 |
| **★★** | vanilla OOM 영역에서 NEO baseline 측정 (3-run avg) | NEO net benefit 영역의 진정한 baseline |
| **★★** | 그 영역에서 SUB_037 (B4 SPARAMX) / SUB_039 (av_amx) 측정 | 진짜 lever 효과 검출 가능성 |
| ⚪ | 본 워크로드 (500p × 8192) 에서 SUB_037/039 진입 | vanilla 가 항상 win — 의미 없음 |
| ⚪ | TSK_019 종료 결정 | A/B/C-tier 모두 noise, vanilla net win → NEO 자체 가치 재정의 |

## 7. raw 자료

| 항목 | 위치 |
|---|---|
| SUMMARY.tsv | `eval/results/20260522_082108_sub036_pathA_500p_baseline/SUMMARY.tsv` |
| t1 NEO 500p result | `eval/results/20260522_082108_sub036_pathA_500p_baseline/t1_neo_500p/result.json` |
| t2 vanilla 500p result | `eval/results/20260522_082108_sub036_pathA_500p_baseline/t2_vanilla_500p/result.json` |
| launcher | `/tmp/run_sub036_pathA_500p_baseline.sh` |
| stdout log | `/tmp/sub036_pathA.log` |
