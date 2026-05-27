# SUB_192 — RESULTS

side-channel **partial KV merge** worker (BF16 mul-add over 256-token × 4096-hidden block,
50 ms cycle, 16 OMP × cores 80-95) — canonical 500p × 3 mix × OFF/ON × 1-run

- 시작 (OFF launcher): 2026-05-27 14:19:27 KST
- 종료 (ON cleanup): 2026-05-27 14:32:55 KST

## 1. 결과 요약 (AGSD 3-mix avg)

| | OFF | ON | Δ |
|---|---:|---:|---:|
| 3-mix avg tps | 4 307.9 | 4 302.4 | **−0.13%** (noise floor) |

verdict: **net positive 아님**. signal magnitude (0.13 pp) 가 1-run variance (≈ 0.7-1.0 pp)
보다 작음 → noise.

## 2. 9-cell table

| mix | scenario | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1 623.4 | 1 611.0 | −0.76% |
| balanced | trident-only | 1 624.0 | 1 616.9 | −0.44% |
| balanced | agsd-gated | 4 035.0 | 4 025.0 | −0.25% |
| sonnet-heavy | vanilla-only | 2 041.9 | 2 045.0 | +0.15% |
| sonnet-heavy | trident-only | 3 155.0 | 3 266.1 | +3.52% |
| sonnet-heavy | agsd-gated | 4 309.2 | 4 428.9 | **+2.78%** |
| code-heavy | vanilla-only | 1 896.3 | 1 884.9 | −0.60% |
| code-heavy | trident-only | 2 964.6 | 2 963.8 | −0.03% |
| code-heavy | agsd-gated | 4 579.4 | 4 453.3 | **−2.75%** |

balanced agsd: −0.25% (noise), sonnet agsd: +2.78%, code agsd: −2.75% — 세 mix 의
signal 이 *완전히 상쇄* 되어 3-mix mean ≈ 0. mix-level variance 가 single-cycle 의 work
load context 에 강하게 의존.

## 3. 부가 지표

- CPU avg: OFF 4.34% → ON 4.68% (+0.34 pp). kv_merge worker 의 duty cycle 0.165 ×
  16 worker / 100 core = +2.6 pp 예측 대비 실측 +0.34 pp — duty cycle 측정대로 0.16
  유효 (cycle/wall = 1688 cycle × 1.648 ms / 84 s = 3.3%, predicted 0.53 pp). 실측
  +0.34 pp 가 prediction 의 같은 order, monitor sampling noise 범위
- GPU avg: 17.41% → 17.54% (+0.13 pp), 동일
- kv_merge worker: 1 688 cycle, avg 1.648 ms/cycle, 50 ms target 대비 duty 3.3% (target
  3-5% ✓)

## 4. 해석 (cold-KV proxy 의 net-positive 가설 reject 확정)

SUB_185 cold-KV proxy (DMA dequant) 가설은 IDE_017 의 paper §4 후보 main lever
였음. 본 SUB 의 side-channel 형태 측정 결과 net 0 → cold-KV decompress 의 isolated
형태도 binding 한 net positive 신호를 만들지 못함.

- SUB_188 softmax precompute +1.84% (regular work, 100 ms cycle, BF16 batch matmul-like)
- SUB_190 tokenize +1.66% (regular work, 20 ms cycle, hash-lookup)
- **SUB_192 kv_merge −0.13%** (regular work, 50 ms cycle, BF16 mul-add)

세 workload 모두 *regular memory access, branch-free*. SUB_192 만 net loss noise.
**가설**: 본 workload 의 256 × 4096 × 64 replicas × 16 worker = 1.0 GiB working-set 이
L3 (96 MiB shared) 와 cross-NUMA HBM 사이의 bandwidth band 와 직접 competition →
vllm 의 cudaMemcpyAsync / pinned alloc 의 bandwidth 와 partial collision. 4 KB block
짧은 packet 흐름인 SUB_188 (152 064 × FP32 = 595 KiB) / SUB_190 (vocab table 17-bit
× 8B = 1 MiB) 와 핵심 차이.

## 5. paper §4 implication

- IDE_017 의 cold-KV path 단독으로는 net positive 0
- side-channel 형태 net positive 의 결정 변수 = **work pattern (regular vs branchy)
  + working-set size** (L3-fit 여부) + **fire rate**
- 본 SUB 의 결과 → IDE_017 paper main 후보군 정리: cold-KV proxy 영역에서의 paper
  §4 main lever 자격 reject. 단 microbench (SUB_166 DMA 35 μs / 54 GB/s) 는 paper §3
  의 background data 로 유효

## 6. 누적 패턴 갱신

lever 18 시도 중 paper-bound net positive 3 (SUB_183 / SUB_188 / SUB_190 모두 1-3%
small), paper main 기준 도달 lever 0. 본 SUB 가 18번째 시도, net 0 으로 추가.

## 7. data / monitor

- benchmark JSON: `measurements/{off,on}/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json`
- monitor csv: `_monitor_{off,on}_{cpu,gpu}.csv`
- kv_merge log: `logs/kv_merge_on.log` (1 688 cycles, avg 1.648 ms)
- aggregate table: `AGGREGATE.md`

## 8. 결론

side-channel partial KV merge **net positive 아님 (−0.13%)**. cold-KV decompress
실측 형태로는 isolated net positive 신호 미확보. paper §4 main lever 자격 IDE_017 영역
에서 추가 reject.
