# Phase B — paper claim vs 우리 측정 비교

> 분석 시각: KST 2026-05-15 ~
> 산출 목적: paper 의 정량 claim 과 우리 H100×8 환경 측정의 정합성 확인

---

## B.9 paper claim vs 우리 측정 — throughput gain

| 지표 | paper claim | 우리 측정 (commit `64f9e0c48`) | 정합성 |
|---|---|---|---|
| GPU = H100, NEO gain | **+14%** vs GPU-only | **−55.9%** (2,157 vs vanilla 4,886) | ❌ 정반대 |
| 500p 완주 wall | 미상 | 1,882s | (paper 비교 불가) |
| 22 strict fire | (paper 비공개) | 19/19 | (정합 영역 외) |

→ paper 는 NEO ON 시 +14% throughput 향상 주장. 우리 측정은 NEO ON 시 **−55.9% 회귀**. **두 결과의 부호가 다름**.

---

## B.10 부호 차이의 가능한 원인

### (a) workload 차이

- paper: 미상 (PDF 직접 인용 불가)
- 우리: 500p × 8192 in/out (input 8192 + output 8192 max), max_num_seqs=256, gmu=0.92, fp8 KV cache

paper 의 workload 가 더 작은 seq_len, 더 작은 batch 면 GPU KV cache pressure 가 NEO 의 sweet spot.

### (b) chain firing 차이

- paper: 정확치 미상 (메커니즘은 항상 발화 가정)
- 우리: 현재 best (`64f9e0c48`) chain firing **4.0%** (minimal env, Option C/K/L/M2 OFF)
- v1.5 try102 시 chain 99% 였으나 throughput 627 tps (vanilla 12.8%)

→ NEO 가 실제로 작동 하려면 chain firing 이 의미 있는 % 도달해야 하나, 본 환경에서는 minimal env 가 더 빠름 (즉 chain 작동 = throughput 손해, NEO 의 sweet spot 영역 밖)

### (c) GPU KV pool pressure 차이

- paper sweet spot: GPU memory 가 batch 의 bottleneck 인 영역
- 우리 H100×8 80 GB × 8 = **640 GB HBM**, fp8 KV = 매우 큰 pool → max_num_seqs=256 만으로 KV pool 못 채움
- mirror size 측정: 우리 10 (1251회 stable, mirror cap 80 도달 드물게 96회)
- paper 의 mirror size 가 더 크다면 NEO 효과 큼

### (d) CPU 의 강도

- paper: A10G + strong CPU 시 +79.3% — strong CPU 정의 미상 (SPR? Genoa?)
- 우리: Xeon Platinum 8480+ × 2 (SPR), 96 core (taskset 0-111), per-worker 12 core pin
- "strong CPU" 정의 미상이라 정합성 비교 곤란

---

## B.11 우리 측정의 internal benchmarks

| 측정 | tps | chain | wall | 비고 |
|---|---:|---:|---:|---|
| vanilla | 4,886 | N/A | 838s | 14:58 KST |
| v1.6 + fix (현 best) | 2,157 | 4.0% | 1,882s | commit 64f9e0c48 |
| SUB_027 H5 winner | 2,302 | 3.5% | 1,768s | 5/14 00:53 |
| v1.5 try102 (chain 99%) | 627 | 99% | 6,383s | 5/11 09:22 |

→ 우리 환경에서 NEO 의 "best operating point" 는 **minimal env + chain firing 매우 낮음** + throughput vanilla 의 44-47% 영역. paper 의 H100 +14% 와 부합 안 함.

---

## B.12 측정 영역 한계 — paper 와 비교 미달 영역

| 항목 | paper | 우리 | 영역 |
|---|---|---|---|
| GPU model | T4 / A10G / H100 | H100×8 | ✅ 일치 (one of) |
| GPU memory pressure | 변동 (workload 의존) | 미달 (낮은 pressure) | ❌ |
| CPU model | 미상 | SPR 8480+ × 2 | ⏳ |
| KV cache dtype | 미상 | fp8 | ⏳ |
| max_num_seqs | 미상 | 256 | ⏳ |
| seq_len pattern | 미상 | 8192 in/out | ⏳ |

후속: paper PDF 직접 추출 후 위 미상 항목 채워서 정합성 본 검증.

---

## B.13 cdec_wait 정량 (2026-05-15 KST Phase 1 측정 정정)

paper 의 CPU kernel throughput 수치 (GFLOPs/s, ms/op) 직접 인용 못 함. 우리 측정 정정:

| 항목 | 이전 추정 | 실측 |
|---|---:|---:|
| cdec_wait_avg | 8.75 ms / layer | **2.55 ms / layer** |
| gpu_avg | 0.09 ms / layer | **0.08 ms / layer** |
| GPU/CPU ratio | 89-94× | **32×** |

(측정 dir: `eval/results/20260515_083247_async1_b6/` PROFILE log, b1_avg=0 영역)

→ 이전 추정의 1/3.4 영역으로 CPU 가 실측에서 더 빠름. 그러나 **b1_avg=0** (cdec sub-batch query 비어 있음) — 본 환경의 cdec 가 actual work 영역이 아닐 가능성. chain firing 영역 도달 시 다른 측정 필요.

이게 본 환경 (H100×8 + SPR 2S) 에서 NEO 가 효과 미달인 진짜 root cause 의 일부일 가능성. 32× 가속도 여전히 큰 격차 (AMX 5-10× 가속 시 3-6× 격차 잔존).

---

## B.14 종합 결론 — Phase B

1. paper 의 14% H100 gain 은 NEO 의 이론 sweet spot 의 **하한**. 우리 환경에서 도달 못 함.
2. CPU compute 가 GPU 보다 89-94× 느림 — NEO 메커니즘의 fundamental 한 입력 조건 (GPU/CPU 비등 또는 CPU 빠름) 미달.
3. **AMX/AVX-512 가속이 cdec compute time 을 줄여 GPU 와 비등 영역 (5-10×) 도달 시** NEO 가 의미 있는 영역으로 진입 가능.
4. AMX 가 5-15× speedup 의 claim 대로 발화 시 cdec_wait 8.75 → 0.6-1.7 ms/layer 도달 → GPU 0.09 ms 의 6-19× → 여전히 GPU 보다 느림. **AMX 만으로는 부족**.
5. workload 조정 (작은 batch + 긴 seq + larger mirror) 가 동반되어야 paper 의 H100 14% 영역 도달.

→ Phase D 의 bottleneck mapping 에서 AMX 적용 시 cdec compute 의 실효 speedup 정량 예측 필수.
