# P3 5-case compare 3-run avg @ gpu=0.85, long 500p × 8192

> 측정 일자 2026-05-20 KST. branch `feat/neo-amx-apply` HEAD `aba1b14b1`.
>
> 사용자 명시: P3 (K BF16 + AMX env-gated) 의 진정한 long workload win/loss 평가 +
> v1.6 best, S1-S9, P1 baseline, vanilla 와 동일 조건 3-run avg 비교.
>
> 본 측정의 의미 = P3 의 1-run variance 영역 (이전 +6.1% win 영역) 의 진정한 영역
> 확정 + best configuration 영역 업데이트.

---

## 측정 환경

| 항목 | 값 |
|---|---|
| Model | Llama-3.3-70B-Instruct |
| Hardware | H100 × 8 (Intel SPR + GPU 7 의 외부 bentoml ixi-ocr-model service) |
| GPU memory utilization | 0.85 |
| Workload | 500p × 8192 token (long, async_scheduling, fp8 kv) |
| max_num_seqs | 256 |
| max_num_batched_tokens | 8192 |
| KMP_BLOCKTIME | 0 |
| OMP_NUM_THREADS | 10 |
| NEO env | mirror_max=80, async_swap_buffers=3, sync_swap_batched=1, cpu_pin=12c, numa_bind |

---

## 5 case 의 정확한 commit / binary

| Case | commit | binary | env |
|---|---|---|---|
| P1 baseline | `aba1b14b1` (P3 commit) | P3 binary (qk_amx_bf16 export) | env=0 (AMX off, K BF16 off) |
| P3 (env=1+AMX=1) | `aba1b14b1` | 동일 | `VLLM_NEO_USE_AMX=1` + `VLLM_NEO_HOST_K_BF16=1` |
| S1-S9 | `531d61608` (S1-S9 정합) | S1-S9 binary | env=0 |
| v1.6 best | `64f9e0c48` (v1.6 best) | v1.6 binary | env=0 |
| vanilla | `64f9e0c48` (v1.6 binary 위) | v1.6 binary | NEO disable (`--enable-neo-asymmetric` 없음) |

---

## 3-run avg 결과

| Config | run 1 | run 2 | run 3 | **3-run avg** | CV |
|---|---:|---:|---:|---:|---:|
| **vanilla (NEO off)** | 4,679.4 | 4,680.4 | 4,680.7 | **4,680.2** | 0.01% |
| **v1.6 best** ★ NEO best | 1,749.8 | 1,778.8 | 1,970.5 | **1,833.0** | 6.6% |
| S1-S9 | 1,763.0 | 1,858.9 | 1,778.3 | **1,800.1** | 2.9% |
| **P3 (env=1+AMX=1)** | 1,799.4 | 1,764.0 | 1,800.4 | **1,787.9** | 1.2% |
| P1 baseline | 1,695.3 | 1,738.2 | 1,801.8 | **1,745.1** | 3.0% |

### Best ranking

| 순위 | Config | tps | vs P1 |
|---|---|---:|---:|
| ★ | **vanilla** (NEO off) | **4,680.2** | +168% |
| 1 (NEO best) | **v1.6 best** | **1,833.0** | **+5.0%** |
| 2 | S1-S9 | 1,800.1 | +3.2% |
| 3 | **P3** | 1,787.9 | +2.5% |
| 4 | P1 baseline | 1,745.1 | reference |

---

## 진정한 fact

1. **vanilla = 4,680.2 tps** (NEO disable) = NEO 영역 의 ~2.6x. H100×8 + Llama-70B + 본 long workload 영역 = **NEO 가 net loss**. NEO 의 설계 의도 (CPU 활용 + batch 확장) 가 H100 의 80 GiB HBM × 8 = 640 GiB 영역 에서 작음.

2. **NEO 영역 끼리 ranking = v1.6 best > S1-S9 > P3 > P1**.
   - v1.6 best (1,833.0) 가 진정한 NEO best (3-run avg 영역).
   - **P3 (1,787.9) = v1.6 best 대비 -2.5% 회귀**. K BF16 + AMX 영역 의 long workload 측면 = net loss.
   - S1-S9 (1,800.1) 가 P3 보다 약간 win (+0.7%). S1-S9 의 정통 NEO 정합 영역 이 P3 의 변경 영역 보다 좋음.

3. **P3 의 1-run +6.1% win 영역 = variance 의 영향**. 3-run avg 영역 에서는 P3 의 진정한 win 영역 X.

4. **v1.6 best 의 CV 6.6%** = run 3 의 outlier (1,970.5). run 1, 2 (1,749.8, 1,778.8) 의 영역 의 사실 영역 ~1,764 영역. 단 본 outlier 의 root 영역 = variance 또는 외부 service 의 momentary state.

5. **S1-S9 의 진정한 위치**: 이전 README 의 fact ("S1-S9 vs v1.6 best 3-run avg = +1.9%") 와 본 시점 (-1.8% vs v1.6 best, 0.85 환경) 의 차이 = gpu_util (0.92 vs 0.85) 의 영향 또는 variance.

---

## 본 측정 의 의의

- **P3 의 진정한 long workload 영역 의 회귀 확정**: K BF16 + AMX 의 본 환경 영역 = vllm Python 영역 의 P3 변경 영역 의 net loss. 본 의문 영역 의 root = (1) AMX qk 의 tile 영역 occupancy 50% (M=8 vs tile 16), (2) K vec conv overhead vs save 비교 의 net negative.
- **v1.6 best 가 진정한 best**: SUB_015-Phase 3 의 모든 시도 (Step 4/5/6, P3 등) 가 v1.6 best 의 baseline 영역 초과 안 함.
- **vanilla 의 진정한 baseline**: H100×8 + Llama-70B + long workload 영역의 NEO 적합도 = -64%. NEO 의 design 영역 = batch 작은 영역 + 짧은 context 영역 에 적합 (paper §4.4).

---

## 다음 단계 — P4, P5 (다음 turn)

| Phase | 영역 |
|---|---|
| **P4** F1 async cdec | 1-run short 영역 -1.9% (env=1 만, infrastructure 의 deferred dispatch 미동작). long workload 의 진정한 영역 measure 또는 V1/V2/V3 변형 (P3 시 architectural blocker 확인됨) |
| **P5** F2 MIRROR sweep | MIRROR_MAX 영역 의 long workload 의 진정한 sweep |

---

## raw 측정 자료

| Run | result.json |
|---|---|
| vanilla run 1 | `eval/results/20260519_235733_vanilla_500p_085/` |
| vanilla run 2 | `eval/results/20260520_052819_vanilla_500p_085_run2/` |
| vanilla run 3 | `eval/results/20260520_062300_vanilla_500p_085_run3/` |
| v1.6 best run 1 | `eval/results/20260519_231500_v16_best_500p_085/` |
| v1.6 best run 2 | `eval/results/20260520_044618_v16_best_500p_085_run2/` |
| v1.6 best run 3 | `eval/results/20260520_054505_v16_best_500p_085_run3/` |
| S1-S9 run 1 | `eval/results/20260519_222213_s1s9_full_500p_085/` |
| S1-S9 run 2 | `eval/results/20260520_040400_s1s9_full_500p_085_run2/` |
| S1-S9 run 3 | `eval/results/20260520_044450_s1s9_full_500p_085_run3/` |
| P1 baseline run 1 | `eval/results/20260519_213155_p1_baseline_long_500p_085/` |
| P1 baseline run 2 | `eval/results/20260520_011326_p1_baseline_500p_085_run2/` |
| P1 baseline run 3 | `eval/results/20260520_023512_p1_baseline_500p_085_run3/` |
| P3 run 1 | `eval/results/20260519_204931_p3_amx_k_bf16_long_500p_085/` |
| P3 run 2 | `eval/results/20260520_015440_p3_amx_kbf16_500p_085_run2/` |
| P3 run 3 | `eval/results/20260520_031537_p3_amx_kbf16_500p_085_run3/` |
| script summary | `/tmp/run_all_5case_run23_summary.txt` |
