# SUB_042 — prefill / decode 분리 측정 (★★★ NEO 의 raison d'être 가설 검증)

> **parent**: TSK_019 / 사용자 turn — "(g) prefill/decode 분리 측정"
> **measurement**: HEAD `0d7dc0334`, 3 시나리오 × 2 mode = 6 runs, gmu=0.85, max_num_seqs=256, CPU+GPU util 1Hz sampling.

---

## 1. 측정 결과

| test | mode | input | output | num_prompts | inf_tps | wall (s) | CPU% | GPU% | crash |
|---|---|---:|---:|---:|---:|---:|---:|---:|:-:|
| t1a | vanilla | 4096 | 128 | 200 | 661.7 | 38.7 | 1.93 | 7.3 | 0 |
| t1b | NEO | 4096 | 128 | 200 | 203.2 | 126.0 | 3.52 | 7.0 | 0 |
| t2a | vanilla | 128 | 8192 | 200 | **8160.0** | 200.8 | 2.10 | 17.1 | 0 |
| **t2b** | **NEO** | 128 | 8192 | 200 | **2039.3** | **799.9** | 3.83 | 22.4 | 0 |
| t3a | vanilla | 2048 | 2048 | 200 | 6488.1 | 63.1 | 2.50 | 11.3 | 0 |
| t3b | NEO | 2048 | 2048 | 200 | 1577.2 | 259.7 | 3.40 | 14.7 | 0 |

## 2. 시나리오별 vanilla vs NEO 비교

| 시나리오 | vanilla tps | NEO tps | vanilla 우위 | NEO CPU% | NEO GPU% |
|---|---:|---:|---|---:|---:|
| **prefill-heavy** (in=4096, out=128) | 661.7 | 203.2 | **3.26× faster** | 3.52% | 7.0% |
| **decode-heavy** (in=128, out=8192) — NEO 가설 영역 | **8160.0** | **2039.3** | **★ 4.00× faster** ⚠️ | 3.83% | 22.4% |
| **balanced** (in=2048, out=2048) | 6488.1 | 1577.2 | **4.11× faster** | 3.40% | 14.7% |

→ **vanilla 가 모든 phase 에서 3.26-4.11× 빠름**. NEO 의 어떤 시나리오에서도 net-positive 없음.

## 3. ★★★ 결정적 발견 — NEO 의 raison d'être 가설 깨짐

### 3.1 가설

NEO 의 KV CPU offload 는 **long output decode 영역에서 net-positive** (KV cache 가 GPU memory 초과 시 vanilla OOM 회피).

### 3.2 실제 결과

decode-heavy (input=128, output=8192) 영역에서도 **vanilla 가 4.00× 빠름**.

### 3.3 근거 — KV memory 분석

| 항목 | 값 |
|---|---:|
| 총 token | 200p × (128 + 8192) = 1.66M |
| KV per token per layer (Llama-70B, FP8) | ~16 KiB (8 KV head × 128 HD × 1 byte × 2 for K/V) |
| 총 KV (80 layer) | 1.66M × 16 KiB × 80 = 2.07 TiB ÷ 8 TP = ~265 GiB per GPU |
| H100 (80 GiB) × gmu 0.85 안 | ~68 GiB available — KV 80% 도달 가능 |

→ **단 max_num_seqs=256 영역으로 batch 한정** → 실제 concurrent KV 가 더 작음. vanilla 가 batch limit + paging 으로 OOM 회피.

### 3.4 결론

본 환경 (H100×8, FP8 KV, gmu=0.85, max_num_seqs=256) 의 **모든 워크로드 영역에서 vanilla 의 batch + paging 으로 충분**. NEO 의 CPU offload 가 항상 unnecessary overhead.

## 4. CLAUDE.md `# Objective` 검증

| 목표 | NEO 결과 (3 시나리오 avg) | 평가 |
|---|---|---|
| "CPU 활용률 **극도로** 끌어올리기" | 3.40-3.83% | ❌ 50-90%+ 목표 미달 |
| "CPU **Idle 허락 안 함**" | CPU idle 96-97% | ❌ 거의 idle |
| "GPU 포함 **서버 전체 throughput 향상**" | 모든 시나리오 vanilla 우위 (3.26-4.11×) | ❌ 미달 |

## 5. SUB_032~042 통합 결론 (11 SUB)

| 영역 | 검증 결과 |
|---|---|
| 단일 job throughput | vanilla 2.46-4.11× faster (모든 워크로드) |
| CPU 활용 (CLAUDE.md 목표) | NEO 3.40-11.93% — 모든 워크로드에서 5-12% 만 |
| GPU 활용 | NEO 가 GPU 도 idle 시키거나 (66 vs 73%) wall 길어져서 nominal 동일 |
| Multi-workload 서버 throughput | vanilla+BG > NEO+BG (-13% NEO 손실) |
| **모든 lever (A1-A5, B1, B3, C1)** | **모두 noise** |
| **모든 워크로드 (100p/500p, prefill/decode/balanced)** | **모두 vanilla 우위** |

→ **본 NEO 구현 (HEAD `0d7dc0334`) 은 본 환경에서 vanilla 의 net-positive 영역이 0**. 

## 6. 다음 path

이전 (e) vanilla OOM 영역 측정 의 가치 재검토:

- 본 SUB_042 = decode-heavy long output 영역 (NEO 의 가설 best fit) 에서도 vanilla 4× 우위
- vanilla 가 OOM 인 진정한 영역 = gmu=0.99 + 매우 큰 max_num_seqs (1000+) + 매우 긴 sequence + 매우 많은 prompt 동시. 이건 본 환경 (H100 80GiB × 8) 에서도 도달 가능성 낮음.
- **본 NEO 자체가 본 hardware 영역에서 raison d'être 없음** 결정 시점

| 후보 | 의미 |
|---|---|
| (a) NEO 적용 영역 좁히기 (vanilla OOM 만) | vanilla OOM 영역 존재 여부 확인 — 매우 좁을 가능성 |
| **(b) TSK_019 종료** + 본 NEO 의 net-negative 결론 문서화 | **본 hardware/환경 에서 NEO 가치 없음 결정** |
| (c) 새 IDE — 다른 CPU 활용 architecture | 본 목표 재추구 |
| (d) CLAUDE.md 목표 재검토 | "CPU 극대화" vs "throughput 극대화" trade-off |
| (e) vanilla OOM threshold 측정 — gmu=0.99 + max_num_seqs sweep | 매우 좁은 win 영역 확인 |
| (f) av_amx (SUB_039) 실제 AMX inner loop 구현 + 측정 | NEO 자체 가속 (단 본 SUB_042 결과 후 ROI 매우 작음) |

## 7. raw 자료

| 항목 | 위치 |
|---|---|
| SUMMARY.tsv | `eval/results/20260522_142840_sub042_prefill_decode/SUMMARY.tsv` |
| per-test 결과 (6 dirs) | `eval/results/20260522_142840_sub042_prefill_decode/t1a~t3b/` |
| util/cpu_util.csv + gpu_util.csv | `<each>/util/*.csv` |
| launcher | `/tmp/run_sub042_prefill_decode.sh` |
| stdout log | `/tmp/sub042_prefill_decode.log` |
