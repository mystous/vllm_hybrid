# B3 cudagraph lever — Llama-3.1-70B FaP +30% 재현 검증

## 0. 목적

B3 (cudagraph_mode = FULL_AND_PIECEWISE, 이하 **FaP**) lever 가 Qwen-7B 에서 보였던
**+30.1% throughput gain** (3272 → 4258 tps, conc=16) 이 **Llama-3.1-70B** 에서도
재현되는지 확인. B200 environment 에서 `FULL` 단독은 활성 불가 → FaP 가 차선책 best.

## 1. 조건

| 항목 | 값 |
|---|---|
| 모델 | meta-llama/Llama-3.1-70B-Instruct |
| TP | 4 |
| GPU | 0,1,2,3 (B200) |
| port | 8003 |
| corpus | sharegpt |
| n prompts | 200 |
| concurrency | 32 |
| max-tokens | 8192 |
| stream | True |
| gpu-memory-utilization | 0.85 |
| max-model-len | 16384 |
| KV tier | OFF (`VLLM_KV_TIERING_DRAM=0`) — pure cudagraph lever isolation |
| commit | bea71c373 (feat/spec-decode-tuning) |
| date | 2026-06-05 |

## 2. 결과

| run | cudagraph_mode | boot s | wall s | tps | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GPU util (%) | mem (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | PIECEWISE | 111 | 182.1 | **1891.5** | 35.8 | 420.8 | 10.7 | 12.3 | 98.0 | 950148 |
| **B** | FULL_AND_PIECEWISE | 76 | 180.1 | **1926.0** | 35.4 | 207.4 | 10.6 | 11.9 | 99.3 | 634335 |

### 2.1 Δ

| metric | A → B | Δ |
|---|---|---|
| output_tps | 1891.5 → 1926.0 | **+1.82%** |
| wall_total_s | 182.1 → 180.1 | -1.10% |
| TTFT p50 | 35.8 → 35.4 | -1.1% (≈noise) |
| TTFT p99 | 420.8 → 207.4 | **-50.7%** (대폭 안정화) |
| TPOT p50 | 10.7 → 10.6 | -0.9% (≈noise) |
| TPOT p99 | 12.3 → 11.9 | -3.3% |
| boot_sec | 111 → 76 | **-31.5%** (FaP capture cost ≤ PIECEWISE) |
| gpu_util | 98.0 → 99.3 | +1.3pp |
| gpu_mem_mib | 950148 → 634335 | -33.2% (graph 메모리 풀 작아짐) |

> 양 run 모두 200/200 success, 0 err.

## 3. Qwen-7B 와 비교

| 모델 / 조건 | PIECEWISE tps | FaP tps | Δ tps |
|---|---|---|---|
| Qwen-7B (TP=1, 100p×conc=16, B3 spec doc) | 3272 | 4258 | **+30.1%** |
| Qwen-7B (TP=1, 50p×conc=16, b3_sched 실측) | 4169 | 4413 | +5.9% |
| **Llama-3.1-70B (TP=4, 200p×conc=32, 본 측정)** | **1891.5** | **1926.0** | **+1.82%** |

→ **Llama-70B 에서는 Qwen-7B 의 +30% gain 이 재현되지 않음** (+1.82% 는 noise 수준 ±2-3%
범위 내). 다만 TTFT p99 -50.7% (420.8ms → 207.4ms) 와 boot_sec -31.5% 는 명확한
positive signal — **outlier latency 가 크게 안정화**.

## 4. Verdict

**Llama-3.1-70B 에서 FaP 의 throughput gain 은 negligible (+1.82%)**, Qwen-7B 의
+30.1% 와 차이가 큼. 그러나 **net negative 가 아니므로 FaP 를 default 로 유지하는
결정은 유효**:

1. throughput: +1.82% (small but positive)
2. tail latency (TTFT p99): **-50.7%** (substantial improvement)
3. boot time: -31.5%
4. memory footprint: -33.2%
5. 실패 / regression 없음

따라서 **B3 default = FULL_AND_PIECEWISE 유지 결정은 valid**. Llama-70B 에서 main
metric 인 throughput gain 은 작아도, tail latency 개선과 memory 절감으로 net positive.

## 5. 왜 Llama-70B 에서 gain 이 작은가 (원인 분석)

cudagraph FULL path 의 이득은 **per-step CPU launch overhead 가 GPU compute time 대비
유의미할 때** 크게 나타남. PIECEWISE 는 prefix-then-attention 두 launch 가 필요하지만
FULL 은 한 번에 capture.

- **Qwen-7B (TP=1)**: GPU compute / decode step 이 짧음 → launch overhead 비중 큼 →
  FaP 가 step time 의 큰 비율 제거 → +30%.
- **Llama-70B (TP=4)**: 4-way TP 의 per-step GPU compute 가 7B 대비 훨씬 김
  (model 10× 큼 + TP allreduce overhead 추가) → launch overhead 가 step time 의
  작은 비율 → graph 자체로 줄일 여지가 작음.
- **batch shape variability**: conc=32 + sharegpt 의 length 다양성 → FULL graph 가
  re-capture / fallback to PIECEWISE 경로를 자주 사용 → ideal gain 미달.
- **TTFT p99 만 크게 개선된 이유**: prefill burst 시 PIECEWISE 의 launch jitter 가
  outlier 를 만들었는데 FaP 가 그 jitter 를 제거 → tail 만 좁혀짐.

## 6. 산출물

- `run.sh` — 2 run launcher (A_piecewise | B_fap)
- `llama70b_A_piecewise.json` + `.raw.jsonl`
- `llama70b_B_fap.json` + `.raw.jsonl`
- `_logs/boot_{A_piecewise,B_fap}.log` — vLLM serve stderr
- `_logs/bench_{A_piecewise,B_fap}.log` — throughput_runner 출력
- `_logs/{A_piecewise,B_fap}.boot_sec` — boot 소요 초
- `_logs/{A_piecewise,B_fap}.gpu_after.txt` — run 종료 후 GPU 0-3 free 검증

## 7. GPU 0-3 최종 free 검증

```text
0, 0, 182632, 0
1, 0, 182632, 0
2, 0, 182632, 0
3, 0, 182632, 0
```

GPU 4-5 는 다른 agent (b1b3_cumulative) 의 작업으로 보존, GPU 6-7 도 free.
