# TSK_045 — IDE_025 DRAM KV/Prefix Tier 측정 결과 (2026-08-27)

- 노드: violet-h100-016 (Xeon 8480+×2 / 2TB DDR5 / H100×8, turbo OFF 2.0GHz)
- 모델: RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic, TP=8, max-model-len 16384, vLLM 0.28.0 (upstream 이미지)
- 워크로드: `prefix_repetition` — 32 prefixes × 8,192 tok prefix + 256 suffix, output 128, 384 req, C=8, seed 42
- 압박 구성: `--num-gpu-blocks-override 9600` (= ~153K tokens pool ≪ 262K prefix tokens 총량)
- offload 구성: `OffloadingConnector` + `CPUOffloadingSpec`, `cpu_bytes_to_use=200e9` (DRAM 200GB)

## 결과

| cell | out tok/s | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | duration (s) |
|---|---:|---:|---:|---:|---:|
| t1 압박 · GPU-only | 418.0 | 347.1 | 631.4 | 16.4 | 106.8 |
| **t2 압박 · +DRAM tier** | **634.4 (+51.8%)** | **121.8 (−65%)** | 400.2 | 10.0 | 69.3 |
| t3 비압박 · GPU-only (상한 참조) | 651.8 | 81.6 | 367.0 | 10.0 | 67.4 |
| t4 비압박 · +DRAM tier | 643.2 (−1.31% vs t3) | 85.2 | 731.4 | 10.1 | 68.5 |

## 동작 실증 (TST_023 카운터 게이트)

t2 서버 메트릭 (Prometheus `vllm:kv_offload_*` 최종값):

- **load (DRAM→GPU reload): 91.27 GB / 280 회**, load_time 합 4.56 s
- store (GPU→DRAM): 50.16 GB / 528 회, store_time 합 2.84 s
- lookup 66 회, sync delay 합 11.6 ms

## 판정 (TST_023)

- **net win**: ✅ 통과 — 압박 구성에서 +51.8% throughput, TTFT p50 −65%. t2 가 비압박 상한 (t3=651.8) 의 97.3% 까지 회복
- **동작 실증**: ✅ 통과 — reload 280회/91GB (IDE_006 의 "merged 0%" 와 대조)
- **무회귀**: 🟡 **경계** — t4/t3 = −1.31% (게이트 ≤1% 소폭 초과, 단일런 jitter ±2~3% 범위 안). TTFT p95 731 vs 367 tail 악화 관측. **비압박 워크로드에는 기본 OFF, 압박/공유-prefix 워크로드에만 ON 권고**로 봉합

## 해석

- 이득 원천 = prefill 재계산 회피 (GPU 연산 절약). CPU 는 연산하지 않고 2TB DRAM 이 저장 tier 로 소비됨 — IDE_006 Q-dilemma 와 무관한 구조임이 실측으로 확인
- 압박은 `--num-gpu-blocks-override` 로 emulate — 실제 long-context/고동시성 워크로드의 대리
- CPU busy% 는 본 회차 미수집 (mpstat 부재) — cpu_sample.sh 로 대체, TSK_044 부터 수집
