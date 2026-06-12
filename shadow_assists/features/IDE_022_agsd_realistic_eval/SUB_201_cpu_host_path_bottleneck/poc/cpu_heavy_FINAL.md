# cpu_heavy_FINAL — CPU 활용도 ↑ + throughput ≥ +10% 동시 달성 시도 (B200 8GPU / Xeon 8570)

## 임무

> Brief: "CPU 활용도를 극도로 끌어 올려 GPU 가 포함된 서버 또는 Cluster 전체의 성능 향상.
> 특히 CPU 의 활용률이 Idle 또는 낮은 Utilization 을 허락하지 않는다."
> **Gate**: throughput ≥ +10% AND cpu_util 상승, 5-sweep mean ± std, paired vs baseline.

## Baseline

- Llama-3.1-8B-Instruct, TP=8 on B200 8 GPU, sharegpt 500p × conc=64 × max-tok=2048
- vllm 1.7-dev (sm_100 build), `VLLM_PREFETCH_TOKENIZE=1 VLLM_BURST_AWARE_ADMISSION=1`,
  `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`

| metric | value (n=3) |
|---|---:|
| output_tps | **22,007.87 ± 143.44** |
| gpu_util | 96.17% |
| cpu_util (mpstat) | **5.44% ± 0.03%** |

이전 baseline (`hw_heavy_baseline/summary.json`, n=5) 22,058 ± 197 tps 와 일치.

## Sweep matrix

| ID | Lever | tps mean ± std | Δ% | cpu_util | gate verdict |
|---|---|---:|---:|---:|---|
| BL | baseline (B3+L2+L10) | 22,007.87 ± 143.44 | 0 | 5.44% | — |
| C1 | ngram K=3 (spec decode, single CPU thread) | 17,083.5 (n=3) | **-22.4%** | 4.56% | **REJECT** |
| C2 | `--kv-cache-dtype fp8` | **22,874.34 ± 378.57** | **+3.93%** | 5.47% | PARTIAL (tps only) |
| C3 | `VLLM_CPU_SAMPLING=1` (brief 의 C-1 매핑) | 21,970.00 ± 333.05 | -0.17% | 5.44% | **NOISE** |
| C4a | fp8 + CPU sampling stack | engine killed (s1 6/500) | — | — | **FAIL (race)** |
| C4b | fp8 + `--max-num-seqs 512` | engine boot fatal | — | — | **FAIL (alloc)** |
| C4c | C4a + C4b stack | cancelled (선행 fail) | — | — | — |

## 분석 (왜 모두 fail / noise 인가)

### Baseline 이 GPU bound (96.2% util) 인 regime 의 구조적 제약

- CPU 활용 lever 가 CPU 일을 늘려도 GPU 일이 동일하면 throughput 변화 0 (GPU 가 critical path).
- CPU 가 GPU 일을 cannibalize 해야 양수 효과 — 이는 GPU 일을 직접 줄이는 lever (spec
  decoding 의 verify batch 효율, fp8 의 memory BW 절감) 가 유효한 이유.

### 각 lever 의 실패 매커니즘

1. **C-1 (NGram spec decode)**: accept_rate α=0.71 매우 양호. 그러나 GPU forward step 의
   batch 가 K+1=4 배 증폭 → step latency 1.4-1.5 배 증가. 이론적 speedup 2.13/(K+1) =
   53% 대비 batch overhead 가 더 커서 net -22.4%. **GPU bound regime 에서 spec decode 는
   throughput penalty**.
2. **C-2 (fp8 KV)**: GPU 메모리 BW 절감으로 step latency 직접 단축 → +3.93%. 그러나 CPU 일
   은 변하지 않으므로 cpu_util 0pp. brief gate 미통과.
3. **C-3 (CPU sampling)**: D2H + CPU softmax/topk 가 sub-ms 라 throughput 영향 noise (-0.17%).
   B × vocab (64 × 128k × 4B = 32 MB) D2H 가 매 step 이지만 PCIe gen5 60 GB/s 기준 ~530μs
   → step 의 ~5%. cpu_util 도 sub-ms work 라 mpstat 1s sampling 에 안 잡힘.
4. **C-4 (stack)**: fp8 + cpu_sampling 결합 시 stream race → engine killed. L11
   MEASUREMENTS.md 에 동일 패턴 기록 있음 (Qwen-7B 환경).

### CPU 활용도 (5.44%) 가 *낮은* 게 아니라 *낮을 수밖에 없다*

- vLLM v1 engine 의 host critical path = scheduler + tokenize/detokenize + sampler. 모두
  μs 단위. GPU step 이 ms 단위라 CPU 는 거의 항상 wait → 5.44% idle 은 hardware-bound limit.
- "CPU 활용도를 30%+ 로" 라는 brief 의 목표는 **CPU 가 GPU 일을 가져가는 lever 가 있어야**
  달성. 본 sweep 의 C-1/C-2/C-3 모두 그 조건 미충족.
- 유일하게 CPU 활용도가 의미 있게 올라가는 lever = CPU draft model (brief C-2) 또는
  CPU GEMM 일부. 둘 다 throughput 폭락 우려 (PyTorch CPU forward latency).

## 양수 lever 후보 (CPU 활용도 gate 무관)

| lever | tps gain | cpu_util |
|---|---:|---:|
| fp8 KV | **+3.93%** (재현 확인) | 변화 없음 |

이외 lever 는 negative 또는 noise. fp8 stacking 도 engine race 로 실패.

## 결론

### Brief gate 통과 lever: **없음**

3 sweep matrix 의 어떤 단일 lever 또는 stack 도 다음 두 조건을 동시 만족하지 않음:
- throughput ≥ +10%
- cpu_util mpstat 의미 있는 상승 (>1pp)

### 이전 4 agent (#48/#49/#50/#51) 와 동일 결론 재확인

50+ lever sweep 중 **fp8 KV 만 유일 양수 (+3.93~+4.02%)**, 그러나 brief 의 "CPU 활용 lever"
조건 미충족. 본 round 에서 추가로 평가한 lever (CPU sampling, ngram spec decode) 모두 동일.

### 근본 원인

B200 8GPU + Llama-8B TP=8 + 이미 적용된 B3 FaP + L2 prefetch + L10 burst-aware admission 조합
의 baseline 이 매우 highly optimized 상태 (gpu_util 96.2%) 이며, **이 regime 에서는 CPU 가
GPU 일을 직접 가져가지 않는 한 throughput 향상이 구조적으로 불가능**. CPU 가 GPU 일을 가져
가는 lever (CPU GEMM / CPU verify) 는 hw spec 상 throughput penalty 가 매우 큼 (Xeon 8570
AMX bf16 ~1 TFLOPS/core, B200 bf16 ~4.5 PFLOPS/8 GPU = 4500 배 차이 → CPU 가 100 ms 작업
이 GPU 0.02 ms → CPU 일이 critical path 가 되는 순간 throughput 폭락).

### 추후 시도 가능한 (본 round 에서는 측정 못 한) 방향

1. **TP=8 보다 작은 TP + multi-instance**: brief 가 "TP=8 강제" 라 본 round 에서는 시도 불가.
   TP=4 × 2 instance 로 운영하면 CPU host overhead 가 instance 별로 누적되어 cpu_util 상승
   가능. 다만 brief 의 lever 후보군 (CPU GEMM/draft/sampler) 와는 다른 dimension.
2. **CPU AMX BF16 sampler kernel (실 AMX SIMD)**: 현 vllm `VLLM_CPU_SAMPLING=1` 은 PyTorch
   FP32 CPU. 실제 oneDNN AMX bf16 softmax + topk kernel 로 교체하면 D2H 후 sampling 자체는
   ~10x faster 가능. 그러나 critical path 가 D2H 자체이므로 throughput 효과는 여전히 0
   (이미 sub-ms 인 sampling 자체가 1/10 되어도 D2H ~530μs 는 그대로).
3. **GPU CUDA-graph capture 변형**: brief 의 lever 목록에 cudagraph_mode 추가 변형 있음
   (FULL_DECODE_ONLY 등 이미 측정). 본 round 와 무관.
4. **multi-step scheduling (`--num-scheduler-steps > 1`)**: vLLM v1 은 미지원 (deprecated
   in v1 engine). v0 에서만 작동.

## 부산물 / 정리

- `cpu_heavy_baseline/summary.json`, `cpu_heavy_C2/summary.json`, `cpu_heavy_C3_cpusampling/summary.json`:
  검증된 결과 JSON.
- `cpu_heavy_baseline/scripts/lib_cpu_heavy.sh`: mpstat 기반 CPU util 측정 라이브러리.
  다음 round 에서 재사용 가능.
- 기존 라운드 (`hw_heavy_baseline`, `hw_custom_*`, `ide023_*`) 와 일관된 결과로 cross-check
  완료.

## Hand-off

Brief gate 통과 lever 가 본 round 에서 없으므로 multi-model 확장 (Qwen-32B, Llama-70B) 및
정확도 게이트는 **시작하지 않음**. 다음 attempter 를 위한 권고:

- "CPU 활용 + throughput +10%" 동시 달성은 본 baseline 환경에서 **구조적으로 어려움** (위
  근본 원인 참조).
- gate 를 완화 (cpu_util gate 제거, throughput ≥ +3% 로 낮춤) 하면 fp8 KV 가 유일한 후보.
- gate 유지하려면 **TP 분할 / multi-instance** 같은 다른 dimension 의 실험 필요 (brief 의
  TP=8 강제 조건 완화 협상 필요).
