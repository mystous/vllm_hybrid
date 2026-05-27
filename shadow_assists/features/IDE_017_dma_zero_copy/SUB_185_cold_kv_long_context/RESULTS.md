# SUB_185 — IDE_017 cold-KV decompress long-context workload e2e first signal

> **parent**: SUB_178 conditional accept follow-up (NEW workload e2e first signal).
> **scope**: 2026-05-27 12:22 ~ 12:31 KST (~9 min wall, OFF + ON chained).
> **status**: ✅ 완료 — long-context 8K input workload + cold-KV concurrent dequant proxy 측정 완료. **net positive 미확인** ⚠

---

## 0. 두괄식 — long-context proxy net positive 미확인

| 측정 | OFF | ON (concurrent cpu_dequant firer) | Δ |
|---|---:|---:|---:|
| **throughput tps** | **103.21** | **103.40** | **+0.18%** (noise) |
| latency p50 (s) | 9.870 | 9.870 | 0.00% |
| **TTFT p50 (s)** | 4.691 | **5.105** | **+8.83% regression** ⚠ |
| TTFT p99 (s) | 9.781 | 9.514 | −2.73% |
| CPU util mean | 3.42% | 3.56% | +0.14 pp (negligible) |
| paper §4 main lever 자격 | — | — | **미확인** (proxy 한계) |

throughput 은 noise floor, **TTFT 만 +8.8% 회귀** (cold-KV firer 가 prefill critical path 와 contending). proxy 측정의 한계 — real KV memory inject 없이 별도 thread 에서 dequant 호출만 fire 한 setup.

---

## 1. workload spec — long-context 8K input

| 항목 | 값 |
|---|---|
| target_tokens | 8,000 |
| sampled_token_lens p50 | **8,711** (target 달성) |
| n_lines_per_prompt | 811 (Pile/SlimPajama subset chunk repeat) |
| tokens_per_line | 10.84 |
| n_prompts | 500 |
| n_ok | 500 (errors 0) |
| concurrency | 32 |
| max_tokens | 32 (decode short, prefill-dominant) |
| max-model-len | 8,192 (8K input fits) |
| wall (per mode) | ~155 sec (boot 70-121 sec + bench 155 sec) |

8K input × max-tokens=32 → **prefill-bound workload** (TTFT >> decode time). 본 setup 의 의도: SUB_178 cold-KV decompress 가 prefill 의 KV 초기화 과정과 overlap 가능성 측정.

## 2. ON mode setup (proxy)

- vllm prefill 진행 중 별도 background thread (`firer`) 가 `cpu_dequant_int8_bf16(...)` (SUB_178 의 .so) 호출 fire
- 실제 vllm prefill path 의 KV 초기화 대체 ❌ (proxy 측정)
- ENV: 그 외 동일 (RAYON/OMP/MKL=4)

**proxy 의 의미**: real integration 은 vllm 의 prefill path 에 inject 필요 (invasive vllm core 수정). 본 SUB 의 first signal = "concurrent CPU dequant 이 GPU prefill 의 critical path 에 미치는 영향" — overlap 이면 net positive / contending 이면 regression.

## 3. 9-cell 상세 (1 cell × 2 mode 만 — long-context single scenario)

| metric | OFF | ON | Δ% |
|---|---:|---:|---:|
| tps | 103.21 | 103.40 | +0.18% |
| latency p50 | 9.870s | 9.870s | 0.00% |
| latency p99 | 12.992s | 12.100s | **−6.87%** (small positive) |
| TTFT p50 | 4.691s | 5.105s | **+8.83%** ⚠ |
| TTFT p99 | 9.781s | 9.514s | −2.73% |
| wall_s | 155.0 | 154.7 | −0.19% |
| total_out_tokens | 16,000 | 16,000 | 동일 |

### 3.1 핵심 observation

1. **throughput noise (+0.18%)**: proxy 측정으로 actual KV initialization 대체 없음 → critical path 변화 없음. 단 cold-KV firer 의 fire-and-forget 호출 자체가 가벼움.
2. **TTFT p50 +8.83% regression**: prefill 진행 중 별도 thread 의 CPU work 가 GPU launch tail / KV allocation step 과 contending (Linux scheduler migration / cache pollution / pinned alloc lock contention 추정 — SUB_184 와 동일 패턴).
3. **TTFT p99 −2.73% / latency p99 −6.87%**: tail latency 약간 개선 — high-load 분기에서 cold-KV firer 의 yield 효과 가능성 (1-run noise 가능성 큼).
4. **CPU util mean +0.14 pp**: concurrent firer 의 dequant 호출이 너무 light (100 Hz × ms-level) — paper target 30%+ 와 무관.

## 4. SUB_178 의 conditional 자격 재평가

SUB_178 microbench 결과 (overlap speedup 1.5-1.71×) 는 **isolated CPU vs GPU dequant 비교**. 본 SUB_185 의 e2e proxy 는 **concurrent CPU-side dequant fire 가 vllm prefill 에 미치는 영향** — 두 측정의 scope 가 다름.

| 측정 형식 | SUB_178 (isolated microbench) | SUB_185 (e2e proxy) |
|---|---|---|
| CPU dequant cost | 측정 (isolated, fast) | 동일 .so 호출 |
| GPU prefill 대체 여부 | yes (계산 path 가 CPU 로 이동) | **no (proxy, vllm path 그대로)** |
| critical path 영향 | speedup 1.5-1.71× | **TTFT +8.83% regression** (contention) |
| accuracy | bit-exact 가능 | by construction PASS (vllm path 미변경) |

→ **isolated speedup → e2e net positive 자동 변환 안 됨**. SUB_178 conditional 자격 유지하려면 **real vllm prefill path integration** 필요 (invasive, 별도 SUB).

## 5. paper §4 implication

- SUB_184 phase-burst reject 이후 cold-KV decompress 가 paper main lever 1순위 후보였으나, **본 SUB_185 proxy 단계 net positive 미확인** + TTFT regression 검출.
- conditional 자격 회복하려면:
  - real KV path integration (vllm prefill KV initialization 의 일부를 CPU dequant 로 대체)
  - INT8 quantized KV pre-store + on-demand CPU dequant + DMA push 의 full pipeline
  - 위 2개 모두 invasive vllm core 수정 (별도 SUB 단위 heavy work)
- **honest assessment**: cold-KV decompress 도 drop-in CPU lever (kernel) 와 동일 패턴 — isolated speedup 이 e2e net positive 로 자동 변환 안 됨. paper §4 main lever 후보 **약화**.

## 6. 누적 패턴 update (SUB_173~185)

| 카테고리 | 시도 | net positive | noise floor | net negative |
|---|---:|---:|---:|---:|
| drop-in CPU kernel | 7 | 0 | 5 | 2 (174, 181) |
| environment-level | 2 | 0 (183 +1.54% < 5%) | 2 | 0 |
| paper main lever (IDE_018) | 1 | 0 (184 −1.75%) | 0 | 1 |
| **NEW workload e2e (proxy)** | **1 (본 SUB)** | **0 (+0.18%)** | **1** | **0** (TTFT만 +8.83%) |
| **누적** | **11** | **0** | **9** | **2 (+1 partial: 185 TTFT)** |

paper §4 main lever 후보 모두 net positive 확보 못함 (11 lever 시도). paper-bound argumentation 은 (a) negative result 의 honest aggregate + (b) microbench feasibility / (c) IDE_015 Phase A 의 CPU 측정 가치 중심으로 재구성 필요.

## 7. 다음 step

- **SUB_186 = SUB_182 + SUB_183 environment stack measurement** (cgroup + hugepages + taskset + NUMA pin 동시) — paper-bound 환경 stack 의 합산 효과 검증
- **SUB_188 = side-channel batch precompute** (NEW workload 후보 별개 lever)
- **SUB_194 = multi-run variance verification** — Top 후보 (SUB_183 +1.54%, SUB_173 +0.86%, SUB_185 +0.18%) 의 3-run 재측정

## 8. raw data

- `measurements/{off,on}/long_sonnet/bench.json` (2 cell × 12 metric)
- `_monitor_{off,on}_{cpu,gpu}.csv` (0.5s interval × 155 sec)
- `logs/{main,vanilla,trident,monitor,chain}_{off,on}.log`
- `launcher.sh` + `chain.sh` + `aggregate.py`
- `src/` — long-context prompt generator + cold-KV firer
