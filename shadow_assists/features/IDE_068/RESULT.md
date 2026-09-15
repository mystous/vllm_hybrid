# IDE_068 결과 — CPU MoE 오프로딩의 서버 한계

> 노드 violet-h100-016 — H100 80 GB ×8 / Xeon Platinum 8480+ ×2 (AMX, 112C/224T) / DDR5 2 TB / **turbo OFF 2.0 GHz 고정**.
> 스택 SGLang 0.5.18 + kt-kernel 0.7.0.post1 (컨테이너 `sgl-kt`). 벤치 `vllm bench serve --backend openai`, sonnet 512/128 (prefix 100), seed 42.
> 진행 로그 `PROGRESS.md`. 원본 `eval/results/20260915_*_ide068_*/`. 표는 `eval/ide068/summarize.py` 로 원본에서 생성.
>
> **모든 CPU 수치는 turbo OFF 하한이다. CPU busy 는 성공 지표가 아니며, binding 지표는 처리량·TTFT·TPOT·greedy 4문항이다.**

## 질문과 답 (요약)

| | 질문 | 답 | 상태 |
|---|---|---|---|
| **B** | GPU-only 로 8장이 필요한 가장 큰 보유 모델(Qwen3-Coder-480B-A35B-FP8, 450 GB)을 오프로딩하면 GPU 를 몇 장까지 줄일 수 있는가 | **1장** — TP1 에서 서빙 성립, greedy 4/4, 43.24 tok/s. TP4·TP2·TP1 처리량이 같다 | 본 셀·B2 완료 |
| **A** | 이 서버가 CPU MoE 오프로딩으로 지원할 수 있는 최대 모델 | 후보 Kimi-K2-Instruct 1.03 TB (공개 최대 MoE) | 다운로드 중 |

---

## 실험 B — 480B 를 GPU 몇 장까지 줄일 수 있는가 (TSK_050)

### B.1 구성

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` — 62층, hidden 6144, 160 experts/층, top-8, moe_intermediate 2560, GQA 96/8, FP8 block 128×128 |
| 크기 | HF snapshot 450 GB. CPU expert 변환본 `kt/qwen3-480b-int4` 232 GB (67 shards, `kt quant -m int4 -i fp8`, TSK_047 산출) |
| 비-expert 파라미터 | attention 62×≈164 M + embedding/lm_head 2×0.93 B + router ≈ **12 B** (산술). 이것이 GPU 에 남는 부분이다 |
| 워크로드 | sonnet 입력 512 / 출력 128 / prefix 100, 64 요청, 동시성 16, request-rate inf, seed 42 |
| 품질 | greedy 4문항 — `The capital of France is` / `def fibonacci(n):` / `1+2+3+...+100 =` / chat "reverse a string" |
| 공통 플래그 | `--attention-backend triton --trust-remote-code`. 하이브리드는 `--disable-cuda-graph --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 0` |

### B.2 결과 — 본 셀 (`eval/results/20260915_092544_ide068_expB_480b/`)

| 셀 | 구성 | 부팅 | greedy | 완료 | 출력 tok/s | 전체 tok/s | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms | TPOT p95 ms | 벤치 s | CPU busy 평균/최대 % | HBM GiB/장 | DRAM used GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **b0** | GPU-only **TP8 + EP8** | HEALTH OK 140 s | **4/4** | 64/64 | **701.60** | 3,453.17 | 335.16 | 455.27 | 20.14 | 21.10 | 11.68 | 5.8 / 9.0 | 71.9~72.4 ×8 | 116 |
| b0a | GPU-only TP4 | **DIED 40 s (OOM)** | — | — | — | — | — | — | — | — | — | — | — | — |
| **b1** | 하이브리드 TP4 | HEALTH OK 70 s | **4/4** | 64/64 | **43.39** | 213.56 | 8,622.12 | 11,317.95 | 299.05 | 302.98 | 188.80 | 45.1 / 51.1 | 16.6~17.1 ×4 | 315 |
| **b2** | 하이브리드 TP2 | HEALTH OK 70 s | **4/4** | 64/64 | **43.33** | 213.27 | 8,643.05 | 11,075.44 | 300.07 | 302.02 | 189.06 | 44.4 / 49.0 | 19.4 ×2 | 309 |
| **b3** | 하이브리드 **TP1** | HEALTH OK 60 s | **4/4** | 64/64 | **43.24** | 212.82 | 8,694.06 | 11,008.23 | 300.12 | 304.42 | 189.46 | 43.9 / 48.3 | **24.0 ×1** | 307 |

b0a 의 OOM 근거 (`b0a_gpu_tp4/error_lines.log`):
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 600.00 MiB. GPU 3 has a total capacity of 79.18 GiB of which 269.25 MiB is free.`

### B.3 읽는 법

1. **GPU-only 기준선을 처음 확보했다.** TSK_047 에서 순수 TP8 은 FP8 block 제약(`output_size 320 not divisible by block_n=128`)으로 SGLang 이 분할하지 못했다. `--ep-size 8` 로 expert 를 통째로 장당 20개씩 나누면 분할 문제가 사라진다. 701.60 tok/s, TPOT 20 ms.
2. **GPU-only 는 8장이 필요하다.** TP4 는 40초 만에 OOM (TSK_047 재현). 450 GB / 4 = 112 GB > 80 GB 이므로 산술상 당연하다.
3. **오프로딩하면 1장으로 줄어든다.** TP1 에서 부팅 60초, 64/64 완료, greedy 4/4. HBM 24.0 GiB — 비-expert 가중치(≈14 GB) + KV pool(max-total-tokens 32768).
4. **처리량은 GPU 장수와 무관하다.** TP4 / TP2 / TP1 = 43.39 / 43.33 / 43.24 tok/s, TPOT ≈ 300 ms 로 동일. 스텝 시간이 CPU expert 구간에 묶여 있어 GPU 를 늘려도 빨라지지 않고, 줄여도 느려지지 않는다. 기준선 대비 **6.2 %** (TPOT 14.9배).
5. 따라서 이 구성에서 GPU 4장 → 1장으로 줄이는 데 **처리량 대가가 없다.** 대가는 GPU-only 를 포기하는 순간 이미 치른 것(701 → 43)이다.
6. b1 은 TSK_047 (44.5 tok/s, C16 64req) 을 재현했다.

### B.4 B2 — 남는 HBM 에 expert 를 올리면 얼마나 회복되는가 (`eval/results/20260915_094613_ide068_expB2_480b_gpuexperts/`)

IDE_030 은 TP4 에서 hot-expert 96개 + hotmap(빈도 기반 배치) + deferral 로 490.9 tok/s 를 얻었다. 그 hotmap 산출물은 소실됐으므로 여기서는 `--kt-num-gpu-experts N` 만 준다 — 물리 id 0..N−1 이 GPU 에 오르며, 이 결과는 **hot-expert 이득의 하한**이다.

| 셀 | 구성 | 부팅 | greedy | 출력 tok/s | 대비 (expert 전량 CPU) | TTFT p50 ms | TPOT p50 ms | 벤치 s | CPU busy % | HBM GiB/장 |
|---|---|---|---|---|---|---|---|---|---|---|
| b3 | TP1, GPU expert 0 (참조) | 60 s | 4/4 | 43.24 | — | 8,694 | 300.1 | 189.5 | 43.9 | 24.0 ×1 |
| **b4** | **TP1 + GPU expert 16** | 70 s | 4/4 | **46.81** | **+8.3 %** | 8,014 | 274.0 | 175.0 | 43.7 | 65.7 ×1 |
| **b5** | **TP2 + GPU expert 40** | 90 s | 4/4 | **49.90** | **+15.2 %** | 6,891 | 268.0 | 164.2 | 43.9 | 70.0 ×2 |
| **b6** | **TP4 + GPU expert 96** | 100 s | 4/4 | **51.55** | **+18.8 %** | 4,548 | 273.4 | 158.9 | 44.1 | 75.7~76.2 ×4 |

- GPU expert 는 FP8 원본에서 올라간다 (층당 1개 ≈ 47 MB → 16개×62층 ≈ 46 GB, b4 의 HBM 24.0 → 65.7 GiB 증가분과 일치).
- 빈도 무관 배치(물리 id 0..N−1)라 hot expert 가 아니다. 160개 중 16개 = 10 %, 40개 = 25 %, 96개 = 60 % 를 GPU 로 옮겼는데 처리량은 +8 / +15 / +19 % 에 그친다 — top-8 라우팅이 앞쪽 id 에 몰리지 않기 때문이다. IDE_030 이 같은 96개를 **빈도 상위**로 배치해 490.9 tok/s 를 얻은 것과의 차이가 hotmap 의 가치다.
- TTFT 는 GPU expert 수에 비례해 줄었다 (8.7 → 8.0 → 6.9 → 4.5 s). prefill 은 모든 expert 를 건드리므로 GPU 에 있는 expert 비율만큼 CPU 구간이 줄어든다.

### B.5 B3 — TP1 에서 cuda graph 를 켜면 (`eval/results/20260915_100411_ide068_expB3_480b_tp1_cudagraph/`)

b3/b4 는 TSK_047 조건(`--disable-cuda-graph`)이었다. IDE_030 이 KT 경로에서 `--cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64` 로 graph 를 켤 수 있음을 보였으므로 같은 TP1 구성에서 켜 보았다.

| 셀 | 구성 | graph | 출력 tok/s | TTFT p50 ms | TPOT p50 ms | HBM GiB |
|---|---|---|---|---|---|---|
| b3 | TP1, expert 0 | OFF | 43.24 | 8,694 | 300.1 | 24.0 |
| b7 | TP1, expert 0 | **ON** | 43.53 | 8,785 | 297.6 | 22.3 |
| b4 | TP1, expert 16 | OFF | 46.81 | 8,014 | 274.0 | 65.7 |
| b8 | TP1, expert 16 | **ON** | 46.10 | 7,710 | 281.4 | 66.0 |

cuda graph 는 이 구성에서 **효과가 없다** (±1 %, run 간 편차 수준). 스텝이 CPU expert 구간에 묶여 있어 GPU 측 launch overhead 를 줄여도 드러나지 않는다. graph 가 효과를 내려면 IDE_030 처럼 hot expert 가 GPU 에 충분히 올라가 GPU 구간이 스텝의 주요 부분이 되어야 한다.

### B.6 B4 — GPU 3·5·6·7 장: TP 가 성립하는 조합이 있는가 (사용자 요청, `eval/results/20260915_101451_ide068_expB4_480b_tp3567/`)

추측으로 제외하지 않고 여섯 구성을 모두 부팅 시도했다. **전부 40초 안에 엔진 단언(assert)으로 종료**했다.

| 셀 | 구성 | GPU | 결과 | 엔진 오류 |
|---|---|---|---|---|
| b9 | 하이브리드 TP3 | 3 | DIED 40 s | `AssertionError: 151936 is not divisible by 3` |
| b10 | 하이브리드 TP5 | 5 | DIED 40 s | `AssertionError: 151936 is not divisible by 5` |
| b11 | 하이브리드 TP6 | 6 | DIED 40 s | `AssertionError: 151936 is not divisible by 6` |
| b12 | 하이브리드 TP7 | 7 | DIED 40 s | `AssertionError: 151936 is not divisible by 7` |
| b13 | GPU-only TP6 + EP6 | 6 | DIED 40 s | `assert num_physical_experts % ep_size == 0` (160 % 6 ≠ 0) |
| b14 | GPU-only TP7 + EP7 | 7 | DIED 40 s | `assert num_physical_experts % ep_size == 0` (160 % 7 ≠ 0) |

**원인은 모델 구조의 정수 제약이다.**

- vocab 151,936 = 2⁷ × 1187 (1187 은 소수). 임베딩·lm_head 를 TP 로 나누려면 vocab 이 TP 로 나뉘어야 하므로 **TP ∈ {1, 2, 4, 8, 16, …, 128}** 만 가능하다. 3·5·6·7 은 KV head(8) 나 attention head(96) 검사에 닿기도 전에 vocab 에서 막힌다.
- expert 160 = 2⁵ × 5. EP 는 160 의 약수여야 하므로 EP6·EP7 은 불가. EP5 는 산술상 가능하나 TP5 가 vocab 에서 막히므로 GPU-only 5장도 불가.

따라서 480B 에서 **TP 로 쓸 수 있는 GPU 장수는 1·2·4·8 뿐**이고, 3·5·6·7 은 TP 로는 성립하지 않는다. 이 결론은 실험이 아니라 산술이 정하는 것이지만, 여섯 셀의 오류 줄로 엔진이 실제로 그렇게 거부함을 확인했다.

3·5·6·7 장을 쓰는 남은 방법은 **DP(복제)** 다 — rank 마다 TP1 모델을 통째로 복제하므로 장수 제약이 없다. 단 rank 마다 CPU expert 풀이 생겨 DRAM 이 rank 수 배가 되고 CPU 스레드를 나눠 써야 한다. 이는 TP 질문의 답이 아니므로 별도 셀(B5)로 1점만 측정한다.

---

## 실험 A — 최대 모델 (TSK_051)

(다운로드 중 — 완료 후 기록)

### A.1 후보 선정

| 후보 | 크기 | 상태 |
|---|---|---|
| DeepSeek-R1-0528 (671 B) | 688.6 GB FP8 | 원본 스냅샷이 삭제되어 있음 (kt INT4 328 GB 만 잔존). 재다운로드 없이는 불가 |
| **Kimi-K2-Instruct (1 T)** | **1,029.2 GB FP8** | 공개 최대 MoE. DeepseekV3 arch, 61층, 384 routed + 1 shared experts, top-8, moe_intermediate 2048, FP8 block 128×128. 다운로드 중 |

DRAM 2 TB 기준 INT4 expert 상한은 산술상 약 3.5~4 T 파라미터이며, 그 이상의 공개 모델은 없다. 따라서 "이 서버가 지원하는 최대" 는 공개 최대 모델이 성립하는지로 답한다.

---

## 한계

- 모든 CPU 수치는 turbo OFF (2.0 GHz) 하한이다.
- 하이브리드 셀은 `--disable-cuda-graph` (TSK_047 과 동일 조건). IDE_030 은 cuda graph + hot expert + deferral 로 훨씬 높은 처리량을 얻었으므로, 본 실험의 43 tok/s 는 "최소 장수 성립" 의 조건이지 이 경로의 성능 상한이 아니다.
- 단일 run, C16 한 점. jitter 는 b1/b2/b3 의 편차(43.24~43.39, 0.3 %)로 가늠할 수 있다.
