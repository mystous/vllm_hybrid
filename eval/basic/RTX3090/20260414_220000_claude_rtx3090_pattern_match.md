# RTX 3090 dev — H100x8 분석 패턴 대조

**작성**: 2026-04-14 22:00 (Claude)
**근거 문서**:
- `eval/basic/H100x8/20260414_213434_claude_h100x8_cpu_execution_path_and_timeframe.md`
- `eval/h100x8/20260414_213415_codex_h100x8_log_analysis.md`

**대조 대상 (RTX3090)**:
- H1: `20260414_143419_GeForce_RTX_3090_x1_Qwen2.5-1.5B-Instruct` (hybrid AFFINITY fix, max_seqs=4, threads=16)
- H7: `20260414_150240_GeForce_RTX_3090_x1_Qwen2.5-7B-Instruct` (hybrid AFFINITY fix, max_seqs=4, threads=16)

---

## 요약

H100x8 문서에서 정리한 **9개 핵심 패턴** 중 **7개가 RTX3090 에서도 동일하게 재현**. 2개는 환경 차이 (1 NUMA / AVX-512 없음) 로 비적용. 추가로 **codex 의 "threads 덮어쓰기" 이슈는 RTX 에서 값이 우연히 일치해 관찰되지 않음**. 또한 RTX 데이터가 H100 문서 §5 의 미확정 가설 하나를 해소.

---

## 패턴 매칭 표

| # | 패턴 | H100x8 | RTX3090 H1 | RTX3090 H7 |
|---|---|---|---|---|
| P1 | Wave-batch lifecycle (close → drain → reset) | ✓ | ✓ | ✓ |
| P2 | Cold-start gate (probe → GPU) | ✓ | ✓ | ✓ |
| P3 | GPU bulk 수초 후 CPU tail 이 wall 결정 | ✓ | ✓ | ✓ |
| P4 | `in_flight_cpu=N/M` 표기 (aggregate/per-engine) | ✓ (2/1, 32/16) | ✓ (4/4) | ✓ (4/4) |
| P5 | IPEX `single_query_cached_kv_attention` decode 경로 | ✓ | ✓ (코드 공통) | ✓ |
| P6 | `chunked_prefill=False` 강제 → varlen 경로 미사용 | ✓ | ✓ | ✓ |
| P7 | TPOT bimodal (GPU 빠른 cluster + CPU 느린 tail) | ✓ (22 vs 15,955 ms) | ✓ (32 vs ~100 ms) | ✓ (40 vs 460 ms) |
| P8 | Multi-NUMA alternating (`_find_wave_open_cpu` strict `<`) | ✓ (2 engines) | — (1 NUMA → 1 engine) | — |
| P9 | AMX BF16 dispatch (ONEDNN_ISA=AVX512_CORE_AMX) | ✓ | ✗ (ISA not set) | ✗ |
| N1 | C++ `init_cpu_threads_env` 가 env threads 덮어씀 | ✓ (32→56) | (값 우연 일치로 미관찰) | (동일) |

---

## P1. Wave-batch lifecycle — 동일

**RTX H1 (1.5B) boot + routing timeline:**
```
14:34:33  [HYBRID-RESOLVE] max_seqs=4 threads=16 numa_nodes=1
14:34:37  [HYBRID-CPU-ENV] OMP=16 ONEDNN_ISA=not set sched_affinity_count=24
14:34:38  [HYBRID-CPU-WORKER] local_omp_cpuid='0,2,4,6,8,10,12,14,16,17,18,19,20,21,22,23'
          (16 physical primaries — P-core 0-14 짝수 + E-core 16-23)
14:34:38  post-init: torch_threads=16 cpu_affinity=1 cores [0]
14:35:05  [HYBRID-ROUTER-INIT] strategy=wave-batch priority=cpu-first
          cpu_max_num_seqs=4 num_cpu_engines=1
14:35:07  [HYBRID-WAVE] engine=0 wave closed (accepted=4, batch_size=4)
14:35:30  [HYBRID-WAVE] engine=0 wave drained (accepted=4) → reset
         ↑ 23초 후 drain — hybrid.json duration 23.06s 와 일치
```

**RTX H7 (7B):**
```
15:02:53  [HYBRID-RESOLVE] max_seqs=4 threads=16 kvcache=16GB
15:02:57  [HYBRID-CPU-ENV] OMP=16
15:02:58  [HYBRID-CPU-WORKER] local_omp_cpuid='0,2,...,23'
15:04:24  [HYBRID-ROUTER-INIT] (91s boot — 7B 모델 로드 시간)
15:04:28  [HYBRID-WAVE] engine=0 wave closed (accepted=4)
15:05:58  [HYBRID-WAVE] engine=0 wave drained (accepted=4)
         ↑ 90초 후 drain — hybrid.json duration 89.56s 와 일치
```

**H100x8 H1 (044922)** 와 비교:
- 구조 동일 (close → drain → reset)
- 차이: engine 수 (RTX 1 vs H100 2), batch_size (RTX 4 vs H100 1), tail 시간

---

## P2. Cold-start gate — 동일

`_route_wave_batch` 의 cold-start 분기 (`hybrid_core.py:486`) 양쪽 환경에서 동일 작동.

- **H1 routing**: probe → GPU, main burst 의 **첫 4 req → engine 0 CPU** (accepted 1,2,3,4 → closed), req 5~500 → GPU
- **H7 routing**: probe → GPU, main burst 의 **첫 4 req → CPU**, 나머지 96 → GPU

---

## P3. GPU bulk 후 CPU tail 이 wall 결정 — 동일

**RTX H1 stats 발췌 (14:35:11 ~ 14:35:13):**
```
14:35:11  finished=501 GPU=29.3 tok/s (497 reqs) CPU=0.0 tok/s (4 reqs) in_flight_gpu=447
14:35:12  finished=501 GPU=28.5 tok/s in_flight_gpu=397
14:35:12  finished=501 GPU=27.7 tok/s in_flight_gpu=347
14:35:12  finished=501 GPU=26.5 tok/s in_flight_gpu=297
14:35:12  finished=501 GPU=25.6 tok/s in_flight_gpu=247
```
2 초 만에 GPU in_flight 447→247. 이후 수 초 내 0. Wall 은 CPU 4 req 가 14:35:30 까지 완료되는 것으로 결정.

**RTX H7 stats 발췌 (7B, 100 req):**
```
15:04:34  finished=101 GPU=20.1 tok/s (97 reqs) in_flight_gpu=72
15:04:35  finished=101 GPU=20.1 tok/s in_flight_gpu=47
15:04:35  finished=101 GPU=20.1 tok/s in_flight_gpu=22
15:05:58  finished=101 GPU=20.0 tok/s CPU=1.2 tok/s in_flight_gpu=0  ← CPU 완료 시점
```
GPU 수초 내 완료, 이후 **80+ 초 동안 CPU 4 req 대기**. `wall = T_cpu_tail = 90s`.

**H100x8 H1 (044922)** 와 동일한 시간 구조. RTX 는 GPU 가 dev 에서 상대적으로 느려 `T_gpu_bulk` 가 좀 더 크지만 여전히 CPU tail 이 wall 지배.

---

## P4. `in_flight_cpu=N/M` 표기 — 동일

- RTX H1/H7: `in_flight_cpu=4/4` (aggregate=4 / per-engine max=4). num_cpu_engines=1 이라 aggregate=per-engine
- H100 H1 (044922): `2/1` → aggregate=2 (engine 0,1 각 1) / per-engine max=1
- H100 H2 (045947): `32/16` → aggregate=32 (engine 0,1 각 16) / per-engine max=16

**overflow 아님** — 분자/분모가 서로 다른 scope 라는 codex 지적이 RTX 에도 그대로 적용.

---

## P5. IPEX decode 경로 — 동일 (코드 공통)

`cpu_attn.py` 의 `_get_paged_attn_impl()` 은 IPEX 설치 여부만 보고 `_IPEXPagedAttention` 선택. RTX dev 에 IPEX 2.8.0 설치돼 있어 H100 과 동일 분기.

단, **AMX 없음** → IPEX 가 내부적으로 AVX2+VNNI BF16 경로로 fallback. dispatcher 는 동일 `single_query_cached_kv_attention` 호출하지만 kernel 내부 ISA 감지 후 slower path. RTX H7 mean_tpot 61ms, p99 460ms 가 H100 H1 7B (mean 37, p99 83) 대비 느린 이유 중 하나.

---

## P6. `chunked_prefill=False` — 동일

`_create_cpu_vllm_config` 가 동일하게 `enable_chunked_prefill=False` 강제. RTX 에서도 IPEX `flash_attn_varlen_func` 미사용.

**증거**: H7 TTFT p99 = **36,755 ms** (4 CPU req 의 prefill 큐 대기). max_seqs=4 이므로 4번째 CPU req 는 앞 3개 prefill 완료 후 차례를 받음. per-prefill ~9s × 4 ≈ 36s ✓

H100 H1 (044922) TTFT p99 = 1,075 ms 는 2 CPU req 만 있었기 때문에 prefill 큐 대기 짧음. RTX H7 은 4 CPU req + 7B 로 큐 대기 길어짐. **chunked_prefill 활성화 시 TTFT p99 개선 가능성** — codex 11.5 에서 제기한 후속 실험 후보.

---

## P7. TPOT bimodal — 동일

| Run | median | mean | p99 | bimodal 비율 |
|---|---:|---:|---:|---|
| H100 H1 044922 (500 req, 2 CPU) | 22.7 | 37 | 83 | 2 / 500 = 0.4% |
| H100 H2 045947 (500 req, 32 CPU) | 21.6 | 1047 | 15,966 | 32 / 500 = 6.4% |
| H100 H3 054010 (500 req, 2 CPU) | 25.5 | 38 | 79 | 2 / 500 = 0.4% |
| **RTX H1 (500 req, 4 CPU)** | 32.1 | 30.4 | 102 | 4 / 500 = 0.8% |
| **RTX H7 (100 req, 4 CPU)** | 40.2 | 61.0 | 460 | 4 / 100 = 4.0% |

- RTX H1 p99/median = 102/32 = 3.2× tail
- RTX H7 p99/median = 460/40 = 11.5× tail (7B 가 CPU 에서 더 느려져 bimodal 격차 심화)
- H100 H2 p99/median = 15,966/22 = 723× tail (max_seqs=16 batch penalty 극단)

**패턴 공통**: CPU 가 GPU 보다 느릴수록 (모델 크기 또는 batch 증가) p99/median 비율 증가.

---

## P8. Multi-NUMA alternating — RTX 비적용

RTX 는 1 NUMA → num_cpu_engines=1 → `_find_wave_open_cpu` 가 항상 engine 0 만 선택. H100 의 alternating 패턴 (`best_accepted < X` strict `<`) 은 RTX 에선 의미 없음.

**시사점**: H100 의 2-NUMA 분산 검증은 RTX 로 불가. RTX 는 single-NUMA hybrid 의 baseline 역할.

---

## P9. AMX BF16 dispatch — RTX 비적용

- H100 boot log: `ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX` (configure_intel_optimizations 가 AMX 감지 후 setdefault)
- RTX boot log: `ONEDNN_ISA=not set` — i9-12900KF 는 AMX 없고 AVX-512 도 없음. IPEX 가 AVX2+VNNI 경로로 자동 fallback

---

## N1. C++ threads 덮어쓰기 — RTX 에서는 발생 안 함

H100 에서 codex 가 짚은 핵심 발견:
> env `HYBRID_CPU_THREADS=32` 설정해도 C++ `init_cpu_threads_env` 가 `omp_set_num_threads(len(local_omp_cpuid))` 로 56 으로 덮어씀.

**RTX 에서는**: env `HYBRID_CPU_THREADS=16` 과 `len(local_omp_cpuid)=16` (16 physical cores) 이 **우연히 일치**. 덮어쓰기는 여전히 발생하지만 값이 같아 관찰되지 않음.

RTX H1/H7 boot log:
```
[HYBRID-CPU-ENV] PID=... OMP=16 ...              ← env 기준
[HYBRID-CPU-WORKER] post-init: torch_threads=16  ← C++ 덮어쓴 값 (여전히 16)
```

**함의**: C++ 덮어쓰기 동작 자체는 공통이므로 RTX 에서 env 값을 바꿔도 결과가 바뀌지 않을 가능성. 예를 들어 `HYBRID_CPU_THREADS=8` 로 세팅해도 autobind 가 16 physical cores 반환하면 최종 threads=16 유지.

**검증 방법**: RTX 에서 `HYBRID_CPU_THREADS=8` 로 재실험 → boot log 에서 `OMP=8` 이지만 `post-init torch_threads=16` 이 나오는지. 만약 그렇다면 codex 지적이 RTX 에도 적용. 현재 dev 환경에서 이 실험은 미수행.

---

## 추가 발견 — RTX 가 H100 batch=1 의 "비정상적으로 큰 per-step" 가설 해소

H100x8 claude 본문 §5 의 미확정 가설:
> CPU per-step batch=1 이 이론 예측 대비 10× 큰 이유 (3079 ms vs 예측 ~200 ms)

RTX 데이터로 부분 검증:
- **H100 H1 7B batch=1**: per-step ≈ 3079 ms (engine 1: 394s/128 tokens)
- **RTX H7 7B batch=4**: per-step ≈ 703 ms (wall 90s / 128 tokens)
- **RTX H7 per-token CPU TPOT**: mean_tpot 역산 ≈ 565 ms/token

RTX CPU 는 AVX2+VNNI only (AMX 없음) — **이론상 H100 AMX 경로보다 5-10× 느려야 함**. 그런데 실측 RTX batch=4 가 H100 batch=1 보다 **4× 빠름** (703 vs 3079 ms/step). 원인:

1. **RTX batch=4 → overhead amortization**: ZMQ/Python/kernel launch 상수 overhead 가 4 tokens 에 분산
2. **H100 batch=1 → overhead 지배 영역**: batch 1 이라 constant overhead 가 per-step 의 대부분 차지
3. AMX 경로 compute 속도만 비교하면 H100 이 당연히 이겨야 하지만, **overhead 지배 구간** 에서 역전 가능

**시사점 (H100 §5 가설 부분 해소)**:
- H100 batch=1 의 3079ms 는 attention/MLP compute 가 아닌 **per-step constant overhead** 가 지배
- 이론 예측 "batch=1 에서 CPU ~200ms/step" 은 compute-only 예측으로 **per-step overhead 를 무시한 것**
- batch 가 작을수록 compute 는 적지만 overhead 는 그대로 → per-step 이 **오히려 증가 가능**
- **wave-batch max_seqs=1 이 max_seqs=16 의 1/16 이 아닌 1/5 배만 빠른 이유** — overhead 는 batch 와 무관하므로

**다음 실험 제안**: H100 에서 max_seqs=2, 4, 8 실측 → batch 증가에 따른 per-step 곡선 knee point 확인. RTX 는 이미 batch=4 지점 데이터 보유.

---

## 본문 수정 권고 (H100 문서 §5/§6)

H100 claude 본문 §5 "batch 크기에 따른 per-step 증가율" 섹션:
- 실측 batch=1 3079ms, batch=16 16390ms, ratio 5.32×
- 단순 모델 예측 (compute-linear) 대비 batch=1 이 10× 큼을 "미확정 가설" 로 표기
- **RTX 데이터로 보강**: batch=1 의 큰 값은 compute 가 아닌 **per-step constant overhead** (ZMQ, Python, kernel launch, sampler) 가 원인. 이 해석으로 §5 미확정 가설 3종 중 하나 해소

---

## 결론

**패턴 9개 중 7개 재현** (P1-P7), **2개는 환경 차이로 비적용** (P8 multi-NUMA, P9 AMX), **1개는 H100 고유 이슈가 RTX 에서 conveniently 회피** (N1 threads overwrite 값 일치).

RTX3090 dev 실험의 가치:
1. **H100 batch=1 per-step 비정상 크기** 를 RTX batch=4 데이터로 **overhead 지배** 로 설명 가능
2. **dev 1 NUMA + affinity fix 이후 16 physical core 정상 활용** → H100x8 분석 결과 (2 NUMA × 56 core) 와 토폴로지만 다르고 메커니즘은 동일
3. **hybrid/gpu_only ratio**: RTX 1.5B 2.85×, RTX 7B 13.7× vs H100 7B (044922) 28.2× (394/14). RTX 가 상대적으로 "덜 망하는" 이유는 RTX 3090 GPU 자체가 H100×4 보다 느려 wall 격차가 작기 때문

RTX3090 은 H100x8 분석의 **단순화된 재현 환경** 으로 작동. dev 에서 빠른 검증 → H100 배포의 CI 흐름이 타당.
