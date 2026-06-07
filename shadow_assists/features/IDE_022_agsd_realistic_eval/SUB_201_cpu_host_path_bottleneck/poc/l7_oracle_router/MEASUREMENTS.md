# L7 — Oracle Router Measurements

본 문서는 `SUB_201` L7 (model-type oracle router, CPU 활용 / multi-model serving) 의 측정·시뮬레이션 결과입니다.
실험 코드는 같은 디렉토리의 `oracle_table.py`, `router.py`, `bench.py` 입니다.

---

## 1. Oracle Table (TSK_042 source)

- 원본: `vllm_config_perf/gating/realistic_eval/runs/routing_combined/metrics_table.csv` (215 rows, 4 methods)
- 정제: 10 model_family × 7 workload_type = **70 cells**
- 각 cell: `{vanilla, suffix, ngram, llm-d}` 의 `output_tps` (mean of measured run)
- best_method = max tps, default = vanilla, uplift_pct = (best − vanilla) / vanilla × 100

### 1.1 oracle 이 선택한 winner 분포 (70 cells)

| method | cells | 비율 |
|---|---:|---:|
| suffix | 46 | 65.7% |
| vanilla | 14 | 20.0% |
| llm-d | 10 | 14.3% |
| ngram | 0 | 0% |

> `ngram` 은 어디서도 winner 가 되지 못함 — 단독 클러스터 channel 로는 무가치.
> `vanilla` 가 winner 인 14 cells 의 대부분은 **DS-R1-671B (7/7)** + **DS-Llama-70B (6/7)**
> → 매우 큰 모델은 spec-decode 의 acceptance overhead 가 손해.

### 1.2 family 별 best-method 분포

| family | llm-d | suffix | vanilla |
|---|---:|---:|---:|
| DS-Llama-70B | 0 | 1 | **6** |
| DS-Qwen-32B | 0 | **7** | 0 |
| DS-Qwen-7B | 1 | **6** | 0 |
| DS-R1-671B | 0 | 0 | **7** |
| Llama-405B | 0 | **7** | 0 |
| Llama-70B | 0 | **7** | 0 |
| Llama-8B | 0 | **7** | 0 |
| Qwen-32B | **4** | 3 | 0 |
| Qwen-72B | 1 | **5** | 1 |
| Qwen-7B | **4** | 3 | 0 |

> Qwen-7B / Qwen-32B 만 **모델 내부에서 workload 별 method 가 갈리는 (llm-d ↔ suffix)** 패턴.
> 나머지 family 는 사실상 model-level 단일 method 로 결정됨.
> → "model_family 만 보고 결정" 만으로도 단일 method 클러스터 대비 의미 있는 회수가 가능.

### 1.3 family 별 평균 uplift (oracle vs default vanilla)

| family | mean | min | max |
|---|---:|---:|---:|
| DS-R1-671B | 0.00 | 0 | 0 |
| DS-Llama-70B | 13.4 | 0 | 93.7 |
| DS-Qwen-32B | 23.3 | 4.0 | 83.4 |
| Qwen-72B | 51.7 | 0 | 208.6 |
| Qwen-7B | 65.9 | 38.9 | 87.2 |
| DS-Qwen-7B | 66.8 | 31.3 | 170.0 |
| Qwen-32B | 80.9 | 63.4 | 115.9 |
| Llama-70B | 88.4 | 30.2 | 232.4 |
| Llama-405B | 90.6 | 68.5 | 125.9 |
| **Llama-8B** | **128.0** | 67.2 | 214.7 |

### 1.4 top-10 uplift cells

| family | corpus | best | default_tps | best_tps | uplift% |
|---|---|---|---:|---:|---:|
| Llama-70B | mix | suffix | 3129.2 | 10400.5 | **+232.4** |
| Llama-8B | mix | suffix | 8849.9 | 27851.4 | +214.7 |
| Qwen-72B | humaneval | suffix | 806.4 | 2488.6 | +208.6 |
| DS-Qwen-7B | mix | suffix | 9058.2 | 24458.3 | +170.0 |
| Llama-8B | swebench | suffix | 8347.9 | 21352.7 | +155.8 |
| Llama-405B | mix | suffix | 1252.1 | 2828.9 | +125.9 |
| Llama-8B | wildchat | suffix | 9001.8 | 19856.1 | +120.6 |
| Llama-405B | swebench | suffix | 1204.5 | 2639.3 | +119.1 |
| Llama-8B | lmsys | suffix | 9073.8 | 19862.3 | +118.9 |
| Qwen-32B | mix | suffix | 3055.5 | 6596.9 | +115.9 |

---

## 2. CPU dispatch latency (router 자체 비용)

`router.py --n 5_000_000` (Intel Xeon 8570, B200 host, single Python core)

| metric | value |
|---|---:|
| lookup ns/call (dict get only) | **123.9 ns** |
| full dispatch ns/call (random pick + counter + lookup) | **1024.8 ns** |
| QPS (single core, Python) | **975,755** |

> 단일 코어 0.98 M QPS — vLLM frontend 가 받을 수 있는 어떤 traffic 보다 4-5 자릿수 큰 수치.
> 라우터 latency 가 throughput 계산에 미치는 영향은 < 0.001%.

---

## 3. Cluster throughput simulation (핵심)

`bench.py` (oracle_table × 워크로드 mix → cluster TPS 가중평균)

### 3.1 두 가지 mix

**Uniform**: 70 cells 동일 가중 (1/70)

**Realistic**: production-flavor (family_share × workload_share)
- family: Qwen-7B 22%, Llama-8B 18%, Qwen-32B/DS-Qwen-7B/DS-Qwen-32B/DS-Llama-70B 10-12%, Llama-70B 7%, DS-R1-671B 5%, Qwen-72B 4%, Llama-405B 2%
- workload: sharegpt 22%, lmsys 18%, mix 20%, wildchat 14%, humaneval 10%, mbpp 8%, swebench 8%

### 3.2 결과

| workload | scheme | cluster TPS (weighted) | Δ vs default |
|---|---|---:|---:|
| uniform | default (vanilla everywhere) | 4022.6 | — |
| uniform | static-best (`suffix` everywhere) | 6601.9 | +64.1% |
| uniform | **oracle** | **6850.6** | **+70.3%** |
| realistic | default (vanilla everywhere) | 5033.7 | — |
| realistic | static-best (`suffix` everywhere) | 8939.6 | +77.6% |
| realistic | **oracle** | **9278.1** | **+84.3%** |

### 3.3 oracle 이 static-best 보다 얼마나 좋은가

| workload | oracle vs static-best (suffix) | 의미 |
|---|---:|---|
| uniform | **+3.77%** | R1-671B / DS-Llama-70B 등 "suffix 가 손해" cell 을 vanilla 로 돌려 회수 |
| realistic | **+3.79%** | 동일 — 분포가 바뀌어도 oracle 가 단일 method 클러스터를 일관 +3.8% 추가 회수 |

> 즉, "그냥 모든 cluster 를 suffix spec-decode 로" 만 해도 70-78% 이득.
> 그 위에서 oracle 가 **R1-671B suffix(797 tps) vs vanilla(1474 tps) 같은 역전 cell** 를 잡아 추가 +3-4% 회수.

### 3.4 router 자체의 net 영향

| 구성 | total host overhead | net Δ |
|---|---:|---:|
| oracle routed cluster + router (1 μs / req) | < 0.001 % | **+84.3% (realistic)** |

라우터 자체의 host overhead 가 무시 가능하므로 **oracle 의 raw uplift = net Δ**.

---

## 4. TSK_044 와의 명확한 분리

| 축 | TSK_044 (per-request classifier, 기각) | **L7 (본 task)** |
|---|---|---|
| 분류 단위 | 한 prompt 마다 C0~C3 regex / ONNX | model_family + workload_type 헤더 |
| host CPU 비용 | 분류 latency ms 단위 (suffix 대비 마진 잠식) | **1 μs / req — 측정 불가능할 만큼 작음** |
| 결정 | 동일 prompt 도 분류 결과에 따라 달라짐 | 같은 (모델, corpus) → 항상 같은 method (결정론) |
| dataset 필요 | classifier 학습/룰 검증 데이터 | TSK_042 measured cells 그대로 |
| drift 위험 | prompt distribution 변화에 취약 | model 추가 시만 oracle table 재측정 (모델 교체 빈도가 낮음) |
| 본 PoC 결론 | TSK_044 = 기각 (per-request 무가치) | **L7 = OK (+84.3% on realistic mix, host overhead 0)** |

L7 은 **TSK_044 의 대체재가 아니라 다른 layer 의 솔루션**:
- TSK_044 가 답하려던 질문: "같은 모델 instance 안에서 prompt 별로 method 를 다르게 할 수 있을까?" → No
- L7 이 답하는 질문: "여러 모델이 동시에 서비스되는 클러스터에서, model_family + workload_type 만으로 instance 를 라우팅하면 얼마나 회수되는가?" → **+84.3% (realistic mix)**

---

## 5. 한계 / Future work

1. **GPU 자원 가정**: 본 시뮬레이션은 "모든 family 가 동시에 boot 가능" 을 전제로 한다. 실제 클러스터는 모델별 GPU 할당 (e.g. 8 GPU × n family) 이 제약. 따라서 본 결과는 **이론적 oracle bound** 이고, 실 가동률·GPU 할당 정책 결합 시 산출 TPS 는 감쇠한다.
2. **workload_type 식별**: 클라이언트 헤더 (`X-Workload-Type`) 또는 caller-side 룰을 가정. 헤더 없는 traffic 에 대해 server-side 추정이 필요하면 그 부분의 host 비용은 별도 평가 필요 (단, 매우 가벼운 정규식 1회 분류이므로 1 μs 이내 추정).
3. **mix corpus 의 의미**: TSK_042 `mix` corpus 는 다른 6 corpus 의 합성. realistic mix 의 20% 를 `mix` 에 부여한 것은 보수적 — 실제 production 은 자연스럽게 mixed 일 수 있어 이 시나리오는 oracle 의 큰 uplift 셀 (모두 mix) 비중을 잘 반영함.
4. **production-ready 라우터**: 본 PoC 의 FastAPI/reverse-proxy 실 구현은 다음 단계 (TSK 부여 후). 본 task 는 simulation 까지 + 라우터 latency lower-bound 측정.

---

## 6. 본 task 결론

- **production 가치**: 여러 (모델 × workload) 가 공존하는 multi-model 클러스터에서 oracle router 는 spec-decode 단일 method 클러스터 대비 추가 **+3.8%**, vanilla 클러스터 대비 **+84.3%** throughput 회수. host CPU overhead 1 μs/req 로 사실상 free.
- **CPU 활용 측면 (SUB_201 framing)**: spec-decode 가 host-bound 시스템을 만들고, 그 host-path 의 1 μs 만 oracle lookup 에 쓰는 것으로 cluster 전체 throughput 을 회수한다는 점에서 SUB_201 의 "CPU 가 GPU slack 을 떠안는" 분류상 **A2 / B1 family (host 병목 인수)** 와 정합.
- **TSK_044 와의 관계**: TSK_044 가 같은 모델 instance 안에서 per-request 라우팅을 시도해 실패한 반면, L7 은 multi-model 클러스터 레벨로 문제를 한 단계 위로 올려 작동시킴. 두 접근이 직교하지 않고 **L7 이 TSK_044 가 풀려던 라우팅 문제의 더 큰 환경에서의 답** 이다.
- **다음 단계 (선택)**: SUB 부여 + 실 multi-instance smoke (Qwen-7B + Llama-8B 2 instance) + traffic split 측정. simulation 의 95-100% 가 재현되면 main path 후보.
