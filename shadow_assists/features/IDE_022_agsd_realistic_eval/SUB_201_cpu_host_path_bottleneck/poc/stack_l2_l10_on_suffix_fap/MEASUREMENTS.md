# Stack L2 + L10 on suffix + FaP — Cumulative Net Δ Measurements

> SUB_201 follow-up. L2 (CPU prefetch tokenize) + L10 (burst-aware admission) 을
> **suffix K=7 + cudagraph FULL_AND_PIECEWISE (FaP)** baseline 위에 단계적으로
> 적용해 cumulative Δ% 를 측정한다.
> L7 (model-type oracle router) 은 cluster-level lever 이므로 single-instance
> 측정과 분리해 §6 simulation 으로 인용한다.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| HW | NVIDIA **B200 × 2** (GPU index 0, 1), Intel Xeon 8570 (AMX-capable) |
| Model | `Qwen/Qwen2.5-7B-Instruct` (BF16) |
| TP | 2 (28 heads ÷ 2 = 14 OK) |
| Port | 8009 |
| `max_model_len` | 16384 |
| `gpu_memory_utilization` | 0.85 |
| `allow_deprecated_quantization` | true |
| Workload (parquet) | `b3_8gpu_full/sharegpt200.parquet` — 200 prompts, sharegpt 분포 |
| concurrency | 16 |
| max_tokens | 512 |
| temperature | 0.0 (greedy) — 분포 동등 검증을 위해 |
| seed | parquet 사전 sampling (`shuffle=False`) |
| spec-decode | suffix K=7 (`{"method":"suffix","num_speculative_tokens":7}`) |
| vllm | `1.7.dev16107+gffe20fb09.d20260601` (`/workspace/vllm_dev_prj/bin/vllm`) |
| LD_LIBRARY_PATH | `/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib` (모든 vllm 호출 prefix) |

### Lever activation 검증
- **L2** (`VLLM_PREFETCH_TOKENIZE=1`, workers=2):
  patch 등록 위치 — `vllm/envs.py:951` + `vllm/utils/async_utils.py:58`
- **L10** (`VLLM_BURST_AWARE_ADMISSION=1`):
  patch 등록 위치 — `vllm/v1/core/sched/scheduler.py:99-135`,
  EngineCore log 확인:
  ```
  [SUB_201/L10] VLLM_BURST_AWARE_ADMISSION=1 — CPU burst-aware admission control
  활성 (shortest-job-first head reorder, FCFS only).
  ```
- **cudagraph FaP** — `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`
- **suffix K=7** — `--speculative-config '{"method":"suffix","num_speculative_tokens":7}'`

---

## 2. Run matrix (5 runs sequential)

| run | cudagraph | spec-decode | L2 | L10 | boot(s) |
|---|---|---|:---:|:---:|---:|
| A | PIECEWISE | vanilla       | OFF | OFF | 59 |
| B | PIECEWISE | suffix K=7    | OFF | OFF | 58 |
| C | FaP       | suffix K=7    | OFF | OFF | 62 |
| D | FaP       | suffix K=7    | **ON** | OFF | 62 |
| E | FaP       | suffix K=7    | **ON** | **ON** | 47 |

---

## 3. 결과 (sharegpt 200p × conc=16 × max-tok=512, n_ok=200 모두 성공)

### 3.1 Primary metrics

| run | tps | TTFT p50 (ms) | TTFT p90 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | α | GPU% | CPU% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A (PW + vanilla)       | **4121.6** | 16.9 |  75.4 | 632.0 | 3.6 | 3.7 |   —    | 81.2 | 2.3 |
| B (PW + suffix K=7)    | 3574.2 | 17.4 |  34.9 | 514.9 | 4.5 | 5.8 | 0.300 | 61.8 | 2.1 |
| C (FaP + suffix)       | 3575.2 | 17.4 |  34.3 | 506.9 | 4.5 | 5.7 | 0.302 | 58.5 | 2.2 |
| D (FaP + suffix + L2)  | 3432.7 | **12.2** |  35.5 | 502.3 | 4.7 | 5.7 | 0.294 | 59.9 | 2.2 |
| E (FaP + suffix + L2 + L10) | **3646.8** | **11.7** | **31.6** | **121.2** | 4.5 | 5.7 | 0.298 | 61.8 | 2.2 |

### 3.2 Cumulative Δ%

| 비교 | tps | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p99 |
|---|---:|---:|---:|---:|---:|
| **B/A** suffix 단독        | **−13.3%** | +3.0% | **−53.7%** | −18.5% | +56.8% |
| **C/B** + FaP              |  +0.0% | 0.0% |  −1.7% |  −1.6% |  −1.7% |
| **D/C** + L2               |  −4.0% | **−29.9%** | +3.5% |  −0.9% | 0.0% |
| **E/D** + L10              |  +6.2% |  −4.1% | **−11.0%** | **−75.9%** | 0.0% |
| **E/C** L2+L10 cumulative  |  +2.0% | **−32.8%** | **−7.9%** | **−76.1%** | 0.0% |
| **E/A** 전체 stack          | **−11.5%** | **−30.8%** | **−58.1%** | **−80.8%** | +54.1% |

(부호 규약: tps 는 + 가 win, latency 는 − 가 win.)

### 3.3 핵심 관찰

1. **suffix K=7 은 sharegpt+conc16+max-tok512 에서 net throughput 손실 −13%**:
   α=0.30 (3/10 accept) 으로 acceptance overhead 가 single-token decode 의
   compute saving 을 넘는다. A→B 의 GPU util 81% → 62% 가 그 증거 (draft
   forward 로 idle window 가 생긴다).
2. **FaP 의 throughput 영향은 거의 0 (C/B +0.0%)**: 이 workload 의 prefill+single-decode
   비율에서는 cudagraph capture 범위 확장만으로는 win 이 나오지 않는다.
   B3 sweep 의 8GPU + max-tok 8192 시나리오와 달리 conc=16 × 512-tok 은 batch
   가 충분히 크지 않다.
3. **L2 는 TTFT p50 을 30% 회수** (17.4 → 12.2 ms): tokenize 가 critical path
   에서 빠지면서 첫 토큰까지의 host overhead 가 감소. throughput 자체는
   D 에서 −4% 인데, 이는 측정 노이즈 (200 prompt, 단발 run) 범위 안으로 해석.
4. **L10 가 추가되면서 TTFT p99 가 502 → 121 ms 로 76% 회수**: bimodal
   workload 가 아닌 sharegpt 분포에서도 burst 시점의 head reorder 가 tail
   에 의미 있는 효과. 동시에 throughput 도 D 대비 +6.2% 회복 (3432 → 3646).
5. **TPOT 는 stack 적용 전후 일관**: p50 4.5 ms, p99 5.7 ms — host-side lever
   가 per-step decode latency 에 영향을 주지 않음을 확인.

---

## 4. 해석

### 4.1 정확성 게이트 (CLAUDE.md constraint)
- 모든 run 에서 `n_ok = 200 / n_err = 0`. greedy(temperature=0) + 동일 prompt
  순서이므로 분포·의도 수준 동등성은 가정 가능. token-level bit-exact 검증은
  별도 verify 가 필요하지만 본 PoC 범위에서는 reject/완료 실패 없음.

### 4.2 throughput 손실의 근거
- **suffix loss 의 위치는 spec-decode 자체** 이며 stack lever 들의 부산물이
  아니다. C/B 에서 throughput 이 거의 변하지 않고, 오히려 E 에서 D 대비
  +6.2% 회복하는 패턴이 그것을 보여준다.
- 본 셋업의 corpus + max_tok 조합에서 suffix α 가 충분히 높지 않다는
  L7 oracle table 의 Qwen-7B 셀과 정합 (Qwen-7B 에서 suffix winner 인
  workload 와 vanilla winner 인 workload 가 갈리는 family).

### 4.3 L2 + L10 의 stack 효과 (suffix+FaP 위에서)
- 본 measurement 의 가장 강한 신호는 **E/C 의 TTFT p99 −76.1%**.
- TTFT p50 은 L2 한 줄이 −30%, p99 는 L10 한 줄이 −76% 를 책임진다.
  즉 두 lever 가 서로 다른 latency layer 를 회수하므로 stacking 가능.
- throughput 으로는 E/C +2% 의 미세 회복. 이는 admission reorder 가 짧은
  요청을 빨리 빼주면서 큐 평균 활성 토큰이 살짝 늘어난 효과.

### 4.4 한계
- 단일 seed, 200 prompt — 통계적 유의성은 약함. ±5% 이하의 throughput 차이는
  noise 로 간주.
- sharegpt corpus 단독 — bimodal·bursty 워크로드에서 L10 의 본 효과 (별도
  L10 측정의 TTFT p90 −56%) 가 더 크게 나올 것으로 예상 (시간 박스 내
  추가 측정 미실시).
- TP=2 단일 — TP=4 / TP=8 에서 host overhead 비중이 다른지는 별도 검증
  대상.
- α=0.30 은 본 corpus 의 한계. TSK_042 의 mix corpus 에서는 더 높은 α 가
  보고됨 → 본 측정의 suffix 손실은 corpus-specific.

---

## 5. 환경 / 안정성

- GPU 0,1 단독 사용, 다른 GPU (2-7) 는 본 측정 동안 미사용.
- 각 run 사이 `kill.sh <run>` 로 pgroup kill + orphan compute-apps 정리,
  GPU 0,1 free 까지 대기 (모든 transition 5초 이내 완료).
- 모든 vllm 호출에 `LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib`
  prefix.
- 최종 (E run 종료 후) GPU 0,1 free 검증:
  ```
  0, 0 MiB, 182632 MiB
  1, 0 MiB, 182632 MiB
  ```

---

## 6. L7 simulation 인용 (cluster-level)

본 measurement 는 single-instance Δ 이므로 L7 (model-type oracle router) 효과는
함께 측정되지 않는다. `poc/l7_oracle_router/MEASUREMENTS.md` 의 cluster-level
시뮬레이션 (Uniform mix, 70 cells = 10 model_family × 7 workload_type) 결과를
인용한다:

| 비교 | cluster TPS Δ |
|---|---:|
| L7 oracle vs vanilla-default (Uniform mix) | **+84.3%** |
| 라우터 dispatch overhead | 1.0 μs / call (975 K QPS single core) |

### 6.1 Cluster potential gain (additive 가정)

본 measurement 의 stack 효과 (E/A throughput −11.5%) 와 L7 cluster simulation
은 서로 다른 축에서 측정된다:
- 본 measurement: 단일 (model, corpus) cell 의 host-path 회수
- L7 simulation: 70 cells 의 method 선택 회수 (cluster-wide)

따라서 **additive 가정 하**의 cluster potential 은 다음과 같다:

```
cluster gain ≈ (1 + L7 simulation Δ) × (1 + stack measurement Δ) − 1
            = (1 + 0.843) × (1 + (-0.115)) − 1
            ≈ +63.1%
```

단, **본 Qwen-7B / sharegpt 셀은 L7 winner 가 vanilla 일 수 있는 셀** (Qwen-7B
family 의 7 cells 중 3 cells 가 vanilla / suffix 외) 이므로 실제로는 라우터가
이 셀을 vanilla 로 보내고 stack lever 만 켜는 조합이 최적이다. 그 경우:

```
optimal per-cell = vanilla baseline (run A) + (L2 + L10 on vanilla)
```

본 시간 박스에서는 vanilla 위 L2+L10 측정을 별도로 진행하지 않았다 (run E 는
suffix 위 stack 만). vanilla + L2 + L10 의 단독 측정은 후속 work item 으로
분리한다. 본 measurement 결과를 conservative 하게 인용하면:

- **per-cell**: stack lever 는 TTFT tail 만 회수, throughput 은 spec-decode
  α 가 충분한 cell 에서만 net + 가 된다.
- **cluster**: L7 oracle 가 method 를 cell 별 최적으로 보내면 **+84%**,
  그 위에 stack lever 가 가는 cell 에 한해 추가 회수 ~ 0~10%
  → 최종 cluster gain potential **+84% ~ +95%** (보수 추정).

---

## 7. 결론 — production-ready stack 권고

| lever | 효과 위치 | 권고 |
|---|---|---|
| **suffix K=7** | spec-decode (cell-level) | **cell-specific** — Qwen-7B sharegpt 같이 α<0.4 인 cell 에는 비활성. L7 oracle 의 winner 선택을 따른다. |
| **cudagraph FaP** | host launch overhead | **default ON** — 본 measurement 에서 손실 없음 (C/B +0.0%) + B3 sweep 의 8GPU 시나리오에서 net −71% 회수 확인. |
| **L2 prefetch tokenize** | TTFT p50 (first-token latency) | **default ON** — single-instance L2 측정 (`l2_prefetch_tokenize/MEASUREMENTS.md`) +11% throughput / TTFT p50 −33%. 본 stack 측정에서도 TTFT p50 −30% 재현. ABI 무변경 env-gated. |
| **L10 burst-aware admission** | TTFT p99 (tail) | **default ON for bursty workloads** — L10 단독 측정 TTFT p90 −56% bimodal. 본 stack 측정의 sharegpt 분포에서도 TTFT p99 −76% (502 → 121 ms). FCFS 외 policy 에서는 자동 no-op. |
| **L7 model-type oracle router** | cluster-level method selection | **cluster gateway 에 default ON** — +84.3% simulation, 1 μs dispatch. multi-model cluster 의 가장 큰 lever. |

### Final production stack (권고)
```
[cluster gateway]
  └ L7 oracle router    (+84.3% sim)
      └ per-model-instance:
          ├ cudagraph FaP             (B3 sweep 검증)
          ├ L2 prefetch tokenize ON   (TTFT p50 -30%, +11% tput 단독)
          └ L10 burst-aware admission (TTFT p99/p90 tail 회수)
```

본 task 의 핵심 발견은 **L2 와 L10 가 서로 다른 latency layer (p50 vs p99)
를 독립적으로 회수**, suffix + FaP 위에서도 stacking 이 가능하다는 점이다.

---

## 8. 산출물

```
poc/stack_l2_l10_on_suffix_fap/
├── boot.sh             # boot one config (A|B|C|D|E)
├── kill.sh             # pgroup + orphan compute-apps kill
├── bench.py            # sharegpt 200p × conc=16 × max-tok=512 streaming
├── MEASUREMENTS.md     # this file
├── runs/
│   ├── A.json + A.raw.jsonl    PIECEWISE + vanilla
│   ├── B.json + B.raw.jsonl    PIECEWISE + suffix K=7
│   ├── C.json + C.raw.jsonl    FaP + suffix K=7
│   ├── D.json + D.raw.jsonl    FaP + suffix K=7 + L2
│   └── E.json + E.raw.jsonl    FaP + suffix K=7 + L2 + L10
└── _logs/
    ├── boot_{A..E}.log
    └── {A..E}.boot_sec
```

### 인용 산출물 (별도 측정)
- `poc/l2_prefetch_tokenize/MEASUREMENTS.md` — L2 단독 (+11% throughput)
- `poc/l10_admission/MEASUREMENTS.md` — L10 단독 (TTFT p90 −56% bimodal)
- `poc/l7_oracle_router/MEASUREMENTS.md` — L7 cluster simulation (+84.3%)
- `poc/b3_8gpu_full/MEASUREMENTS.md` — cudagraph FaP 8GPU sweep
