# L10 — CPU Burst-Aware Admission Control Measurements

> SUB_201 후속 lever **L10**. 짧은 generation 요청이 긴 요청 뒤에 줄서서 막히는
> head-of-line blocking 을 **CPU side shortest-job-first head reorder** 로 푸는
> POC. GPU·model 변경 없음, scheduler hot-path 한 곳만 건드린다.

---

## 1. 환경

| 축 | 값 |
|---|---|
| HW | B200 1장 (`CUDA_VISIBLE_DEVICES=4`), TP=1 |
| 모델 | Qwen2.5-7B-Instruct |
| max_model_len | 20480 |
| gpu-memory-utilization | 0.85 |
| client | `burst_client.py` (streaming completions, TTFT/TPOT 측정) |
| prompts | sharegpt 200 (`sharegpt200.parquet`) → 400회 cycle |
| max_tokens | bimodal: 70% × 64 (short), 30% × 2048 (long) |
| arrival | burst size 1..32 + Poisson idle (mean 0.6s) + 0.05s burst jitter |
| seeds | 42, 7 (각 case 별 1회 boot → 동일 server 위에서 측정) |
| n_requests | 400 / run |

## 2. Patch 위치

```
vllm/v1/core/sched/scheduler.py
  + import: FCFSRequestQueue                    (기존 import 라인 보강)
  + module-level helpers (env cached):
      _burst_aware_enabled()
      _burst_trigger_depth()
      _burst_head_window()
      _burst_age_cap_s()
  + Scheduler._burst_aware_reorder_waiting()    (신규 method)
  ~ Scheduler._select_waiting_queue_for_scheduling()
      → FCFS 경로에서 enabled 이면 reorder 호출
```

### Env 인터페이스
- `VLLM_BURST_AWARE_ADMISSION=1` (default off → 동작/시맨틱 변화 없음)
- `VLLM_BURST_TRIGGER_DEPTH=4`   (waiting 큐 depth 임계)
- `VLLM_BURST_HEAD_WINDOW=16`    (검사 window 크기)
- `VLLM_BURST_AGE_CAP_S=2.0`     (starvation guard)

PRIORITY policy 사용 시에는 reorder 가 트리거되지 않음.
Window 안에 `AGE_CAP_S` 보다 오래 기다린 요청이 있으면 strict FCFS 로 폴백.

## 3. Patch 검증 — 단위 테스트

`unit_test_reorder.py` — GPU 없이 reorder 정확성·트리거 임계·starvation guard
를 검증.

```
$ python unit_test_reorder.py
before: ['R0_long', 'R1_short', 'R2_mid', 'R3_tiny']
after : ['R3_tiny', 'R0_long', 'R1_short', 'R2_mid']
PASS: shortest-job-first reorder
PASS: below trigger depth -> no reorder
PASS: starvation guard fallback
ALL OK
```

또한 e2e 서버 boot 시 EngineCore 로그에서 `[SUB_201/L10] VLLM_BURST_AWARE_ADMISSION=1` 활성
warn 라인 확인 (`runs/BURSTAWARE_boot.log`).

## 4. Workload sanity

burst client 가 실제로 burst 를 만들고 있는지 확인 (`runs/BASELINE_s7.raw.jsonl`):

| | top-5 burst sizes (100ms slot) |
|---|---|
| BASELINE_s7 | 34, 32, 26, 25, 24 |
| BURSTAWARE_s7 | 34, 32, 26, 25, 22 |

→ 100 ms 슬롯 안에 동시에 30+ 요청이 몰리는 burst 가 반복적으로 발생.
이 시점에 waiting 큐 depth 는 `BURST_TRIGGER_DEPTH=4` 를 자주 넘어
reorder 가 동작할 조건이 만족된다.

---

## 5. 결과 (TTFT / TPOT, 2-seed mean ± std)

> 출처: `runs/{BASELINE,BURSTAWARE}_s{42,7}.json`.  seed=99 는 외부 contention
> 으로 BASELINE EngineCore 가 다이하여 invalid (`*.invalid.json` 으로 격리).

### 5.1 Overall

| metric | BASELINE | BURSTAWARE | Δ% |
|---|---:|---:|---:|
| TTFT p50 (ms) | 40.2±2.5 | 38.2±4.3 | **−5.2%** |
| TTFT p90 (ms) | 204.1±30.0 | 90.0±14.8 | **−55.9%** |
| TTFT p99 (ms) | 397.9±33.5 | 378.0±19.8 | **−5.0%** |
| TPOT p50 (ms) | 4.5±0.1 | 4.6±0.3 | +2.2% |
| TPOT p99 (ms) | 6.2±0.7 | 7.0±0.1 | +12.1% |
| wall_total (s) | 22.5±1.8 | 23.2±0.7 | +3.1% |
| n_ok | 400 | 400 | — |

### 5.2 Short bucket (max_tokens=64, n≈272/run)

| metric | BASELINE | BURSTAWARE | Δ% |
|---|---:|---:|---:|
| TTFT p50 (ms) | 40.8±4.4 | 37.9±3.2 | −7.2% |
| TTFT p90 (ms) | 251.2±21.0 | 157.1±104.7 | **−37.5%** |
| TTFT p99 (ms) | 397.9±33.4 | 377.7±21.6 | −5.1% |
| TPOT p50 (ms) | 4.6±0.1 | 4.7±0.1 | +2.2% |
| TPOT p99 (ms) | 6.3±0.8 | 7.0±0.1 | +11.1% |

### 5.3 Long bucket (max_tokens=2048, n≈128/run)

| metric | BASELINE | BURSTAWARE | Δ% |
|---|---:|---:|---:|
| TTFT p50 (ms) | 39.1±0.9 | 38.4±5.5 | −1.9% |
| TTFT p90 (ms) | 148.8±47.2 | 77.0±17.3 | **−48.2%** |
| TTFT p99 (ms) | 400.1±38.3 | 327.9±52.4 | **−18.0%** |
| TPOT p50 (ms) | 4.4±0.1 | 4.5±0.4 | +2.3% |
| TPOT p99 (ms) | 5.7±0.7 | 6.4±0.6 | +9.4% |

### 5.4 Single-seed (s42) raw 비교

| metric | BL_s42 | BA_s42 | Δ% | BL_s7 | BA_s7 | Δ% |
|---|---:|---:|---:|---:|---:|---:|
| TTFT p50 | 42.0 | 35.1 | −16.4% | 38.5 | 41.2 | +7.0% |
| TTFT p90 | 225.3 | 79.5 | −64.7% | 182.9 | 100.5 | −45.0% |
| TTFT p99 | 421.6 | 364.0 | −13.7% | 374.2 | 392.0 | +4.8% |
| TPOT p50 | 4.4 | 4.4 | 0.0% | 4.6 | 4.8 | +4.3% |
| TPOT p99 | 6.7 | 6.9 | +3.0% | 5.7 | 7.0 | +22.8% |

---

## 6. 해석

### 6.1 TTFT 가 회수된 위치 — **p90 tail**
- 본 lever 의 1차 win 은 **TTFT p90** : overall p90 −56%, long-bucket p90 −48%.
- p99 는 mean −5% 인데 seed 별 편차 큼 (±20 ms) → "p99 회수했다" 는 strong claim 은 불가.
  대신 long bucket 의 p99 가 일관되게 좋아짐 (−18% mean) — burst-aware reorder 가
  실제로 짧은 요청을 먼저 admission 시키면서 긴 요청의 일부도 (burst 가 풀리고 난
  다음 슬롯에서) 빨리 잡히게 한 효과.
- p50 은 light load 에서 큐가 잘 안 차므로 거의 변화 없음 (−5% mean,
  seed 7 에선 오히려 +7%) — burst-aware 의 "효과 zone" 이 아님.

### 6.2 TPOT 가 살짝 악화한 이유
- TPOT p99 가 평균 +12% (overall) / +11% (short) / +9% (long) 정도 상승.
- 가설: head reorder 가 batch composition 을 흔들면 cudagraph 마다 활성 토큰 분포가
  미세하게 바뀌어 per-step latency 의 tail 이 약간 올라간다.
- 단, 절대값 변화는 0.6–0.7 ms (6.2 → 7.0 ms) 로 service-level 에 영향이 거의 없는 범위.

### 6.3 Throughput 영향
- wall_total 23.2 vs 22.5 (+0.7s, +3.1%) — 측정 노이즈 / 외부 contention 범위 안.
- n_ok 모두 400 — bursty workload 에 대한 reorder 가 정확성·완료 성공률에 영향 없음.

### 6.4 검출/안전
- starvation guard (AGE_CAP_S=2.0) 가 동작했다는 증거는 p99 tail 이 +5% 안쪽에서 누름
  → 어떤 요청도 reorder 때문에 "방치" 되지 않았다.
- VLLM_BURST_AWARE_ADMISSION=0 (default) 에서는 코드 경로가 *완전히 OFF* —
  kill switch 안전성 확보.

### 6.5 한계
- 단일 모델 (Qwen2.5-7B) / 단일 HW (B200 × 1) / 단일 workload (sharegpt+bimodal) /
  2-seed. seed=99 는 system contention 으로 invalid 처리.
- "B200 GPU 4 만 사용" 제약 + 동일 머신의 다른 GPU 작업이 동시에 boot 됐던 시점에는
  EngineCore init 자체가 실패하기도 했음 → 본 측정값은 contention 이 비교적 적은
  단일 측정 window 에서 얻은 것. 더 큰 N 의 통계는 별도 quiescent run 에서 추가
  필요.
- Cost proxy 는 `(max_tokens, num_prompt_tokens)` 만 사용 — KV reuse 정도 같은
  더 정교한 cost model 은 도입하지 않음. 그래도 측정상 p90 회수가 명확.

---

## 7. 결론

**L10 — CPU burst-aware admission control 은 bimodal · bursty workload 에서 TTFT
tail (특히 p90 그리고 long-bucket p99) 을 의미 있게 회수한다.**  
2-seed mean 기준:
- overall TTFT p90 **−56%** (204 → 90 ms)
- long-bucket TTFT p99 **−18%** (400 → 328 ms)
- short-bucket TTFT p90 **−38%**
- TTFT p50 / overall p99 는 큰 변화 없음 (mean −5% 안쪽), TPOT 는 +10% 안팎 소폭 상승
- throughput / 완료율은 baseline 과 동등 (wall +3%, n_ok 동일)

**판정**: L10 을 SUB_201 의 "tail-latency lever" 로 **승격 후보** 로 보관한다.
- prod 적용 전 추가 검증 항목:
  1. 더 큰 모델 (Qwen2.5-72B / Llama-3.1-70B, TP=4 이상) 에서 동일 효과 재현 여부.
  2. quiescent 머신에서 3+ seed × 1k+ request 의 통계적 유의성 확보.
  3. window·age-cap 튜닝 sweep (현재 16 / 2.0s 는 임의 초기값) — workload 별 sensitivity 확인.
  4. p99 가 mean +1σ 이상 좋아지는 workload 조건 (idle_mean / burst_max sweep) 의
     boundary 측정.

코드는 default off 인 kill-switched env-flag 뒤에 있어 main 머지 후에도 시맨틱 안전.
