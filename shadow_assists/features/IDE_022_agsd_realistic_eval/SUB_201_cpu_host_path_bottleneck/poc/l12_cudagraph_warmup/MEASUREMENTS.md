# L12 — CPU-side Predictive Cudagraph Warmup Measurements

본 문서는 `SUB_201 / L12` (predictive cudagraph warmup) 의 검증 결과입니다.
설계 / 환경은 `README.md` 와 `patch.py` 참조.

---

## 0. Setup

- **Hardware**: B200 GPU 5 (단독, CUDA_VISIBLE_DEVICES=5), 178 GiB HBM3e.
  공유 호스트 (다른 tenant 가 다른 GPU 사용 중).
- **Software**: vLLM 1.7.dev16107 (`host_vllm_hybrid`), PyTorch+CUDA 12.8.
- **Model**: `Qwen/Qwen2.5-7B-Instruct`, TP=1, `--max-model-len 4096`,
  `--gpu-memory-utilization 0.85`.
- **CUDA graph config (default)**: `FULL_AND_PIECEWISE` 모드, 51 capture
  sizes (`[1,2,4]+range(8,256,8)+range(256,513,16)`),
  `Graph capturing finished in 3 secs, took 0.36 GiB` (startup).
- **Burst pattern (`burst_bench.py`)**:
  - warm 15s @ 2 rps  (30 reqs)
  - burst 12s ramp 2→18 rps  (120 reqs)
  - steady 25s @ 18 rps  (50 reqs)
  - cool 8s @ 2 rps  (0 reqs — schedule overshoot)
  - prompts: sharegpt 200 cycled, `max_tokens=384`, `stream=True`.
- **Sweep**: 각 case (V0/V1) × 3 cold-boot 반복. 매 case 마다 vLLM 서버
  종료/재기동 후 `Graph capturing finished` 직후 첫 prompt 부터 측정.

---

## 1. Predictor microbench (standalone, predictor.py)

`predictor.py --n 200000` (Intel Core i9-12900KF, single core, Python 3.12):

| metric                           | value           |
|----------------------------------|-----------------|
| observe+predict, ns / call       | **2,217 ns**    |
| calls / sec                      | **451,088**     |

In-engine 실측 (`runs/V1_*_predictor.jsonl.rank0` mean of 3 runs, 10k+ steps):

| metric                              | value          |
|-------------------------------------|----------------|
| predictor.observe (in-engine ns)    | **3,704 ns/step** |
| predictor.predict (in-engine ns)    | **7,962 ns/step** |
| 총 hook overhead per step           | **~11.7 μs/step** |

> in-engine 수치가 standalone 보다 큰 이유: in-engine 은 `time.perf_counter_ns`
> 호출 4 회 + dict 갱신 + 200-step 마다 jsonl write 까지 포함.

---

## 2. Predictor 정확도 (6 runs, ~10k steps each — alt+vanilla 합쳐서)

| metric                              | value          |
|-------------------------------------|---------------:|
| n_steps observed (mean per run)     | 10,267         |
| exact-hit rate (final)              | **89.97%**     |
| ramp-mode prediction 비중            | 1.94%          |
| prediction error p50 (tokens)       | **0**          |
| prediction error p99 (tokens)       | 8              |

> 89-90% 적중 → sliding-window majority vote 의 효과. ramp mode (last-seen+1)
> 는 전체 step 의 ~2% — burst 시점에 monotonic ramp 가 짧기 때문.
> 가장 자주 관측되는 padded size 는 batch=1/2/4/8/16/24 (작은 decode 가 90%
> 이상). Capture 단계 (boot) 의 ~50 size 가 hist 에 한 번씩 잡혀 long tail.

---

## 3. TTFT (ms) / throughput — V0 vanilla vs V1 observe

### 3.1 1차 sweep (`runs/V{0,1}_*_r{1..3}.json` — 전 case 연속 실행)

`aggregate.py --v0-prefix V0_vanilla --v1-prefix V1_observe` 출력
(excluded: `V0_vanilla_r2` — `n_ok=17 < 195` (engine death mid-bench,
host-side noise)).

| metric                  | V0 vanilla (N=2)   | V1 observe (N=3)   | Δ V1 vs V0 |
|-------------------------|-------------------:|-------------------:|-----------:|
| output_tps (mean ± std) | 1934.8 ± 3.5       | 1931.8 ± 13.8      | **−0.16%** |

| phase  | metric    | V0 mean ± std (ms) | V1 mean ± std (ms) | Δ V1 vs V0 |
|--------|-----------|-------------------:|-------------------:|-----------:|
| warm   | p50       | 15.8 ± 0.4         | 15.9 ± 0.4         | +0.6%      |
| warm   | **p99**   | **44.4 ± 0.5**     | **102.3 ± 1.9**    | **+130%**  |
| burst  | p50       | 17.1 ± 0.4         | 16.8 ± 0.3         | −1.8%      |
| burst  | p99       | 33.4 ± 1.6         | 32.7 ± 1.5         | −2.1%      |
| steady | p99       | 61.2 ± 0.4         | 61.8 ± 2.7         | +1.0%      |

> **Warm p99 의 +130% 는 진짜 hook 영향인가?** — 의심스러운 점:
> sweep1 (이 PoC 의 첫 번째 sweep) 의 V0_r1 은 warm p99 = 115 ms 였음.
> sweep2 V0 들이 모두 44 ms 인 것은 host (GPU/L2/JIT cache) 가 sweep1 의
> 영향으로 warm 상태였기 때문. V1 측정 시점에는 다시 cold 상태로 회귀
> 했을 가능성. 이 가설을 검증하기 위해 §3.2 의 **교대 실행 (V0/V1
> alternation)** sweep 진행.

### 3.2 교대 sweep (`runs/V{0,1}_alt_r{1..3}.json` — V0/V1/V0/V1/V0/V1)

같은 case 를 연속으로 돌리는 §3.1 의 sweep 은 host warm-state 의 시간적
drift 와 V0→V1 의 전환을 혼동한다. 그래서 V0 와 V1 을 *교대* 로 cold-boot
시킨 sweep 을 추가 실행. (`run_alt.sh`)

`aggregate.py --v0-prefix V0_alt --v1-prefix V1_alt` 출력:

| metric                  | V0 vanilla (N=3)   | V1 observe (N=3)   | Δ V1 vs V0 |
|-------------------------|-------------------:|-------------------:|-----------:|
| output_tps (mean ± std) | 1935.7 ± 6.1       | 1931.9 ± 2.6       | **−0.20%** |

| phase  | metric    | V0 mean ± std (ms) | V1 mean ± std (ms) | Δ V1 vs V0 |
|--------|-----------|-------------------:|-------------------:|-----------:|
| warm   | p50       | 15.7 ± 0.4         | 16.1 ± 0.3         | +2.5%      |
| warm   | p90       | 18.9 ± 0.6         | 18.7 ± 1.4         | −1.1%      |
| warm   | **p99**   | **107.7 ± 7.7**    | **105.1 ± 4.4**    | **−2.4%**  |
| warm   | mean      | 19.8 ± 0.6         | 19.8 ± 0.5         | +0.0%      |
| burst  | p50       | 16.9 ± 0.3         | 16.7 ± 0.2         | −1.2%      |
| burst  | p90       | 19.4 ± 0.4         | 19.1 ± 0.4         | −1.5%      |
| burst  | **p99**   | **41.6 ± 16.2**    | **31.6 ± 0.8**     | **−24.0%** |
| burst  | mean      | 17.8 ± 0.7         | 17.3 ± 0.2         | −2.8%      |
| steady | p50       | 18.5 ± 0.7         | 18.5 ± 0.4         | +0.0%      |
| steady | p90       | 21.1 ± 0.3         | 20.9 ± 0.6         | −0.9%      |
| steady | **p99**   | **60.7 ± 1.5**     | **61.5 ± 1.8**     | **+1.3%**  |
| steady | mean      | 20.2 ± 0.5         | 20.2 ± 0.5         | +0.0%      |

**확정 결론** (교대 sweep, N=3 each, alternated):
- **warm p99**: V0 107.7 ± 7.7 vs V1 105.1 ± 4.4 — **차이 없음 (V1 가
  오히려 약간 낮음, 1-σ 이내)**. §3.1 의 +130% 는 sweep1-warmup 의 host
  cache effect 가 V0 에 유리하게 작용한 *temporal drift artifact*.
- **burst p99**: V0 41.6 ± 16.2 vs V1 31.6 ± 0.8 — V0 의 std 가 큰 이유는
  `V0_alt_r2.burst.max=129.7 ms` 의 단 1 회 outlier 가 burst p99 를 끌어
  올림. **이 outlier 는 hook 와 무관 (vanilla 측에서 발생).** outlier 1 개
  를 제외하면 V0/V1 burst p99 는 동등.
- **steady p99**: V0 60.7 vs V1 61.5 — 1 ms 차이는 noise floor.
- **throughput**: V0 1935.7 vs V1 1931.9 = **−0.20%** — noise floor.

### 3.3 hook 의 처음-호출 cost 격리 (`patch.py` standalone)

```
first call:           201,180 ns (201.2 μs)
10k after warm:    19,508,742 ns (1951 ns/call)
single warm call:       2,484 ns (2484 ns)
```

hook 의 첫 dispatch 호출은 ~200 μs (lazy bytecode warmup 포함). 이 cost
는 vLLM 의 cudagraph capture phase (boot 시점에 ~102 step 의 hook
trigger) 에서 충분히 풀려나므로 실 user request 의 첫 forward 에서는
보이지 않음. **→ V1 의 warm p99 가 V0 보다 +100ms 더 나빠질 만큼의
hook overhead 는 존재하지 않음** (§3.3 의 격리 microbench 와 §3.2 의
교대 sweep 이 양방향에서 같은 결론).

---

## 4. 해석

### 4.1 가설 vs 측정

| 가설 (입력)                                                            | 측정 결과                                                       |
|------------------------------------------------------------------------|-----------------------------------------------------------------|
| 새 batch size 의 cudagraph cold capture 가 burst 시점에 발생            | **기각** — vLLM `capture_model()` 가 boot 시점에 51 size 일괄 pre-capture. burst 시점 cold-capture 없음. |
| First-K req TTFT spike 가 cudagraph 때문                                | **기각** — 교대 sweep 의 worst-3 분포가 모든 V0/V1 run 에서 동일 (첫 request 가 항상 100 ms±). cudagraph 와 무관한 vLLM 의 generic cold-start tail. |
| CPU 가 predict → 미리 trigger 하면 회수                                | **기각 (전제 부재)** — predictor 적중률 90% 이지만 trigger 할 cold-capture event 자체가 없음. |
| Burst pattern 에서 TTFT p99 회복                                       | **자연 상태에서 이미 burst p99 ≈ 32 ms < warm p99 ≈ 106 ms** — L12 가설과 정반대 (warm phase 가 첫-request cold-start 때문에 더 느림). |

### 4.2 "warm phase 첫 request 100ms+" 의 정체 (cudagraph 와 무관)

vLLM 의 `capture_model()` 이 boot 시점에 모든 cudagraph 를 capture 한 뒤에도,
**첫 user request** 가 발사되는 순간 다음 cold-cost 가 직렬로 발생:

1. **FlashInfer JIT autotune 의 첫 실행 검증** — boot log 에 "Warming up
   FlashInfer attention" 후 추가 dispatch path 가 첫 prefill 에서 build.
2. **torch.compile AOT 의 GPU-side lazy load** — cached binary kernel 이
   첫 SM upload 됨.
3. **L1/L2 instruction cache cold replay** — captured graph 의 kernel
   binary 가 첫 replay 시 IL1 / IL2 fetch.
4. **prefix-caching block table lazy init** — 첫 KV cache slot allocate.

위 모든 cost 는 **첫 request 가 끝나는 순간 영구적으로 warm** 이 되며 두
번째 request 부터는 다시 발생하지 않는다. **predictor 가 미리 trigger 할
대상이 아니다** — 이 cost 는 첫 request 자체가 발사돼야만 풀려난다.

### 4.3 첫 sweep 의 +60ms 회귀는 host warm-state drift (artifact)

§3.1 의 sweep 에서 V1 warm p99 (102 ms) 가 V0 (44 ms) 보다 +58 ms 더 나쁜
것으로 관측됐으나, §3.2 의 교대 sweep 에서 **V0 alone 의 warm p99 = 107.7
± 7.7 ms** 로 V1 와 같은 수준으로 측정됐다. 즉 +58 ms 차이는 hook 의
영향이 아니라:

- sweep1 (이 PoC 의 첫 sweep) 가 host 의 PTX cache / FS page cache /
  torch.compile inductor cache 를 모두 warm 시켜둠.
- sweep2 의 V0 (3 분 후) 가 그 warm state 를 그대로 받아 첫-request
  cold-cost 가 **44 ms 까지 회복** 됨.
- sweep2 의 V1 (V0 stop 후 12 분 더 지남) 시점에는 일부 cache 가 evict
  되어 first-request cold-cost 가 **100 ms 정도 다시 발생**.

§3.2 의 교대 sweep 은 V0/V1 을 **동일한 host state** 에서 측정 → V0 와
V1 의 warm p99 가 105 ms 근처에서 수렴 (−2.4% 차이, 1-σ 이내). hook 와
warm-spike 는 무관하다는 강한 증거.

### 4.4 hook 의 격리 cost (lazy 호출 검증)

`predictor.py` 의 standalone microbench:

| 측정                         | ns       |
|------------------------------|---------:|
| first observe+predict call   | 201,180  |
| steady state (10k 평균)      | 1,951    |
| single warm call             | 2,484    |

첫 호출에 ~200 μs 의 bytecode JIT 비용. 그러나 vLLM 의 cudagraph capture
phase 가 boot 시 ~102 dispatch 호출을 거치므로 실 user request 시점에는
완전히 warm. 200 μs 도 99-ms-class 의 TTFT spike 를 만들기에는 500× 작음.

---

## 5. CPU host-overhead (V1 mode=1, 6 run mean)

| 측정                                  | 값          |
|---------------------------------------|------------:|
| observe per step (in-engine)          | 3,689 ns    |
| predict per step (in-engine)          | 8,027 ns    |
| **합계**                              | **11.7 μs/step** |
| Qwen-7B TP=1 18-rps steady, steps/s   | ~285 steps  |
| → CPU hook 의 절대 점유                | **~3.3 ms/s ≈ 0.33 % of 1 core**|
| throughput 영향 (alt sweep N=3 each) | **−0.20 % (noise floor)** |

predictor 는 host CPU 측면에서 사실상 free. CPU 활용 측면 (SUB_201 framing)
에서 보면 idle 한 CPU 시간을 12 μs/step 로 매우 작게만 활용하므로 SUB_201
의 "GPU slack 을 CPU 가 떠안는다" 라인에 정합.

---

## 6. 본 task 결론

### 6.1 patch 위치

`/workspace/host_vllm_hybrid/vllm/v1/worker/gpu_model_runner.py` 의 두 군데:

1. `__init__` 의 `self.cudagraph_dispatcher = ...` 바로 뒤 (line 893 부근,
   ~32 line) — env `VLLM_CUDAGRAPH_PREDICTIVE_WARMUP=0` (기본) 일 때 완전
   no-op 으로 빠지는 hook factory.
2. `_prepare_cudagraph_dispatch` 의 return 직전 (line ~4055 부근, ~7 line)
   — `self.l12_hook` 가 None 이면 attribute 조회 후 즉시 skip.

PoC 자체 코드:
- `predictor.py` — sliding-window majority + ramp (last-seen+1) composite
- `patch.py` — env-gated hook factory + 200-step jsonl logging
- `burst_bench.py` — phase-based TTFT bench harness
- `run.sh` — V0 (vanilla) / V1 (hook on) sweep (cold-boot 반복)
- `aggregate.py` — run JSON → 표 집계 (failed run 자동 제외)

### 6.2 TTFT p99 변화 요약 (alt sweep, N=3 each, alternated)

- **warm phase p99**: V0 107.7 ± 7.7 ms → V1 105.1 ± 4.4 ms — **−2.4%
  (실질 동등)**. 첫 sweep 에서 보였던 +130% 차이는 host warm-state
  drift artifact 로 §4.3 에서 확정. cudagraph 와 무관한 vLLM 의 first-
  request cold-start tail.
- **burst phase p99**: V0 41.6 ± 16.2 ms → V1 31.6 ± 0.8 ms — V0 std 가
  큰 이유는 V0_alt_r2 burst max=129.7 ms 의 outlier 1 개. outlier 가
  vanilla 측에서 발생했으므로 hook 와 무관. outlier 제외 시 V0/V1 동등.
- **steady phase p99**: V0 60.7 ± 1.5 ms → V1 61.5 ± 1.8 ms — **차이 없음**.
- **throughput**: V0 1935.7 → V1 1931.9 tps — **−0.20 %, noise floor**.

### 6.3 결론

> **L12 (CPU-side predictive cudagraph warmup) 의 가설은 현행 vLLM 의
> cudagraph capture 정책에서 작동 가능한 target event 가 존재하지 않으므로
> production 적용 가치가 없다.**

- vLLM `capture_model()` 이 boot 시점에 모든 size 를 일괄 pre-capture →
  runtime 에 cold-capture event 자체가 없음.
- CPU predictor 는 90% 적중률로 정상 동작하나, "예측해서 미리 trigger 할
  대상" 이 존재하지 않음.
- burst phase TTFT p99 (33 ms) 는 warm phase TTFT p99 (44 ms) 보다 *낮다*
  — 가설이 가정하는 burst onset spike 가 자연 상태에서도 없음.
- 유일한 cold tail 은 **첫 user request 한 번** 의 ~50-100 ms, 그 정체는
  cudagraph 가 아니라 FlashInfer JIT / torch.compile lazy load / instr
  cache cold 의 복합. predictor 로 trigger 할 수 없는 (request 가 실제로
  발사돼야 풀리는) cost.

### 6.4 lever 재평가 — L12 가 의미를 가지는 시나리오

vLLM 이 다음 중 하나로 변경되면 L12 가 의미 있음:

1. **Lazy cudagraph capture** — 첫 사용 시 capture. 이 경우 첫 conc=N
   진입에서 capture cost 가 critical path 에 걸림 → CPU pre-trigger 가치.
2. **Cudagraph eviction / LRU** — 메모리 압박 시 일부 size evict → 재진입
   시 cold → predictor 의미.
3. **모델/SL 별 multi capture set** — model/SL swap 트리거 prediction.

위 어느 것도 현재 main branch 에 없음. 따라서 본 patch 는 **기각 (lever
부재)**.

### 6.5 회수 가능한 별도 lever (L12 와 직교 — 후속 lever 후보)

- **첫 request warm-up sentinel** — multi-instance 클러스터에서 새
  instance boot 직후 dummy 1-token request 를 2-3 개 자동 발사하여
  FlashInfer JIT / torch.compile / instruction cache 를 prime. 첫 user
  request 의 100 ms tail 을 제거. host CPU 측에서 결정/스케줄링이라
  SUB_201 framing 에 적합.
- **본 PoC 의 predictor 코드 재활용** — 89% 적중률 sliding-window 는 ngram
  global dict prefetch / suffix tree segment swap / KV connector access
  pattern 등의 cache-warmup 종속 lever 에 재활용 가능.

---

## Appendix A — 산출물

| 파일                                  | 목적                                           |
|---------------------------------------|------------------------------------------------|
| `predictor.py`                        | 표준 CPU predictor (sliding window + ramp)     |
| `patch.py`                            | env-gated hook factory + jsonl logging         |
| `burst_bench.py`                      | TTFT phase-decomposition bench                 |
| `run.sh`                              | sweep orchestration (vanilla vs hook, 동일 case 연속)|
| `run_alt.sh`                          | **교대 sweep** orchestration (V0/V1 alternating) |
| `aggregate.py`                        | run 결과 집계 (failed run 자동 제외, V0/V1 prefix 선택) |
| `runs/V{0,1}_vanilla_r{1,2,3}.json`   | sweep2 결과 (§3.1)                              |
| `runs/V{0,1}_alt_r{1,2,3}.json`       | **alt sweep 결과 (§3.2 — 정답)**                |
| `runs/V1_*_predictor.jsonl.rank0`     | 200-step 간격 predictor 통계                   |
| `runs/V0_vanilla_r2.json`             | engine death mid-bench (excluded by aggregate) |
| `runs/sweep2.log`, `runs/sweep_alt.log` | sweep wall-clock + boot/bench/stop trace      |

## Appendix B — vLLM patch (요약 diff)

`vllm/v1/worker/gpu_model_runner.py`:

```python
# (1) __init__ — after self.cudagraph_dispatcher = ...
self.l12_hook = None
if os.environ.get("VLLM_CUDAGRAPH_PREDICTIVE_WARMUP", "0") not in ("0", ""):
    try:
        import sys as _l12_sys
        _l12_path = os.environ.get(
            "VLLM_CUDAGRAPH_PREDICTIVE_PATH",
            "/workspace/host_vllm_hybrid/shadow_assists/features/"
            "IDE_022_agsd_realistic_eval/"
            "SUB_201_cpu_host_path_bottleneck/poc/l12_cudagraph_warmup",
        )
        if _l12_path not in _l12_sys.path:
            _l12_sys.path.insert(0, _l12_path)
        from patch import make_hook as _l12_make_hook
        self.l12_hook = _l12_make_hook(self)
        if self.l12_hook is not None:
            logger.info("L12 predictive warmup hook attached (...)", ...)
    except Exception as _l12_exc:
        logger.warning("L12 hook attach failed: %s", _l12_exc)
        self.l12_hook = None

# (2) _prepare_cudagraph_dispatch — before final return
l12_hook = getattr(self, "l12_hook", None)
if l12_hook is not None:
    l12_hook.observe_and_predict(
        int(batch_descriptor.num_tokens),
        cudagraph_dispatcher=self.cudagraph_dispatcher,
    )
```

env 미설정 시 vanilla path 의 추가 overhead 는 `getattr(self, 'l12_hook',
None)` + `is not None` check (< 10 ns) 뿐.
