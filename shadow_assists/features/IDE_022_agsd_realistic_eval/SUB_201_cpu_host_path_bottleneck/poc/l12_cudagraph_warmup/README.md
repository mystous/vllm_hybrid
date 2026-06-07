# L12 — CPU-side predictive cudagraph warmup

본 PoC 는 SUB_201 의 후속 lever 로, **CPU 가 다음 forward step 의 cudagraph
batch size 를 예측해 GPU 의 cudagraph cold capture (또는 cache cold replay)
를 미리 trigger** 하면 TTFT p99 가 회복되는지 검증한다.

## 가설 (입력)

> "새 batch size 의 cudagraph cold capture 가 늦음 — first-K req TTFT spike.
> CPU 가 다음 가능한 batch size 예측 → 미리 trigger 해서 warm cache 유지."

## 코드 읽기로 본 vLLM 의 실제 cudagraph life-cycle

`vllm/v1/worker/gpu_model_runner.py::capture_model()` 는 **서버 부팅 시점에
모든 batch size 의 cudagraph 를 한번에 capture** 한다. 기본 capture set 은
`[1, 2, 4] + range(8, 256, 8) + range(256, 512+1, 16)` = **51 개 size**
(Qwen2.5-7B 의 경우).  Runtime 에서 `_prepare_cudagraph_dispatch()` →
`CudagraphDispatcher.dispatch()` 는 *padded* batch size 를 보고 사전에
캡쳐된 graph 의 dict lookup 만 수행한다.

즉, **runtime 에 "cold capture" 는 발생하지 않는다.** 모든 (FULL, PIECEWISE)
graph 는 startup `Graph capturing finished in 3 secs` 시점에 이미 device
memory 에 올라가 있다.

이 사실은 가설 자체를 약화시키지만, 다음 두 가지 잔존 가능성은 측정으로
확인할 가치가 있다:

1. **GPU instruction/data cache cold replay** — 한 동안 사용되지 않은
   batch size 의 cudagraph 노드 (kernel 코드 + descriptor) 가 L2 에서
   evict 되어, burst 시점 첫 replay 가 느려진다. 미리 일종의 *warm-up*
   replay 를 trigger 하면 회수 가능?
2. **첫 request cold-start tail** (FlashInfer JIT, FA autotuner,
   torch.compile cache miss 등) — 이 부분은 cudagraph 와 무관하지만, 본
   bench 가 잡아내는 "warm phase p99" 의 진짜 원인일 수 있다.

## PoC 구성

- ``predictor.py`` — sliding-window mode + last-seen+1 ramp 의 composite
  predictor (pure CPU, microbench: ~2 μs / observe+predict).
- ``patch.py`` — env-gated hook (``VLLM_CUDAGRAPH_PREDICTIVE_WARMUP=1``)
  를 ``GPUModelRunner`` 에 attach 하여 매 forward step 에서
  observe+predict + jsonl 로깅.
- ``vllm/v1/worker/gpu_model_runner.py`` 에 ~40 line 의 in-place 패치
  (init + dispatch hook 호출). env 미설정 시 no-op.
- ``burst_bench.py`` — phase-based burst pattern (warm → burst-ramp →
  steady → cool) 으로 TTFT p50/p90/p99/max 를 phase 별로 분해.
- ``run.sh`` — Qwen2.5-7B TP=1 / GPU 5 / port 8112 sweep
  (`V0_vanilla` vs `V1_observe`).
- ``aggregate.py`` — 결과 집계 (mean ± std, V0 vs V1 Δ%, predictor stats).

## 측정 결과 → ``MEASUREMENTS.md`` 참조

## 본 PoC 가 *하지 않는* 것

- 실제 cudagraph replay 를 *side stream* 에서 미리 trigger 하지 않는다.
  `_dummy_run` 은 persistent buffer (input_ids, positions, …) 를 덮어쓰기
  때문에 정상 forward 와 race 하면 정합성이 깨진다. side-stream replay
  를 안전하게 하려면 (a) graph 별 입력 버퍼 copy, (b) 별도 stream 동기화
  가 필요해 patch 가 커진다 — 그 비용은 측정된 cold-replay 의 jitter 가
  유의한 경우에만 정당화된다. **본 PoC 는 그 전제 조건을 먼저 검증한다.**
- AMX/AVX-512 등 CPU SIMD 와 무관 — pure dispatcher-level Python.

## 결론

`MEASUREMENTS.md` 6절.
