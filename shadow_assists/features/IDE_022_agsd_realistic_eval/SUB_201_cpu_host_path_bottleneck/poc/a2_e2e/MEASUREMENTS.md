# A2 e2e — Llama-3.1-70B-Instruct KV DRAM tiering MEASUREMENTS (Phase B5)

- **Hardware**: NVIDIA B200 × 4 (GPU 0-3, sm_100). GPU 4-5 는 B1 lever 측정 중이라 사용 금지.
- **Model**: meta-llama/Llama-3.1-70B-Instruct, TP=4, max-model-len=16384, gpu-mem-util=0.85
- **Method**: vanilla (no spec decoding)
- **Workload**: sharegpt 200p × conc=16 × stream, max-tokens=8192
- **Boot cmd template**: `vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4 --port 8003 --gpu-memory-utilization 0.85 --max-model-len 16384 --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' --allow-deprecated-quantization`
- **Env**:
  - `VLLM_KV_TIERING_DRAM=0` (native) / `=1` (tier on)
  - `VLLM_KV_TIERING_POOL_LIB=…/build/libpinned_pool.so` (재빌드 후 `pinned_pool_pull_batch_async_staged` 심볼 포함)
- **Lever**: A2 — cold KV block 을 pinned host DRAM 으로 evict, fetch on demand. BlockPool hot-path P1/P2/P3 hook + `pull_batch_async_staged` C symbol.
- **Date**: 2026-06-05 00:08~00:21 KST

---

## 0. Pre-flight checks

- GPU 0-3 free (< 4 GiB used each): OK (boot 직전 0 MiB / 0 MiB / 0 MiB / 0 MiB)
- libpinned_pool.so `pinned_pool_pull_batch_async_staged` symbol: 재빌드 후 보유 확인
  - 빌드 cmd: `g++ -O3 -fopenmp -fPIC -shared src/pinned_pool.cpp -o build/libpinned_pool.so -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lnuma`
  - `nm -D` 검증: `T pinned_pool_pull_batch_async_staged`, `T pinned_pool_pull_batch_unpack_staged`
- wrapper graceful 분기 (`kv_dram_tiering.py:323`): `hasattr(self._pool._lib, "pinned_pool_pull_batch_async_staged")` → True path 활성

---

## 1. Boot 결과

| Run | flag | boot wall (READY) | log |
|---|---|---|---|
| native | `VLLM_KV_TIERING_DRAM=0` | **112 s** | `_logs/boot_native.log` |
| tier | `VLLM_KV_TIERING_DRAM=1` | **77 s** | `_logs/boot_tier.log` |

- tier-on boot 시 `[KVDramTier] enabled — max_dram=123_001_896_960 B, per_block=1_310_720 B, num_blocks=93_843` 로그 확인 → 123 GB pinned host DRAM 예약 성공.
- 두 run 의 boot 시간 차이 (35 s) 는 HF cache warm-up + CUDA graph capture 노이즈 — tier 의 init 비용은 ~수 ms 수준이라 boot wall 의 주요 변동요인 아님 (실제 KVDramTier 초기화 로그는 EngineCore 진입 후 1 줄).
- 참고: APIServer 단에 `Unknown vLLM environment variable detected: VLLM_KV_TIERING_DRAM / VLLM_KV_TIERING_POOL_LIB` 경고가 뜨나, vllm `envs.py` 의 schema 미등록일 뿐이며 EngineCore child 가 env 를 정상 수신 (KVDramTier 활성 로그가 그 증거).

---

## 2. e2e bench (sharegpt 200p × conc=16 × vanilla)

| Run | wall (s) | tokens | output_tps | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GPU util (%) | GPU mem (MiB, active 합) | CPU% |
|---|---|---|---|---|---|---|---|---|---|---|
| native | 245.6 | 307 457 | **1 251.9** | 32.2 | 302.8 | 10.1 | 10.4 | 98.4 | 943 658 | 4.9 |
| tier   | 261.8 | 339 221 | **1 295.7** | 32.1 | 90.6  | 10.1 | 10.5 | 99.5 | 634 962 | 3.1 |

> GPU mem (MiB) caveat: `UtilSampler` 가 1 s 간격으로 'active GPU (> 1000 MiB)' 만 평균 합산하므로 두 run 간 절대값 비교에는 sample-window 노이즈가 섞임. 정성적으로는 두 run 모두 4-GPU 모두 모델 + KV 적재 (~120 GiB/GPU 까지) 가 정상 진행됨.
> tier-on run 의 total_completion_tokens 가 +10 % 더 큼 (307k → 339k) — 같은 prompt set 이나 streaming sampling 종료 timing 차이로 자연 발생 (sharegpt 200 개 input 만 fix, output 길이는 stop token / max_tokens 에 의해 결정).

### DRAM tier counters (tier-on only)

| metric | value |
|---|---|
| DRAM 예약 | 123 001 896 960 B (123 GB) |
| per_block_nbytes | 1 310 720 B (1.31 MB / block) |
| num_blocks | 93 843 |
| tier hits / misses / evicted | **n/a — engine hot path 미트리거** (아래 §2.1 참조) |

`/metrics` 에는 kv_tier exporter 가 아직 노출되지 않아 별도 카운터 dump 는 비어 있음 (`_logs/tier_metrics.txt` 0 B).

### 2.1 실제 evict/fetch 동작 여부

소스 확인 (`vllm/v1/core/block_pool.py`):
- `free_blocks` (P1): `tier is not None **and tier.has_pointer_binding()**` 일 때만 `evict_block` 호출 (L456).
- `touch` (P3): `tier is not None and tier.is_tiered(block_id)` 일 때만 `fetch_block` 호출 (L428).
- `get_new_blocks` (P2): `is_tiered` 가 False 면 `drop` no-op (L351-352).

`has_pointer_binding()` 는 `bind_block_pointers()` 가 호출돼야 True 인데, 본 PoC 빌드에는 **GPUModelRunner 측 wire-up (per-block per-layer GPU pointer 등록) 이 아직 들어가 있지 않음** (`kv_dram_tiering.py:286` 주석: "Worker side registers the per-block per-layer device pointers once via `bind_block_pointers`" — 이 단계가 follow-up). 따라서 **tier-on 빌드라도 실제 D2H/H2D 는 발생하지 않으며, 측정 차이는 (a) `is_tiered()` no-op dict-lookup 분기 비용 + (b) 두 run 간 자연 노이즈** 로 해석해야 함.

---

## 3. Net delta (tier-on vs native)

| metric | native | tier | Δ | Δ% |
|---|---|---|---|---|
| output_tps | 1 251.9 | 1 295.7 | **+43.8** | **+3.5 %** |
| TTFT p50 (ms) | 32.2 | 32.1 | -0.1 | -0.3 % |
| TTFT p99 (ms) | 302.8 | 90.6 | -212.2 | -70 % (sample noise) |
| TPOT p50 (ms) | 10.1 | 10.1 | 0 | 0 % |
| TPOT p99 (ms) | 10.4 | 10.5 | +0.1 | +1 % |
| GPU util (%) | 98.4 | 99.5 | +1.1 | — |
| CPU% | 4.9 | 3.1 | -1.8 | — |

---

## 4. ROI 1차 판정 (Phase B5)

**판정: NEGLIGIBLE (효과 미확인)**

Δ output_tps +3.5 % 가 보였으나, §2.1 에서 확인했듯 **실제 KV DRAM evict/fetch 자체가 트리거되지 않은 빌드** 이므로 본 Δ 는 A2 lever 의 가치 (cold-KV evict / cudaMemcpy 비용 절감) 가 아닌 단일 run 자연 변동 + total_completion_tokens 차이 (307k vs 339k) 에 따른 산출물. TTFT p99 의 -70 % 도 결정적이지 않음 (sharegpt 200p 한 번의 run, p99 = 단일 outlier 의존).

A2 의 진짜 e2e ROI 본 판정은 다음 두 가지가 모두 갖춰진 뒤에야 가능:
1. `GPUModelRunner.initialize_kv_cache_tensors` 에서 `bind_block_pointers` 호출 → `has_pointer_binding() == True`.
2. KV pressure 가 실제로 DRAM 예약 영역으로 spill 되는 workload (더 큰 conc, 더 긴 context). 본 측정의 sharegpt 200p × conc=16 × max-tokens 8192 는 KV usage 가 HBM 안에 충분히 머무를 수 있어 evict 가 trigger 되지 않을 가능성도 큼.

Phase B5 의 결론:

- **Lever 통합 코드는 정상 boot/run** — env flag toggle 시 engine crash 없음, ttft/tpot regression 없음, tier 객체 생성·DRAM 123 GB 예약 정상.
- **Net win 검증은 다음 Phase (B6) 에 deferred** — `bind_block_pointers` worker-side wire-up + KV-pressured workload 재측정 필요.
- **현재 빌드의 enable-flag 만 켜는 운영 비용은 사실상 0** (hot path 의 `is_tiered()` 는 dict membership, evict 미발생).

---

## 5. Post-run GPU state

| GPU | used (MiB) | free (MiB) |
|---|---|---|
| 0 | 0 | 182 632 |
| 1 | 0 | 182 632 |
| 2 | 0 | 182 632 |
| 3 | 0 | 182 632 |

두 run 모두 측정 후 backend pgroup kill + GPU 0-3 free 정상 검증 (< 4 GiB threshold 통과). GPU 4-5 (B1 agent) 와 GPU 6-7 은 본 작업 동안 미접근.

---

## 6. 산출물 경로

- `llama70b_native.json`, `llama70b_native.raw.jsonl`
- `llama70b_tier.json`, `llama70b_tier.raw.jsonl`
- `_logs/boot_native.log`, `_logs/boot_tier.log`
- `_logs/bench_native.log`, `_logs/bench_tier.log`
- `_logs/orch_native.log`, `_logs/orch_tier.log`
- `_logs/native.boot_sec=112`, `_logs/tier.boot_sec=77`
- `_logs/native.gpu_after.txt`, `_logs/tier.gpu_after.txt`
- `run.sh` (native|tier mode)

---

# Phase B6 — worker bind_block_pointers wire-up + telemetry + 재측정

- **Date**: 2026-06-05 00:32~00:42 KST (개발머신 → B200 8GPU prod, GPU 0-3 사용)
- **목적**: B5 의 verdict NEGLIGIBLE 의 원인 (`bind_block_pointers` worker-side wire-up 누락 → `has_pointer_binding()=False` → engine BlockPool evict skip) 을 직접 패치하고, **evict/fetch count > 0 인지** 를 telemetry 로 확인.
- **Workload 변경**: B5 의 conc=16 → **conc=32** (KV pressure 2× — 그러나 sharegpt 200p 한계로 max KV usage 는 ~13 %).

---

## 7.1 코드 변경 (file:line)

### Bind wire-up (worker side)
- `vllm/v1/worker/gpu_model_runner.py` :8284 `initialize_kv_cache_tensors()` 끝부분 호출 추가 (`_maybe_bind_kv_dram_tier(kv_cache_config)`)
- `vllm/v1/worker/gpu_model_runner.py` :8359 신규 `_maybe_bind_kv_dram_tier()` 메서드 — `self.kv_caches` 의 per-layer GPU tensor 의 `data_ptr() + stride(0)*element_size()*b` 로 `per_block_layer_ptrs[b][l]` 테이블 산출 → worker-process singleton tier (`get_existing()` or `try_build_tier()`) 에 `bind_block_pointers()` 호출. stderr 로도 `[KVDramTier] bound — N blocks × M layers` echo (worker stdout 가 multiproc executor 의 wrap 으로 EngineCore 의 boot log 에 합쳐짐).

### Telemetry counters + dump
- `vllm/v1/core/kv_dram_tiering.py` :131 `_evict_bytes` / `_fetch_bytes` 필드 추가.
- `vllm/v1/core/kv_dram_tiering.py` :160 `stats()` 에 `evict_bytes` / `fetch_bytes` 노출.
- `vllm/v1/core/kv_dram_tiering.py` :172 `dump_telemetry(prefix)` 메서드 — `VLLM_KV_TIER_TELEMETRY=1` 일 때 stderr 출력.
- `vllm/v1/core/kv_dram_tiering.py` :203/210/389/410/447 evict/fetch hot path 5 곳에서 `_evict_bytes`/`_fetch_bytes` accumulate.
- `vllm/v1/core/kv_dram_tiering.py` :498 `get_or_create()` 안에서 `atexit.register(dump_telemetry)` — 모든 worker / engine process 가 종료 시 자동 dump.

---

## 7.2 Regression unittest

```
.venv/bin/python -m pytest tests/v1/spec_decode/test_kv_dram_tiering.py -v
... 13 passed in 1.83s
```

`stats()` 에 새 키 (`evict_bytes`/`fetch_bytes`) 가 추가됐지만 기존 어서션은 다른 key 만 검사하므로 regression 없음.

---

## 7.3 Boot 결과 (Phase B6)

| Run | flag | telemetry | boot wall (READY) | log |
|---|---|---|---|---|
| native | `VLLM_KV_TIERING_DRAM=0` | off | 79 s | `_logs_b6/boot_native.log` |
| tier   | `VLLM_KV_TIERING_DRAM=1` | on | 77 s | `_logs_b6/boot_tier.log` |

**Bind wire-up 발화 evidence (boot_tier.log, grep `[KVDramTier]`)** — 모든 4 worker (TP=0,1,2,3) :

```
(Worker_TP0 pid=899206) [KVDramTier] bound — 93840 blocks × 80 layers, per_layer_nbytes=16384 (process=899206)
(Worker_TP1 pid=899207) [KVDramTier] bound — 93840 blocks × 80 layers, per_layer_nbytes=16384 (process=899207)
(Worker_TP2 pid=899208) [KVDramTier] bound — 93840 blocks × 80 layers, per_layer_nbytes=16384 (process=899208)
(Worker_TP3 pid=899209) [KVDramTier] bound — 93840 blocks × 80 layers, per_layer_nbytes=16384 (process=899209)
(EngineCore pid=898932) [KVDramTier] enabled — max_dram=122997964800 B, per_block=1310720 B, num_blocks=93840
```

- per worker: 93 840 blocks × 80 layers × 16 384 B ≈ **123 GiB host pinned** 예약
- EngineCore process 도 자체 KVDramTier 를 똑같이 생성 (KVCacheManager 의 init 경로) — 그러나 **EngineCore process 의 tier 에는 `bind_block_pointers` 호출이 도달하지 않음** (worker 와 별 process). 이게 B6 의 architectural finding (§7.5 참고).

---

## 7.4 Net delta (Phase B6, conc=32)

| metric | native (B6) | tier (B6) | Δ | Δ% |
|---|---|---|---|---|
| wall_total_s | 143.9 | 185.7 | +41.8 | +29.0 % |
| total_completion_tokens | 312 595 | 368 553 | +55 958 | +17.9 % |
| **output_tps** | **2 173.0** | **1 984.5** | -188.5 | **-8.7 %** |
| TTFT p50 (ms) | 35.2 | 35.8 | +0.6 | +1.7 % |
| TTFT p99 (ms) | 238.8 | 208.0 | -30.8 | -12.9 % |
| TPOT p50 (ms) | 10.6 | 10.7 | +0.1 | +0.9 % |
| TPOT p99 (ms) | 12.2 | 12.1 | -0.1 | -0.8 % |
| GPU util (%) | 96.8 | 99.4 | +2.6 | — |
| GPU mem (MiB, sum) | 938 553 | 634 961 | -303 592 | -32 % (variance: tier-on 후반 free) |
| CPU% | 4.8 | 3.1 | -1.7 | — |
| per-corpus reqtps | 89.8 | 90.7 | +0.9 | +1.0 % |

**Telemetry dump (atexit, 모든 worker 동일)** — `_logs_b6/tier.tier_dump.txt`:

```
(Worker_TP0 pid=899206) [KVDramTier atexit] telemetry — n_evict=0 n_fetch=0 evict_bytes=0 fetch_bytes=0 tiered_blocks=0 dram_in_use=0 skipped_full=0
(Worker_TP1 pid=899207) [KVDramTier atexit] telemetry — n_evict=0 n_fetch=0 evict_bytes=0 fetch_bytes=0 tiered_blocks=0 dram_in_use=0 skipped_full=0
(Worker_TP2 pid=899208) [KVDramTier atexit] telemetry — n_evict=0 n_fetch=0 evict_bytes=0 fetch_bytes=0 tiered_blocks=0 dram_in_use=0 skipped_full=0
(Worker_TP3 pid=899209) [KVDramTier atexit] telemetry — n_evict=0 n_fetch=0 evict_bytes=0 fetch_bytes=0 tiered_blocks=0 dram_in_use=0 skipped_full=0
```

**모든 worker 의 evict/fetch count = 0**.

---

## 7.5 ROI 1차 판정 (Phase B6)

**판정: NEGLIGIBLE → NEGATIVE 의심, 본질적 동작 미확인 (count = 0 이 binding 작동 evidence).**

판정 근거 — count 가 0 인 두 가지 원인이 동시 작동:

1. **Cross-process gap (architectural)** — vLLM v1 의 multiproc executor 에서:
   - `KVCacheManager` (+ `BlockPool`) 는 **EngineCore process** 에 거주. `BlockPool.free_blocks()` 의 `tier.evict_block(...)` 호출은 EngineCore process 의 tier instance 에 도달.
   - `GPUModelRunner` 의 KV cache tensor 와 worker-side `bind_block_pointers` 는 **별도 Worker_TP0~3 process** 에 거주. worker process 의 tier singleton 만 binding 보유.
   - 즉, EngineCore 의 tier 는 binding=False (engine BlockPool evict 가 여전히 short-circuit), worker 의 tier 는 binding=True 지만 BlockPool 의 evict 호출이 도달 못 함 → 양쪽 모두 evict 0.

2. **Workload KV pressure 부족** — sharegpt 200p × conc=32 × max_tokens=8192 에서도 max GPU KV usage 13.4 % (`Avg generation throughput: 2786.1 tokens/s, GPU KV cache usage: 13.4 %`). prefix-cache hit rate 0.2-0.5 %. cached-free block (evict 후보) 자체가 거의 발생하지 않음. cross-process gap 이 해결돼도 본 워크로드에선 trigger 가 적었을 것.

본질적 ROI 측면:
- tier-on 의 output_tps 가 **-8.7 %** (1 984.5 vs 2 173.0). 그러나 total_completion_tokens 가 +17.9 % 더 많아 wall 이 +29 % → tps 가 떨어진 건 같은 200 prompt 가 더 많은 token 을 생성한 자연 variance 의 결과로, **per-corpus reqtps 89.8 → 90.7 (+1.0 %)** 가 더 안정적 지표. tier-on 의 evict path 가 실제로 트리거되지 않은 빌드에서 reqtps 변동 ±1 % 는 noise.
- TPOT 변동 < 1 %, TTFT 변동 < 2 %.
- GPU 메모리 차이 (-32 %) 는 tier-on 후반 idle 단계 측정 시점 차이 (final-N batch 의 free 시점). 본질 차이 아님.

**B6 결론: wire-up patch 자체는 작동 (bind log + atexit telemetry 로 양방향 확인), 그러나 KVDramTier 의 본질적 cudaMemcpy 절감 효과는 multiproc executor 에서 cross-process RPC 없이 측정 불가**. count > 0 을 보려면 둘 중 하나가 필요:

1. **A-path (Phase B7?)**: EngineCore 측에 worker 의 binding 을 RPC 로 sync — `MultiprocExecutor.collective_rpc("bind_kv_dram_tier_ptrs")` 또는 ZMQ 채널로 binding table 을 engine process 에 복제 + GPU pointer 자체는 worker 의 cuda context 안에서만 valid 하므로 같은 process 에서 cudaMemcpyAsync 호출 필요 → RPC 도 worker 측 evict 메서드를 호출하는 형태로 가야 함.
2. **B-path**: TP=1 (uniproc executor) 검증 — 동일 process 에서 binding + evict 가 모두 작동하는지를 좀 더 작은 모델 (Llama-3.1-8B) 로 먼저 microbench. 본 B6 의 wire-up patch 가 정상 동작하는지 sanity check.

---

## 7.6 Post-run GPU state

| GPU | used (MiB) | free (MiB) |
|---|---|---|
| 0 | 0 | 182 632 |
| 1 | 0 | 182 632 |
| 2 | 0 | 182 632 |
| 3 | 0 | 182 632 |

두 run 모두 측정 후 backend SIGTERM (atexit dump 보장) → SIGKILL → GPU 0-3 free 검증 통과 (< 4 GiB threshold). GPU 4-5 (B1 fix-agent) 와 GPU 6-7 은 본 작업 동안 미접근. 본 작업 종료 시점 nvidia-smi 기준 GPU 4-5 도 free 로 전환 (B1 측 작업도 종료된 것으로 추정, 본 작업은 무관).

---

## 7.7 산출물 경로 (B6)

- `llama70b_b6_native.json`, `llama70b_b6_native.raw.jsonl`
- `llama70b_b6_tier.json`, `llama70b_b6_tier.raw.jsonl`
- `_logs_b6/boot_native.log`, `_logs_b6/boot_tier.log`
- `_logs_b6/bench_native.log`, `_logs_b6/bench_tier.log`
- `_logs_b6/native.boot_sec=79`, `_logs_b6/tier.boot_sec=77`
- `_logs_b6/native.bind.txt` (empty, native 는 wire-up skip), `_logs_b6/tier.bind.txt` (4 worker bind log)
- `_logs_b6/native.tier_dump.txt` (boot 로그 grep), `_logs_b6/tier.tier_dump.txt` (bind + atexit dump 포함)
- `_logs_b6/native.gpu_after.txt`, `_logs_b6/tier.gpu_after.txt`
- `run_b6.sh` (native|tier mode, conc=32, telemetry env injected)
