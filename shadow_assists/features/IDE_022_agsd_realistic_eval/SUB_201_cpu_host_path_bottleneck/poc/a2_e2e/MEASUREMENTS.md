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
