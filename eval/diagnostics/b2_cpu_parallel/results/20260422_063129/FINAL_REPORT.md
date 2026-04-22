# B2 CPU Parallelism 검증 통합 보고서

- 실행 시각 (KST): 20260422_063129
- 결과 디렉토리: `eval/diagnostics/b2_cpu_parallel/results/20260422_063129`
- (재생성 시각: 2026-04-22 15:51:53 KST)

## 1. Phase 1 — 정적 dispatch 분석

파일: [`phase1/dispatch_static.txt`](phase1/dispatch_static.txt)

```
=== Section A — _trace_decode_path 호출 site (각 path 의 gating) ==============
  [custom_avx]  called at line 986
  [sdpa_loop]  called at line 997
  [sdpa_batched]  called at line 1077
  L 1056 |                     if num_kv_heads != num_heads:
  L 1064 |                     for t in range(tokens_for_seq):
  L 1075 |                 return
  L 1077 |             _trace_decode_path("sdpa_batched", num_seqs, num_tokens)
  [ipex]  called at line 1266
  L 1266 |         _trace_decode_path("ipex", context_lens.shape[0], query.shape[0])
=== Section B — IPEX entry point 언급 =======================================
  L 1204 | class _IPEXPagedAttention(_PagedAttention):
  L 1274 |         if not hasattr(_IPEXPagedAttention, '_decode_call_count'):
  L 1275 |             _IPEXPagedAttention._decode_call_count = 0
  L 1276 |             _IPEXPagedAttention._decode_total_ms = 0.0
  L 1277 |             _IPEXPagedAttention._decode_batch_histogram = {}
  L 1278 |         _IPEXPagedAttention._decode_call_count += 1
  L 1279 |         _IPEXPagedAttention._decode_total_ms += _elapsed * 1000
  L 1283 |         hist = _IPEXPagedAttention._decode_batch_histogram
  L 1291 |                          and _IPEXPagedAttention._decode_call_count
  L 1294 |             avg_ms = (_IPEXPagedAttention._decode_total_ms
  L 1295 |                       / _IPEXPagedAttention._decode_call_count)
  L 1299 |                 _IPEXPagedAttention._decode_call_count,
  L 1301 |         elif _IPEXPagedAttention._decode_call_count <= 5 or \
  L 1302 |                 _IPEXPagedAttention._decode_call_count % 100 == 0:
  L 1305 |                 _IPEXPagedAttention._decode_call_count,
  L 1312 |         return _IPEXPagedAttention
  L 1004 |                 if num_tokens < num_seqs:
```

## 2. Phase 3 — Live introspection


### Engine PID 1162225

- Flame graph: [`phase3/engine_1162225_flame.svg`](phase3/engine_1162225_flame.svg) (브라우저에서 열기)
- Info: [`phase3/engine_1162225_info.txt`](phase3/engine_1162225_info.txt)

#### Process 요약
```
### PID 1162225 — VLLM::CPU_Engin
threads: 318
cpus   : 0

### OMP/BLAS 라이브러리 로드 상태 (duplication 체크)
/vllm_dev_prj/lib/python3.12/site-packages/opencv_python_headless.libs/libopenblasp-r0-59ffcd50.3.15.so
/vllm_dev_prj/lib/python3.12/site-packages/torch/lib/libgomp.so.1

### Thread 이름 분포
    173 python
    139 VLLM::CPU_Engin
      2 ZMQbg/Reaper
      2 ZMQbg/IO/0
      1 pt_tcpstore_uv
      1 cuda00002000009

### ps -L top-10 by %CPU
1162225 Rl+    0 47.2 VLLM::CPU_Engin
1163330 Sl+   43 10.8 VLLM::CPU_Engin
1163309 Sl+   22 10.4 VLLM::CPU_Engin
1163296 Sl+    9 10.3 VLLM::CPU_Engin
1163333 Sl+   46 10.0 VLLM::CPU_Engin
1163327 Sl+   40 10.0 VLLM::CPU_Engin
1163322 Sl+   35 10.0 VLLM::CPU_Engin
1163288 Sl+    1  9.7 VLLM::CPU_Engin
1163334 Sl+   47  9.6 VLLM::CPU_Engin
1163329 Sl+   42  9.6 VLLM::CPU_Engin

```

#### Top hot functions (60 초 flame graph 샘플 상위 20)
```
  948  decorate_context (torch/utils/_contextlib.py:120) (948 samples, 31.94%)
  670  execute_model (vllm/v1/worker/cpu_worker.py:718) (670 samples, 22.57%)
  576  schedule (vllm/v1/core/sched/scheduler.py:380) (576 samples, 19.41%)
  411  get_computed_blocks (vllm/v1/core/kv_cache_manager.py:181) (411 samples, 13.85%)
  407  decorate_context (torch/utils/_contextlib.py:120) (407 samples, 13.71%)
  383  execute_model (vllm/v1/worker/gpu_model_runner.py:1483) (383 samples, 12.90%)
  371  find_longest_cache_hit (vllm/v1/core/kv_cache_coordinator.py:231) (371 samples, 12.50%)
  248  decorate_context (torch/utils/_contextlib.py:119) (248 samples, 8.36%)
  238  decorate_context (torch/utils/_contextlib.py:119) (238 samples, 8.02%)
  151  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:275) (151 samples, 5.09%)
  124  _update_states (vllm/v1/worker/gpu_model_runner.py:612) (124 samples, 4.18%)
  112  schedule (vllm/v1/core/sched/scheduler.py:545) (112 samples, 3.77%)
  107  run_busy_loop (vllm/v1/engine/core.py:703) (107 samples, 3.61%)
  106  step (vllm/v1/engine/core.py:276) (106 samples, 3.57%)
  103  get_cached_block (vllm/v1/core/block_pool.py:89) (103 samples, 3.47%)
   82  schedule (vllm/v1/core/sched/scheduler.py:552) (82 samples, 2.76%)
   81  _update_states (vllm/v1/worker/gpu_model_runner.py:610) (81 samples, 2.73%)
   73  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:268) (73 samples, 2.46%)
   69  schedule (vllm/v1/core/sched/scheduler.py:337) (69 samples, 2.32%)
   68  clone (torch/autograd/grad_mode.py:294) (68 samples, 2.29%)
```


### Engine PID 1162226

- Flame graph: [`phase3/engine_1162226_flame.svg`](phase3/engine_1162226_flame.svg) (브라우저에서 열기)
- Info: [`phase3/engine_1162226_info.txt`](phase3/engine_1162226_info.txt)

#### Process 요약
```
### PID 1162226 — VLLM::CPU_Engin
threads: 318
cpus   : 56

### OMP/BLAS 라이브러리 로드 상태 (duplication 체크)
/vllm_dev_prj/lib/python3.12/site-packages/opencv_python_headless.libs/libopenblasp-r0-59ffcd50.3.15.so
/vllm_dev_prj/lib/python3.12/site-packages/torch/lib/libgomp.so.1

### Thread 이름 분포
    173 python
    139 VLLM::CPU_Engin
      2 ZMQbg/Reaper
      2 ZMQbg/IO/0
      1 pt_tcpstore_uv
      1 cuda00002000009

### ps -L top-10 by %CPU
1162226 Rl+   56 45.0 VLLM::CPU_Engin
1163129 Sl+   57 10.9 VLLM::CPU_Engin
1163134 Sl+   62 10.8 VLLM::CPU_Engin
1163130 Sl+   58 10.7 VLLM::CPU_Engin
1163166 Sl+   94 10.4 VLLM::CPU_Engin
1163131 Sl+   59  9.3 VLLM::CPU_Engin
1163183 Sl+  111  9.1 VLLM::CPU_Engin
1163176 Sl+  104  9.1 VLLM::CPU_Engin
1163175 Sl+  103  9.1 VLLM::CPU_Engin
1163174 Sl+  102  9.1 VLLM::CPU_Engin

```

#### Top hot functions (60 초 flame graph 샘플 상위 20)
```
  900  decorate_context (torch/utils/_contextlib.py:120) (900 samples, 30.84%)
  625  execute_model (vllm/v1/worker/cpu_worker.py:718) (625 samples, 21.42%)
  555  schedule (vllm/v1/core/sched/scheduler.py:380) (555 samples, 19.02%)
  397  get_computed_blocks (vllm/v1/core/kv_cache_manager.py:181) (397 samples, 13.61%)
  368  decorate_context (torch/utils/_contextlib.py:120) (368 samples, 12.61%)
  353  find_longest_cache_hit (vllm/v1/core/kv_cache_coordinator.py:231) (353 samples, 12.10%)
  342  execute_model (vllm/v1/worker/gpu_model_runner.py:1483) (342 samples, 11.72%)
  259  decorate_context (torch/utils/_contextlib.py:119) (259 samples, 8.88%)
  236  decorate_context (torch/utils/_contextlib.py:119) (236 samples, 8.09%)
  150  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:275) (150 samples, 5.14%)
  123  run_busy_loop (vllm/v1/engine/core.py:703) (123 samples, 4.22%)
  116  get_cached_block (vllm/v1/core/block_pool.py:89) (116 samples, 3.98%)
  114  schedule (vllm/v1/core/sched/scheduler.py:545) (114 samples, 3.91%)
  110  _update_states (vllm/v1/worker/gpu_model_runner.py:612) (110 samples, 3.77%)
   98  step (vllm/v1/engine/core.py:276) (98 samples, 3.36%)
   86  schedule (vllm/v1/core/sched/scheduler.py:552) (86 samples, 2.95%)
   74  schedule (vllm/v1/core/sched/scheduler.py:333) (74 samples, 2.54%)
   72  _update_states (vllm/v1/worker/gpu_model_runner.py:610) (72 samples, 2.47%)
   67  execute_model (vllm/v1/worker/cpu_worker.py:651) (67 samples, 2.30%)
   66  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:268) (66 samples, 2.26%)
```


## 3. Phase 2 — TRACE counter 실측

파일: [`phase2/trace_counters.txt`](phase2/trace_counters.txt)

```
=== [HYBRID-CPU-ATTN] counter (from server run log) ===
(no [HYBRID-CPU-ATTN] markers)

=== [HYBRID-CPU-ATTN-IPEX] counter (from server boot log) ===
(no [HYBRID-CPU-ATTN-IPEX] markers)

=== [HYBRID-CPU-ATTN] counter (from server boot log — fallback) ===
(no [HYBRID-CPU-ATTN] in boot log)
```

## 4. 판정 참고

Top hot functions 를 본 뒤, 아래 범주 중 지배적인 것 식별:

| 주요 hot path | 의미 |
|---|---|
| `find_longest_cache_hit` / `get_computed_blocks` 가 dominant | **prefix cache 탐색** 이 bottleneck. heavy workload 에서 16K prompt × 1024 hash block 의 Python loop 매칭 |
| `_update_states` 가 dominant | GPU 상속 state update 의 PP 분기 + block_ids loop 가 heavy 에서 비용 |
| `execute_model` (gpu_model_runner) 가 dominant 이지만 내부가 C++ (forward) | 정상 compute, 최적화 대상 아님 |
| `decorate_context` / `__enter__` 같은 context manager 가 많음 | inclusive time (stack 공통), self-time 아님 |
| IPEX kernel 이 dominant | native kernel 최적화 대상 |

해석은 `super_power/draft/B2/` 문서에 반영.
