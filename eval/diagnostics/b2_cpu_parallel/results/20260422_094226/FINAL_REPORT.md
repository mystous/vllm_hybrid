# B2 CPU Parallelism 검증 통합 보고서

- 실행 시각 (KST): 20260422_094226
- 실행 브랜치: investigate/b2-cpu-parallelism
- 실행 커밋: 868a7119a
- 결과 디렉토리: `eval/diagnostics/b2_cpu_parallel/results/20260422_094226`

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


### Engine PID 1539959

- Flame graph: [`phase3/engine_1539959_flame.svg`](phase3/engine_1539959_flame.svg) (브라우저에서 열기)
- Info: [`phase3/engine_1539959_info.txt`](phase3/engine_1539959_info.txt)

#### Process 요약
```
### PID 1539959 — VLLM::CPU_Engin
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
1539959 Rl+    0 44.4 VLLM::CPU_Engin
1541119 Sl+   49 10.3 VLLM::CPU_Engin
1541125 Sl+   55  9.8 VLLM::CPU_Engin
1541090 Sl+   20  9.6 VLLM::CPU_Engin
1541120 Sl+   50  9.1 VLLM::CPU_Engin
1541117 Sl+   47  9.1 VLLM::CPU_Engin
1541116 Sl+   46  9.1 VLLM::CPU_Engin
1541115 Sl+   45  9.1 VLLM::CPU_Engin
1541114 Sl+   44  9.1 VLLM::CPU_Engin
1541113 Sl+   43  9.1 VLLM::CPU_Engin

```

#### Top hot functions (flame graph 샘플 상위)
```
  793  get_computed_blocks (vllm/v1/core/kv_cache_manager.py:181) (793 samples, 26.85%)
  718  find_longest_cache_hit (vllm/v1/core/kv_cache_coordinator.py:231) (718 samples, 24.31%)
  281  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:275) (281 samples, 9.52%)
  227  get_cached_block (vllm/v1/core/block_pool.py:89) (227 samples, 7.69%)
  223  schedule (vllm/v1/core/sched/scheduler.py:545) (223 samples, 7.55%)
  180  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:268) (180 samples, 6.10%)
  169  run_busy_loop (vllm/v1/engine/core.py:703) (169 samples, 5.72%)
  155  schedule (vllm/v1/core/sched/scheduler.py:552) (155 samples, 5.25%)
  148  schedule (vllm/v1/core/sched/scheduler.py:333) (148 samples, 5.01%)
  110  schedule (vllm/v1/core/sched/scheduler.py:337) (110 samples, 3.73%)
   87  __bool__ (vllm/v1/core/sched/request_queue.py:124) (87 samples, 2.95%)
   84  _make_cached_request_data (vllm/v1/core/sched/scheduler.py:657) (84 samples, 2.84%)
   83  _process_input_queue (vllm/v1/engine/core.py:712) (83 samples, 2.81%)
   81  peek_request (vllm/v1/core/sched/request_queue.py:94) (81 samples, 2.74%)
   73  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:271) (73 samples, 2.47%)
   70  has_requests (vllm/v1/core/sched/interface.py:121) (70 samples, 2.37%)
   70  _process_input_queue (vllm/v1/engine/core.py:723) (70 samples, 2.37%)
   69  &lt;lambda&gt; (&lt;string&gt;:1) (69 samples, 2.34%)
   67  step_with_batch_queue (vllm/v1/engine/core.py:304) (67 samples, 2.27%)
   66  schedule (vllm/v1/core/sched/scheduler.py:511) (66 samples, 2.24%)
```


### Engine PID 1539960

- Flame graph: [`phase3/engine_1539960_flame.svg`](phase3/engine_1539960_flame.svg) (브라우저에서 열기)
- Info: [`phase3/engine_1539960_info.txt`](phase3/engine_1539960_info.txt)

#### Process 요약
```
### PID 1539960 — VLLM::CPU_Engin
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
1539960 Rl+   56 43.6 VLLM::CPU_Engin
1540918 Sl+   84 10.5 VLLM::CPU_Engin
1540905 Sl+   71 10.5 VLLM::CPU_Engin
1540891 Sl+   57 10.5 VLLM::CPU_Engin
1540901 Sl+   67 10.1 VLLM::CPU_Engin
1540892 Sl+   58  9.7 VLLM::CPU_Engin
1540933 Sl+   99  9.5 VLLM::CPU_Engin
1540926 Sl+   92  9.5 VLLM::CPU_Engin
1540944 Sl+  110  9.3 VLLM::CPU_Engin
1540923 Sl+   89  9.2 VLLM::CPU_Engin

```

#### Top hot functions (flame graph 샘플 상위)
```
  755  get_computed_blocks (vllm/v1/core/kv_cache_manager.py:181) (755 samples, 25.86%)
  685  find_longest_cache_hit (vllm/v1/core/kv_cache_coordinator.py:231) (685 samples, 23.46%)
  277  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:275) (277 samples, 9.49%)
  205  schedule (vllm/v1/core/sched/scheduler.py:545) (205 samples, 7.02%)
  200  get_cached_block (vllm/v1/core/block_pool.py:89) (200 samples, 6.85%)
  175  run_busy_loop (vllm/v1/engine/core.py:703) (175 samples, 5.99%)
  165  schedule (vllm/v1/core/sched/scheduler.py:552) (165 samples, 5.65%)
  146  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:268) (146 samples, 5.00%)
  132  schedule (vllm/v1/core/sched/scheduler.py:333) (132 samples, 4.52%)
  116  schedule (vllm/v1/core/sched/scheduler.py:337) (116 samples, 3.97%)
   97  _make_cached_request_data (vllm/v1/core/sched/scheduler.py:657) (97 samples, 3.32%)
   87  _process_input_queue (vllm/v1/engine/core.py:712) (87 samples, 2.98%)
   80  step_with_batch_queue (vllm/v1/engine/core.py:320) (80 samples, 2.74%)
   79  has_requests (vllm/v1/core/sched/interface.py:121) (79 samples, 2.71%)
   79  __bool__ (vllm/v1/core/sched/request_queue.py:124) (79 samples, 2.71%)
   77  peek_request (vllm/v1/core/sched/request_queue.py:94) (77 samples, 2.64%)
   77  &lt;lambda&gt; (&lt;string&gt;:1) (77 samples, 2.64%)
   75  schedule (vllm/v1/core/sched/scheduler.py:511) (75 samples, 2.57%)
   72  has_unfinished_requests (vllm/v1/core/sched/interface.py:101) (72 samples, 2.47%)
   70  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:271) (70 samples, 2.40%)
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

| 증거 | 결론 |
|---|---|
| Phase 2 counter `sdpa_loop` dominant | **C** — dispatch 조건 수정 |
| Phase 2 counter `ipex` dominant 이고 여전히 느림 | **A** — IPEX 자체가 long-ctx 에서 single-thread |
| Phase 3 py-spy stack 이 Python attention 함수 | **B** — Python/GIL serialize |
| Phase 3 py-spy `<native>` + perf top `ipex_*_paged_attention` | **A** |
| Phase 3 py-spy `torch::sdpa` 경로 | **C** |
| Phase 3 threads.txt 대부분 S(sleep) + 소수 R | 스케줄 안 됨 → B 또는 native lock |

위 표와 실제 데이터 대조 → 가설 A/B/C 중 하나 선택 → `super_power/draft/B2/` 분석문서의 §8 레이어 3 / §11.1 B1 해석 갱신.
