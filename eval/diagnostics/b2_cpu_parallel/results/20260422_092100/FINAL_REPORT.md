# B2 CPU Parallelism 검증 통합 보고서

- 실행 시각 (KST): 20260422_092100
- 실행 브랜치: investigate/b2-cpu-parallelism
- 실행 커밋: 64bde900d
- 결과 디렉토리: `eval/diagnostics/b2_cpu_parallel/results/20260422_092100`

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


### Engine PID 1490899

- Flame graph: [`phase3/engine_1490899_flame.svg`](phase3/engine_1490899_flame.svg) (브라우저에서 열기)
- Info: [`phase3/engine_1490899_info.txt`](phase3/engine_1490899_info.txt)

#### Process 요약
```
### PID 1490899 — VLLM::CPU_Engin
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
1490899 Rl+    0 45.0 VLLM::CPU_Engin
1491836 Sl+   14  9.6 VLLM::CPU_Engin
1491870 Sl+   48  9.3 VLLM::CPU_Engin
1491861 Sl+   39  9.3 VLLM::CPU_Engin
1491877 Sl+   55  9.2 VLLM::CPU_Engin
1491869 Sl+   47  9.2 VLLM::CPU_Engin
1491868 Sl+   46  9.2 VLLM::CPU_Engin
1491867 Sl+   45  9.2 VLLM::CPU_Engin
1491866 Sl+   44  9.2 VLLM::CPU_Engin
1491865 Sl+   43  9.2 VLLM::CPU_Engin

```

#### Top hot functions (flame graph 샘플 상위)
```
  713  get_computed_blocks (vllm/v1/core/kv_cache_manager.py:181) (713 samples, 24.85%)
  645  find_longest_cache_hit (vllm/v1/core/kv_cache_coordinator.py:231) (645 samples, 22.48%)
  226  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:275) (226 samples, 7.88%)
  201  schedule (vllm/v1/core/sched/scheduler.py:545) (201 samples, 7.01%)
  174  run_busy_loop (vllm/v1/engine/core.py:703) (174 samples, 6.06%)
  163  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:268) (163 samples, 5.68%)
  162  get_cached_block (vllm/v1/core/block_pool.py:89) (162 samples, 5.65%)
  153  schedule (vllm/v1/core/sched/scheduler.py:552) (153 samples, 5.33%)
  146  schedule (vllm/v1/core/sched/scheduler.py:333) (146 samples, 5.09%)
  102  schedule (vllm/v1/core/sched/scheduler.py:337) (102 samples, 3.56%)
   92  _process_input_queue (vllm/v1/engine/core.py:712) (92 samples, 3.21%)
   91  __bool__ (vllm/v1/core/sched/request_queue.py:124) (91 samples, 3.17%)
   87  schedule (vllm/v1/core/sched/scheduler.py:511) (87 samples, 3.03%)
   81  _make_cached_request_data (vllm/v1/core/sched/scheduler.py:657) (81 samples, 2.82%)
   79  prepend_requests (vllm/v1/core/sched/request_queue.py:105) (79 samples, 2.75%)
   77  has_requests (vllm/v1/core/sched/interface.py:121) (77 samples, 2.68%)
   71  schedule (vllm/v1/core/sched/scheduler.py:422) (71 samples, 2.47%)
   68  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:271) (68 samples, 2.37%)
   65  peek_request (vllm/v1/core/sched/request_queue.py:94) (65 samples, 2.27%)
   65  get_computed_blocks (vllm/v1/core/kv_cache_manager.py:187) (65 samples, 2.27%)
```


### Engine PID 1490900

- Flame graph: [`phase3/engine_1490900_flame.svg`](phase3/engine_1490900_flame.svg) (브라우저에서 열기)
- Info: [`phase3/engine_1490900_info.txt`](phase3/engine_1490900_info.txt)

#### Process 요약
```
### PID 1490900 — VLLM::CPU_Engin
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
1490900 Rl+   56 47.5 VLLM::CPU_Engin
1492009 Sl+   57 10.8 VLLM::CPU_Engin
1492063 Sl+  111 10.6 VLLM::CPU_Engin
1492052 Sl+  100 10.6 VLLM::CPU_Engin
1492043 Sl+   91 10.6 VLLM::CPU_Engin
1492015 Sl+   63 10.6 VLLM::CPU_Engin
1492011 Sl+   59 10.6 VLLM::CPU_Engin
1492010 Sl+   58 10.4 VLLM::CPU_Engin
1492042 Sl+   90  9.5 VLLM::CPU_Engin
1492055 Sl+  103  9.3 VLLM::CPU_Engin

```

#### Top hot functions (flame graph 샘플 상위)
```
  693  get_computed_blocks (vllm/v1/core/kv_cache_manager.py:181) (693 samples, 23.85%)
  625  find_longest_cache_hit (vllm/v1/core/kv_cache_coordinator.py:231) (625 samples, 21.51%)
  237  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:275) (237 samples, 8.16%)
  192  schedule (vllm/v1/core/sched/scheduler.py:545) (192 samples, 6.61%)
  174  run_busy_loop (vllm/v1/engine/core.py:703) (174 samples, 5.99%)
  172  get_cached_block (vllm/v1/core/block_pool.py:89) (172 samples, 5.92%)
  159  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:268) (159 samples, 5.47%)
  154  schedule (vllm/v1/core/sched/scheduler.py:552) (154 samples, 5.30%)
  144  schedule (vllm/v1/core/sched/scheduler.py:333) (144 samples, 4.96%)
  106  schedule (vllm/v1/core/sched/scheduler.py:337) (106 samples, 3.65%)
  102  schedule (vllm/v1/core/sched/scheduler.py:511) (102 samples, 3.51%)
   95  _process_input_queue (vllm/v1/engine/core.py:712) (95 samples, 3.27%)
   90  __bool__ (vllm/v1/core/sched/request_queue.py:124) (90 samples, 3.10%)
   87  prepend_requests (vllm/v1/core/sched/request_queue.py:105) (87 samples, 2.99%)
   81  has_requests (vllm/v1/core/sched/interface.py:121) (81 samples, 2.79%)
   77  _make_cached_request_data (vllm/v1/core/sched/scheduler.py:657) (77 samples, 2.65%)
   73  find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:271) (73 samples, 2.51%)
   73  step_with_batch_queue (vllm/v1/engine/core.py:320) (73 samples, 2.51%)
   70  peek_request (vllm/v1/core/sched/request_queue.py:94) (70 samples, 2.41%)
   70  get_computed_blocks (vllm/v1/core/kv_cache_manager.py:187) (70 samples, 2.41%)
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
