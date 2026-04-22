# B2 CPU Parallelism 검증 통합 보고서

- 실행 시각 (KST): 20260422_113211
- 실행 브랜치: investigate/b2-cpu-parallelism
- 실행 커밋: ba33a2541
- 결과 디렉토리: `eval/diagnostics/b2_cpu_parallel/results/20260422_113211`

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


### Engine PID 1789662

- Flame graph: [`phase3/engine_1789662_flame.svg`](phase3/engine_1789662_flame.svg) (브라우저에서 열기)
- Info: [`phase3/engine_1789662_info.txt`](phase3/engine_1789662_info.txt)

#### Process 요약
```
### PID 1789662 — VLLM::CPU_Engin
threads: 374
cpus   : 0

### OMP/BLAS 라이브러리 로드 상태 (duplication 체크)
/vllm_dev_prj/lib/python3.12/site-packages/opencv_python_headless.libs/libopenblasp-r0-59ffcd50.3.15.so
/vllm_dev_prj/lib/python3.12/site-packages/torch/lib/libgomp.so.1

### Thread 이름 분포
    195 VLLM::CPU_Engin
    173 python
      2 ZMQbg/Reaper
      2 ZMQbg/IO/0
      1 pt_tcpstore_uv
      1 cuda00002000009

### ps -L top-10 by %CPU
1789662 Sl+    0 16.4 VLLM::CPU_Engin
1790594 Sl+    2  9.9 VLLM::CPU_Engin
1790636 Sl+   44  9.7 VLLM::CPU_Engin
1790593 Sl+    1  9.1 VLLM::CPU_Engin
1790631 Sl+   39  9.0 VLLM::CPU_Engin
1790619 Sl+   27  9.0 VLLM::CPU_Engin
1790616 Sl+   24  9.0 VLLM::CPU_Engin
1790612 Sl+   20  9.0 VLLM::CPU_Engin
1790611 Sl+   19  9.0 VLLM::CPU_Engin
1790607 Sl+   15  9.0 VLLM::CPU_Engin

```

#### Top hot functions (flame graph 샘플 상위)
```
   99  all (99 samples, 100%)
   99  _bootstrap (threading.py:1032) (99 samples, 100.00%)
   99  _bootstrap_inner (threading.py:1075) (99 samples, 100.00%)
   99  run (threading.py:1012) (99 samples, 100.00%)
   99  _worker (concurrent/futures/thread.py:93) (99 samples, 100.00%)
   99  run (concurrent/futures/thread.py:59) (99 samples, 100.00%)
   99  execute_model (vllm/v1/executor/abstract.py:87) (99 samples, 100.00%)
   99  collective_rpc (vllm/executor/uniproc_executor.py:58) (99 samples, 100.00%)
   99  run_method (vllm/utils/__init__.py:2948) (99 samples, 100.00%)
   99  decorate_context (torch/utils/_contextlib.py:120) (99 samples, 100.00%)
   99  execute_model (vllm/v1/worker/cpu_worker.py:718) (99 samples, 100.00%)
   99  decorate_context (torch/utils/_contextlib.py:120) (99 samples, 100.00%)
   99  execute_model (vllm/v1/worker/gpu_model_runner.py:1584) (99 samples, 100.00%)
   99  _wrapped_call_impl (torch/nn/modules/module.py:1775) (99 samples, 100.00%)
   99  _call_impl (torch/nn/modules/module.py:1786) (99 samples, 100.00%)
   99  forward (vllm/model_executor/models/qwen2.py:496) (99 samples, 100.00%)
   99  __call__ (vllm/compilation/decorators.py:206) (99 samples, 100.00%)
   99  forward (vllm/model_executor/models/qwen2.py:361) (99 samples, 100.00%)
   99  _wrapped_call_impl (torch/nn/modules/module.py:1775) (99 samples, 100.00%)
   99  _call_impl (torch/nn/modules/module.py:1786) (99 samples, 100.00%)
```


### Engine PID 1789663

- Flame graph: [`phase3/engine_1789663_flame.svg`](phase3/engine_1789663_flame.svg) (브라우저에서 열기)
- Info: [`phase3/engine_1789663_info.txt`](phase3/engine_1789663_info.txt)

#### Process 요약
```
### PID 1789663 — VLLM::CPU_Engin
threads: 374
cpus   : 56

### OMP/BLAS 라이브러리 로드 상태 (duplication 체크)
/vllm_dev_prj/lib/python3.12/site-packages/opencv_python_headless.libs/libopenblasp-r0-59ffcd50.3.15.so
/vllm_dev_prj/lib/python3.12/site-packages/torch/lib/libgomp.so.1

### Thread 이름 분포
    195 VLLM::CPU_Engin
    173 python
      2 ZMQbg/Reaper
      2 ZMQbg/IO/0
      1 pt_tcpstore_uv
      1 cuda00002000009

### ps -L top-10 by %CPU
1789663 Sl+   56 16.3 VLLM::CPU_Engin
1790773 Sl+   58 10.2 VLLM::CPU_Engin
1790787 Sl+   72  9.9 VLLM::CPU_Engin
1790809 Sl+   93  9.6 VLLM::CPU_Engin
1790772 Sl+   57  9.6 VLLM::CPU_Engin
1790812 Sl+   96  9.5 VLLM::CPU_Engin
1790820 Sl+  104  9.4 VLLM::CPU_Engin
1790814 Sl+   98  9.4 VLLM::CPU_Engin
1790800 Sl+   84  9.4 VLLM::CPU_Engin
1790797 Sl+   81  9.4 VLLM::CPU_Engin

```

#### Top hot functions (flame graph 샘플 상위)
```
   99  all (99 samples, 100%)
   99  _bootstrap (threading.py:1032) (99 samples, 100.00%)
   99  _bootstrap_inner (threading.py:1075) (99 samples, 100.00%)
   99  run (threading.py:1012) (99 samples, 100.00%)
   99  _worker (concurrent/futures/thread.py:93) (99 samples, 100.00%)
   99  run (concurrent/futures/thread.py:59) (99 samples, 100.00%)
   99  execute_model (vllm/v1/executor/abstract.py:87) (99 samples, 100.00%)
   99  collective_rpc (vllm/executor/uniproc_executor.py:58) (99 samples, 100.00%)
   99  run_method (vllm/utils/__init__.py:2948) (99 samples, 100.00%)
   99  decorate_context (torch/utils/_contextlib.py:120) (99 samples, 100.00%)
   99  execute_model (vllm/v1/worker/cpu_worker.py:718) (99 samples, 100.00%)
   99  decorate_context (torch/utils/_contextlib.py:120) (99 samples, 100.00%)
   99  execute_model (vllm/v1/worker/gpu_model_runner.py:1584) (99 samples, 100.00%)
   99  _wrapped_call_impl (torch/nn/modules/module.py:1775) (99 samples, 100.00%)
   99  _call_impl (torch/nn/modules/module.py:1786) (99 samples, 100.00%)
   99  forward (vllm/model_executor/models/qwen2.py:496) (99 samples, 100.00%)
   99  __call__ (vllm/compilation/decorators.py:206) (99 samples, 100.00%)
   99  forward (vllm/model_executor/models/qwen2.py:361) (99 samples, 100.00%)
   99  _wrapped_call_impl (torch/nn/modules/module.py:1775) (99 samples, 100.00%)
   99  _call_impl (torch/nn/modules/module.py:1786) (99 samples, 100.00%)
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
