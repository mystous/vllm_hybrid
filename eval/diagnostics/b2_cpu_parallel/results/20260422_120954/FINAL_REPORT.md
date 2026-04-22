# B2 CPU Parallelism 검증 통합 보고서

- 실행 시각 (KST): 20260422_120954
- 실행 브랜치: investigate/b2-cpu-parallelism
- 실행 커밋: 728a38723
- 결과 디렉토리: `eval/diagnostics/b2_cpu_parallel/results/20260422_120954`

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


### Engine PID 1880753

- Flame graph: [`phase3/engine_1880753_flame.svg`](phase3/engine_1880753_flame.svg) (브라우저에서 열기)
- Info: [`phase3/engine_1880753_info.txt`](phase3/engine_1880753_info.txt)

#### Process 요약
```
### PID 1880753 — VLLM::CPU_Engin
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
1880753 Sl+    0 17.1 VLLM::CPU_Engin
1881887 Sl+   10 10.7 VLLM::CPU_Engin
1881895 Sl+   18  9.9 VLLM::CPU_Engin
1881897 Sl+   20  9.6 VLLM::CPU_Engin
1881906 Sl+   29  9.3 VLLM::CPU_Engin
1881896 Sl+   19  9.3 VLLM::CPU_Engin
1881878 Sl+    1  9.3 VLLM::CPU_Engin
1881925 Sl+   48  9.2 VLLM::CPU_Engin
1881924 Sl+   47  9.2 VLLM::CPU_Engin
1881923 Sl+   46  9.2 VLLM::CPU_Engin

```

#### Top hot functions (flame graph 샘플 상위)
```
   94  all (94 samples, 100%)
   94  _bootstrap (threading.py:1032) (94 samples, 100.00%)
   94  _bootstrap_inner (threading.py:1075) (94 samples, 100.00%)
   94  run (threading.py:1012) (94 samples, 100.00%)
   94  _worker (concurrent/futures/thread.py:93) (94 samples, 100.00%)
   94  run (concurrent/futures/thread.py:59) (94 samples, 100.00%)
   94  execute_model (vllm/v1/executor/abstract.py:87) (94 samples, 100.00%)
   94  collective_rpc (vllm/executor/uniproc_executor.py:58) (94 samples, 100.00%)
   94  run_method (vllm/utils/__init__.py:2948) (94 samples, 100.00%)
   94  decorate_context (torch/utils/_contextlib.py:120) (94 samples, 100.00%)
   94  execute_model (vllm/v1/worker/cpu_worker.py:718) (94 samples, 100.00%)
   94  decorate_context (torch/utils/_contextlib.py:120) (94 samples, 100.00%)
   94  execute_model (vllm/v1/worker/gpu_model_runner.py:1584) (94 samples, 100.00%)
   94  _wrapped_call_impl (torch/nn/modules/module.py:1775) (94 samples, 100.00%)
   94  _call_impl (torch/nn/modules/module.py:1786) (94 samples, 100.00%)
   94  forward (vllm/model_executor/models/qwen2.py:496) (94 samples, 100.00%)
   94  __call__ (vllm/compilation/decorators.py:206) (94 samples, 100.00%)
   94  forward (vllm/model_executor/models/qwen2.py:361) (94 samples, 100.00%)
   94  _wrapped_call_impl (torch/nn/modules/module.py:1775) (94 samples, 100.00%)
   94  _call_impl (torch/nn/modules/module.py:1786) (94 samples, 100.00%)
```


### Engine PID 1880754

- Flame graph: [`phase3/engine_1880754_flame.svg`](phase3/engine_1880754_flame.svg) (브라우저에서 열기)
- Info: [`phase3/engine_1880754_info.txt`](phase3/engine_1880754_info.txt)

#### Process 요약
```
### PID 1880754 — VLLM::CPU_Engin
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
1880754 Sl+   56 16.4 VLLM::CPU_Engin
1881729 Sl+   86 10.5 VLLM::CPU_Engin
1881701 Sl+   58 10.5 VLLM::CPU_Engin
1881703 Sl+   60 10.1 VLLM::CPU_Engin
1881732 Sl+   89  9.6 VLLM::CPU_Engin
1881725 Sl+   82  9.6 VLLM::CPU_Engin
1881717 Sl+   74  9.5 VLLM::CPU_Engin
1881746 Sl+  103  9.4 VLLM::CPU_Engin
1881727 Sl+   84  9.2 VLLM::CPU_Engin
1881722 Sl+   79  9.2 VLLM::CPU_Engin

```

#### Top hot functions (flame graph 샘플 상위)
```
   96  all (96 samples, 100%)
   96  _bootstrap (threading.py:1032) (96 samples, 100.00%)
   96  _bootstrap_inner (threading.py:1075) (96 samples, 100.00%)
   96  run (threading.py:1012) (96 samples, 100.00%)
   96  _worker (concurrent/futures/thread.py:93) (96 samples, 100.00%)
   96  run (concurrent/futures/thread.py:59) (96 samples, 100.00%)
   96  execute_model (vllm/v1/executor/abstract.py:87) (96 samples, 100.00%)
   96  collective_rpc (vllm/executor/uniproc_executor.py:58) (96 samples, 100.00%)
   96  run_method (vllm/utils/__init__.py:2948) (96 samples, 100.00%)
   96  decorate_context (torch/utils/_contextlib.py:120) (96 samples, 100.00%)
   96  execute_model (vllm/v1/worker/cpu_worker.py:718) (96 samples, 100.00%)
   96  decorate_context (torch/utils/_contextlib.py:120) (96 samples, 100.00%)
   96  execute_model (vllm/v1/worker/gpu_model_runner.py:1584) (96 samples, 100.00%)
   96  _wrapped_call_impl (torch/nn/modules/module.py:1775) (96 samples, 100.00%)
   96  _call_impl (torch/nn/modules/module.py:1786) (96 samples, 100.00%)
   96  forward (vllm/model_executor/models/qwen2.py:496) (96 samples, 100.00%)
   96  __call__ (vllm/compilation/decorators.py:206) (96 samples, 100.00%)
   96  forward (vllm/model_executor/models/qwen2.py:361) (96 samples, 100.00%)
   96  _wrapped_call_impl (torch/nn/modules/module.py:1775) (96 samples, 100.00%)
   96  _call_impl (torch/nn/modules/module.py:1786) (96 samples, 100.00%)
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
