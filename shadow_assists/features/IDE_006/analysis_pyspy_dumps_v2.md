# [ANAL.v2] Phase 2-A/B/3 — py-spy stack 분석

**작성**: 2026-05-08
**소스**: try44 (5 dumps) / try45 (6 dumps) / try46 (5 dumps) — 모두 EngineCore PID 대상

---

## 1. 분석 한계 — 무엇을 보고 무엇을 못 보는가

| 영역 | py-spy attach 대상 | 가시 |
|---|---|---|
| EngineCore main thread (스케줄링 + dispatch wait) | ✅ | scheduler.py / shm_broadcast / sched_yield |
| EngineCore 보조 thread (process_input_sockets, process_output_sockets) | ✅ | ZeroMQ poll, socket I/O |
| **Worker (TP0~TP7) — 실제 GPU forward + CPU pacpu thread pool** | ❌ | 별도 process, 본 dump 에서 미가시 |

→ NEO 의 *cdec dispatch* (`_get_neo_cdec_executor` 의 ThreadPoolExecutor) 는 **Worker 측** 에서 실행되므로 본 dump 들은 *EngineCore 의 step 진행 상태* 만 보여줌. CPU pacpu thread 의 active/idle 상태는 별도 worker 측 py-spy 필요.

## 2. EngineCore main thread state — try44 vs try46 비교

### try44 (v1.2, chain 미발화)

**dump 1 (warmup phase)** — `dump_01_*.txt`
```
Thread 1660323 (idle): "MainThread"
    poll (zmq/sugar/poll.py:106)
        timeout: 60000  (= 60s waiting)
    wait (vllm/distributed/device_communicators/shm_broadcast.py:186)
        timeout_ms: 60000
        current_time: 32055.568059768
    acquire_read (shm_broadcast.py:674)
    dequeue (shm_broadcast.py:755)
```

→ MainThread idle, **shm_broadcast 의 dequeue 대기** (60s timeout). worker 측 처리 결과 도착 대기 중. step 진행은 정상이지만 CPU 측 idle.

### try46 (v38, chain 발화 중)

**dump 3 (peak generation)** — `dump_03_223550.txt`
```
Thread 1892357 (active): "MainThread"
    sched_yield (vllm/distributed/utils.py:48)
    wait (shm_broadcast.py:184)
```

→ MainThread **active** — `sched_yield` 적극 호출. busy synchronization with workers. v38 의 step rate 가 v1.2 보다 빠른 점과 정합 (more steps/s → more sync points).

**보조 thread**:
| Thread | state | 위치 |
|---|---|---|
| MainThread | active | sched_yield + shm_broadcast.wait |
| MultiprocWorkerMonitor | idle | watchdog |
| Thread-1 (process_input_sockets) | idle | ZeroMQ poll |
| Thread-2 (process_output_sockets) | idle | ZeroMQ poll |
| signal-callback | idle | signal handler |

총 5 thread 모두 EngineCore 측 — IPC 와 sync 관련. **CPU pacpu 처리는 worker 측 별도** (본 dump 미가시).

## 3. try44 chain 미발화 의 직접 증거 — *no* cdec executor thread

try44 의 모든 dump 에서 **`_get_neo_cdec_executor` 관련 thread 부재**. 이는:
- adapter 가 cdec_ids=[] 로 attach (P5 trace 와 일치)
- worker 가 _neo_b0_eff_for_step / _neo_b1_eff_for_step 를 None 으로 set (P6=0 과 일치)
- forward_context.neo_cdec_token_slice/seq_slice/req_ids 가 set 안 됨 (P7 _interesting=False 와 일치)
- attention.py 의 cdec_future submit 분기 미진입 → **cdec executor thread 자체가 spawn 안 됨**

## 4. try45 OOM crash 직전 stack — `forward_double` 진입 확인

try45 의 마지막 dump (5번째 또는 break-on-crash dump) 에는 OOM 직전 worker side stack 이 부분 보일 수 있으나, EngineCore PID 만 attach 했으므로 실제 OOM 발생 worker thread 는 미가시. crash log (`gpu_model_runner.py:3808` traceback) 가 worker 의 정확 좌표를 보여줌:

```
File "vllm/v1/worker/gpu_model_runner.py", line 3792
    outputs = _inner_model.forward_neo_pipelined(...)
File "vllm/model_executor/models/llama.py", line 607
    out0, out1 = executor.forward_pipeline(sub_batches, embeddings_list)
File "vllm/v1/worker/sub_batch_executor.py", line 297
    q1, k1, v1, next_emb0 = self.forward_double(...)
File "vllm/v1/worker/sub_batch_executor.py", line 231
    q0_next, k0_next, v0_next = self.cb.preproj(...)
File "vllm/model_executor/models/llama.py", line 563
    hidden, residual = layer.neo_preproj(emb, residual)
File "vllm/model_executor/models/llama.py", line 361
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
→ torch.cuda.OutOfMemoryError: CUDA out of memory.
   Tried to allocate 126.00 MiB. GPU 5 has 92.00 MiB free.
```

→ **chain 정상 firing** (P5/P6/P7 fire 후) → forward_double 진입 → preproj 의 input_layernorm allocation 시점 OOM.

## 5. 다음 *수정 plan* 의 input — Worker side py-spy 필요

본 분석에서 *못 본* 것들:
- CPU pacpu thread pool 의 active/idle 비율 (CLAUDE.md objective 의 CPU util 측정)
- forward_double 의 preproj/postproj 실행 시간 (b0/b1 overlap 효율)
- Q D2H transfer 의 실제 발화 시점 (`qkvtr_e.synchronize()`)

→ 다음 verification phase 에서 **Worker_TP0 같은 worker process 에 py-spy attach** 해야 NEO 의 *진짜 동작* (CPU 가 active 인지) 측정 가능.

## 6. Reference

- `eval/results/20260508_211625_try44_anal_v2_phase2A/pyspy_dumps/` — 5 dumps
- `eval/results/20260508_222510_try45_anal_v2_phase2B/pyspy_dumps/` — 6 dumps (crash 전후)
- `eval/results/20260508_223224_try46_anal_v2_phase3_v38/pyspy_dumps/` — 5 dumps (v38 chain firing)

## 7. Change log

| 일자 | 변경 |
|---|---|
| 2026-05-08 | 초안 — try44/45/46 dump 분석 |
