# Dev 로직 검증 + Abort 시 CPU slot 영구 누수 버그 발견/수정

**Timestamp (KST)**: 2026-04-11 06:50:41
**대상**: TODO v1 §1 의 dev 로직 검증 잔여 항목 4건 (Tech_done 에 아직 기록 안 된 영역)
**결과**: **신규 버그 발견 및 수정** — aborted request 가 CPU slot 을 영구 점유하는 치명적 슬롯 누수. H100 "capacity 멈춤" 증상의 직접 원인 후보 1 로 확정.

---

## 1. 실행 환경

이전 보고서 `20260411_063046_dev_rtx3090_1.5B_7B_hybrid_verify/environment.json` 와 동일.
- Host: `i9-12900KF` + `RTX 3090` (dev, AVX2 + NUMA 1)
- vLLM `0.1.dev8475+g78fa48cb8`, torch `2.9.0+cu130`, CUDA 13.0
- Git branch `h100_cu13`, 테스트 시작 시점 HEAD `a0d15b3788d40fd85a19e1635bd2d30b08a5bc71` (clean)
- 테스트 종료 시점: `vllm/v1/engine/core_client.py` **uncommitted patch 적용됨** (본 세션의 수정)
- 서버: `./serve.sh hybrid envs/dev_rtx3090_500.env` (Qwen2.5-1.5B-Instruct, 1 CPU engine, cpu_max_num_seqs=1, capacity/cpu-first)

---

## 2. 검증 항목

TODO v1 §1 의 4 건:

| # | 항목 | 결과 |
|---|---|---|
| A | 50+ req **순차 반복** → CPU slot 누수 확인 | **PASS**: 60/60, 평균 1.58s/req, 편차 ±0.05s, CPU=100% 라우팅, 누수 0건 |
| B | `output.finished` 다양 종료 조건 (**length** / **stop** / **abort**) | **버그 발견** → **수정** → PASS |
| C | V1 scheduler `cpu_max_num_seqs=1` 경계 코드 경로 추적 | 코드 리딩으로 경로 확정 (§4 참조) |
| D | H100 "capacity 멈춤" 증상 dev 재현 시도 | **재현 성공** — 원인은 B 의 abort 슬롯 누수. 수정 후 재현 안 됨 |

---

## 3. 핵심 발견 — `abort_requests*` 가 CPU slot 을 반납하지 않음

### 3.1 증상

- Client 가 CPU 에서 실행 중인 request 를 mid-stream 에 disconnect
- Router 의 `cpu_in_flight` 카운터가 `1` 로 stuck, 이후 모든 요청이 GPU 로 overflow
- Router stats 기준 `in_flight=1/1` 영구 유지

### 3.2 재현 로그 (수정 전)

Server log `/tmp/seq_hybrid_serve.log` (= `server_before_fix.log`):
```
06:43:07  dispatch stop_case → cpu:0
06:43:07  dispatch abort_case → gpu   (동시각, cpu in_flight=1)
06:43:07  stop_case finishes, tokens=2                # CPU slot 복귀 (정상)
06:43:08  client disconnect on abort_case             # GPU 에서 abort
06:43:10  dispatch post_abort → gpu                   # ??? CPU 여야 하는데 GPU
...
06:44:08  dispatch long_req_for_cpu_abort → gpu       # 1분 뒤에도 CPU 반납 안 됨
06:44:10~12  dispatch probe[0..9] all → gpu           # 10개 모두 GPU
```

직접 재현 스크립트: `cpu_abort_test.py` (Step 1 에서 long CPU req 를 2초 후 abort, Step 2 에서 10개 probe 보내 어느 엔진으로 가는지 확인).

**수정 전 결과**:
```
long req: aborted=True elapsed=2.01s
probe 0: elapsed=0.027s -> GPU?  (CPU 면 ~0.4s 여야 함)
probe 1: elapsed=0.035s -> GPU?
... (전 10개 동일)
```

### 3.3 근본 원인 (코드 분석)

`vllm/v1/engine/core_client.py::HybridAsyncMPClient.abort_requests_async` (수정 전, line 1480):

```python
async def abort_requests_async(self, request_ids):
    for req_id in request_ids:
        entry = self._hybrid_reqs_in_flight.get(req_id)   # get, not pop
        engine = entry[0] if entry is not None else self._gpu_engine
        by_engine[engine].append(req_id)
    for engine, req_ids in by_engine.items():
        await self._send_input(EngineCoreRequestType.ABORT, req_ids, engine)
```

1. `.get()` 만 사용 → `_hybrid_reqs_in_flight` 에서 제거 안 함
2. `on_request_finished` 호출 없음 → `cpu_in_flight` 감소 안 함
3. Engine 은 ABORT 받아서 scheduler `finish_requests()` → `_free_request()` 호출하지만,
4. Scheduler 의 `finished_req_ids_dict` 는 **`data_parallel_size > 1` 에서만 populate 됨** (`core.py:122`):
   ```python
   include_finished_set=vllm_config.parallel_config.data_parallel_size > 1
   ```
5. DP=1 (dev, H100x4 모두 해당) 에서는 `finished_req_ids_dict is None` →
   `EngineCoreOutputs.finished_requests` 필드가 **영원히 비어있음**
6. Aborted request 는 새 token 을 만들지 않으므로 `outputs.outputs` 에도 `output.finished=True` 가 emit 안 됨
7. 결과: `process_engine_outputs` 의 어느 경로로도 slot 반납이 일어나지 않음 → 영구 누수

### 3.4 수정

`vllm/v1/engine/core_client.py::HybridAsyncMPClient.abort_requests_async` (패치 후):

```python
async def abort_requests_async(self, request_ids):
    for req_id in request_ids:
        entry = self._hybrid_reqs_in_flight.pop(req_id, None)   # pop
        if entry is not None:
            engine, engine_path = entry
            num_tokens = self._hybrid_req_token_counts.pop(req_id, 0)
            self._hybrid_router.on_request_finished(             # 명시적 반납
                req_id, engine_path, num_tokens=num_tokens)
        else:
            engine = self._gpu_engine
        by_engine[engine].append(req_id)
    ...
```

동일 수정을 `HybridSyncMPClient.abort_requests` (line 1653) 에도 적용.

**이중 반납 방지**: `process_engine_outputs` 도 `.pop(req_id, None)` 을 사용하므로, 엔진 쪽에서 뒤늦게 output 이 오더라도 entry 가 이미 None 이면 skip → 안전.

패치 파일: `patch_core_client.diff`

### 3.5 검증 (수정 후)

동일 스크립트 `cpu_abort_test.py` 재실행 결과:

```
long req: aborted=True elapsed=2.03s
probe 0: elapsed=0.470s -> CPU?   (CPU latency: 4 tokens / 9.9 tok/s ≈ 0.4s)
probe 1: elapsed=0.378s -> CPU?
probe 2: elapsed=0.378s -> CPU?
probe 3: elapsed=0.379s -> CPU?
probe 4: elapsed=0.376s -> CPU?
probe 5: elapsed=0.378s -> CPU?
probe 6: elapsed=0.377s -> CPU?
probe 7: elapsed=0.377s -> CPU?
probe 8: elapsed=0.378s -> CPU?
probe 9: elapsed=0.372s -> CPU?
```

Server log `server_after_fix.log`:
```
06:47:13 dispatch long req → cpu:0
06:47:15 Request finished: cmpl-3815... on cpu:0, tokens=21 (cpu_count=1)    # abort 가 slot 반납
06:47:15 dispatch probe 0 → cpu:0
06:47:16 Request finished: cmpl-d57d... on cpu:0, tokens=4 (cpu_count=2)
06:47:16 dispatch probe 1 → cpu:0
...
06:47:20 dispatch probe 9 → cpu:0
06:47:21 Request finished: cmpl-b846... on cpu:0, tokens=4 (cpu_count=11)   # 11/11 모두 CPU cycle
```

11 req 전부 CPU 에서 정상 cycle. `cpu_count` 가 1→11 순차 증가로 slot 반납이 매번 일어남을 증명.

---

## 4. 보조 검증 결과 (Task A, B, C)

### 4.1 Task A — 60 req 순차 반복 (누수 probe)

`seq_repeat_test.py` — `max_tokens=16` 으로 60개 요청 순차 (앞이 완료된 후 다음) 송출.

결과 (`seq_repeat_results.json`):
- 60/60 성공, 모두 `finish_reason=length`
- 평균 1.58s/req, 편차 ±0.05s — **누수/slowdown 없음**
- Router stats: `CPU=10.2 tok/s (50 reqs), cpu_ratio=100.0%, in_flight=0/1` — 매 req 마다 slot 0→1→0 cycle
- 60/60 → CPU 라우팅 (sequential 이므로 매번 in_flight=0 에서 출발 → cpu-first 가 항상 CPU 선택)

### 4.2 Task B — finish variety (length / stop / abort)

`finish_variety_test.py`

| Case | Result (after fix) |
|---|---|
| length (max_tokens=10) | 0.91s, `finish_reason=length`, CPU ✓ |
| stop (stop=["."]) | 0.21s, `finish_reason=stop`, CPU ✓ |
| abort (stream cancel @ 1s) | client 측 aborted, 19 chunks 수신 ✓ |
| post_abort probe | 0.39s on CPU ✓ (수정 전: 0.04s on GPU) |

length / stop 은 원래부터 정상 (`output.finished=True` 경로). abort 만 수정 필요했음.

### 4.3 Task C — V1 scheduler `cpu_max_num_seqs=1` 경계 경로

`vllm/v1/core/sched/scheduler.py` + `vllm/v1/engine/hybrid_core.py::_create_cpu_vllm_config` 코드 추적 결과:

1. `cpu_sched.max_num_seqs = resolved.cpu_max_num_seqs` (= 1) — standard `max_num_running_reqs` 메커니즘으로 강제
2. **`cpu_sched.enable_chunked_prefill = False`**, `chunked_prefill_enabled = False` — CPU 엔진에서는 chunked prefill 명시 비활성화. Decode 와 interleave 시 극심하게 느려지기 때문. 덕분에 prefill chunk 경계의 edge case 는 존재하지 않음.
3. `cpu_max_model_len = min(gpu_max, cpu_max_batched_tokens × cpu_max_num_seqs)` — chunked prefill 끄면 `max_num_batched_tokens >= max_model_len` 조건 필요
4. 정상 종료 (`length` / `stop`): scheduler → `_free_request()` → `finished_req_ids.add(...)` → update_from_output 에서 per-output `finished=True` emit → `process_engine_outputs` 의 line 1507 경로로 slot 반납 ✓
5. Abort 종료: scheduler `finish_requests()` → `_free_request()` 는 호출되지만, `include_finished_set=False` (DP=1) 조건 때문에 `finished_req_ids_dict` 에 누적 안 됨 → `EngineCoreOutputs.finished_requests` 필드 empty → engine 측에서 slot 반납 신호가 안 나감 → **router 측에서 직접 반납해야 함** (이 세션의 수정)
6. Preemption (KV cache exhaustion): `num_cpu_engines=1`, `cpu_kvcache=8~16GB`, max_model_len 제한 있어 실제 발생 확률 매우 낮음

### 4.4 Task D — H100 capacity 멈춤 원인

증상 "capacity 경계에서 router 가 멈춤" 은 본 abort 누수 버그의 직접 결과임을 dev 에서 재현 완료. H100 운영 환경에서 일어나는 자연스러운 client disconnect (timeout, 네트워크, LB health check) 한 번이면 CPU slot 이 영구 stuck → 이후 모든 요청이 GPU 로만 흐름.

수정 이후 dev 에서는 11 req burst abort 시나리오에서 모두 정상 cycle. H100 재현 테스트는 실제 환경 이관 시 (TODO §3) 다시 확인 필요.

---

## 5. 파일 구성

```
20260411_065041_dev_logic_verify_abort_slot_leak_fix/
├── README.md                          ← 본 문서
├── patch_core_client.diff             ← 수정 diff (abort_requests_async + abort_requests)
├── seq_repeat_test.py                 ← Task A 스크립트
├── seq_repeat_results.json            ← 60 req 순차 결과
├── finish_variety_test.py             ← Task B 스크립트
├── finish_variety_results.json        ← length/stop/abort 결과 (after fix)
├── cpu_abort_test.py                  ← CPU abort 재현/검증 스크립트
├── cpu_abort_results.json             ← after-fix 10 probe 결과
├── server_before_fix.log              ← seq_hybrid 서버 로그 (버그 재현 포함)
└── server_after_fix.log               ← abortfix 서버 로그 (수정 후 동작)
```

## 6. 다음 단계

1. **코드 리뷰 + 커밋** (사용자 지시 후): 본 패치 `vllm/v1/engine/core_client.py` 2 함수 수정 — 단일 commit, 메시지 예: `fix: release CPU slot on abort — prevents permanent in_flight leak`
2. **TODO.md v4 섹션 append**: Task A/B/C/D 완료 표기, §1 거의 완전 해소 (H100 실측만 남음)
3. **Tech_done.md v3 섹션 append**: 본 abort 버그 + 수정 + 재현 증명 기록 (검증 완료 사실만)
4. **TODO §3 H100 이관 전**: 이 패치 적용된 상태로 H100 재시도 — capacity 멈춤 증상이 해소됐는지 확인 필요
