# G — NEO 원본 정통 rewrite plan (2026-05-17 KST)

> 사용자 지시 (2026-05-17): "barrier 고치고 NEO 원래 논문 구현 그대로 구현. NEO 의 한계 알겠고, 거기서 부터 출발한다."
> 진행: 정적 영향도 분석 → fix 순서 → 단계별 적용 + 22 strict 검증 → 측정.

## 1. 우리 implement vs NEO 원본 차이 (8 곳)

### NEO 원본 (swiftllm/worker/layers/transformer_layer.py + model.py)

```python
# 단일 main stream + cpu_communication_stream (별도 1 개)
cpu_communication_stream = torch.cuda.Stream()

# Transfer event handler (QKV D2H transfer event)
events[stage].qkvtr_e = torch.cuda.Event()

def _comm_wait_compute(self):
    self.cpu_communication_stream.wait_stream(torch.cuda.default_stream())

def _compute_wait_comm(self):
    torch.cuda.default_stream().wait_stream(self.cpu_communication_stream)

def _transfer_qkv(self, q, k, v, batch, cur_stage):
    self._comm_wait_compute()
    with torch.cuda.stream(self.cpu_communication_stream):
        # GPU→CPU async copy (non_blocking=True)
        self.events[cur_stage].qkvtr_e.record()

def _attention(self, q, k, v, batch, cur_stage):
    # prefill (flash_attn) — main stream
    # GPU decode (paged_attention) — main stream
    self.events[cur_stage].qkvtr_e.synchronize()              # host blocking, QKV transfer 끝
    torch.ops.pacpu.paged_attention_cpu(...)                  # CPU compute, 직접 호출 (ThreadPool X)
    with torch.cuda.stream(self.cpu_communication_stream):
        o[-batch.num_cdecs:].copy_(oc, non_blocking=True)
    self._compute_wait_comm()                                 # GPU level sync, host non-blocking

def _forward_pipeline_stage(self, q1, k1, v1, batches, cur_stage):
    self._transfer_qkv(q1, k1, v1, batches[cur_stage^1], cur_stage)   # cpu_comm_stream
    self._swap_out_blocks(batches[cur_stage])                          # cpu_comm_stream
    self._postproj(batches[cur_stage])                                 # main stream
    self._preproj(e0, batches[cur_stage], layer_off=1)                 # main stream
    self._attention(q1, k1, v1, batches[cur_stage^1], cur_stage)       # main stream, layer-end CPU pacpu
```

### 우리 implement (vllm/v1/worker/sub_batch_executor.py + attention.py)

```python
# 3 stream: default + s0 + s1 (Phase 3.3 priority)
self._batch_streams = (
    torch.cuda.Stream(priority=-1),  # s0 (Phase 3.3: gdec high)
    torch.cuda.Stream(priority=0),   # s1 (cdec default)
)

# ThreadPoolExecutor for cdec dispatch
_neo_cdec_executor = ThreadPoolExecutor(max_workers=2)

# Pending queue for Option B 시도 (deque)
_neo_pending_cdec_queue: deque = deque()

def attention(...):
    # GPU forward + cdec dispatch (executor.submit)
    cdec_future = _neo_cdec_executor.submit(_neo_cdec_compute_cpu, ...)
    self.impl.forward(...)                                   # GPU 영역
    out_buf = cdec_future.result()                           # ★ Option A: main thread blocking

def forward_double(...):
    with torch.cuda.stream(s1):                              # ★ 우리 추가
        attn1 = attention(b1, layer_idx)
    with torch.cuda.stream(s0):                              # ★ 우리 추가
        preproj(b0, layer_idx + 1)
    cur_stream.wait_stream(s0); cur_stream.wait_stream(s1)
    attn0_next = attention(b0, layer_idx + 1)
    postproj(attn1, b1, layer_idx)
    preproj(b1, layer_idx + 1)
    _drain_pending()                                          # ★ 우리 추가
    postproj(attn0_next, b0, layer_idx + 1)
```

### 차이 표 (8 곳)

| # | 영역 | NEO 원본 | 우리 implement | 회귀 위험 (정적) |
|---|---|---|---|:-:|
| C1 | CUDA stream 수 | default + cpu_communication_stream (2 개) | default + s0 + s1 (3 개) + cpu_communication_stream | Phase 3.3 priority 효과 제거 |
| C2 | cdec dispatch | `torch.ops.pacpu.paged_attention_cpu(...)` 직접 호출 | `ThreadPoolExecutor.submit(...)` | main thread blocking 영역 동일, GIL race 제거 |
| C3 | cdec wait | `qkvtr_e.synchronize()` + `_compute_wait_comm()` | `cdec_future.result()` | host blocking → host blocking (sync 영역 동일) |
| C4 | `_neo_pending_cdec_queue` deque | 없음 | 있음 (Option B 시도, 안 됨) | starvation 해소 |
| C5 | `_neo_async_cdec_mode` flag | 없음 | 있음 | scope context manager 제거 |
| C6 | `_compute_wait_comm` / `_comm_wait_compute` stream helper | 있음 (GPU level sync) | 없음 | 새 helper 추가 |
| C7 | `qkvtr_e` event | per-stage event (TransformerEvents) | 없음 (cdec_future 가 대체) | 새 event 추가 |
| C8 | forward_double 의 stream context | `_forward_pipeline_stage()` 함수 ordering 만 (cuda.stream context 없음) | `with cuda.stream(s0):` / `with cuda.stream(s1):` 분리 | s0/s1 제거, ordering 정합 |

### 주요 fact

- NEO 원본도 **layer 안 cdec sync** (sync wait). 단 *batch 간 layer offset* 으로 wall hide
- NEO 원본의 batch interleave win = `_forward_pipeline_stage(cur_stage=0)` 의 *batch[0] postproj + preproj || batch[1] attention*
- **barrier 자체는 NEO 원본도 있음**. 단 *batch interleave* + *cpu_communication_stream* 으로 wall path 효과 최소화

## 2. fix 순서 (root → leaf, 회귀 위험 작은 영역부터)

| 단계 | fix | 회귀 위험 | 검증 영역 | 시간 |
|---|---|:-:|---|---|
| **S1** | `_compute_wait_comm` / `_comm_wait_compute` stream helper 추가 (C6) | 작음 | 신규 함수, 호출 위치 없음 | 30 min |
| **S2** | `cpu_communication_stream` lazy init (C1 의 일부, helper 만) | 작음 | 신규 stream object, 호출 위치 없음 | 15 min |
| **S3** | `_neo_async_cdec_mode` flag + `_neo_pending_cdec_queue` deque + context manager 제거 (C4 C5) | 중 | 의미상 dead code, Option B 시도 영역 정리 | 30 min |
| **S4** | `forward_double` 의 `with cuda.stream(s0):` / `with cuda.stream(s1):` 제거 (C8) | 큼 | sub_batch_executor.py 의 forward_double 재작성, 22 strict 검증 | 1-2 일 |
| **S5** | `ThreadPoolExecutor.submit` → `paged_attention_cpu` 직접 호출 (C2) | 큼 | attention.py 의 cdec dispatch path 재작성, 22 strict 검증 | 1-2 일 |
| **S6** | `qkvtr_e` event 도입 + `_transfer_qkv` 영역 정합 (C7) | 큼 | QKV D2H 전용 event 신규, 별도 stream 사용 | 1 일 |
| **S7** | `s0` / `s1` 별 stream 영역 완전 제거 → default + cpu_communication_stream 만 (C1) | 매우 큼 | sub_batch_executor.py 전체 재작성, Phase 3.3 효과 제거 | 1-2 일 |
| **S8** | `_forward_pipeline_stage(cur_stage)` 함수 패턴 도입 — NEO 원본 정통 ordering | 매우 큼 | forward_double 의 기존 패턴 폐기, batch interleave 정합 | 1-2 일 |

### 단계별 회귀 위험 영역

| 영역 | S1-S2 | S3 | S4 | S5 | S6 | S7 | S8 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 22 strict #8 (shape mismatch) | OK | OK | 위험 | 위험 | 위험 | 위험 | 위험 |
| 22 strict #12 (b0/b1 정렬) | OK | OK | 위험 | OK | OK | 위험 | 위험 |
| 22 strict #14 (KV migration) | OK | OK | OK | OK | 위험 | 위험 | 위험 |
| engine_dead | OK | OK | 위험 | 위험 | 위험 | 위험 | 위험 |
| Phase 3.3 priority 효과 | OK | OK | OK | OK | OK | 제거 | 제거 |

## 3. 작업 plan (S1 → S8 순)

| step | 작업 | 산출 | 검증 |
|---|---|---|---|
| 0 | working tree clean (Option B 변경 revert 완료) | git status clean | ✓ |
| **1** | S1 — `_compute_wait_comm` / `_comm_wait_compute` 신규 함수 (attention.py 또는 별도 module) | 함수 정의 + lazy stream init | unit test (호출 시 stream wait_stream 만) |
| **2** | S2 — `cpu_communication_stream` lazy init helper | 함수 + stream object | smoke test (vllm 시작 정상) |
| **3** | short test (100p × 8192) — S1+S2 회귀 없음 확인 | starvation 없음 + 22 strict 유지 | 5 min |
| **4** | S3 — `_neo_async_cdec_mode` 영역 cleanup | 코드 size 감소, dead code 제거 | smoke test |
| **5** | S4 — forward_double 의 `with cuda.stream(s0):` 제거 (단순화 1 단계) | sub_batch_executor.py 변경 | short test (100p) + 22 strict 검증 |
| **6** | S5 — ThreadPoolExecutor 제거, paged_attention_cpu 직접 호출 | attention.py 변경 | short test + 22 strict |
| **7** | S6 — qkvtr_e event 도입 | attention.py + sub_batch_executor.py 변경 | short test + 22 strict |
| **8** | S7 — s0/s1 stream 영역 완전 제거 | fundamental rewrite, sub_batch_executor.py | short test + 22 strict (★ critical) |
| **9** | S8 — `_forward_pipeline_stage(cur_stage)` 패턴 도입 | NEO 원본 정통 ordering 완성 | short test + 22 strict (★ critical) |
| **10** | 500p × 8192 3-run 측정 | v1.6 best 대비 비교 | 3-run avg + CV |

## 4. 회귀 fix iterative

각 step 끝 시 **22 strict 19/19 보존** + **engine_dead = 0** + **shape_mismatch = 0** 검증. 회귀 발생 시:
- 단계 revert + root cause 분석
- 추가 fix 또는 단계 분할
- 다시 검증

각 step 1-2 일 예상. total 1-2 주.

## 5. 예상 효과

| 단계 | wall 영향 | 22 strict |
|---|---|:-:|
| S1-S3 (cleanup) | 0 | 유지 |
| S4 (forward_double 단순화) | ±5% (s0/s1 priority 효과 제거) | 유지 |
| S5 (ThreadPool 제거) | ±0% (sync wait 영역 동일, GIL race 제거) | 유지 |
| S6 (qkvtr_e 도입) | +5% (transfer 영역 정확화) | 유지 |
| S7 (stream 영역 완전 제거) | ±0% (Phase 3.3 priority 효과 제거되지만 NEO 원본 정합) | 유지 |
| **S8 (batch interleave 정합)** | **+10-20%** (NEO 원본의 wall hide 정합) | 유지 |
| total | **+15-25%** | 19/19 보존 |

추정 fact (Option A 영역 baseline v1.6 best 2,197 tps):
- S1-S8 완성 시 throughput **2,500-2,750 tps** (vanilla 의 53-59%)
- paper claim H100 +14% 영역 도달 — 사실상 불가 (vanilla 가 paper baseline 보다 빠름)
- 단 NEO §4.4 algorithm 정통 implement 도달

## 6. 진행 상태 (2026-05-17 KST)

### 적용 완료

| step | 결과 | 검증 |
|---|---|---|
| **S1** | `_neo_comm_wait_compute` + `_neo_compute_wait_comm` 신규 함수 추가 (attention.py:1382-1399) | syntax + AST verify ✓ |
| **S2** | `_get_neo_communication_stream()` (line 1372) 이미 있음 — 변경 X | ✓ |
| **S3a** | attention.py 의 async deque path block (line 1133-1151) 제거 → Option A sync path 만 유지 | syntax ✓ |

### 남은 작업

| step | 작업 | 위험 |
|---|---|---|
| S3b | sub_batch_executor.py 의 `neo_async_cdec_scope` import 정리 — 단 env gate (default 0) 라 효과 동일 | 작음 |
| S3c | attention.py 의 dead code module-level (`_neo_async_cdec_mode`, `_neo_pending_cdec_queue`, `_neo_drain_pending_cdec` 등) 제거 | 작음 |
| **S4** | `forward_double` 의 `with cuda.stream(s0):` / `with cuda.stream(s1):` 제거 → NEO 원본 ordering | **큼** |
| **S5** | `ThreadPoolExecutor` 제거 → `torch.ops.pacpu.paged_attention_cpu` 직접 호출 | **큼** |
| **S6** | `qkvtr_e` event 도입 + `_transfer_qkv` 정합 | **큼** |
| **S7** | `s0` / `s1` 별 stream 영역 완전 제거 | **매우 큼** |
| **S8** | `_forward_pipeline_stage(cur_stage)` 패턴 도입 — NEO 원본 정통 ordering | **매우 큼** |

### S4 design (forward_double rewrite)

```python
# 현재 우리 implement
def forward_double(self, batches, layer_idx, q1, k1, v1, next_emb0):
    s0, s1 = self._get_batch_streams()
    cur_stream = torch.cuda.current_stream()
    s0.wait_stream(cur_stream); s1.wait_stream(cur_stream)
    with torch.cuda.stream(s1):
        attn1 = attention(b1, layer_idx)
    with torch.cuda.stream(s0):
        preproj(b0, layer_idx + 1)
    cur_stream.wait_stream(s0); cur_stream.wait_stream(s1)
    # Stage 1 ...

# NEO 원본 정합 후 (S4 적용)
def forward_double(self, batches, layer_idx, q1, k1, v1, next_emb0):
    # NEO §4.4 forward_double = 2 회 _forward_pipeline_stage 호출
    # Stage 0 (cur_stage=0): batch[0] postproj/preproj | batch[1] attention
    # Stage 1 (cur_stage=1): batch[1] postproj/preproj | batch[0] attention
    q1, k1, v1, next_emb0 = self._forward_pipeline_stage(
        q1, k1, v1, batches, next_emb0, layer_idx, cur_stage=0
    )
    q0, k0, v0, next_emb1 = self._forward_pipeline_stage(
        q1, k1, v1, batches, next_emb0, layer_idx, cur_stage=1
    )
    return q0, k0, v0, next_emb0

def _forward_pipeline_stage(self, q1, k1, v1, batches, next_emb, layer_idx, cur_stage):
    # NEO 원본 의 정확한 ordering
    other = cur_stage ^ 1
    # transfer (cpu_communication_stream)
    self.cb.transfer(q1, k1, v1, batches[other], layer_idx)
    # batch[cur_stage] postproj + 다음 layer preproj (main stream)
    self.cb.postproj(prev_attn, batches[cur_stage], layer_idx)
    q_next, k_next, v_next = self.cb.preproj(next_emb, batches[cur_stage], layer_idx + 1, 0)
    # batch[other] attention (main stream, cdec dispatch + wait)
    attn_other = self.cb.attention(q1, k1, v1, batches[other], layer_idx)
    return q_next, k_next, v_next, attn_other_processed
```

이 design 의 핵심 차이:
- `with cuda.stream(s0/s1):` *제거*
- 모든 main stream 위 sequential — NEO 원본 그대로
- batch interleave 가 *함수 호출 ordering 만* 으로 달성 — NEO §4.4 fact

회귀 위험:
- 22 strict #12 (b0/b1 정렬) — 큰 영향
- shape_mismatch — `attn1` reference 영역 변경
- Phase 3.3 priority 효과 제거 (s0/s1 stream X)

검증: short test (100p × 8192) + 22 strict
