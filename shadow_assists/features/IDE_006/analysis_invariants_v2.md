# [ANAL.v2] Phase 5 — 4 곳 invariant 의 vLLM v1 git history 추적

**작성**: 2026-05-08 / Phase 5 (analysis only)
**목적**: NEO 가 위반하는 vLLM v1 의 4 곳 invariant 의 *도입 commit* + *원래 의도* + *NEO 충돌 영역* 매핑.

---

## INVARIANT 1 — `prev_step_scheduled_req_ids` 무결성

### 도입 commit

| 항목 | 값 |
|---|---|
| commit | `2ce5c5d3d6` |
| 제목 | `[BugFix] Handle unscheduled requests properly when async scheduling (#27756)` |
| 작성자 | Nick Hill <nhill@redhat.com> |
| 일자 | 2025-10-29 |

### 코드 (scheduler.py)

```python
# in __init__:
self.prev_step_scheduled_req_ids: set[str] = set()

# at end of schedule():
self.prev_step_scheduled_req_ids.clear()
self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())
```

### vLLM 표준 의도

`prev_step_scheduled_req_ids` = 직전 step 에서 token 을 schedule 받은 req 들의 집합. async scheduling 환경에서 *이전 step 의 결과 (sampled tokens) 가 아직 도착 안 한 reqs* 를 식별하는 데 사용. async 가 아닌 경우에도 resumed-from-preemption 의 정확한 분기 위해 유지.

핵심 가정: **req 가 prev_step_scheduled 에 있다 = 직전 step 에서 token 1 개를 받았다 = 그 req 의 num_computed_tokens 은 이미 prev step 에서 ++ 되었다.**

### NEO 와의 충돌

NEO 의 swap_out path 에서 SWAPPED_OUT 으로 status 전환된 reqs 는 prev_step 에서 RUNNING 이었다 (즉 schedule 받았다) → `prev_step_scheduled_req_ids` 에 포함됨. 다음 step 에 다시 RUNNING 으로 schedule 하려면 (cdec dispatch path 활성화 시) 이 set 을 통과해야 하는데, 이 set 의 의미 ("이미 token 받음") 와 cdec 의 "CPU 에서 다시 처리" 의도가 충돌. 단순 set 추가/제거 로 해결 불가 — **재해석 (semantic redefine) 필요**.

---

## INVARIANT 2 — `_make_cached_request_data` 의 `assert not scheduled_in_prev_step`

### 도입 commit

동일: `2ce5c5d3d6 (#27756)` (위 INVARIANT 1 과 같은 PR).

### 코드 (scheduler.py:1271)

```python
num_running_reqs = len(running_reqs)
for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
    req_id = req.request_id
    ...
    scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
    if idx >= num_running_reqs:
        assert not scheduled_in_prev_step  # ← 본 invariant
        resumed_req_ids.add(req_id)
```

### vLLM 표준 의도

이 함수는 (running_reqs ⊕ resumed_reqs) 의 cached request data 를 빌드. 두 그룹의 *분리* 가 핵심 invariant:
- `running_reqs` = 이전 step 에서도 RUNNING 이었던 reqs (decode 진행). idx ∈ [0, num_running_reqs)
- `resumed_reqs` = 이전 step 까지 *RUNNING 아니었던* reqs (preempted 후 재진입 또는 새로 SWAPPED → RUNNING). idx ∈ [num_running_reqs, len(running) + len(resumed))

`assert not scheduled_in_prev_step` 의 의미: **resumed slice 에 있는 req 는 prev step 에서 schedule 안 받은 req 여야 한다** (preempted/swapped 인 동안 schedule 안 됨이 자명 — 그래야 resumed 이지). 이게 깨지면 resumed slice 와 running slice 분리가 무너짐.

### NEO 와의 충돌

`scheduler.py:407` 의 try22 skip (`SWAPPED_OUT outer-loop skip`) 을 제거하면, NEO 시도는 SWAPPED_OUT req 도 매 step 정상 schedule (cdec dispatch path 로). 그러면 SWAPPED_OUT req 가 매 step `prev_step_scheduled_req_ids` 에 등록됨. 다음 step 에 *swap_in 으로 RUNNING 복귀* 하면 그 req 는 `resumed_reqs` slice 에 들어감 (idx ≥ num_running_reqs) BUT `scheduled_in_prev_step=True` → **assertion fire**.

이게 try30/32 의 직접 root. 본 invariant 는 *swap_out 시 prev_step 에서 제거* 해 줘야 NEO 와 양립 가능.

---

## INVARIANT 3 — `scheduled_running_reqs` vs `scheduled_resumed_reqs` 분류

### 도입 commit

| 항목 | 값 |
|---|---|
| commit | `30b44a1598` |
| 제목 | `GPU Model Runner V2 (#25266)` |
| 작성자 | Woosuk Kwon |

### 코드 위치

`vllm/v1/core/sched/output.py` 의 `SchedulerOutput` dataclass:
```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_resumed_reqs: list[Request]
    scheduled_running_reqs: list[Request]
    ...
```

### vLLM 표준 의도

3-way 분류. 각 그룹의 input_batch 처리 path 가 다름:
- `scheduled_new_reqs` — full prefill (prompt → KV alloc + tokens)
- `scheduled_resumed_reqs` — resume from PREEMPTED/SWAPPED (KV 재 alloc 또는 swap-in 후 decode)
- `scheduled_running_reqs` — 이미 RUNNING 인 reqs (decode 1 token)

worker 쪽 input_batch 갱신: new → 새 row 추가, resumed → 기존 row 갱신 (block_table 재 set), running → 그대로 + token.

invariant: 세 그룹은 **서로 disjoint** (∩ = ∅). 한 req 는 한 step 에 정확히 한 그룹에만 속함.

### NEO 와의 충돌

cdec req (SWAPPED_OUT 잔류 + token decode) 는 어느 그룹에 속해야 하는가?
- `scheduled_running_reqs` 로 넣으면 worker 가 정상 RUNNING 처리 → block_table 그대로 사용 → SWAPPED_OUT 의 GPU KV 가 freed 라 stale → cross-req KV contamination → CUDA assert.
- `scheduled_resumed_reqs` 로 넣으면 worker 가 block_table 재 set 시도 → KV CPU 에 있어 GPU pointer 못 셋업 → fail.
- 둘 다 안 됨 → **새 그룹** (예: `scheduled_cdec_reqs`) 필요. 또는 기존 그룹 중 하나의 *sub-class* 처리.

NEO 의 현 구현은 `cdec_ids` 를 별도 list 로 attach 하지만 input_batch 갱신 path 는 `scheduled_running_reqs` 를 따름 — 그래서 try10~15 의 silent crash root (input_batch.block_table.np[req_idx] stale).

---

## INVARIANT 4 — `KVCacheBlocks.empty_kv_cache_blocks` 의 의미론

### 도입 commit (vLLM 표준)

| 항목 | 값 |
|---|---|
| commit | `acaa2c0a4a` |
| 제목 | `[Core] Reuse empty block lists whenever possible in KVCacheBlocks to mitigate GC costs (#24964)` |

### NEO 첫 사용 commit

| 항목 | 값 |
|---|---|
| commit | `9993352008` |
| 제목 | `feat(IDE_006/TSK_015.B-2.b): Scheduler SWAPPED_OUT 정상 schedule + KV alloc 우회` |

### 코드 (kv_cache_manager.py:345)

```python
from vllm.v1.request import RequestStatus as _RS
if request.status == _RS.SWAPPED_OUT:
    return self.empty_kv_cache_blocks
```

### vLLM 표준 의도

`empty_kv_cache_blocks` 는 GC 비용 최소화 목적으로 reuse 되는 *empty list* singleton. 정상 path 에서 `KVCacheBlocks.blocks == []` 인 경우 (e.g. KV 가 이미 cached 되어 새 alloc 불필요) 에 반환되어 caller 가 "alloc 결과가 있긴 하지만 새 block 은 없음" 으로 처리.

vLLM 표준에서 alloc 결과의 의미는 "**새로 할당된 GPU KV blocks**". empty 면 = "이번엔 새 alloc 없음" = "KV 가 이미 모두 prefix cache 에 있거나 외부 컴퓨트 결과로 채워짐".

### NEO 와의 충돌

NEO 는 SWAPPED_OUT req 가 정상 schedule path 를 통과해 token 이 schedule 되도록 `allocate_slots` 의 시작에 `empty_kv_cache_blocks` 반환을 박음. 이는 vLLM 표준의 의도 ("외부 컴퓨트로 KV 채워짐") 와 다른 의도 ("KV 가 GPU 에 *없고* CPU 에 있음, GPU forward 도 안 함, cdec dispatch 로 처리").

caller (scheduler) 는 alloc 결과를 받고 worker 로 보내는데, worker 의 input_batch 갱신 path 는 alloc 결과의 의미를 "새 KV block 없음 = 기존 block_table 그대로" 로 해석 → block_table 에 stale GPU pointer 가 그대로 → cross-req contamination.

본 invariant 의 NEO 충돌 = **alloc 결과의 *의미* 가 두 가지 (vLLM 정상 vs NEO cdec) 로 갈라지는데 caller/worker 는 단일 의미 가정**. semantic dispatch 필요.

---

## 4 invariant 종합 매핑 표

| # | invariant | vLLM 표준 의도 | NEO 충돌 핵심 | 수정 path 후보 |
|---|---|---|---|---|
| 1 | `prev_step_scheduled_req_ids` | "직전 step 에 token 받음" | swap_out 시 set 에 잔류 | swap_out 시 명시적 remove |
| 2 | `assert not scheduled_in_prev_step` | resumed slice 의 prev-step 정합 | swap_in 복귀 시 trigger | invariant 1 fix 와 동치 |
| 3 | `scheduled_running` vs `scheduled_resumed` | 3-way 분리 + worker 처리 분기 | cdec req 의 그룹 미정 | 새 group `scheduled_cdec_reqs` 도입 또는 sub-class |
| 4 | `empty_kv_cache_blocks` 의미 | "새 alloc 없음" | NEO 가 "GPU KV 부재 + cdec 처리" 의미로 박음 | alloc 결과에 *type tag* 추가 (e.g. `cdec_marker=True`) |

## 핵심 결론 (Phase 6 input)

1. **try22 skip 제거 만으로는 chain 못 살림** — invariant 2 가 fire 함. 추가로 invariant 1 (prev_step_scheduled set 관리) 를 NEO swap_out path 에서 갱신 필요.
2. **invariant 1 + 2 fix 후에도 invariant 3 가 막음** — input_batch 갱신 path 가 cdec req 를 RUNNING 으로 처리 → block_table stale. 새 group 또는 marker 도입.
3. **invariant 4 는 단독으로는 통과 가능** — empty_kv_cache_blocks 는 worker 까지 가지만 worker 가 정상 그룹 (running/resumed) 분기 후 처리. invariant 3 fix 시 4 도 자연 정합.
4. **수정 plan 의 최소 변경 path** = (1, 2 fix) + (3 의 새 group `scheduled_cdec_reqs` 도입) + 워커 fork branch 의 cdec 분기 추가. 4 는 자동 정합.

---

## Phase 6 으로 넘기는 파라미터

- **수정 좌표 1**: `scheduler.py` 의 `prev_step_scheduled_req_ids.update(...)` 영역 — swap_out 된 reqs 제거.
- **수정 좌표 2**: `scheduler.py:1271` — assert 의 분기 (NEO cdec req 면 통과) 또는 (1) 의 set 관리로 자동 회피.
- **수정 좌표 3**: `output.py` — `SchedulerOutput.scheduled_cdec_reqs` 신규 field 또는 `scheduled_running_reqs.cdec_marker` 추가.
- **수정 좌표 4**: `gpu_model_runner.py` 의 input_batch 갱신 path — cdec marker 시 block_table 갱신 우회 + cdec dispatch hook 으로 wiring.
