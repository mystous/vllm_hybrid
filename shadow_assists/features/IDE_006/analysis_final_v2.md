# [ANAL.v2] Phase 6 — 종합 + 다음 *수정 plan* input

**작성**: 2026-05-08 / Phase 6 synthesis
**입력**: Phase 1~5 의 산출물 + try44/45/46 measured + try47 진행 중 (Phase 4)
**목적**: 다음 *수정 plan* 의 input — "NEO chain 을 발화 시키려면 어디를 어떻게 수정하는가" 의 정확한 좌표.

---

## 1. Executive summary

| 지표 | v1.2 (try44, HEAD) | v38 (try46, eeed0d46fc) | gain |
|---|---|---|---|
| generate wall | 3,548 s | 2,075 s | **-41.5%** |
| output_tps | 1,154 | 1,804 | **+56.3%** |
| prompt_tps | 763 | 1,305 | +71% |
| chain firing (active cdec dispatch) | **0 / 4,210 swap** | **11,443+ / 12,300 fork (93%)** | — |
| AssertionError | 0 | 0 | — |
| OOM | 0 | 0 | — |

**핵심 발견**:
1. v38 의 NEO chain 활성화가 vanilla 대비 **+56.3% throughput** 의 직접 원인 — TSK_019 의 목표 (NEO > vanilla) 달성 가능 영역 확인
2. v1.2 의 chain 차단 (`scheduler.py:407` try22 skip) 이 그 56.3% 의 *전부* 손실
3. **try22 skip 제거 시 우려한 4 invariant 가 trigger 안 함** (try45 P3=0) — 4 invariant 는 *fundamental block* 가 아님
4. try22 skip 제거 시 OOM at forward_double — GPU memory budget 이 *실제 block*
5. swap_in path 와 mode select 가 항상 미발화 — NEO 의 *migration loop* 미완

## 2. Phase 별 결과 통합

### Phase 1 — runtime trace (analysis_runtime_trace_v2.md)
- 8 probe 적재 + throttle (try44 의 generate wall 51% 손실 — acceptable)
- v1.2 try44 의 chain break point 정확히 식별: `scheduler.py:407` try22 skip 이 SWAPPED_OUT 을 outer loop 에서 미리 제거 → cdec_ids=[] → 후속 chain 전체 미발화
- max SWAPPED_OUT 동시 잔류 224 (try22 skip 으로 self.running 에 누적)

### Phase 2-A — try44 (v1.2 baseline)
- output_tps 1,154 / generate wall 3,548 s
- swap_evt 4,210 / P2 (try22 skip) 4,207 — 매 swap 마다 skip fire
- AssertionError 0, OOM 0, 정상 종료
- chain firing 0 → throughput 이득 0%

### Phase 2-B — try45 (try22 skip 제거)
- chain ACTIVATES at step=154: cdec_ids=195, b0_eff=35, b1_eff=195, P6=8 (worker × step)
- **AssertionError 0, P3 assert_will_fire=True 0** — invariant 2 (`assert not scheduled_in_prev_step`) 가 trigger 안 됨
- forward_double 진입 → CUDA OOM @ `preproj/input_layernorm` (`llama.py:361`) — alloc 126 MiB / GPU free 92 MiB
- 즉 *invariant 가 막은 게 아니라 GPU memory budget 이 막음*

### Phase 3 — try46 (v38 대조군)
- output_tps 1,804 / generate wall 2,075 s
- NEO FORK STAT: `total=12,300 / eligible=11,443 / active=11,443 / reject_no_subs=856` — 93.0% chain firing
- AssertionError 0, OOM 0, 정상 종료 (gpu_memory_utilization=0.75 로 forward_double 마진 확보)
- v1.2 대비 +56.3% throughput 우월 = NEO chain firing 의 직접 이득

### Phase 4 — try47 (minimal repro 완료)
- 5 reqs × 8192 max_tok + gpu_memory_utilization=0.40 (KV pool 작게) + VLLM_ANAL_DISABLE_TRY22_SKIP=1
- generate wall 110 s / output_tps 369.7 / total output 40,739 tokens
- **swap_evt 0 / chain firing 0 / AssertionError 0** — KV pressure 가 threshold (50%) 미도달
- → 5 reqs 만으로는 H100×8 KV pool 의 50% 도달 불가. **swap firing 이 upstream gate 임을 재확인**.
- skip 제거 setting 도 swap 미발화 시 의미 없음 (SWAPPED_OUT 등장 자체 X)

### Phase 5 — invariants (analysis_invariants_v2.md)
- 4 invariant 의 vLLM v1 도입 commit 확인:
  - INVARIANT 1, 2: PR `2ce5c5d3d6` (#27756, Nick Hill, async scheduling fix)
  - INVARIANT 3: PR `30b44a1598` (#25266, GPU Model Runner V2)
  - INVARIANT 4: PR `acaa2c0a4a` (#24964, KVCacheBlocks GC 최적화)
- NEO 충돌 매핑 + 수정 좌표 도출

## 3. 4 invariant 의 *실제* trigger 여부 — Phase 2-B 결과로 재해석

| invariant | 가설 (Phase 5 사전) | 실제 (Phase 2-B 측정) | 재해석 |
|---|---|---|---|
| 1 (`prev_step_scheduled_req_ids`) | swap_out 시 set 잔류 → 다음 step 의 invariant 2 위반 | trigger 안 함 | swap_in 미발화로 *resumed slice* 진입 자체가 없음 → invariant 1 fire 영역 도달 안 함 |
| 2 (`assert not scheduled_in_prev_step`) | swap_in 복귀 시 fire | **fire 0회** | swap_in 미발화로 resumed_reqs 가 SWAPPED_OUT 출신이 아님 |
| 3 (running vs resumed 분류) | cdec req 의 group 미정 → 잘못된 처리 | 결과 아직 미관찰 | cdec req 가 `scheduled_running_reqs` 에 들어가지만 input_batch.block_table 갱신 path 가 cdec marker 따로 처리 (gpu_model_runner b0_eff/b1_eff swap_states) → 우회 가능 |
| 4 (`empty_kv_cache_blocks`) | 의미 충돌 → block_table stale → CUDA assert | 결과 아직 미관찰 | TSK_015 4.5.2.c B-NEW fix (auto-swap-in 제거) 가 stale 회피. cdec marker chain 으로 우회 |

→ **수정 plan 에서 4 invariant 는 *우선순위 하향*. 진짜 block 은 GPU memory + swap_in/migration 부재.**

## 4. 진짜 chain break — 정확한 좌표

| break | 위치 | 원인 | 수정 path |
|---|---|---|---|
| **B-1** | `scheduler.py:407~414` (try22 SWAPPED_OUT skip) | TSK_019 try22 deadlock fix 의 부산물 — chain 통로 자체 차단 | 조건부 활성화 (chain firing path 만 우회) 또는 *별도 schedule branch* 도입 |
| **B-2** | forward_double GPU memory (try45 OOM @ `llama.py:361`) | b0/b1 dual path 의 추가 buffer alloc — gpu_memory_utilization=0.85 시 마진 부족 | gpu_memory_utilization 자동 reserve (NEO ON 시 0.75 cap) 또는 forward_double workspace pool 도입 |
| **B-3** | `neo_scheduler.py` Step 3 (swap_in) 항상 0 | swap_in candidate evaluation 미구현 또는 조건 불충족 | LRU policy + threshold 기반 swap_in fire (NEO 표준 의 6 단계 algorithm 정합) |
| **B-4** | `neo_scheduler_adapter.py:128` mode select 항상 sequential | TablePerfPredictor 미작성 | heuristic mode select (b1_count > threshold 면 pipelined) 또는 ModelProfiler 적재 |

## 5. 다음 *수정 plan* 의 입력 (Phase 6 deliverable)

### 5.1 · 우선순위

| # | 수정 항목 | 영향 | 난이도 |
|---|---|---|---|
| 1 | B-1 (try22 skip 조건부 활성화) | chain 통로 열기 — *throughput +56% 잠재* | 중 |
| 2 | B-2 (forward_double GPU memory budget) | OOM 회피 — chain firing 안정 | 중 |
| 3 | B-3 (swap_in path 활성화) | NEO 의 *migration loop* 완성 — KV CPU 잔류 + GPU 복원 | 고 |
| 4 | B-4 (mode select 활성화) | sequential vs pipelined 동적 선택 — 적정 workload 매칭 | 저 (heuristic) / 고 (predictor) |

### 5.2 · 좌표

| break | 수정 좌표 | 변경 의도 |
|---|---|---|
| B-1 | `vllm/v1/core/sched/scheduler.py:407` | `if (...): continue` 의 조건에 *NEO chain firing* path 인지 분기 추가. SWAPPED_OUT 이지만 cdec dispatch 대상이면 skip 안 함 |
| B-2 | `vllm/v1/worker/sub_batch_executor.py:231` (`forward_double`) + `vllm/model_executor/models/llama.py:361,533~607` (`neo_preproj`, `forward_neo_pipelined`) | preproj 의 hidden+residual buffer 를 step 시작 시 미리 reserve (workspace pool). 또는 LLM init 시 gpu_memory_utilization 자동 cap (NEO ON 시 0.75) |
| B-3 | `vllm/v1/core/sched/neo_scheduler.py:Step 3` + `vllm/v1/engine/core.py:_handle_neo_swaps` | LRU policy 적재 + swap_in candidate evaluation logic (`gpu_block_needed + get_block_needed(candidate) <= swap_in_threshold` 시 fire) + KV CPU→GPU 복원 worker wiring |
| B-4 | `vllm/v1/core/sched/neo_scheduler_adapter.py:128~169` (predictor init) + `neo_scheduler.py:Step 5` | heuristic 적재: `if b1_count > 32: pipelined`. ModelProfiler 적재 시 TablePerfPredictor 의 table 채우기 (별도 phase) |

### 5.3 · 검증 게이트 (수정 plan 의 종료 조건)

| gate | 측정 | 목표값 |
|---|---|---|
| chain firing | NEO FORK STAT active 비율 | ≥ 90% (v38 수준) |
| throughput | output_tps @ 500p × 50:50 | > vanilla baseline + 3% |
| token correctness | per-token logprob max abs diff (vs vanilla) | ≤ ε |
| stability | run 정상 완료 + AssertionError + OOM = 0 | — |
| CPU utilization | py-spy thread state ratio (cpu pacpu thread active) | ≥ 50% |

## 6. *수정 plan* 별도 작성 시점

본 문서는 *수정 plan 의 input* 이며, 수정 plan 자체는 *별도 phase* 에서 작성 (사용자 명시 명령 후). 단계 분기:

| 단계 | 조건 | 다음 |
|---|---|---|
| 6 (현재) | 본 synthesis 작성 완료 | 사용자 review |
| 7 (Phase 7) | 분석 patch 전체 revert (in-tree only) | git status 클린 |
| 8 | 사용자 명시 *수정 plan 작성 명령* | 별도 *수정 plan v3* 작성 |
| 9 | 수정 plan 승인 + 명시 *수정 실행 명령* | 코드 수정 |

---

## 7. Reference

- `Objective-for-NEO-porting.md` — NEO 19 mechanism 의 구현/동작/근거 표
- `analysis_runtime_trace_v2.md` — Phase 1 runtime trace
- `analysis_invariants_v2.md` — Phase 5 invariant git history + NEO 충돌
- `eval/results/20260508_211625_try44_anal_v2_phase2A/` — try44 measurement
- `eval/results/20260508_222510_try45_anal_v2_phase2B/` — try45 measurement (OOM)
- `eval/results/20260508_223224_try46_anal_v2_phase3_v38/` — try46 measurement (v38 대조군)
- `eval/results/20260508_230921_try47_anal_v2_phase4_repro/` — try47 measurement (chain dormant — minimal workload)

## 8. Change log

| 일자 | 변경 |
|---|---|
| 2026-05-08 | 본 문서 초안 — try44/45/46 결과 + Phase 5 invariant 분석 통합. try47 결과는 turn 후 보강 |
