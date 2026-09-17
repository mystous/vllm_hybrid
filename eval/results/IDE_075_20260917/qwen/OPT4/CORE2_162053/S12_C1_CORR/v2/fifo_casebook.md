# fifo_casebook — S12_C1_CORR (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

해당 표본 없음

## 큰 배치 p95 부근

해당 표본 없음

## 큰 배치 최대 tail

해당 표본 없음

## 작은 배치 (숨겨진 계산)

- ID: graph 35 replay 857 consumer layer 58 ← producer layer 57 slot 802 epoch 1154 def_task_seq 365804 step step[DECODE bs=1]
- 부분순서 (µs, 절대): go(p) 1789631274189584.5 → enqueue+2.5 → exec_start +5.25 (선행 실행 점유 0.0, 미분리 5.25) → service 125.0 (numa0 110.75 / numa1 111.5, skew 2.0) → pub +103.0
- GPU: hot_ready 1789631274189651.2 / pub 1789631274189592.0 (delta -59.25) / h2d_start 1789631274189662.2 (pre gap 11.0, idle 11.0) / h2d 3.0 / combine +2.5
- 규모: cold assignments 1, unique numa0/1 1/1, max rows 0/0, AMX/AVX numa0 0/3; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 0,0,6,66,1,1,33,3,110
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## empty-cold

- ID: graph 35 replay 0 consumer layer 1 ← producer layer 0 slot 745 epoch 297 def_task_seq 257815 step step[DECODE bs=1]
- 부분순서 (µs, 절대): go(p) 1789631259519000.5 → enqueue+4.0 → exec_start +30.75 (선행 실행 점유 25.5, 미분리 5.25) → service 33.0 (numa0 7.5 / numa1 11.0, skew 4.75) → pub +267.75
- GPU: hot_ready 1789631259519075.5 / pub 1789631259519007.2 (delta -68.25) / h2d_start 1789631259519085.0 (pre gap 9.5, idle 9.5) / h2d 3.0 / combine +4.5
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 0,0,0,1,0,0,1,3,6
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

