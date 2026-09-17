# fifo_casebook — S10_CORR (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 201 consumer layer 53 ← producer layer 52 slot 115 epoch 458 def_task_seq 184294 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789631113300688.5 → enqueue+4.0 → exec_start +1518.25 (선행 실행 점유 1517.5, 미분리 0.75) → service 1464.75 (numa0 1439.75 / numa1 1408.5, skew 29.75) → pub +1.0
- GPU: hot_ready 1789631113301067.2 / pub 1789631113301915.5 (delta 848.25) / h2d_start 1789631113301926.8 (pre gap 859.5, idle 859.5) / h2d 30.0 / combine +4.25
- 규모: cold assignments 33, unique numa0/1 19/19, max rows 7/7, AMX/AVX numa0 0/57; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 8,23,40,811,29,9,497,18,1439
- 선행 task 분해: {"pred_deferred_us": 1517.75, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 p95 부근

- ID: graph 2 replay 237 consumer layer 28 ← producer layer 27 slot 90 epoch 494 def_task_seq 188744 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789631117157028.8 → enqueue+5.0 → exec_start +1767.5 (선행 실행 점유 1766.5, 미분리 1.0) → service 2762.0 (numa0 2736.5 / numa1 2684.75, skew 50.25) → pub +0.25
- GPU: hot_ready 1789631117157340.2 / pub 1789631117159555.8 (delta 2215.5) / h2d_start 1789631117159575.2 (pre gap 2235.0, idle 2235.0) / h2d 28.25 / combine +3.75
- 규모: cold assignments 148, unique numa0/1 24/24, max rows 28/28, AMX/AVX numa0 0/72; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 10,30,115,1516,40,23,979,19,2735
- 선행 task 분해: {"pred_deferred_us": 1766.5, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 최대 tail

- ID: graph 2 replay 117 consumer layer 25 ← producer layer 24 slot 87 epoch 374 def_task_seq 173238 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789631103337132.2 → enqueue+4.25 → exec_start +2229.0 (선행 실행 점유 2227.75, 미분리 1.25) → service 7048.25 (numa0 6964.75 / numa1 3314.5, skew 3658.0) → pub +0.75
- GPU: hot_ready 1789631103337384.8 / pub 1789631103343946.0 (delta 6561.25) / h2d_start 1789631103343955.0 (pre gap 6570.25, idle 6570.25) / h2d 19.75 / combine +7.25
- 규모: cold assignments 116, unique numa0/1 25/25, max rows 18/18, AMX/AVX numa0 0/75; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 38,27,86,2974,91,95,3597,50,6963
- 선행 task 분해: {"pred_deferred_us": 2228.0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 5 consumer layer 15 ← producer layer 14 slot 697 epoch 134 def_task_seq 191843 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789631119701499.5 → enqueue+3.25 → exec_start +2.75 (선행 실행 점유 0.25, 미분리 2.5) → service 193.0 (numa0 178.0 / numa1 178.0, skew 1.25) → pub +66.0
- GPU: hot_ready 1789631119701590.2 / pub 1789631119701505.2 (delta -85.0) / h2d_start 1789631119701604.5 (pre gap 14.25, idle 14.25) / h2d 5.75 / combine +6.0
- 규모: cold assignments 2, unique numa0/1 2/2, max rows 1/1, AMX/AVX numa0 0/6; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,0,7,93,2,2,60,7,177
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## empty-cold

- ID: graph 2 replay 1 consumer layer 1 ← producer layer 0 slot 63 epoch 258 def_task_seq 158690 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789631092551708.5 → enqueue+5.5 → exec_start +74.5 (선행 실행 점유 70.75, 미분리 3.75) → service 64.5 (numa0 31.75 / numa1 38.25, skew 7.75) → pub +416.0
- GPU: hot_ready 1789631092552014.5 / pub 1789631092551715.5 (delta -299.0) / h2d_start 1789631092552027.8 (pre gap 13.25, idle 13.25) / h2d 21.0 / combine +3.0
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,11,0,1,0,0,0,14,30
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

