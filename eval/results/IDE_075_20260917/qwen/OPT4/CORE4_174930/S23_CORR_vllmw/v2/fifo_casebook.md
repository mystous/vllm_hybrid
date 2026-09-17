# fifo_casebook — S23_CORR_vllmw (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 124 consumer layer 42 ← producer layer 41 slot 104 epoch 381 def_task_seq 120147 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789635280576815.5 → enqueue+5.5 → exec_start +760.25 (선행 실행 점유 759.0, 미분리 1.25) → service 915.25 (numa0 890.5 / numa1 849.0, skew 40.0) → pub +1.25
- GPU: hot_ready 1789635280577171.5 / pub 1789635280577499.8 (delta 328.25) / h2d_start 1789635280577513.8 (pre gap 342.25, idle 342.25) / h2d 29.75 / combine +8.5
- 규모: cold assignments 17, unique numa0/1 12/12, max rows 2/2, AMX/AVX numa0 0/36; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 6,21,20,488,20,7,306,18,890
- 선행 task 분해: {"pred_deferred_us": 759.0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 p95 부근

- ID: graph 2 replay 189 consumer layer 43 ← producer layer 42 slot 105 epoch 447 def_task_seq 129024 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789635287062109.0 → enqueue+6.0 → exec_start +670.5 (선행 실행 점유 669.25, 미분리 1.25) → service 1386.75 (numa0 1359.75 / numa1 1301.0, skew 58.0) → pub +1.5
- GPU: hot_ready 1789635287062444.0 / pub 1789635287063270.2 (delta 826.25) / h2d_start 1789635287063279.5 (pre gap 835.5, idle 835.5) / h2d 29.5 / combine +6.75
- 규모: cold assignments 32, unique numa0/1 18/18, max rows 7/7, AMX/AVX numa0 0/54; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 8,22,51,741,29,9,476,19,1358
- 선행 task 분해: {"pred_deferred_us": 669.5, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 최대 tail

- ID: graph 2 replay 189 consumer layer 2 ← producer layer 1 slot 64 epoch 446 def_task_seq 128817 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789635286949498.5 → enqueue+5.5 → exec_start +806.5 (선행 실행 점유 805.5, 미분리 1.0) → service 7882.5 (numa0 1085.0 / numa1 7842.0, skew 6769.5) → pub +1.5
- GPU: hot_ready 1789635286949930.8 / pub 1789635286957161.2 (delta 7230.5) / h2d_start 1789635286957172.5 (pre gap 7241.75, idle 7232.25) / h2d 30.0 / combine +8.25
- 규모: cold assignments 14, unique numa0/1 12/12, max rows 2/2, AMX/AVX numa0 0/36; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 10,32,17,603,22,15,352,30,1084
- 선행 task 분해: {"pred_deferred_us": 805.75, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 25 consumer layer 9 ← producer layer 8 slot 691 epoch 154 def_task_seq 140456 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789635291821924.2 → enqueue+2.5 → exec_start +3.75 (선행 실행 점유 0.0, 미분리 3.75) → service 194.25 (numa0 176.0 / numa1 173.75, skew 1.25) → pub +55.0
- GPU: hot_ready 1789635291822016.2 / pub 1789635291821931.5 (delta -84.75) / h2d_start 1789635291822024.0 (pre gap 7.75, idle 7.75) / h2d 6.0 / combine +3.5
- 규모: cold assignments 2, unique numa0/1 2/2, max rows 1/1, AMX/AVX numa0 0/6; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,0,7,90,4,3,57,8,175
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## empty-cold

- ID: graph 2 replay 0 consumer layer 1 ← producer layer 0 slot 63 epoch 257 def_task_seq 104565 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789635273147298.5 → enqueue+7.25 → exec_start +1117.0 (선행 실행 점유 1115.5, 미분리 1.5) → service 94.0 (numa0 51.25 / numa1 50.5, skew 0.5) → pub +159.25
- GPU: hot_ready 1789635273147444.8 / pub 1789635273147307.2 (delta -137.5) / h2d_start 1789635273147455.5 (pre gap 10.75, idle 10.75) / h2d 24.25 / combine +4.0
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,13,0,1,0,0,1,32,50
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

