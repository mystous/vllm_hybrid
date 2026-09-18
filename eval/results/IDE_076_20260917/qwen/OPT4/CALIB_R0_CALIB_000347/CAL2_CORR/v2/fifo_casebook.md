# fifo_casebook —  (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 18 consumer layer 15 ← producer layer 14 slot 77 epoch 275 def_task_seq 106343 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789657754548600.8 → enqueue+3.0 → exec_start +689.75 (선행 실행 점유 689.0, 미분리 0.75) → service 1049.0 (numa0 1024.75 / numa1 985.0, skew 38.75) → pub +0.5
- GPU: hot_ready 1789657754548926.8 / pub 1789657754549401.0 (delta 474.25) / h2d_start 1789657754549415.0 (pre gap 488.25, idle 488.25) / h2d 29.75 / combine +3.75
- 규모: cold assignments 18, unique numa0/1 14/14, max rows 2/2, AMX/AVX numa0 0/42; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 7,17,24,579,20,7,349,18,1024
- 선행 task 분해: {"pred_deferred_us": 689.0, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 큰 배치 p95 부근

- ID: graph 2 replay 227 consumer layer 48 ← producer layer 47 slot 110 epoch 484 def_task_seq 133159 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789657772327295.0 → enqueue+4.75 → exec_start +356.0 (선행 실행 점유 355.5, 미분리 0.5) → service 1796.0 (numa0 1754.0 / numa1 1713.75, skew 40.5) → pub +0.25
- GPU: hot_ready 1789657772327750.2 / pub 1789657772328801.2 (delta 1051.0) / h2d_start 1789657772328810.2 (pre gap 1060.0, idle 1060.0) / h2d 27.25 / combine +5.75
- 규모: cold assignments 44, unique numa0/1 25/25, max rows 4/4, AMX/AVX numa0 0/75; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 9,18,31,992,36,9,636,19,1753
- 선행 task 분해: {"pred_deferred_us": 356.0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 큰 배치 최대 tail

- ID: graph 2 replay 124 consumer layer 21 ← producer layer 20 slot 83 epoch 381 def_task_seq 119605 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789657762368234.8 → enqueue+3.0 → exec_start +1215.25 (선행 실행 점유 1211.75, 미분리 3.5) → service 9743.0 (numa0 9717.0 / numa1 2741.25, skew 6974.75) → pub +1.25
- GPU: hot_ready 1789657762368636.5 / pub 1789657762377733.8 (delta 9097.25) / h2d_start 1789657762377744.2 (pre gap 9107.75, idle 9107.75) / h2d 29.0 / combine +2.0
- 규모: cold assignments 40, unique numa0/1 25/25, max rows 5/5, AMX/AVX numa0 0/75; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 16,28,34,8921,42,9,643,19,9716
- 선행 task 분해: {"pred_deferred_us": 1212.0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 18 consumer layer 47 ← producer layer 46 slot 729 epoch 147 def_task_seq 139157 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789657775246953.5 → enqueue+2.5 → exec_start +3.25 (선행 실행 점유 0.25, 미분리 3.0) → service 134.25 (numa0 111.5 / numa1 112.5, skew 1.5) → pub +121.75
- GPU: hot_ready 1789657775247041.2 / pub 1789657775246959.0 (delta -82.25) / h2d_start 1789657775247050.8 (pre gap 9.5, idle 9.5) / h2d 4.0 / combine +4.75
- 규모: cold assignments 1, unique numa0/1 1/1, max rows 1/1, AMX/AVX numa0 0/3; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,0,3,64,1,0,31,7,111
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## empty-cold

- ID: graph 2 replay 0 consumer layer 1 ← producer layer 0 slot 63 epoch 257 def_task_seq 104065 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789657753760881.5 → enqueue+6.25 → exec_start +777.25 (선행 실행 점유 775.75, 미분리 1.5) → service 73.25 (numa0 43.75 / numa1 45.5, skew 2.25) → pub +188.5
- GPU: hot_ready 1789657753761069.8 / pub 1789657753760892.0 (delta -177.75) / h2d_start 1789657753761079.8 (pre gap 10.0, idle 10.0) / h2d 27.5 / combine +7.75
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,12,0,2,0,0,0,23,43
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

