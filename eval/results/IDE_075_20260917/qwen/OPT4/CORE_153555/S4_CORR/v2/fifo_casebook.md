# fifo_casebook — S4_CORR (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 63 consumer layer 53 ← producer layer 52 slot 115 epoch 576 def_task_seq 162419 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789627310031822.8 → enqueue+3.5 → exec_start +552.0 (선행 실행 점유 551.5, 미분리 0.5) → service 908.5 (numa0 883.0 / numa1 880.5, skew 1.75) → pub +0.5
- GPU: hot_ready 1789627310032191.0 / pub 1789627310032506.0 (delta 315.0) / h2d_start 1789627310032520.2 (pre gap 329.25, idle 329.25) / h2d 32.5 / combine +8.75
- 규모: cold assignments 17, unique numa0/1 12/12, max rows 3/3, AMX/AVX numa0 0/36; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 7,25,20,483,20,5,301,17,882
- 선행 task 분해: {"pred_deferred_us": 551.75, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 큰 배치 p95 부근

- ID: graph 2 replay 233 consumer layer 12 ← producer layer 11 slot 74 epoch 746 def_task_seq 184212 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789627323107379.0 → enqueue+3.5 → exec_start +1444.25 (선행 실행 점유 1443.5, 미분리 0.75) → service 1462.5 (numa0 1436.5 / numa1 1045.75, skew 390.0) → pub +0.5
- GPU: hot_ready 1789627323107773.0 / pub 1789627323108596.0 (delta 823.0) / h2d_start 1789627323108616.0 (pre gap 843.0, idle 843.0) / h2d 28.75 / combine +4.0
- 규모: cold assignments 21, unique numa0/1 13/13, max rows 3/3, AMX/AVX numa0 0/39; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 19,46,197,777,21,7,350,16,1435
- 선행 task 분해: {"pred_deferred_us": 1443.5, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 큰 배치 최대 tail

- ID: graph 2 replay 246 consumer layer 19 ← producer layer 18 slot 81 epoch 759 def_task_seq 185851 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789627323987357.8 → enqueue+11.5 → exec_start +2056.0 (선행 실행 점유 2052.75, 미분리 3.25) → service 3654.75 (numa0 3626.25 / numa1 2551.25, skew 1076.25) → pub +0.5
- GPU: hot_ready 1789627323987795.5 / pub 1789627323990783.8 (delta 2988.25) / h2d_start 1789627323990795.2 (pre gap 2999.75, idle 2999.75) / h2d 25.0 / combine +8.5
- 규모: cold assignments 23, unique numa0/1 16/16, max rows 3/3, AMX/AVX numa0 0/48; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 14,112,43,2422,66,60,882,23,3625
- 선행 task 분해: {"pred_deferred_us": 2052.75, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 36 consumer layer 8 ← producer layer 7 slot 690 epoch 293 def_task_seq 191704 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789627325477544.8 → enqueue+3.25 → exec_start +4.75 (선행 실행 점유 0.0, 미분리 4.75) → service 129.0 (numa0 109.5 / numa1 112.5, skew 4.5) → pub +107.5
- GPU: hot_ready 1789627325477634.5 / pub 1789627325477551.0 (delta -83.5) / h2d_start 1789627325477646.5 (pre gap 12.0, idle 12.0) / h2d 3.75 / combine +1.75
- 규모: cold assignments 1, unique numa0/1 1/1, max rows 1/1, AMX/AVX numa0 0/3; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,0,3,61,2,1,32,6,109
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## empty-cold

- ID: graph 2 replay 0 consumer layer 1 ← producer layer 0 slot 63 epoch 513 def_task_seq 154440 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789627306752014.5 → enqueue+6.75 → exec_start +880.0 (선행 실행 점유 878.25, 미분리 1.75) → service 81.75 (numa0 38.25 / numa1 49.25, skew 13.0) → pub +177.5
- GPU: hot_ready 1789627306752160.2 / pub 1789627306752027.0 (delta -133.25) / h2d_start 1789627306752170.5 (pre gap 10.25, idle 10.25) / h2d 27.0 / combine +5.5
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,11,0,1,0,0,0,21,37
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

