# fifo_casebook — S3_CORR (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 31 consumer layer 9 ← producer layer 8 slot 71 epoch 288 def_task_seq 108206 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789627183848743.2 → enqueue+5.5 → exec_start +675.0 (선행 실행 점유 674.0, 미분리 1.0) → service 932.0 (numa0 907.25 / numa1 864.75, skew 41.0) → pub +0.5
- GPU: hot_ready 1789627183849119.0 / pub 1789627183849440.0 (delta 321.0) / h2d_start 1789627183849454.8 (pre gap 335.75, idle 335.75) / h2d 32.5 / combine +8.75
- 규모: cold assignments 16, unique numa0/1 12/12, max rows 2/2, AMX/AVX numa0 0/36; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 6,13,23,521,20,5,299,15,906
- 선행 task 분해: {"pred_deferred_us": 674.0, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 큰 배치 p95 부근

- ID: graph 2 replay 95 consumer layer 11 ← producer layer 10 slot 73 epoch 352 def_task_seq 116210 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789627187635132.0 → enqueue+4.25 → exec_start +908.5 (선행 실행 점유 908.0, 미분리 0.5) → service 1440.75 (numa0 1415.25 / numa1 1366.75, skew 47.25) → pub +0.5
- GPU: hot_ready 1789627187635549.5 / pub 1789627187636340.5 (delta 791.0) / h2d_start 1789627187636350.5 (pre gap 801.0, idle 801.0) / h2d 31.0 / combine +4.5
- 규모: cold assignments 31, unique numa0/1 19/19, max rows 4/4, AMX/AVX numa0 0/57; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 8,39,26,789,28,8,496,17,1414
- 선행 task 분해: {"pred_deferred_us": 908.0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 큰 배치 최대 tail

- ID: graph 2 replay 246 consumer layer 10 ← producer layer 9 slot 72 epoch 503 def_task_seq 135708 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789627199707243.2 → enqueue+6.5 → exec_start +676.0 (선행 실행 점유 674.5, 미분리 1.5) → service 6084.25 (numa0 6046.25 / numa1 1138.25, skew 4907.25) → pub +1.0
- GPU: hot_ready 1789627199707581.5 / pub 1789627199713091.0 (delta 5509.5) / h2d_start 1789627199713103.0 (pre gap 5521.5, idle 5521.5) / h2d 25.5 / combine +7.75
- 규모: cold assignments 22, unique numa0/1 14/14, max rows 5/5, AMX/AVX numa0 0/42; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 13,31,23,678,25,7,5181,83,6045
- 선행 task 분해: {"pred_deferred_us": 675.0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 36 consumer layer 36 ← producer layer 35 slot 718 epoch 165 def_task_seq 141635 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789627201236743.2 → enqueue+3.25 → exec_start +3.75 (선행 실행 점유 0.25, 미분리 3.5) → service 134.5 (numa0 118.75 / numa1 120.25, skew 3.0) → pub +103.25
- GPU: hot_ready 1789627201236838.2 / pub 1789627201236752.8 (delta -85.5) / h2d_start 1789627201236849.8 (pre gap 11.5, idle 11.5) / h2d 3.75 / combine +1.75
- 규모: cold assignments 1, unique numa0/1 1/1, max rows 1/1, AMX/AVX numa0 0/3; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,0,3,70,1,1,31,7,118
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## empty-cold

- ID: graph 2 replay 0 consumer layer 1 ← producer layer 0 slot 63 epoch 257 def_task_seq 104315 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789627182462137.5 → enqueue+7.75 → exec_start +732.0 (선행 실행 점유 730.5, 미분리 1.5) → service 82.0 (numa0 50.75 / numa1 55.25, skew 5.5) → pub +163.5
- GPU: hot_ready 1789627182462282.2 / pub 1789627182462146.8 (delta -135.5) / h2d_start 1789627182462292.0 (pre gap 9.75, idle 9.75) / h2d 29.75 / combine +7.0
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,28,1,1,0,1,1,15,50
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

