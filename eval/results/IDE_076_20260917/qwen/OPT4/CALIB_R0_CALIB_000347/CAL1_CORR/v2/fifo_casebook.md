# fifo_casebook —  (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 253 consumer layer 53 ← producer layer 52 slot 115 epoch 254 def_task_seq 86544 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789657625467137.0 → enqueue+4.75 → exec_start +1064.0 (선행 실행 점유 1063.25, 미분리 0.75) → service 1119.75 (numa0 1096.0 / numa1 1068.25, skew 26.5) → pub +0.75
- GPU: hot_ready 1789657625467560.0 / pub 1789657625468010.0 (delta 450.0) / h2d_start 1789657625468018.2 (pre gap 458.25, idle 458.25) / h2d 29.25 / combine +6.0
- 규모: cold assignments 21, unique numa0/1 15/15, max rows 3/3, AMX/AVX numa0 0/45; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 8,43,19,596,23,7,377,17,1094
- 선행 task 분해: {"pred_deferred_us": 1063.25, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 큰 배치 p95 부근

- ID: graph 2 replay 97 consumer layer 54 ← producer layer 53 slot 116 epoch 98 def_task_seq 66421 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789657611467513.0 → enqueue+5.75 → exec_start +982.25 (선행 실행 점유 981.5, 미분리 0.75) → service 1698.5 (numa0 1671.5 / numa1 1607.5, skew 62.75) → pub +0.75
- GPU: hot_ready 1789657611467954.8 / pub 1789657611468970.0 (delta 1015.25) / h2d_start 1789657611468981.5 (pre gap 1026.75, idle 1026.75) / h2d 30.75 / combine +2.5
- 규모: cold assignments 36, unique numa0/1 24/24, max rows 3/3, AMX/AVX numa0 0/72; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 8,22,30,948,35,9,581,33,1671
- 선행 task 분해: {"pred_deferred_us": 981.75, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 큰 배치 최대 tail

- ID: graph 2 replay 129 consumer layer 40 ← producer layer 39 slot 102 epoch 130 def_task_seq 71018 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789657616978022.5 → enqueue+4.75 → exec_start +306.5 (선행 실행 점유 305.25, 미분리 1.25) → service 6545.0 (numa0 6468.25 / numa1 3602.0, skew 2866.25) → pub +1.0
- GPU: hot_ready 1789657616978211.2 / pub 1789657616984328.2 (delta 6117.0) / h2d_start 1789657616984343.5 (pre gap 6132.25, idle 6132.25) / h2d 22.75 / combine +3.25
- 규모: cold assignments 54, unique numa0/1 7/7, max rows 35/35, AMX/AVX numa0 0/21; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 7,23,124,4490,64,229,1478,49,6467
- 선행 task 분해: {"pred_deferred_us": 305.0, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 116 consumer layer 44 ← producer layer 43 slot 726 epoch 117 def_task_seq 101526 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789657627945534.0 → enqueue+3.25 → exec_start +3.25 (선행 실행 점유 0.25, 미분리 3.0) → service 146.25 (numa0 119.0 / numa1 130.5, skew 12.0) → pub +120.5
- GPU: hot_ready 1789657627945625.2 / pub 1789657627945539.0 (delta -86.25) / h2d_start 1789657627945637.2 (pre gap 12.0, idle 12.0) / h2d 4.0 / combine +8.75
- 규모: cold assignments 1, unique numa0/1 1/1, max rows 1/1, AMX/AVX numa0 0/3; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,0,3,67,1,0,37,5,118
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## empty-cold

- ID: graph 2 replay 0 consumer layer 1 ← producer layer 0 slot 63 epoch 1 def_task_seq 54190 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789657605108668.0 → enqueue+6.25 → exec_start +515.25 (선행 실행 점유 514.25, 미분리 1.0) → service 73.25 (numa0 44.0 / numa1 46.75, skew 3.75) → pub +186.0
- GPU: hot_ready 1789657605108856.0 / pub 1789657605108677.0 (delta -179.0) / h2d_start 1789657605108869.0 (pre gap 13.0, idle 11.5) / h2d 27.25 / combine +6.25
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,11,0,1,0,0,1,26,43
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

