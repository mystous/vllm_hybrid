# fifo_casebook — S18_CORR_pinned (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 51 consumer layer 9 ← producer layer 8 slot 71 epoch 308 def_task_seq 111081 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789633277072682.8 → enqueue+3.75 → exec_start +1335.5 (선행 실행 점유 1335.0, 미분리 0.5) → service 1451.5 (numa0 1426.75 / numa1 1291.75, skew 133.75) → pub +0.5
- GPU: hot_ready 1789633277073017.0 / pub 1789633277073899.0 (delta 882.0) / h2d_start 1789633277073918.2 (pre gap 901.25, idle 901.25) / h2d 29.25 / combine +3.0
- 규모: cold assignments 43, unique numa0/1 11/11, max rows 21/21, AMX/AVX numa0 0/33; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 8,18,76,871,20,13,401,16,1425
- 선행 task 분해: {"pred_deferred_us": 1335.25, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 p95 부근

- ID: graph 2 replay 107 consumer layer 36 ← producer layer 35 slot 98 epoch 364 def_task_seq 118135 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789633283260416.8 → enqueue+6.5 → exec_start +2269.75 (선행 실행 점유 2269.0, 미분리 0.75) → service 2796.5 (numa0 2769.5 / numa1 2748.75, skew 19.75) → pub +0.5
- GPU: hot_ready 1789633283260720.5 / pub 1789633283262982.5 (delta 2262.0) / h2d_start 1789633283262989.2 (pre gap 2268.75, idle 2268.75) / h2d 32.25 / combine +8.0
- 규모: cold assignments 129, unique numa0/1 31/31, max rows 18/18, AMX/AVX numa0 0/93; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 11,27,84,1535,45,19,1024,20,2768
- 선행 task 분해: {"pred_deferred_us": 2269.0, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 최대 tail

- ID: graph 2 replay 49 consumer layer 42 ← producer layer 41 slot 104 epoch 306 def_task_seq 110897 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789633276929746.0 → enqueue+4.25 → exec_start +2267.5 (선행 실행 점유 2266.25, 미분리 1.25) → service 7078.25 (numa0 7046.0 / numa1 4336.75, skew 2708.5) → pub +0.75
- GPU: hot_ready 1789633276930073.8 / pub 1789633276936605.2 (delta 6531.5) / h2d_start 1789633276936617.8 (pre gap 6544.0, idle 6544.0) / h2d 22.5 / combine +4.25
- 규모: cold assignments 117, unique numa0/1 23/23, max rows 19/19, AMX/AVX numa0 0/69; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 15,24,80,4763,47,34,2054,24,7045
- 선행 task 분해: {"pred_deferred_us": 2266.25, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 47 consumer layer 45 ← producer layer 44 slot 727 epoch 176 def_task_seq 143278 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789633301736869.0 → enqueue+2.5 → exec_start +5.25 (선행 실행 점유 0.0, 미분리 5.25) → service 137.75 (numa0 119.5 / numa1 117.25, skew 1.0) → pub +117.5
- GPU: hot_ready 1789633301736961.8 / pub 1789633301736877.5 (delta -84.25) / h2d_start 1789633301736975.2 (pre gap 13.5, idle 13.5) / h2d 4.0 / combine +9.75
- 규모: cold assignments 1, unique numa0/1 1/1, max rows 1/1, AMX/AVX numa0 0/3; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,0,3,67,1,1,36,6,119
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## empty-cold

- ID: graph 2 replay 1 consumer layer 1 ← producer layer 0 slot 63 epoch 258 def_task_seq 104815 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789633273448577.0 → enqueue+3.75 → exec_start +64.0 (선행 실행 점유 63.0, 미분리 1.0) → service 62.25 (numa0 34.75 / numa1 37.75, skew 4.25) → pub +434.75
- GPU: hot_ready 1789633273448886.5 / pub 1789633273448589.0 (delta -297.5) / h2d_start 1789633273448894.8 (pre gap 8.25, idle 8.25) / h2d 20.0 / combine +7.0
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,10,0,1,0,0,0,17,33
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

