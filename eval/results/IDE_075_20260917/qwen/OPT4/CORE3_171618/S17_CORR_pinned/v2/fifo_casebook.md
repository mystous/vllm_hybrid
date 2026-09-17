# fifo_casebook — S17_CORR_pinned (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 112 consumer layer 49 ← producer layer 48 slot 111 epoch 113 def_task_seq 68786 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789633160996910.0 → enqueue+3.0 → exec_start +2280.5 (선행 실행 점유 2279.75, 미분리 0.75) → service 1517.5 (numa0 1472.0 / numa1 1491.25, skew 20.5) → pub +0.5
- GPU: hot_ready 1789633160997317.2 / pub 1789633160998186.8 (delta 869.5) / h2d_start 1789633160998199.5 (pre gap 882.25, idle 882.25) / h2d 30.0 / combine +2.25
- 규모: cold assignments 46, unique numa0/1 18/18, max rows 11/11, AMX/AVX numa0 0/54; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 8,18,53,798,29,10,533,18,1471
- 선행 task 분해: {"pred_deferred_us": 2279.75, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 p95 부근

- ID: graph 2 replay 119 consumer layer 21 ← producer layer 20 slot 83 epoch 120 def_task_seq 69605 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789633161779948.5 → enqueue+4.5 → exec_start +1649.5 (선행 실행 점유 1648.5, 미분리 1.0) → service 2777.25 (numa0 2753.0 / numa1 2699.0, skew 52.5) → pub +0.5
- GPU: hot_ready 1789633161780264.8 / pub 1789633161782500.0 (delta 2235.25) / h2d_start 1789633161782514.2 (pre gap 2249.5, idle 2249.5) / h2d 29.5 / combine +9.25
- 규모: cold assignments 115, unique numa0/1 27/27, max rows 22/22, AMX/AVX numa0 0/81; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 11,25,82,1606,40,18,945,22,2752
- 선행 task 분해: {"pred_deferred_us": 1648.75, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 최대 tail

- ID: graph 2 replay 216 consumer layer 28 ← producer layer 27 slot 90 epoch 217 def_task_seq 82244 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789633173225048.8 → enqueue+7.5 → exec_start +2429.0 (선행 실행 점유 2410.25, 미분리 18.75) → service 11159.5 (numa0 11131.0 / numa1 11005.75, skew 123.5) → pub +1.0
- GPU: hot_ready 1789633173225349.5 / pub 1789633173235982.8 (delta 10633.25) / h2d_start 1789633173236001.8 (pre gap 10652.25, idle 10652.25) / h2d 25.0 / combine +5.75
- 규모: cold assignments 126, unique numa0/1 31/31, max rows 22/22, AMX/AVX numa0 0/93; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 11,41,161,3391,104,81,7273,64,11130
- 선행 task 분해: {"pred_deferred_us": 2410.25, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 15 consumer layer 11 ← producer layer 10 slot 693 epoch 16 def_task_seq 89210 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789633178327296.2 → enqueue+2.5 → exec_start +5.5 (선행 실행 점유 0.0, 미분리 5.5) → service 140.75 (numa0 123.0 / numa1 124.5, skew 2.75) → pub +101.25
- GPU: hot_ready 1789633178327388.2 / pub 1789633178327303.5 (delta -84.75) / h2d_start 1789633178327396.2 (pre gap 8.0, idle 8.0) / h2d 5.0 / combine +4.0
- 규모: cold assignments 1, unique numa0/1 1/1, max rows 1/1, AMX/AVX numa0 0/3; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,0,3,71,1,0,35,6,122
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## empty-cold

- ID: graph 2 replay 1 consumer layer 1 ← producer layer 0 slot 63 epoch 2 def_task_seq 54815 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789633150672123.8 → enqueue+7.25 → exec_start +78.0 (선행 실행 점유 74.0, 미분리 4.0) → service 74.75 (numa0 34.0 / numa1 38.75, skew 5.25) → pub +395.75
- GPU: hot_ready 1789633150672428.5 / pub 1789633150672132.5 (delta -296.0) / h2d_start 1789633150672437.2 (pre gap 8.75, idle 8.75) / h2d 19.25 / combine +9.5
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,11,0,1,0,0,0,16,33
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

