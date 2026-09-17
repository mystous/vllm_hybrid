# fifo_casebook — S9_CORR (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 215 consumer layer 55 ← producer layer 54 slot 117 epoch 216 def_task_seq 133548 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789630135365240.0 → enqueue+3.75 → exec_start +1892.0 (선행 실행 점유 1891.5, 미분리 0.5) → service 1569.75 (numa0 1544.5 / numa1 1495.5, skew 48.0) → pub +0.25
- GPU: hot_ready 1789630135365609.2 / pub 1789630135366581.0 (delta 971.75) / h2d_start 1789630135366597.0 (pre gap 987.75, idle 987.75) / h2d 29.75 / combine +7.5
- 규모: cold assignments 50, unique numa0/1 20/20, max rows 6/6, AMX/AVX numa0 0/60; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 8,19,42,856,32,10,554,18,1543
- 선행 task 분해: {"pred_deferred_us": 1891.25, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 p95 부근

- ID: graph 2 replay 77 consumer layer 37 ← producer layer 36 slot 99 epoch 78 def_task_seq 115762 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789630118798391.2 → enqueue+6.0 → exec_start +2522.25 (선행 실행 점유 2520.5, 미분리 1.75) → service 2813.75 (numa0 2776.5 / numa1 2657.25, skew 118.5) → pub +1.0
- GPU: hot_ready 1789630118798689.5 / pub 1789630118800985.0 (delta 2295.5) / h2d_start 1789630118800998.2 (pre gap 2308.75, idle 2308.75) / h2d 25.75 / combine +7.25
- 규모: cold assignments 108, unique numa0/1 34/34, max rows 13/13, AMX/AVX numa0 0/102; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 10,25,75,1522,50,15,1032,42,2774
- 선행 task 분해: {"pred_deferred_us": 2520.5, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 최대 tail

- ID: graph 2 replay 240 consumer layer 28 ← producer layer 27 slot 90 epoch 241 def_task_seq 136619 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789630138044782.0 → enqueue+5.5 → exec_start +3763.75 (선행 실행 점유 3762.75, 미분리 1.0) → service 7814.0 (numa0 7436.75 / numa1 7784.0, skew 349.5) → pub +0.5
- GPU: hot_ready 1789630138045081.0 / pub 1789630138052362.2 (delta 7281.25) / h2d_start 1789630138052374.8 (pre gap 7293.75, idle 7293.75) / h2d 31.0 / combine +8.25
- 규모: cold assignments 150, unique numa0/1 30/30, max rows 32/32, AMX/AVX numa0 0/90; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 15,37,150,4029,59,121,2989,31,7435
- 선행 task 분해: {"pred_deferred_us": 3762.5, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 72 consumer layer 51 ← producer layer 50 slot 733 epoch 73 def_task_seq 147790 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789630141426497.8 → enqueue+2.75 → exec_start +3.75 (선행 실행 점유 0.25, 미분리 3.5) → service 136.75 (numa0 120.0 / numa1 121.0, skew 1.75) → pub +122.75
- GPU: hot_ready 1789630141426589.0 / pub 1789630141426504.2 (delta -84.75) / h2d_start 1789630141426603.2 (pre gap 14.25, idle 14.25) / h2d 4.0 / combine +7.0
- 규모: cold assignments 1, unique numa0/1 1/1, max rows 1/1, AMX/AVX numa0 0/3; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,0,3,69,1,0,35,6,119
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## empty-cold

- ID: graph 2 replay 129 consumer layer 1 ← producer layer 0 slot 63 epoch 130 def_task_seq 122690 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789630128111811.8 → enqueue+4.5 → exec_start +264.5 (선행 실행 점유 263.5, 미분리 1.0) → service 71.75 (numa0 48.25 / numa1 41.75, skew 5.5) → pub +166.75
- GPU: hot_ready 1789630128112037.5 / pub 1789630128111823.2 (delta -214.25) / h2d_start 1789630128112046.8 (pre gap 9.25, idle 9.25) / h2d 21.25 / combine +5.75
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,14,0,2,0,0,0,25,47
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

