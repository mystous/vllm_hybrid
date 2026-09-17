# fifo_casebook — S22_CORR_vllmw (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 136 consumer layer 14 ← producer layer 13 slot 76 epoch 139 def_task_seq 72716 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789635160705848.2 → enqueue+3.0 → exec_start +442.75 (선행 실행 점유 442.0, 미분리 0.75) → service 1020.75 (numa0 995.75 / numa1 930.25, skew 64.5) → pub +-141328.0
- GPU: hot_ready 1789635160706100.2 / pub 1789635160706492.2 (delta 392.0) / h2d_start 1789635160706507.2 (pre gap 407.0, idle 407.0) / h2d 31.25 / combine +9.5
- 규모: cold assignments 17, unique numa0/1 13/13, max rows 2/2, AMX/AVX numa0 0/39; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 7,21,21,560,22,6,337,18,995
- 선행 task 분해: {"pred_deferred_us": 442.25, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 p95 부근

- ID: graph 2 replay 41 consumer layer 34 ← producer layer 33 slot 96 epoch 46 def_task_seq 59756 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789635151933351.8 → enqueue+4.0 → exec_start +778.5 (선행 실행 점유 777.75, 미분리 0.75) → service 1201.5 (numa0 1170.25 / numa1 1003.5, skew 163.0) → pub +181392.75
- GPU: hot_ready 1789635151873667.0 / pub 1789635151933631.2 (delta 59964.25) / h2d_start 1789635151873675.0 (pre gap 8.0, idle 8.0) / h2d 22.25 / combine +8.25
- 규모: cold assignments 19, unique numa0/1 14/14, max rows 3/3, AMX/AVX numa0 0/42; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 8,22,22,570,137,7,382,18,1169
- 선행 task 분해: {"pred_deferred_us": 778.25, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 최대 tail

- ID: graph 2 replay 86 consumer layer 59 ← producer layer 58 slot 121 epoch 91 def_task_seq 65806 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789635154736929.2 → enqueue+3.25 → exec_start +1067.0 (선행 실행 점유 1065.75, 미분리 1.25) → service 112062.5 (numa0 53252.5 / numa1 1332.75, skew 51910.5) → pub +2.25
- GPU: hot_ready 1789635154737333.2 / pub 1789635154848760.5 (delta 111427.25) / h2d_start 1789635154848765.5 (pre gap 111432.25, idle 111432.25) / h2d 18.0 / combine +2.0
- 규모: cold assignments 14, unique numa0/1 14/14, max rows 1/1, AMX/AVX numa0 0/42; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 8,20,28,1321,74,48,27751,23997,53250
- 선행 task 분해: {"pred_deferred_us": 1066.0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 13 consumer layer 28 ← producer layer 27 slot 710 epoch 14 def_task_seq 88869 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789635168294528.2 → enqueue+3.0 → exec_start +3.25 (선행 실행 점유 0.0, 미분리 3.25) → service 184.5 (numa0 170.0 / numa1 170.0, skew 1.25) → pub +45.0
- GPU: hot_ready 1789635168294618.2 / pub 1789635168294533.8 (delta -84.5) / h2d_start 1789635168294627.0 (pre gap 8.75, idle 8.75) / h2d 5.75 / combine +2.75
- 규모: cold assignments 2, unique numa0/1 2/2, max rows 1/1, AMX/AVX numa0 0/6; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,1,7,92,2,2,54,7,169
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## empty-cold

- ID: graph 2 replay 0 consumer layer 1 ← producer layer 0 slot 63 epoch 1 def_task_seq 54440 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789635149650348.2 → enqueue+7.0 → exec_start +879.75 (선행 실행 점유 878.5, 미분리 1.25) → service 79.75 (numa0 49.25 / numa1 51.5, skew 3.25) → pub +165.5
- GPU: hot_ready 1789635149650499.5 / pub 1789635149650358.5 (delta -141.0) / h2d_start 1789635149650512.2 (pre gap 12.75, idle 12.75) / h2d 25.75 / combine +8.0
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,11,0,1,0,0,0,31,48
- 선행 task 분해: NOT_LINKED
- clock: CLOCK_INDETERMINATE; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

