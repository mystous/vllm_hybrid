# fifo_casebook — S29_CORR_vllmw_fp8fixso (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

- ID: graph 2 replay 232 consumer layer 18 ← producer layer 17 slot 80 epoch 489 def_task_seq 133974 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789638291388558.5 → enqueue+4.5 → exec_start +1113.25 (선행 실행 점유 1112.25, 미분리 1.0) → service 944.0 (numa0 920.25 / numa1 906.5, skew 12.25) → pub +0.5
- GPU: hot_ready 1789638291388931.8 / pub 1789638291389266.5 (delta 334.75) / h2d_start 1789638291389280.8 (pre gap 349.0, idle 349.0) / h2d 31.0 / combine +4.0
- 규모: cold assignments 15, unique numa0/1 13/13, max rows 2/2, AMX/AVX numa0 0/39; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 7,18,21,510,20,6,314,19,919
- 선행 task 분해: {"pred_deferred_us": 1112.5, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 p95 부근

- ID: graph 2 replay 83 consumer layer 22 ← producer layer 21 slot 84 epoch 340 def_task_seq 114732 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789638279488066.0 → enqueue+3.75 → exec_start +1213.25 (선행 실행 점유 1212.75, 미분리 0.5) → service 1379.5 (numa0 1341.0 / numa1 1263.75, skew 76.5) → pub +0.75
- GPU: hot_ready 1789638279488399.8 / pub 1789638279489221.8 (delta 822.0) / h2d_start 1789638279489233.8 (pre gap 834.0, idle 834.0) / h2d 25.0 / combine +3.0
- 규모: cold assignments 29, unique numa0/1 18/18, max rows 4/4, AMX/AVX numa0 0/54; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 7,24,30,770,27,8,451,18,1340
- 선행 task 분해: {"pred_deferred_us": 1212.75, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 큰 배치 최대 tail

- ID: graph 2 replay 65 consumer layer 52 ← producer layer 51 slot 114 epoch 322 def_task_seq 112542 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789638278306603.5 → enqueue+11.25 → exec_start +1475.25 (선행 실행 점유 1474.75, 미분리 0.5) → service 7665.25 (numa0 7624.0 / numa1 1729.25, skew 5890.25) → pub +0.5
- GPU: hot_ready 1789638278306958.8 / pub 1789638278314046.0 (delta 7087.25) / h2d_start 1789638278314060.2 (pre gap 7101.5, idle 7101.5) / h2d 26.25 / combine +2.75
- 규모: cold assignments 23, unique numa0/1 15/15, max rows 4/4, AMX/AVX numa0 0/45; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 13,27,28,7110,26,7,393,16,7623
- 선행 task 분해: {"pred_deferred_us": 1474.5, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## 작은 배치 (숨겨진 계산)

- ID: graph 32 replay 124 consumer layer 3 ← producer layer 2 slot 685 epoch 253 def_task_seq 152569 step step[DECODE bs=2]
- 부분순서 (µs, 절대): go(p) 1789638295168552.8 → enqueue+2.5 → exec_start +3.75 (선행 실행 점유 0.25, 미분리 3.5) → service 134.0 (numa0 114.0 / numa1 110.75, skew 2.25) → pub +120.0
- GPU: hot_ready 1789638295168642.8 / pub 1789638295168559.0 (delta -83.75) / h2d_start 1789638295168653.0 (pre gap 10.25, idle 10.25) / h2d 5.5 / combine +9.25
- 규모: cold assignments 1, unique numa0/1 1/1, max rows 1/1, AMX/AVX numa0 0/3; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,0,3,65,1,1,32,6,113
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.25, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## empty-cold

- ID: graph 2 replay 0 consumer layer 1 ← producer layer 0 slot 63 epoch 257 def_task_seq 104315 step step[DECODE bs=63]
- 부분순서 (µs, 절대): go(p) 1789638274736588.5 → enqueue+14.25 → exec_start +1400.75 (선행 실행 점유 1398.5, 미분리 2.25) → service 86.75 (numa0 47.5 / numa1 45.5, skew 1.75) → pub +167.5
- GPU: hot_ready 1789638274736733.8 / pub 1789638274736601.2 (delta -132.5) / h2d_start 1789638274736742.8 (pre gap 9.0, idle 9.0) / h2d 26.0 / combine +3.75
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,16,0,1,0,0,1,24,47
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

