# fifo_casebook — S26_C1_CORR_vllmw (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

해당 표본 없음

## 큰 배치 p95 부근

해당 표본 없음

## 큰 배치 최대 tail

해당 표본 없음

## 작은 배치 (숨겨진 계산)

- ID: graph 35 replay 615 consumer layer 1 ← producer layer 0 slot 745 epoch 765 def_task_seq 331315 step step[DECODE bs=1]
- 부분순서 (µs, 절대): go(p) 1789635528883822.2 → enqueue+3.5 → exec_start +24.5 (선행 실행 점유 20.75, 미분리 3.75) → service 187.5 (numa0 172.75 / numa1 175.25, skew 4.0) → pub +15.0
- GPU: hot_ready 1789635528883890.8 / pub 1789635528883828.0 (delta -62.75) / h2d_start 1789635528883900.0 (pre gap 9.25, idle 9.25) / h2d 4.0 / combine +4.0
- 규모: cold assignments 2, unique numa0/1 2/2, max rows 0/0, AMX/AVX numa0 0/6; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 0,0,9,93,2,2,58,6,172
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.0, "pred_imm_us": 20.75, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

## empty-cold

- ID: graph 35 replay 0 consumer layer 1 ← producer layer 0 slot 745 epoch 150 def_task_seq 253940 step step[DECODE bs=1]
- 부분순서 (µs, 절대): go(p) 1789635518571313.2 → enqueue+6.5 → exec_start +38.0 (선행 실행 점유 28.5, 미분리 9.5) → service 24.25 (numa0 8.25 / numa1 7.5, skew 0.25) → pub +267.0
- GPU: hot_ready 1789635518571388.5 / pub 1789635518571319.8 (delta -68.75) / h2d_start 1789635518571402.0 (pre gap 13.5, idle 13.5) / h2d 3.25 / combine +6.5
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 0,0,0,1,0,0,0,4,7
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING(slot/epoch count + map_at_go + order)

