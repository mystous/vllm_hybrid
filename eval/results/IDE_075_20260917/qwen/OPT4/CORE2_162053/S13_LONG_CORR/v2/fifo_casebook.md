# fifo_casebook — S13_LONG_CORR (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

해당 표본 없음

## 큰 배치 p95 부근

해당 표본 없음

## 큰 배치 최대 tail

해당 표본 없음

## 작은 배치 (숨겨진 계산)

- ID: graph 26 replay 111 consumer layer 24 ← producer layer 23 slot 582 epoch 112 def_task_seq 530986 step step[DECODE bs=8]
- 부분순서 (µs, 절대): go(p) 1789631651367903.0 → enqueue+2.0 → exec_start +320.75 (선행 실행 점유 319.25, 미분리 1.5) → service 355.75 (numa0 294.75 / numa1 295.25, skew 1.5) → pub +1.0
- GPU: hot_ready 1789631651368033.5 / pub 1789631651368057.2 (delta 23.75) / h2d_start 1789631651368070.2 (pre gap 36.75, idle 36.75) / h2d 9.0 / combine +8.5
- 규모: cold assignments 5, unique numa0/1 3/3, max rows 3/3, AMX/AVX numa0 0/9; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 4,3,16,149,5,5,91,17,294
- 선행 task 분해: {"pred_deferred_us": 319.5, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 2, "pending_at_enq": "2"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## empty-cold

- ID: graph 26 replay 0 consumer layer 1 ← producer layer 0 slot 559 epoch 1 def_task_seq 517065 step step[DECODE bs=8]
- 부분순서 (µs, 절대): go(p) 1789631647812608.5 → enqueue+4.0 → exec_start +2776.75 (선행 실행 점유 2774.75, 미분리 2.0) → service 91.0 (numa0 33.75 / numa1 37.0, skew 4.5) → pub +136.25
- GPU: hot_ready 1789631647812721.5 / pub 1789631647812623.0 (delta -98.5) / h2d_start 1789631647812735.5 (pre gap 14.0, idle 14.0) / h2d 9.5 / combine +5.0
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,0,0,1,0,0,0,29,33
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

