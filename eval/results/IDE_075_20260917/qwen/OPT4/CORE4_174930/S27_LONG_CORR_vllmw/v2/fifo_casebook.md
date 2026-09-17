# fifo_casebook — S27_LONG_CORR_vllmw (CORR)

각 사례는 원 ID·timestamp·부분순서·미분리 구간을 제시. 사례는 전체 통계의 대체가 아님 (전체: dependency_summary_v2.json).

## 큰 배치 전형 (p50)

해당 표본 없음

## 큰 배치 p95 부근

해당 표본 없음

## 큰 배치 최대 tail

해당 표본 없음

## 작은 배치 (숨겨진 계산)

- ID: graph 26 replay 162 consumer layer 18 ← producer layer 17 slot 576 epoch 163 def_task_seq 533849 step step[DECODE bs=8]
- 부분순서 (µs, 절대): go(p) 1789635933280913.0 → enqueue+3.0 → exec_start +4.25 (선행 실행 점유 0.0, 미분리 4.25) → service 185.75 (numa0 130.0 / numa1 121.0, skew 7.75) → pub +217.5
- GPU: hot_ready 1789635933281103.0 / pub 1789635933280922.2 (delta -180.75) / h2d_start 1789635933281111.2 (pre gap 8.25, idle 8.25) / h2d 9.25 / combine +7.0
- 규모: cold assignments 1, unique numa0/1 1/1, max rows 1/1, AMX/AVX numa0 0/3; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 2,0,3,69,1,0,37,14,129
- 선행 task 분해: {"pred_deferred_us": 0, "pred_done_setter_us": 0.0, "pred_imm_us": 0, "pred_other_us": 0, "n_predecessors_in_window": 1, "pending_at_enq": "1"}
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

## empty-cold

- ID: graph 26 replay 0 consumer layer 1 ← producer layer 0 slot 559 epoch 1 def_task_seq 512940 step step[DECODE bs=8]
- 부분순서 (µs, 절대): go(p) 1789635925999421.2 → enqueue+5.75 → exec_start +1507.0 (선행 실행 점유 1504.75, 미분리 2.25) → service 89.25 (numa0 30.25 / numa1 27.75, skew 1.25) → pub +136.0
- GPU: hot_ready 1789635925999514.2 / pub 1789635925999438.2 (delta -76.0) / h2d_start 1789635925999526.0 (pre gap 11.75, idle 11.75) / h2d 8.25 / combine +7.0
- 규모: cold assignments 0, unique numa0/1 0/0, max rows 0/0, AMX/AVX numa0 0/0; stage numa0 (prepare,cpy,q_in,up_gate,act,q_down,down,weight,total) = 1,0,0,0,0,0,1,25,29
- 선행 task 분해: NOT_LINKED
- clock: OK; 매핑: VALIDATED_HEURISTIC_MAPPING+ANCHOR_RESYNC

