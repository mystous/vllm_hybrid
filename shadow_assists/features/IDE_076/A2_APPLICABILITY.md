# A2_APPLICABILITY — 같은 Cold job 내부 descriptor 묶음 실행의 적용 조건 (PLAN.md §7.1). 초안 2026-09-17 23:10 KST

| 질문 | 확인 방법·증거 |
|---|---|
| 현재 어느 단위로 pool 작업을 공급하는가? | `forward_prefill` (moe_base.hpp) 가 job 당 서브풀별 7 회 `do_work_stealing_job` (S0 gather qlen task / S1 A 양자화 expert task / S2 gate·up 20 task/expert / S3 act nth task/expert / S4 down 양자화 expert task / S5 down 48 task/expert / S6 가중합 qlen task). CODE_MAP §1 |
| 반복 parallel 호출이 존재하는가? | 있음: 7 회 dispatch + 7 회 장벽(부모 spin wait). task 수 ≈ 1,500/서브풀/job (unique 18). 각 dispatch 는 워커 48 개에 mutex+cv notify (`worker_pool.cpp:149`) |
| 장벽이 필요한 데이터 의존 때문인가? | S2→S3(act 는 gate·up 전체 N 필요: expert 별로만 필요, 전역 장벽은 아님), S3→S4(양자화는 act 전체 행 필요: expert 별), S4→S5(down 은 양자화된 전체 K=1,280 필요: expert 별), S5→S6(가중합은 토큰의 top-k expert 전부 필요: 토큰 별). **모든 장벽이 expert 단위(또는 토큰 단위) 의존이지 job 전역 의존이 아니다** → expert 단위 readiness 로 대체 가능 |
| 줄일 비용이 서비스 경과 시간에 노출되는가? | (a) 격리 replay 의 n 스케일 절편 ≈ 0 → dispatch·장벽의 순수 고정비는 <50 µs/job (A1_GAP_ANALYSIS §6). (b) 그러나 S29 스테이지 합 중 GEMM 외 6 스테이지 = 94 µs (10 %) 이며, 이 안에 '작은 task 의 워커 tail(각 장벽에서 가장 늦은 워커를 기다림)' 이 들어 있다. 장벽 7 개 각각의 tail 이 노출 비용의 실체인지 → **측정 필요 (Q-A2-1)** |
| 현재 이미 충분히 묶여 있는가? | S2/S5 는 expert×N 블록(128 열) 단위로 잘게 분할되어 워커 균형은 좋다. 반대로 S1/S4(양자화) 는 expert 당 1 task(18 개 < 워커 48) 이라 워커 30 개가 유휴 → 장벽 tail 후보 |
| 묶음화로 새로운 불균형이 생길 수 있는가? | expert 단위 파이프라인(gate/up 완료 → 즉시 그 expert 의 act·양자화·down) 은 expert 크기가 같아 균형이 유지되지만 down 48 task 가 뒤로 몰림. 시험으로 판정 |

## 측정 계획 (Q-A2-1: 장벽 tail 의 실제 노출 비용)
1. 격리 replay 에서 job 을 재생하며 각 dispatch 의 (dispatch 시작 → wait 종료) 구간과 워커별 마지막 task 종료 시각을 기록 (PROFILE_BALANCE 유사 계측을 kt_opt proof 모드에만 켬) → 스테이지별 `barrier_tail_us` = wait 종료 − 워커 작업 종료 중앙값.
2. 서버 CORR 계측(v3 기록기의 stage_us) 은 이미 스테이지 합을 주므로, replay 의 stage 합과 대조.
3. 판정: 스테이지 tail 합이 서비스의 ≥3 % 이면 A2 구현(expert 단위 readiness 로 S2~S5 를 한 dispatch 로: descriptor = (expert, stage, N 블록), 의존 카운터), 그 미만이면 `NOT_APPLICABLE_WITH_EVIDENCE`.

## 유지 계약 (§7.2)
상위 FIFO·job 순서·done 게시·H2D·결합 순서·expert/rows·NUMA-local shard·기존 worker pool 유지. 변경은 forward_prefill 안의 스테이지 descriptor 구성뿐. 다음 step 대기·NUMA stealing·새 pool 없음.

## Q-A2-1 1차 측정 (v6, KT_OPT_PROOF=1, replay hot_r1_e18/hot_r2_e18/seq_mixed 각 5+20 회; 2026-09-17 23:16)
| 지표 | 값 |
|---|---|
| dispatch 수 / task 수 | 721 / 697,900 |
| Σ wall (dispatch 시작→wait 종료) | 1,835 ms |
| Σ parent (부모 스레드 자기 task 종료까지) | 1,780 ms (97.0 % of wall) |
| Σ tail (wait 종료 − 워커 종료 중앙값) | 68.0 ms (**3.7 % of wall**), 최대 20.4 ms (1 건, 워커 cv 수면 후 wake 로 추정) |
| Σ 워커 종료 시각 폭 (최초→최후) | 90.3 ms (4.9 % of wall) |
한계: 이 실행은 A1 proof 카운터(RB 블록 호출마다 공유 atomic, 3.79 M 회) 때문에 커널이 62 % 느려진 상태(hot_r1_e18 1,358 µs vs 838) 라 비율이 왜곡될 수 있다 → 카운터를 스레드별로 바꾼 v7 로 재측정(Q-A2-1b). 판정 규칙(§측정 계획 3): tail 합 ≥ 3 % 이면 A2 구현 검토. 1차 값 3.7 % 는 경계.

## 구현 설계 (적용 판정 시; PLAN.md §7.3~7.5 계약)
- 변경 위치: `moe_base.hpp::forward_prefill` 의 S1~S5 (A 양자화 → gate/up → act → down 양자화 → down). S0(gather)·S6(가중합) 은 그대로.
- descriptor: `(job, numa, expert e, stage ∈ {QA, GU(n블록), ACT, QD, DN(n블록)}, output_tile_range)`. 소유권: GU/DN 의 n 블록은 기존 nth 분할 그대로(한 worker 가 한 (e, n블록) 출력을 독점), ACT/QA/QD 는 expert 당 1 descriptor.
- 실행: 한 번의 `do_work_stealing_job(total_descriptors)` 로 공급하되 stage readiness 를 expert 단위 원자 카운터로 관리: GU(e) 20 개 완료 → ACT(e) 가능 → QD(e) → DN(e) 48 개. 완료되지 않은 의존을 가진 descriptor 를 뽑은 worker 는 해당 expert 의 다음 준비 상태를 기다리지 않고 **큐의 다른 descriptor 를 먼저 처리**(순서: 모든 expert 의 GU 를 앞에 배치, 그 뒤 ACT/QD/DN 을 expert 완료 순으로 push 하는 2 단계 큐). 이렇게 하면 장벽 5 → 1 (S6 앞) 로 줄고, expert 별 pipeline 이 겹친다.
- 보존: 각 stage 의 계산 함수(do_gate_up_gemm/apply_activation/from_mat/do_down_gemm/to_mat) 호출 인자·순서·수치 불변; W 공유 없음; NUMA 간 이동 없음; 새 스레드 없음. 기존 pool 의 nested 대기 금지 계약을 지키기 위해 worker 는 blocking wait 를 하지 않는다(준비 안 된 descriptor 는 뒤로 재삽입).
- 검증: descriptor coverage(각 (e, stage, n블록) 정확히 1 회), readiness 위반 0(카운터 검사), 출력 bitwise == 기존 경로(같은 산술 순서), task 수·FIFO 상위 계약 불변, replay 서비스·tail, OFF MAIN.
- 반증·기각 조건: 준비 안 된 descriptor 재삽입으로 spin 이 늘어 p95 악화; 또는 tail 절감 < 3 % 로 이득이 판정 한도 미만.

## Q-A2-1b 재측정 (v7, 스레드별 카운터 → 커널 왜곡 제거; hot_r1_e18 881 µs ≈ 비-proof 838)
| 지표 | A1=0 | A1=1 |
|---|---|---|
| dispatch / task | 721 / 697,900 | 721 / 697,900 |
| Σ wall | 1,555 ms | 1,571 ms |
| Σ parent (부모 task 종료까지) | 96.6 % | 96.4 % |
| Σ tail (wait 종료 − 워커 종료 중앙값) | 56.9 ms = **3.7 %** | 68.6 ms = **4.4 %** |
| Σ 워커 종료 폭 | 80.2 ms = 5.2 % | 94.0 ms = 6.0 % |
| 최대 tail | 18.2 ms (1 건) | 18.8 ms |
판정: 줄일 수 있는 내부 비용(장벽 tail) 이 dispatch 시간의 3.7~4.4 % 로 확인됨 → §7.1 조건 충족, **A2 구현** (설계 §구현 설계). 기대 상한: 장벽 7 → 2 (S0/S6 유지) 로 tail 의 대부분 제거 시 서비스 −3 % 안팎. E2E 판정 한도(2 %) 를 넘을지는 실측으로.
- 구현: `eval/ide076/kt_opt_a2_patch.py` (moe_base.hpp `forward_prefill_a2_`; QA→GU→ACT→QD→DN 위상 순서, expert 단위 atomic readiness, 준비 안 된 task 는 낮은 id 대기 spin). 빌드 v8 은 E2E 블록 종료 후.
