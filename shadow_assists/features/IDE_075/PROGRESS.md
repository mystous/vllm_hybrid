# PROGRESS — IDE_075

## 09-17 15:25 30분 보고 (착수 14:45)
- A00: 잔여 예산 세션 7·부팅 5·품질 76 확정. IDE_075 등록, 브랜치 feat/cpu-moe-bottleneck-followup-20260917.
- A01: tsparse 단위시험 PASS; 17 세션 창 재구성(WINDOW_RECONSTRUCTION.md); PCM 정정: forward_envelope 창 read 237.0/251.0/247.8 GB/s (legacy 144.8/143.4/139.4). perf NOT_RECOVERABLE.
- A02: v2 지표·cohort·민감도(0/1/2/5 µs)·항등식 residual 을 IDE_074 P1 6세션에 적용. bs=63 cold_present_nonempty 15,597~15,601 행: cold_pub_delta p50 307.8/312.5/334.0 µs, late 비율 0.898/0.897/0.909 (thr5: 0.894/0.893/0.906), pre_h2d gap 안 GPU idle = gap 전체(다른 op 0), residual 0.
- A03: 기록기 v2 (stamp task 제거) 구현·재빌드(.so SHA a4e9045a…), 합성 시험 PASS (링 overflow 검출, 순서, SPSC stale, clock 27 ns/call).
- A11: OFF_OPEN → CORE(S2 CORE, S3/S4 CORR, S5 RESOURCE) → OFF_CLOSE 체인 실행 중 (3 부팅, 6 세션).

## 09-17 15:45 A11 완료
- 세션 6/7: S1_OFF_A 700.9, S6_OFF_B 706.9, S2_CORE 708.2, S3_CORR 699.1, S4_CORR 701.5, S5_RESOURCE 781.6 tok/s. 부팅 시도 5/5 (OFF_OPEN·CORE_152703 은 하네스 결함으로 세션 없이 소진). 예비 S7 없음.
- 분석 체인(v2 지표·FIFO 분해·자원·검증·렌더) 실행 중.

## 09-17 15:59 종료 보고
- 세션 6/7 (예비 없음), 부팅 시도 5/5 (2회 하네스 결함 소진). 분석·검증·보고서 생성 완료. READINESS: 후보 A·B READY_WITH_LIMITED_SCOPE. 게시 진행.

## 09-17 16:13 확장 단계 30분 보고
- 기록기 v3 (경쟁 조건 수정·스레드 이름·expert 표본) 재빌드·설치 (.so SHA d659ca0b…). clock anchor: GPU→host offset +3.5 µs 이내. 클라이언트 barrier(probe_client.py)·TID CPU 시간·FOCUS sched 수집기 추가.
- 기존 S3/S4 재분석: post-MoE all-reduce 에서 rank 0 이 항상 마지막 도착(15,809/15,809), rank 1~3 커널 437 µs vs rank 0 12.5 µs, 도착 시차 p50 577 µs (post-attention 은 4 µs). prefill(EXTEND) 층 682/682 연결: CPU 게시가 hot 보다 1.3~3.1 ms 빠름(늦음 0 %), AMX 경로, GPU 노출 대기 10~12 µs.
- 확장 체인 OFF_OPEN2 → CORE2(S9~S14) → OFF_CLOSE2 실행 중. GLM FP8 경로 코드 검토 진행 중.

## 09-17 16:27 확장 단계 30분 보고
- OFF_OPEN2: S8 OFF(vllm) 708.0 tok/s; S8b(probe 클라이언트) 는 컨테이너 마운트 경로 오류로 실패(rc 1) → 경로 수정. CORE2 부팅(v3 .so) 서버는 살아 있어 ATTACH 로 S9~S14 진행, 이어 OFF_CLOSE2(S15, S15b).
- 완료: 기록기 v3, clock anchor(+3.5 µs 이내), TP collective 위치 귀속(rank 0 마지막 도착 100 %, rank1~3 post-MoE AR 437 µs), prefill EXTEND 682 층 연결(CPU 게시 1.3~3.1 ms 빠름, AMX 경로).
- 진행 중: GLM FP8 경로 코드 검토(에이전트), 확장 세션 실행.

## 09-17 16:56 확장 단계 30분 보고
- CORE2(ATTACH, v3): S9_CORR 472.4, S10_CORR 506.8, S11_RESOURCE 505.8, S12_C1_CORR 57.8 tok/s (모두 probe 클라이언트). S8 OFF(vllm bench) 708.0 보다 낮음 → 클라이언트 차이 가능성; S15/S15b(OFF_CLOSE2 의 vllm vs probe) 로 동등성 판정 예정. 내부 타이밍(kt_evt·트레이스)은 별도 검증.
- GLM: 코드 검토(G00_code_review.md) 결과 H1 = routed_scaling_factor 2.5 가 GPU 기여에만 적용되고 디코드 경로에서 합계에 재적용 → 최소 수정 패치(glm_rsf_patch.py)·진단 스크립트(glm_diag.py D1/D2/D3) 준비. Qwen 체인 뒤 실행.
- 남은 세션: S13 LONG, S14 FOCUS, OFF_CLOSE2(S15, S15b).

## 09-17 17:27 확장 단계 30분 보고
- CORE3(pinned probe): S17 500.6, S18 498.6, S19 FOCUS 487.9, S20 RESOURCE 500.8 tok/s. OFF_X: S16 vllm 693.5, S16b probe-pinned 410.6, S16c vllm 788.6 → probe 클라이언트는 고정해도 느림(KT 코어 간섭 가설 기각). vllm bench 코드 그대로의 barrier 래퍼(vllm_bench_wrapper.py) 구현, OFF_Y/CORE4 체인 예약.
- GLM: 업스트림 클린 빌드 .so 로 FP8 per-channel·블록 FP8 시험 PASS(0.58 %) vs 설치본 출력 0 → 로컬 kt-kernel 패치가 FP8 CPU 경로를 깨뜨림. 그룹별 이분 탐색(빌드·시험) 진행 중. D3/D4/D5 진단 실행 중.
- 분석: CORE2 파이프라인(S12 C1 258 MB 트레이스) 진행 중, CORE3 분석 시작.

## 09-17 18:15 확장 단계 30분 보고
- 클라이언트 동등성 확정 (OFF_Y, 같은 부팅): vllm 705.1 / vllm 래퍼(vllmw) 703.1 / vllm 재실행 705.7 tok/s (TPOT 63.0/63.9/65.3 ms) → 래퍼 동등. probe 는 비동등(410~510). CORE4(S22~S27, vllmw) 측정 완료, 분석 중.
- CORE2 재분석: S12(C1 bs=1) anchor 재동기화로 126,976 쌍 복원 → cold_pub_delta p50 −59 µs(CPU 가 먼저 게시), enq→start 4.0 µs, service 148.8 µs. S9/S10(probe) 는 pub_delta 848~972 µs·service 1427~1556 µs 로 vllm 클라이언트 CORE(315~334 / 902~912) 보다 큼 → probe 간섭이 CPU 서비스 시간을 늘림. FOCUS perf script 파싱 수정, S9 트레이스 링크 복구.
- GLM 근본 원인 확정: bisect → `moe_base.hpp`. IDE_046-b/051 hunk 의 `if constexpr (requires ...) if (env) {...} else ORIGINAL` dangling-else 로 FP8/BF16 커널(BufferABF16Impl) 에서 원본 gather·양자화 경로가 컴파일에서 제거 → 출력 0 (D1~D5 전부 비정상과 일치). 수정 패치 적용·재빌드 중 → FP8 단독 테스트 → D6(rsf 수정) 게이트·G_OFF/G_CORR → D7(rsf 미수정) 게이트.
- 다음 30분: 재빌드 .so 설치·테스트, GLM D6/D7, CORE4 분석, 보고서 렌더 확장(§8b 클라이언트별 비교, §8c G00).

## 09-17 18:17 확장 단계 30분 보고
- GLM 근본 원인 수정 검증: 수정 .so(12926df2…) 로 kt FP8 단독 테스트 2종 PASS(0.5829 %), **GLM D6 4문항 게이트 통과 4/4** (18/3/70000/540, 전부 stop). G_OFF(vllm)·G_CORR(vllmw+프로파일러) 세션 실행 중, 이어 D7(rsf 미수정 게이트) → Qwen FOCUS2(시스템 전체 sched_switch, -k CLOCK_MONOTONIC) + S29 CORR(수정 .so 회귀 확인).
- FOCUS 기존 데이터(S14/S19/S25): `perf record -p` 는 switch-in 을 기록하지 않아 off-CPU 길이 산출 불가(NOT_RECOVERABLE) → 회수 가능 값만 산출: 워커 96 스레드 switch-out 738/835 건(R 87 % 선점, S 13 %), 워커당 초당 0.93~0.95 회, tail 창 안 78/84 건, perf 손실 6~10 %.
- CORE3(pinned probe) 의존성: pub_delta p50 870/882 µs, service 1447/1459 µs (CORE vllm 315~321 / 902~912) → probe 간섭 재확인. CORE4(vllmw) 의존성 분석 실행 중 (anchor 재동기화 빈 DtoH 행 보호 수정).
- 다음 30분: CORE4 값 확정 → READINESS/HANDOFF 갱신, GLM 세션 분석(KT_ROW_BYTES 10240), FOCUS2, 보고서 렌더·검증·게시 준비.

## 09-17 18:30 확장 단계 30분 보고
- CORE4(vllm 래퍼, 기록기 v3) 의존성 확정: S22/S23 cold_pub_delta 346/327 µs · enq→start 670/668 · service 921/916 µs → 기본 단계 S3/S4(321/315 · 663/650 · 911/902) 와 일치. probe 클라이언트 값(848~972 / 1,427~1,556) 은 간섭 산물로 확정. C1(bs=1) 은 CPU 게시가 63 µs 먼저(늦음 1.8 %).
- GLM: D6 게이트 4/4 후 G_OFF 42.8 tok/s(TPOT 1.09 s). 단, qwen480 토크나이저 사용 문제로 D8(G_OFF2/G_CORR2/G_OFF3) 재실행 예약. G_CORR(D6) 진행 중 → D7 → Qwen FOCUS2/S29 → D8.
- FOCUS: S25 도 -p 한계(손실 40 %). 시스템 전체 기록 FOCUS2 대기.
- 문서: READINESS·OPTIMIZATION_HANDOFF·CONFIG_DIFF·PROBE_MAP_V2·G00_code_review §8·registry TSK_060 갱신. 렌더러 §8b/§8c 추가 확인.

## 09-17 18:40 확장 단계 30분 보고
- CORE4 나머지: LONG(S27, bs=8) cold_pub_delta −181 µs(CPU 먼저)·service 246 µs; C1(S26) −63 µs. → bs 큰 디코드(63) 에서만 CPU 가 GPU 를 기다리게 함. S27 prefill(EXTEND ~4,080 tok) 층: CPU 게시가 hot 보다 약 0.5 ms 늦음(늦음 비율 0.56~0.75, service 6.6 ms, AMX 경로) — 짧은 prefill(S3/S4: CPU 1.3~3.1 ms 빠름) 과 반대 → READINESS prefill 항목 갱신 예정.
- GLM: D6 G_CORR 43.3 tok/s(트레이스 확보, CPU 레코드는 KT_EVT 미설정으로 없음 → D8 부팅에 KT_EVT 추가). D7(rsf 미수정 게이트) 실행 중 → FOCUS2/S29 → D8.
- 문서 갱신 완료: READINESS 확장 절, HANDOFF, CONFIG_DIFF, PROBE_MAP_V2, G00 §8, registry TSK_060, WORK_LOG.

(정정 18:42: 위 18:17/18:30/18:40 항목은 처음에 18:45/19:15/19:45 로 잘못 표기되었고, 'nerdctl 30 분 지연' 은 시각 오인에 따른 오판으로 철회. G_OFF 는 18:18 에 정상 시작해 382 s 걸림.)

## 09-17 18:52 확장 단계 30분 보고
- D7(rsf 미패치) 게이트 4/4 이나 전 GPU 기준(IDE_073 G-GPU8) 과 첫 토큰부터 분기; D6(두 수정) 은 4문항 중 3문항 완전 일치·1문항 129자 공통 후 분기 → GLM CPU 경로는 두 수정 모두 필요 (G00 §8-5).
- FOCUS2(S28, 시스템 전체 sched, 손실 0): 디코드 tail 31/23,808 의 창 안 워커 off-CPU 최대 284 µs, >1 ms 0 → tail 의 스케줄링 원인 기각. 수정 .so v4 회귀 없음 (703.5 tok/s). probe 클라이언트 세션은 tail 이 1,485~1,564 로 vllmw(31~44) 의 수십 배.
- prefill: 4~5k 토큰 EXTEND 스텝에서 CPU 게시 0.1~1.1 ms 늦음 (조건부 진단 항목).
- 남은 실행: S29 CORR(v4 회귀·의존성) → 분석 → GLM D8(G_OFF2/G_CORR2/G_OFF3, KT_EVT·glm47 토크나이저) → GLM 분석 → 최종 렌더·검증·게시·cron 삭제.

## 09-17 19:15 확장 단계 30분 보고
- S29 (수정 .so v4, vllmw): 689.7 tok/s, cold_pub_delta 335 µs·service 921·enq→start 673 → 부팅 4개·.so v2/v3/v4·클라이언트 vllm/vllmw 에서 대표값 재현. v4 회귀 없음.
- GLM D8: 게이트 4/4 (전 GPU 기준 접두 D6 과 동일), G_OFF2 44.1 tok/s(TPOT 1.05 s). G_CORR2/G_OFF3 진행 중. 단 GLM BASIC4 는 callback-free 경로가 아니라 CPU 기록기(poll_loop_) 가 동작하지 않아 CPU 레코드가 없음 → GLM 의존성 분석은 OPT 형 구성 여부 확인 후 결정.
- 다음: D8 종료 → (가능하면 GLM OPT 구성 1회) → 최종 렌더·검증·게시·cron 삭제·종료 보고.

## 09-17 19:37 확장 단계 30분 보고
- GLM D8 완료: G_OFF2 44.1 / G_CORR2 44.0 / G_OFF3 44.2 tok/s (TPOT 1.05 s). CPU 레코드는 BASIC4 경로 특성상 없음 → D9(callback-free env) 실행 중 (게이트 → G_CORR3 → G_OFF4).
- 다음 30분: D9 종료·분석 → 최종 렌더(FULL_REPORT §8b/§8c)·검증·게시·cron 삭제·종료 보고.

## 09-17 20:27 확장 단계 30분 보고
- GLM D9 (callback-free): 게이트 4/4, G_CORR3 44.0 / G_OFF4 43.9 tok/s (D8 과 동일 → 전달 경로 효과 없음). CPU 레코드는 확보했으나 GPU 층 분절 휴리스틱이 Qwen 그래프 전용이라 GLM 의존성은 NOT_ANALYZED (데이터 보존, 적응 방법 기록).
- 최종 검증에서 S22/S23/S27 의 clock 순서 위반(창 경계 zip 어긋남 ~4 그룹) 발견 → 순서 검사 기반 재동기화로 수정, CORR 11 세션 재분석·전 세션 재검증 중 (약 20 분). 이후 렌더·게시·cron 삭제·종료 보고.

## 09-17 20:56 확장 단계 종료 보고
- 실행: 확장 부팅 14 (Qwen 9: OFF_OPEN2/CORE2/OFF_CLOSE2/OFF_X/CORE3/OFF_Y/CORE4/FOCUS2 + ATTACH; GLM 5: D2~D9 중 세션 부팅), 세션 Qwen 22 (S8~S29) + GLM 8 (G_OFF~G_OFF4). 예산 상한 밖(사용자 승인, 원장 phase=extended).
- 결과 요약: (1) 대표 의존성 값이 부팅 4개·.so v2/v3/v4·클라이언트 vllm/vllmw 에서 재현 (cold_pub_delta 315~392 µs, enq→start 650~673, service 902~922). (2) probe 클라이언트 비동등 확정·폐기, vllm 래퍼 동등. (3) bs=1/8 디코드는 CPU 가 먼저, 4~5k 토큰 EXTEND 스텝만 CPU 0.1~1.1 ms 늦음. (4) TP 도착 시차는 CPU 대기 반영. (5) FOCUS2: 디코드 tail 은 워커 선점/off-CPU 원인 아님. (6) expert rows ≤2 가 67.7 %. (7) GLM: kt-kernel dangling-else 결함(TSK_060) + rsf 결함 수정 → 전 GPU 기준과 greedy 3/4 완전 일치, 44 tok/s; GLM 의존성 값은 GPU 층 분절 미지원으로 NOT_ANALYZED.
- 문서: FULL_REPORT §8b/§8c, READINESS(확장 절), OPTIMIZATION_HANDOFF, CONFIG_DIFF(v3/v4·복원), PROBE_MAP_V2, G00_code_review §8, MEASUREMENT_COMPLETION_STATUS, VALIDATION_REPORT, registry TSK_060, README 트리.
- 검증: 39 세션 중 20 all_pass; 19 건 불합격은 사유 특정 (WORK_LOG 20:55). 게시 후 cron 삭제.
