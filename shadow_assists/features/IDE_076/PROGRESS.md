# PROGRESS — IDE_076 (A1·A2·B 구현·검증)

## 09-17 22:49 착수 (O0)
- 완료 질문: Q0-1 현재 소스·바이너리·모듈·모델·클라이언트·manifest 출처 확인 (BASELINE_PROVENANCE.md, SOURCE_MANIFEST.json). Q0-2 횟수 guard 조사·비활성화·합성 시험 (EXECUTION_POLICY_TESTS.md). Q0-3 Cold job 실행 구조 확인: qlen=64 → forward_prefill → 스테이지 5개(gate/up 20 task/expert·activation·down 양자화·down 48 task/expert·가중 합) 각각 `do_work_stealing_job` 장벽(워커별 mutex+cv notify, 부모 spin wait, block=1 guided).
- 활성 variant/build: R0_REF (reference_binary v4 12926df2…), 부팅 진행 중 (MAIN×3, C1, LONG OFF).
- unresolved_required_questions: R0 원값·변동, ACCEPTANCE_CRITERIA 확정, CODE_MAP(A1 RB 잔여 비용·A2 호출 트리) 작성, 신규 플래그 빌드.
- 자원: GPU 0~3 부팅 중, 다른 작업 없음.

## 09-17 23:00 진행 (O0 완료 → O2/O3 착수)
- 완료 질문: Q0-4 R0_REF(v4, H0, OFF) 원값 확보 — MAIN 745.4/746.0/747.5, C1 53.2, LONG 163.0 tok/s (WORK_LOG). Q3-0 A1 asm 근거(A1_GAP_ANALYSIS §3).
- 활성: R0_REF 종료(서버 다운). CPU replay(RB ON/OFF, expert 단위 R=1/2/3) 실행 중.
- unresolved_required_questions: (1) R=1/2/3 시간 의존성(대역 한계 vs 명령 한계) → A1 이득 상한, (2) ACCEPTANCE_CRITERIA 수치 확정(R0 변동 0.3 % 기반), (3) 신규 플래그 빌드(KT_OPT_A1/A2_ENABLE) 와 flags-OFF 대조 R0, (4) A2 적용성(내부 dispatch 비용 측정).

## 09-17 23:12 진행 (O2→O3)
- 완료 질문: Q3-1 R 의존성 — R=1→2 +7 %(n160)/+15 %(n16) → W 스트리밍 한계(≈190~230 GB/s/socket), 서버 expert 당 52 µs 와 replay n16 51.9 µs 일치. Q-A2-0 dispatch 고정비 절편 ≈ 0 (<50 µs/job). RB ON/OFF 출력 bit 동일.
- 활성: PF 진단 replay(KT_AVX_PF 1/2/4) 실행 중 → 이어 A1-a(GFNI 언팩) 패치 빌드 v5 → 플래그 파서·FP8 회귀·A1 OFF/ON/PROOF replay 체인 대기 중.
- unresolved: Q3-2 A1-a 의 bitwise 동일·시간, Q-A2-1 장벽 tail 노출 비용, R0(신규 빌드 flags OFF) 대조, ACCEPTANCE 등록 완료(23:02).

## 09-17 23:15 진행 (O3 빌드 / O6 도구)
- 완료: A1-a 패치 적용(kt_opt.h 플래그 파서 0/1·proof 카운터·GFNI 언팩, ext_bindings kt_opt_status), 빌드 진행 중(23:06~). A2 장벽 tail 계측 패치 준비(빌드 v6 에 포함 예정). B 도구 `hotmap_tools.py` (validate/diff/swap) 작성, H0 동결(placements/H0, hot_sets_digest 68f74531…), 자체 시험.
- 대기 체인: a1 chain(플래그 시험·FP8 회귀·A1 OFF/ON/PROOF replay) → e2e1 chain(v6 빌드·FP8 회귀·A2 probe replay → R0(v6 OFF)·A1(v6 ON) 각 MAIN×3/C1/LONG 부팅).
- unresolved: Q3-2 A1-a bitwise·시간, Q3-3 A1 OFF MAIN vs R0, Q-A2-1 장벽 tail, Q-B-0 calibration 수집 설계(R_CPU 확정 후).

## 09-17 23:20 진행 (O3 검증 / O5 준비)
- 완료: A1-a(GFNI 언팩) v5 빌드·플래그 시험·FP8 회귀 PASS·replay: 출력 bit 동일, 시간 −0~2 %(판정 한도 아래). 기존 RB 프리페치 knob(KT_AVX_PF) 진단: n16 R=1 −12 %, R=2 −10 % → A1-e subvariant 로 E2E 비교 예정. A1_BRANCH_PROOF.json 저장. B screening 모델 v0: calibration 빈도(측정 기준선 세트) 가 현재 워크로드의 cold 사용(관측 unique 12~16 vs 모델 2~6) 과 불일치 → 대상 워크로드에서 calibration 재수집 필요(§8.2) 확인.
- 활성: v6 빌드(A1-a + A2 probe) → FP8 회귀 → A2 장벽 tail probe replay → R0(v6 OFF)·A1(v6 ON) E2E 부팅.
- unresolved: Q3-3 A1 E2E, Q3-4 A1e/A1ae E2E, Q-A2-1 tail, Q-B-1 calibration 수집(recorder API), Q-B-2 ΔG 측정 방법.

## 09-17 23:33 진행 (O3 E2E 1차)
- 완료: v6 R0/A1 부팅 각 1 (MAIN 735.3 vs 746.3, smoke 동일). v6 OFF 경로가 v4 보다 1.5 % 낮은 원인(런타임 분기) 을 템플릿·hoist 로 제거해 asm diff 0 확인 → v7 빌드 체인(e2e2) 시작: v7 → FP8 회귀 → A2 probe(스레드별 카운터) → A1e(PF=1)·A1ae(GFNI+PF=1)·R0 2차 부팅 → CALIB 수집.
- unresolved: Q3-3(A1 판정: v7 블록 ≥3 필요), Q3-4(A1e/A1ae), Q-A2-1b, Q-B-1(calibration), Q-B-2(ΔG).

## 09-17 23:45 진행 (O4 착수)
- 완료: v7 빌드(A1 템플릿·hoist·스레드별 카운터) 설치·FP8 회귀 PASS. A2 probe 재측정: 장벽 tail 3.7~4.4 % → A2 구현 결정, 패치 작성·소스 적용(빌드는 E2E 블록 종료 후).
- 활성: e2e2 체인 — A1e(PF=1) 부팅 진행 중 → A1ae → R0 2차 → CALIB 수집; 이어 e2e3 블록(R0·A1ae·A1e·A1·R0).
- unresolved: Q3-3/3-4(A1·PF E2E 블록), Q4-1(A2 bitwise·replay·E2E), Q-B-1/2.

## 09-18 00:05 진행 (O3 E2E 블록 2 / O5 calibration)
- 완료: A1e(PF=1) 부팅 MAIN 평균 759.1, A1ae(GFNI+PF=1) 부팅 진행/완료, R0 2차 부팅 진행. A2 executor 소스 적용·문법 통과(빌드 v8 은 블록 3 뒤 e2e4 체인). calibration 분석기(calib_build.py: v2 분석 → ΔG 회귀 → b_cost_model → HB 후보 생성) 작성·예약(e2e4 뒤).
- 활성 체인: e2e2(R0 b2 → CALIB 수집) → e2e3(R0·A1ae·A1e·A1·R0) → e2e4(v8: A2 replay·proof → A2/A12/R0/A12e 부팅) → calib1.
- unresolved: Q3-3/3-4 블록 ≥3 판정, Q4-1 A2 검증, Q-B-1/2 calibration·ΔG, SELECTION/CONFIRMATION 계획.

## 09-18 00:13 진행 (사고 대응 / O5)
- 사고: v7 부팅 2 건(A1ae, R0 b2) 이 두 번째 MAIN 중 NaN 확률 assert 로 서버 사망 (WORK_LOG 00:08). 같은 v7 의 A1e·CALIB(CORR 2 세션) 부팅은 정상. 사망 부팅 세션은 성능 통계에서 제외(invalid_reason=server_died_in_boot). e2e3 첫 R0(v7) 부팅을 재현 시험으로 진행 중; 재현 시 v7 구성요소 이분.
- 완료: CALIB 수집(recorder .pt 4 개 + CORR 계측, 618.9/622.4 tok/s). v2 분석·비용표·후보 생성은 e2e4 뒤(calib1 체인).
- unresolved: 사고 원인, A1/PF/A2 E2E 판정(블록 3~5), B 후보·SELECTION.

## 09-18 00:40 진행 (사고 대응 계속)
- 사고 상태: known-good v4 부팅도 M3 에서 같은 NaN assert 로 사망(4 번째) → 바이너리 무관, 23:45 이후 환경성. 런타임 파일 SHA·per-layer budget·GPU ECC/remap 이상 없음. crash 는 C64 웨이브 경계에서 발생. 조치: GPU reset 은 Fabric Manager 로 거부됨 → 페이지 캐시 drop 후 v4 sanity 부팅(ENVCHK2) 진행 중. 재발 시 E2E 부팅은 BLOCKED_ENVIRONMENT 로 두고 오프라인 항목(A2 빌드·replay 검증, B 후보 문서) 을 계속.
- E2E 유효 결과(현재): R0_REF(v4) 746.3 / R0(v6) 735.3 / A1(v6) 746.3 / A1e(v7) 759.1 (각 부팅 1, 3 세션); 사망 부팅 세션은 제외.
- unresolved: 사고 원인·복구, 블록 판정(A1/A1e/A1ae/A2/B), B SELECTION.

## 09-18 01:07 진행 (사고 원인 분리 → 완화 패치 시험)
- 완료: 원인 분리 — 기존 callback 경로(KT_CALLBACK_FREE 해제) 진단 부팅은 MAIN 3 세션 생존(657.6/655.0/659.8, callback-free 대비 −12 %). 사고는 callback-free 전달 경로의 디코드 웨이브 종료→빈 배치→prefill(eager) 전환 경쟁이며 바이너리·A1/A2/PF 무관.
- 조치: eager 스텝 layer 0 진입 시 `cpu_infer.sync()` 완화 패치(`eval/ide076/cf_transition_sync_patch.py`, 파이썬 래퍼만 수정, 그래프 경로 불변) 적용. 재현 시험 CFFIX_v4 MAIN3 ×2 부팅 진행 중: 부팅 1 M1 736.2 / M2 743.9 tok/s 생존, M3 실행 중(이전 사망은 대부분 M3·웨이브 경계).
- 활성 variant/build: CFFIX_v4 (설치 .so v4 12926df2… + 래퍼 수정 79cfa0c6…). 소스는 v8 후보(A2 포함) 상태, 미빌드.
- 병행: B calibration v1 분석(`calib_build.py`, CAL1/CAL2 v2 분석) 호스트 nice 19 로 실행 중.
- 마지막 유효 결과: R0_REF(v4) 746.3 / R0(v6) 735.3 / A1(v6) 746.3 / A1e(v7) 759.1 (사망 부팅 세션 제외). 수정 후 결과는 별도 모집단(§14).
- 자원: GPU0-3 77 GB 사용(서버 가동), load 78. 오류 없음(현 부팅).
- unresolved_required: 완화 패치 유효성(CFFIX 2 부팅 무사망), 수정 스택 R0 기준선 재측정, A1/A1e/A1ae/A2 블록 판정, v8 빌드·A2 replay, B SELECTION/CONFIRMATION, 통합 결정·게시(사용자 승인).

## 09-18 01:32 진행 (근본 원인 수정 → 검증 부팅)
- 완료: (1) 첫 완화(eager layer 0 sync)는 CFFIX 부팅 2 에서 같은 사망 → 폐기. (2) 근본 원인 특정: overlap 스케줄러의 호스트 선행 + callback-free eager 패킷 rearm 무대기 + row 수별 pinned 버퍼 → '작은 prefill→8,192 청크' 전환에서 이전 go 가 다음 스텝 args 로 실행(사망 7 건 로그 패턴 일치). (3) 수정 `kt_cf_rearm_fix_patch.py`(C++ 슬롯 세대 대기 + Python 슬롯 링 2) → v8f 빌드·설치(a4add14d…), 플래그 검사·FP8 회귀(0.5829 %, PASS) 통과. (4) calibration v1 분석 완료: 모델 unique cold 13.3 vs 관측 14.8 정합, ΔG 미검출(r² 0.005), 단일 교환 후보 12 개는 0.1 % 로 제외, HB_DEC1/MIX1 SELECTION 예정. (5) 모집단 분리(pre_fix/sync_only/post_fix) 도구·manifest 반영, ROLLBACK.md 초안.
- 활성: fixtest 체인 — R0(v8f OFF) MAIN3_C1_LONG ×3 부팅 중(f1 부팅·워밍업 단계). post2 체인 무장(무사망 확인 시 v4f 빌드 → A2 replay → 블록 p1 → B SELECTION → p2 → p3, ≈ 26 부팅).
- 마지막 유효 결과: pre_fix R0_REF(v4) 746.3 / R0(v6) 735.3 / A1(v6) 746.3 / A1e(v7) 759.1; sync_only CFFIX_v4 부팅 1 736.2/743.9/770.8 (참고). post_fix 결과 아직 없음.
- 오류·복구: post 체인 v1 이 게이트 오판으로 v8(수정 없음) 빌드 시작 → 수동 중단, 설치 안 됨(기록). 현재 오류 없음.
- 자원: GPU0-3 79 GB, util 51~100 %(부팅/워밍업), load 69. 호스트 분석 프로세스 없음.
- unresolved_required: 수정 유효성(3 부팅 무사망), post_fix R0·R0_REF(v4f) 기준선, A1/A1e/A1ae/A2/A12/A12e 판정(≥3 블록), A2 replay bitwise, B SELECTION/CONFIRMATION, 통합 결정·게시(사용자 승인).

## 09-18 02:00 진행 (수정 검증 실패 → 진단 체인 2)
- 완료: fixtest 3 부팅 중 f1·f2 완주(MAIN 732~753, C1 63, LONG 163~166), **f3 는 같은 패턴으로 사망**(웨이브 종료 → 3,557 tok prefill → 8,192 청크에서 NaN) → CF-rearm 수정은 불충분. rearm 대기 경고 0 건. post2 체인은 게이트가 정상 차단.
- 새 사실: 커널 Xid 43 이 9-16 01:30 / 9-16 07:38 / 9-17 08:56 에도 4 GPU 동시 발생 → IDE_076 이전부터 있던 잠재 결함이며 현재 발생률(전환당 ≈0.07)이 높아진 상태. 새 pinned 버퍼 할당 여부는 변별력 없음. GPU 리타이어/리셋 필요 없음, 컨테이너 좀비 4,593 개(자원 미점유).
- 활성: 진단 체인 2 (v8f OFF, 각 MAIN3 ×3 = 36 전환): D0 기존 callback 경로 → D1 `--disable-overlap-schedule` → D2 SKIP_EMPTY_IMM 해제. D0 부팅 1 진행 중(워밍업). 러너에 `--sarg` 추가.
- 마지막 유효 결과: post_fix R0(v8f) f1/f2 = MAIN 732.1/743.4/748.4, 735.3/748.8/752.9; C1 63.5/63.1; LONG 166.5/163.1 (사망 f3 제외). pre_fix 값은 이전 항목.
- 오류·복구: f3 사망(기록·제외). 현재 오류 없음.
- 자원: GPU0-3 77 GB(부팅 중), load 23.
- unresolved_required: 사고 재현 경로 특정(D0/D1/D2 결과), 근본 수정 또는 회피 구성 확정 → post_fix 기준선·블록·B SELECTION·CONFIRMATION·결정·게시(사용자 승인).

## 09-18 02:25 진행 (진단 체인 2 중간)
- 완료: D0 기존 callback 경로 3 부팅 36 전환 무사망(앞선 진단 포함 48 전환 0 사망; 전환당 0.07 기준 우연 확률 ≈3 %) → callback-free 경로 결함 판정 강화. MAIN 평균 ≈642 (callback-free 대비 −13~14 %). D1 overlap 해제 d1 무사망(MAIN 699~710, −5~6 %).
- 활성: D1 d2 부팅 시작(v8f OFF + `--disable-overlap-schedule`), 이후 d3 → D2(SKIP_EMPTY_IMM 해제) ×3. 남은 5 부팅 ≈ 35 분.
- 마지막 유효 결과(post_fix R0 v8f): f1/f2 MAIN 732~753, C1 63, LONG 163~166. 진단 구성 값은 성능 판정에 쓰지 않음.
- 오류·복구: 없음(이번 구간). 자원: GPU 4 MiB(부팅 전환 중), load 44.
- unresolved_required: D1/D2 결과 → 회피·수정 경로 확정(후보: overlap 해제 −5 %, legacy −14 %, 또는 callback-free 내부 재수정) → post_fix 기준선·블록·B SELECTION·CONFIRMATION·결정·게시(사용자 승인).

## 09-18 03:01 진행 (진단 체인 2 완료 → NaN 발생 지점 프로브)
- 완료: 진단 체인 2 — 기존 callback 경로 48 전환 0 사망(MAIN ≈642), overlap 해제 36 전환 0 사망(≈704), 빈 imm 생략 해제 36 전환 0 사망(≈651). 사고는 callback-free ∧ overlap ∧ 빈 imm 생략 세 조건이 모두 있을 때만 발생(기본 구성 ≈100 전환에 7 사망). 코드 정독(TaskQueue 순차성, 폴러 go 순서, S6 전 행 기록, worker_pool 장벽·NUMA join, 호스트 NUMA 2 노드) 으로는 결함 미발견.
- 활성: NaN 발생 지점 프로브(`kt_nan_probe_patch.py`, 호스트 동기화 없는 비동기 기록) 를 적용한 사고 구성 부팅 PROBE_nan p1 진행 중(워밍업). 최대 4 부팅, 사망 시 프로브 로그 확보 후 원복. 설치 바이너리 v8f.
- 마지막 유효 결과: post_fix R0(v8f) f1/f2 MAIN 732~753, C1 63, LONG 163~166. 진단 구성 값은 회피 비용 참고.
- 오류·복구: 없음(이번 구간). 자원: GPU 79~81 GB(부팅 중), load 247(가중치 적재).
- unresolved_required: NaN 기원(CPU 출력 vs MoE 입력)·층 위치 → 근본 수정 또는 회피 구성(overlap 해제 −5 % / 생략 해제 −12 % / legacy −14 %) 확정 → post_fix 기준선·블록·B SELECTION·CONFIRMATION·결정·게시(사용자 승인).

## 09-18 03:25 진행 (프로브 재현 체인 3)
- 완료: 프로브 부팅 두 차례 OOM 의 원인 특정·수정 — (1) isnan/isinf 임시 텐서 → amax/amin 축소로 교체, (2) import 시점 pinned 할당이 각 rank 를 GPU 0 컨텍스트(≈520 MiB×3)로 묶어 TP0 여유 −2 GB → 첫 사용 시 지연 초기화. r1 부팅은 OOM 없이 완주(MAIN 723/748/748, 프로브 부하 무시 가능, 12 전환 무사망).
- 활성: PROBE_nan r2 진행 중(사고 구성 v8f OFF + KT_NANPROBE=1; 최대 r4, 사망 시 프로브 로그 확보 후 원복).
- 마지막 유효 결과: post_fix R0(v8f) f1/f2 MAIN 732~753, C1 63, LONG 163~166. 진단 체인 2 결론 유지(callback-free ∧ overlap ∧ 빈 imm 생략에서만 사고).
- 오류·복구: 프로브 OOM 2 회(원인 수정, 성능 데이터 아님). 자원: GPU 79 GB, util 100 %(세션 중), load 62.
- unresolved_required: 프로브 로그(첫 NaN 위치: CPU 출력 vs 입력, 층·rows·step) → 근본 수정 또는 회피 구성 확정 → post_fix 기준선·블록·B SELECTION·CONFIRMATION·결정·게시(사용자 승인).

## 09-18 03:55 진행 (프로브 체인 3~5)
- 완료: 인-스트림 프로브 4 부팅(r1~r4) 48 전환 0 사망 → 측정 커널이 forward 스트림을 층당 ≈0.4 ms 늦춰 경쟁을 가림(측정 부하가 회피책이 되는 수준). 별도 스트림 프로브 첫 판(s1)은 record_stream 로 인한 정점 메모리 증가로 OOM → record_stream 제거·한 층 뒤 join 으로 수정(메모리 증가 96 MB, 검출 검증 통과). 비순차 저장(NT store) 가설은 출력 경로가 일반 저장이라 기각.
- 활성: 프로브 체인 5 t1 진행 중(M1 740.4, M2 777.1, M3 진행; 사고 구성 v8f OFF + side 프로브). 최대 t6, 사망 시 프로브 로그 확보 후 원복.
- 마지막 유효 결과: post_fix R0(v8f) f1/f2 MAIN 732~753, C1 63, LONG 163~166. 진단 체인 2 결론 유지.
- 오류·복구: 프로브 OOM 3 회(모두 프로브 구현 문제, 수정 완료). 자원: GPU 77 GB(세션 사이), load 62.
- unresolved_required: 프로브 로그(첫 NaN 위치) 또는 재현 실패 시 회피 구성 결정 → post_fix 기준선·블록·B SELECTION·CONFIRMATION·결정·게시(사용자 승인).

## 09-18 04:30 진행 (NaN 기원 특정 → 재계산 프로브)
- 완료: 프로브 v3 가 사고를 재현·판독 — 스텝 69(8,192 행) 층 0 에서 입력·CPU 출력은 정상, **GPU hot-expert(fused_moe FP8) 출력에 NaN** → 결합 → 다음 층 입력 672 행 NaN 으로 전파. 앞선 재현(층 11 입력 첫 검출)과 정합. **NaN 기원은 CPU 핸드오프·KT 커널·A1/A2 가 아니라 SGLang GPU fused_moe 경로.** GPU 경로 정독(−1 expert 처리, 정렬 버킷, 중간 버퍼, 층별 예산 적용 여부) 으로는 결정적 결함 미발견. 층별 예산 패치는 하네스가 부팅마다 적용·원복함을 확인(B 유효성 문제 없음).
- 활성: 프로브 v4 체인 7 v1 부팅 시작(사고 구성 v8f OFF; gpu_nan 감지 시 입력 복제본으로 fused_moe 2 회 재계산 + 입력 덤프 /tmp/ide076_nan_dump_*.pt). 최대 6 부팅.
- 마지막 유효 결과: post_fix R0(v8f) f1/f2 MAIN 732~753, C1 63, LONG 163~166. 진단 체인 2 결론 유지(세 조건 조합에서만 사고).
- 오류·복구: 프로브 OOM 3 회(모두 수정). 자원: GPU 4 MiB(부팅 직전), load 22.
- unresolved_required: 재계산 결과(결정적 결함 vs 일시적 경쟁) → GPU 경로 수정 또는 회피 구성 확정 → post_fix 기준선·블록·B SELECTION·CONFIRMATION·결정·게시(사용자 승인).

## 09-18 04:55 진행 (재계산 프로브 체인 7)
- 완료: 프로브 v4(입력 복제 링 + gpu_nan 시 재계산) 부팅 v1~v3 무사망(36 전환, MAIN 724~741) → forward 스트림의 100 MB 복제(≈50 µs)가 경쟁을 가릴 가능성 → 프로브 v5(제출 시점 side 스트림 복제, fused_moe 직전 대기만; 단위 검증 통과) 를 적용해 남은 부팅에 사용. 단위 시험 중 실행 중 체인의 프로브를 약 1 분 원복한 실수가 있었으나 그 사이 새 부팅 없음(WORK_LOG 04:48).
- 활성: 체인 7 v4 부팅(프로브 v4) — M1 739.5 완료 후 M2 진행 중이며 04:54 현재 GPU1~3 메모리가 해제되어 서버 사망(재현) 가능성 있음. 종료 시 재계산 로그(`KT-NANPROBE-RECOMPUTE`)·입력 덤프 확인 예정. 남은 부팅 v5·v6 은 프로브 v5.
- 마지막 유효 결과: post_fix R0(v8f) f1/f2 MAIN 732~753, C1 63, LONG 163~166.
- 오류·복구: 없음(이번 구간; 프로브 원복 실수는 영향 없음). 자원: GPU0 79.5 GB / GPU1~3 ≈1 GB(전환 중), load 313(가중치 적재 또는 종료 처리).
- unresolved_required: 재계산 결과(결정적 결함 vs 일시적 경쟁) → GPU fused_moe 경로 수정 또는 회피 구성 확정 → post_fix 기준선·블록·B SELECTION·CONFIRMATION·결정·게시(사용자 승인).

## 09-18 05:25 진행 (GPU 측 캡처 프로브)
- 완료: 체인 7 v4 에서 3 번째 재현(스텝 66 층 0 gpu_nan) — 그러나 호스트 폴링이 GIL 에 밀려 한 스텝 뒤에 깨어나 입력 링이 덮임(ring miss). → 프로브 v6: GPU 위에서 `where` 로 첫 gpu_nan 층의 입력(x·ids·w·층·스텝)을 조건부 캡처해 지연과 무관하게 재계산·덤프(단위 검증 통과). v6 는 부하(층당 ≈0.9 ms 상당 side 트래픽) 로 MAIN −7 %·3 부팅 무사망(가림) → v6.2 경량(캡처 층 0~12, GPU 출력 검사만) 적용.
- 활성: 체인 8 w4 부팅(M1 693.9, M2 699.2 진행) → w5·w6 은 v6.2; 재현 없으면 체인 9(x1~x8, v6.2) 자동. 재현 시 `KT-NANPROBE-RECOMPUTE` 로그(재계산 NaN 재현 여부)와 `/tmp/ide076_nan_dump_*.pt` 확보.
- 마지막 유효 결과: post_fix R0(v8f) f1/f2 MAIN 732~753, C1 63, LONG 163~166. 프로브 부팅 값은 성능 판정에 쓰지 않음.
- 오류·복구: 없음(이번 구간). 자원: GPU 77 GB(세션 사이), load 69.
- unresolved_required: 재계산 판독(결정적 vs 일시적) → GPU fused_moe 경로 수정 또는 회피 구성 확정 → post_fix 기준선·블록·B SELECTION·CONFIRMATION·결정·게시(사용자 승인).

## 09-18 05:55 진행 (경량 캡처 프로브 재현 대기)
- 완료: 체인 8 종료(w1~w6 무사망; w5·w6 은 경량 v6.2 로 처리량 정상 725~771 → 측정 부하가 타이밍을 바꾸지 않음). 체인 9 x1·x2 무사망(경량 프로브 누적 48 전환).
- 활성: 체인 9 x3 부팅 진행(사고 구성 v8f OFF + v6.2: 층 0~12 입력을 GPU 위에서 조건부 캡처, gpu_nan 감지 시 재계산·덤프). x8 까지 자동, 재현 시 중단·원복.
- 마지막 유효 결과: post_fix R0(v8f) f1/f2 MAIN 732~753, C1 63, LONG 163~166. 재현 3 회(035604·040906·044802) 모두 GPU hot-expert 출력이 첫 NaN.
- 오류·복구: 없음(이번 구간). 자원: GPU 79~80 GB, util 64~100 %(세션 중), load 47.
- unresolved_required: 재계산 판독(결정적 vs 일시적) → GPU fused_moe 경로 수정 또는 회피 구성 확정 → post_fix 기준선·블록·B SELECTION·CONFIRMATION·결정·게시(사용자 승인).

## 09-18 06:25 진행 (경량 프로브 미재현 → 대조 부팅 예약)
- 완료: 체인 9 x1~x7 무사망(경량 v6.2 누적 108 전환, MAIN 717~771 정상 처리량). p≈0.07 기준 우연 확률 <0.1 % → 경량 프로브도 타이밍을 바꾸거나, 사고 발생률 자체가 낮아졌음. 구분용 대조 실험(프로브 없는 사고 구성 MAIN3 ×4, CTRL_noprobe) 을 체인 9 종료 후 자동 실행하도록 예약.
- 활성: 체인 9 x8 부팅(마지막). 이어서 대조 부팅 c1~c4(사망 시 중단).
- 마지막 유효 결과: post_fix R0(v8f) f1/f2 MAIN 732~753, C1 63, LONG 163~166. 재현 3 회(v3/v4 프로브) 모두 GPU hot-expert 출력이 첫 NaN; GPU 캡처 프로브로는 아직 재현 없음.
- 오류·복구: 없음(이번 구간). 자원: GPU 77 GB(세션 사이), load 46.
- unresolved_required: (1) 대조 결과 — 사망하면 프로브 가림(프로브 추가 축소 필요), 무사망이면 발생률 저하(환경 요인 추정) → 이 경우 사고 구성 그대로 post_fix 기준선·블록 재개 가능 여부 판단; (2) GPU fused_moe 경로 결함의 근본 원인(재계산 판독) 또는 회피 구성 확정; (3) post_fix 기준선·블록·B SELECTION·CONFIRMATION·결정·게시(사용자 승인).

## 09-18 06:55 진행 (지시서 비교 재개)
- 사용자 지시(06:40): 사망 부팅은 실패로 처리. 근본 원인 추적(지시서 §17.2 범위 밖)은 중단하고 원래 구성으로 비교 재개. 대조 부팅 c1(프로브 없음) 은 같은 패턴으로 사망(발생률 유지, 경량 프로브가 가렸음) → 기록만.
- 완료: v4f 기준 바이너리(v4 소스 SHA 일치 + rearm 수정) 빌드·검증. A2 replay(v8f): A2 ON·A1+A2 ON 모두 OFF 와 bitwise 동일; 시간 p50 큰 job −2 %(A2) / −1.7~3.4 %(A1+A2), 작은 job 변화 없음; proof 적격 142/142 묶음, fallback 0, 장벽 tail 2.8 %.
- 활성: post3 체인 블록 p1 — R0_REF(v4f) 부팅 진행(M1 735.1, M2 741.8). 이후 R0/A1ae/A1e/A1/A2/A12/A12e → p2 → B SELECTION ×2 → p3~p6. 사망 부팅은 server_died 로 기록·제외.
- 마지막 유효 결과: post_fix R0(v8f) f1/f2 MAIN 732~753, C1 63, LONG 163~166.
- 오류·복구: 없음(이번 구간). 자원: GPU 79 GB, util 60~100 %(세션 중), load 84.
- unresolved_required: 유효 블록 ≥3 확보 후 A1/A1e/A1ae/A2/A12/A12e 판정, B SELECTION → CONFIRMATION → B_DECISION, 통합 결정·회귀(C1/LONG)·원복 시험·보고·게시(사용자 승인).

## 09-18 07:25 진행 (E2E 블록 p1)
- 완료: p1 R0_REF(v4f) 735.1/741.8/748.5, C1 62.9, LONG 166.3 · R0(v8f OFF) 732.5/747.8/743.8, C1 63.5, LONG 166.6 · A1ae(GFNI+PF) 769.7/772.4/767.9(R0 대비 +3.9 %), C1 63.1, LONG 165.6 — 모두 완주. **A1e(PF=1) p1 은 M2 에서 사망 → 실패 처리·제외**(지시). 새 바이너리 OFF 경로는 기준 v4f 와 같은 범위.
- 보강: 부팅별 실효 플래그를 서버 로그에 남기는 `[kt-opt] status` 출력(성능 경로 무관) 07:22 적용.
- 활성: p1 A1 부팅 진행 중 → A2 → A12 → A12e → p2 → B SELECTION ×2 → p3~p6.
- 마지막 유효 결과(post_fix): 위 p1 값 + R0 f1/f2. A2 replay: bitwise 동일, 큰 job −2 %.
- 오류·복구: A1e p1 서버 사망(NaN, 기록·제외). 자원: GPU 79 GB, util 58~100 %, load 38.
- unresolved_required: 유효 블록 ≥3 (변형별) → A1/A1e/A1ae/A2/A12/A12e 판정, B SELECTION → CONFIRMATION → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 07:55 진행 (블록 p1 종료)
- 완료: 블록 p1 (부팅 각 1, MAIN 평균 / R0 p1 대비): R0_REF 741.8 (+0.1 %) · R0 741.4 · A1 761.0 (+2.6 %) · A1ae 770.0 (+3.9 %) · A2 772.9 (+4.2 %) · A12 819.3 (+10.5 %) · A12e 791.1 (+6.7 %); A1e 사망(제외). C1 62.9~63.7, LONG 163.2~167.5, TTFT p95 비 0.98~1.01, TPOT p95 비 0.88~1.01 — 회귀 한도 내. 실효 플래그 로그(A2/A12/A12e 부팅에서 a1/a2 effective 확인).
- 활성: 블록 p2 (R0 → A12e → A12 → A2 → A1 → A1e → A1ae) 시작. 이후 B SELECTION ×2 → p3~p6.
- 마지막 유효 결과: 위 p1 표. 판정은 변형별 유효 블록 ≥3 이후.
- 오류·복구: A1e p1 사망(기록·제외). 자원: GPU 77 GB(부팅 전환), load 32.
- unresolved_required: 유효 블록 ≥3 확보 → A1/A1e/A1ae/A2/A12/A12e 판정(특히 A12 vs A12e 순서 확인), B SELECTION → CONFIRMATION → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 08:25 진행 (블록 p2 중)
- 완료(p2): R0 743.6 (731.5/744.8/754.6), C1 63.5, LONG 163.6 · A12e 803.7 (+8.1 % vs R0 p2), C1 63.1, LONG 164.2 · **A12 M1 사망(제외)** · A2 788.5 (774.8/780.1/810.6, +6.0 %), C1 63.5, LONG 167.0. 비교기 짝 비교를 '같은 블록의 R0' 기준으로 변경.
- 활성: p2 A1 부팅 진행 → A1e → A1ae → B SELECTION(H0/DEC1/MIX1 ×2) → p3~p6.
- 누적 유효 부팅: R0_REF 1, R0 4, A1 1, A1ae 1, A2 2, A12 1, A12e 2, A1e 0. 사망: f3(R0), A1e p1, A12 p2.
- 오류·복구: A12 p2 사망(기록·제외). 자원: GPU 부팅 전환 중(72 GB 적재), load 28.
- unresolved_required: 변형별 유효 블록 ≥3 → A1/A1e/A1ae/A2/A12/A12e 판정, B SELECTION → CONFIRMATION → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 08:55 진행 (블록 p2 종료 → B SELECTION)
- 완료(p2 나머지): A1 747.2 (+0.5 % vs R0 p2), C1 63.6, LONG 164.3 · A1e 769.4 (+3.5 %), C1 63.3, LONG 165.4 · A1ae M1 사망(제외). p2 요약: R0 743.6 / A12e 803.7 / A2 788.5 / A1 747.2 / A1e 769.4; A12·A1ae 사망.
- B SELECTION 시작(v8f OFF, seed 20261011, 세션 2): H0 s1 = 634.2 / 656.5 tok/s(SELECT 워크로드는 MAIN 과 다름, 후보 간 비교 전용).
- 활성: B_DEC1_SEL s1 부팅 → MIX1 s1 → H0/DEC1/MIX1 s2 → 블록 p3~p6.
- 누적 유효 부팅: R0_REF 1, R0 4, A1 2, A1e 1, A1ae 1, A2 2, A12 1, A12e 2. 사망 5(R0 f3, A1e p1, A12 p2, A1ae p2 + 대조 c1).
- 오류·복구: A1ae p2 사망(기록·제외). 자원: GPU 77 GB, util 9~100 %(세션 중), load 33.
- unresolved_required: 변형별 유효 블록 ≥3 → A 판정, B SELECTION 결과 → CONFIRMATION → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 09:25 진행 (B SELECTION 완료 → 블록 p3)
- 완료: B SELECTION(배치당 2 부팅×2 세션): H0 645.2 / HB_DEC1 504.1 (−21.9 %, TTFT 2.1 배) / HB_MIX1 682.3 (+5.7 %, TTFT 동등·TPOT −9 %) → **HB_MIX1 선별, DEC1 기각**. B 수치 기준(§8.10) 사전 등록: PPL 상대차 최대 ≤3 %, |Δlogprob| 평균 ≤0.05. runner 에 CONFIRM 플랜(고정 입력 + 새 seed 20261021 IND×2)과 수치 프로브 훅 추가, post4 체인(R0C/B/A12B/A2B/A1B ×3) 무장. p3 R0_REF(v4f) 765.7 완주.
- 활성: p3 R0 부팅 → A1ae → A1e → A1 → A2 → A12 → A12e → p4~p6 → post4(CONFIRMATION).
- 누적 유효 부팅: R0_REF 2, R0 4, A1 2, A1e 1, A1ae 1, A2 2, A12 1, A12e 2; B SEL 6/6.
- 오류·복구: 없음(이번 구간). 자원: GPU 72 GB(적재 중), load 34.
- unresolved_required: 변형별 유효 블록 ≥3 → A 판정, B CONFIRMATION(수치·독립 입력·A×B) → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 09:55 진행 (블록 p3 중)
- 완료(p3): R0_REF 765.7 (C1 63.7, LONG 165.3) · R0 751.9 (C1 63.4, LONG 161.4) · A1ae M3 사망(제외; A1ae 유효 1, 사망 2) · A1e 765.4 (+1.8 % vs R0 p3; C1 63.2, LONG 163.8).
- 활성: p3 A1 부팅 → A2 → A12 → A12e → p4~p6 → post4 CONFIRMATION.
- 누적 유효 부팅: R0_REF 2, R0 5, A1 2, A1e 2, A1ae 1, A2 2, A12 1, A12e 2. 사망 6 건 제외.
- 오류·복구: A1ae p3 사망(기록·제외). 자원: GPU 79 GB, util 100 %, load 76.
- unresolved_required: 변형별 유효 블록 ≥3 → A 판정, B CONFIRMATION → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 10:25 진행 (블록 p3 종료 → p4)
- 완료(p3): R0_REF 765.7 · R0 751.9 · A1e 765.4 (+1.8 %) · A1 768.8 (+2.3 %) · A2 786.0 (+4.5 %) · A12 785.7 (+4.5 %); A1ae·A12e 사망(제외).
- 누적(같은 블록 R0 대비 짝 이득): A1 3 블록 +2.6/+0.5/+2.3 (평균 +1.9 %) · A1e 2 블록 +3.5/+1.8 · A1ae 1 블록 +3.9 · A2 3 블록 +4.3/+6.0/+4.5 (평균 +5.1 %) · A12 2 블록 +10.5/+4.5 · A12e 2 블록 +6.7/+8.1 · R0_REF 2 블록 +0.1/+1.8. C1·LONG·TTFT p95·TPOT p95 모두 한도 내. 사망 제외 7 부팅.
- 활성: 블록 p4 (R0 → A12e → A12 → A2 → A1 → A1e → A1ae) 시작. 이후 p5·p6 → post4 CONFIRMATION(R0C/B/A12B/A2B/A1B ×3, 수치 프로브).
- 오류·복구: A12e p3 사망(기록·제외). 자원: GPU 77 GB(부팅 전환), load 49.
- unresolved_required: A1e/A1ae/A12/A12e 유효 블록 ≥3 → A 판정(A1 은 3 블록 확보, 평균 +1.9 % 로 기준 2 % 경계), B CONFIRMATION → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 10:55 진행 (블록 p4 중)
- 완료(p4): R0 739.2 (C1 63.1, LONG 162.0) · A12e 808.9 (+9.4 %) · A12 781.8 (+5.8 %) · A2 776.4 (+5.0 %) — 모두 완주, NaN 0.
- 누적 유효 블록(짝 이득 %): A1 3 (+2.6/+0.5/+2.3) · A1e 2 (+3.5/+1.8) · A1ae 1 (+3.9) · A2 4 (+4.3/+6.0/+4.5/+5.0) · A12 3 (+10.5/+4.5/+5.8) · A12e 3 (+6.7/+8.1/+9.4) · R0_REF 2 (+0.1/+1.8). 회귀 항목 한도 내.
- 활성: p4 A1 부팅 → A1e → A1ae → p5 → p6 → post4 CONFIRMATION.
- 오류·복구: 없음(이번 구간). 자원: GPU 72 GB(적재 중), load 27.
- unresolved_required: A1e·A1ae 유효 블록 ≥3 → A 판정 완결(A2/A12/A12e/A1 은 3 블록 확보), B CONFIRMATION → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 11:25 진행 (블록 p4 종료 → p5)
- 완료(p4): A1e 768.3 (+3.9 %) 완주 · A1ae M2 사망(제외; A1ae 4 부팅 중 3 사망). p4 요약: R0 739.2 / A12e 808.9 / A12 781.8 / A2 776.4 / A1 745.8 / A1e 768.3.
- 누적 짝 이득(유효 블록): A1 +1.6 % (4; 기준 2 % 미만) · A1e +3.2 % (3) · A1ae +3.5 % (1) · A2 +5.0 % (4) · A12 +7.0 % (3) · A12e +7.7 % (3) · R0_REF +1.3 % (2). 모든 짝 양수, 회귀 항목 한도 내.
- 활성: p5 R0_REF(v4f) 부팅 적재 중 → R0 → A1ae → A1e → A1 → A2 → A12 → A12e → p6 → post4 CONFIRMATION.
- 오류·복구: A1ae p4 사망(기록·제외). 자원: GPU0 79 GB, GPU1~3 적재 시작 전(1 GB), load 292(가중치 적재).
- unresolved_required: A1ae 유효 블록 ≥3(현재 1) 외 A 판정은 충족; B CONFIRMATION → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 11:55 진행 (블록 p5 중)
- 완료(p5): R0_REF(v4f) M1 사망(제외; 기준 바이너리도 같은 사고) · R0 744.3 (C1 63.3, LONG 166.7) · A1ae 773.7 (+4.0 %) · A1e 767.8 (+3.2 %).
- 누적 유효 블록(짝 이득 %): A1 4 (+2.6/+0.5/+2.3/+0.9) · A1e 4 (+3.5/+1.8/+3.9/+3.2) · A1ae 2 (+3.9/+4.0) · A2 4 (+4.3/+6.0/+4.5/+5.0) · A12 3 (+10.5/+4.5/+5.8) · A12e 3 (+6.7/+8.1/+9.4) · R0_REF 2.
- 활성: p5 A1 부팅(적재 중) → A2 → A12 → A12e → p6 → post4 CONFIRMATION. 사용자 문의에 p5·p6 단축 선택지를 제시했으며 지시 없으면 계획대로 진행.
- 오류·복구: R0_REF p5 사망(기록·제외). 자원: GPU0 79 GB, GPU1~3 적재 전, load 338(적재).
- unresolved_required: A1ae 유효 블록 3(현재 2), B CONFIRMATION → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 12:25 진행 (블록 p5 종료 → p6)
- 완료(p5): R0 744.3 · A1ae 773.7 (+4.0 %) · A1e 767.8 (+3.2 %) · A12 775.2 (+4.2 %, LONG 160.5 = −3.7 %, 한도 내) · A12e 796.2 (+7.0 %); R0_REF·A1·A2 사망(제외).
- 누적 유효 블록: A1 4, A1e 4, A1ae 2, A2 4, A12 4, A12e 4, R0_REF 2, R0 6. 짝 이득 평균: A1 +1.6 / A1e +3.1 / A1ae +4.0 / A2 +5.0 / A12 +6.3 / A12e +7.6 %.
- 활성: 블록 p6 (R0 → A12e → A12 → A2 → A1 → A1e → A1ae, 마지막) 적재 시작 → post4 CONFIRMATION(R0C/B/A12B/A2B/A1B ×3, 수치 프로브).
- 오류·복구: p5 사망 3 건(기록·제외). 자원: GPU 72 GB(적재), load 31.
- unresolved_required: A1ae 유효 블록 3(현재 2), B CONFIRMATION → B_DECISION, 통합 결정·회귀·원복 시험·보고·게시(사용자 승인).

## 09-18 12:55 진행 (E2E 종료 → CONFIRMATION)
- 완료: E2E(post3) 종료. 최종 유효 블록·짝 이득 평균: A1 4 (+1.6 %) 불채택 · A1e 4 (+3.1 %) 채택 · A1ae 3 (+3.7 %) 채택 · A2 4 (+5.0 %) 채택 · A12 5 (+5.9 %) · A12e 5 (+7.7 %) · R0_REF 2 (+1.3 %, OFF 경로 이상 없음). 사망 12 부팅 제외. A1_DECISION·A2_DECISION 확정, task.md C02/C07/C10 완료 표기. 사용자 시간 제약으로 p6 은 A1ae 만, CONFIRMATION 은 A1B 제외(A1 단독 기준 미달로 조합 질문 없음).
- 활성: CONFIRMATION(post4 축소판) c1 — R0C(H0 기준) 부팅 적재 중 → B → A12B → A2B → c2 → c3 (12 부팅, 각 MAIN×3/C1/LONG + 새 seed IND×2 + 수치 프로브). 약 1 시간 40 분.
- 오류·복구: 없음(이번 구간). 자원: GPU 72 GB(적재), load 32.
- unresolved_required: B CONFIRMATION(고정 입력·독립 입력·수치 기준) → B_DECISION, INTEGRATION_DECISION(A12e vs A12B/A2B), 회귀(C1/LONG 수집 완료)·원복 시험·COMPLETION_STATUS·FULL_REPORT·게시(사용자 승인).

## 09-18 13:25 진행 (CONFIRMATION c1)
- 완료: R0C c1 (H0) 743.7/745.9/751.0, C1 62.9, LONG 166.1, IND 650.6/652.1 완주 + 수치 프로브 정상. **B c1 (HB_MIX1) M3 사망(제외)**; 완주한 M1/M2 664.3/671.3 은 H0 대비 −10.7 %(SELECTION +5.7 % 와 반대). 수치: PPL 상대차 최대 10.1 %(기준 3 %), |Δlogprob| 평균 0.080(기준 0.05) → 사전 등록 기준 미달. **A12B c1 M2 사망(제외)**, M1 705.3(H0 의 A12e ≈790 대비 −11 %).
- 활성: A2B c1 부팅 → c2(R0C/B/A12B/A2B) → c3. 프로브 잡음 기준은 R0C c2 로 확인.
- 잠정 해석: HB_MIX1 은 SELECT 워크로드(seed 20261011)에는 유리했으나 고정 평가 입력(MAIN_SHORT)에서는 −10 % 대 손실, 분포도 뚜렷이 달라짐 → B 는 채택 불가 방향. CONFIRMATION 남은 부팅으로 확정.
- 오류·복구: B c1·A12B c1 사망(기록·제외). 자원: GPU 0 MiB(부팅 전환), load 79.
- unresolved_required: B CONFIRMATION 완료 → B_DECISION(현재 불채택 방향), INTEGRATION_DECISION(A12e 최종 후보), 원복 시험·COMPLETION_STATUS·FULL_REPORT·게시(사용자 승인).

## 09-18 13:55 진행 (CONFIRMATION c2 중)
- 완료: A2B c1 완주 — MAIN 703.0(H0 배치 A2 대비 −10.6 %), IND 713.5/717.1(R0C 대비 +9.8 %) → HB_MIX1 이득은 calibration 분포 전용, 고정 입력에서는 손실. R0C c2 752.0 완주. **프로브 부팅 간 잡음 0(H0 vs H0)** → B 수치 기준 미달 판정 타당. B c2 M1 사망(제외). B_DECISION(불채택 잠정)·INTEGRATION_DECISION(A12e 최종 후보) 잠정본 작성.
- 활성: A12B c2 부팅(세션 중) → A2B c2 → c3(R0C/B/A12B/A2B). 약 50 분.
- 오류·복구: B c2 사망(기록·제외). 자원: GPU 79 GB, util 0~82 %, load 45.
- unresolved_required: c2·c3 완료 → B_DECISION·INTEGRATION_DECISION 확정, 원복 시험(v4 + H0 + 플래그 unset 부팅 smoke 동일 확인), COMPLETION_STATUS, FULL_REPORT 렌더·SHA, 게시(사용자 승인).

## 09-18 14:25 진행 (CONFIRMATION c3 중)
- 완료: c2 — A12B M1 사망(제외), A2B 완주(MAIN 698.2, R0C c2 대비 −7.2 %; IND +6.0 %). c3 — R0C 742.2 완주(C1 63.5, LONG 161.4, IND 663.6/661.9). B_DECISION 표에 c2 반영.
- 활성: B c3 부팅 세션 중(M1 667.0, M2 717.7) → A12B c3 → A2B c3 → 원복 시험(v4 원본 .so + 원본 래퍼 + 플래그 unset + H0, MAIN3; 이후 채택 스택 복원) 자동.
- HB_MIX1 CONFIRMATION 누적: B 0/2 완주, A12B 0/2, A2B 2/2 완주(MAIN −10.6/−7.2 %, IND +9.8/+6.0 %). 수치 기준 미달(프로브 잡음 0 확인).
- 오류·복구: 없음(이번 구간). 자원: GPU 79 GB, util 100 %, load 73.
- unresolved_required: c3 종료 → B_DECISION·INTEGRATION_DECISION 확정, 원복 시험 결과 → ROLLBACK §4, COMPLETION_STATUS, FULL_REPORT·ARTIFACT_INDEX·SHA256SUMS 렌더, 게시(사용자 승인) 후 CronDelete.

## 09-18 14:55 진행 (CONFIRMATION·원복 시험 종료)
- 완료: CONFIRMATION c3 — B 완주(MAIN −6.1 %, IND +7.2 %), A12B 사망, A2B 완주(MAIN −0.8 %, IND +12.7 %). **B_DECISION 불채택 확정**(완주 4 부팅 고정 입력 전부 음수, 수치 기준 미달, 높은 사망률). **INTEGRATION_DECISION 확정: A12e** (v8f + GFNI + A2 + PF, H0, +7.7 %). COMPLETION_STATUS 작성, task.md C13~C15 완료, FULL_REPORT 렌더(601 줄, 아티팩트 6,837).
- 원복 시험(14:47~14:53): v4 원본 .so + 원본 래퍼 + 플래그 unset + H0 → MAIN 741.7/752.8/744.9 (평균 746.5; pre_fix R0_REF 746.3 과 동일 범위), NaN 0. 시험 후 채택 스택(v8f + 래퍼) 복원.
- 활성: 실행 중인 부팅 없음(GPU 0 MiB). 남은 작업: ROLLBACK §4 기록·smoke 대조, 최종 렌더·SHA, 게시(사용자 승인) → CronDelete.
- 오류·복구: 없음. 자원: GPU 0 MiB, load 14.
- unresolved_required: 게시 승인만 남음(그 외 필수 질문 모두 해결).
