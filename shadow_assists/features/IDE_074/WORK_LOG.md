# WORK_LOG — IDE_074 (계측 수정 이유·조사 중 가설·선택/보류 근거. 원시 결과와 분리)

## 2026-09-17
- 11:20 착수. 지시서(PLAN.md) 만 업로드됨; 인계서 H/R/O 는 없음 → 인용 절 번호는 지시서 기준으로만 해석. 브랜치 `feat/cpu-moe-bottleneck-20260917`, IDE_074 등록.
- M0: `m0_collect.py`. HEAD 82ad7bca4 (기준 67d24dd84 의 후손). 컨테이너 SGLang 71de97b + 로컬 수정 7파일, kt_kernel_ext.so SHA e29357f7… (재빌드 전). hotmap_v2 SHA 7ce02cad…, layer_budget_5952 SHA c8ba4247… (per_layer 62개 합 5,952, 75~142).
- M1: `gpu_union.py` 로 §4.2 정의 구현, §4.3 합성 8건 PASS. 첫 `recompute_d2.py` 는 MoE 경계 114k × op 825k 의 2중 루프로 15분 이상 무출력 → UnionIndex(prefix sum + bisect) 로 교체, 동일 정의임을 selftest 에 추가 검증. 결과 `profiles/corrected_metrics.csv`: rank 1~3 의 원 보고 busy(31.7 s, 0.775) 는 stream 합계였고 union 은 23.4 s(0.572) — 원 보고값과 나란히 보존.
- 코드 지도(읽기 전용 에이전트, `profiles/code_map_survey.md`)에서 확인된 사실 3건이 측정 설계를 바꿈: (a) deferred=8=top-k → 모든 cold 가 deferred, GPU 는 층 L 에서 층 L−1 의 CPU 결과를 기다림 (immediate 없음). (b) 디코드 그래프 재생 시 층별 Python 실행 0회 → 층별 GPU 이벤트는 트레이스의 graph id/node id + memcpy 패턴으로만 분절 가능. (c) job id/epoch 부재, slot 이 유일 키 → C++ 에 slot 별 epoch 카운터를 추가해야 GPU replay 와 대응 가능.
- 기존 D2(retry2) 트레이스 패턴 조사(`trace_pattern_probe.py`): graph 2(decode bs=63) 첫 replay 에 HtoD 62 / DtoH 248 / fused_moe 124 → 층당 4·1·2 의 고정 패턴. HtoD 직전 커널은 항상 `moe_sum_reduce_kernel`(hot MoE 마지막), 직후는 elementwise add(결합). 같은 stream(96). 이로써 "hot 종료 → cold HtoD 시작" 유휴를 층·replay 별로 뽑을 수 있음 (`gpu_layer_timeline.py`). 이는 기존 자료의 재집계이며 층 번호는 휴리스틱(HEURISTIC_LAYER_ID).
- 코드 지도 §12-1 의 "KT_GPU_EXPERTS_PER_LAYER 코드 없음" 은 부팅 시 `patch_per_layer_experts.sh` 로 적용 후 종료 시 revert 되는 구조 때문 (IDE_073 runtime_proofs 에 62층 값 존재). 새 캠페인도 같은 스크립트로 적용.
- M3/M4 구현 선택: C++ 최소 diff(`kt_evt_patch.py`, 4파일 + 신규 헤더). 타임스탬프는 (A) poller 의 go 관측, (B) imm/def enqueue, (C) done-setter 실행, (D) deferred 작업 전후 stamp 태스크(단일 FIFO 워커이므로 정확히 감쌈), (E) NUMA 서브풀 fork-join 전후. 디바이스 동기화 0회, 기존 event/stream 의존 불변. `clock_gettime(CLOCK_REALTIME)` 로 kineto 의 host 시계와 같은 도메인. 레코드 링 1<<20, 200 ms 플러셔, 누락 카운터(#dropped). env `KT_EVT` 미설정이면 rec==nullptr 로 기존 경로.
- 부수 변경: `moe_base.hpp::forward_decode` 의 게이트 없는 printf(qlen==1 마다 stdout 2행) 를 KT_PHASE_PROF+64회 게이트로 통일. C=1 세션에만 영향. OFF/ON 공통 바이너리이므로 IDE_074 내부 비교엔 중립, IDE_073 C1 결과와는 차이 요인 → CONFIG_DIFF 기록.
- 재빌드: 기존 CMake 캐시(Release, AMX/AVX512 ON) 로 incremental build (`cmake --build … -j48`, 약 3분, error 0). 새 .so 8,306,720 B SHA adf49e48…; 원본 8,290,336 B SHA e29357f7… 은 `/sgl-workspace/ide074_backup/` 보존. Python `experts_base.py`(site-packages, IDE_033 패치본) 에 slot→layer 맵 기록 4행 추가.
- rank 1~3 트레이스: 층당 all-reduce 커널 2개 중 짝수 순번(휴리스틱) 의 duration p50 445 µs(합 7.69 s / 트레이스 41 s) — rank 0 의 cold 대기(p50 343 µs)에 대응하는 rank 1~3 측 대기 후보. 순번↔(attention 뒤 / MoE 뒤) 귀속은 미확정.
- 세션 배분: KT_EVT 가 부팅 시 getenv 로 고정되므로 OFF/ON 을 한 부팅에서 전환 불가 → 부팅 단위 모드 (PLAN_RESOLVED §4). 지시서 §9.3 의 묶음 순서(P0→P1→P2 …)는 실행 불가로 기록.

- 13:10~13:41 M5 SMOKE(P1) 통과 후 M6 체인 B1(OFF 6세션)·B2(P1 3)·B3(P2 3) 완료, 서버 사망·실패 없음. P2 의 perf stat 은 스케줄러 4 프로세스 합산(≈59 CPUs utilized, IPC 1.29), pcm-memory 는 1 s 표본 38개 전부 요청 창 안(경계 표본 0).
- cpu_only_metrics 의 rows 키는 slot→rows 맵의 마지막 기록을 쓰므로 eager(prefill) 슬롯은 마지막 rearm 의 rows 로 표기됨 (prefill chunk 별 qlen 은 kt_evt.csv 의 qlen 열에 있음). 보고서에 한계로 기록.
- 14:05 조건부 F1(C1·긴 입력 OFF/ON, 부팅 2) → GLM 4문항 게이트(부팅 1) 체인 시작. GLM 은 IDE_073 에서 20문항 전부 절단·8 오류(출력 텍스트 비정상)였으므로 게이트 결과에 따라 BLOCKED_NORMAL_OUTPUT 가능.
- 14:25 P1_r3 에서 `t_go < t_dtoh_last_end` 1,659/23,808 행 (50/384 replay, 세션 중 2.4 s 구간에 집중). 크기는 −0.25 ~ −2.25 µs (p50 −0.75), 매핑·계수 불일치 0, 양수 쪽 p50 5.0 µs. 크기가 ms 급이 아니므로 slot↔replay 매핑 오류가 아니라 CPU CLOCK_REALTIME ↔ CUPTI host 시계 정렬 지터로 분류. 판정 규칙: 5 µs 초과 음수만 순서 위반, 이내는 지터로 계수(원값 보존, 행 유효). 관측된 정렬 오차 상한 2.25 µs 를 clock 한계로 보고. SMOKE·P1_r1~r3 재분석.
- 14:35 GLM 게이트: native FP8 BASIC4, 4문항 모두 1024 토큰 절단 + 텍스트 비정상(단어 뒤섞임) → BLOCKED_NORMAL_OUTPUT. IDE_073 의 20문항 결과와 같은 양상. INT4 변환·converter 수정은 범위 밖으로 남김.
