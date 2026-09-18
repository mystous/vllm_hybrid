# ANALYSIS_AND_DECISION — IDE_076 A1·A2·B (작성 중; 마지막 갱신 2026-09-17 23:30 KST)

원값은 FULL_REPORT.md. 이 문서는 가설·실제 줄인 경로·반증·혼입·한계·단독/조합 효과·채택 이유를 적는다.

## 0. 기준선 (O0)
- reference_binary v4 R0_REF: MAIN 745.4/746.0/747.5 (같은 부팅 변동 0.3 %), C1 53.2, LONG 163.0.
- 신규 빌드 v6 플래그 OFF R0(부팅 1): MAIN 730.0/736.8/739.1 (평균 735.3, −1.5 % vs v4), C1 63.1, LONG 165.4, smoke 텍스트 4/4 동일. §3.3 "설명되지 않은 변화" 후보: (a) A1 런타임 분기가 OFF 경로 hot 루프에 남음(v6) — v7 에서 템플릿화·asm diff 0 으로 제거, (b) 부팅 간 변동(IDE_075 OFF 부팅 간 3 %). 판정은 v4/v7 부팅 블록 교대 결과로.

## 1. A1 — 기존 RB 내부 R=1/2 잔여 비용
- 가설: RB(avx_rb_rows) 에 남은 동적 비용(언팩·브로드캐스트·spill·to_mat) 을 줄이면 Cold 서비스가 준다.
- 관측 (A1_GAP_ANALYSIS): R=1 경로 명령의 6 % 만 dot; 언팩·로드가 2 배. 그러나 replay 에서 R 1→2 가 +7 % 뿐 → W 스트리밍(≈190~230 GB/s/socket) 이 지배. 서버 expert 당 52 µs = replay n16 51.9 µs.
- A1-a(GFNI 언팩): 명령 −5 %, 출력 bitwise 동일, replay −0~2 % (판정 한도 아래). E2E 1차: A1(v6) MAIN 746.3 vs R0(v6) 735.3 (+1.5 %, 부팅 1) — v6 OFF 경로 손실을 되찾는 수준(v4 R0_REF 746.3 과 동일).
- A1-e(기존 프리페치 knob KT_AVX_PF=1): replay n16 R=1 −12 %, R=2 −10 %, n160 −4~6 %. E2E 1차: A1e(v7) MAIN 759.1 (부팅 1) — R0_REF 대비 +1.7 %; A1ae 1 세션 769.9 (부팅은 이후 사망 → 무효). 블록 반복은 사고 해결 후.
- 이미 구현된 것: row blocking 자체(RB), R=1/2 정적 블록 → ALREADY_IMPLEMENTED 로 계수하지 않음.
- 판정: (대기)

## 2. A2 — 같은 Cold job 내부 descriptor 묶음
- 가설: 스테이지 장벽 7 개·작은 task 공급 비용이 서비스 경과 시간에 노출된다.
- 관측: 격리 replay n 스케일 절편 ≈ 0(job 고정비 <50 µs); v7 probe(왜곡 제거): 장벽 tail 3.7 %(A1=0)/4.4 %(A1=1), 워커 종료 폭 5.2/6.0 % of dispatch wall. S29 비-GEMM 스테이지 합 94 µs(10 %).
- 결정: 줄일 수 있는 내부 비용이 확인됨(≥3 %) → 구현 (`forward_prefill_a2_`: S1~S5 expert 단위 readiness 단일 dispatch, 소스 적용·문법 통과; 빌드 v8·bitwise replay·E2E 는 사고 복구 후). 기대 상한 ≈ 서비스 −3 % (E2E 판정 한도 2 % 경계).
- v8f replay (2026-09-18 06:47, n=40): A2 ON 출력 = OFF 와 bitwise 동일(6 장면). 시간 p50 변화: hot_r1 −0.3 %, hot_r2 −0.5 %, hot_r3 −2.0 %, seq_mixed −1.9 %, hot_r1_e18 +2.1 %, hot_r2_e18 +0.4 %. A1+A2: hot_r3 −3.4 %, seq_mixed −1.7 %, hot_r1 −1.0 %, hot_r2 +0.7 %. proof: 적격 job 142/142 묶음 실행, fallback 0, 장벽 tail 2.8 % of dispatch wall. → 큰 job 에서만 −2 % 안팎(기대 상한 −3 % 아래), 작은 job 은 효과 없음. E2E 판정은 post3 블록.
- 판정: (대기: E2E 블록 p1~p6)

## 3. B — 생산자 비용 기반 정적 Hot 배치
- 가설: CPU 가 늦는 층에 슬롯을 더 주고 이른 층에서 빼면(합 5,952) 소비 지연이 준다.
- 관측: v0 모델(기존 빈도) 은 현재 워크로드의 cold 사용을 과소 추정 → CALIB 재수집 예약. 같은 층 빈도 교환은 v0 에서 순이득 없음.
- SELECTION (09-18 09:15): HB_MIX1 +5.7 % (4/4 세션 > H0 max), HB_DEC1 −21.9 % (prefill 커버리지 손실) → HB_MIX1 선별, DEC1 기각. CONFIRMATION(post4) 대기.
- CONFIRMATION (09-18 12:54~14:47): 완주 4 부팅(B c3, A2B c1~c3) 고정 입력 MAIN −6.1/−10.6/−7.2/−0.8 %, 독립 입력 +7.2/+9.8/+6.0/+12.7 %; 수치 기준 미달(프로브 잡음 0). **판정: 불채택** — 배치가 calibration 분포에 과적합(B_DECISION).

## 4. 통합·최종 선택
- 최종: **A12e** (v8f + GFNI + A2 + PF=1, H0) 짝 이득 +7.7 %(5 블록, 전부 양수), 회귀 한도 내. A2 단독 +5.0 %, PF 단독 +3.1 %, GFNI 단독 +1.6 %(불채택이나 결합에서는 +1~2 %p). B 제외. 상세 INTEGRATION_DECISION.

## 5. 정상성 사고와 모집단 분리 (2026-09-18 00:08~01:30)
- 사고: 디코드 웨이브 종료 후 새 웨이브의 prefill 첫 스텝들에서 CUDA assert(NaN) 로 서버 사망(7 부팅; v4 포함, A1/A2/PF 무관). 기존 callback 경로는 생존.
- 원인: callback-free eager 패킷의 `rearm_packet` 이 이전 스텝의 go 소비를 기다리지 않는데 SGLang overlap 스케줄러가 호스트를 한 스텝 앞서 실행 → 다른 row 수(다른 pinned 버퍼)로 패킷이 덮여 이전 go 에 잘못된 args 로 작업 실행. 상세 WORK_LOG 01:20.
- 수정: 슬롯 세대 대기(C++) + eager 슬롯 링(Python) (`kt_cf_rearm_fix_patch.py`). 첫 완화(layer 0 sync)는 효과 없어 폐기.
- 모집단: pre_fix(수정 전; 사망 부팅 세션 제외) / sync_only(폐기된 완화) / post_fix(v8f·v4f). **최종 판정은 post_fix 만** 사용하고 pre_fix 는 방향성 참고로만 둔다(§14).
- 2026-09-18 06:40 최종 상태: 원인 **미해결**. 프로브(호스트 동기화 없는 스트림 측정)로 3 회 재현했고 첫 NaN 은 모두 GPU hot-expert(SGLang fused_moe FP8) 출력이었다(MoE 입력·CPU 출력 정상). 발현 조건은 callback-free ∧ overlap ∧ 빈 imm 생략(각 대안 구성 36~48 전환 0 사망)이며, 층당 ≈0.2 ms 이상의 측정 부하가 붙으면 사라지는 타이밍 경쟁이다. 커널 Xid 43 이력상 IDE_076 이전(9-16·9-17)에도 존재. 근본 원인 추적은 지시서 §17.2 범위를 넘어 사용자 지시로 중단. **이후 모든 비교는 원래 구성으로 진행하고 사망 부팅은 실패로 기록·제외한다(§12.1).** 제거된 완화: eager layer-0 sync. 유지: rearm 세대 대기(무해; 기준 바이너리 v4f 에도 동일 적용).
