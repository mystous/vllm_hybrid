# IDE_024 — test.md (검증 게이트)

## G0. CPU 단위 (GPU 불필요)

| 테스트 | 내용 | 합격 기준 |
|---|---|---|
| G0-1 | SUB_213 `test_pad_uniform.py` (U1a~g, U2a~b) | ALL PASS |
| G0-2 | α-EMA 게이트 단위: 합성 accept-len 시퀀스 → ON/OFF 전이 + 히스테리시스 | 플래핑 0, 전이 지연 ≤ 2×반감기 |
| G0-3 | oracle 테이블 생성기: SUB_212 70-cell 입력 → 라우팅 재현 | 기측정 winner 와 78.6% 이상 일치 |
| G0-4 | OFF 기본값 = upstream no-op (모든 신규 env 부재 시) | propose()/runner 출력 bit-동일 |

## G1. 기능 검증 (GPU, 단셀)

| 테스트 | 내용 | 합격 기준 |
|---|---|---|
| G1-1 | SUB_213 E1/E2 — FaP 재귀속 | E1 8,850±10%, E2 27,851±10% |
| G1-2 | SUB_213 P1~P4 — pad lever | P1 > 27,851 (GO) / P3 ≈ P4 (모델 정합) |
| G1-3 | FULL graph 적중 확인 | boot/runtime log 의 cudagraph dispatch 통계 또는 P1−P3 유의차 |
| G1-4 | 동적 게이트 (T2): low-α corpus (mbpp) | 정적 pad 대비 손실 회수, high-α 손실 ≤ 3% |

## G2. 정확도 게이트 (root CLAUDE.md Constraint 운영 해석)

- binding: per-token logprob max abs diff + 시퀀스 PPL relative diff (분포 유사성)
- informational: token-level 일치율
- 비교쌍: {pad ON vs OFF} × {FaP vs PIECEWISE}, greedy seed 고정 8 prompt × 32 tok

## G3. 회귀 게이트 (oracle 통합 후)

- 70-cell 축약판 (10 모델 × mix) 재측정: oracle 라우팅 가중 평균 ≥ 각-셀 최적의 95%
- MoE (R1-671B): vanilla 강제 확인 (suffix 음수 회피)

## G4. steady-state 스킵 (T3)

- metadata 캐시 hit ratio 로그 + 캐시 ON/OFF 출력 bit-동일 (G2 동일 프로토콜)
- step CPU time 프로파일 (RDTSC/perf) −2% 이상
