# SUB_213 — MEASUREMENTS

> **status**: 측정 대기 (GPU 가 타 실험 점유 중 — 사용자 지시로 실행 보류, 2026-06-11)
> 사전 예측은 README.md §2/§3 에 측정 **이전** commit (IDE_023 P0 프로토콜).

## 셀 현황

| 셀 | 구성 | 사전 예측 (gen tps) | 실측 | Δ vs 예측 | 판정 |
|---|---|---:|---:|---:|---|
| E1 | vanilla + PIECEWISE, mix | ~8,850 | — | — | — |
| E2 | suffix K32 + PIECEWISE, mix | ~27,851 | — | — | — |
| P1 | suffix K8 + **pad** + FaP, mix | 30.6k~34.8k | — | — | — |
| P2 | suffix K15 + **pad** + FaP, mix | (정보용) | — | — | — |
| P3 | suffix K8 + nopad + FaP, mix | ≈ P4 | — | — | — |
| P4 | suffix K8 + nopad + PIECEWISE, mix | (기준점) | — | — | — |

공통: Llama-3.1-8B-Instruct, TP=8, conc=32, limit=500, max-tokens=8192, gmu=0.85,
host DSA WQ **enabled 그대로** (E1 으로 host DSA 효과=0 을 확정하는 설계).

## 기준점 (기존 측정)

| 출처 | 구성 | gen tps |
|---|---|---:|
| TSK_042 (06-02) | vanilla + PIECEWISE | 8,850 |
| SUB_212 (06-10) | vanilla + FaP | 12,089 (+36.6%) |
| TSK_042 (06-02) | suffix K32 + PIECEWISE | **27,851** |
| SUB_212 (06-10) | suffix K32 + FaP | 24,407 (−12.4%) |

## 판정 로직

1. **E1 ≈ 8,850 (±10%)** → H-FaP 확정: "+36% = host DSA WQ" (SUB_212 결론) 기각,
   FaP 로 재귀속 → SUB_212 문서 4종 정정.
2. **P1 > 27,851** → uniform padding GO (FaP×suffix 양립 lever 성립).
   P1 < 27,851 → kill (verify 낭비 > launch 이득).
3. **P3 ≈ P4** 이어야 함 (FULL 미적중이면 FaP 무관) — P3 ≫ P4 면 모델 오류 신호.

## 단위 테스트 (CPU, GPU 불필요)

| 테스트 | 내용 | 결과 |
|---|---|---|
| U1a~g | pad/truncate/경계/멀티요청 uniform | (실행 대기 — `test_pad_uniform.py`) |
| U2a~b | OFF no-op + flag file fallback | (실행 대기) |

## 결과 해석

(측정 후 작성)
