# IDE_026 — SCED: Speculation-Coupled Expert Dispatch

> 논문 주력 후보. 아이디어 본문: [`../../brainstorming/paper_novelty_candidates_20260827.md`](../../brainstorming/paper_novelty_candidates_20260827.md) §2
> 실험 플랜 (이론·가설·절차): [`PLN_004.md`](PLN_004.md)

한 문장: **spec depth K 는 CPU-상주 MoE expert 의 arithmetic intensity 를 조종하는 1급 제어변수다** — 검증 배치 증폭이 CPU expert 를 GEMV→GEMM 화하고, memory-bound 구간에선 투기 낭비가 공짜이므로 최적 K 가 GPU-only 보다 커진다 (반증 가능한 예측). (K, expert 배치, deferral) 공동 최적화 제어기가 시스템 기여.

상태: 구상·플랜 단계. **실험 미실행** (E0 시뮬레이션만 하드웨어 불요로 착수 가능).
