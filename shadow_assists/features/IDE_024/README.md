# IDE_024 — CPU Co-location: 서버 합산 throughput

> parent: `PLN_003` / `TSK_044` / `TST_022` / 진행 로그: [`../IDE_023/PROGRESS_20260827.md`](../IDE_023/PROGRESS_20260827.md)

## 1. 배경

- **실증 승계**: SUB_041 (2026-05-22) — vanilla vLLM + CPU BG 56 proc 병행 시 **−0.04% (무손실)**. 유휴 100+ core 는 GPU serving 에 사실상 공짜 자원.
- **미봉합 승계**: SUB_049 (2026-05-23) — CPU LLM co-run 시 메인 −3.6%. 원인 추정 = core 미격리 (vLLM 프로세스와 BG 의 코어 경합). verdict 미기록 상태로 방치 → 본 IDE 에서 닫는다.
- **재정의**: "CPU 로 GPU inference 를 돕는다" (1~4세대, 전부 기각) 가 아니라 **"CPU 는 별도의 가치 있는 독립 워크로드를 수행하고, 서버 합산 throughput 으로 판정"** 한다.

## 2. 설계

- GPU serving: TSK_046 baseline config (70B-FP8 TP=8 + spec decode).
- BG 워크로드: CPU-bound 재현 가능 작업 (SHA256 멀티프로세스 — SUB_041 과 동일 계열) + (확장) CPU 효율이 좋은 실용 워크로드.
- **격리**: vLLM 프로세스 코어 예약 대비, BG 는 `numactl --physcpubind` 로 잔여 코어에 pin. GPU 4/4 = socket 0/1 분할이므로 BG 는 HT sibling 회피.
- 판정 (TST_022): t2 (격리 BG) 손실 ≤1% && CPU busy ≥50%.

## 3. 비고

CPU LLM co-run (0.5B, ~160 tok/s) 은 교환비가 나쁨 (70B 토큰 −390/s ↔ 0.5B +160/s, SUB_049 실측) — 본 IDE 의 BG 는 LLM decode 가 아닌 워크로드를 기본으로 한다.
