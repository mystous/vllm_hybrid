# SUB_230 — eBPF run-queue latency 조기경보, 2026-06-15

> **판정: runq latency 는 메모리-BW 간섭에 BLIND (전용코어 serving).** SLO 신호로
> 부적합 — 메모리-지연 차원(SUB_243 CANARY)이 보완 필요. (README 자체 주석과 일치.)

## 측정 (bpftrace v0.20.2, sched_wakeup→sched_switch 지연, victim-필터)
Phase A (victim 단독, cores 0-7):
```
@vic_runq_us:
[0]      596068 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[1]        1419 | ...   [2,4) 1728 ...   tail 미미
```
→ victim runq latency **거의 전부 0µs** (596068/~599000). 전용코어라 큐 대기 없음.

## 판정
1. **전용코어 victim 은 런큐 대기 0** (Phase A). 메모리 aggressor(코어 8-23)가
   돌아도 victim 은 여전히 전용코어 → **runq latency 변화 없음(~0 유지)**.
2. 그러나 victim 작업률은 메모리 stall 로 **ns/load 90→220 (2.5× 악화)** (SUB_222/224/227
   에서 동일 셋업 실측). 이 악화는 **on-CPU stall** 이라 sched_wakeup→switch 지연에
   전혀 안 잡힘.
3. → **runq latency 는 메모리-BW 간섭의 SLO 신호로 부적합**. CPU-시간 경합(공유코어)
   만 감지. 코어 분리 harvest 설계의 지배적 간섭(메모리)을 놓침.
4. **함의**: governor 입력은 runq 가 아니라 **메모리-지연 직접 측정**(LLC miss latency,
   mbm, 또는 victim canary probe = SUB_243)이어야 함. D17 게이트(감지≤20ms) 불충족
   (감지 자체 불가).

## 비고
- comm 필터는 kernel comm 15자 truncate("victim_aggresso") 주의 — PID 범위 필터 권장.
- bpftrace 오버헤드 ~1% 확인.

산출물: `runs/vic_runq_A.txt`.
