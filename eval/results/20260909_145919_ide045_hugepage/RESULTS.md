# IDE_045 — expert 가중치 버퍼 2MB huge page (`KT_HUGEPAGE=1`) — **기각 (효과 없음)**

가설: cold expert 스트리밍 332 GB/s (torch 96스레드 읽기 390 의 85%) 의 잔여 격차가 TLB 미스·페이지 워크.
패치: `moe_base.hpp` BufferB 3곳 `aligned_alloc(64)` → 2MB 정렬 + `madvise(MADV_HUGEPAGE)`. 적용 확인: 서버 프로세스 AnonHugePages 133 GB → 245 GB (expert 버퍼 232 GB 전량).

## 마이크로벤치 (rows 스윕, 동일 프로세스 내 base → hp)
| n_cold / T | base µs/expert | hp µs/expert |
|---|---|---|
| 8 / 1 | 58.4 | 59.5 |
| 32 / 4 | 48.7 | 48.8 |
차이 없음 (소규모 할당은 THP always 로 이미 huge page 였을 가능성).

## 서빙 (최선 구성: hot-96 α0.25, CF+τ, AMX 선택, FP8 KV 110k, graph 160; 480 프롬프트 = 3×C)
| 구성 | C160 r1 | C160 r2 | C64 |
|---|---|---|---|
| huge page ON | 947.4 / 122.3 / TTFT 6.07s | 970.1 / 120.6 / 5.78s | 722.1 / 64.5 / 3.14s |
| huge page OFF (같은 시간대 대조) | 965.3 / 120.4 / 5.90s | 984.6 / 119.4 / 5.62s | (IDE_044) 766.2 / 60.6 / 2.99s |
GSM40 (ON) 97.5.

**판정**: C160·C64 모두 대조와 동일 (노이즈 ±2%). 스트리밍 잔여 격차는 TLB 가 아님 → 기각.

## 주의: 1차 C64 이상치 (1093.5 tok/s, TTFT 551 ms)
1차 체인의 C64 (192 프롬프트) 는 직전 C160 이 192 프롬프트·seed 42 로 같은 프롬프트 집합을 보내 **prefix cache 전량 적중** 한 결과. 재측정 (C160 480 프롬프트 뒤) 에서 재현되지 않음 (722). 부수 정보: prefill 이 사라지면 C64 처리량 +43%, TPOT 60.6→54.6 → prefill 이 C64 처리량의 약 30% 를 차지하며 decode 를 끊는다는 정량 근거 (→ IDE_046 동기).

파일: `mb_rows_{base,hp}.json`, `hp_c160.log`(192 프롬프트), `hp_c64.log`(prefix cache 적중), `hp480_*.log`, `base480_*.log`, `gsm40_hp.json`, `RUN.log`.
