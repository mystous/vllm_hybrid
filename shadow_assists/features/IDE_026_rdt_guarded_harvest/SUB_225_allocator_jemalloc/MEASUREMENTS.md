# SUB_225 — Allocator interference (jemalloc) on suffix draft (2026-06-14)

> **판정: 기각 (end-to-end 향상 없음)** — 단, 레버 자체는 작동(마이크로벤치 −17%).
> **메타-발견**: 70B serving은 GPU-bound, suffix draft CPU 는 GPU 와 완전 오버랩
> (critical path 밖) → CPU 가속이 tps 로 안 이어짐. 233/225/231 (CPU-side) 계열의
> 70B 구조적 천장을 jemalloc 이 증명.

## 1. 가설 / 레버
SUB_233 진단: suffix walk 는 Draft `std::vector` push_back 할당 지배(포인터체이스 아님).
→ fast allocator(jemalloc) LD_PRELOAD 로 할당 비용 절감. 설치: `libjemalloc2`(apt, 승인).
무설치 대안 glibc 튜너블(arena/tcache)은 마이크로벤치 무효(단일스레드 할당엔 효과 없음).

## 2. CPU 마이크로벤치 (격리, taskset -c 0, 동일 walk 1,192,308 토큰)
| allocator | ns/call | vs glibc |
|---|---:|---:|
| glibc (default) | 8,137 | — |
| **jemalloc** | **6,736** | **−17.2% (빠름)** |
→ jemalloc 은 할당-bound walk 를 실제로 17% 가속. (SUB_233 prefetch 는 +2% 악화였음 — 대조)

## 3. end-to-end 70B (K6 pad, taskset 0-47,56-103, 동일세션 A/B)
| corpus | baseline (glibc) | jemalloc | 델타 |
|---|---:|---:|---:|
| mix | 6,232.4 | 6,209.3 | −0.4% |
| swebench | 8,023.1 | 7,947.6 | −0.9% |
| lmsys | 6,349.6 | 6,332.3 | −0.3% |
→ 전부 평탄(노이즈 내, 약간 음). tps 향상 없음.

## 4. 진단 — 왜 walk −17% 가 tps 0 인가
70B TP8 conc32 는 **GPU 연산이 병목**. suffix draft(CPU)는 GPU verify 와 **오버랩**되어
critical path 밖. draft 를 17% 빠르게 해도 GPU shadow 안에서 끝나므로 tps 불변.
= SUB_233(prefetch null)과 동일 구조. **CPU-side 미세최적화는 70B serving tps 를 못 올림**
(레버가 진짜 작동해도 — jemalloc 이 결정적 증거).

## 5. 함의
- 233/225/231 (전부 CPU-side) 는 70B 에서 구조적 천장. 재시도(다른 allocator/config)도
  같은 천장 → 추가 70B 부팅은 null 재확인일 뿐.
- CPU-opt 가 의미있는 곳은 **CPU 가 critical path 인 regime** (소형 모델·초고 conc·
  CPU-bound 단계). 70B 는 해당 안 됨.
- jemalloc 자체는 무해(−0.4% 노이즈) + draft 빨라짐 → **운영 기본값으로 둬도 손해 없음**
  (다른 host-bound regime 에서 이득 가능). 단 70B serving 이득은 0.

산출물: `driver_ab.sh`, `runs/`. 마이크로벤치: SUB_233 `microbench.py` 재사용.
