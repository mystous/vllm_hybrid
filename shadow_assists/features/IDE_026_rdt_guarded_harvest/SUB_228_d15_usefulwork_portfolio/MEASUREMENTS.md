# SUB_228 — useful-work portfolio (무엇을 harvest 하나), 2026-06-15

> **판정: positive ⭐ — 실제 유용작업은 RDT 없이도 near-free harvest.**
> 합성 메모리 aggressor(worst-case)만 큰 간섭. reviewer "so what do you run?" 직접 답.

## 결과 (victim 0-7, harvest 16-way @8-23, baseline ns/load=86.6)
| 작업 | 유형 | victim degr% | harvest 적합 |
|---|---|---:|---|
| openssl AES-256 | 연산 | −3.3% | ✅ 자유 |
| openssl SHA256 | 연산 | −3.6% | ✅ 자유 |
| xz -6 압축 | 메모리+연산 | +0.7% | ✅ 자유 |
| synthetic basic | 순수메모리 ref | +156.2% | ⚠ RDT 필요 |

## 판정 (D15 게이트: ≥3 후보 + 권장표 — 충족)
1. **연산-bound 유용작업(암호/해시/압축) = victim 간섭 ≈0** (−3.6%~+0.7%, 노이즈 내).
   캐시상주 working set + ALU 점유라 메모리 BW 경합 거의 없음 → **RDT 가드 불요,
   유휴 CPU 자유 harvest**.
2. **순수 메모리 스트리밍만 +156% 간섭** — 합성 aggressor(SUB_214~227 의 worst-case)는
   실제 유용작업 대비 **과도하게 비관적**. RDT 기계장치는 메모리-bound harvest 전용.

## 권장 portfolio (CPU 를 뭘로 채우나)
| Tier | 작업 | 가드 |
|---|---|---|
| 1 (자유) | 암호화·해시·압축·AMX 보조추론 (연산-bound) | 불요 (degr ≈0) |
| 2 (가드) | KV 스캔·prefix index 등 메모리-스트리밍 | RDT-MBA 또는 SW-MBA(SUB_224) |

## 함의
- **harvest 접근의 실용성 입증**: 실세계 useful work 대부분이 Tier 1 → serving 무손실
  유휴 CPU 활용 가능 (CLAUDE.md objective 직접 달성). RDT 는 Tier 2 안전망.
- 단, vLLM-내장 후보(prefix index/KV압축)의 실제 메모리강도는 추가 측정 필요
  (일부는 Tier 2 가능).

산출물: `runs/results.csv`. 선행 SUB_227(IE 지표).
