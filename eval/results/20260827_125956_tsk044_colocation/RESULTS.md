# TSK_044 — IDE_024 CPU Co-location 측정 결과 (2026-08-27)

- 노드: violet-h100-016 / 모델: Llama-3.3-70B-FP8 TP=8 (vLLM 0.28) / 워크로드: sonnet 2048/400 × 500p × C=64, seed 42
- BG: SHA256 멀티프로세스 (`bg_hash.py`), 1MB buf. 격리 = physical 상위 절반 (28-55,84-111) taskset
- **주의**: t1 (cold) 은 이후 셀과 prefix cache 상태가 달라 비교 불가 → **warm 셀 (t2/t3/t4/t5) 간 비교만 유효** (t4 = warmup 패스 후 solo)

## 결과 (warm 비교)

| cell | BG | out tok/s | Δ vs t4 solo | CPU busy | BG 산출 (hash/s) |
|---|---|---:|---:|---:|---:|
| t4 solo warm | — | 4,921.0 | — | 4.5% | — |
| **t3 BG 56 비격리** | 56 proc free | **4,896.3** | **−0.50%** ✅ | 29.4% | 51,451 |
| t2 BG 56 격리 | 56 proc pinned | 4,827.5 | −1.90% | 29.4% | 51,480 |
| t5 BG 112 비격리 | 112 proc free | 4,739.6 | −3.69% | **54.6%** | 99,496 |

참고: t1 solo cold = 3,037.8 (TSK_046 t1 재현 −0.04% — TST_020 재현성 검증)

## 판정 (TST_022)

- **손실 ≤1% 게이트**: ✅ BG 56 비격리 구성 (−0.50%)
- **CPU busy ≥50% 게이트**: ✅ BG 112 구성 (54.6%) — 단 이때 손실 −3.69%
- **두 게이트 동시 충족 구성은 미확보** — trade-off 곡선상 BG 60~80 proc 사이에 존재 추정 (후속 미세 sweep 후보)
- **격리(taskset)는 불리** (−1.9% vs −0.5%): 이 부하 수준에선 커널 스케줄러의 동적 배치가 고정 pin 보다 우수 — SUB_049 의 −3.6% 원인이 "격리 부재"라는 가설은 **기각**, 실제 원인은 BG 강도였음

## 운영 권고

- SLA 우선: BG ≤56 proc (CPU ~30%) — 사실상 무손실로 51K hash/s 급 CPU 작업 co-host
- CPU 활용 우선 (CLAUDE.md idle 불허): BG 112 proc — GPU −3.7% 비용으로 CPU 55% 가동 + 2× BG 산출
- 서버 합산 throughput 은 어느 구성에서든 양수 (BG 산출 ≫ GPU 손실분의 기회비용)
