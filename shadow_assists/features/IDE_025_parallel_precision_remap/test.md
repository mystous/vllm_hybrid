# IDE_025 — test.md

## G1. 공정성 사전 점검 (셀 실행 전)

- DP=N 셀의 총 부하 = 기준 셀 × N (conc 스케일) — runner 인자 검증
- boot log 에서 실제 적용 확인: `data_parallel_size=N`, `kv_cache_dtype=fp8_e4m3`,
  `enable_dbo=True`, `enable_sp=True` grep (묵시 기본값 confounder 방지)
- 호스트 상태 기록 (DSA WQ clients / HugePages / clocks)

## G2. 처리량 판정 (TSK_047)

| 비교 | GO | kill |
|---|---|---|
| 8B TP1/DP8 vs TP8 | cluster tps ≥ +20% | < +20% → N1 기각 + 프로파일 재해석 |
| 32B/70B 하이브리드 | ≥ +15% | < +5% |
| +DBO / +SP 증분 | ≥ +3% (3-run 유의) | 음수 |
| +FP8 KV 증분 | ≥ +10% **AND** G3 통과 | G3 탈락 시 즉시 제외 |

## G3. 정확도 게이트 (FP8 KV 만 — 나머지는 수치 경로 동일로 면제)

- root CLAUDE.md Constraint 운영 해석: 분포 유사성 binding
- greedy seed 고정 8 prompt × 32 tok, per-token logprob max abs diff + PPL relative diff
- bf16 기준 대비 PPL rel diff > 1% 시 탈락 (informational: token 일치율 기록)

## G4. 시스템 지표 (Objective 검증)

- DP=8 시 CPU 사용률 (8 EngineCore) — 224 thread 점유율 상승 확인 (mpstat)
- TTFT/TPOT p50/p99 병기 — DP 가 tail 을 해치지 않는지

## G5. 재현성

- winner 구성 3-run, CV < 2%
