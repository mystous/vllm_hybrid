# SUB_035 — C1 layer-fusion / OMP team overhead 감소 plan

> **parent**: TSK_019 / O 분석 §7.2 ★★ C-tier 항목
> **출처**: 사용자 명시 (turn 19) — "B, C 진행"
> **현 status**: plan 단계, SUB_034 (B1) 결과 대기

---

## 1. C1 의 본래 가설 vs 코드 surface 현실

### 1.1 본래 가설 (N 문서)

**libgomp 43.75% (`gomp_team_barrier_wait`) 의 직접 원인** = NEO 의 workload-driven OMP fork/join (layer 당 80회). cross-layer 묶음으로 80회 → 20-40회 로 줄이면 libgomp 비율 직접 감소.

### 1.2 코드 surface 현실 — cross-layer 묶음 불가능

| 영역 | 현 상태 |
|---|---|
| `csrc/cpu/pacpu/core.h::ispc_attention_tasks` | **한 layer = 한 OMP team launch** (line 359 `#pragma omp parallel`) |
| layer i output → layer i+1 input | model-level 의존성 chain — attention layer 사이에 GPU forward (preproj/postproj) 가 끼어 있음 |
| 80 layer cross-fusion | **불가능** — layer 사이의 GPU 작업이 dependency barrier |

→ **layer-level fusion 자체는 model architecture 변경 (LISA / CLAA) 필요** = N 문서 #5.3 C-tier, large effort (2-3 주).

## 2. 실현 가능한 C-tier 대체 옵션 (SUB_034 결과 본 후 선택)

### Option C1a — OMP team launch overhead 측정 (instrumentation only)

| 항목 | 값 |
|---|---|
| 목적 | `#pragma omp parallel` 진입 / 종료 의 wall 비용 실측 → 정량적 시야 확보 |
| 변경 | `core.h` 에 OMP team launch 직전/직후 timer 추가 (env VLLM_NEO_OMP_LAUNCH_PROFILE) |
| effort | 30 min |
| 의미 | C-tier 의 진정한 ROI 확인. launch overhead 가 cdec_wait 의 < 10% 면 cross-layer fusion 가치 없음 |

### Option C1b — OMP `nowait` clause 시도 (N 문서 #3.3)

| 항목 | 값 |
|---|---|
| 목적 | Step 1 → Step 2 사이 barrier 만 명시적 (line 453) — Step 0 → Step 1 의 implicit barrier 영역 nowait 가능성 점검 |
| 변경 | `core.h::ispc_attention_tasks` 의 OMP barrier 영역 분석 → 의존성 없는 barrier 제거 |
| effort | 1-2 hr |
| risk | KV cache write/read race 가능 → 정확도 게이트 깨짐 |

### Option C1c — OMP chunk schedule 재시도 (N 문서 #3.5 + F6 재검토)

| 항목 | 값 |
|---|---|
| 목적 | F6 (dynamic schedule) 회귀의 root cause 분석 후 `schedule(guided)` 또는 `schedule(static, N)` 영역 sweep |
| 변경 | `core.h::ispc_attention_tasks` 의 task 분배 알고리즘 (현 token-level balance) 와 OMP schedule clause 결합 |
| effort | 2-3 hr |
| risk | F6 = -1.4% 회귀 의 root (atomic counter overhead) 가 다른 schedule 에도 발생 가능 |

### Option C1d — CDEC executor concurrency 재시도 (SUB_023 후속)

| 항목 | 값 |
|---|---|
| 목적 | SUB_023 CW=2 sweet spot 의 다른 분배 시도 — CW=2 + 다른 OMP=N (현재 OMP=10) |
| 변경 | `attention.py` 의 CW envvar + `core.h` 의 OMP team size 조합 sweep |
| effort | 2-3 hr |
| risk | SUB_023 CW=4+OMP=5 = -52% 회귀 — concurrency × OMP 경합 위험 |

### Option C1e — Persistent OMP team (Phase 3.1 회귀 root cause 분석)

| 항목 | 값 |
|---|---|
| 목적 | Phase 3.1 (Persistent OMP) 의 -1.4% 회귀 root cause 분석 후 재설계 |
| 변경 | `core.h` 의 OMP team init 패턴 (현 매 call setter, `omp_set_dynamic(0)` 등) 정밀 조정 |
| effort | 2-3 hr |
| risk | Phase 3.1 회귀의 진짜 원인 미파악 시 동일 회귀 |

## 3. 권장 sequencing

```
[SUB_034 B1 async depth sweep] 결과 출현
       ↓
   depth ≥ 2 가 noise 이면 → CPU pacpu compute 자체가 bottleneck 확정
   depth ≥ 2 가 win 이면 → CPU sync wait 가 bottleneck 확정
       ↓
[Option C1a — OMP launch overhead 측정 (30 min, 정량적 시야)]
       ↓
   launch overhead < 10% 면 → C-tier 전체 가치 없음, B-tier 다른 항목 (B4 SPARAMX, B5 libxsmm) 진입
   launch overhead ≥ 10% 면 → C1b (nowait) 또는 C1c (chunk schedule) 시도
       ↓
[Option C1b/c/d/e 중 1개 선택 적재]
       ↓
   결과 정리 후 다음 C-tier 항목 또는 B-tier 항목 진입
```

## 4. 종료 조건

- C1a 측정 결과가 "C-tier 가치 없음" 으로 나오면 SUB_035 종료 + B-tier 잔여 (B2 APEX, B4 SPARAMX, B5 libxsmm, B6 IPEX threadpool) 진입
- C1b~e 중 1+ 가 ≥5% win 이면 정식 채택 + 3-run avg + 정확도 verify
- C1b~e 모두 noise 이면 C-tier 전체 폐기 + 측정 환경 재정의 (Path A 권고 복귀)

## 5. 다음 turn 의 deliverables

1. SUB_034 결과 분석 → C-tier 의 진정한 ROI 추정
2. Option C1a (OMP launch overhead 측정) 우선 적재 + 측정 (30 min + 15 min)
3. C1a 결과 → C1b~e 중 1 선택 또는 SUB_035 종료
4. 측정 / 결과 doc / id_registry / INDEX 갱신
