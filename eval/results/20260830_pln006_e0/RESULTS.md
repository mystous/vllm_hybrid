# PLN_006 E0 — 마이크로벤치 + 계산기 + 재예측 결과

## E0a — 모델 입력 (전부 스펙/마이크로벤치, end-to-end 수치 아님)

| 입력 | 값 | 출처 |
|---|---|---|
| DRAM 소켓 로컬 / 원격 / interleave96 | **195 / 112 / 276 GB/s** | STREAM triad 실측 (금일, 컨테이너 numactl) |
| PCIe H2D / D2H | **51.9 / 55.1 GB/s** | torch pinned copy 실측 (금일) |
| AMX 곡선 F_C(n_e), C_eff 21TF, BW_eff 70~136, call 고정비 ~20μs | E1 microbench (기보유) | `20260829_001433_pln004_e1_amx_microbench` |
| 모델 구조 480B: L62 h6144 m2560 E160 k8 (expert INT4 23.6MB) / 30B: L48 h2048 m768 E128 | config.json 추출 | 스냅샷 |
| H100: HBM 3.35TB/s | 스펙 | — |

## E0b — 계산기 `features/IDE_029/e0_model/placement_bound.py` (v2 roofline)

구조: GPU 항 (attn 가중치/step + KV×C + per-layer 오버헤드) + CPU 항 roofline (max(활성 expert 바이트/BW, FLOPs/C_eff) + call 고정비) × **pool 간 straggler 계수** (이항 분산, 파라미터-프리: 1+√(2/πn)) + PCIe 활성값 항. 하한: "활성 expert 가중치는 최소 1회/step 읽혀야 함" (양소켓 로컬 낙관치 390GB/s).

## E0c — 재예측 판정 (게이트: ±30% 또는 bracket 포함)

| # | 대상 | 예측 (v2) | 실측 | 판정 |
|---|---|---|---|---|
| r1 | 30B GPU-only/hybrid 비율 | 12.3× | ~15× | ✅ (−18%) |
| r2 | 480B S C=16 | bracket [15.5, 164] (균등↔완전집중 라우팅) | 44.5 | 🟡 bracket 포함 — 폭이 넓음, **ρ 미지** |
| r3 | 480B D 인스턴스 C=16 | bracket [8.3, 90.4] | 30.6 | 🟡 동일 |
| r4 | AMX knee 위치 | 구성상 재현 (곡선이 입력) | 75~129 | ✅ |
| r5 | K3 부호 (D vs S) | C=4 D승 ✓ / **C≥16 S승 ✗** | 전 구간 D승 | ❌ **구조 결함 발견** |
| r6 | kt+GPU-only 동거 무간섭 | 자원 분리 → 구조상 0 | ≈0% | ✅ |

**★ 하한 대비 gap (헤드라인 지표의 첫 계산)**: 균등-라우팅 하한 47.6 tok/s vs 실측 S 44.5 → **93%**. 단 라우팅이 집중이면 하한이 올라가 gap 은 더 벌어짐 — P0a 후 확정.

## r5 실패의 해부 (E0 의 핵심 산출)

v1 (E1 곡선 스케일링) 은 부호·격차축소 패턴을 재현했으나 절대값 3×↓, v2 (roofline) 는 절대값 bracket 은 맞지만 고부하에서 S 승 예측 (실측과 반대). 원인 = 스펙+마이크로벤치만으로 결정되지 않는 kt 내부 자유도 2개:

1. **P0a — 라우팅 집중 계수 ρ**: 균등 가정 (U=89.6 @C16) 은 실측 처리량과 모순 (물리적으로 44.5 불가) — 실제 활성 expert 수는 절반 이하로 추정. 라우팅 트레이스로 측정 (모델 속성 측정이지 성능 fitting 아님).
2. **P0b — 구성별 유효 BW/C_eff**: E1 은 96thr/2pool 구성만 측정. 48thr/1pool (D 구성) 의 BW_eff·C_eff 를 동일 microbench 로 측정해야 S/D 차등이 결정됨. threadpool=2 의 pool-join 비용도 여기서 분리.

→ **E0c 판정: 부분 통과 (4/6). 게이트 규정 ("대폭 이탈 → 모델 구조 재검토") 에 따라 P0a/P0b 를 E1 앞에 필수 삽입.** 모델 수정은 새 측정 입력 추가이지 end-to-end fitting 이 아니므로 §0 원칙 유지.
