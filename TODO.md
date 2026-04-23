# vLLM Hybrid — Ninja Gap 성능 향상 작업 (v7, 2026-04-20)

작업 이력: `Task_done.md` / 기술 검증 결론: `Tech_done.md` / 설계 단일 진실 공급원: `docs/paper/main.tex` / 프로젝트 구성: `CLAUDE.md`.

**운용 규칙**: "남은 성능 향상 작업만" 유지. 이전 버전 `old_doc/TODO_v5_20260415.md` 에 스냅샷 보존. 완료는 `Task_done.md` 에 append.

**v7 변경 (2026-04-20)**: `Tech_done.md` v8 SSOT 반영 — 대표 workload 를 Qwen2.5-32B TP=8 로 고정, Gate 를 baseline-relative 로 재정의, §06-1 상태 동기화, §11 핵심축 강등, "경로 1 누적 이론 상한" 표 INVALIDATED, §18 spec decode scenario 확률 강등.

---

## 핵심 메시지 (2026-04-20 기준)

> 대표 workload: **Qwen2.5-32B-Instruct × H100x8 (TP=8) × 500 req × 128/128**. 실측 기준 **gpu_only outTP = 11,523 tok/s**, **hybrid 최고치 (§06-1 v1 seqs=1) = 1,196 tok/s = gpu_only 의 10.4%**.
>
> 주원인: **CPU engine batch 병렬화 자체가 구조적으로 불작동**. §06-1 v1 에서 seqs=1 → seqs=8 까지 outTP 가 1196 → 272 (4.4× 감소), warmup 시간 기하급수 증가. 현재 CPU kernel 군 어느 것도 M>1 에서 amortize 를 못함.
>
> 상세 원인 트리: `Tech_done.md` v8 §SSOT-3 참조. 문서 원인 서술은 그쪽을 참조, 여기는 요약만.
>
> 7B + RTX3090 는 dev 검증용 secondary. 의사결정은 32B 기준만.

---

## 하지 말아야 할 것 (Guardrails)

1. batch scaling 확인 전 `cpu_max_num_seqs` 확대
2. `wave-batch` 를 기본 전략으로 유지
3. NUMA/pinning bring-up 재증명 (이미 실측 검증 완료)
4. 이미 구현된 기능 (chunked prefill off, NUMA membind, ISA 감지) 을 **신규 gain 항목으로 재계산**
5. VNNI/AMX 토대 존재만으로 "INT8/AMX 사용 중" 판단 — **hot path 연결 여부는 별개**
6. CPU handled request 수만 늘어도 wall 악화 시 성공으로 판단
7. 외부 논문 speedup 수치를 우리 코드에 직접 곱하기

---

## 1. 성공 판정 4축

Stage 종료 시마다 **동시 확인**:

| 축 | 측정 | 방향 |
|---|---|---|
| CPU scaling | `cost(batch=N) / cost(batch=1)` | N 보다 훨씬 작아야 |
| CPU throughput | CPU-only tok/s, req/s | batch↑ 에 따라 ↑ |
| Tail | GPU bulk 완료 후 CPU drain | ↓ |
| Wall ratio | `hybrid wall / gpu_only wall` | ↓ |
| CPU contribution | handled req | ↑ |

**4축 동시 개선 아니면 성공 아님**.

### Stop/Go Rules

| Case | 관측 | 판정 | 조치 |
|---|---|---|---|
| 1 | CPU handled↑ but wall↓ | 실패 | `cpu_max_num_seqs` 확대 중단, kernel 단계 복귀 |
| 2 | CPU tok/s↑ but tail 그대로 | 부분 실패 | routing/gate/prefill 재검토 |
| 3 | 단일 req 만 빨라짐, batch scaling 없음 | 실패 | single-req 최적화로만 분류 |
| 4 | Kernel 수정 후 metric 변화 없음 | hot path 미타격 | marker 로 실제 호출 확인, 다음 kernel 금지 |

---

## 2. Gate 정의 (2026-04-20 재정의, baseline-relative)

이전 Gate 조건 (`cost(4)/cost(1) ≤ 2×`) 은 `Tech_done.md` v7 §06 측정에서 baseline 이 이미 통과 (ratio=1.53) 함이 확인됨. Gate 조건으로 유지 불가 → 재정의:

| Gate | **재정의 조건** (baseline-relative) | 현재 상태 |
|---|---|---|
| G0 | seq sweep + sublayer breakdown 확보 | ✅ 통과 |
| G1 | **hybrid outTP ≥ base(§06 off) outTP × 1.0 at seqs 1/2/4/8/16** — CPU engine 이 순손실 아님 | ✗ 미통과 (§06-1 v1 기준 seqs 일부 우위 / 일부 열세) |
| G2 | **hybrid outTP ≥ gpu_only outTP × 0.30 at any seqs** — α > 0 의 실효적 증거 | ✗ 미통과 (현재 최고 10.4%) |
| G3 (Ninja Gap) | **hybrid outTP ≥ gpu_only outTP at any seqs** | ✗ 미통과 |

단일 진실 공급원: `Tech_done.md` v8 §SSOT-2.

---

## 3. G0 — 계측 재정의 (모든 후속 기법의 전제) ✅ 완료 (2026-04-17, 22afea529)

- [x] `eval/cpu_profile*.sh` 에 `num_seqs=1/2/4/8/16` sweep 고정
- [x] CPU-only 와 hybrid CPU engine 동일 shape 비교 harness
- [x] `cpu_worker.py` attn/mlp coarse hook → QKV/O/Gate/Up/SiLU/Down/Norm 세분화
- [x] per-step barrier/sync time, memory wait, packing/repacking marker
- [x] H100x8 + dev (RTX3090) 결과 동일 CSV schema 로 저장
- [x] 산출물: `batch_scaling_ratio`, `per_req_cost`, sublayer top bottleneck, `num_seqs` 증가 시 폭증 sublayer

---

## 3-A. 현재 진행축 (2026-04-21 업데이트 — §28 Phase 0 결과 반영)

§11 Phase 1 기각 이후 방향 재정립. §16 기각 (2026-04-20) + §28 보류 (2026-04-21) 반영.

| 순위 | 후보 | 보고 수치 | 측정 HW 일치도 | 증거 등급 | 모델 변경 |
|:---:|---|---|---|---|---|
| **1** | [§22 NEO asymmetric](NinjaGap_Todo/22_neo_asymmetric.md) | throughput **14.3%** | **H100 + 70B 동일** | B (MLSys'25 실측) | 없음 (routing) |
| **2** | [§13 T-MAC LUT INT4](NinjaGap_Todo/13_tmac_lut_gemv_int4.md) | INT4 **4×** | edge CPU (ARM) | C (이식 리스크 큼, SPR 재검증 필수) | 있음 (weight INT4 quant) |
| ⏸ | [§28 xFasterTransformer 이식](NinjaGap_Todo/28_xft_kernel_porting.md) | Intel 공식 SPR 실측 | SPR production | **보류 (2026-04-21)** | 없음, 단 Intel closed binary (xDNN) 의존 |

**우선순위 근거**: §22 는 우리와 동일 HW + 모델 규모 (H100+70B) 실측, 모델 변경 없음. §13 은 weight INT4 quantization (§06 Q8_0 의 공격적 버전). §28 은 Phase 0 조사에서 AMX kernel 이 xFT 소스가 아닌 Intel 내부 xDNN 라이브러리에 있음이 확인돼 "Apache-2.0 kernel 이식" 전제 붕괴. 사용자 판단 대기 중.

**기각 / 보류**:
- ~~§16 SparAMX~~ (2026-04-20 기각) — unstructured pruning 은 GPU tensor core sparse 미지원, 2:4 로 바꾸면 SparAMX 논문 수치 근거 깨짐
- ⏸ §28 xFT (2026-04-21 보류) — AMX 성능이 xDNN closed binary 에 있음. 3 분기 (A: 자체 AMX intrinsic 구현 / B: xDNN 런타임 의존 수용 / C: §22 전환) 사용자 판단 대기. 상세는 [§28 문서](NinjaGap_Todo/28_xft_kernel_porting.md) "Phase 0 조사 결과" 섹션

**착수 규율**: 한 번에 한 후보만. G1 (hybrid outTP ≥ base) 또는 G2 (hybrid outTP ≥ gpu_only × 0.30) 재판정 후 다음으로 이동. 실패 시 다음 후보. 모두 실패 시 Tier 2 backlog 에서 선정.

---

## 4. 현재 남은 구현 작업 (2026-04-20)

현재 코드 상태 / 진행 중인 항목만. **기각 / 보류 / 이력 은 `old_doc/NinjaGap_backlog_tier2_20260420.md` 로 분리**.

### 4.1 §06 / §06-1 현재 상태 (Tier 1 후보 착수 전 baseline)

| 항목 | 상태 | 상세 |
|---|---|---|
| **§06 Q8_0 dispatch** | 🔶 dispatch 완료, kernel 결함 | `hot_path_wiring.py` + `_Q8_0LinearMethod` + CLI arg (`6f904b39b`). seqs=1 outTP +18%, seqs≥2 역효과. kernel 결함은 §06-1 에서 수정 |
| **§06-1 v1 (최종)** | 🔶 v1 kernel 최종 채택 (v2 기각) | v1 `0c066f0e7` (weight reuse GEMM). §06 on 대비 seqs 2/4/8 +21~34% 회복, base 대비 일부 열세. M>1 구조적 결함은 v1 으로 해결 안 됨 (SSOT: `Tech_done.md` v8 §SSOT-3) |

v2 (VNNI `vpdpbusd`) 기각 상세: [§06-1 문서](NinjaGap_Todo/06-1_m_aware_mlp_kernel.md), 측정: `measurement_results/H100x8/g0_06_1_qwen2.5_32b_v2(fail)/`

### 4.2 Tier 1 후보 (§3-A 착수 대상)

우선순위 1~3 은 §3-A 참조. 각 기법 상세는 개별 NinjaGap_Todo 문서:
- [§22 NEO asymmetric](NinjaGap_Todo/22_neo_asymmetric.md)
- [§28 xFasterTransformer 이식](NinjaGap_Todo/28_xft_kernel_porting.md)
- [§13 T-MAC LUT INT4](NinjaGap_Todo/13_tmac_lut_gemv_int4.md)

### 4.3 기각 / 보류 항목

Tier 2 / 장거리 / 기각 / 강등 전체 목록은 [old_doc/NinjaGap_backlog_tier2_20260420.md](old_doc/NinjaGap_backlog_tier2_20260420.md). 본문 중복 제거.

---

## 5. 실행 순서

1. **§3-A 우선순위 1 (§22 NEO asymmetric) 착수** — 모델 변경 없는 routing 축. H100+70B 14.3% 실측
2. G1 / G2 재판정 (`Tech_done.md` v8 §SSOT-2 기준)
3. 성공 시 우선순위 2 (§28 xFT) 또는 Tier 2 재평가. 실패 시 우선순위 2 로 이동
4. Tier 1 후보 3개 모두 실패 시 Tier 2 backlog 재평가

---

## 6. 참조 (이력 / 기각 / 보류)

현재 TODO 본문에서 제외된 항목들:

| 종류 | 위치 |
|---|---|
| Tier 2 (보류) / 장거리 / 기각 / 강등 목록 | [old_doc/NinjaGap_backlog_tier2_20260420.md](old_doc/NinjaGap_backlog_tier2_20260420.md) |
| "경로 1 누적 이론 상한" 표 (invalidated) | [old_doc/NinjaGap_path1_theoretical_upper_bound_INVALIDATED_20260420.md](old_doc/NinjaGap_path1_theoretical_upper_bound_INVALIDATED_20260420.md) |
| 실패 측정 데이터 | `measurement_results/H100x8/g0_06_1_qwen2.5_32b_v2(fail)/`, `measurement_results/H100x8/g0_11_qwen2.5_32b_phase1(fail)/` |
| 과거 TODO 스냅샷 | `old_doc/TODO_v5_20260415.md`, `old_doc/TODO_v4_20260411.md` |
| SSOT (원인 트리 / Gate 정의 / 대표 workload) | [Tech_done.md](Tech_done.md) v8 §SSOT-* |
| 완료 작업 이력 | [Task_done.md](Task_done.md) |
| 프로젝트 구성 요약 | [CLAUDE.md](CLAUDE.md) |
| 설계 단일 진실 공급원 | `docs/paper/main.tex` |

---

## 7. 코드 수정 위치 총괄

- **계측**: `vllm/v1/worker/cpu_worker.py` (sublayer hook, barrier marker), `eval/cpu_profile*.sh` (num_seqs sweep)
- **라우팅**: `vllm/v1/engine/hybrid_core.py` (default strategy, cpu_max_num_seqs, wave-batch, throughput-adaptive), `vllm/v1/engine/core_client.py`
- **CPU hot path**: `vllm/v1/attention/backends/cpu_attn.py` (IPEX vs custom 분기), `csrc/cpu/*` (VNNI / fusion / LUT / AVX/AMX), `vllm/v1/worker/cpu_model_runner.py`

---

