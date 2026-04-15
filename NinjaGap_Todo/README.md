# 00. Overview — Ninja Gap TODO 디렉토리

**본 디렉토리 목적**: `TODO.md` 의 각 Ninja Gap 기법을 **독립 문서**로 분리해 기술적 배경 / 관련 참고 문헌 / 구현 메모 / 성공 조건 / 의존성 을 충실히 담는다. TODO.md 는 전체 요약 + 진행률 인덱스 역할.

---

## 목표

**Ninja Gap**: `T_hybrid < T_gpu_only` 달성. 현재 H100x8 (7B, 500×128/128) 에서 hybrid wall 이 GPU-only 대비 26–143× 느리다.

**실패 원인** (진단 확정):
- "CPU 가 느리다" 가 아님
- **`num_seqs` 증가에도 per-req cost 가 안 내려감 — batch scaling 실패**
- `cpu_max_num_seqs` 확대는 throughput 이 아닌 tail amplification

---

## 실행 순서 원칙

```
G0 (계측) → G1 (hot path 연결) → G2 (batch scaling) → G3 (big wins) → routing 재활성
```

이 순서를 건너뛰면 어떤 기법도 tail 만 만든다.

---

## 문서 인덱스 및 진행 체크

| # | 문서 | 기법 | Tier | 상태 | 진행 |
|---|---|---|:---:|:---:|:---:|
| 01 | [G0_measurement](./01_G0_measurement.md) | Tier -1 계측 재정의 | -1 | ⭕ | ☐ |
| 02 | [tier0_baseline_defense](./02_tier0_baseline_defense.md) | 기준선 방어 (cpu_max_num_seqs=1, wave-batch off) | 0 | 부분 | ☐ |
| 03 | [huge_pages](./03_huge_pages.md) | Huge Pages 1GB | 0 | ⭕ | ☐ |
| 04 | [ipex_woq_int8](./04_ipex_woq_int8.md) | IPEX WoQ INT8 | 0 | ⭕ | ☐ |
| 05 | [omp_env_finalize](./05_omp_env_finalize.md) | OMP env 마무리 (KMP_BLOCKTIME) | 0 | ✅ 거의 완료 | ☐ |
| 06 | [hot_path_wiring](./06_hot_path_wiring.md) | VNNI/pre-pack hot path 연결 (G1) | 1 | ⭕ | ☐ |
| 07 | [isa_binary_dispatch](./07_isa_binary_dispatch.md) | ISA Binary Dispatch (AVX-512 ↔ AMX) | 1 | 🔶 | ☐ |
| 08 | [kernel_fusion](./08_kernel_fusion.md) | Kernel Fusion (QKV / Gate-Up / Residual+Norm) | 1 | 🔶 | ☐ |
| 09 | [softmax_silu_lut](./09_softmax_silu_lut.md) | Softmax + SiLU LUT | 1 | ⭕ | ☐ |
| 10 | [head_folding](./10_head_folding.md) | Head Folding (GEMV → GEMM) | 1 | ⭕ | ☐ |
| 11 | [batch_aware_decode_attn](./11_batch_aware_decode_attn.md) | Batch-aware Decode Attention | 2 | 🔶 | ☐ |
| 12 | [barrier_sync_reduction](./12_barrier_sync_reduction.md) | Barrier/Sync 감소 (OMP persistent region) | 2 | ⭕ | ☐ |
| 13 | [tmac_lut_gemv_int4](./13_tmac_lut_gemv_int4.md) | T-MAC LUT GEMV INT4 | 2 | ⭕ | ☐ |
| 14 | [avx_amx_cascade](./14_avx_amx_cascade.md) | AVX/AMX Cascade Pipeline | 2 | ⭕ | ☐ |
| 15 | [amx_weight_prepack](./15_amx_weight_prepack.md) | AMX Weight Pre-pack (독자 제어) | 2 | 🔶 | ☐ |
| 16 | [sparamx_bitmask_sparse](./16_sparamx_bitmask_sparse.md) | SparAMX AVX-512 Bitmask Sparse | 2 | 🔶 | ☐ |
| 17 | [core_group_pipeline](./17_core_group_pipeline.md) | Core Group Systolic Pipeline | 3 | ⭕ | ☐ |
| 18 | [spec_decode_cpu_drafter](./18_spec_decode_cpu_drafter.md) | Spec Decode CPU Drafter (DuoDecoding) | 병행 | 🔶 | ☐ |
| 19 | [pd_disaggregation](./19_pd_disaggregation.md) | P/D Disaggregation | 장거리 | 🔶 | ☐ |
| 20 | [kv_offload](./20_kv_offload.md) | KV Cache CPU Tier Offload (InfiniGen) | 장거리 | 🔶 | ☐ |
| 21 | [scout_attention](./21_scout_attention.md) | ScoutAttention Layer-Ahead | 장거리 | ⭕ | ☐ |
| 22 | [neo_asymmetric](./22_neo_asymmetric.md) | NEO Asymmetric Batch Split | 장거리 | ⭕ | ☐ |

**진행 표시**: ☐ 미시작 / ▶ 진행중 / ✅ 완료 / ⏸ 보류 / ✗ 드롭

---

## Gate 정의

| Gate | 통과 조건 |
|---|---|
| G0 | seq=1/2/4/8/16 CPU-only scaling + sublayer breakdown 확보 |
| G1 | 4req cost ≤ 2× single / tail < 100s / wall ratio < 8× |
| G2 | 4req cost ≤ 1.5× / tail < 10s / wall ratio < 1.5× |
| G3 | CPU req↑ + tail 제거 + wall ≤ GPU-only (**Ninja Gap 달성**) |

Gate 숫자는 방향성. G0 기준선 재측정으로 조정.

---

## 성공 판정 4축

| 축 | 측정 | 방향 |
|---|---|---|
| CPU scaling | `cost(batch=N) / cost(batch=1)` | N 보다 훨씬 작아야 |
| CPU throughput | CPU-only tok/s, req/s | batch↑ 에 따라 ↑ |
| Tail | GPU bulk 완료 후 CPU drain | ↓ |
| Wall ratio | `hybrid wall / gpu_only wall` | ↓ |
| CPU contribution | handled req | ↑ |

**4축 동시 개선 아니면 성공 아님**.

---

## Stop/Go Rules

| Case | 관측 | 판정 | 조치 |
|---|---|---|---|
| 1 | CPU handled↑ but wall↓ | 실패 | `cpu_max_num_seqs` 확대 중단, kernel 단계 복귀 |
| 2 | CPU tok/s↑ but tail 그대로 | 부분 실패 | routing/gate/prefill 재검토 |
| 3 | 단일 req 만 빨라짐, batch scaling 없음 | 실패 | single-req 최적화로만 분류 |
| 4 | Kernel 수정 후 metric 변화 없음 | hot path 미타격 | marker 로 실제 호출 확인, 다음 kernel 금지 |

---

## 하지 말아야 할 것 (Guardrails)

1. batch scaling 확인 전 `cpu_max_num_seqs` 확대
2. `wave-batch` 를 기본 전략으로 유지
3. NUMA/pinning bring-up 재증명 (이미 실측 검증 완료)
4. 이미 구현된 기능 (chunked prefill off, NUMA membind, ISA 감지) 을 신규 gain 으로 재계산
5. VNNI/AMX 토대 존재만으로 "사용 중" 판단 — hot path 연결 여부는 별개
6. CPU handled request 수만 늘어도 wall 악화 시 성공으로 판단
7. 외부 논문 speedup 수치를 직접 곱하기

---

## 이미 구현된 항목 (gain 재계산 금지)

아래 항목은 완료 상태. **신규 성능 향상분으로 산출하지 않음**.

| 항목 | 근거 파일 |
|---|---|
| CPU engine launch (dual-process) | `hybrid_core.py`, `core_client.py` |
| wave-batch routing | `CapacityAwareRouter` |
| throughput-adaptive routing | `CapacityAwareRouter` |
| `cpu_max_num_seqs=1` auto baseline | `_resolve_cpu_params()` |
| CPU `chunked_prefill=False` | `_create_cpu_vllm_config()` |
| CPU core pinning (1:1) | `CPUWorker.init_device()` + `_C_utils.init_cpu_threads_env` |
| NUMA node 기반 CPU 선택 | `hybrid_config.numa_bind_node`, `_get_autobind_cpu_ids()` |
| NUMA memory strict bind | `numa_set_membind()` + `numa_set_strict(1)` in `csrc/cpu/utils.cpp` |
| affinity reset after fork | `_setup_cpu_process_env()` |
| feature 기반 ONEDNN ISA 설정 | `intel_cpu_utils.py` |
| VNNI INT8 GEMM 커널 빌드 | `csrc/cpu/gemm_vnni.*` (call-site 연결은 §06) |
| attn/mlp coarse profiling hook | `cpu_worker.py` forward hook (세분화는 §01) |

---

## 경로 1 누적 이론 상한

순차 적용 (diminishing returns 50% 가정):

| 기법 | 단독 이득 | 누적 | Gate |
|---|---:|---:|---|
| Baseline | 1× | 1× | |
| + Huge Pages (§03) | 1.1× | 1.1× | |
| + WoQ INT8 (§04) | 2.0× | 2.1× | |
| + OMP env (§05) | 1.05× | 2.2× | |
| + ISA binary (§07) | 2.0× | **3.3×** | G1 |
| + Fusion (§08) | 1.7× | 4.7× | |
| + LUT ops (§09) | 1.3× | 5.7× | |
| + Head Folding (§10) | 1.5× | **7.4×** | |
| + LUT GEMV INT4 (§13) | 3.0× | 13× | (WoQ 대체) |
| + Cascade (§14) | 1.7× | **19×** | G2 |
| + Pre-pack (§15) | 1.15× | 21× | |
| + Sparse (§16) | 1.35× | 27× | |
| + Batch-aware Attn (§11) | 1.5× | **35×** | G3 Ninja Gap |
| + Systolic (§17) | 2× | 70× | (overshoot) |

현재 cost_cpu / cost_gpu ≈ 28×. 경로 1 단독 역전 이론상 가능, 실제 30% 효율 가정 시 10–20× 구간 예상 → 경로 2 (spec decode) 조합 필요.

---

## Ninja Gap 달성 시나리오

| 시나리오 | 조건 | 확률 |
|---|---|---:|
| 경로 1 단독 승리 | T-MAC 4× + cascade 1.7× + batch-aware attn 12× scaling | 30% |
| **경로 1 + Spec Decode 조합 ★** | Stage C 후 15–20× + DuoDecoding 2× 추가 | **50%** |
| 구조 변경 필요 | 현 workload Ninja Gap 포기 → 70B/long-ctx 전환 | 20% |

---

## 근거 등급

- **A** (로컬 실측): H100x8 wall 394/2098/14s, RTX3090 wall 23/90/8.1/6.5s
- **B** (유사 HW 논문): SparAMX 1.42× (Xeon SPR), KTransformers ISA batch>4 경계
- **C** (edge/NPU/MoE 논문, 이식 시 재검증): T-MAC 48 tok/s, T-MAN 3.1×, DuoDecoding 2.61×
- **D** (강한 가설, 환경 미검증): AVX/AMX cascade, LUT GEMV on SPR+AMX, staging cache-fit

D 에 머무는 기법이 3개 연속 실패 시 드롭.

---

## 측정 모드 (VLLM_HYBRID_PROFILE)

G0 계측과 기법 ablation 은 **측정 전용 env flag** 로 제어. Production 실행은 flag off → hook overhead 없음.

### 2단 분리 구조

**Layer 1 — 측정 모드**
```bash
VLLM_HYBRID_PROFILE=0   # 기본. production.
VLLM_HYBRID_PROFILE=1   # 측정 모드. sublayer hook + manifest + 로그 활성
VLLM_HYBRID_PROFILE_EVERY=10   # N step 마다 로그 출력 (기본 10)
VLLM_HYBRID_PROFILE_SUBLAYER=1 # attn/mlp → qkv/o/gate_up/down/norm 세분화
```

**Layer 2 — 기법 feature flag** (아래 표. §03~§22 각 기법)

Layer 1 과 독립. 기법 flag 는 실제 동작 변경 (production 에서도 켤 수 있음), Layer 1 은 관찰만.

### 기법 Feature Flag 테이블

| Flag | 기법 | 기본값 | 현재 |
|---|---|:---:|:---:|
| `HYBRID_HUGEPAGES` | §03 Huge Pages 1GB | 0 | ⭕ |
| `HYBRID_WOQ_INT8` | §04 IPEX WoQ INT8 | 0 | ⭕ |
| `HYBRID_KMP_BLOCKTIME` | §05 OMP env (KMP_BLOCKTIME=0) | auto | ✅ |
| `HYBRID_VNNI_HOT_PATH` | §06 VNNI hot path wiring | 0 | ⭕ |
| `HYBRID_ISA_DISPATCH` | §07 ISA binary dispatch | auto | 🔶 |
| `HYBRID_KERNEL_FUSION` | §08 Kernel fusion | 0 | 🔶 |
| `HYBRID_LUT_SOFTMAX` | §09 Softmax LUT | 0 | ⭕ |
| `HYBRID_LUT_SILU` | §09 SiLU LUT | 0 | ⭕ |
| `HYBRID_HEAD_FOLDING` | §10 Head Folding | 0 | ⭕ |
| `HYBRID_BATCH_AWARE_ATTN` | §11 Batch-aware attn (v1/v2/off) | off | 🔶 |
| `HYBRID_PERSISTENT_OMP` | §12 Barrier/Sync 감소 | 0 | ⭕ |
| `HYBRID_TMAC_LUT_INT4` | §13 T-MAC LUT GEMV | 0 | ⭕ |
| `HYBRID_AVX_AMX_CASCADE` | §14 AVX/AMX Cascade | 0 | ⭕ |
| `HYBRID_AMX_PREPACK` | §15 AMX pre-pack (auto/ipex/custom) | auto | 🔶 |
| `HYBRID_SPARSE_BITMASK` | §16 SparAMX sparse | 0 | 🔶 |
| `HYBRID_CORE_GROUP_PIPELINE` | §17 Core group systolic | 0 | ⭕ |
| `HYBRID_SPEC_DECODE_CPU` | §18 Spec Decode CPU drafter | 0 | 🔶 |
| `HYBRID_PD_DISAGG` | §19 P/D disaggregation | 0 | 🔶 |
| `HYBRID_KV_OFFLOAD` | §20 KV offload (predictive) | 0 | 🔶 |
| `HYBRID_SCOUT_ATTN` | §21 ScoutAttention | 0 | ⭕ |
| `HYBRID_NEO_ASYMMETRIC` | §22 NEO asymmetric split | 0 | ⭕ |

**auto**: Intel feature 감지 기반 자동 결정 (기본 켜짐). **0**: 명시 opt-in 기법.

### 측정 모드 사용 흐름

**A. Production 실행** (기본):
```bash
# env 에 VLLM_HYBRID_PROFILE 언급 없음
./eval/serve.sh h100x8_qwen7b_hybrid.env
# hook 없음, overhead 0
```

**B. G0 측정 실행**:
```bash
# env 에 추가
VLLM_HYBRID_PROFILE=1
VLLM_HYBRID_PROFILE_SUBLAYER=1
HYBRID_VNNI_HOT_PATH=1   # 이번 주 적용 기법
HYBRID_BATCH_AWARE_ATTN=v2

./eval/serve.sh h100x8_qwen7b_hybrid.env
./eval/bench.sh h100x8_qwen7b_hybrid.env
```

Result dir 에 자동 생성:
- `applied_features.json` — 활성 flag + git sha + 모델 정보
- `hybrid_server_run.log` 에 `[HYBRID-APPLIED-FEATURES]` (boot) + `[HYBRID-CPU-PROFILE]` (per step) 마커
- `env_snapshot.txt` — `HYBRID_*`, `VLLM_HYBRID_*` 환경변수 덤프

**C. Feature ablation (같은 조건에서 1 flag 만 차이)**:
```bash
# run 1
VLLM_HYBRID_PROFILE=1 HYBRID_LUT_SOFTMAX=0 ./serve.sh ...

# run 2
VLLM_HYBRID_PROFILE=1 HYBRID_LUT_SOFTMAX=1 ./serve.sh ...

./eval/g0_compare.sh run1_dir run2_dir
# manifest diff 로 LUT_SOFTMAX 차이만 추출 + 성능 delta + attribution
```

### 주간 운영 (최소 2점, 여유 시 5점)

```
Week 0 (baseline):           [seqs=1, 16]  → curve_0
Week N (기법 적용 후):        [seqs=1, 16]  → curve_N
필요 시 중간점 추가:          [seqs=4]  → knee 확정
```

`eval/<HW>/g0_<NN>/seqs<N>/` 구조로 저장 (`<NN>` = 적용 TODO 번호, `00` = baseline, `05` = §05 KMP_BLOCKTIME 적용 후, `06` = §06 hot path 적용 후 …). `g0_compare.sh` 로 라운드 간 diff.

---

## 공통 참조 문서

- `/vllm_hybrid/TODO.md` — 전체 요약
- `/vllm_hybrid/ideation/20260415_1630_ninja_gap_superset.md` — 전체 기법 카탈로그
- `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md` — 실행 규율 + Stop/Go
- `/vllm_hybrid/ideation/20260415_094130_claude_ninja_gap_comprehensive_plan.md` — 3겹 실패 모델 + 이론 상한
- `/vllm_hybrid/ideation/20260415_1629_deep-research-report.md` — 두 계열 통합 + 불확실성 정리
- `/vllm_hybrid/ideation/20260415_092738_claude_HPC_breakthrough_principles_v4.md` — HPC 원칙 v4
- `/vllm_hybrid/eval/basic/H100x8/` — 실측 데이터
- `/vllm_hybrid/eval/basic/H100x8/analysis_h100.ipynb` — 분석 노트북
- `/vllm_hybrid/docs/paper/main.tex` — 설계 단일 진실 공급원
