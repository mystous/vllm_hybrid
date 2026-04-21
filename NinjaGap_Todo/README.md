# 00. Overview — Ninja Gap TODO 디렉토리

**본 디렉토리 목적**: `TODO.md` 의 각 Ninja Gap 기법을 **독립 문서**로 분리해 기술적 배경 / 관련 참고 문헌 / 구현 메모 / 성공 조건 / 의존성 을 충실히 담는다. TODO.md 는 전체 요약 + 진행률 인덱스 역할.

---

## 목표 (2026-04-20 업데이트)

**Ninja Gap**: `T_hybrid < T_gpu_only` 달성.

**대표 workload**: Qwen2.5-32B × H100x8 (TP=8) × 500 req × 128/128. 7B + RTX3090 는 dev secondary.

**현재 실측 gap** (32B 기준): gpu_only outTP = 11,523 tok/s, hybrid 최고 (§06-1 v1 seqs=1) = 1,196 tok/s = **gpu_only 의 10.4%**.

**주원인**: CPU engine batch 병렬화가 구조적으로 불작동. seqs=1 → 8 에서 §06-1 v1 outTP 가 1196 → 272 (4.4× 감소), warmup 시간 기하급수 증가. 현재 CPU kernel 군 어느 것도 M>1 에서 amortize 를 못함.

**상세 원인 트리**: `Tech_done.md` v8 §SSOT-3 — 본 문서의 원인 서술은 그쪽을 따름.

---

## 실행 순서 원칙

```
G0 (계측) → G1 (hot path 연결) → G2 (batch scaling) → G3 (big wins) → routing 재활성
```

이 순서를 건너뛰면 어떤 기법도 tail 만 만든다.

---

## 활성 항목 (작업 축)

현재 코드 / 측정 상태에 직접 관여하는 항목만. **Tier 1 후보 우선순위는 §3-A.우선순위 참조**.

| # | 기법 | 상태 | 근거 등급 |
|---|---|---|---|
| 01 | G0 measurement | ✅ 완료 (2026-04-17, `22afea529`) | A |
| 02 | Tier 0 baseline defense | ✅ 완료 (2026-04-17, `22afea529`) | A |
| 05 | OMP env (KMP_BLOCKTIME=0) | ✅ 완료 (2026-04-15) | A |
| 06 | Q8_0 dispatch | 🔶 dispatch 완료 / kernel 결함 — §06-1 로 해결 (`6f904b39b`) | A |
| 06-1 | M-aware MLP kernel | 🔶 **v1 최종** (v2 기각) — `0c066f0e7`, 측정: `g0_06_1_qwen2.5_32b_v1/` | A |
| **22** | NEO asymmetric | ⭕ — **Tier 1 후보 (순위 1)** | B |
| **28** | xFasterTransformer 이식 | ⭕ — **Tier 1 후보 (순위 2)** | B |
| **13** | T-MAC LUT GEMV INT4 | ⭕ — **Tier 1 후보 (순위 3)** | C |

## 보관 / 기각 / 보류 (별도 backlog)

§03 (Huge Pages Phase 2 기각), §04 (WoQ INT8 기각), §07~§12/§14/§15/§17 (Tier 2 보류), §18 (강등), §19~§21 (장거리 보류), §23/§24/§25 (Tier 2 보류), §11 (Phase 1 기각), **§16 (SparAMX 기각 2026-04-20 — unstructured pruning 은 GPU 이득 없음, 2:4 로는 SparAMX 수치 근거 깨짐)** 전체 목록:

→ [old_doc/NinjaGap_backlog_tier2_20260420.md](../old_doc/NinjaGap_backlog_tier2_20260420.md)

**본 인덱스에서 제거된 이유**: 사용자 지적 (2026-04-20) — "활성 작업 문서에 보관/기각 항목 함께 두면 독자가 큰 표를 먼저 따라감". 본 인덱스는 진짜 작업 축만, 나머지는 backlog 로 분리.

---

## Gate 정의 (2026-04-20 재정의, baseline-relative)

이전 Gate (`cost(4)/cost(1) ≤ 2×`) 는 baseline §06 off 에서 이미 통과 (ratio=1.53, `Tech_done.md` v7) 로 gate 조건 자기 무효화. 재정의:

| Gate | **조건** | 현재 상태 |
|---|---|---|
| G0 | seq sweep + sublayer breakdown 확보 | ✅ 통과 |
| G1 | **hybrid outTP ≥ base(§06 off) outTP at seqs 1/2/4/8/16** — CPU engine 이 순손실 아님 | ✗ 미통과 |
| G2 | **hybrid outTP ≥ gpu_only outTP × 0.30 at any seqs** — α > 0 실효적 증거 | ✗ 미통과 (현 10.4%) |
| G3 (Ninja Gap) | **hybrid outTP ≥ gpu_only outTP at any seqs** | ✗ 미통과 |

SSOT: `Tech_done.md` v8 §SSOT-2.

---

## Gate ↔ NinjaGap 기법 매핑

각 Gate 를 통과하기 위해 적용되는 기법(문서 #). 측정 결과 저장 시 "어느 Gate 단계에서 어떤 기법을 적용했는지" 기록 기준.

| Gate | 통과 조건 | Tier | 적용 기법 (NinjaGap §) |
|---|---|:---:|---|
| **G0** | sublayer breakdown + seq sweep 계측 확보 | **-1** | **§01** G0 measurement |
| (G0 baseline) | 실험 오염 방지 가드 | **0** | **§02** Tier 0 baseline defense |
| (Tier 0 gain) | infra 기법 — gate 아닌 독립 이득 | **0** | ~~§03 Huge Pages~~ (**기각 2026-04-19** — Phase 1 기본 on, Phase 2 SPR TLB 구조상 역효과) · ~~§04 IPEX WoQ INT8~~ (**기각 2026-04-19** → §06 편입) · §05 OMP env (✅) |
| **G1** | hybrid outTP ≥ base outTP at seqs 1/2/4/8/16 | **1** | **§06** Q8_0 dispatch · **§06-1** kernel M-aware (v1 최종) |
| **G2** | hybrid outTP ≥ gpu_only × 0.30 at any seqs | **Tier 1 후보** | **§22 NEO** · **§13 T-MAC LUT INT4** · ⏸ §28 xFT 이식 (2026-04-21 보류) |
| **G3 (Ninja Gap)** | hybrid outTP ≥ gpu_only at any seqs | **Tier 1 후보** | Tier 1 후보 누적 + **§22 NEO asymmetric** (H100 70B 14.3% 실측) |
| 기각 / 강등 | — | — | **§11** Batch-aware attn (✗ Phase 1 기각 2026-04-20) · **§18** Spec Decode (⏸ 강등 2026-04-20, CPU drafter balance 미충족) |
| 보류 (Tier 2) | Tier 1 후보 후 재평가 | 2 | §07 ISA · §08 Fusion · §09 LUT · §10 Head fold · §12 Barrier · §14 Cascade · §15 Pre-pack · §17 Core group · §23 Q8_0/Q4_K · §24 W8A8 · §25 GQA batched |
| 장거리 (context 의존) | 현 workload 기여 제한적 | — | §19 P/D disagg · §20 KV offload · §21 ScoutAttn |

### 해석 (2026-04-20 업데이트)

- **G0 통과**, Tier 0 gain (§05) 적용, §06 dispatch + §06-1 v1 kernel 까지 반영된 현재 상태에서도 **G1 미통과**. 이유: §06-1 v1 이 base 대비 scope 일부 (seqs 2/4/8) 는 회복했지만 다른 seqs 에서 열세. 근본 원인은 CPU kernel 의 M>1 super-linear cost (SSOT: `Tech_done.md` v8 §SSOT-3)
- **다음 단계는 Tier 1 후보 중 선택**: §22 NEO asymmetric (우선순위 1) / §13 T-MAC LUT INT4 (우선순위 2). §16 (2026-04-20 기각) + §28 xFT (2026-04-21 보류, xDNN closed binary 의존 문제) 상태
- **§11, §18 은 재평가 대기 상태** (실패 가설 / 근거 불충분). 계획 본문에서 중심축으로 참조하지 않음
- **Tier 2 (§10/§14/§15/§17/§24/§25)** 는 원리만 있고 실측 수치 없음. Tier 1 후보 착수 후 실측 데이터 확보한 뒤에 재평가

### 측정 결과 저장 네이밍

`measurement_results/<HW>/g0_<NN>_<model>[_<suffix>]/seqs<N>/` 의 `<NN>` 을 기법 번호로:
- `g0_00_qwen2.5_32b_base` — Ninja Gap flag 전부 off (표준 baseline)
- `g0_06_qwen2.5_32b` — §06 dispatch only
- `g0_06_1_qwen2.5_32b_v1` — §06 + §06-1 v1 (kernel 최종 채택)
- `g0_06_1_qwen2.5_32b_v2(fail)` — §06-1 v2 (VNNI `vpdpbusd`) 기각 데이터 보존
- `g0_11_qwen2.5_32b_phase1(fail)` — §11 Phase 1 기각 데이터 보존
- 앞으로 Tier 1 후보 실측 시 각 §N 번호로 (예: `g0_13_qwen2.5_32b_v1/`)

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

## 무효화된 프레임 (2026-04-20, 본문에서 제거)

아래 항목들은 이전 README 에 있었으나 근거 붕괴로 본문에서 제거. 이력만 보관:

| 제거된 것 | 사유 | 보관 위치 |
|---|---|---|
| "경로 1 누적 이론 상한" 표 (Baseline 1× → §11 35× → Systolic 70×) | 논문 수치 직접 곱하기 — Guardrails 자기 위반. §06-1 v1 / §11 Phase 1 실측으로 전제 붕괴 | [../old_doc/NinjaGap_path1_theoretical_upper_bound_INVALIDATED_20260420.md](../old_doc/NinjaGap_path1_theoretical_upper_bound_INVALIDATED_20260420.md) |
| "Ninja Gap 달성 시나리오" 확률표 (경로 1 단독 30% / Spec Decode 조합 50% / 구조 변경 20%) | 위 이론 상한 표 전제. 시나리오 사고방식 자체가 소망 → 데이터로 대체 | 본 섹션 주석만 유지 |

**대체 기준** (scenario 확률 대신):
- **실측 gap**: gpu_only 11,523 / hybrid 최고 1,196 = 9.6×
- **기법 선택**: Tier 1 후보 3개 중 우선순위 1 (§22) → 2 (§28) → 3 (§13) 순차 검증. 각 단계에서 G1/G2 (SSOT: `Tech_done.md` v8 §SSOT-2) 재판정
- **곱셈 금지**: 논문 개별 수치를 누적 곱으로 표현 안 함

---

## 근거 등급

- **A** (로컬 실측): H100x8 wall 394/2098/14s, RTX3090 wall 23/90/8.1/6.5s
- **B** (유사 HW 논문): NEO 14.3% (H100 70B MLSys'25), xFasterTransformer (Intel 공식 SPR), KTransformers ISA batch>4 경계
- **C** (edge/NPU/MoE 논문, 이식 시 재검증): T-MAC 48 tok/s, T-MAN 3.1×, DuoDecoding 2.61×
- **D** (강한 가설, 환경 미검증): AVX/AMX cascade, LUT GEMV on SPR+AMX, staging cache-fit

D 에 머무는 기법이 3개 연속 실패 시 드롭.

### Tier 1 후보 (2026-04-21 업데이트 — §28 보류 반영)

보고된 실측 수치 + 우리 환경 일치도 + **모델 변경 여부** 순.

| 순위 | § | 보고 수치 | 측정 HW 일치도 | 등급 | 모델 변경 |
|:---:|---|---|---|:---:|---|
| **1** | §22 NEO asymmetric | throughput 14.3% | **H100 + 70B 동일** | **B** | 없음 (routing) |
| **2** | §13 T-MAC LUT INT4 | 4× | edge CPU (ARM) — **이식 리스크 큼** | **C** | 있음 (weight INT4 quant) |
| ⏸ | §28 xFasterTransformer 이식 | Intel 공식 SPR (블로그) | SPR production | **보류 (2026-04-21)** | 없음, 단 Intel closed binary 의존 |

**등급 근거 차이**: §22 는 우리 HW 에서 직접 실측, 모델 불변. §13 은 ARM edge 실측으로 SPR 재현 리스크 + weight INT4 quantization 모델 변경 수반. §28 은 Phase 0 조사에서 AMX kernel 이 xFT 소스가 아닌 xDNN (Intel 내부 라이브러리) 에 있음이 확인돼 "Apache-2.0 kernel 소스 이식" 전제 붕괴.

**기각 (2026-04-20)**: ~~§16 SparAMX~~ — unstructured pruning 이 GPU 에 이득 없음 (tensor core sparse 지원은 2:4 structured 전용). 2:4 로 바꾸면 SparAMX 논문의 1.42× 근거 깨짐.

**보류 (2026-04-21)**: ⏸ §28 xFT — AMX 성능이 xDNN closed binary 에 있음. 3 분기 (자체 AMX intrinsic 직접 구현 / xDNN 런타임 의존 수용 / §22 전환) 사용자 결정 대기. 상세는 [§28 문서](./28_xft_kernel_porting.md) "Phase 0 조사 결과" 섹션

**Tier 2 (원리만, 실측 수치 없음)**: §07~§12/§14/§15/§17/§23/§24/§25 — [backlog 참조](../old_doc/NinjaGap_backlog_tier2_20260420.md).

**실패 기록** (재시도 참조): §03 Phase 2 / §04 / §06-1 v2 / §11 Phase 1 — [backlog "기각" 표](../old_doc/NinjaGap_backlog_tier2_20260420.md#기각-확정-재시도-없음).

**공통 교훈**: 원리/인용만으로 Tier 2 기법을 Tier 1 확신으로 추진하지 말 것. SSOT: `Tech_done.md` v8 §SSOT-3.

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
| `HYBRID_HUGEPAGES` | §03 Huge Pages (**기각**, Phase 1 호스트 default, Phase 2 역효과) | 0 | ✗ |
| ~~`HYBRID_WOQ_INT8`~~ | §04 IPEX WoQ INT8 (**기각** 2026-04-19, §23 편입) | — | ✗ |
| `HYBRID_KMP_BLOCKTIME` | §05 OMP env (KMP_BLOCKTIME=0) | auto | ✅ |
| `HYBRID_VNNI_HOT_PATH` (→ `--hybrid-vnni-hot-path`) | §06 Q8_0 hot path wiring | 0 | 🔶 Phase A |
| `HYBRID_ISA_DISPATCH` | §07 ISA binary dispatch | auto | 🔶 |
| `HYBRID_KERNEL_FUSION` | §08 Kernel fusion | 0 | 🔶 |
| `HYBRID_LUT_SOFTMAX` | §09 Softmax LUT | 0 | ⭕ |
| `HYBRID_LUT_SILU` | §09 SiLU LUT | 0 | ⭕ |
| `HYBRID_HEAD_FOLDING` | §10 Head Folding | 0 | ⭕ |
| `HYBRID_BATCH_AWARE_ATTN` | §11 Batch-aware attn | off | ✗ **Phase 1 기각 (2026-04-20)**, flag 유지는 infra 존재 표시일 뿐. 재설계 보류 |
| `HYBRID_PERSISTENT_OMP` | §12 Barrier/Sync 감소 | 0 | ⭕ |
| `HYBRID_TMAC_LUT_INT4` | §13 T-MAC LUT GEMV | 0 | ⭕ |
| `HYBRID_AVX_AMX_CASCADE` | §14 AVX/AMX Cascade | 0 | ⭕ |
| `HYBRID_AMX_PREPACK` | §15 AMX pre-pack (auto/ipex/custom) | auto | 🔶 |
| ~~`HYBRID_SPARSE_BITMASK`~~ | ~~§16 SparAMX sparse~~ | — | ✗ 기각 (2026-04-20, pruning 모델 변경 필요 + GPU 이득 없음) |
| `HYBRID_CORE_GROUP_PIPELINE` | §17 Core group systolic | 0 | ⭕ |
| `HYBRID_SPEC_DECODE_CPU` | §18 Spec Decode CPU drafter | 0 | 🔶 |
| `HYBRID_PD_DISAGG` | §19 P/D disaggregation | 0 | 🔶 |
| `HYBRID_KV_OFFLOAD` | §20 KV offload (predictive) | 0 | 🔶 |
| `HYBRID_SCOUT_ATTN` | §21 ScoutAttention | 0 | ⭕ |
| `HYBRID_NEO_ASYMMETRIC` | §22 NEO asymmetric split | 0 | ⭕ |
| `HYBRID_CPU_NATIVE_QUANT` | §23 CPU Native Quant (`q8_0` / `q4_k` / `q8_0_w8a8` / `xft_int8`) | — | ⭕ |
| `HYBRID_W8A8_SMOOTHQUANT` | §24 SmoothQuant offline migration | 0 | ⭕ |
| `HYBRID_W8A8_OUTLIER_LOG` | §24 activation outlier 통계 기록 (debug) | 0 | ⭕ |
| `HYBRID_GQA_BATCHED_ATTN` | §25 GQA-aware batched attention | 0 | ⭕ |
| `HYBRID_XFT_KERNEL` | §28 xFasterTransformer bridge | 0 | ⭕ |

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

결과는 `eval/results/<ts>_.../`. PROFILE=1 은 sublayer hook + `applied_features.json` meta 기록만 추가. sweep 단위 정리 (`measurement_results/<HW>/g0_<NN>/seqs<N>/`) 는 **사용자가 수동 mv**. `g0_analyze.py` 도 수동 실행.

사용 흐름:

1. `cp eval/envs/g0_h100x8_qwen7b.env /tmp/run.env` (또는 dev template)
2. `HYBRID_TODO_NN`, `HYBRID_CPU_MAX_SEQS` 만 수정
3. `./eval/serve.sh hybrid /tmp/run.env` + `./eval/bench.sh hybrid /tmp/run.env` → 결과가 `eval/results/<ts>_.../` 에 `applied_features.json` 포함해서 저장
4. 필요 시 수동 `mv eval/results/<ts>_... measurement_results/<HW>/g0_<NN>/seqs<N>/`
5. `python3 eval/g0_analyze.py <sweep_dir>` — 표 + PNG + md 생성

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
