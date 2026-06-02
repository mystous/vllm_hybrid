# Qwen2.5-32B vanilla / trident / AGSD on DGX B200 (6 workload × 3 scenario)

> **위치**: TSK_020 측정 산하 B200 결과 문서 (**SUB 미부여** — 단순 하드웨어 재측정/적용)
> **대조**: SUB_097 (H100) 의 B200 hardware cross-check + 확장
> **status**: ✅ **완료** (2026-06-01) — 18 cells, 전건 500/500
> **HW**: **DGX B200** (8× B200 183GB sm_100, Intel Xeon Platinum 8570 224스레드 2 NUMA, 2TB RAM) — docker 컨테이너
> **모델**: Qwen/Qwen2.5-32B-Instruct (bf16)
> **raw**: [`vllm_config_perf/gating/runs/agsd_32b_b200_v2_20260601/`](../../../../../vllm_config_perf/gating/runs/agsd_32b_b200_v2_20260601/) (v2 = 동일 8 GPU 공정 비교 + max_model_len 20480)

---

## 0. 설정 (★ 동일 자원 8-GPU 공정 비교)

모든 scenario 가 **8 GPU 전부** 사용하도록 구성 (이전 v1 의 vanilla/trident TP=4 → 절반만 쓰던 문제 교정):

| scenario | 구성 | GPU |
|---|---|---|
| **vanilla** | **TP=8 단일 인스턴스** (:8001, gmu 0.85, spec OFF, PIECEWISE) | 8 GPU |
| **trident** | **TP=8 단일 인스턴스** (:8002, gmu 0.80, **suffix K=32** + PIECEWISE) | 8 GPU |
| **AGSD** | TP=4×2 (vanilla GPU0-3 + trident GPU4-7) + router(:8000, chat→vanilla / sonnet·code→trident) | 4+4 GPU |

- **workload 6종** (각 500 prompts): pure sonnet / chat / code + mix balanced(34:33:33) / sonnet-heavy(60:20:20) / code-heavy(10:20:70)
- **length**: 8192 input × 8192 output, **max_model_len 20480** (입력 ~8.4k + 출력 8.2k 가 이전 16640 한계를 1토큰 초과해 400 에러 나던 문제 해결 → 전건 500/500 성공), concurrency 32, 1-run each
- **accuracy**: vanilla=trident=router 출력 텍스트 동일 확인 (suffix 출력 동등, CLAUDE.md 정확도 제약 충족)

### 선행 환경 작업 (B200)
- vllm 를 **sm_100 으로 재빌드** (`host_vllm_hybrid`, `TORCH_CUDA_ARCH_LIST=10.0`, 224 cores) — 기존 install 은 H100(sm_90) 빌드라 `cuTensorMapEncodeTiled` undefined 로 실패.
- `mxfp4_experts_quant.cu` 는 CUDA≥12.9 요구 → CMakeLists.txt 에 12.9 가드로 skip (FP4 MoE 전용, dense 추론 무관). 이 머신 CUDA 툴킷 12.8.
- `arctic-inference==0.1.1` 설치 (suffix). 상세: 메모리 `b200-vllm-build`.

---

## 1. 결과 — output_tps (전건 500/500, 동일 8 GPU)

| workload | vanilla TP=8 | trident TP=8 | AGSD TP=4×2 | trident vs vanilla | AGSD vs vanilla | **AGSD vs trident** |
|---|---:|---:|---:|---:|---:|---:|
| sonnet       | 3,040.5 | **6,757.2** | 6,574.9 | +122.2% | +116.2% | −2.7% |
| chat         | 2,932.8 | **4,616.3** | 2,568.0 | +57.4%  | −12.4%  | **−44.4%** |
| code         | 3,083.3 | **6,657.4** | 6,246.3 | +115.9% | +102.6% | −6.2% |
| balanced     | 2,913.4 | 6,309.1 | **6,930.2** | +116.6% | +137.9% | **+9.8%** ⭐ |
| sonnet-heavy | 2,908.1 | 6,474.3 | **6,797.8** | +122.6% | +133.7% | **+5.0%** ⭐ |
| code-heavy   | 3,011.3 | 6,527.0 | **7,183.6** | +116.7% | +138.6% | **+10.1%** ⭐ |

---

## 2. 핵심 관찰

### 2.1 trident(suffix) — vanilla 대비 6/6 net positive
동일 8 GPU·동일 500/500 조건에서 **suffix 가 모든 workload 에서 +57~+123%**. vanilla 는 spec OFF 라 2,908~3,083 으로 평탄, trident 는 4,616~6,757. chat 만 +57% 로 가장 작고(이미 acceptance 높아 suffix 여유가 작음), 나머지는 +115% 이상. → **순수 suffix 최적화 효과** (자원·완전성 동일).

### 2.2 AGSD(게이팅) — mix 에서 단일 trident 도 능가, pure 에선 손해
- **mix 3종**: AGSD 가 **단일 trident TP=8 보다 +5.0~+10.1%** (balanced +9.8 / sonnet-heavy +5.0 / code-heavy +10.1). 같은 8 GPU 인데도 **두 백엔드 병렬 게이팅이 단일 인스턴스보다 우수** → AGSD 설계 가치 입증.
- **pure 3종**: AGSD 가 단일 trident 보다 낮음. 특히 **pure chat −44.4%** — 라우터 정책상 chat→vanilla(spec 없는 4 GPU)로 가서 최악. pure sonnet/code 는 trident(TP=4)로 가서 trident TP=8 에 −3~−6% (4 GPU 라 약간 손해).

### 2.3 결론 (배포 권고)
- **mixed traffic** → **AGSD** (단일 trident 대비 +5~10%, 두 소켓 병렬 활용).
- **동질 traffic (단일 workload)** → **단일 trident TP=8** (특히 pure chat 은 AGSD 회피).
- chat→vanilla 라우팅 정책이 AGSD 의 약점 — mix 부하분산용으로 pure chat 을 희생. [IDE_022](../../idea/IDE_022_agsd_realistic_eval.md) 의 분류기·라우팅 재설계로 개선 여지.

---

## 3. vs SUB_097 (H100×8) 비교

| 구분 | SUB_097 (H100) | 본 측정 (B200) |
|---|---|---|
| vanilla/trident | TP=8 single (Phase A), 일부 cell 누락 | TP=8 single, **전건 500/500** |
| AGSD | TP=4×2 (Phase B, 200p×256max) | TP=4×2, **500p×8192×8192** |
| trident vs vanilla | sonnet +63.8% / code −0.8%(tie) | **sonnet +122% / code +116%** |

→ B200 + 전건 측정에서 **code 의 suffix 가속이 H100(거의 tie)과 달리 +116%** 로 크게 나옴. 단 setup(length·max_model_len·완전성)이 달라 직접 절대수치 비교보다 **"suffix 6/6 net positive + mix 에서 AGSD>trident"** 패턴의 hardware 재현으로 해석.

---

## 4. 전체 결과 병합

본 B200 18 cell 은 `measurements/_ALL_MEASUREMENTS.csv` 에 `sub=b200` / `model=Qwen2.5-32B-B200` 로 append 완료 (SUB 미부여, hardware 로 구분). vanilla/trident = `tp=8 / kind=single-tp8-b200`, AGSD = `tp=4 / kind=router-2inst-b200`.
