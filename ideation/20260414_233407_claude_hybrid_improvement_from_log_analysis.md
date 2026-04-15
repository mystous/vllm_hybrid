# Hybrid 개선 방향 — HPC 돌파구 × H100x8/RTX3090 실측 연결

**작성**: 2026-04-14 23:34 KST (Claude), **재작성**: 2026-04-15 (HPC 돌파구 반영)

**전제 문서**:
- ideation 2026-04-13: `0916_vLLM CPU-GPU`, `0933_B_plans`, `0944_deep-research`, `1400_consolidated_roadmap`
- ideation 2026-04-14 아침: `0942_intel_hetero_survey`, `0950_computation_dataflow_redesign` ⭐, `0950_cpu_llm_optimization_techniques`, `1005_HPC 베어메탈`
- 오늘 codex: `20260414_220744_codex_...`
- 실측 분석: `eval/basic/H100x8/20260414_213434_claude_...`, `eval/basic/RTX3090/20260414_220000_claude_...`

## 이 문서의 입장 (이전 판 정정)

이전 판은 "CPU 는 memory BW 5-10× 낮아 batch 이득 불가" 로 **BW 를 고정 상수** 처럼 다룸. **잘못.** `20260414_0950_computation_dataflow_redesign.md` 의 핵심 관찰:

> 7B BF16 decode 연산량 14 GFLOPs, AMX 30 TFLOPS → **이론 0.47 ms**. 실측 **400 ms**. 연산 0.5 ms + **데이터 대기 399.5 ms (99.9%)**.
>
> NPU/systolic 이 CPU 보다 빠른 것은 연산 속도가 아니라 **데이터 재사용 + dataflow 제어**.

즉 현재 CPU 구현의 BW 손실 399.5 ms 를 **7가지 HPC 돌파구** 로 해소하면 batch 이득도 복원 가능. T-MAC 실험이 증거: **NPU 10.4 tok/s < CPU 22 tok/s (2× 빠름)**.

---

## 1. 실측 재해석 (log analysis 와 HPC 연결)

### 1-1. Per-step 의 실체 재규정

| 환경 | batch | per-step | 분해 |
|---|---:|---:|---|
| H100x8 H1 (7B, AMX) | 1 | 3079 ms | compute ~10 ms + **data/overhead ~3069 ms** |
| H100x8 H2 (7B, AMX) | 16 | 16,390 ms | compute ~160 ms + **data/overhead ~16,230 ms** |
| RTX3090 H7 (7B, AVX2) | 4 | 703 ms | compute ~40 ms + **data/overhead ~660 ms** |

모든 환경에서 **data/overhead 가 95%+**. 이건 architecture 한계 아니라 **dataflow 미설계**. 이전 판에서 "overhead 지배 영역" 이라고 부른 것의 정체가 이것.

### 1-2. Batching 비선형 악화 재해석

이전 해석: "L3 scatter pressure"
HPC 문서 해석: **AMX tile layout 불일치 + 3-stage pipeline 부재 + kernel fusion 부재** 의 복합.

- batch↑ → weight read 는 amortize (여기까지는 이득)
- **그러나** kernel fusion 없이 sublayer 간 메모리 왕복 → 중간 결과 × batch 만큼 DDR write-back
- AMX tile 이 row-major 가 아닌 weight 를 만나면 매번 runtime 재배치 → batch 마다 반복

즉 batch=16 재앙은 architecture 문제 아니고 **구현 수준의 데이터 재사용 실패**.

### 1-3. codex/이전 판의 "spec decode / NEO / Scout" 방향 재평가

codex 는 "CPU 를 느린 복제본으로 쓰면 tail 생긴다 → 다른 역할 부여" 로 해석. 절반만 맞음. **다른 절반**: CPU 가 현재 속도에서 "느린 복제본" 인 것은 구현 미성숙. HPC 돌파구 7개를 적용해 CPU 를 **"복제본 but 훨씬 빠른"** 으로 만들면 request-level hybrid 도 유효해질 수 있음. 둘 다 추진 가치.

---

## 2. 7가지 돌파구 (ideation 0414 morning 문서 통합)

### B-1. LUT 기반 GEMV (T-MAC, Microsoft EuroSys'25)

**기법**: INT4 weight 는 16 값만 존재. `x × {0..15}` 16개 결과를 32B LUT 에 사전 계산 → 곱셈을 **`vpshufb` 1-cycle 테이블 참조** 로 대체.

**효과 (ideation 0950 redesign §3)**:
- 곱셈 + 역양자화 **완전 제거**
- INT4 에서 **2-4× 가속**
- Llama-2-7B 4bit: CPU T-MAC 22 tok/s > **NPU 10.4 tok/s 의 2×**
- INT2 로 낮추면 **선형 2× 추가** (일반 기법은 INT4→INT2 에 추가 이득 없음)

**batch 관점**: LUT access 가 **register 내** → batch 증가해도 memory BW 무관. batch 선형 이득.

**코드 위치**: `csrc/cpu/lut_gemv.cpp` 신규 + oneDNN dispatch 우회 + IPEX INT4 WoQ 대신 native 호출.

### B-2. ISA 동적 전환 (KTransformers, SOSP'25)

**핵심 발견**: AMX 는 **batch=1 에서 AVX-512 보다 2.22× 느림**. 이유:
- AMX tile 초기화 고정 비용
- AMX 활성 시 turbo 3.8→2.0 GHz down-clock
- 8 tile regs (8KB) 가 decode 의 작은 working set 에 oversized

**전략 (0950 optimization §1.1)**:
| 단계 | Ops/Byte | 최적 ISA |
|---|---|---|
| prefill 512+ tok | 50-500 | **AMX BF16** |
| decode batch=1 | 1-2 | **AVX-512 VNNI** |
| decode batch≥8 | 8-32 | **AMX INT8** |

**함의**: 우리 H100x8 H1 (batch=1) 은 **AMX path 로 돌아 3079ms** 일 가능성. AVX-512 로 전환하면 **1/2 이하** 가능. H2 (batch=16) 는 AMX 가 맞지만 **tile layout 불일치** 로 이득 못 봄.

**코드 위치**: `cpu_worker.py` 에서 decode step 진입 시 OP dispatch — IPEX 내부 dispatcher 무시하고 직접 선택.

### B-3. 3-Stage DMA-Vector-Matrix Pipeline (T-MAN NPU → CPU)

**NPU 원리**: DMA (tile 로드) + Vector (역양자화) + Matrix (GEMM) 가 **동시 실행** → 데이터 이동이 완전히 숨겨짐.

**CPU 대응 (0950 redesign §2)**:
```
Stage 1: Intel DSA 가 tile k+2 를 DDR → L3 비동기 prefetch
Stage 2: AVX-512 가 tile k+1 을 INT4 → BF16 역양자화
Stage 3: AMX TMUL 이 tile k 를 실제 GEMM
```

**효과**: T-MAN 실측 prefill 1.4×, **decode 3.1×**, 에너지 84% 절감.

**핵심 전제**: Intel DSA (Sapphire Rapids 이상) 사용. `enqcmd` 명령으로 프로그래머 제어. 우리 Xeon 8480+ 에 있음.

**코드 위치**: csrc/cpu 내 custom kernel. `libxsmm` JIT 조합 시 루프 언롤링까지.

### B-4. Core Group Pipeline (systolic 멀티코어 적용)

**현재 구조**: 56 cores 가 각자 독립 DDR read + 독립 gemm → **코어 간 데이터 공유 0**, 전부 weight 를 DDR 에서 중복 read.

**Systolic 매핑 (0950 redesign §6)**:
```
코어 그룹 A (0-13):  QKV proj            → L3 로 출력
코어 그룹 B (14-27): Attention + O proj  → L3 에서 QKV 읽고 L3 로 출력
코어 그룹 C (28-41): MLP (gate+up+down) → L3 로 출력
코어 그룹 D (42-55): 다음 layer QKV (pipeline 시작)

→ 4 layer 동시 pipeline 실행
```

**이점**:
- L3 공유 (300 GB/s) > DDR (250 GB/s) — L3 가 병목 아님
- 각 그룹은 **자기 sublayer weight 만 L2 에 유지** (working set 축소)
- **batch=16 의 L3 scatter thrash 회피** — 각 그룹이 다른 weight 만 봄

**GPU SM 간 shared memory** 와 유사한 dataflow 를 CPU L3 로 구현. **현재 IPEX 에는 이 구조 없음**.

### B-5. Kernel Fusion + Operator Substitution

여러 기법의 묶음:

**a) Gate + Up interleave (0950 redesign §7.1)**:
```
W_combined = interleave(W_gate, W_up)   # [M, 2K]
combined = W_combined × x               # 1 GEMV
gate, up = combined[::2], combined[1::2]
```
입력 x 를 **1회만 로드**. 2 GEMV → 1 GEMV. 메모리 접근 절반 감소.

**b) QKV fusion**: 같은 입력 x 로 3 GEMV → 1 GEMV. `W_qkv = concat(W_q, W_k, W_v)`.

**c) Softmax LUT**: `exp()` 20 cycles → 1-cycle 테이블 참조. **2.2× 가속** (T-MAN).

**d) SiLU LUT (TARDIS, arXiv 2501.10054)**:
- 입력의 80-90% 가 좁은 범위 → "hot range" 에서 `SiLU(x) ≈ a×x + b` 선형 근사
- 선형이면 **Gate × Up constant folding** 가능
- FFN weight **80% 감소**, vLLM **1.6× 가속**

**e) RMSNorm + Residual fusion**: 메모리 왕복 2회 → 1회.

각 1.1-1.8×, **곱해지면 9-26× 이론 누적** (0950 redesign §8).

### B-6. AVX-512 Bitmask Native Sparsity (ideation 1005 HPC)

**문제**: CSR/CSC 압축 포맷 은 branch misprediction + 메모리 낭비. Zero-padding 도 BW 낭비.

**해결**: AVX-512 `KMOVQ k1, rax` 로 **64-bit sparsity pattern 을 mask register 에 직접 로드**. `_mm512_mask_fmadd_ps(v, k1, w, x, acc)` 로 **유효 원소만 계산**.

**효과**:
- Unstructured 50% sparsity 에서 **1.5× + memory footprint 감소**
- SparAMX (Intel Labs, arXiv 2502.12444): linear 1.42×, attention 1.14×

**batch 불변성**: Polar Sparsity 는 "batch 시 MLP 희소성 소멸" 경고했지만 **attention head sparsity 는 batch-invariant** → B-6 는 batch 커도 유효.

### B-7. Huge Pages + AMX Weight Pre-pack (설정만, 즉시)

**Huge Pages (0950 optimization §3.2)**: 70B INT4 기준 4KB 페이지 → TLB 엔트리 **900만개** 필요 → 심각한 TLB miss. 1GB huge page 로 **35개** 로 감소.

```bash
sudo grubby --update-kernel=ALL --args="hugepagesz=1G hugepages=40 default_hugepagesz=1G"
```
→ **5-15% 향상** (설정 외 코드 변경 없음).

**AMX tile pre-pack (KTransformers)**: weight 를 로드 시점에 AMX tile layout (16×64 byte) 로 사전 배치. 런타임 재배치 오버헤드 제거. **10-20% 추가**.

---

## 3. 우리 코드와의 매핑 + 우선순위

### Tier 0 (즉시, ≤1주) — 설정 + 단일 호출 교체

| 기법 | 위치 | 비용 | 예상 |
|---|---|---|---|
| Huge Pages 1GB | 부팅 grub + mmap MAP_HUGETLB | 0.5일 | **5-15%** |
| IPEX WoQ INT8 | `cpu_worker.py` `ipex.llm.optimize` 인자 추가 | 2-3일 | decode **2×** |
| OMP env 최적화 검증 | `hybrid_core.py _setup_cpu_process_env` | 1일 | 5-15% |

이것만으로도 H100 H1 per-step 3079 → **~1000 ms** 가능.

### Tier 1 (1-2주) — ISA 동적 전환 + Fusion

| 기법 | 위치 | 예상 |
|---|---|---|
| B-2 ISA 동적 전환 (decode→AVX-512) | `cpu_worker.execute_model` pre-dispatch + csrc kernel 등록 | decode **1.5-2.2×** |
| B-5 Gate+Up interleave | IPEX 커널 wrapping + weight layout 변환 | **1.5-1.8×** |
| B-5 QKV fusion | 동일 | **1.2-1.3×** |
| B-5 Softmax/SiLU LUT | `csrc/cpu/lut_ops.cpp` | **1.1-1.2×** |

누적 ~3-5× 기대.

### Tier 2 (3-4주) — LUT GEMV + Weight Pre-pack

| 기법 | 위치 | 예상 |
|---|---|---|
| B-1 T-MAC LUT GEMV (INT4) | `csrc/cpu/lut_gemv.cpp` + dispatcher | **2-4×** |
| B-7 AMX weight pre-pack | model loader hook | **10-20%** |
| B-6 AVX-512 bitmask sparse | `csrc/cpu/sparse_amx.cpp` (SparAMX 기반) | **1.4×** |

이 단계 끝나면 7B CPU decode 이론 71 tok/s 에 근접 (현재 0.32 tok/s → **수십 tok/s**). GPU per-token 22ms (45 tok/s) 와 격차 2-3×.

### Tier 3 (1-2달) — 3-stage Pipeline + Core Group

| 기법 | 위치 | 예상 |
|---|---|---|
| B-3 3-stage (DSA+AVX+AMX) | csrc 대규모 재설계 | decode **3.1×** |
| B-4 Core group pipeline | scheduler + worker 분리 재설계 | latency **2-3×** |
| Spec decode drafter (A1) | 별도 엔진 + ZMQ | TPOT **2×** |

### 병렬 트랙 (우선순위 독립)

- **Property 2 gate 재검증**: CPU 가 실제로 빨라지면 router 가 CPU 선택하기 시작 → hybrid 수치가 `1.4×` 같은 현실적 값으로 돌아오는지 확인
- **cos similarity 측정** (ScoutAttention B-2 전제): 1일 실험
- **FP8 KV + chunked_prefill=True**: codex 권고, 즉시

---

## 4. 7가지 돌파구 ↔ 이전 문서 방안 매핑

| 이전 방안 | 재분류 |
|---|---|
| A1 spec decode | **Tier 3 병렬 트랙** (여전히 유효) |
| A2 KV offload / B3 KV INT4 | Tier 3 (long-context 조건부) |
| A3 P/D disagg | Tier 3 (long-context 조건부) |
| A4 AMX-INT8 / D1 IPEX WoQ | **Tier 0** (Huge Pages + INT8 WoQ) 로 흡수 |
| B1 NEO asymmetric | **B-4 Core Group Pipeline** 의 특수 케이스 |
| B2 ScoutAttention | Tier 2-3 (cos sim 검증 후) |
| B4 활성화 희소성 (TEAL) | **B-6 AVX-512 bitmask sparse** 로 흡수 |
| B5 AMX-INT8 WoQ | Tier 0 에 흡수 |
| Head Folding / VNNI pre-pack (사용자 제안) | **B-7 AMX weight pre-pack** + B-2 ISA 전환 |
| Sandwich (decode thread 축소) | B-2 ISA 전환의 한 측면 |

**요약**: 이전 방안들이 대부분 **돌파구 7개 중 하나의 특수 케이스**. 돌파구 프레임에서 재정렬하면 구현 순서가 훨씬 깨끗.

---

## 5. Success Metric (codex 권고 준수)

codex 문서 §3-3 의 "tail 제거가 1차 성공 기준" 을 유지하되 **정량 기준** 추가:

| 단계 | 성공 지표 |
|---|---|
| Tier 0 통과 | H100 H1 wall **394s → <200s** (Huge Pages + WoQ INT8) |
| Tier 1 통과 | H100 H1 per-step **3079ms → <1000ms** (ISA + fusion) |
| Tier 2 통과 | 7B CPU decode **>20 tok/s** (T-MAC LUT) |
| Tier 3 통과 | hybrid wall ≤ gpu_only × 1.5 (B-3/B-4 + spec decode) |

각 Tier 종료 시 `basic/H100x8/` 재실험 + codex `inspect.txt` 재생성으로 검증.

---

## 6. 본 문서 vs 이전 판 차별점 요약

| 축 | 이전 판 (초안) | 본 판 (재작성) |
|---|---|---|
| 전제 | BW 를 고정 상수로 | BW 는 고정이지만 **Ops/Byte 를 올릴 수 있음** |
| CPU 한계 | architecture 탓 | 구현 미성숙 (T-MAC 이 NPU 이긴 증거) |
| 1순위 | overhead 분해 측정 | **Tier 0 설정 + WoQ INT8 즉시** |
| 구조 변경 (A1/B1/B2) | 우선순위 | **돌파구 적용 후 평가** (CPU 가 빨라지면 일부 돌파구로 대체 가능) |
| roadmap 축 | "측정 3축" | **7가지 돌파구 + Tier 0-3** |

---

## 7. 최종 명제

**CPU 가 GPU 의 batch 구조를 못 쓴다는 건 거짓**. ideation 0414 문서들이 제시한 T-MAC, KTransformers, SparAMX, NPU 3-stage pipeline, AVX-512 bitmask 등은 **CPU 가 Ops/Byte 를 GPU 비슷하게 끌어올려 batch 이득을 얻는 구체적 경로**. CPU 가 NPU 이길 수 있다는 실증 (T-MAC 22 vs NPU 10.4 tok/s) 이 이미 존재.

따라서 우리 hybrid 최적화의 첫 번째 축은 **"CPU 를 포기하고 다른 역할 주기"** (codex) 가 아니라 **"CPU 를 제대로 구현하기"** (HPC 돌파구). 두 축 병행하되 **Tier 0-2 설정+커널 투자로 per-step 10× 이상 개선 가능** 하므로 이것부터.

---

## 참고

- ideation 20260414_0950_computation_dataflow_redesign.md ⭐ (가장 핵심)
- ideation 20260414_0950_cpu_llm_inference_optimization_techniques.md (KTransformers, SGLang, SparAMX 실측)
- ideation 20260414_1005_HPC 베어메탈 (RDMA, bitmask, LibXSMM, DMA)
- ideation 20260414_0942_intel_cpu_llm_hetero_survey.md (T-MAC, PowerInfer, Dovetail 연구)
- codex 20260414_220744_codex_hybrid_improvement_directions_after_h100x8_rtx3090_log_review.md
- eval/basic/H100x8/20260414_213434_claude_h100x8_cpu_execution_path_and_timeframe.md
- eval/basic/RTX3090/20260414_220000_claude_rtx3090_pattern_match.md
