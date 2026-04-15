# vLLM Hybrid 프로젝트 — HPC 돌파구 원칙 v2

**작성**: 2026-04-15 00:02 KST (Claude)
**전신**: `20260414_235037_claude_HPC_breakthrough_principles.md` (v1)
**반영**:
- codex `20260414_220744_codex_*.md` 의 5가지 보강 포인트 (tail 기준, bring-up 완료 인식, machine-specific vs policy, B1 POC 기준, 전략 전환 조건)
- 외부 출처 재검증: T-MAC (EuroSys'25) / KTransformers (SOSP'25) / DuoDecoding (2025-03)

본 v2 는 **갈림길마다 내리는 구체적 선택** 이라는 v1 의 골격을 유지하면서, **성공 지표를 3축으로 명시** 하고 **전략 전환 조건을 결정 트리로 명문화** 한다.

---

## Part A — 우리가 가는 방향 (Destination)

### A-1. 최종 목표 상태

> H100x8 서버에서 Qwen2.5-7B, 500 req × 128/128 burst workload 에 대해
> **hybrid wall ≤ gpu_only × 1.0** 를 달성한다.

현재 hybrid wall = gpu_only × **28×** (394s vs 14s) → 이 28× 를 뒤집는 것이 프로젝트 존재 증명.

### A-2. 경유 지점 (순서)

3개 경유 지점 각각에 **3축 성공 기준** 을 둔다. codex §3-3 지적 반영:

| 경유 | 속도 축 (A) | Tail 축 (B) | Wall ratio 축 (C) |
|---|---|---|---|
| **G1. CPU 자체 10× 가속** | per-step 3079 → **<300 ms** | tail 길이 394s → **<50s** | hybrid/gpu × 28 → **×3.5 이하** |
| **G2. Property 2 gate 전환점** | per-step **<100 ms** | tail **<10s**, inflight 고착 없음 | **×1.5 이하** |
| **G3. 최종 돌파** | per-step **<50 ms** | **tail 제거** (GPU 완료 ≈ CPU 완료) | **×1.0 이하** |

**3축을 동시 통과** 해야 경유. 한 축만 좋으면 이전했다고 보지 않음.

### A-3. 왜 이 순서 (codex v2 보강)

codex 지적: "bring-up 은 완료됐다. 남은 질문은 CPU 가 어떤 시간축/역할을 맡으면 wall 을 안 늘리는가." → 이것은 G3 의 정의에 해당.

하지만 G1 을 건너뛰면 안 되는 이유: CPU 가 여전히 per-req 기준 GPU 의 28× 느리면, **어떤 역할 재정의 (spec decode / NEO) 도 Amdahl 에 걸림**. G1 (CPU 자체 10× 가속) 은 G3 도달을 위한 **선행 조건** 이지 경쟁 경로가 아님.

codex 입장 (CPU 를 다른 역할로) 과 본 문서 입장 (CPU 자체 최적화) 은 **병행**. 단 **진입 순서는 G1 먼저**.

---

## Part B — 우리가 내리는 10가지 선택 (v1 유지 + 근거 보강)

### 선택 1. CPU 자체 최적화를 1순위, 역할 재정의는 병행 트랙

**갈림길**: A) CPU kernel 10× 가속 / B) CPU 를 다른 역할 (spec decoder) 로 재배치

**선택: A 먼저, B 는 병렬**.

**근거 (외부 검증)**:
- [T-MAC (EuroSys'25)](https://arxiv.org/pdf/2407.00088) 가 CPU 에서 22 tok/s 를 달성해 **NPU 10.4 tok/s 를 2× 이김** (Llama-2-7B 4bit, Surface Laptop 7). "CPU 는 GPU/NPU 의 느린 복제본" 전제를 반증.
- [DuoDecoding (arXiv 2503.00784)](https://arxiv.org/abs/2503.00784) 의 2.61× 가속은 **CPU drafter 가 충분히 빠를 때만** 성립. "Llama-68m on CPU" 의 8 token auto-regressive 시간이 "Llama-2-7B on GPU" 의 8 token 병렬 verify 시간과 **일치** 해야 함. CPU drafter 자체가 느리면 DuoDecoding 도 안 통함.

→ B (역할 재정의) 도 결국 **CPU kernel 속도에 의존**. A 가 선행.

### 선택 2. Kernel 을 직접 작성 (vs framework 위에서 설정)

**갈림길**: A) csrc/cpu 에 custom kernel / B) IPEX/oneDNN 옵션 튜닝

**선택: A**.

**근거**:
- [KTransformers (SOSP'25)](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) 는 PyTorch native 대비 **3.9× 빠름** — IPEX/oneDNN 위에 올리지 않고 전용 AMX kernel 작성.
- AMX 21.3 TFLOPS 달성 (peak 112 TFLOPS 의 19%) — 동일 HW 에서 framework 기본 대비 훨씬 peak 에 근접.

**대상**: LUT GEMV (INT4), kernel fusion, ISA dispatcher, AMX weight pre-pack.

### 선택 3. Ops/Byte 를 측정하고 추적 (vs tok/s 만)

**갈림길**: A) 매 기법마다 roofline Ops/Byte 측정 / B) 최종 tok/s 만 비교

**선택: A**.

**근거**: 현재 decode Ops/Byte ~1-2. 목표 ~8-32 (10× 영역). tok/s 는 **결과**, Ops/Byte 는 **원인**. 원인을 추적해야 개선 여지 판단 가능.

**구현**: `cpu_profile_dev.sh` 확장해 per-kernel Ops/Byte 출력. 각 Tier 종료 시 전후 비교.

### 선택 4. ISA 를 batch 크기에 따라 동적 선택 (vs AMX 고정)

**갈림길**: A) batch=1 → AVX-512, batch≥4 → AMX / B) 항상 AMX

**선택: A**.

**근거 (외부 검증)**:
- [KTransformers AMX doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md) 에 명시: "AMX kernels are automatically selected during long prompt prefill phases (where each expert handles **more than 4 tokens on average**), while short prompt prefill and decode phases dynamically switch to **AVX-512 kernels**."
- 동일 HW 에서 **AMX peak 5.4 TFLOPS vs AVX-512 peak 1.8 TFLOPS** (고ARI 영역). 저ARI (decode) 에서는 AMX 가 **AVX-512 보다 2.22× 느림** (tile 초기화 + clock-down).
- 현재 우리 IPEX 구성은 batch 무관 AMX → H100x8 H1 (batch=1) per-step 3079ms 의 일부가 이 origin 가능성.

**구현**: `cpu_worker.execute_model` pre-dispatch hook, batch size > 4 → AMX, else → AVX-512.

### 선택 5. Weight 를 모델 로드 시 1회 재배치 (vs runtime 재배치)

**갈림길**: A) AMX tile layout 사전 pre-pack / B) PyTorch row-major 유지

**선택: A**.

**근거**: KTransformers "AMX tile-aware memory layout" — **런타임 재배치 오버헤드 제거**, 10-20% 이득. `tileloadd` 명령이 연속 16 cache line 을 한 번에 로드하도록 layout 최적화.

**구현**: CPUWorker `load_model` 후 hook 으로 weight 변환. layer 단위.

### 선택 6. Memory hierarchy 를 수동 제어 (vs OS/HW 자동)

**갈림길**: A) Huge Pages + `_mm_prefetch` + Intel DSA / B) 4KB 페이지 + HW prefetcher 의존

**선택: A**.

**근거**:
- 70B INT4 기준 4KB → TLB 900만 엔트리. 1GB huge page → 35 엔트리. 5-15% 즉시 이득.
- KTransformers 의 "지그재그 순회 패턴" + L2 prefetch 로 L2 활용률 극대화.

**구현 순서**: (1) Huge Pages 1GB grub 설정, (2) `_mm_prefetch` hotspot 삽입, (3) Intel DSA 는 Tier 2+.

### 선택 7. Sublayer 묶음 (vs 독립 kernel 호출)

**갈림길**: A) QKV concat, Gate+Up interleave, RMSNorm+Residual fused / B) 각자 torch op

**선택: A**.

**근거**: SGLang SiLU+up fused 가 **12% 이득** (단일 측). 우리 sublayer 8개 체인 중 최소 4개 가능. 누적 1.5-2×.

**구현**: `csrc/cpu/fused_*.cpp` 단위. 각 fused kernel 별 Ops/Byte 개선 측정.

### 선택 8. 느린 연산을 LUT 로 대체 (vs compute unit 에 더 태움)

**갈림길**: A) exp/SiLU/INT4 dequant/곱셈 → LUT / B) AMX 에 태워 가속

**선택: A**.

**근거 (외부 검증)**:
- T-MAC 의 핵심: **INT4 weight 16개 값 × input 을 pre-compute → `vpshufb` 1-cycle 테이블 참조**. 곱셈 + 역양자화 **완전 제거**. INT4→INT2 에서도 **선형 2× 추가** 가능 (기존 기법은 이 영역에서 추가 이득 없음).
- T-MAN (arXiv 2511.11248) 의 exp LUT: 20 cycles → 1 cycle, **2.2× Softmax 가속**.
- SiLU 근사 (TARDIS, arXiv 2501.10054): sigmoid+mul 25 cycles → linear approx 1 cycle, Gate × Up constant folding 가능 → **FFN weight 80% 감소, vLLM 1.6×**.

**구현**: `csrc/cpu/lut_ops.cpp`, `csrc/cpu/lut_gemv.cpp`. INT4 LUT 32B 는 register 상주.

### 선택 9. Batch 는 layout/fusion 선행 후 증가 (vs batch 먼저)

**갈림길**: A) 선택 5+7 완료 후 batch 실험 / B) 지금 `max_seqs=16` 다시 시도

**선택: A**.

**근거**: H100x8 H2 실측 — 현재 구현 batch=16 이 batch=1 의 5.3× 느림. "batch 자체" 가 아닌 **layout 불일치 + fusion 부재** 가 원인. 먼저 고치고 batch 재시도.

**경로**: Tier 1 (fusion) + Tier 2 (pre-pack) 완료 후 batch 2-8 knee point 탐색.

### 선택 10. 매 단계 실측으로 원칙 검증 (vs 이론만)

**갈림길**: A) 기법 도입 후 원칙대로 개선됐는지 측정 / B) 논문 수치 믿고 다음 단계

**선택: A**.

**근거**: 이론 9-26× 누적이 실측 3× 로 끝날 수 있음. 측정 없이 Tier 계속 쌓으면 **잘못된 방향 과투자**.

**구현**: 각 Tier 종료 시 `basic/H100x8/` 재실험, Ops/Byte 기록, codex `inspect.txt` 재생성. **예상 대비 50% 미만 달성 시 해당 Tier stop** (Part F 전략 전환).

---

## Part C — 우리가 하지 않는 것

### 비-1. 하드웨어 세대 업그레이드 기대

SPR → Granite Rapids 는 BW/compute 증가하지만 **원리 안 고치면 도달률 0.1% → 0.2%**. 본 프로젝트는 현재 HW 에서 peak **10%+ 도달** 목표.

### 비-2. "더 많은 thread" 실험

56 cores 이상은 L3 thrash + barrier cost. thread 수 탐색은 **1회 검증 후 고정**. profile 에서 peak 32-48 확인 완료.

### 비-3. Python-level 최적화

`asyncio` 튜닝, ZMQ 옵션 등은 overhead 1-2%. Hot path 는 **C++/intrinsic 만**. Python 은 orchestration 에만.

### 비-4. 알고리즘-only 개선 (HW-unaware)

양자화, spec decode, sparse attention 은 **HW-aware 구현과 병행해야** 효과. 알고리즘만 IPEX 에 얹으면 이득 절반.

### 비-5. 범용성 목표

**Intel Xeon SPR+ / H100 / AMX-capable** 특화. AMD EPYC / ARM 지원 없음. 전용 kernel 은 이 특화 전제.

### 비-6. Long-context (16K+) / 70B workload 우선 추적

baseline 은 7B × 128/128. KV offload / B2 Scout / A3 P-D 등 long-ctx 전용은 **Tier 3 별도 트랙**.

### 비-7. Bring-up 재증명 실험 (codex v2 보강)

CPU engine launch, NUMA 선택, OMP pinning, IPEX path 사용 — **이미 확정**. 다음 실험은 "CPU path 가 도는지" 가 아니라 **"성능이 개선되는지"**. bring-up 검증용 소규모 실험은 stop.

---

## Part D — 실행 순서 + Tier 성공 기준 (codex v2 3축 기준 적용)

```
[Tier 0] 설정 + 단일 호출 교체  (~1주)
   ├─ Huge Pages 1GB (선택 6)
   ├─ IPEX WoQ INT8 (선택 3 측정 포함)
   └─ OMP env 검증 (선택 6 일부)
        │
        ▼ 3축 성공 조건 (G1 경유):
        │   A. per-step 3079ms → <1000ms  (속도)
        │   B. tail 394s → <200s           (tail)
        │   C. hybrid/gpu 28× → <20×       (wall ratio)
        │
[Tier 1] ISA 분기 + 기본 Fusion  (2-3주)
   ├─ AVX-512 decode kernel 명시 호출 (선택 4)
   ├─ Gate+Up / QKV concat (선택 7)
   └─ Softmax/SiLU LUT (선택 8 first)
        │
        ▼ 3축 성공 조건:
        │   A. per-step <500ms, Ops/Byte >4
        │   B. tail <100s
        │   C. hybrid/gpu <10×
        │
[Tier 2] LUT GEMV + Pre-pack      (3-4주)
   ├─ T-MAC INT4 LUT (선택 8)
   ├─ AMX weight pre-pack (선택 5)
   └─ AVX-512 bitmask sparse (선택 8 ext)
        │
        ▼ 3축 성공 조건 (G2 경유):
        │   A. per-step <100ms, Ops/Byte >8
        │   B. tail <10s, inflight 고착 없음
        │   C. hybrid/gpu <1.5×   ← Property 2 gate 전환점
        │
[Tier 3] 구조 변경 + 장거리 워크로드  (1-2달)
   ├─ Spec decode CPU drafter (DuoDecoding 참조, 2.61× 이론)
   ├─ 3-stage DMA+Vec+Mat pipeline
   ├─ Core group pipeline (systolic)
   └─ 70B / long-context baseline
        │
        ▼ 3축 성공 조건 (G3 최종):
        │   A. per-step <50ms (CPU drafter path)
        │   B. tail 제거 (GPU ≈ CPU 완료 시점)
        │   C. hybrid/gpu ≤1.0×
```

**중요 규칙**:
- 한 축만 통과는 "경유" 아님. **3축 동시** 통과해야 다음 Tier.
- Tier 내 기법 각각은 Part B 선택 10 에 따라 측정 후 진행.

---

## Part E — 결정 트리 (새로 추가)

여러 길목에서 어떻게 판단할지.

### E-1. Tier 성공 조건 불충족 시

```
Tier N 종료 시 3축 성공 조건 체크
│
├─ 3축 모두 통과 → Tier N+1 진행
│
├─ 속도 축 (A) 만 통과, tail/ratio 불통과
│  └─ 원인 분석: CPU 빨라졌는데 wall 이 그만큼 안 준 이유?
│     ├─ router 가 CPU 선택 안 함 → Property 2 gate 파라미터 재조정
│     ├─ CPU prefill 직렬화 (chunked_prefill=False) → 실험 후보
│     └─ TPOT bimodal 극단 → batch knee 미도달, Tier 재설계
│
├─ 속도 축 (A) 불통과 (이론 대비 50% 미만)
│  └─ **해당 Tier 의 선택 N 재검토**
│     ├─ Tier 0 WoQ INT8 실패 → IPEX 호환성 문제 → Tier 1 로 점프 불가, 멈춤
│     ├─ Tier 1 ISA 분기 실패 → KTransformers 의 가정 무효 → 전략 전환 (Part F-3)
│     └─ Tier 2 LUT 실패 → T-MAC 의 가정 무효 → 전략 전환
│
└─ 하나도 통과 못 함
   └─ Part F-3 전략 전환 조건 발동
```

### E-2. 전략 전환 조건 (선택 1 의 A → B 전환)

**선택 1 은 "CPU 자체 최적화 1순위"**. 이게 실패하면 codex 의 B (CPU 역할 재정의) 로 전환.

**전환 발동 조건** (다음 중 2개 이상):
1. Tier 1 종료 시 per-step **>1000 ms** (목표 <500 ms 의 2배 초과)
2. Tier 2 종료 시 Ops/Byte **<4** (목표 >8 의 50% 미달)
3. Tier 2 종료 시 hybrid/gpu_only **>3×** (목표 <1.5 의 2배 초과)

**전환 시 즉시 할 일**:
1. Tier 2 까지 얻은 속도 (비록 목표 미달이라도) 를 baseline 으로 고정
2. Spec decode CPU drafter 에 리소스 재분배
3. 측정: DuoDecoding 의 "draft speed 가 verify speed 와 balance" 조건이 우리 HW 에서 성립하는지 — Qwen2.5-0.5B on CPU 의 8-token 시간 vs Qwen2.5-7B on GPU 의 8-token verify 시간
4. 성립 시 → 구조 변경 진행 (Part D Tier 3 로 점프)
5. 불성립 시 → **본 프로젝트의 request-level partition 구조로는 G3 달성 불가** 결론. Paper 에 negative result 로 기록, ideation 의 B3 (KV offload long-context) 로 방향 선회.

### E-3. machine-specific vs policy 판정 (codex §3-1 반영)

새 이슈 발견 시:
```
이슈 X 관찰
│
├─ RTX3090 dev 에서도 재현 → policy/code 문제
│  └─ dev 에서 fix + 검증 → H100 에서 재현 (codex 의 "dev 에서 완벽 검증 후 H100" 원칙)
│
├─ H100 에서만 나타남 → machine-specific
│  └─ 필요 조건: 2-NUMA / AMX / SMT / large cache / large BW
│     └─ 그 특정 기능 디버깅 (dev 재현 불가하므로 H100 직접)
│
└─ 판단 불가 → dev 먼저 돌려봄
```

RTX3090 pattern match 문서의 **P1-P7 재현 관찰** 이 이 원칙의 적용례.

---

## Part F — 예외 조건

**원칙 위반 허용** 은 다음 3가지 경우에만, 그리고 **명시적 기록** 조건부:

### F-1. 실측 반증

선택 X 기반 기법 구현 후 예상 대비 50% 미만 → Part E-1 결정 트리 따라 재검토. 해당 선택 철회 가능하나 **대안 명시** 필요.

### F-2. 새 HW 출시로 전제 변경

Granite Rapids FP8 AMX 지원, Intel XPU 통합 등으로 선택 4 (ISA 동적) 의 batch 경계 재정의 등. **원칙 v3 issuing** 으로 처리.

### F-3. 우선순위 충돌 (명시적 원칙 위반 계획)

인력/시간 제약으로 원칙 순서와 다르게 진행해야 할 때:
- **기록 필수**: 어떤 원칙을 위반했는지, 이유, 복귀 시점
- **예시**: "Tier 1 Gate+Up fusion 을 건너뛰고 Tier 2 LUT GEMV 로 직행 — 이유: 팀원 A 의 T-MAC 경험, 복귀 계획: LUT 완료 후 fusion 재방문"

---

## Part G — 외부 근거 (검증된 출처)

| 원칙 관련 | 근거 | URL |
|---|---|---|
| 선택 1 (CPU 자체 최적화) | T-MAC CPU 22 tok/s > NPU 10.4 tok/s | [T-MAC arxiv 2407.00088](https://arxiv.org/pdf/2407.00088) |
| 선택 1 (역할 재정의 병행) | DuoDecoding 2.61× (CPU drafter + GPU verifier) | [DuoDecoding arxiv 2503.00784](https://arxiv.org/abs/2503.00784) |
| 선택 2 (kernel 직접) | KTransformers 21.3 TFLOPS, PyTorch 대비 3.9× | [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) |
| 선택 4 (ISA 동적) | AMX 5.4 TFLOPS vs AVX-512 1.8 TFLOPS, batch>4 경계 | [KTransformers AMX doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md) |
| 선택 8 (LUT) | T-MAC vpshufb 기반 INT4 LUT, 선형 bit scaling | [T-MAC github](https://github.com/microsoft/T-MAC/) |
| 선택 8 (LUT Softmax) | T-MAN NPU, exp 20→1 cycle | [T-MAN arxiv 2511.11248](https://arxiv.org/html/2511.11248v1) |
| Part E-2 (전환 조건) | DuoDecoding CPU/GPU balance 원리 | [DuoDecoding github](https://github.com/KaiLv69/DuoDecoding) |
| 추가 참고 | LUT Tensor Core co-design | [LUT Tensor Core ISCA'25](https://dl.acm.org/doi/10.1145/3695053.3731057) |
| 추가 참고 | SAIL SRAM-accelerated LUT GEMV | [SAIL arxiv 2509.25853](https://arxiv.org/html/2509.25853) |
| 추가 참고 | T-SAR ternary CPU in-place SIMD | [T-SAR arxiv 2511.13676](https://arxiv.org/html/2511.13676v1) |

---

## Part H — v1 대비 변경점 요약

| 축 | v1 | v2 |
|---|---|---|
| 성공 기준 | 속도 축 (per-step) 단일 | **3축** (속도 + tail + wall ratio) |
| 경유 지점 정의 | 모호 (×1.5, ×1.0) | **G1/G2/G3 각각 3축 기준** |
| codex 입장 | 간략 언급 | **Part E-2 전략 전환 트리** 에 통합 |
| Bring-up 재증명 | 없음 | **비-7** 추가 |
| machine-specific 판정 | 없음 | **Part E-3** 결정 트리 |
| 외부 근거 | 간접 언급 | **Part G** 출처 검증 매트릭스 |

---

## Part I — 사용법

1. **새 기법 제안 시**: Part B 10개 선택 비교 → 부합 수 ≥7 이면 채택 후보
2. **Tier 재설계 시**: Part D 순서 유지. 뒤집으면 Part E 일관성 깨짐
3. **"이 방향 맞나?" 검토 시**: Part A G1/G2/G3 중 현재 어디인지 + 3축 체크
4. **속도만 개선되고 wall 안 줄 때**: Part E-1 결정 트리
5. **Tier 실패 시**: Part F-1 (실측 반증) + Part E-2 (전략 전환)

본 원칙은 **고정이 아님**. 실측이 반증하면 Part F 로 수정. 단 **즉흥적 예외 없이 기록 의무**.

---

## 참고 문서

- v1: `20260414_235037_claude_HPC_breakthrough_principles.md` (본 문서가 대체)
- 돌파구 구체 기법: `20260414_233407_claude_hybrid_improvement_from_log_analysis.md`
- codex 우선순위 논의: `20260414_220744_codex_hybrid_improvement_directions_after_h100x8_rtx3090_log_review.md`
- 실측 분석: `eval/basic/H100x8/20260414_213434_claude_*.md`, `eval/basic/RTX3090/20260414_220000_claude_*.md`
- ideation morning 4종 (2026-04-14): 돌파구 정의 기반
