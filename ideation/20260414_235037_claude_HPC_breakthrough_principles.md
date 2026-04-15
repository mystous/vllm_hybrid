# vLLM Hybrid 프로젝트 — HPC 돌파구 원칙

**작성**: 2026-04-14 23:50 KST (Claude)

본 문서는 **우리가 무엇을 할지, 어디로 갈지, 무엇을 하지 않을지** 를 결정하는 기준이다. 추상적 금언이 아니라 **갈림길마다 내리는 구체적 선택** 이다.

개별 로드맵/방안 문서가 바뀌어도 본 문서의 선택은 유지된다. 새 기법 등장 시 본 원칙 기준으로 채택/기각 판단.

---

## Part A — 우리가 가는 방향 (Destination)

**최종 목표 상태**:

> H100x8 서버에서 Qwen2.5-7B 500 req × 128/128 burst workload 에 대해 **hybrid wall < gpu_only × 1.0** 를 달성한다. 즉 CPU 가 실제 throughput 에 기여해 wall 을 줄인다.

이 목표가 유효한 이유: 현재 hybrid wall = gpu_only × **28×** (394s vs 14s). 28× 손해에서 +α 이득으로 뒤집는 것이 본 프로젝트의 존재 증명.

**경유 지점 (순서)**:

1. **CPU 자체 속도를 10× 올린다** — per-step 3079 ms → 300 ms. 이 단계에서는 hybrid wall 이 여전히 gpu_only 보다 느리지만, **격차가 3× 이하** 로 좁혀짐.
2. **Wall ≤ gpu_only × 1.5 달성** — 이 지점에서 Property 2 gate 가 실제로 CPU 를 선택하기 시작함.
3. **Wall < gpu_only × 1.0** — 최종 목표.

이 3단계를 뒤섞지 않는다. 1을 건너뛰고 2 를 시도 (구조 변경 only) 하면 CPU 가 느려서 Amdahl 에 걸림. 2 를 건너뛰고 3 을 노리지 않는다 (그건 다른 workload 이야기).

---

## Part B — 우리가 내리는 10가지 선택

### 선택 1. **CPU 자체 최적화를 1순위로** (vs CPU 역할 재정의)

**갈림길**: A) CPU kernel 을 10× 가속 / B) CPU 를 "다른 역할" (spec decoder, KV store) 로 재배치.

**우리 선택: A 먼저, B 는 병렬 트랙으로**.

- 근거: T-MAC 이 NPU 이긴 실측 (CPU 22 tok/s > NPU 10.4 tok/s). CPU 가 "느린 복제본" 인 것은 구현 문제. 구현 투자 없이 B 로 가면 **"felt slow CPU 를 그대로 두고 다른 곳에 붙임"**. Amdahl 재발.
- B 는 polish 단계에서 덧붙임. A 와 배타적 아님.

### 선택 2. **Kernel 을 직접 작성한다** (vs framework 위에서 설정)

**갈림길**: A) csrc/cpu 에 custom kernel 추가 / B) IPEX/oneDNN 최적화 옵션만 튜닝.

**우리 선택: A**.

- 근거: IPEX/oneDNN 은 범용성 목적으로 **보수적**. T-MAC / KTransformers 는 이들을 bypass 하고 직접 intrinsic 으로 구현해 2-4× 얻음. 우리가 같은 수준 가려면 csrc/cpu/ 확장이 필수.
- 대상: LUT GEMV (INT4), kernel fusion (Gate+Up, QKV concat), ISA dispatcher, AMX weight pre-pack.

### 선택 3. **Ops/Byte 를 측정하고 추적한다** (vs tok/s 만 본다)

**갈림길**: A) 매 기법마다 roofline 상 Ops/Byte 측정 / B) 최종 tok/s 만 비교.

**우리 선택: A**.

- 근거: tok/s 는 결과 지표. 개선 여지를 판단하려면 "이 기법이 Ops/Byte 를 몇 에서 몇 으로 올렸나" 가 필요. 현재 ~1-2 → 목표 ~8-32 (10× 영역 진입).
- 구현: `cpu_profile_dev.sh` 확장해 per-kernel Ops/Byte 출력. 각 Tier 종료 시 기록.

### 선택 4. **ISA 를 batch 크기에 따라 동적 선택** (vs AMX 고정)

**갈림길**: A) batch=1 에 AVX-512, batch≥8 에 AMX / B) 항상 AMX (IPEX 기본).

**우리 선택: A**.

- 근거: KTransformers 실측 AMX 가 batch=1 에서 AVX-512 보다 **2.22× 느림** (tile 초기화 + clock-down). 우리 H1 H100 per-step 3079 ms 의 일부가 이 origin 가능성.
- 구현: `cpu_worker.execute_model` 에서 current batch size 확인 후 dispatch 분기.

### 선택 5. **Weight 를 모델 로드 시 1회 재배치** (vs 매 step runtime)

**갈림길**: A) AMX tile layout 으로 pre-pack / B) PyTorch row-major 유지하고 kernel 이 매번 재배치.

**우리 선택: A**.

- 근거: KTransformers 10-20% 이득. 매 step 재배치 제거 = 매 step memory write 제거.
- 구현: CPUWorker `load_model` 후 hook 으로 weight 변환. Granularity = layer 단위.

### 선택 6. **Memory hierarchy 를 수동 제어** (vs OS/HW 자동)

**갈림길**: A) Huge Pages + `_mm_prefetch` + Intel DSA / B) 기본 4KB 페이지 + HW prefetcher 의존.

**우리 선택: A**.

- 근거: 70B 기준 TLB 엔트리 900만개 → 35개. 5-15% 즉시 이득. 코드 변경 최소 (grub 설정 + mmap flags).
- 구현 순서: (1) Huge Pages 1GB 설정 → (2) `_mm_prefetch` hotspot 삽입 → (3) DSA 는 Tier 2+ 에서.

### 선택 7. **Sublayer 를 묶는다** (vs 독립 kernel 호출)

**갈림길**: A) QKV concat, Gate+Up interleave, RMSNorm+Residual fused / B) 각자 torch op 호출.

**우리 선택: A**.

- 근거: 이미 SGLang 12% 이득 보고. 우리 sublayer 8개 체인 중 최소 4개 묶을 수 있음. 누적 1.5-2×.
- 구현: csrc/cpu/fused_*.cpp 단위. 각 fused kernel 별로 Ops/Byte 개선 측정.

### 선택 8. **느린 연산을 LUT 로 대체** (vs 그대로 두고 compute unit 늘림)

**갈림길**: A) exp/SiLU/INT4 dequant/곱셈 → LUT / B) AMX 에 더 태움.

**우리 선택: A**.

- 근거: T-MAN 실측 exp 20 cycles → 1 cycle (20×), SiLU 25 cycles → 1 cycle (25×). AMX 에 태워도 같은 연산이 빠를 수 없음 (근본적으로 transcendental).
- 구현: csrc/cpu/lut_ops.cpp. INT4 LUT 는 register 상주 (32B).

### 선택 9. **Batch 는 layout/fusion 선행 후 증가** (vs batch 먼저 키움)

**갈림길**: A) 선택 5+7 완료 후 `cpu_max_num_seqs` 실험 / B) 지금 `max_seqs=16` 다시 시도.

**우리 선택: A**.

- 근거: H100x8 H2 실측 — 현재 구현으로 batch=16 은 batch=1 의 5.3× 느림. 재앙. 원인은 "batch 자체" 가 아니라 layout 불일치 + fusion 부재. 먼저 고치고 batch 재시도.
- 경로: Tier 1 (fusion) + Tier 2 (pre-pack) 완료 후 batch 2-8 knee point 탐색.

### 선택 10. **매 단계 실측으로 원칙 검증** (vs 이론만 믿음)

**갈림길**: A) 기법 도입 후 원칙대로 개선됐는지 측정 / B) 논문 수치 믿고 다음 단계.

**우리 선택: A**.

- 근거: 이론 9-26× 누적이 실측에서 3× 로 끝날 수도 있음. 측정 없이 Tier 계속 쌓으면 **잘못된 방향에 시간 과투자**.
- 구현: 각 Tier 종료 시 `basic/H100x8/` 재실험 + Ops/Byte 기록 + codex `inspect.txt` 재생성. 예상 대비 50% 미만 달성이면 해당 Tier stop.

---

## Part C — 우리가 하지 않는 것

### 비-1. 하드웨어 세대 업그레이드 기대
Sapphire Rapids → Granite Rapids 는 BW/compute 증가지만 **원리 안 고치면 0.1% → 0.2% 도달률**. 본 프로젝트는 현재 HW 에서 peak 의 **10%+ 도달** 을 목표로.

### 비-2. "더 많은 thread" 실험
56 cores 이상은 L3 thrash + barrier cost. KVM 26.5 GB/s 환경은 16-24 에서 이미 포화. thread 수 탐색은 **1회 검증** 후 고정.

### 비-3. Python-level 최적화
`asyncio` 튜닝, ZMQ 옵션 조정 등은 overhead 1-2%. Hot path 는 **C++/intrinsic 만**. Python 은 orchestration 역할에만.

### 비-4. 알고리즘-only 개선 (HW-unaware)
양자화, spec decode, sparse attention 은 **HW-aware 구현과 병행해야** 효과. 알고리즘만 바꾸고 IPEX 에 얹으면 이득의 절반.

### 비-5. 범용성 목표
이 프로젝트는 **Intel Xeon SPR+ / H100 / AMX-capable** 타겟. AMD EPYC / ARM Graviton 지원 없음. 전용 kernel 은 이 특화를 전제로.

### 비-6. Long-context (16K+) / 70B workload 우선 추적
우리 baseline 은 7B × 128/128. KV offload / B2 Scout / A3 P-D 같은 "long-ctx 전용" 기법은 **Tier 3 별도 트랙**. 지금 추적 안 함.

---

## Part D — 실행 순서 (Sequencing)

원칙은 **순서가 있다**. 앞 단계 안 끝나면 뒤 단계가 무의미해질 수 있음.

```
[Tier 0] 설정 + 단일 호출 교체  (1주)
   ├─ Huge Pages 1GB (선택 6)
   ├─ IPEX WoQ INT8 (선택 3 측정 포함)
   └─ OMP env 검증 (선택 6 일부)
        │
        ▼ 성공 조건: per-step 3079 → <1000 ms
        │
[Tier 1] ISA 분기 + 기본 Fusion  (2-3주)
   ├─ AVX-512 decode kernel 명시 호출 (선택 4)
   ├─ Gate+Up / QKV concat (선택 7)
   └─ Softmax/SiLU LUT (선택 8 first)
        │
        ▼ 성공 조건: per-step <500 ms, Ops/Byte >4
        │
[Tier 2] LUT GEMV + Pre-pack      (3-4주)
   ├─ T-MAC INT4 LUT (선택 8)
   ├─ AMX weight pre-pack (선택 5)
   └─ AVX-512 bitmask sparse (선택 8 extension)
        │
        ▼ 성공 조건: per-step <200 ms, Ops/Byte >8
        │
[Tier 3] 구조 변경 + 장거리 워크로드  (선택 1 의 B track, 1-2달)
   ├─ Spec decode CPU drafter (A1)
   ├─ 3-stage DMA+Vec+Mat pipeline
   ├─ Core group pipeline (systolic)
   └─ 70B / long-context baseline
        │
        ▼ 최종 목표: hybrid wall < gpu_only × 1.0
```

각 Tier **성공 조건 불충족 시**: 다음 Tier 로 가지 않는다. 원칙 10 대로 원인 진단 후 **되돌아가거나 해당 Tier 재설계**.

---

## Part E — 우리 결정의 일관성 검증

본 원칙들은 서로 충돌하지 않아야 한다:

- 선택 1 (CPU 자체 최적화) × 선택 2 (kernel 직접 작성) — **부합**: 자체 최적화는 kernel 수준에서만 가능
- 선택 9 (layout 먼저 batch 나중) × 선택 5 (pre-pack) — **부합**: pre-pack 이 layout, 그 후 batch
- 선택 4 (ISA 동적) × 선택 11 (정적 parallelism) — **부합**: ISA 선택은 batch size 기반, batch 는 이미 결정된 상태
- 선택 3 (Ops/Byte 측정) × 선택 10 (실측 검증) — **부합**: 같은 방향

충돌 없음.

---

## Part F — 예외 조건

원칙 위반을 허용하는 경우:

1. **실측이 반증 시**: 선택 X 기반 기법 구현 후 예상 대비 50% 미만 → 해당 선택 재검토. (예: ISA 동적 전환이 AMX 대비 느리면 선택 4 철회)
2. **새 하드웨어 출시로 전제 변경 시**: Granite Rapids FP8 AMX 지원 등장 → 선택 4 의 "batch≥8 에 AMX" 재정의
3. **우선순위 충돌 시 — 지금은 이 길, 기록**: A 와 B 중 선택 1 이 A 우선이지만 resource 제약으로 B 를 먼저 해야 할 경우 → **명시적으로 기록** ("원칙 위반 계획, 근거, 복귀 시점").

---

## Part G — 본 문서 사용법

1. **새 기법 제안될 때**: Part B 10개 선택과 비교. 부합 수 >=7 이면 채택 후보.
2. **Tier 재설계 시**: Part D 순서 그대로 유지. 뒤집으면 Part E 일관성 깨짐.
3. **"이 방향이 맞나?" 질문 시**: Part A 의 3단계 경유 지점 중 현재 어디인지 먼저 확인.
4. **상위 우선순위 결정 갈등 시**: Part F 예외 조건에 해당하는지 명시.

이 원칙들은 **고정이 아님**. 실측이 반증하면 Part F 로 수정. 단 **즉흥적 예외** 는 피한다.

---

## 참고

- 돌파구 구체 기법: `20260414_233407_claude_hybrid_improvement_from_log_analysis.md`
- codex 우선순위 논의: `20260414_220744_codex_hybrid_improvement_directions_after_h100x8_rtx3090_log_review.md`
- 실측 근거: `eval/basic/H100x8/20260414_213434_claude_*.md`, `eval/basic/RTX3090/20260414_220000_claude_*.md`
- ideation morning 4종 (2026-04-14): 돌파구 정의 기반
