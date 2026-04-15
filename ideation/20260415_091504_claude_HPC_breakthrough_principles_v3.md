# vLLM Hybrid 프로젝트 — HPC 돌파구 원칙 v3

**작성**: 2026-04-15 09:15 KST (Claude)
**전신**:
- v1: `20260414_235037_claude_HPC_breakthrough_principles.md`
- v2: `20260415_000227_claude_HPC_breakthrough_principles_v2.md`

**본 v3 반영**:
- codex `20260415_085858_codex_hybrid_improvement_integrated_rewrite.md` 의 4가지 유효 지적
  - ① compute/data 분해는 **모델링**, 실측 미검증 → tagging
  - ② 정량 기대치 (wall 394→<200) 근거 약함 → 논문 수치 × baseline 공식으로 재정의
  - ③ **true batch scaling** 이 중심 문제 (per-step 속도 아님)
  - ④ Tier 0 앞에 **계측 단계** 필요
- v2 의 유지할 장점:
  - WebSearch 검증된 외부 출처 매트릭스
  - 전략 전환 결정 트리 (선택 1 실패 시 DuoDecoding 방향 발동)
  - machine-specific vs policy 판정 트리

---

## Part A — 우리가 가는 방향 (Destination)

### A-1. 최종 목표 상태

> H100x8 서버에서 Qwen2.5-7B, 500 req × 128/128 burst workload 에 대해 **hybrid wall ≤ gpu_only × 1.0** 를 달성한다.

현재 hybrid wall = gpu_only × **28×** (394s vs 14s). 이 28× 를 뒤집는 것이 프로젝트 존재 증명.

### A-2. 문제의 중심 — Batch Scaling (v3 재정의)

v2 까지 "per-step 속도" 를 성공 지표 중심에 두었다. **v3 는 codex 의 지적을 수용해 "true batch scaling" 을 중심에 둔다.**

> 여러 request 를 CPU 에 동시에 보내도 per-request cost 가 충분히 내려가지 않는 것 — 이것이 현재 구조적 실패 모드.

실측 근거:
- H100x8 `cpu_max_num_seqs=1`: CPU tail 394s (2 req × single cost)
- H100x8 `cpu_max_num_seqs=16`: CPU tail 2098s (32 req × 거의 같은 single cost)
- RTX3090 `cpu_max_num_seqs=4`: tail 90s (4 req × batch=4 cost)

즉 **batch 를 늘려도 per-req cost 가 선형 감소 안 함** → `inflight` 증가가 throughput gain 이 되지 않음 → tail 확대.

**따라서 v3 의 중심 문제**: "num_seqs 1→N 에서 per-req cost 가 1/N 에 근접하는 scaling 을 만드는 것."

### A-3. 경유 지점 — 3축 성공 기준 재정의

| 경유 | **Batch scaling 축 (A, 신규)** | Tail 축 (B) | Wall ratio 축 (C) |
|---|---|---|---|
| **G0. 계측 기준선** | num_seqs 1→4 scaling ratio 실측 | 현재 tail 실측 | baseline 고정 |
| **G1. CPU scaling 개선** | num_seqs=4 per-req cost 가 num_seqs=1 대비 **≤2×** | tail **<50s** | hybrid/gpu **≤3.5×** |
| **G2. Property 2 gate 전환점** | num_seqs=4 per-req cost **≤1.5×** | tail **<10s**, inflight 고착 없음 | **≤1.5×** |
| **G3. 최종 돌파** | num_seqs=4 per-req cost **≤1.2×** | **tail 제거** (GPU ≈ CPU 완료) | **≤1.0×** |

**v2 대비 변경**:
- 축 A (속도) → **축 A (batch scaling ratio)** 로 재정의
- G0 계측 기준선 신설 (아무 기법도 적용 안 한 현재 상태의 scaling 실측)
- 3축 동시 통과 원칙 유지

### A-4. 왜 이 경로 (v2 동일 + codex 입장 수용)

codex 입장: "CPU 가 다른 시간축/역할을 맡도록 바꿔야" (role-level)
claude 입장: "CPU 자체 구현을 제대로 하면 되돌아온다" (kernel-level)

**v3 는 codex 의 "batch scaling 이 중심 문제" 를 수용하되, kernel-level 투자를 통해 batch scaling 을 만들 수 있다는 claude 입장을 유지**. Part F-3 전환 조건 발동 시 role-level 로 자동 스위치.

---

## Part B — 우리가 내리는 10가지 선택 (v2 유지 + 근거 정정)

### 선택 1. CPU 자체 최적화를 1순위, 역할 재정의는 병행 트랙

**선택: A 먼저 (CPU kernel 가속), B 는 병렬 (spec decode)**.

**근거** (WebSearch 검증):
- [T-MAC (EuroSys'25)](https://arxiv.org/pdf/2407.00088): 3B BitNet single-core 20 tok/s, 4-core **48 tok/s**, 기존 CPU low-bit baseline 대비 **4-5× speedup** (codex 2-2 재인용). "CPU 가 batch/low-bit 에 원천적으로 불리" 가설 반증.
- [DuoDecoding (2025-03)](https://arxiv.org/abs/2503.00784): CPU drafter + GPU verifier **2.61×**. 단 draft throughput 이 verify throughput 과 balance 되어야 성립 — **CPU kernel 속도가 선행 조건**.

### 선택 2. Kernel 을 직접 작성 (vs framework 설정)

**근거**: [KTransformers (SOSP'25)](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) 는 PyTorch 대비 3.9× — IPEX/oneDNN 위가 아닌 전용 kernel 작성. AMX 21.3 TFLOPS (peak 112 TFLOPS 의 19%).

**v3 주의**: KTransformers 수치는 MoE 중심이므로 우리 Dense workload 에 직접 이식 시 **수치 재검증 필요** (codex 지적 §3-2 반영).

### 선택 3. Batch Scaling Ratio 를 주 지표로 추적 (v3 신설, Ops/Byte 대체)

**갈림길**: A) num_seqs sweep per-req cost 감소율 / B) 기존 "Ops/Byte" 기준

**선택: A (v2 의 선택 3 에서 변경)**.

**근거 (codex ③ 수용)**: Ops/Byte 는 compute-centric 지표. batch scaling 실패는 **compute 가 아니라 sync/memory/parallel efficiency** 에서도 옴. num_seqs sweep 이 더 직접적.

**구현**: `cpu_profile_dev.sh` 에 num_seqs 1/2/4/8/16 별 per-req tok/s 측정 + sublayer 시간 분해 + NUMA diff.

### 선택 4. ISA 를 batch 크기에 따라 동적 선택

**근거 (v2 검증 유지)**: [KTransformers AMX doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md): "AMX kernels... more than 4 tokens on average, while... decode phases dynamically switch to **AVX-512 kernels**." AMX 5.4 TFLOPS vs AVX-512 1.8 TFLOPS (고 ARI).

### 선택 5. Weight 를 모델 로드 시 1회 재배치

(v2 동일)

### 선택 6. Memory hierarchy 를 수동 제어

(v2 동일)

### 선택 7. Sublayer 묶음 + **Attention batch-aware 분리** (v3 보강)

**v3 에서 추가**: codex 4-2 후보 #1, #3 수용.
- **Head Folding** (SGLang blog): GEMV → GEMM 변환, AMX tile full 활용
- **Batch-aware decode attention**: IPEX `single_query_cached_kv_attention` 의 KV paged scatter 문제 해소 — 별도 kernel 필요

### 선택 8. 느린 연산을 LUT 로 대체

**근거 (v2 동일)**: T-MAC `vpshufb` INT4, T-MAN exp/SiLU LUT.

**v3 주의**: T-MAC 은 edge CPU (Snapdragon X, Surface) 검증. Xeon SPR + AMX 조합에서 **수치 재검증 필요**.

### 선택 9. Batch 는 layout/fusion 선행 후 증가

(v2 동일)

### 선택 10. **계측 우선 실증주의** (v3 강화)

**v2 "매 단계 실측" → v3 "계측을 Tier 0 이전 단계로 승격"**.

**근거 (codex ④)**: 계측 없이 기법 적용하면 "어느 기법이 어느 축 (compute/BW/sync) 에 영향" 판단 불가. Part E-1 결정 트리도 실측 없으면 작동 안 함.

**구현**: Part D 의 Tier -1 (신설) 에 통합.

---

## Part C — 우리가 하지 않는 것 (v2 동일)

비-1~비-7 전부 유지. 특히:
- **비-7 Bring-up 재증명** (codex 수용)
- **비-3 Python-level 최적화**
- **비-6 long-context / 70B workload 우선** (현재 workload 7B × 128/128 고정)

---

## Part D — 실행 순서 (v3 재구성: Tier -1 계측 추가)

```
[Tier -1] 계측 기준선 (NEW, ~3일)         ← codex ④ 수용
   ├─ num_seqs=1/2/4/8/16 CPU-only tok/s sweep
   ├─ per-req latency 감소율 측정 (scaling ratio)
   ├─ Sublayer 분해 (QKV / Attn / O / Gate+Up / SiLU / Down / Norm)
   ├─ NUMA-local vs non-local diff
   ├─ compute vs sync vs memory-wait 비중 (PROFILE marker 활용)
   └─ 현재 hybrid wall 구성 분해 (GPU bulk 완료 시점 / CPU tail 길이)
        │
        ▼ G0 기준선 확보: num_seqs=4 가 num_seqs=1 대비 per-req cost 몇 배인가
        │   (현재 실측 미기록 — Tier -1 종료 시 확정)
        │
[Tier 0] 설정 + 단일 호출 교체 (~1주)
   ├─ Huge Pages 1GB (선택 6)
   ├─ IPEX WoQ INT8 (선택 3 측정 포함)
   └─ OMP env 검증
        │
        ▼ 3축 성공 조건 (G1):
        │   A. num_seqs=4 per-req cost ratio ≤ baseline × 0.5 (개선 50%)
        │   B. tail 394s → <200s
        │   C. hybrid/gpu 28× → <15×
        │
[Tier 1] ISA 분기 + 기본 Fusion (2-3주)
   ├─ AVX-512 decode kernel (선택 4)
   ├─ Gate+Up / QKV concat (선택 7)
   ├─ Head Folding (선택 7 보강)
   └─ Softmax/SiLU LUT (선택 8 first)
        │
        ▼ G1 통과 조건:
        │   A. num_seqs=4 per-req cost ≤ 2× num_seqs=1
        │   B. tail <100s
        │   C. hybrid/gpu <8×
        │
[Tier 2] LUT GEMV + Pre-pack + Batch-aware Attention (3-4주)
   ├─ T-MAC INT4 LUT (선택 8)
   ├─ AMX weight pre-pack (선택 5)
   ├─ Batch-aware decode attention (선택 7)
   └─ AVX-512 bitmask sparse (선택 8 ext)
        │
        ▼ G2 통과 조건 (Property 2 gate 전환점):
        │   A. num_seqs=4 per-req cost ≤ 1.5×
        │   B. tail <10s, inflight 고착 없음
        │   C. hybrid/gpu <1.5×
        │
[Tier 3] 구조 변경 + 장거리 워크로드 (1-2달)
   ├─ Spec decode CPU drafter (DuoDecoding)
   ├─ 3-stage DMA+Vec+Mat pipeline
   ├─ Core group pipeline (systolic)
   └─ 70B / long-context baseline
        │
        ▼ G3 최종 조건:
        │   A. num_seqs=4 per-req cost ≤ 1.2×
        │   B. tail 제거
        │   C. hybrid/gpu ≤ 1.0×
```

**v2 대비 변경**:
- **Tier -1 (계측)** 신설
- 각 Tier 성공 조건의 A 축을 **batch scaling ratio** 로 재정의
- 정량 기대치 (하이브리드 ratio) 는 **논문 수치 × baseline 공식** 으로 유도:
  - Tier 0: T-MAC INT8 2× × Huge Pages 1.1× = 2.2× 가속 → 28×/2.2 ≈ 12.7× → **ratio <15×** 설정
  - Tier 1: + ISA 2.22× × fusion 1.5× = 총 7.3× 가속 → 28/7.3 ≈ 3.8× → **ratio <8×**
  - Tier 2: + LUT 4× × pre-pack 1.15× = 총 34× 가속 → 28/34 ≈ 0.82× → **ratio <1.5×**

숫자는 여전히 가설이나 **출처 추적 가능** 해짐.

---

## Part E — 결정 트리 (v2 유지 + 보강)

### E-1. Tier 성공 조건 불충족 시 (v2 동일)

Tier N 종료 후 3축 중 일부 불통과:
- 속도/batch scaling 만 통과: router 정책 재조정 (chunked_prefill, Property 2 gate 파라미터)
- 속도 축 불통과: 해당 Tier 의 선택 N 재검토

### E-2. 전략 전환 조건 (v2 동일, batch scaling 축으로 재정의)

**선택 1 (CPU 자체 최적화) 실패 발동 조건** (2개 이상):
1. Tier 1 종료 시 num_seqs=4 per-req cost **>3× num_seqs=1** (목표 ≤2× 의 1.5× 초과)
2. Tier 2 종료 시 num_seqs=4 scaling **>2×** (목표 ≤1.5 의 초과)
3. Tier 2 종료 시 hybrid/gpu_only **>3×**

발동 시: DuoDecoding 방향 (spec decode CPU drafter) 으로 Tier 3 즉시 진입. Qwen2.5-0.5B on CPU 의 draft throughput 이 Qwen2.5-7B on GPU 의 verify throughput 과 balance 되는지 실측 → 성립 시 진행, 불성립 시 **request-level hybrid 의 구조적 한계 결론**, paper negative result.

### E-3. machine-specific vs policy 판정 (v2 동일)

이슈 → RTX3090 재현 시도 → dev 에서 재현되면 policy/code 문제 → dev 에서 수정 → H100 검증.

---

## Part F — 예외 조건 (v2 동일)

F-1 실측 반증 / F-2 HW 변경 / F-3 우선순위 충돌 시 명시적 기록 조건부 원칙 위반 허용.

---

## Part G — 외부 근거 (WebSearch 검증, v2 + codex 추가)

| 원칙 | 근거 | URL |
|---|---|---|
| 선택 1a (CPU 자체 가능성) | T-MAC 3B BitNet 4-core 48 tok/s, 4-5× baseline | [T-MAC EuroSys'25](https://arxiv.org/pdf/2407.00088), [github](https://github.com/microsoft/T-MAC) |
| 선택 1b (역할 재정의) | DuoDecoding 2.61× + TTFT 17%↓ | [DuoDecoding arxiv 2503.00784](https://arxiv.org/abs/2503.00784) |
| 선택 2 (kernel 직접) | KTransformers 21.3 TFLOPS, PyTorch 3.9× | [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) |
| 선택 4 (ISA 동적) | AMX 5.4 TFLOPS vs AVX-512 1.8 TFLOPS, batch>4 경계 | [KTransformers AMX doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md) |
| 선택 7 (Head Folding + batch-aware attn) | codex 4-2 #1, #3 | codex `20260415_085858` |
| 선택 7 (sublayer fusion) | SGLang SiLU+up 12% 이득 | [LMSYS blog](https://lmsys.org/blog/2025-10-22-KTransformers/) |
| 선택 8 (LUT GEMV) | T-MAC 선형 bit scaling | [T-MAC github](https://github.com/microsoft/T-MAC/) |
| 선택 8 (LUT Softmax/SiLU) | T-MAN exp 20→1, SiLU 25→1 | [T-MAN arxiv 2511.11248](https://arxiv.org/html/2511.11248v1) |
| **신규 NEO (CPU+GPU batch 재설계)** | NEO H100 14% throughput, 같은 latency | [NEO OpenReview MLSys'25](https://openreview.net/forum?id=umgy9tWBLA) |
| 신규 SparAMX (AMX+sparsity) | linear 1.42×, attn 1.14× | [SparAMX HF paper](https://huggingface.co/papers/2502.12444) |
| 추가 참고 | LUT Tensor Core ISCA'25 | [ACM DL](https://dl.acm.org/doi/10.1145/3695053.3731057) |
| 추가 참고 | SAIL LUT GEMV | [SAIL arxiv 2509.25853](https://arxiv.org/html/2509.25853) |
| 추가 참고 | T-SAR ternary CPU | [T-SAR arxiv 2511.13676](https://arxiv.org/html/2511.13676v1) |

---

## Part H — 명시적 추정/근거 구분 (codex ① 수용)

**모델링 기반 추정 (실측 미검증)**:
- 7B decode 연산량 14 GFLOPs / AMX 30 TFLOPS = **이론 0.47 ms/step** — 이론 계산, 실측 아님
- "compute 10ms + data 3069ms" 분해 — 이론 compute 를 AMX 이론치로 추정한 것. 실제 compute-only 부분은 **Tier -1 에서 PROFILE 으로 측정 예정**

**외부 출처 수치 (이식 시 재검증 필요)**:
- KTransformers 21.3 TFLOPS — MoE 중심, Dense 워크로드에 직접 이식 시 수치 달라질 수 있음
- T-MAC 48 tok/s — Snapdragon X Elite / Surface Laptop 7 기준. Xeon SPR + AMX 에서는 재측정 필요
- NEO 14% throughput — H100 실측이지만 SwiftLLM 기반, vLLM 포팅 시 달라질 수 있음
- SparAMX 1.42× — Xeon SPR 공유 실측. 우리 구현에선 가장 신뢰도 높음

**로컬 실측 확정**:
- H100x8 H1 wall **394s**, H2 wall **2098s**, gpu_only **14s**
- RTX3090 H1 wall **23s**, H7 wall **90s**, gpu_only 1.5B **8s**, 7B **6.5s**
- 3축 `batch scaling ratio, tail, hybrid/gpu ratio` 수치는 **이것들 기반 계산**

---

## Part I — v2 대비 변경점 요약

| 축 | v2 | **v3 (변경)** |
|---|---|---|
| 중심 문제 | per-step 속도 | **batch scaling ratio** |
| Tier 구조 | Tier 0-3 | **Tier -1 (계측) 추가** + 0-3 |
| 성공 지표 축 A | 속도 (ms) | **batch scaling ratio (per-req cost)** |
| 정량 기대치 | 임의 숫자 | **논문 수치 × baseline 공식** |
| compute/data 분해 | 사실처럼 서술 | **"모델링 기반 추정" 명시** |
| Part H (근거 구분) | 없음 | **추정/외부/로컬 3 분리** |
| 선택 7 | sublayer fusion 만 | **Head Folding + batch-aware attention 추가** |
| NEO 출처 | 없음 | **Part G 에 포함** (codex 2-1) |
| Tier 1 에 Head Folding 추가 | 없음 | **codex 4-2 수용** |

---

## Part J — 사용법 (v2 동일)

1. **새 기법 제안 시**: Part B 10개 선택 비교 → 부합 수 ≥7 이면 채택 후보
2. **Tier 재설계 시**: Part D 순서 유지. Tier -1 건너뛰기 금지
3. **방향 점검 시**: Part A-2 batch scaling 프레이밍 + G0-G3 중 현재 위치
4. **성공 지표 불균형 시**: Part E-1 결정 트리
5. **Tier 반복 실패 시**: Part F-1 / E-2 (전략 전환)

**원칙은 고정 아님**. 실측 반증 시 Part F 로 수정. 단 **즉흥적 예외 없이 기록 의무**.

---

## 참고

- v1/v2: 같은 디렉토리
- codex 통합 rewrite: `20260415_085858_codex_hybrid_improvement_integrated_rewrite.md`
- 실측 분석: `eval/basic/H100x8/20260414_213434_claude_*.md`, `eval/basic/RTX3090/20260414_220000_claude_*.md`
- 돌파구 구체 기법: `20260414_233407_claude_hybrid_improvement_from_log_analysis.md`
- ideation morning 4종 (2026-04-14): 돌파구 정의 기반
