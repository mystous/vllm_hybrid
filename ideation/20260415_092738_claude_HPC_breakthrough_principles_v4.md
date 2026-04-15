# vLLM Hybrid 프로젝트 — HPC 돌파구 원칙 v4

**작성**: 2026-04-15 09:27 KST (Claude)
**전신**:
- v1: `20260414_235037_claude_HPC_breakthrough_principles.md`
- v2: `20260415_000227_claude_HPC_breakthrough_principles_v2.md`
- v3: `20260415_091504_claude_HPC_breakthrough_principles_v3.md`

**본 v4 반영** (codex `20260415_085858_codex_*.md` 추가 업데이트 수용):
- **§2-5 AVX/AMX cascade**: ISA 선택을 "둘 중 하나" 가 아닌 **"타일 파이프라인 연속 단계"** 로 재설계. 선택 4 확장 + Tier 에 추가.
- **§4-5 Stop/Go 3 cases**: Tier 종료 판정을 3가지 실패 패턴으로 구조화. Part E-1 강화.

---

## Part A — 우리가 가는 방향 (Destination) [v3 유지]

### A-1. 최종 목표 상태

> H100x8, Qwen2.5-7B, 500 req × 128/128 → **hybrid wall ≤ gpu_only × 1.0**.

현재 × 28 (394s vs 14s).

### A-2. 중심 문제 — True Batch Scaling

**"여러 request 를 CPU 에 동시에 보내도 per-req cost 가 1/N 에 근접하지 않음"** (codex 수용).

실측:
- `cpu_max_num_seqs=1`: 2 req × single cost
- `cpu_max_num_seqs=16`: 32 req × 거의 같은 single cost (scaling ≈ 0)
- `cpu_max_num_seqs=4` (RTX): 4 req × batch=4 cost

→ **batch scaling 확립이 선결 과제**. inflight 증가 = tail 확대 가 되는 구조를 끊어야.

### A-3. 경유 지점 (v3 유지, 3축)

| 경유 | Batch scaling (A) | Tail (B) | Wall ratio (C) |
|---|---|---|---|
| G0. 계측 기준선 | num_seqs 1→4 현재 실측 | 현재 tail | baseline |
| G1. CPU scaling 개선 | 4req cost ≤ 2× single | <50s | ≤3.5× |
| G2. Gate 전환점 | 4req cost ≤ 1.5× | <10s, 고착 없음 | ≤1.5× |
| G3. 최종 돌파 | 4req cost ≤ 1.2× | 제거 | ≤1.0× |

**3축 동시 통과** 원칙. codex §4-5 **Stop/Go 3 cases** 가 Part E-1 에서 판정.

---

## Part B — 우리가 내리는 11가지 선택 (v4: 1개 확장)

### 선택 1. CPU 자체 최적화 1순위, 역할 재정의 병행 (v3 동일)

근거: T-MAC 48 tok/s (baseline 4-5×), DuoDecoding 2.61× (CPU drafter balance 조건 전제).

### 선택 2. Kernel 직접 작성 (v3 동일)

근거: KTransformers 21.3 TFLOPS, PyTorch 3.9×.

### 선택 3. Batch Scaling Ratio 를 주 지표로 (v3 동일)

근거: batch scaling 실패는 sync/memory/parallel efficiency 에서도 옴. num_seqs sweep 이 Ops/Byte 보다 직접적.

### 선택 4. ISA 를 **동적 dispatch + cascade pipeline** 으로 병용 (v4 확장) ⭐

**v3 (단순 dispatch)**: batch=1 → AVX-512, batch≥4 → AMX (binary switch).

**v4 추가 축 — cascade pipeline (codex §2-5)**:

두 ISA 는 경쟁 관계만이 아니라 **연속 단계** 로 쓸 수 있다:
- **AVX-512**: dequant, pack, norm, softmax 같은 벡터/원소별 작업
- **AMX**: tile-friendly matmul
- **조합**: tile k+2 load (DSA/prefetch) → tile k+1 dequant/pack (AVX-512) → tile k matmul (AMX)

**주의 (codex)**:
- AVX `zmm` 와 AMX `tile register` 는 **별도 레지스터 파일** → register-to-register 연쇄 불가
- **타일 버퍼 (L1/L2 상주) 기반 cascade** 로 설계
- 잘못하면 중간 write/read 비용만 늘 수 있음

**핵심 설계 질문 (codex)**:
1. 어떤 batch/shape 에서 어느 쪽이 hot path 인가
2. 어느 시점에 AVX → AMX 로 넘길 것인가
3. staging 이 L1/L2/L3 안에서 닫히는가

→ dataflow 문제 (원칙 3 "reuse" 와 연결).

**구현 우선순위 (v4)**:
- Tier 1: 단순 dispatch (batch 기반) 먼저
- Tier 2: cascade pipeline (타일 버퍼) 추가
- **shape 별 측정 필수**. 작은 decode batch 에서 전환/버퍼링 고정비가 더 클 수 있음.

**근거**:
- KTransformers AMX doc (binary dispatch 검증)
- T-MAN 3-stage pipeline (NPU DMA + Vector + Matrix) — CPU 의 DSA + AVX-512 + AMX 대응, codex §2-5 와 일치
- **실측 미검증**. 강한 가설 (Part H 에 tagging).

### 선택 5. Weight 로드 시 1회 재배치 (v3 동일)

### 선택 6. Memory hierarchy 수동 제어 (v3 동일)

### 선택 7. Sublayer 묶음 + Attention batch-aware 분리 (v3 동일)

### 선택 8. 느린 연산 LUT 대체 (v3 동일)

### 선택 9. Batch 증가는 layout/fusion 후 (v3 동일)

### 선택 10. 계측 우선 실증주의 (v3 동일)

### 선택 11. **Staging 이 cache 안에서 닫히는지 항상 확인** (v4 신설)

**갈림길**: A) cascade/fusion 을 구현하면서 L1/L2/L3 fit 검증 / B) 기능만 구현하고 cache 는 OS/HW 가 관리

**선택: A**.

**근거 (codex §2-5)**: "staging 이 L1/L2/L3 안에서 닫히는가" — AVX/AMX cascade 든, sublayer fusion 이든, LUT 참조든, **중간 버퍼가 cache 에 fit 해야** dataflow 이득이 실현됨. Fit 실패 시 DDR 왕복으로 오히려 느려짐.

**구현 체크리스트**:
- Cascade pipeline 도입 시: tile k/k+1/k+2 의 총 크기 < L2 (코어당 2MB) 검증
- Sublayer fusion 시: 중간 activation 크기 × batch < L1 (48KB) 또는 L2
- LUT 크기: INT4 LUT 32B → register 상주 OK, INT8 LUT 512B → L1 OK

**실패 징후**: PROFILE marker 에서 memory-wait 비중이 50% 초과 → cache overflow 의심.

---

## Part C — 우리가 하지 않는 것 (v3 동일, 비-1~비-7)

---

## Part D — 실행 순서 (v4: Tier 1/2 에 cascade 반영)

```
[Tier -1] 계측 기준선 (~3일)
   ├─ num_seqs=1/2/4/8/16 CPU-only tok/s sweep
   ├─ per-req latency scaling ratio
   ├─ Sublayer 시간 분해 (QKV/Attn/O/Gate+Up/SiLU/Down/Norm)
   ├─ NUMA-local vs non-local diff
   ├─ **cache hit/miss 비중** (memory-wait %) — 선택 11 준비
   └─ 현재 hybrid wall 구성 분해
        │
        ▼ G0 baseline 확정
        │
[Tier 0] 설정 + 단일 호출 (~1주)
   ├─ Huge Pages 1GB
   ├─ IPEX WoQ INT8
   └─ OMP env 검증
        │
        ▼ G1: scaling ≤2×, tail <200s, ratio <15×
        │
[Tier 1] ISA 분기 + 기본 Fusion + 단순 dispatch (2-3주)
   ├─ ISA binary dispatch (선택 4 1차: batch 기반 AVX/AMX 선택)
   ├─ Gate+Up / QKV concat (선택 7)
   ├─ Head Folding
   ├─ Softmax/SiLU LUT (선택 8 first)
   └─ **staging cache-fit 검증** (선택 11 first)
        │
        ▼ G1 통과: 4req cost ≤2×, tail <100s, ratio <8×
        │
[Tier 2] LUT GEMV + Pre-pack + Cascade Pipeline (3-4주)
   ├─ T-MAC INT4 LUT (선택 8)
   ├─ AMX weight pre-pack (선택 5)
   ├─ Batch-aware decode attention (선택 7)
   ├─ **AVX dequant/pack → AMX matmul 타일 파이프라인 (선택 4 cascade)** ⭐ NEW
   ├─ AVX-512 bitmask sparse (선택 8 ext)
   └─ cache-fit 재검증 (선택 11)
        │
        ▼ G2: 4req cost ≤1.5×, tail <10s, ratio <1.5×
        │
[Tier 3] 구조 변경 (1-2달)
   ├─ Spec decode CPU drafter (DuoDecoding)
   ├─ 3-stage DMA+Vec+Mat pipeline (선택 4 cascade 의 완성형, DSA 포함)
   ├─ Core group pipeline (systolic)
   └─ 70B / long-context baseline
        │
        ▼ G3: 4req cost ≤1.2×, tail 제거, ratio ≤1.0×
```

**v4 변경**:
- Tier 1 "단순 dispatch" (선택 4 1차) 로 한정
- Tier 2 에 **AVX/AMX cascade pipeline** 신규 추가 (선택 4 2차)
- Tier -1 과 Tier 1/2 에 **staging cache-fit 검증** (선택 11) 삽입
- Tier 3 의 "3-stage pipeline" 은 cascade 의 DSA 포함 완성형

---

## Part E — 결정 트리 (v4: Stop/Go 3 cases 통합)

### E-1. Tier 종료 시 Stop/Go 판정 (codex §4-5 수용)

3축 체크 결과 3가지 실패 패턴:

#### 경우 1. 속도 축만 좋아지고 tail 축이 안 좋아짐

**가능한 해석**:
- CPU kernel 빨라졌지만 router 가 여전히 잘못 wave 를 채움
- CPU prefill/decode 경계 직렬화 잔존
- batch scaling 생겼지만 wave close 정책이 tail 을 다시 만듦

**조치**:
- routing / gate 파라미터 재검토
- `wave-batch` 유지 여부 재판단
- inflight/finished/drained 로그로 상태 전이 재확인
- **다음 Tier 진행 보류**

#### 경우 2. 속도 축 자체가 거의 안 좋아짐

**가능한 해석**:
- 해당 기법이 hot path 를 못 건드림
- memory traffic 이 줄지 않음
- runtime 재배치 / sync / per-seq loop 지배 잔존

**조치**:
- **다음 Tier 로 가지 않음**
- 기법 보류, 계측 단계 (Tier -1 data) 로 되돌아가 원인 재분석
- 선택 11 (cache-fit) 재검증

#### 경우 3. CPU handled requests 는 늘었는데 wall ratio 가 안 좋아짐

**가능한 해석**:
- 처리한 request 수 증가했지만 CPU req 당 비용 여전히 높음
- CPU 증가분이 pure throughput 이 아니라 tail 로 남음

**조치**:
- **성공으로 보지 않음**
- `cpu_max_num_seqs` 확대 실험 중단
- batch scaling 재검토 후 다시 시도

**최종 판정 기준** (4개 모두 충족):
1. CPU 가 더 많이 처리했다
2. CPU batch tok/s 가 올랐다
3. tail 이 줄었다
4. wall ratio 가 좋아졌다

**한 가지만 좋아지고 다음 Tier 진행 금지** (codex §4-5).

### E-2. 전략 전환 (v3 동일)

선택 1 실패 발동 조건 2개 이상 충족 시 → DuoDecoding (role-level) 방향으로 Tier 3 진입.

### E-3. machine-specific vs policy 판정 (v3 동일)

---

## Part F — 예외 조건 (v3 동일)

---

## Part G — 외부 근거 (v4 보강)

| 원칙 | 근거 | URL |
|---|---|---|
| 선택 1a | T-MAC 4-5× baseline | [T-MAC](https://arxiv.org/pdf/2407.00088) |
| 선택 1b | DuoDecoding 2.61× | [DuoDecoding](https://arxiv.org/abs/2503.00784) |
| 선택 2 | KTransformers 3.9× | [SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) |
| 선택 4 binary | KTransformers AMX doc, batch>4 | [AMX doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md) |
| **선택 4 cascade** | **codex §2-5, T-MAN 3-stage** | [T-MAN](https://arxiv.org/html/2511.11248v1), codex `20260415_085858` |
| 선택 7 | SGLang 12% | [LMSYS](https://lmsys.org/blog/2025-10-22-KTransformers/) |
| 선택 8 LUT | T-MAC | [T-MAC github](https://github.com/microsoft/T-MAC/) |
| 선택 8 LUT transcendental | T-MAN exp/SiLU | 동일 |
| **선택 11 cache-fit** | **codex §2-5 staging**, NPU TCM 설계 | 상동 |
| NEO | H100 14% throughput | [NEO](https://openreview.net/forum?id=umgy9tWBLA) |
| SparAMX | linear 1.42× | [SparAMX](https://huggingface.co/papers/2502.12444) |

---

## Part H — 추정/근거 구분 (v4 강화)

### 모델링 기반 추정 (실측 미검증, Tier -1 에서 측정 예정)
- 7B decode 14 GFLOPs / AMX 30 TFLOPS = 이론 0.47 ms/step
- "compute 10ms + data 3069ms" 분해

### 외부 출처 수치 (이식 시 재검증 필요)
- KTransformers 21.3 TFLOPS — MoE 중심
- T-MAC 48 tok/s — edge CPU (Snapdragon X)
- NEO 14% — SwiftLLM 기반
- SparAMX 1.42× — Xeon SPR (가장 신뢰도 높음)

### 강한 가설 (외부 원리 확인, 우리 환경 미검증) ⭐ v4 신설
- **AVX/AMX cascade pipeline** (선택 4 2차): T-MAN NPU 에서 3-stage 증명됐으나 **x86 CPU 직접 이식 시 staging overhead 가 이득을 상쇄할 가능성**. Tier 2 에서 shape 별 실측 필요.
- **Staging cache-fit** (선택 11): 원리적 필수 조건. 실패 시 다른 기법 모두 무효화 가능.

### 로컬 실측 확정
- H100x8 H1 394s, H2 2098s, gpu_only 14s
- RTX3090 H1 23s, H7 90s

---

## Part I — v3 대비 변경점

| 축 | v3 | v4 |
|---|---|---|
| 선택 개수 | 10 | **11** (선택 11 cache-fit 추가) |
| 선택 4 | binary dispatch 만 | **binary + cascade pipeline** (2 축) |
| Tier 2 | LUT + pre-pack | **+ AVX/AMX cascade** |
| Part E-1 | 간략 결정 트리 | **Stop/Go 3 cases** 구조화 (codex §4-5) |
| Part H 가설 층 | "외부 재검증" 만 | **"강한 가설" 층 신설** (cascade, cache-fit) |
| Tier -1 계측 | 4개 항목 | **+ cache hit/miss 비중** 추가 |

---

## Part J — 사용법 (v3 동일)

1. 새 기법: Part B 11개 비교, ≥7 부합 채택
2. Tier 재설계: Part D 순서 유지, Tier -1 건너뛰기 금지
3. Tier 종료 판정: Part E-1 **3 cases + 4 기준 모두 충족** (1개만 좋으면 fail)
4. 전략 전환: Part E-2
5. 원칙 위반: Part F 기록 조건부

---

## 참고

- v1/v2/v3: 같은 디렉토리
- codex integrated rewrite: `20260415_085858_codex_hybrid_improvement_integrated_rewrite.md` (§2-5, §4-5 가 v4 추가 원천)
- 실측 분석: `eval/basic/H100x8/20260414_213434_claude_*.md`, `eval/basic/RTX3090/20260414_220000_claude_*.md`
- 돌파구 구체 기법: `20260414_233407_claude_hybrid_improvement_from_log_analysis.md`
