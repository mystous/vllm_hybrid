# 08. Kernel Fusion (QKV / Gate+Up / Residual+Norm)

**Tier**: 1
**상태**: 🔶 부분 구현 (GPU 경로 only, CPU 전용 fused kernel 없음)
**예상 이득**: 1.5–2× (SGLang SiLU+up 12% × 4 sublayer 누적)

---

## 왜 필요한가

**실패 3** (Claude 3겹 진단): sublayer 8개 체인 = 독립 kernel = **DDR 왕복 8회 × batch**.

Transformer layer 의 sublayer 당 weight read + intermediate write/read 가 각각 독립적. 예:
- QKV projection: `x` read, `W_q`, `W_k`, `W_v` read, `q`, `k`, `v` write
- Gate projection: `x` read (Residual after QKV 이미 write), `W_gate` read, `gate` write
- Up projection: `x` read (중복!), `W_up` read, `up` write
- SiLU: `gate` read, `gate_silu` write
- Down: `up * gate_silu` read, `W_down` read, `out` write
- Residual add: `out + residual` read, write

동일 `x` 가 QKV + Gate + Up + Residual 에서 **4회 read** 되고, 중간 activation 이 DDR 에 write 후 즉시 read (L3/L2 에서 hit 하면 됨 — 크기 따라). **Fusion 은 이 중복 read 와 중간 write 를 제거**.

---

## 기술적 배경

### Fusion 대상 3종

**1. QKV concat fusion**
- BF16 `x @ [W_q || W_k || W_v]` = `[q || k || v]` 한 번에 계산
- 공통 `x` 를 한 번만 read
- GEMM 의 N dimension 을 3× 로 확장 (AMX tile 입장에선 efficient)
- vLLM GPU 경로 `QKVParallelLinear` 이미 구현. **CPU 경로는 IPEX 가 내부적으로 하지만 custom kernel 없음**

**2. Gate + Up interleave fusion (SwiGLU)**
- `x @ W_gate → gate` / `x @ W_up → up` 를 동일 x 로 2회 read
- Interleaved execution: tile k 마다 gate_partial + up_partial 동시 계산, `x[k]` 1회 read
- SGLang 실측 **12% 개선** (CPU 백엔드 blog). 4 sublayer 누적하면 1.5×

**3. Residual + RMSNorm fusion**
- `x_new = RMSNorm(x_old + attn_out)` — 두 pass 가 아니라 one-pass
- Intermediate `residual_sum` 을 메모리에 안 쓰고 register 에서 norm 계산으로 이동
- IPEX 내부에 이미 있음. custom 구현 시 fused_add_rmsnorm → 그 위에 quant 경로 붙임

### Fusion 의 cache-behavior

- `x` 는 `shape=(M, hidden_dim)` — batch=16, hidden=4096 → 128KB (BF16). L2 (L2 per core 2MB) 안에 편안히 들어감
- Weight `W` 는 `shape=(hidden_dim, out_dim)` — 수십 MB. DDR read 필수
- **Fusion 의 이득은 W read 감소가 아니라 `x` read 중복 제거 + intermediate write/read 생략**
- batch 작을수록 (M 작을수록) `x` 재사용 이득 작음 — **batch↑ 에서 fusion 효과 확대** (batch scaling 에 기여)

### 현재 상태 확인

- `vllm/model_executor/layers/linear.py` 의 `QKVParallelLinear`, `MergedColumnParallelLinear (gate_up_proj)` — vLLM GPU 경로용 merged linear
- CPU 경로에서 이 merged module 이 IPEX 에 의해 어떻게 치환되는지는 IPEX 내부 구현에 의존. IPEX 는 내부적으로 Llama 계열의 QKV/GateUp 를 fuse 하지만, **우리 VNNI INT8 경로 (§06 에서 연결 예정) 와는 별개**

---

## 관련 참고 문헌

- **SGLang CPU 백엔드 blog "AMX optimized kernels"**: https://lmsys.org/blog/2025-10-22-KTransformers/ — SiLU+up 12% 실측
- **Llama 2 paper §"Architectural details"**: SwiGLU activation 의 Gate + Up 구조
- **Touvron et al. (2023) Llama 2** — SwiGLU rationale
- **T-MAC Gate+Up interleave 설계**: https://arxiv.org/pdf/2407.00088 §"LUT kernel fusion"
- **Shazeer (2020) "GLU Variants Improve Transformer"**: https://arxiv.org/abs/2002.05202 — SwiGLU 원전
- **RMSNorm 원전**: Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
- **Flash Attention paper (Dao et al. 2022)**: Fusion 의 성능 이득 원리 (동일 원리 CPU 에도 적용)
- **TVM / AutoTVM fused kernel**: fused kernel 자동 생성 참조
- **현재 vLLM 코드**: `vllm/model_executor/layers/linear.py` 의 `QKVParallelLinear`, `MergedColumnParallelLinear`

---

## 구체 작업

### QKV Fusion
- [ ] **현재 IPEX 가 QKV fuse 하는지 확인**: VTune profile 에서 `qkv_proj` 가 3회 독립 matmul 인지 1회 fused matmul 인지
- [ ] **이미 fused 라면 스킵**. 아니면 `csrc/cpu/fused_qkv.cpp` 구현
- [ ] AMX path: N 을 3× 확장한 GEMM 한 번
- [ ] AVX-512 VNNI path: 3-way concat GEMM

### Gate + Up Fusion
- [ ] **`csrc/cpu/fused_gate_up_silu_down.cpp`** 구현
- [ ] k-direction tiling: 각 tile 마다 `x[k]` 1회 load, `W_gate[k,:]` 와 `W_up[k,:]` 에 대해 gate_partial += x*Wg, up_partial += x*Wu
- [ ] 마지막에 SiLU(gate) * up 를 register 에서 직접 계산 (중간 tensor write 생략)
- [ ] 이어서 Down projection 까지 연결 (extended fusion)

### Residual + RMSNorm Fusion
- [ ] **`csrc/cpu/fused_add_rmsnorm.cpp`** 구현
- [ ] IPEX 내부 fused kernel 사용 가능한지 먼저 확인 (대부분 이미 있음)
- [ ] 우리 INT8/AMX path 로 연결 시 fused 유지

### 공통
- [ ] **torch ops 등록**: `torch.ops._C_cpu_ops.fused_qkv_bf16`, `fused_gate_up_silu_bf16`, `fused_add_rmsnorm_bf16`
- [ ] **Linear module patch**: QKVParallelLinear / MergedColumnParallelLinear 를 CPU 경로에서 fused op 로 치환
- [ ] **정확도 테스트**: fused vs 분리 결과가 tolerance 내
- [ ] **batch scaling 측정**: fusion 전후 `batch_scaling_ratio` 비교

---

## 성공 조건

1. ✅ 각 fusion 이 정확도 tolerance 통과 (BF16 relative error <1e-2)
2. ✅ `batch=4` 에서 fusion 전후 per-step time 10–20% 감소
3. ✅ `batch=16` 에서 fusion 이득 더 커짐 (x read 중복 제거의 영향 확대)
4. ✅ **`batch_scaling_ratio` 개선**: fusion 전 5.3× → fusion 후 3× 이하 (목표)
5. ✅ G0 baseline 대비 decode step time 1.5–2× 감소

---

## 의존성

- **선행**: §06 hot path wiring (fused op 를 호출할 infra), §07 ISA dispatch (AMX/AVX-512 각각에 대한 fused kernel)
- **병행**: §09 Softmax/SiLU LUT (SiLU 를 LUT 로 대체하면 fusion 내부에서 swap)
- **후속**: §13 T-MAC LUT GEMV (INT4 LUT 경로에서도 동일 fusion 적용), §14 cascade (fusion + pipeline 조합)

---

## 리스크

- **IPEX 기존 fused kernel 과 중복/충돌**: IPEX 이 이미 QKV/Gate+Up fuse 하는 경우 우리 구현이 의미 없음. 측정으로 IPEX fusion 유효성 먼저 확인
- **fused kernel 의 유지보수 비용**: model architecture 변경 시 (예: GQA → MLA, SwiGLU → SiLU-only) 재작성
- **AMX tile layout 과 fusion 의 상호작용**: QKV 3× N 확장이 AMX tile (16-col) 경계 와 맞지 않으면 padding/align overhead
- **정확도 열화**: fused 연산 순서 변경이 BF16 accumulation 에서 미세 차이 유발 — tolerance 설정

---

## 스택 호환성

- §07 ISA Binary Dispatch 의 각 dispatch 안에서 fused kernel 호출
- §13 T-MAC LUT GEMV: INT4 LUT path 에서도 동일 fusion 로직 적용
- §14 AVX/AMX Cascade: fused kernel 이 각 stage 의 primitive 가 됨
- §09 Softmax/SiLU LUT: fused_gate_up_silu 내부에서 SiLU 를 LUT 로 대체

---

## 관련 코드 위치

- `csrc/cpu/fused_qkv.cpp` — (신규)
- `csrc/cpu/fused_gate_up_silu_down.cpp` — (신규)
- `csrc/cpu/fused_add_rmsnorm.cpp` — (신규, IPEX 없으면)
- `csrc/cpu/torch_bindings_hybrid.cpp` — 등록
- `vllm/model_executor/layers/linear.py` — `QKVParallelLinear`, `MergedColumnParallelLinear`
- `vllm/v1/worker/cpu_model_runner.py` — load_model 후 patch
