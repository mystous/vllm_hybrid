# 15. AMX Weight Pre-pack (독자 제어)

**Tier**: 2
**상태**: 🔶 부분 구현 (IPEX 내부 자동, 독자 제어 없음)
**예상 이득**: 1.1–1.2× (KTransformers 실측 10–20%)

---

## 왜 필요한가

AMX `tileloadd` 는 weight 를 **연속 16 cache line (16 × 64 bytes = 1KB)** 로 읽는다. 모델 weight 가 일반 row-major layout 이면 tileloadd 마다 stride read → cache line fragmentation.

**Pre-pack**: 모델 로드 시 1회 weight 를 **AMX tile layout (16×64 block)** 으로 재배치. runtime tileloadd 가 연속 cache line 로드 → cache miss 감소.

IPEX 의 `ipex.llm.optimize(weights_prepack=True)` 가 이미 암묵 활성 (H100 SPR AVX-512 환경 자동). 그러나 **IPEX 내부 layout 은 우리의 custom kernel (T-MAC LUT, cascade) 과 불일치 가능**. 독자 pre-pack 필요.

---

## 기술적 배경

### AMX Tile Layout

AMX tile 은 16 행 × 최대 64 bytes (16 BF16 elements) 의 2D 영역:
- `tileloadd tmm0, [rax + rbx]` — base + row stride 로 read
- Weight 가 `W[M, N]` 면 `W[0..15, i:i+32]` 를 한 번에 tile load
- 이상적으로: M 방향 16 row 가 메모리에서 연속 → stride = row_size

### Pre-pack 형식

Standard row-major: `W[M, N]` → address = `M*N_size + N`
- M 방향 16 row 가 연속 stride N_size
- 원래도 `tileloadd` 작동하지만 stride = N_size (large) → TLB pressure

T-MAC / AMX pre-pack:
- Block size 16×32 (BF16) 로 재배치
- Block 내부는 contiguous 16 row × 32 col
- Block 간 order 는 M-major or N-major
- `tileloadd` 가 block 단위 → stride 가 작음 (L1/L2 line-aligned)

### IPEX 내부 pre-pack vs 독자 pre-pack

**IPEX**:
- `ipex.llm.optimize(weights_prepack=True)` 가 oneDNN primitive 의 `weights_pd.desc()` 에서 preferred layout 을 조회
- SPR AVX-512 환경에서 자동 AMX-friendly layout
- dev (AVX2 only) 환경에선 `weights_prepack=False` 로 자동 fallback (`intel_cpu_utils.py:891`)
- **Layout 이 우리 custom kernel 과 호환되지 않을 수 있음**

**독자 pre-pack**:
- 우리 kernel 이 요구하는 정확한 layout 강제
- Load-time 1회 변환 + cache to disk (`~/.cache/vllm_hybrid/prepack/<model>_amx.pt`)
- IPEX 와 별도로 자체 관리

### Memory 부담

Pre-pack 은 원본 weight 와 별도 메모리 보유 → **메모리 2×**. 7B BF16 = 14GB + 14GB = 28GB. KV cache 와 경쟁. NUMA 2-node 에서 각 engine 이 local weight copy 가지면 추가 부담.

**해결**:
- Pre-pack 후 원본 weight 삭제 (`torch.Tensor.set_()` 로 underlying storage replace)
- 단 IPEX 가 원본 참조 중이면 삭제 불가 → IPEX 우회 or 순서 조정

---

## 관련 참고 문헌

- **Intel AMX Programming Guide §Tile Layout**: https://cdrdv2-public.intel.com/671368/architecture-instruction-set-extensions-programming-reference.pdf
- **KTransformers AMX doc**: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md — pre-pack 10-20% 실측
- **oneDNN Weights Pre-packing**: https://oneapi-src.github.io/oneDNN/dev_guide_primitive_desc.html
- **IPEX `weights_prepack` option**: https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/api_doc.html#ipex.llm.optimize
- **T-MAC group layout**: https://arxiv.org/pdf/2407.00088 §"Weight Layout"
- **현재 코드**: `vllm/platforms/intel_cpu_utils.py:891` — AVX2 fallback
- **Claude Part 2-10 AMX Pre-pack**: `/vllm_hybrid/ideation/20260415_094130_claude_ninja_gap_comprehensive_plan.md`

---

## 구체 작업

### 설계
- [ ] **Pre-pack layout 스펙 확정**: 우리 kernel (§07 AMX dispatch, §14 cascade, §13 T-MAC) 이 요구하는 정확한 format 문서화
- [ ] **IPEX layout 과의 호환성 확인**: IPEX `weights_prepack=True` 결과를 dump 후 layout 비교
- [ ] **Cache 경로 설계**: `~/.cache/vllm_hybrid/prepack/<model_hash>_<kernel>_<numa>.pt`

### 구현
- [ ] **`csrc/cpu/weight_prepack.cpp`** (신규)
  - `prepack_amx_bf16(W_input, W_output, M, N)` — 16×32 block layout
  - `prepack_tmac_int4(W_input, W_output, M, N, group_size)` — §13 용
  - `prepack_vnni_int8(W_input, W_output, M, N)` — §07 AVX-512 VNNI 용
- [ ] **Load-time hook**: `cpu_model_runner.py` 의 `load_model` 후
  ```python
  for layer in model.layers:
      layer.qkv_proj.weight = torch.ops._C_cpu_ops.prepack_amx_bf16(layer.qkv_proj.weight)
      # original weight storage replaced
  ```
- [ ] **Cache to disk**: pre-pack 결과를 파일로 저장, 재부팅 시 skip
- [ ] **NUMA-local replication**: 2-NUMA 에서 engine 0 은 NUMA 0 에, engine 1 은 NUMA 1 에 별도 copy
- [ ] **torch op 등록**

### 검증
- [ ] **정확도**: pre-pack 후 연산 결과가 원본과 동일 (loss-less repack)
- [ ] **성능**: AMX GEMM 시간 비교 (pre-pack on/off)
- [ ] **Memory**: `/proc/<pid>/status` 로 RSS 변화 확인 (2× 증가 예상)
- [ ] **Cache hit**: 두 번째 실행에서 pre-pack 재생성 skip 되는지

---

## 성공 조건

1. ✅ Pre-pack 후 AMX GEMM 10–20% 가속
2. ✅ Cache line miss `perf stat -e l2_rqsts.miss` 감소
3. ✅ Load-time overhead <10s (cache 재사용 시 <1s)
4. ✅ 원본 weight memory 회수로 RSS 증가 <1.5× (2× 아님)
5. ✅ §13, §14 kernel 이 pre-pack weight 사용 확인

---

## 의존성

- **선행**: §06 hot path wiring (pre-pack weight 를 사용할 kernel path), §07 AMX kernel (소비자)
- **병행**: §13 T-MAC pre-pack (다른 layout)
- **후속**: §14 Cascade 의 matmul stage 가 pre-packed weight 사용

---

## 리스크

- **메모리 2× 부담**: 원본 weight 삭제가 IPEX 와 충돌 시 RSS 2× 유지 → KV cache 공간 축소
- **Layout 버전 drift**: Intel CPU 세대 (SPR → GNR) 마다 optimal layout 달라질 수 있음 → cache invalidation 필요
- **IPEX weights_prepack 과의 이중 변환**: IPEX 가 먼저 변환 → 우리가 또 변환 → 성능 저하. 순서/비활성 제어 필요
- **Load-time 증가**: 7B 기준 pre-pack 10-30s 예상. cache 로 해소 가능하나 첫 실행 UX 부담

---

## 스택 호환성

- §13 T-MAC LUT GEMV: T-MAC group layout 독자 pre-pack (같은 infra, 다른 layout)
- §14 AVX/AMX Cascade: matmul stage 의 tile 로 pre-packed weight
- §07 ISA Dispatch: path 별 pre-pack (AMX 용, VNNI 용 각각)
- §03 Huge Pages: pre-pack 결과가 1GB page 에 상주 시 TLB 이득 최대화

---

## 관련 코드 위치

- `csrc/cpu/weight_prepack.cpp` — (신규)
- `csrc/cpu/torch_bindings_hybrid.cpp` — 등록
- `vllm/v1/worker/cpu_model_runner.py` — load_model 후 hook
- `vllm/platforms/intel_cpu_utils.py:891` — 기존 AVX2 fallback
- `~/.cache/vllm_hybrid/prepack/` — cache 저장 위치
