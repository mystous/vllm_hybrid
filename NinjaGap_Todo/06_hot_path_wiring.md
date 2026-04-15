# 06. Hot Path 연결 증명 (G1 진입 필수)

**Tier**: 1
**상태**: ⭕ 미구현 (build artifact 는 있지만 call-site 없음)
**중요도**: **G0 계측 다음 본선 진입의 첫 관문**

---

## 왜 필요한가

현재 vLLM hybrid 의 CPU 경로는 IPEX `ipex.llm.optimize` 에 의존. IPEX 는 내부적으로 oneDNN primitive 를 호출하며, 어떤 primitive 가 선택되는지는 IPEX/oneDNN dispatcher 가 결정. 우리가 빌드한 `_C_cpu_ops.abi3.so` 의 VNNI INT8 GEMM (`csrc/cpu/gemm_vnni.cpp`) 은 **build 는 되지만 실제 Python 쪽에서 호출하는 경로 (call-site) 가 없음**.

**Codex 원칙**:
> "VNNI 가 있다" 도 신규 gain 아님 — hot path 에 실제 연결되어 batch scaling 을 만든 경우만 gain.

이 섹션은 **Tier 2/3 kernel 실험의 전제**. hot path 를 실제로 바꾸는 infra 없이는 §07~§17 의 kernel 개선도 "IPEX 가 계속 사용됨" 으로 귀결.

---

## 기술적 배경

### 현재 CPU linear path

```
Python model forward
  → vllm.model_executor.layers.linear.Linear.forward
  → (ipex.llm.optimize 에 의해 치환) IPEX optimized Linear
  → IPEX MLP/Attn fused primitive
  → oneDNN primitive (IPEX dispatcher 선택)
  → AMX BF16 matmul (SPR 자동) 또는 AVX-512 VNNI INT8 (quant 경로)
```

우리가 구현한 `int8_gemm_vnni` (`csrc/cpu/gemm_vnni.cpp`) 는 이 chain 어디에도 **callable 연결 안 됨**.

### 연결 지점 후보

**Option A — IPEX bypass**:
- `vllm.model_executor.layers.linear.Linear.forward` 를 IPEX 치환 **전에** 우리 kernel 로 override
- 장점: 명시적 제어
- 단점: IPEX 의 다른 최적화 (pre-pack, kernel fusion) 잃음

**Option B — IPEX hook**:
- IPEX 의 `_IPEXLinearFusionCPU` 혹은 oneDNN custom primitive 로 등록
- 장점: IPEX 최적화 유지
- 단점: IPEX 내부 구조 의존, 버전별 깨짐

**Option C — Torch custom op**:
- `torch.library.define("vllm_cpu::gemm_vnni", ...)` 로 등록
- model load 후 `post_process_model_hook` 으로 특정 Linear 를 torch.ops 호출로 치환
- 장점: Python 에서 명시 dispatch, IPEX 와 공존
- 단점: hook 설계 필요

**권장**: Option C (torch custom op + post-load hook). `csrc/cpu/torch_bindings_hybrid.cpp` 가 이미 일부 op (VNNI) 를 torch ops 로 등록했을 가능성 — 확인 필요.

### Pre-pack cache

AMX tile layout (16×64 bytes) 또는 VNNI block layout (16×4 bytes) 로 weight 를 load-time 에 1회 재배치. runtime repack 을 방지:
- IPEX 가 이미 `weights_prepack=True` 암묵 활성
- **그러나 우리 kernel (VNNI INT8) 은 IPEX 의 pre-pack format 과 다를 수 있음** — 별도 pre-pack 필요

### Shape 별 dispatch log

decode 시 `(M=batch, N=hidden_dim, K=intermediate)` shape 이 매 step 거의 동일. prefill 은 M=prompt_len. 우리가 VNNI INT8 에 대해 어느 shape 에서 언제 호출됐는지 로그:
```
[HYBRID-KERNEL] vnni_int8 M=16 N=4096 K=11008 time=2.3ms
[HYBRID-KERNEL] ipex_amx_bf16 M=1 N=4096 K=4096 time=0.8ms
[HYBRID-KERNEL] sdpa_fallback M=1 N=4096 K=128 time=1.1ms
```

`ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI` 같은 설정이 있다고 해서 실제 VNNI 가 호출된다는 보장은 없음. dispatcher 결정은 shape / alignment / pre-pack format 에 따름.

---

## 관련 참고 문헌

- **Codex playbook §6 Tier 1 Mainline Hot Path 연결**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md`
- **KTransformers AMX**: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md — AMX-specialized kernel 을 실제 hot path 에 연결하는 방식
- **KTransformers SOSP'25 paper**: https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf — CPU/GPU hybrid 에서 synchronization overhead 감소 방식
- **SGLang + KTransformers blog**: https://lmsys.org/blog/2025-10-22-KTransformers/ — ISA switching + coordination overhead 감소
- **oneDNN Primitive Dispatch**: https://oneapi-src.github.io/oneDNN/
- **Torch custom op**: https://pytorch.org/tutorials/advanced/torch_library.html
- **Tech_done v1 Q1**: `_C_cpu_ops` 빌드 검증 기록

---

## 구체 작업

- [ ] **torch_bindings_hybrid.cpp 검토**: 현재 어떤 op 가 torch ops 로 등록되어 있는지 확인 (`torch.ops._C_cpu_ops.xxx` 로 callable 한 것 나열)
- [ ] **VNNI INT8 GEMM op 등록 완비**: `torch.ops._C_cpu_ops.int8_gemm_vnni(..)` callable 확인, schema 문서화
- [ ] **Linear module patch 설계**: `vllm_patch_linear_to_vnni(model)` — 특정 layer (`o_proj`, `down_proj` 등 memory-bound 선별) 를 torch custom op 호출로 치환
- [ ] **Load-time pre-pack cache**: `cpu_model_runner.py` 의 `load_model` 후 hook 에서 weight 를 VNNI layout 으로 1회 변환 + serialized tensor 로 cache
  - `~/.cache/vllm_hybrid/prepack/<model_hash>_vnni.pt`
- [ ] **Runtime repack 계측**: step 마다 weight layout 전환이 일어나는지 marker `[HYBRID-KERNEL-REPACK]` 기록 (0 이어야 함)
- [ ] **Shape 별 dispatch log marker**: `[HYBRID-KERNEL] shape=... kernel=... time=...`
  - env variable `VLLM_HYBRID_KERNEL_TRACE=1` 일 때만 출력 (overhead 방지)
- [ ] **`ONEDNN_MAX_CPU_ISA` 실제 dispatch 확인**: oneDNN verbose `ONEDNN_VERBOSE=1` 와 교차 검증, primitive 이름 (`brg_matmul:avx512_core_vnni` vs `brg_matmul:avx10_1_512_amx`) 기록
- [ ] **성공 판정 실험**: `num_seqs=1/2/4/8` 각각에서 VNNI kernel hit rate + per-req cost

---

## 성공 조건

1. ✅ `[HYBRID-KERNEL]` 로그에서 VNNI INT8 kernel 이 실제 호출됨을 확인
2. ✅ `ONEDNN_VERBOSE` 가 우리 custom op 의 호출을 보여줌 (또는 직접 kernel 이면 IPEX 우회 확인)
3. ✅ runtime repack count = 0 (pre-pack cache 작동)
4. ✅ `num_seqs=4` 에서 per-request cost 감소 (G0 baseline 대비)
5. ✅ **단일 req 만 빨라지고 scaling 없으면 다음 Tier 금지** (Stop/Go Case 3)

---

## 의존성

- **선행**: §01 G0 계측 (baseline + sublayer breakdown 확보), §05 OMP env
- **병행**: §03 Huge Pages (pre-pack cache 효율에 유리)
- **후속**: §07 ISA dispatch 가 본 infra 를 사용해 batch shape 별 분기. §15 AMX pre-pack 은 본 pre-pack cache 확장
- **ultimately**: §10, §11, §13, §14 모두 hot path 연결 이후에만 의미

---

## 리스크

- **IPEX 가 내부적으로 이미 VNNI 쓰고 있음**: 우리 kernel 이 더 빠르다고 보장 없음. KTransformers 수치 (1.5–2×) 는 IPEX 부재 조건
- **pre-pack format 불일치**: IPEX weights_prepack 과 우리 VNNI pre-pack 이 다르면 모델 weight 를 두 벌 보관 (메모리 2×)
- **Torch custom op 의 fx graph / compile 친화성**: vLLM 의 model compile 경로 (`torch.compile`) 와 충돌 가능. graph break 발생 시 성능 영향 측정
- **dispatch 분기 overhead**: 매 layer 마다 shape check → Python overhead. `torch.ops` 는 빠르지만 layer 수 80+ 면 누적 가능. 측정

---

## 스택 호환성

- §07 ISA Binary Dispatch: 본 infra 위에 batch 조건 기반 분기 추가
- §13 T-MAC LUT GEMV: 본 infra 를 모델로 삼아 INT4 LUT op 등록
- §14 AVX/AMX Cascade: 본 infra + tile buffer 관리
- §15 AMX Pre-pack: 본 pre-pack cache 의 AMX tile layout 확장

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_VNNI_HOT_PATH` | `0` (기본) / `1` | VNNI kernel 을 hot path 에 실제 연결 |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `csrc/cpu/gemm_vnni.cpp` — VNNI INT8 GEMM 구현
- `csrc/cpu/torch_bindings_hybrid.cpp` — torch ops 등록 (확인/보강)
- `vllm/v1/worker/cpu_model_runner.py` — load_model 후 hook 위치
- `vllm/_custom_ops.py` — `HAS_CPU_OPS` flag + callable 노출
- `vllm/model_executor/layers/linear.py` — Linear module patch 대상
- `vllm/v1/attention/backends/cpu_attn.py` — `_decode_path_counts` (기존 counter)
