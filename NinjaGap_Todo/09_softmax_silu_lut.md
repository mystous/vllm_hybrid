# 09. Softmax + SiLU LUT 대체

**Tier**: 1
**상태**: ⭕ 미구현
**예상 이득**: Softmax 2.2×, SiLU 1.2× (TARDIS vLLM 1.6× 보고)
**조건부**: scalar transcendental 이 프로파일 **top bottleneck 일 때만**

---

## 왜 필요한가

Transformer 에서 `exp()` 는 **scalar transcendental function** — hardware 가속 없음. x86 libm 의 `expf` 는 ~20 cycles, `vpermi2ps` 같은 vectorized approx 는 ~10 cycles, **`vpshufb` 기반 LUT 는 1 cycle**.

**경로**:
- Softmax = `softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))` — `exp` 가 hot
- SiLU = `x * sigmoid(x) = x / (1 + exp(-x))` — `exp` 가 hot

batch=16, seq_len=1 (decode), head=32, context=128 기준, softmax 대상 값 수 = 16 × 32 × 128 = 65k per step. `exp` 20 cycles × 65k = 1.3M cycles per softmax per step. 동일 계산을 layer 수 (80) × 2 (attn + swiglu) = 160 회 반복 → 총 **2억 cycles**, 3GHz 에서 **70ms**. 이건 step time 의 상당 비중.

단 **프로파일에서 top bottleneck 인지 G0 에서 확인 후 진입** (Codex 조건).

---

## 기술적 배경

### LUT 의 원리

`exp(x)` 를 `x ∈ [-8, 0]` 구간에서 근사:
1. `x` 를 16/32/64 bin 으로 quantize (예: `y = floor((x + 8) * 8)` → `y ∈ [0, 64]`)
2. `y` 를 `vpshufb` 의 index 로 사용, precomputed `exp` table 에서 한 번에 lookup
3. `vpshufb` 는 single-cycle (Port 5 on SKL+)

### `vpshufb` 의 제약

- 입력 index 는 **low nibble 만** 사용 (4 bits, 16 entries per lane)
- 512-bit zmm 은 4×128-bit lane → 16 entries × 4 replicated = 64 independent lookups
- 더 정밀한 lookup (128/256 entries) 은 `vpermi2b` 사용 가능 (but higher latency)

### Softmax 2-pass → 1-pass optimization

표준:
```
max_val = max(x)     // pass 1
sum = Σ exp(x - max_val)  // pass 2
out = exp(x - max_val) / sum  // pass 3
```

**Flash Attention 스타일 online softmax**:
- Running max, running sum 을 tile 마다 갱신
- CPU 에도 동일 원리 적용 가능, memory traffic 감소

### SiLU LUT vs 선형 근사

SiLU 는 부드러운 함수. 두 방식:
- **LUT (range [-8, 8] 64 bins)**: 정확도 0.5% 손실, 1-cycle
- **Piece-wise linear (hot range [-2, 2] 선형, 외곽 0/x)**: 정확도 2% 손실, 0.5-cycle

### TARDIS vLLM 구현

TARDIS (arXiv 2501.10054) 는 vLLM 에 LUT 기반 transcendental 을 도입하여 **1.6× 전체 decode throughput** 개선 보고.

### T-MAC 의 LUT 인프라 공유

§13 T-MAC LUT GEMV 는 INT4 weight 에 대한 LUT 기반 GEMV. 둘 다 `vpshufb` / `vpermi2b` 를 사용하므로 **동일 infra 재사용** 가능.

---

## 관련 참고 문헌

- **T-MAN (arXiv 2511.11248)**: https://arxiv.org/html/2511.11248v1 — unified table lookup for LLM
- **TARDIS (arXiv 2501.10054)**: — vLLM 1.6× with LUT transcendental (구체 수치 citation 필요)
- **Intel Intrinsics Guide `vpshufb`, `vpermi2b`**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- **Flash Attention paper (Dao et al. 2022)**: online softmax
- **Milakov & Gimelshein (2018) "Online normalizer calculation for softmax"**: one-pass softmax
- **`exp` approximation (Schraudolph 1999)**: fast integer-based exp using IEEE754 bit representation
- **SIMD `log`/`exp` implementations (Agner Fog's tables)**: https://www.agner.org/optimize/
- **HuggingFace `softmax` CUDA 구현**: 참조 algorithm (CPU 에 이식)

---

## 구체 작업

### Pre-조건 확인
- [ ] **§01 G0 계측에서 softmax/silu 가 top bottleneck 인지 확인**. 아니면 본 항목 후순위로 밀림

### LUT 인프라 구축
- [ ] **`csrc/cpu/lut_ops.cpp`** (신규) — `vpshufb` 기반 LUT kernel 공통 infra
  - `lut_exp_bf16(input, output, length)` — BF16 → LUT 의 16-entry exp 테이블
  - `lut_sigmoid_bf16(...)`
  - `lut_silu_bf16(...)`
- [ ] **LUT table 초기화** (load-time): 32-byte table register resident
  - `exp_table[16] = {exp(-8), exp(-7.5), ..., exp(-0.5)}`
  - normalize 후 BF16 변환
- [ ] **정확도 검증 harness**: `input ∈ [-10, 10]` sweep 에서 reference `expf` 대비 max rel error

### Softmax 적용
- [ ] **`csrc/cpu/lut_softmax.cpp`** — `online softmax + LUT exp`
- [ ] Attention path (`cpu_attn.py`) 에서 기존 softmax 를 custom op 로 치환
- [ ] batch-aware: softmax 의 input shape `(M, heads, seq_len)` 에서 `M × heads` 방향 병렬

### SiLU 적용
- [ ] **`csrc/cpu/lut_silu.cpp`** — SiLU LUT 또는 piece-wise linear
- [ ] §08 Gate+Up fusion 내부에서 SiLU 대체
- [ ] 단독 SiLU 경로도 지원 (fusion 없을 때)

### 통합
- [ ] **torch ops 등록**: `torch.ops._C_cpu_ops.lut_softmax_bf16`, `lut_silu_bf16`
- [ ] **Model patch**: `softmax` / `silu` 호출을 custom op 로 치환
- [ ] **정확도 비교**: PPL (WikiText-2), MMLU 변화 <2%

---

## 성공 조건

1. ✅ LUT 기반 exp/silu/sigmoid 의 max rel error <1% (BF16 tolerance 내)
2. ✅ Softmax 단독 측정에서 2× 이상 가속
3. ✅ SiLU 단독 측정에서 1.2× 이상 가속
4. ✅ PPL 열화 <2% (WikiText-2 기준)
5. ✅ decode step time 10–30% 감소 (Softmax/SiLU 가 top bottleneck 일 때)

---

## 의존성

- **선행**: §01 G0 계측 (softmax/silu 가 top 인지 확인), §06 hot path wiring
- **병행**: §08 Kernel Fusion (SiLU 가 Gate+Up fused 내부로 들어감)
- **후속**: §13 T-MAC LUT GEMV (동일 `vpshufb` infra 재사용)

---

## 리스크

- **정확도 열화가 특정 모델에서 큼**: long-context 또는 low-temperature 생성에서 softmax 미세 차이가 누적되어 token-level divergence. 측정 필수
- **`vpshufb` 의 4-bit index 한계**: 더 정밀한 근사 (8-bit index, 256 entries) 필요 시 `vpermi2b` 사용 — latency 증가 (5 cycles)
- **AMX path 와 통합 어려움**: AMX 는 tile register 에만 쓰기, `vpshufb` 는 zmm 에 쓰기. Softmax 는 fused attention 의 중간 산출 → AMX GEMM 결과를 zmm 으로 옮겨야 함 → 추가 copy
- **scalar-math hot 이 이미 IPEX 에서 해결됨**: IPEX 의 oneDNN softmax 가 SIMD exp (vectorized, 10 cycles) 사용. 우리 LUT 대체 이득 축소될 수 있음

---

## 스택 호환성

- §08 Gate+Up fusion 내부에서 LUT SiLU 대체
- §13 T-MAC LUT GEMV 와 동일 infra (`lut_ops.cpp` 공유 helper)
- §11 Batch-aware decode attention 내부 softmax 를 LUT 으로 대체

---

## 관련 코드 위치

- `csrc/cpu/lut_ops.cpp` — (신규) 공통 LUT infra
- `csrc/cpu/lut_softmax.cpp` — (신규)
- `csrc/cpu/lut_silu.cpp` — (신규)
- `csrc/cpu/torch_bindings_hybrid.cpp` — 등록
- `vllm/v1/attention/backends/cpu_attn.py` — softmax 치환
- `vllm/model_executor/layers/activation.py` — SiLU 치환
