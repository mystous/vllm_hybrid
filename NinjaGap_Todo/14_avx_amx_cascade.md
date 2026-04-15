# 14. AVX/AMX Cascade Pipeline

**Tier**: 2
**상태**: ⭕ 미구현
**예상 이득**: 1.5–3× (T-MAN NPU 3.1× decode, CPU 이식 보수적 1.5–2×)
**근거 등급**: D → 강한 가설 (x86 재검증 필수)

---

## 왜 필요한가

AVX-512 와 AMX 는 서로 다른 register file 을 사용하며 직접 연결되지 않는다. 둘을 binary dispatch (§07) 로만 쓰면 각자 독립. 하지만 **decode 의 실제 연산 sequence 는 load → dequant/pack → matmul** 로 3-stage 로 분해 가능. 이를 겹쳐 실행하면 pipeline 이득.

T-MAN (NPU) 에서 decode **3.1×** 달성. CPU 이식은 강한 가설 — cache-fit 이 성패를 가른다.

---

## 기술적 배경

### 3-Stage Pipeline 설계

```
Tile k   : AMX matmul    (tmm register)
Tile k+1 : AVX-512 dequant/pack (zmm register)
Tile k+2 : prefetch/DSA load   (L1/L2 cache)
```

각 tile 은 독립 data. k 가 matmul 중일 때 k+1 은 이미 dequant/pack 완료, k+2 는 load 중. OMP 가 아닌 **single-thread 내 instruction-level pipelining + software prefetch**.

### AVX `zmm` ↔ AMX `tile` 데이터 이동

- AVX → AMX: 중간 buffer (메모리) 에 store 후 `tileloadd` 로 tile 에 load
- AMX → AVX: `tilestored` 후 `vmovdqa` 로 zmm 에 load
- **직접 연결 없음** → 중간 buffer 가 L2 에 상주해야 함

### Cache-fit 전제

Tile size (16×64 bytes = 1KB) × 8 tiles × 3 stages = 24KB. L1d 32KB 에 들어가기엔 빡빡 (input/weight/output tile 모두). **L2 (2MB per core)** 에서 편안.

Stage 간 buffer 가 DDR 로 밀리면 **pipeline 이 아니라 DDR 왕복 증가** — 이득 역전. 따라서:
- Tile schedule 을 L2-fit 되도록 설계
- 중간 buffer 를 lazy-write (write-combining) 로 DDR 방지

### Intel DSA (Data Streaming Accelerator)

SPR 에는 DSA (Accelerator Interfaces Architecture AIA) 가 있음. 별도 가속기로 DMA 전용:
- `enqcmd` 인스트럭션으로 비동기 memcpy 요청
- CPU core 가 matmul 하는 동안 DSA 가 다음 tile DMA
- Stage 3 (prefetch/DSA) 의 구현 후보

하지만 DSA 사용은 kernel 영역 (IOMMU 설정, user-mode submission). 현실적 어려움. **대안**: 소프트웨어 prefetch (`PREFETCHT0/T1/T2`) 가 단순하면서 효과 있음.

### T-MAN 의 NPU 3-stage 와 CPU 차이

T-MAN 은 NPU 에서:
- DMA engine (load)
- Vector unit (dequant/pack)
- Matrix unit (matmul)
이들이 **물리적으로 분리된 하드웨어**.

x86 CPU 에서는:
- Load: cache hierarchy (prefetcher)
- Dequant/Pack: AVX-512 (port 0, 1, 5)
- Matmul: AMX (port 5 공유)

**port 5 공유** 가 병목 가능성. 실험으로 측정 필요.

### Shape-aware dispatch

```
batch=1  → AVX-only path (AMX 이득 없음)
batch=4-8 → cascade 진입, 소규모 tile
batch=16+ → full AMX path, cascade 의미 축소
```

§07 binary dispatch 의 확장.

---

## 관련 참고 문헌

- **T-MAN (arXiv 2511.11248)**: https://arxiv.org/html/2511.11248v1 — 3-stage pipeline 원리
- **T-MAC GitHub t-man**: https://github.com/microsoft/T-MAC/tree/main/t-man
- **Intel AMX + AVX-512 관계**: https://cdrdv2-public.intel.com/671368/architecture-instruction-set-extensions-programming-reference.pdf §Chapter 3
- **Intel DSA documentation**: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/data-streaming-accelerator.html
- **Software prefetching tutorial (Intel)**: https://www.intel.com/content/www/us/en/developer/articles/technical/memory-performance-in-a-nutshell.html
- **Codex playbook Tier 3 AVX/AMX cascade**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md` — cache-fit 실패 시 DDR 왕복 증가 경고
- **KTransformers AMX**: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md
- **Agner Fog "The Microarchitecture of Intel, AMD, and VIA CPUs"**: https://www.agner.org/optimize/microarchitecture.pdf — port 5 공유 분석

---

## 구체 작업

### 사전 검증
- [ ] **§01 G0 계측 결과로 AMX path 가 DDR-bound 인지 확인**: memory BW utilization
- [ ] **§07 binary dispatch 성숙** 후에 진입

### 설계
- [ ] **Tile schedule 설계**:
  - Tile size: 16×64 bytes (AMX tile) × K ratio
  - Stage 개수: 3 (load / dequant / matmul)
  - Buffer 배치: L2 fit 목표 — (3 tile size × 3 stage) 가 L2 절반 이하
- [ ] **Shape-aware dispatch 확장** (§07 binary → 3-way):
  - batch=1 → AVX-only
  - batch=4-8 → cascade
  - batch=16+ → full AMX
- [ ] **Tile config 비용 측정**: `ldtilecfg` 를 loop 밖으로 제거 (kernel 진입 시 1회)

### 구현
- [ ] **`csrc/cpu/cascade_gemm.cpp`** (신규)
  - 3-stage pipeline with software prefetch
  - Option: DSA integration (feasibility 확인 후)
  - LUT 연동 (§13 T-MAC GEMV) → cascade 의 dequant stage 를 LUT lookup 으로
- [ ] **Intermediate buffer pool**: 각 thread 마다 L2 resident 3× tile buffer
- [ ] **Prefetch intrinsic 적용**: `_mm_prefetch(addr, _MM_HINT_T1)` 로 k+2 tile

### 검증
- [ ] **Cache hit ratio**: `perf stat -e l2_rqsts.all_demand_references,l2_rqsts.miss`
- [ ] **Port utilization**: `perf stat -e uops_dispatched_port.port_0,port_1,port_5`
- [ ] **단일 shape 비교**: standalone AVX, standalone AMX, cascade 3 path
- [ ] **Shape sweep**: batch, hidden_dim, seq_len 변화에 따른 성능 cross-over

---

## 성공 조건

1. ✅ 특정 shape 에서 cascade 가 standalone AVX 또는 standalone AMX 보다 빠름
2. ✅ Memory wait 비중이 증가하지 않음 (staging 이 DDR 로 밀리지 않음)
3. ✅ L2 hit ratio 95%+ 유지
4. ✅ Port 5 (공유 port) 에서 bubble 비율 감소
5. ✅ decode step time 1.5–2× 개선 (batch 4-8 구간)

**Stop 조건**: cache-fit 실패 시 DDR 왕복 증가가 이득 상쇄 → staging schedule 재설계 또는 본 기법 드롭.

---

## 의존성

- **선행**: §07 ISA Binary Dispatch (cascade 는 binary 의 발전형), §15 AMX Pre-pack (weight 가 tile layout 이어야)
- **병행**: §13 T-MAC LUT (cascade 의 dequant stage 를 LUT 으로 대체 가능), §09 LUT ops
- **후속**: §17 Core Group Pipeline 은 multi-core cascade 확장

---

## 리스크

- **⚠ Staging overhead 가 이득 상쇄 (높음)**: Codex 경고 — 잘못 설계하면 중간 write/read 비용만 증가. prototype 검증 필수
- **Cache-fit 실패**: tile size × stages 가 L2 초과 → pipeline 이 아니라 DDR 왕복 폭증
- **port 5 공유 병목**: AVX-512 `vpshufb` 와 AMX 가 같은 port 사용 → 이론적 parallel 이 실제로 serial
- **DSA feasibility**: kernel 모드 설정 필요 — 컨테이너 환경에서 차단 가능성
- **T-MAN 원리의 x86 이식성 강한 가설 D**: 검증 전엔 수치 신뢰 불가

---

## 스택 호환성

- §07 ISA binary dispatch 의 발전형
- §13 T-MAC LUT: dequant stage → LUT lookup
- §15 AMX pre-pack: cascade 의 matmul stage 가 pre-packed weight 사용
- §17 Core group pipeline: single-core cascade → multi-core systolic

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_AVX_AMX_CASCADE` | `0` (기본) / `1` | 3-stage cascade pipeline 활성 |

전체 flag 테이블: [00_Overview.md](./00_Overview.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `csrc/cpu/cascade_gemm.cpp` — (신규)
- `csrc/cpu/torch_bindings_hybrid.cpp` — 등록
- `vllm/v1/worker/cpu_worker.py` — dispatch 분기
- `csrc/cpu/mem_opt.cpp` — NT memcpy, prefetch helper (기존)
