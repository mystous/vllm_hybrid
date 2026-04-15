# 12. Barrier/Sync 감소 (OMP Persistent Region)

**Tier**: 2
**상태**: ⭕ 미구현 (미계측)
**예상 이득**: 프로파일 의존 (G0 산출물 기반)

---

## 왜 필요한가

Transformer layer 당 sublayer 는 8개 (QKV, O, Gate, Up, SiLU, Down, RMSNorm×2). 각 sublayer 마다 OMP parallel region 이 열리고 닫히면:
- **thread team 생성/파괴 오버헤드** (~수십 us per region)
- **barrier 동기화** (모든 thread 완료 대기)
- layer 80개 × sublayer 8 = **640 barrier per step**

640 × 20us = **12.8ms overhead** (decode step 25ms 의 절반). 이게 실제로 발생하고 있는지는 **§01 G0 계측 전까진 미확정** 하지만 강한 가설.

---

## 기술적 배경

### OMP parallel region 의 비용

```cpp
// 매 sublayer 마다 반복되는 패턴
void sublayer_forward(...) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        // 병렬 작업
    }
    // 여기서 implicit barrier (모든 thread 완료 대기)
    // parallel region 닫힘
}
```

각 `#pragma omp parallel` 은:
1. **thread team 할당** (KMP_BLOCKTIME 에 따라 재사용 or 재생성)
2. **work distribution**
3. **barrier 동기화** (region 끝)
4. **region 종료**

`KMP_BLOCKTIME=0` (§05) 으로 spin 비활성 시 team wake-up cost 더 커짐 (~5-10us per region).

### Persistent parallel region 패턴

```cpp
#pragma omp parallel
{
    for (int layer = 0; layer < n_layers; layer++) {
        // All sublayers within ONE parallel region
        compute_qkv(tid, nthreads, ...);
        #pragma omp barrier  // 명시 필요한 동기화만
        compute_attention(tid, nthreads, ...);
        #pragma omp barrier
        compute_mlp(tid, nthreads, ...);
        // ...
    }
}
```

- **Team 1회 생성** → 전체 model forward 동안 유지
- Barrier 는 명시적으로 필요한 곳에만
- Thread 가 sublayer 간 spin (data dependency 없는 부분은 barrier 생략)

### Chunk scheduling

OMP `schedule` 최적화:
- `schedule(static)` (기본): 작업을 thread 수로 균등 분할 — steal 없음
- `schedule(static, chunk_size)`: `chunk_size` 블록 단위 분배
- `schedule(dynamic)`: runtime 작업 할당 — overhead 크지만 불균형 해소
- `schedule(guided)`: dynamic 의 저오버헤드 변형

Dense GEMM 에선 `static` 이 최적 (load balance 자연스러움). Sparse/불균형 작업에선 `dynamic`.

### Data dependency 분석

Sublayer 간 dependency:
- QKV → Attention (Q/K/V 필요)
- Attention → O (attn_out 필요)
- O → Residual + RMSNorm
- RMSNorm → Gate, Up (같은 input)
- Gate → SiLU → ... → Down
- Down → Residual
- 다음 layer 로

**Dependency 없는 것**:
- Gate 와 Up: 같은 input 에서 독립 연산 → **병렬 가능** (§08 fusion 에서 해결)
- Attn head 간: 독립 → heads 차원으로 parallelize

### oneDNN scratchpad reuse

oneDNN primitive 가 매번 scratchpad 할당하면 malloc overhead 반복. `primitive_desc.query(scratchpad_md)` 로 크기 조회 + pre-allocate + `exec_args[DNNL_ARG_SCRATCHPAD]` 로 reuse.

---

## 관련 참고 문헌

- **OpenMP Specification 5.2 §"parallel construct"**: https://www.openmp.org/spec-html/5.2/openmpsu29.html
- **Intel OpenMP Optimization Guide**: https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/openmp-parallelism-and-loop-parallelism.html
- **Chandra et al. (2001) "Parallel Programming in OpenMP"**: Morgan Kaufmann — fork-join overhead 분석
- **Codex playbook Tier 2 barrier/sync 감소**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md`
- **LLVM OpenMP Runtime documentation**: https://openmp.llvm.org/
- **KMP_BLOCKTIME 영향**: Intel 공식 문서 (§05 참조)
- **oneDNN scratchpad**: https://oneapi-src.github.io/oneDNN/dev_guide_attributes_scratchpad.html

---

## 구체 작업

### 계측
- [ ] **§01 G0 계측 산출물 확보**: step 시간 중 barrier/sync 비중
- [ ] **VTune "OpenMP Analysis"**: serial 영역 + parallel efficiency
- [ ] **Wait time per sublayer**: `omp_get_wtime()` 기반 직접 측정

### Persistent region 구현
- [ ] **전체 forward 를 1개 parallel region 으로 재구성** 가능성 탐색
  - 제약: Python layer (HuggingFace `model.forward`) 는 각 layer 가 별도 Python call
  - Workaround: C++ level 에서 persistent region 을 만들고 Python 에서 1회 호출 (`full_forward_cpu`)
  - Option A: 전체 model forward 를 `torch.ops._C_cpu_ops.full_model_forward` 로 C++ 이식 (큰 작업)
  - Option B: layer 단위 persistent (layer 당 1 region) — 현실적 절충
  - Option C: 특정 sublayer group (Attn+O, MLP) 만 persistent — 작은 시작
- [ ] **`csrc/cpu/persistent_forward.cpp`** (신규) — Option B/C 구현

### Chunk/schedule 최적화
- [ ] **기존 OMP directive 들에 `schedule` clause 명시 추가**
- [ ] **작은 sublayer 는 thread 수 제한** (`#pragma omp parallel for num_threads(N)`)

### Scratchpad reuse
- [ ] **oneDNN primitive 의 scratchpad pre-allocate**
- [ ] IPEX 가 이미 하는지 확인, 없으면 custom path 에 적용

---

## 성공 조건

1. ✅ §01 G0 결과에서 barrier/sync 비중 측정
2. ✅ Persistent region 후 barrier 횟수 640 → 80 (layer 단위) or 그 이하
3. ✅ step time 중 wait time 비중 감소 (VTune 비교)
4. ✅ `num_seqs` 증가 시에도 wait time 이 선형 이하로 증가 (batch scaling 에 기여)
5. ✅ decode step time 5–15% 감소

---

## 의존성

- **선행**: §01 G0 계측 (barrier 비중 확인), §05 OMP env (KMP_BLOCKTIME)
- **병행**: §08 Kernel Fusion (sublayer 수 자체 감소)
- **후속**: §13, §14 kernel 투자의 ceiling 를 높임

---

## 리스크

- **Persistent region 구현 복잡도**: Python ↔ C++ 경계가 자연스러운 region 경계. 이를 넘어 persistent 로 가려면 C++ 에서 model forward 재구현 필요 — **매우 큰 작업**
- **Dependency 분석 실수 시 race condition**: sublayer 간 data dependency 가 미묘. synchronization 생략으로 bug 유발
- **IPEX 내부 OMP 와 충돌**: IPEX primitive 내부에서도 OMP region 엶 → nested parallelism overhead. `OMP_NESTED=TRUE` 로 해결 or IPEX 직접 수정
- **이득이 예상보다 작음**: barrier 가 이미 team reuse (`KMP_BLOCKTIME=infinite`) 로 저렴할 수도. 계측이 결정적

---

## 스택 호환성

- §08 Kernel Fusion: sublayer 수 감소 → region 수 감소 (fusion 과 persistent 가 곱셈적)
- §13 T-MAC LUT, §14 Cascade: 이들이 추가 parallel region 열면 본 개선분 상쇄 — 일관 설계 필수

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_PERSISTENT_OMP` | `0` (기본) / `1` | Persistent OMP region 모드 |

전체 flag 테이블: [00_Overview.md](./00_Overview.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `csrc/cpu/utils.cpp` — `init_cpu_threads_env` (OMP team 초기화, 이미 persistent 성격)
- `csrc/cpu/persistent_forward.cpp` — (신규, Option B/C 구현 시)
- `vllm/v1/worker/cpu_worker.py` — execute_model
- `vllm/v1/worker/cpu_model_runner.py` — model forward driver
- 각 `csrc/cpu/fused_*.cpp` — `#pragma omp` 사용 부분 검토
