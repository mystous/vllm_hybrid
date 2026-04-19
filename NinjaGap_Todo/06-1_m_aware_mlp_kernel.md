# 06-1. M-aware MLP Q8_0 Kernel (batch 처리 결함 수정)

**Tier**: 1 (§06 의 구현 완성)
**상태**: ⭕ 미구현 (원인 확정, 설계만 정리된 상태)
**선행**: §06 (Q8_0 dispatch 경로) 구현 완료 (commit `6f904b39b`, 2026-04-19)
**중요도**: §06 의 batch 영역 역효과를 정상화하는 필수 후속 작업

---

## 왜 §06 에서 분리했는가 (정직한 이력)

§06 최초 commit 시점에 `06_hot_path_wiring.md` 에 "Phase A 구현 완료" 라고 적었지만, Phase B 는 **내용 없는 placeholder** 였다. TP=8 baseline 대조 측정 (2026-04-20) 에서 §06 의 batch 영역 역효과를 발견한 뒤 그 수정 작업을 "Phase B" 에 사후적으로 채워 넣는 건 기록 품질이 나쁘다는 지적을 받아, **정식 § 번호 06-1 로 분리**하고 Phase 용어는 철회한다.

즉 §06-1 은 "처음부터 계획된 후속" 이 아니라 **측정으로 드러난 실체 결함의 정정 작업**이며, 이 점을 문서에 명시한다.

---

## 왜 필요한가

§06 (`6f904b39b`, 2026-04-19) 는 "Qwen2 MLP → Q8_0 dispatch" 경로 구축에 성공했고 seqs=1 에서 명확한 이득을 보였다. 하지만 2026-04-20 의 TP=8 baseline (`g0_00_qwen2.5_32b_base`) 과의 대조 측정에서 seqs ≥ 2 부터 **§06 on 이 역효과**임이 확정됐다:

| seqs | base outTP (§06 off) | §06 on outTP | Δ |
|---:|---:|---:|---:|
| 1 | 908.9 | **1069.7** | **+18%** (이득) |
| 2 | 895.9 | 654.6 | −27% |
| 4 | 595.3 | 370.0 | −38% |
| 8 | 575.2 | 211.2 | −63% |
| 16 | 637.8 | 118.2 | −81% |
| 32 | 423.1 | 63.7 | −85% |
| 64 | 339.7 | 32.2 | **−90%** |

wall time 도 같은 방향 (seqs=64 에서 §06 off 181.7 s vs §06 on 1918.3 s). routing 정책은 양쪽 동일하므로 **CPU 처리 속도 자체의 차이**이며, 이는 §06 Q8_0 kernel 의 구현 결함에서 온다.

---

## 원인 확정 (`csrc/cpu/quant_q8_0.cpp` 241-247 행)

```cpp
void q8_0_linear_impl(...) {
    const int M = input.size(0);
    // ... input 을 M별 dynamic quantize (병렬) ...

    // Compute output
    for (int m = 0; m < M; ++m) {                           // ← M 축 순차 반복
        q8_0_gemv_vnni_impl(xq_ptr + m*K, xs_ptr[m],
                            w_ptr, out_f32 + m*N, N, K);    // ← GEMV 를 M 번 호출
    }
}

// q8_0_gemv_vnni_impl:
#pragma omp parallel for schedule(static)
for (int n = 0; n < N; ++n) { ... }                         // ← N 축만 병렬
```

**§06 Q8_0 kernel 은 GEMV 를 M 번 순차 호출하는 batch-oblivious 구현**. M 축이 GEMM 차원으로 활용되지 않아 wall 이 M 에 선형 증가. 반면 IPEX (AMX BF16, baseline) 는 M 을 GEMM 차원으로 활용해 batch amortize. 그래서 batch 가 커질수록 §06 이 IPEX 대비 급격히 불리해 보이는 것.

즉 "AMX 가 VNNI 보다 빠르다" 가 이 현상의 원인이 아니라, **§06 kernel 이 M 축 처리를 안 했다**가 원인이다. AMX / VNNI 의 compute throughput 우열은 별개 논점.

---

## 개선 방향

### M 분기 (kernel 내부 shape-aware dispatch)

| M 구간 | 처리 경로 | 근거 |
|---|---|---|
| **M = 1** | 기존 GEMV 유지 | decode 는 memory-bound. weight BW 절반화 이득 보존. 현재 이미 +18% outTP 확인 |
| **M > 1 (Phase 1)** | VNNI INT8 **GEMM** (A 경로) | M 축을 GEMM tile 로 활용해 amortize. 기존 `gemm_vnni.cpp::int8_gemm_vnni` 재활용 |
| **M ≥ 임계값 (Phase 2, 선택)** | AMX-INT8 tile op (B 경로) | SPR AMX 는 INT8 도 지원 (peak ~3958 TFLOPS). compute-bound 전환 구간 커버 |

임계값은 **실측으로 결정**. 하드코드 대신 kernel 내부 shape check.

### (A) VNNI INT8 GEMM path

기존 자산:
- `csrc/cpu/gemm_vnni.cpp::int8_gemm_vnni` — 6×16 micro-kernel, M 축 tile 이미 있음
- Q8_0 block 포맷 (34 bytes = fp16 scale + int8[32])

작업:
1. Q8_0 block → VNNI 입력 포맷 adapter. scale 은 kernel 외부에서 output 에 FP32 apply
2. `q8_0_linear_impl` 의 `for (m=0; m<M; ++m) q8_0_gemv_vnni_impl(...)` 를 `if (M == 1) gemv(...) else gemm(...)` 분기로 교체
3. GEMV 경로 M=1 유지

### (B) AMX-INT8 path (Phase 2, 선택)

SPR AMX intrinsics (`_tile_loadconfig`, `_tile_dpbssd`) 로 INT8 tile matmul. Q8_0 dequant fusion 필요.

Phase 2 는 (A) 완료 후 seqs≥16 에서 여전히 baseline 대비 손해면 착수.

---

## 구체 작업

### (A) Phase 1 — VNNI INT8 GEMM
- [ ] Q8_0 block → VNNI 입력 포맷 adapter 구현
- [ ] `q8_0_linear_impl` 에 M>1 GEMM 경로 분기 (M=1 GEMV 유지)
- [ ] dev smoke test: M=4/8/16 으로 dummy tensor 호출, 결과 tensor shape/dtype 일치
- [ ] H100x8 sweep 재측정 (seqs 1/2/4/8/16/32/64)

### (B) Phase 2 — AMX-INT8 (조건부)
- [ ] (A) 측정에서 seqs≥16 outTP 가 baseline 대비 여전히 열세일 때 착수
- [ ] AMX tile config + INT8 matmul intrinsics
- [ ] Q8_0 dequant fusion

### 정확도 검증
- [ ] 1차: greedy top-1 token sequence 동일 (100 샘플)
- [ ] 2차 fallback: exact match ≥ 95%
- [ ] 3차 fallback: PPL 열화 < 0.5 (WikiText-2)

---

## 성공 조건

### 1차 판정 — `g0_06_qwen2.5_32b` (§06 on) 대비 회복

§06-1 의 직접 목적은 "§06 의 batch 영역 역효과 해소" 다. 따라서 primary baseline 은 **§06 on 상태 (`g0_06_qwen2.5_32b`)** 이며 여기 대비 개선이 나와야 한다.

- seqs = 2/4/8 에서 **wall/TPOT 가 g0_06 대비 회복** — 즉 §06 on → (§06 on + §06-1) 로 가면서 wall 감소, TPOT 감소
- seqs = 1 은 §06 on 과 **동등** (GEMV 경로 변경 없음 — M=1 dispatch 그대로)

### 2차 판정 — `g0_00_qwen2.5_32b_base` (baseline) 대비 역효과 해소

1차 판정이 긍정이어도 §06-1 적용본이 baseline (§06 off) 보다 나쁘면 "부분 성공": kernel 경로는 유효하나 VNNI compute 한계로 완전 회복 안 된 경우.

- seqs 2/4/8 에서 §06-1 적용본이 baseline 과 **동등 이상** 이면 완전 성공
- 열세이면 §06-1 scope 안에서의 최대치를 달성한 것이고, 추가 이득은 §24 병합 영역 판단으로 이관

### Scope 밖

- **seqs ≥ 16 은 §06-1 판정 대상이 아님**. Compute-bound 전환 구간이며 VNNI peak 가 AMX BF16 에 열세인 band. 이 구간 이득은 Q8_0 포맷 제약상 AMX-INT8 로만 가능하고, 이는 §24 (W8A8) 와 병합 재설계 영역
- `measurement_results/H100x8/g0_06_1_qwen2.5_32b/seqs{16,32,64}` 수치는 기록용으로 수집하되 **합격/불합격 판정에 쓰지 않는다**

### 정확도 게이트 (3 단 fallback)

1. **1차**: Greedy top-1 token sequence 동일 (100 prompt)
2. **2차 fallback**: Short generation exact match ≥ 95% (100 sample)
3. **3차 fallback**: WikiText-2 PPL 열화 < 0.5

1차 충족 시 다음 fallback 생략.

---

## 의존성

- **선행**: §06 (dispatch 경로) 완료. 이 위에 kernel 만 손봄
- **병행**: 없음 (kernel level 수정이라 다른 §와 직교)
- **후속**: §24 W8A8 (activation INT8) 은 §06-1 의 GEMM infra 위에 activation 도 INT8 확장

---

## 리스크

- **VNNI GEMM 이 AMX BF16 을 이기지 못함 (large M)**: compute-bound 구간에서 AMX peak > VNNI peak. §06-1 만으로 seqs ≥ 16 break-even 어려움. 이 구간은 의도적으로 scope 밖
- **Kernel shape 경계 curation**: M 임계값 (현재 16) 은 실측 iteration 후 조정 가능
- **정확도 누적**: GEMM accumulator 가 GEMV 와 정확히 같게 구현돼야 token-identical 유지. FP32 accumulate 명시
- **Write locality / false sharing**: `output[m*N + n]` 을 N 축 parallel thread 가 인접 n 으로 쓰면 같은 M 행의 cache line 경합 가능. 관찰 시 thread 별 local buffer → 마지막에 scatter 로 회피
- **AMX-INT8 path 는 §06-1 scope 밖**: large M 구간은 Q8_0 포맷 제약으로 AMX 직접 적용 어려움. §24 와 병합 재설계 영역

---

## 실패 시 디버깅 우선순위 (kernel 알고리즘 의심 전에 확인)

측정 결과가 약하거나 §06 on 대비 변화가 없을 때, "내가 짠 kernel 알고리즘이 틀린가?" 보다 먼저 아래 순서로 확인한다. §06 진행에서 겪었던 "실제로 코드가 실행되지 않던" 사례의 재발 방지:

1. **AVX-512 / VNNI 경로가 빌드에 실제로 포함됐는가**
   ```bash
   nm -D vllm/_C_cpu_ops.abi3.so | grep -iE 'q8_0_gemm_vnni_impl|q8_0_gemv_vnni_impl'
   ```
   - 양쪽 심볼 모두 나와야 정상. 한 쪽만 or 없음 = `#if defined(__AVX512F__) && defined(__AVX512VNNI__)` guard 가 walked off. gcc 버전 (< 12.3) 또는 `-mavx512vnni` flag 누락 확인

2. **gcc 버전**
   ```bash
   g++ --version | head -1
   ```
   - `< 12.3` 이면 `cmake/cpu_hybrid_extension.cmake` 의 gate 에서 VNNI OFF. §06 때 겪은 이슈. `g++-12` 설치 + `CXX=g++-12` 로 재빌드

3. **Kernel 이 실제로 dispatch 되는가 — runtime 검증**
   - `VLLM_HYBRID_KERNEL_TRACE=1` + `HYBRID_CPU_MAX_SEQS=4` 로 서버 짧게 띄운 뒤 log 에 M=4 호출 확인
   - M 분기 진단 로그는 §06-1 commit 자체에는 포함 안 시킴 (production 노이즈). 필요 시 임시 `fprintf(stderr, ...)` 후 debug
   - `q8_0_linear_impl` 의 M 값이 실제로 기대한 M 으로 들어오는지 (IPEX 의 layer 중 M=1 만 오는 경우도 가능)

4. **Write locality / false sharing**
   - `output[m*N + n]` 을 여러 thread 가 인접 n 으로 쓸 때 같은 cache line 에 다수 thread 가 동시 접근
   - `perf stat -e cache-misses,cache-references` 로 cache miss ratio 확인
   - 확인되면 thread-local scratch buffer 후 마지막에 한 번 scatter, 또는 n 축 tile 을 cache line 경계 (16 float = 64 bytes) 에 맞추기

5. **Q8_0 block alignment 및 dispatch 경로**
   - `n_blocks_per_row` 값 확인 (K/32 배수). non-32 K 면 tail scalar path
   - M 가 정말 1<M<16 범위로 들어오는지. IPEX 가 layer 를 wrap 해서 M 값을 다르게 만들 가능성 없는지

6. **위 모두 통과하면 그때 kernel 알고리즘 의심**
   - madd+reduce 대신 VNNI `_mm512_dpbusd_epi32` 사용 (u8×s8), s8s8 compensation
   - Weight block load 재사용 실제 효과 측정 (block 1개에 대해 M inner 가 L1 에서 돌아가는지)

이 디버깅 순서는 §06 에서 확인된 원칙 그대로 따른다: **"현상만 보고 kernel 로직 의심하지 말고 빌드/실행 경로부터 검증"**.

---

## 실행 flag

§06 의 기존 flag 그대로:
- `HYBRID_VNNI_HOT_PATH=1` / `--hybrid-vnni-hot-path`

§06-1 은 kernel 내부 수정이라 **별도 flag 없이** §06 on 이면 자동 적용. 단 Phase 1/2 선택을 위한 kernel-internal guard (ISA detection) 는 필요:
- AMX 감지 + M ≥ 임계값 → (B) 경로
- 그 외 → (A) 경로
- M = 1 → GEMV

---

## 관련 코드 위치

- `csrc/cpu/quant_q8_0.cpp` — **주 수정 대상** (M>1 GEMM 분기 추가)
- `csrc/cpu/gemm_vnni.cpp` — `int8_gemm_vnni` 재활용 (adapter 작성)
- `vllm/v1/worker/hot_path_wiring.py` — 변화 없음 (kernel-only 수정)
- `csrc/cpu/torch_bindings_hybrid.cpp` — op 이름 / signature 그대로

---

## 관련 기록

- §06 dispatch 완료 commit: `6f904b39b` (2026-04-19)
- §06 on 측정: `measurement_results/H100x8/g0_06_qwen2.5_32b/`
- TP=8 §06 off baseline: `measurement_results/H100x8/g0_00_qwen2.5_32b_base/` (2026-04-20)
- batch 역효과 실측 발견: `Tech_done.md v7`
- §06 완결 표기 철회 + §06-1 도입 기록: `Task_done.md v7`
