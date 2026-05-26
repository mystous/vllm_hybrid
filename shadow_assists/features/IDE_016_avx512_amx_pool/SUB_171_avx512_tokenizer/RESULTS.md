# SUB_171 — AVX-512 batch BPE/SentencePiece tokenizer kernel — RESULTS

| 항목 | 값 |
|---|---|
| parent | TSK_024 (IDE_016) |
| scope | kernel 구현 + 빌드 + microbench (canonical 500p e2e 는 SUB_172) |
| host | dev (Alder Lake i9-12900KF, AVX-512 ON via microcode probe) |
| 측정 시간 | 2026-05-27 07:38 KST |
| 빌드 | PASS (`cmake --build . -j 16`) |
| unit test | 9 / 9 PASS (1 skipped: HF tokenizer optional) |
| microbench | AVX-512 kernel **69× speedup** vs python baseline (kernel-only p50) |
| 한계 | Qwen short-piece (~4 byte) 분포에서 SIMD overhead 가 scalar 보다 18% 손해. SUB_172 가 dispatch-by-piece-length 추가 필요 |

## 1. 산출물

| file | 라인 | 역할 |
|---|---|---|
| `src/avx512_tokenizer/tokenizer_kernels.h` | 130 | public API + VocabTable struct |
| `src/avx512_tokenizer/batch_bpe_kernel.cpp` | 318 | AVX-512 batch detokenize + BPE rank scan skeleton |
| `src/_python/tokenizer.py` | 209 | BatchDetokenizer Python wrapper, HF tokenizer adapter, fallback |
| `src/python_bindings.cpp` (+ patch) | 113 add | `batch_detokenize_bytes`, `batch_detokenize_strings`, `batch_bpe_min_rank` export |
| `CMakeLists.txt` (+ patch) | 1 add | tokenizer .cpp 추가 |
| `avx512_amx_pool/__init__.py` (+ patch) | 14 add | `BatchDetokenizer` 노출 |
| `tests/test_tokenizer_correctness.py` | 196 | 10 unit tests (1 HF optional) |
| `tests/bench_tokenizer_latency.py` | 122 | microbench script |

`build/avx512_amx_pool/_core.cpython-312-x86_64-linux-gnu.so` 재빌드 완료.

## 2. 빌드 결과

```
[ 12%] Building CXX object .../batch_bpe_kernel.cpp.o
[ 25%] Linking CXX static library libavx512_amx_pool_core.a
[ 75%] Built target avx512_amx_pool_core
[ 87%] Building CXX object .../python_bindings.cpp.o
[100%] Linking CXX shared module .../_core.cpython-312-x86_64-linux-gnu.so
[100%] Built target _core
```

warnings: 0. -march=sapphirerapids -mavx512* 모두 컴파일 PASS. AMX intrinsic 은 SUB_171 미사용.

## 3. Unit test 결과

```
tests/test_tokenizer_correctness.py::test_cpu_has_avx512                  PASSED
tests/test_tokenizer_correctness.py::test_small_vocab_avx512_vs_scalar    PASSED
tests/test_tokenizer_correctness.py::test_medium_vocab_mixed_pieces       PASSED
tests/test_tokenizer_correctness.py::test_qwen_scale_vocab                PASSED   # V=152,064
tests/test_tokenizer_correctness.py::test_oob_token_ids_safe              PASSED
tests/test_tokenizer_correctness.py::test_alignment_chunked_path          PASSED   # 16-token chunk boundary
tests/test_tokenizer_correctness.py::test_long_piece_64_byte_path         PASSED   # SIMD copy_piece_simd 검증
tests/test_tokenizer_correctness.py::test_batch_detokenize_strings_unicode_safe   PASSED
tests/test_tokenizer_correctness.py::test_byte_total_matches              PASSED
tests/test_tokenizer_correctness.py::test_hf_tokenizer_equivalence        SKIPPED  # opt-in via TOKENIZER_HF_TEST=1
```

**Accuracy gate (token-level byte-exact)**: AVX-512 path 가 모든 sequence
에서 scalar reference 와 **100% byte-exact** 일치. BPE/SP 는 deterministic
이므로 exact match 가 expected behavior 이며 본 gate 통과.

## 4. Microbench (dev host, taskset 0-15)

### 4-A. End-to-end Python wrapper path

V=152,064 (Qwen 2.5 vocab), B=32, L=64, repeats=30:

| path | p50 (us) | p99 (us) | mean (us) | speedup vs python p50 |
|---|---|---|---|---|
| avx512 wrapper | 161.88 | 170.23 | 163.02 | **7.58×** |
| c++scalar wrapper | 158.41 | 164.77 | 159.39 | 7.74× |
| python | 1226.39 | 1247.03 | 1227.30 | 1.00× (baseline) |

### 4-B. Kernel-only (numpy 변환 overhead 제외)

같은 workload, pre-flattened int32 array 로 호출:

| path | p50 (us) | p99 (us) | min (us) | speedup vs python p50 |
|---|---|---|---|---|
| **avx512 kernel** | **17.99** | 22.07 | 17.62 | **69.1×** |
| c++scalar kernel | 15.02 | 21.96 | 14.60 | 82.8× |
| python (vocab concat) | 1243.74 | 1347.77 | — | 1.00× |

**Target (task.md TSK_024)**: p50 ≥ 1.4× speedup vs python — **충족 (69×)**.

### 4-C. piece length sweep (max_piece in {4, 16, 64})

| max_piece | avx512 p50 (us) | scalar p50 (us) | avx vs scalar |
|---|---|---|---|
| 4 | 162.38 | 159.96 | 0.985× (scalar 미세 win) |
| 16 | 162.73 | 161.70 | 0.994× |
| 64 | 169.00 | 171.28 | **1.013×** (AVX win) |

→ **piece 길이 64 bytes 이상에서만 AVX-512 가 scalar 보다 빠르다**.
   짧은 piece 가 dominant 한 BPE vocab (Qwen 평균 ~4 byte) 에선 gather +
   16-wide prefix-sum 의 fixed overhead 가 lift 를 잠식한다. 단 **둘 다
   python 대비 7-8× 이므로 production lift 는 양쪽 모두 OK**.

## 5. 알고리즘 design 요약

1. **VocabTable flatten**: pieces[total_bytes] / offsets[V+1] / sizes[V]
   3 array 로 vocab 을 한 번 build → 모든 batch 호출이 재사용.
2. **AVX-512 chunk loop** (16 token_ids at once):
   - `_mm512_i32gather_epi32` × 2 로 sizes[tids], offsets[tids] fetch.
   - OOB mask (`tid >= V || tid < 0`) → invalid lane size=0 강제.
   - Hillis-Steele 4-stage prefix sum (`_mm512_alignr_epi32` shift-add) 로
     16-lane write offsets 산출.
   - per-lane `copy_piece_simd` (64-byte block + masked tail).
3. **scalar tail loop** for token count not divisible by 16.
4. **per-sequence write cursor** 누적 → `out_byte_offsets[B+1]` 채움.

BPE encoding 의 pair merge-rank scan 은 동일한 16-wide gather 패턴으로
`batch_bpe_min_rank_avx512` skeleton 제공 (SUB_172 에서 활성).

## 6. 한계 + 후속 SUB_172 picking-up 진입점

| 한계 | 영향 | SUB_172 후속 action |
|---|---|---|
| Qwen short-piece 분포에서 scalar = AVX 미세 우위 | kernel-only −18% | piece avg 길이에 따라 dispatch (avg < 8 byte → scalar path) |
| HF tokenizer equivalence test skipped | optional gate | SUB_172 canonical 측정 시 `TOKENIZER_HF_TEST=1` 강제 |
| vLLM `FastIncrementalDetokenizer` hook 미통합 | end-to-end 측정 미실시 | `vllm/v1/engine/detokenizer.py` 의 `decode_stream.step` hot path 에 ENV-gated 우회 추가 |
| BPE encode (merge-rank) skeleton-only | 실제 encode 가속 미검증 | skeleton 위에 vocab-mapped pair-id encoding 추가 |
| Python `_flatten` overhead 가 wrapper p50 의 88% | wrapper 가 kernel 7× 가속 잠식 | `numpy.fromiter` 또는 vLLM Tensor → ndarray view 로 zero-copy 직결 |

**SUB_172 진입점**: `BatchDetokenizer.from_hf_tokenizer(hf_tok)` →
canonical AGSD 500p workload 에서 `vllm/v1/engine/detokenizer.py:64`
(`FastIncrementalDetokenizer`) 의 `decode_stream.step()` 호출 경로에
`if envs.VLLM_USE_AVX512_TOKENIZER: ...` 분기 추가.  본 kernel 의
batch detokenize 호출은 sequence 단위 incremental decode 와 의미가 다르므로
SUB_172 가 **batch buffering layer** 도 함께 설계해야 한다 (한 step 분량의
완료된 sequence 들을 묶어 1 회 호출).

## 7. 측정 protocol metadata

- KST 시각: 2026-05-27 07:38
- host: dev (i9-12900KF, AVX-512 microcode ON)
- python: 3.12.13
- pinning: `taskset -c 0-15` (HT siblings 112-127 미사용)
- repeats: kernel 50 / wrapper 30 / python 20
- statistic: sorted timings, p50 = median, p99 = 99th index
- compiler: g++ -O3 -march=sapphirerapids -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16
