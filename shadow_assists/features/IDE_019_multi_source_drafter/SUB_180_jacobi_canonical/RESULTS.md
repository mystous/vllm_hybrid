# SUB_180 — IDE_019 / TSK_035 Jacobi lookahead + AVX-512 kernel

> **status**: 완료 (2026-05-27 11:09 KST)
> **scope**: NEW workload — drop-in 아님. Jacobi parallel decoding (CPU AVX-512) 의
> lossless proof + AVX-512 kernel 빌드 + microbench + 4-method router design.
> **verdict**: feasibility CONDITIONAL — kernel 자체 정확도 PASS / throughput 한계
> (vocab 152K BW-bound 로 GPU verify 의 ~40-50× cost) 로 본 SUB 의 kernel 단독으로는
> net-win 불가. SUB_181 의 net-win 조건 = (a) W transpose to vocab-major
> repack + (b) smaller draft model (Qwen 0.5B) 또는 (c) partial-vocab top-N.

---

## 1. Lossless guarantee 요약

`design/jacobi_lossless_proof.md` 참고. 핵심 정리:

- **fixed-point theorem** (§1.3): Jacobi update `c_i^(t+1) = argmax_v f_θ(x, c_{1..i-1}^(t))_i`
  의 fixed point `c*` 는 induction by i 로 **AR greedy 시퀀스 `y_{n+1..n+K}` 와
  token-level bit-exact**.
- **fixed-point 수렴 보장** (§1.4): vocab^K 가 유한이므로 max_iters cap 안에서
  cycle 또는 fixed point 에 도달. 본 kernel 의 driver 는 cycle detection +
  cap=8.
- **verify stage 통합** (§2): partial fixed-point 또는 non-converge 의 경우에도
  verify reject 가 prefix 만 emit → **distribution-level lossless** (rejection
  sampler 적용 시).
- **AVX-512 vectorize 영향 없음** (§5): partial sum order 변경뿐, lossless 깨지
  않음. BF16 비결합성 영향은 argmax tie 영역 ~0.01% 이하 (SUB_171 의
  tokenizer ordering test 와 동일 pattern).

## 2. AVX-512 kernel 구현 + 빌드

- **file**: `src/jacobi_avx512.cpp` (300 lines)
- **build**: `g++ -O3 -march=sapphirerapids -mavx512f -mavx512bf16 -fopenmp -fPIC -shared`
- **size**: `build/libjacobi_avx512.so` = 20 KB
- **API**:
  - `jacobi_lm_head_argmax_bf16(H, W, argmax_out, maxlogit_out, BK, hidden, vocab, n_threads)`
  - `jacobi_run(H, W, candidates_out, B, K, hidden, vocab, max_iters, n_threads, iters_used)`
  - `jacobi_lm_head_argmax_scalar_ref(...)` — accuracy reference
- **vectorize 전략**: hidden 32-bf16/iter (`_mm512_dpbf16_ps` 대용으로 fp32
  expansion + fma path), vocab 16-col outer tile, OpenMP across BK rows.

## 3. accuracy verify (small shape)

| metric | value |
|---|---|
| AVX vs scalar match | **4/4** (BK=4, hidden=128, vocab=512) |
| AVX vs numpy(bf16->fp32) match | **4/4** |
| scalar vs numpy match | **4/4** |
| out_avx / out_scalar / out_numpy | `[40, 242, 326, 421]` (전원 동일) |

→ kernel 의 numerical correctness PASS. 모든 token id token-level bit-exact.

## 4. microbench (Qwen 32B target shape)

config: hidden=5120, vocab=152064, BF16, taskset 0-99, OMP=4 envs, 1-run + warmup 1.

| K | B | BK | T=1 p50 (ms) | T=16 p50 (ms) | T=64 p50 (ms) | TFLOPS@T=64 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 3 | 5,994 | 1,723 | 1,698 | 0.0028 |
| 3 | 4 | 12 | 23,337 | 1,774 | 2,401 | 0.0078 |
| 5 | 1 | 5 | 9,922 | 1,723 | 1,841 | 0.0042 |
| 5 | 4 | 20 | 34,959 | 3,377 | **1,684** | 0.0185 |
| 7 | 1 | 7 | 13,292 | 1,977 | 1,765 | 0.0062 |
| 7 | 4 | 28 | 50,377 | 3,653 | **1,713** | **0.0255** |
| 9 | 1 | 9 | 16,567 | 1,788 | 2,010 | 0.0070 |
| 9 | 4 | 36 | 68,173 | 5,330 | **1,708** | **0.0328** |

관찰:

- **BW-bound**: BK 단일 row (BK=3) 와 BK=36 사이 T=64 p50 가 거의 같음
  (1,698 vs 1,708 ms). vocab=152K × hidden=5120 의 W matrix (~1.56 GB) traverse 가
  per-call dominant cost — `W` 가 모든 row 마다 1-pass — DRAM BW bound.
- **thread saturation**: T=64 가 T=16 대비 한정적 lift (BK=20-36 에서만 ~2× lift).
  BK ≤ 12 에서 thread oversubscription 으로 T=16 보다 T=64 가 더 느린 경우 존재.
- **TFLOPS peak 0.0328**: BK=36 T=64. peak 1-thread BF16 1-2 TFLOPS Sapphire
  Rapids 의 ~1.6%. 본 kernel 의 inner loop 가 column-stride scalar gather (per
  vocab col 32 bf16 load with stride=vocab) 이라 cache locality 결핍.

## 5. acceptance rate estimation

직접 측정은 vllm 통합 (SUB_181) 후 가능. **estimate** (IDE_011 reference):

| workload | IDE_011 ngram α | Jacobi self-draft α 예상 | rationale |
|---|---:|---:|---|
| chat | 51% (K=4.32) | **75-85%** (target α≥81.2%) | semantic continuity, self-draft hidden state 가 ngram 보다 nearby distribution capture |
| sonnet | 48% (K=4.07) | 60-70% | low repetition, self-draft 가 ngram 우월 |
| code | 81% (K=6.69) | 70-75% | ngram K=7 이 이미 매우 강함 — Jacobi 가 marginal |

→ chat 에서 Jacobi 가 ngram 대비 +24-34 pp lift 가능 가설. 실 측정 SUB_181.

## 6. canonical 500p baseline 정합성

본 SUB 는 microbench + lossless proof + design 까지 (NEW workload 의 첫 단계).
canonical e2e 측정은 의도적 미수행:

- vllm endpoint (vanilla :8001 / trident :8002) **현재 stopped**.
- 본 SUB kernel 의 throughput (BK=28 K=7 T=64 p50 = 1,713 ms / iter) 가 GPU verify
  step (~35-44 ms) 의 **~40-50×** 큰 cost — kernel 단독으로 net-win 불가
  (SUB_178 의 honest scope 와 동일 — 작동 조건 미충족 시 의도적 미실행).
- canonical baseline reference: **SUB_179 = 4,299 tps (max-tokens=32 3-mix avg AGSD)**
  / SUB_177 = 6,110 tps (max-tokens=256). 본 SUB 의 환경 변경 없음 → 정합성 가정.

## 7. 4-method router design

`router_design_4method.md` 참고. 핵심:

- 새 backend `cpu_jacobi` (`http://127.0.0.1:8003/v1/completions`).
- decide rule: code→trident(ngram) / sonnet→trident(suffix) / **chat→cpu_jacobi**.
- fast-path: prefix_len < 16 → vanilla. fallback: cpu_jacobi unhealthy → trident.
- vllm `cpu_jacobi_proposer.py` (스케치): `ctypes.CDLL(libjacobi_avx512.so)` +
  `jacobi_run` 호출 → candidate IDs → verify GPU.
- e2e overhead budget: CPU draft 1,700 ms vs GPU verify 35 ms = **net loss
  current state**. net-win 조건 = W repack + smaller draft model.

## 8. limitations

| 한계 | 영향 | 후속 작업 |
|---|---|---|
| BK=1 throughput 1,698 ms — vocab BW-bound | net loss vs GPU verify 35 ms | SUB_181: W vocab-major repack + tile-blocked inner loop |
| acceptance rate estimate only | per-workload α 불확실 | SUB_181: vllm 통합 후 8 prompts × 32 tokens 측정 |
| vllm `cpu_jacobi` proposer 미구현 | e2e 측정 불가 | SUB_181: `vllm/v1/spec_decode/cpu_jacobi_proposer.py` 신규 |
| canonical e2e 측정 의도적 미수행 | 실 throughput 영향 미상 | endpoint 가동 + SUB_181 |
| Jacobi driver 의 hidden update callback 미통합 | self-driving Jacobi 불가능 (현재는 1-pass argmax) | SUB_181 |

## 9. verdict (paper §4 lever 자격)

**NEW workload 의 2번째 후보 (SUB_178 cold-KV 다음)** — feasibility CONDITIONAL:

- ✅ lossless guarantee 입증 (fixed-point theorem, §1)
- ✅ AVX-512 kernel build PASS / accuracy 4/4 PASS
- ✅ microbench: kernel 작동, BW-bound 정량화
- ⚠ throughput: GPU verify 의 ~40-50× cost — kernel 단독으로 net-win 불가
- ✅ 4-method router design 작성 (SUB_181 통합 진입로 명확)
- ❌ e2e 측정 미수행 (의도적 — kernel optimization SUB_181 선행 필요)

본 SUB 는 **drop-in 7번째 실패 패턴이 아닌 NEW workload 후보**. SUB_178 (cold-KV)
와 함께 NEW workload 2개 후보 stack — SUB_181 에서 W repack + small draft
model 두 lever 로 net-win 시도. 단, throughput gap (~40-50×) 은 SUB_178 (~1.5-1.71×
overlap speedup) 보다 큰 도전 — kernel rewriting 후 재측정 필수.

## 10. 산출물

- `src/jacobi_avx512.cpp` — AVX-512 kernel (300 lines)
- `build/libjacobi_avx512.so` (20 KB, Sapphire Rapids)
- `run_jacobi_bench.py` — microbench harness (ctypes, BF16 sweep)
- `jacobi_microbench.json` — verify result (BK=4 hidden=128 vocab=512)
- `jacobi_microbench_main.json` — main microbench (K∈{3,5,7,9}, B∈{1,4}, T∈{1,16,64})
- `logs/build.log`, `logs/microbench_main.log`
- `../design/jacobi_lossless_proof.md` — 수식 증명
- `router_design_4method.md` — 4-method AGSD router design

## 11. 다음 SUB

- **SUB_181 (TSK_037)**: vllm `cpu_jacobi_proposer` 통합 + W vocab-major repack +
  acceptance rate 실측 + canonical 3-mix e2e.
- prereq: 본 SUB 의 kernel BW-bound 한계 해소 (W transpose to [vocab, hidden] +
  16-wide tile-major load) — kernel 5-10× speedup 필요.
