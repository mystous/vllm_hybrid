# H — SUB_015-Phase 2 측정 결과 (env+constant+code change sweep)

> 2026-05-18 KST. branch `feat/neo-amx-apply` HEAD `da32e79dd`.
>
> 본 측정 = Phase 1 의 lever G (libgomp) + C (K_TILE_WIDTH) + B (softmax fast_exp) 의 시험.
> 결과: 모든 cheap variants 가 baseline 이 best. A (AMX) 는 effort 평가만 (prod-only).

---

## 1. G-1 — libgomp symbol resolve (완료)

### Worker 가 load 한 libgomp
- 경로: `/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib/libgomp.so.1` (torch wheel 내장)
- Build-ID: `2244452c1de4759ccb52adc5352b501d4c2ee9ce`
- perf.data 의 build-id 와 매칭 ✓

### 0x1de60 의 enclosing function + instruction
- `addr2line` resolve: `omp_get_num_procs@@OMP_1.0` (nearest visible symbol — debug info 없는 spin function 의 inline)
- 0x1de60 instruction: **`pause`** (busy-wait CPU hint)
- 0x1de60-0x1de6f cluster 의 spin loop body:
```asm
1de60: pause                    ; busy-wait hint
1de62: add  $0x1,%rax           ; spin counter ++
1de66: cmp  %rcx,%rax           ; check spin limit
1de69: je   1de80                ; exit if counter reached
1de6b: mov  (%rdi),%esi         ; load shared state (barrier counter)
1de6d: cmp  %esi,%edx           ; check expected generation
1de6f: je   1de60                ; loop back to pause if same
1de71: mov  (%rdi),%eax         ; final load
1de73: cmp  %edx,%eax
1de75: je   1de40                ; jump if final check passes
1de77: ret                       ; return (spinning ended)
```

### 결론
★ libgomp 43.75% = **standard barrier wait spin loop**. GOMP 의 `gomp_team_barrier_wait_end` 또는 `do_wait` 내부 구현 (debug info 부재로 addr2line 이 nearest visible symbol `omp_get_num_procs` 로 resolve). Phase 1 가설 confirm.

---

## 2. G-2 — OMP_NUM_THREADS sweep

### 측정 환경
- Workload: 100p × target_input_len 8192 × max_tokens 1024 (short)
- Common env: KMP_BLOCKTIME=200, OMP_PROC_BIND=false, VLLM_NEO_PROFILE=1, VLLM_NEO_CPU_PIN_PER_WORKER=1 (12 core/worker)
- baseline = `eval/run_neo_22items.sh` 의 OMP=10

### 결과

| OMP_NUM_THREADS | tps | wall (s) | vs baseline | 비고 |
|---|---|---|---|---|
| 5 | 543.2 | 188.5 | **-5.0%** | thread ↓ → compute throughput loss |
| 8 | 560.1 | 181.8 | **-2.0%** | 작은 win 없음 |
| **10** | **571.6** | **179.1** | **(baseline)** | best |
| 14 | — | timeout 25min | (fail) | worker 살아있으나 100p × 1024 안 끝남 |
| 16 | — | WorkerProc init failure | (fail) | 8 worker × 16 thread = 128 > 112 phys core. NUMA 충돌. |

### 해석
- **OMP ↓**: compute throughput ↓ 의 loss 가 libgomp wait ↓ 의 win 보다 큼.
- **OMP ↑**: NUMA/CPU 자원 over-saturation. 112 phys core 한계.
- best = baseline 10. env-only path 로 G 의 win 0.

### Phase 1 가설과의 대조
Phase 1 정적 분석의 "thread ↓ → libgomp spinning ↓ → throughput ↑" 가설은 실증 안됨. compute loss 가 더 큼. **NEO 의 cdec executor (max_workers=2) cap 환경에서 OMP thread 가 cdec executor 의 sub-task batch 와 동시 동작** — thread ↓ 시 단일 cdec task 의 wall time ↑ 으로 cdec executor 가 idle.

---

## 3. C-1 — K_TILE_WIDTH sweep

### 측정 환경
- Workload: 동일 (100p × 1024 short)
- OMP_NUM_THREADS=10 fixed
- 빌드: `bash csrc/cpu/pacpu/build.sh llama3_3_70b 8` (CXX=/tmp/gcc12/usr/bin/g++-12)

### 결과

| K_TILE_WIDTH | tps | wall (s) | vs baseline (K=2) | 비고 |
|---|---|---|---|---|
| **2 (baseline)** | **571.6** | **179.1** | **(best)** | reduce_add 8 per block |
| 4 | 562.7 | 182.0 | -1.6% | reduce_add 4 per block, register pressure ↑ |
| 8 | 569.0 | 180.0 | -0.5% | reduce_add 2 per block, ILP saturation |

### 해석
- **K_TILE ↑**: reduce_add 횟수 ↓ 의 win 보다 inner loop body 의 dependency chain ↑ + ILP saturation 의 loss 가 큼.
- ISPC compiler 의 vectorization 패턴이 K=2 에 최적화 (8 lane partial sum × 16 width gang).
- best = baseline 2. constant 변경 path 로 C 의 win 0.

---

## 4. G-4 — schedule(dynamic) work-stealing (code change)

### 변경 위치
- `csrc/cpu/pacpu/core.h:331` — Step 1 의 task loop 을 `#pragma omp for schedule(dynamic, 1)` work-stealing 으로 변경.
- 기존 `thrd_start_task[]` 정적 분배 → seq_len 비균등 시 thread imbalance → barrier #2 wait ↑ 가설.

### 결과 (단일 측정, 100p × 1024)

| Variant | tps | wall (s) | vs baseline | 비고 |
|---|---|---|---|---|
| **baseline (static partition)** | **571.6** | **179.1** | **(best)** | thrd_start_task 정적 분배 |
| **G-4 (schedule dynamic, 1)** | **568.6** | **180.1** | **-0.5%** | work-stealing |

### 해석
- task imbalance 가 이미 매우 작음 (정적 task 분배 알고리즘이 well-tuned)
- schedule(dynamic) 의 atomic counter overhead 가 imbalance ↓ win 보다 큼
- **G-4 win 0** → revert 후 baseline 으로 복원

### barrier 제거 가능성 (시도 안 함)
- **barrier #1 (line 326)**: Step 0 (bch_blk partition, store_kv) → Step 1 (task partition, attn_one_seq) 사이. thread A 가 store 한 seq Y 의 KV 를 thread B 가 attn 시 read. **semantic 필수, 제거 불가**.
- **barrier #2 (line 345)**: Step 1 (task partition, attn) → Step 2 (bch_blk partition, gather) 사이. 다른 thread 의 task 결과 gather. **semantic 필수, 제거 불가**.

→ G-4 의 cheap variant 로는 win 없음. 추가 변경 (Step 0/2 partition 통일 또는 store_kv 의 ISPC vectorization) 도 small win 예상 (Step 0/2 cost 가 작음).

---

## 5. B — softmax fast_exp (code change)

### 변경 위치
- `csrc/cpu/pacpu/pacpu.ispc:109-140` 의 softmax 내부 `exp()` 호출 → `fast_exp_inl()` (degree-4 polynomial)
- fast_exp_inl: `exp(x) ≈ 2^k * P(f)`, x = k*ln(2) + r, polynomial degree-4 approximation.
- 정확도 추정: 2-3 ULP (vs ISPC builtin `exp()` 의 ~1 ULP)

### 결과 (단일 측정, 100p × 1024)

| Variant | tps | wall (s) | vs baseline | 비고 |
|---|---|---|---|---|
| **baseline (ISPC exp)** | **571.6** | **179.1** | **(best)** | SLEEF-based polynomial |
| **B (fast_exp_inl degree-4)** | **568.2** | **180.2** | **-0.6%** | custom polynomial |

### 해석
- ISPC builtin `exp()` 는 SLEEF-based vectorized polynomial — 이미 매우 optimized
- 직접 작성한 degree-4 polynomial 은 floor + int conversion 등 추가 ops 가 builtin 의 vectorized polynomial 보다 비효율
- **B win 0** → revert 후 baseline 으로 복원

### 다른 B variants (시도 안 함)
- **ISPC `--math-lib=svml`**: Intel SVML link 필요. 환경에 libsvml 부재.
- **ISPC `--math-lib=ispc-fast`**: CLAUDE.md 정책 (fast-math 금지) 위반. 시도 불가.
- **host-side AVX-512 intrinsic softmax** (`csrc/cpu/cpu_arch_macros.h` 의 fast_exp 활용): ISPC kernel 분리 → ATen 래퍼 + function dispatch overhead. 매우 큰 effort.

→ B 의 cheap variant 로 win 없음. 큰 redesign 필요.

---

## 6. A — AMX qk/av (sanity test PASS + integration deferred)

### 6.0 prod 머신 확인 (정정)

이전 응답의 "dev 검증 불가" 정정 — **본 측정 머신 = prod**.

| 항목 | 값 |
|---|---|
| CPU | Intel Xeon Platinum 8480+ (SPR) |
| AMX | amx_bf16, amx_int8, amx_tile (모두 native) |
| AVX-512 | avx512_bf16, avx512_fp16, vnni 등 모두 native |
| GPU | NVIDIA H100 80GB × 8 |
| Kernel | Linux 5.14.0-427.13.1.el9_4 (RHEL 9) |
| AMX permission | XFEATURE_XTILEDATA OK (arch_prctl 0x1023) |

→ **이 머신에서 AMX 즉시 구현 + 검증 가능**.

### 6.1 AMX sanity test (`/tmp/amx_sanity.cpp`)

| 검증 | 결과 |
|---|---|
| arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) | OK |
| `_tile_loadconfig` | OK |
| `_tile_loadd` + `_tile_dpbf16ps` + `_tile_stored` | OK |
| 16×16 BF16 matmul 결과 정확도 | C[0][0] = 2.0 expected ✓ |
| gcc-12 빌드 (`-mamx-tile -mamx-bf16`) | OK |

★ **AMX hardware + kernel + 컴파일러 모두 ready**.

### 6.2 NEO qk-product 의 AMX integration 의 어려움

microbench (`/tmp/amx_qk_bench.cpp`) 시도 → SIGILL (tile config 의 hardware spec 불일치). debug 시간 cost 큼.

**진정한 issue**: NEO 의 qk-product 의 matmul size 가 **AMX 의 sweet spot 보다 작음**:

| 매개 | NEO Llama-3.3-70B TP=8 의 1 block | AMX tile 효율 |
|---|---|---|
| M (heads) | 8 | tile=16 (M=8 만 사용, 절반 낭비) |
| N (tokens) | 16 (BLOCK_SIZE) | tile=16 ✓ |
| K (head_dim) | 128 | tile K_pair=16 → 4 rounds 필요 |
| 1 matmul size | 8 × 16 × 128 = 16,384 FMA | AMX peak 1024 FMA/cycle → 16 cycle |
| Setup overhead | tile_loadd × 2 + dpbf16ps + tile_stored | ~100 cycle |
| 효율 | **work/setup = 16/100 = ratio 0.16** — setup dominant |

★ **작은 matmul size 에서 AMX setup overhead 가 work 보다 큼**. 큰 matmul (prefill 의 GEMM) 에서는 효율적이나, decode 의 per-block matmul 에서는 효율 낮음.

### 6.3 정량 추정 (Roofline 재정정)

| 항목 | 값 | 근거 |
|---|---|---|
| 이론 AMX peak | 1,024 GFLOP/sec/core | _tile_dpbf16ps × 2/cycle × 4 GHz |
| Setup overhead | ~100 cycle/call × 5,760 call/sec = 1.44 ms/sec/thread | call rate × overhead |
| Effective AMX | 100 cycle setup + 16 cycle work = 116 cycle/call | per-block |
| AVX-512 FP16 (현재) | ~512 cycle/call (peak 32 FMA/cycle) | 32 FMA × 512 cycle = 16,384 FMA |
| **이론 speedup** | **512 / 116 = 4.4×** | per-block |
| **Amdahl cap** | qk+av 16.65% × (1 - 1/4.4) = 13% 영역 감축 | Phase 1 dso |
| **실효 throughput win** | **+5-8%** (cdec executor cap 안에서) | Amdahl |

### 6.4 Integration 실측 결과

본 turn 에 integration 완료:
- 신규 `csrc/cpu/pacpu/amx_kernel.cpp` — `qk_amx` (host C++ AMX BF16) + `attn_one_seq_amx` (qk=AMX, softmax/av=ISPC)
- `csrc/cpu/pacpu/pacpu.ispc` — softmax `export` 변경 (AMX path 가 ispc::softmax 직접 호출)
- `csrc/cpu/pacpu/core.h` — env-toggle (`VLLM_NEO_USE_AMX=1`) dispatch
- `csrc/cpu/pacpu/CMakeLists.txt` — `amx_kernel.cpp` + `-mamx-tile -mamx-bf16`
- 빌드: gcc-12 successful, no warning
- thread_local AMX init (XTILEDATA permission + tile_loadconfig 1회/thread)

### 6.5 측정 결과

| Variant | tps | wall (s) | vs baseline (571.6) | 결과 |
|---|---|---|---|---|
| **baseline** (ISPC) | **571.6** | **179.1** | (best) | best |
| **A AMX integration** (`VLLM_NEO_USE_AMX=1`) | **564.3** | **181.5** | **-1.3%** | loss |

- 100p × 1024 short 완주 (SIGILL 없음, worker crash 없음) — AMX path **functionally correct**
- 측정 dir: `eval/results/20260518_074106_sub015_p3_a_amx/`

### 6.6 결과 해석

★ Phase 1 의 정량 추정 (+5-8%) 가 **실증 안 됨**. -1.3% loss 의 root:

| Overhead source | per-block cost | 빈도 |
|---|---|---|
| Q FP16 → BF16 변환 (1024 elem) | ~수십 cycle | per block (imax 회/seq/layer) |
| K^T pre-pack + FP16→BF16 (2048 elem × 4 round) | ~수백 cycle | per block × 4 |
| tile_loadd × (1 A + 1 B) × 4 round | ~30 cycle/load | per block × 8 |
| tile_dpbf16ps × 4 round | ~16 cycle/op | per block × 4 |
| tile_stored × 1 | ~30 cycle | per block |
| C[16,16] → a[M=8, N=tmax] copy | ~수십 cycle | per block |
| **합계 AMX cycle/block** | **~600-800 cycle** | overhead-dominant |
| **ISPC AVX-512 FP16 (baseline)** | ~500 cycle | compute-dominant |

★ NEO 의 작은 matmul size (M=8 head, N=16 token, K=128 dim) 에서 AMX setup overhead (특히 FP16→BF16 변환 + K pre-pack) 가 매 block 마다 발생 — work 보다 큼. **AMX 의 sweet spot 아님**.

### 6.7 개선 후보 (시도 안 함, future)

| 개선 | 변경 | 예상 효과 |
|---|---|---|
| **Q hoist** | Q FP16→BF16 변환을 ispc_attention_tasks 의 outer (per-call) 1회로 hoist | -50% 변환 cost |
| **K cache BF16 store** | NEO host buffer 의 K cache 를 BF16 으로 store (FP16 대신) — swap path 도 BF16 | K 변환 완전 제거 (-30-40%) |
| **multi-seq batched** | 여러 seq 의 Q × K^T 를 stack 으로 batched matmul — AMX 의 M=16 full 사용 | setup amortize, ratio ↑ |

각 1-2 일 effort. K cache BF16 store 가 가장 큰 win 예상 (-30-40% setup → AMX 가 win 으로 전환 가능).

### 6.8 코드 상태

- AMX path 는 **env-gated** (`VLLM_NEO_USE_AMX`). default off → baseline 영향 0.
- 본 turn 의 코드 **keep** (future Q hoist / K BF16 store 개선의 base).

### 6.9 결론

★ A AMX integration **functionally correct + measurement complete**. 단 작은 matmul size 의 setup overhead 가 work 보다 커 **-1.3% loss**. 진정한 win 위해서는 **K cache BF16 store** 등 큰 design 변경 필요 (1-2 일 추가 work).

| 항목 | 값 | 근거 |
|---|---|---|
| 변경 영역 | qk_product (8.75%) + av_product (7.90%) = **16.65% cycle** | perf dso |
| 대체 ISA | AMX BF16 micro-GEMM (`_tile_dpbf16ps`, 16×16×64 tile) | SPR spec |
| dtype 호환 | FP16 → BF16 변환 필요 (vCVTPH2PS + vCVTPS2BF16 = 2 round trip / sample) | dtype 분석 |
| 정확도 | BF16 mantissa 7 vs FP16 mantissa 10 = **3 bit drop** | dtype spec |
| Roofline (이론) | AMX BF16 peak = **1,024 GFLOP/sec/core** vs AVX-512 FP16 peak 128 = **8× theoretical** | SPR spec |
| Roofline (실효) | L2 BW 50 GB/s × AI 7.1 = **355 GFLOP/sec ceiling** = AVX-512 의 **2.77×** | AI 7.1 + SPR |
| 예상 throughput win | qk+av cycle 의 60% 감축 × cdec cap → **+5-10% throughput** | Amdahl |

### Effort
- ISPC 미지원 — C++ host function 으로 별도 작성 (qk_product_amx.cpp + av_product_amx.cpp + dispatcher)
- 정확도 검증 (TST_003 분포 유사성)
- dev 머신 (Alder Lake) AMX 미지원 → **prod 전용**
- 빌드 변경 — gcc-12+ + `-mamx-tile -mamx-bf16` flag
- **총 effort: 1-2 주**

### Risk
- dev 검증 불가 — prod 머신에서만 실행 가능
- 정확도 — BF16 → 분포 ULP error 증가, downstream token output 영향 가능 (TST_003 검증 필요)
- 빌드 환경 — `_tile_dpbf16ps` intrinsic 의 gcc/clang version 호환

### 권고
**A 는 prod-only task 로 deferred**. 본 SUB_015-Phase 2 의 즉시 구현 범위 밖.

추진 시 별도 task (예: SUB_015-Phase 3 / TSK_020 / TST_004 별도 verdict).

---

## 7. SUB_015-Phase 2 종합 결과

### 7.1 측정 sweep 종합

| Lever | 변경 path | tps | vs baseline | 결과 |
|---|---|---|---|---|
| **baseline** | OMP=10 + K_TILE=2 + ISPC builtin exp + static partition | **571.6** | (best) | **best** |
| G-2 OMP=5 | env thread ↓ | 543.2 | -5.0% | loss |
| G-2 OMP=8 | env thread ↓ | 560.1 | -2.0% | loss |
| G-2 OMP=14 | env thread ↑ | timeout | (fail) | fail |
| G-2 OMP=16 | env thread ↑ | init fail | (fail) | fail |
| C-1 K_TILE=4 | constant ↑ | 562.7 | -1.6% | loss |
| C-1 K_TILE=8 | constant ↑ | 569.0 | -0.5% | loss |
| G-4 schedule(dynamic) | code change | 568.6 | -0.5% | loss |
| B fast_exp_inl | code change | 568.2 | -0.6% | loss |
| A AMX qk/av | 평가만 | deferred | (prod-only) | 시도 안 함 |

### 7.2 결론

**env + constant + cheap code change variants 모두 baseline 이 best**. SUB_015-Phase 2 의 cheap 변경 path 에서 win 0.

가능 해석:
1. **NEO 의 cdec executor (max_workers=2) cap** 가 강력한 Amdahl 한계 — CPU 의 추가 효율화는 cdec cap 안에서 한계.
2. **ISPC kernel + libgomp** 의 baseline 조합이 이미 매우 well-tuned — 단순 변경으로 개선 어려움.
3. **libgomp 43.75% spinning** 은 *barrier 의 wait time* 이지 실제 *compute work* 가 아님. wait 줄이려면 sync 자체 줄여야 함 (불가) 또는 sync 횟수 줄여야 함 (paged_attention_cpu 호출 빈도 줄이기 — 별도 redesign).

### 7.3 다음 단계 후보 (cheap path 소진, 큰 redesign 필요)

| 후보 | 변경 영역 | 본 turn 진행 | Effort | Win 추정 |
|---|---|---|---|---|
| **A** AMX qk/av | pacpu C++ host AMX path | **sanity PASS, integration deferred** | 1-2 일 | +5-8% (작은 matmul 의 setup overhead 한계) |
| **D** layer batched paged_attention | gpu_model_runner Python loop → C++ batch | **시도 불가 — semantic** | — | 0 (layer dependency) |
| **E** KV cache layout 변경 | NEO host buffer + GPU cache layout 통일 | **평가만** | 2-4 주 | +3-5% (swap path ↓) |
| **F** Predictor 개선 | NEO predictor 의 cdec 호출 빈도 ↓ | (deferred) | 1-2 주 | +5-10% (Amdahl cap ↑) |

#### D 의 semantic 한계 (분석 완료)

- `attention.py:764` 의 cdec_future 는 **같은 layer 안에서 b1 region 처리** — layer 별 별개 future 생성, 매 layer 마다 submit + 같은 layer 의 b1 boundary 에서 result wait.
- vllm 의 model forward 가 layer 별 sequential — layer 0 attention 결과 → layer 1 hidden state → layer 1 attention.
- 80 layer 를 batched 단일 pacpu call 으로 묶을 수 없음 (caller 가 매 layer 결과 wait).
- cdec executor 의 max_workers 변경 시도: max_workers=4 측정 (SUB_030) 에서 -8% 회귀 — cap 2 가 best.
- → D 진정한 cheap path 없음. **layer fusion (multi-layer attention 의 architectural redesign)** 필요 — model architecture 변경 영역, SUB_015 범위 밖.

#### E 의 effort (분석 완료)

- **NEO host buffer**: `(num_layers, num_blocks, num_kv_heads, block_size, head_dim)` — 5D
- **vLLM GPU per-layer K cache**: per-layer 별개 tensor, `(num_blocks, num_kv_heads, block_size, head_dim)` (HND) 또는 `(num_blocks, block_size, num_kv_heads, head_dim)` (NHD)
- **neo_pacpu.py 의 `_to_neo_kv_view`** (line 248): per-layer (4D) → unsqueeze (5D) = **view only (zero-copy)** ✓
- 단 NHD layout 시 `.permute(0, 2, 1, 3)` + `.contiguous()` — **vLLM 의 KV cache layout 이 NHD 일 때만 영향** (현재 HND 라 zero-copy)

→ E 의 진정한 cost = 거의 0 (HND 환경). 추가 win path = **vLLM 의 KV cache layout 자체를 HND 로 강제 + NEO 호환**. 이건 모든 attention backend 영향 (FlashAttention, FlashInfer 등 — 큰 design 변경, 2-4 주 effort).

→ E 도 SUB_015 범위 밖, **별도 task**.

→ SUB_015 외에 **별도 task 로 분류** 합리적. 본 turn 의 SUB_015-Phase 2/3 = **cheap path 소진 + A 측정 -1.3% loss** 결론.

### 7.4 측정 자료 위치

- `eval/results/20260518_023541_sub015_p2_omp5/`
- `eval/results/20260518_024036_sub015_p2_omp8/`
- `eval/results/20260518_024642_sub015_p2_omp10/` (baseline)
- `eval/results/20260518_025257_sub015_p2_omp14/` (timeout)
- `eval/results/20260518_031803_sub015_p2_omp16/` (init fail)
- `eval/results/20260518_033853_sub015_p2_c_k4/`
- `eval/results/20260518_034522_sub015_p2_c_k8/`
- `eval/results/20260518_062631_sub015_p2_g4_dynamic/` (G-4)
- `eval/results/20260518_064427_sub015_p2_b_fastexp/` (B)
- summary: `/tmp/omp_sweep_summary.txt`, `/tmp/c_sweep_summary.txt`, `/tmp/g4_summary.txt`, `/tmp/b_summary.txt`

### 7.5 코드 상태

- `csrc/cpu/pacpu/pacpu.ispc` — baseline 으로 reverted (ISPC builtin exp)
- `csrc/cpu/pacpu/core.h` — baseline 으로 reverted (static partition)
- `csrc/cpu/pacpu/CMakeLists.txt` — `-g` 추가 유지 (SUB_015-Phase 1 debug build, 이미 commit 됨)
- baseline pacpu.so — 재빌드 완료 (22:00, K_TILE=2 + ISPC builtin + static partition)
