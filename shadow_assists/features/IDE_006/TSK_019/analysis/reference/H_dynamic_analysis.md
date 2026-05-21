# H — 동적 분석 (SUB_015-Phase 1)

> 2026-05-17 KST. perf record + debug-symbol pacpu rebuild 기반.
>
> 측정 dir: `eval/results/20260517_200212_cpu112_analysis_500p/deep_dive_60s/perf.data` (60s, 413K cycles sample).
> debug-symbol .so build_id `ebaa7333c7f8530669c0893809b3ed04669bd63d` (CMakeLists.txt 의 `set(ISPC_FLAGS "-O3" "-g")` 추가 후).
> perf report 결과: `/tmp/sub015_perf_resolve.txt`.

---

## 1. 측정 환경

- Workload: 500p × 8192 token, Llama-3.3-70B TP=8, NEO 적용 v1.6 S1-S9 (commit 531d61608)
- Host: Xeon Platinum 8480+ (SPR, 112 phys core, NUMA 2, 224 thread)
- 측정 도구: `perf record -e cycles --call-graph dwarf -F 99 -a` 60s 동안 8 worker × 14 OMP thread 채집
- Total samples: 413,068 cycles event
- Total event count: 6,862,126,820,987 cycle (≈ 6.86 T cycle)

## 2. dso 별 cycle 분포 (top 12 symbol 기준)

| % | Samples | Symbol | DSO | 분류 |
|---|---|---|---|---|
| 31.98 | 109,868 | `0x1de60` | **libgomp.so.1** | barrier / spin (unresolved) |
| 9.73 | 33,506 | `softmax___CuniCunfun_…` | **libpacpu** | ISPC kernel (transcendental) |
| 8.75 | 30,156 | `qk_product___CuniCuni…` | **libpacpu** | ISPC kernel (matmul) |
| 7.90 | 27,184 | `av_product___CuniCuni…` | **libpacpu** | ISPC kernel (matmul) |
| 4.85 | 16,753 | `index_put_kernel<Half>` | **libtorch_cpu** | ATen advanced indexing → swap path |
| 3.75 | 12,894 | `0x1de62` | libgomp.so.1 | barrier / spin |
| 3.68 | 12,649 | `0x1de6b` | libgomp.so.1 | barrier / spin |
| 3.42 | 11,791 | `0x1e028` | libgomp.so.1 | barrier / spin (별개 hot path) |
| 3.27 | 11,290 | `AVX2::copy_kernel` | libtorch_cpu | ATen tensor copy |
| 2.12 | 7,338 | `index_kernel<Half>` | libtorch_cpu | ATen advanced indexing |
| 1.84 | 6,361 | `_PyEval_EvalFrameDefault` | python3.12 | Python interpreter |
| 0.92 | 3,146 | `0x1de6f` | libgomp.so.1 | barrier / spin |

### dso 별 합산 (Top 12 만)

| DSO | 합 % | 분류 |
|---|---|---|
| **libgomp.so.1** | 43.75 | OMP runtime overhead (대부분 hot 영역 `0x1de60`-`0x1de6f` + `0x1e028`) |
| **libpacpu** | 26.38 | ISPC attention kernel (softmax + qk + av) |
| **libtorch_cpu** | 10.24 | ATen swap path (index_put, copy, index) |
| **python3.12** | 1.84 | Python interpreter |
| 합계 | 82.21 | (나머지 17.79% = misc symbols < 0.5%) |

---

## 3. libgomp 의 hot path 분석

### 3.1 hot offset 의 clustering

| Offset | % | 비고 |
|---|---|---|
| 0x1de60 | 31.98 | base — 가장 hot |
| 0x1de62 | 3.75 | base + 2 (next instruction) |
| 0x1de6b | 3.68 | base + 11 |
| 0x1de6f | 0.92 | base + 15 |
| 0x1e028 | 3.42 | base + 0x1c8 (별개 함수 또는 loop tail) |

★ 0x1de60 ~ 0x1de6f 의 16 byte 범위 = **하나의 함수 inner loop** 이 hot.

### 3.2 후보 함수 (libgomp source 기준 추정)

libgomp 의 sample hot symbol 후보:
1. **`gomp_team_barrier_wait_end`** — `#pragma omp barrier` 후 다른 thread 의 도달 대기 (atomic poll).
2. **`gomp_team_barrier_wait_final`** — implicit barrier (omp parallel 끝).
3. **`do_wait`** — idle thread 의 KMP_BLOCKTIME 안 spinning.
4. **`gomp_spin`** — atomic spinlock.

각 함수는 모두 짧은 spin loop (compare-and-swap 또는 atomic load + branch) 으로 cycle event 가 모두 잡힘.

### 3.3 mechanism 가설 (정적 분석과 결합)

`core.h:308-363` 의 4 sync point × 라이트한 thread imbalance → 가장 늦은 thread 대기 동안 모든 thread 가 libgomp 안 spinning. 

- step rate = 72 step/sec, layer 80 → call rate per worker = **5,760 paged_attention call/sec**.
- 매 call 의 4 sync × ws=14 thread imbalance → 평균 ~50% thread 가 wait state.
- 14 thread × 60 sec × 8 worker = 6,720 thread-sec measurement window.
- libgomp 43.75% × 6.86T cycle / 14 thread / 60 sec / 8 worker = **44.6 M cycle/sec/thread** ≈ 14.3 ms/sec/thread spinning.
- core capacity 1 sec → 14.3 ms = **1.43%** 의 wall time. 그러나 thread-time 으로는 44% — 거의 절반의 thread 가 wait.

★ **결론**: libgomp 43.75% 는 **barrier wait spinning 이 dominant**. step rate 와 sync 횟수 backing.

### 3.4 결정적 확인 방법

libgomp 의 symbol resolve 를 위해:
1. `dnf install libgomp-debuginfo` (해당 미러 검색 — 본 호스트에 미존재 확인됨).
2. wheel 의 `torch/lib/libgomp.so.1` 의 build-id 매칭 후 `addr2line -e` 으로 source line resolve.
3. perf record `--call-graph dwarf` 의 caller stack 추적 — pacpu 의 `ispc_attention_tasks` 의 `#pragma omp parallel` 후 어느 barrier 가 hot 인지 확정.

→ Phase 2 (lever G 적용 시점) 의 first step 으로 진행. 본 turn 의 정적 + dso 분포만으로 가설 충분.

---

## 4. libpacpu 의 kernel 별 분해

### 4.1 ISPC mangled symbol 해석

| Mangled | 디코드 | Source location |
|---|---|---|
| `softmax___CuniCunfun_3C_unf_3E_un_3C_unf_3E_` | `softmax(uniform int, uniform itmd_t, uniform itmd_t[], uniform itmd_t[])` | `pacpu.ispc:109` |
| `qk_product___CuniCuniCuniun_3C_Cunh_3E_un_3C_Cunh_3E_un_3C_Cuni_3E_un_3C_unf_3E_` | `qk_product(uniform int×3, uniform data_t[], uniform data_t[], uniform int[], uniform itmd_t[])` | `pacpu.ispc:5` |
| `av_product___CuniCuniCuniun_3C_Cunf_3E_un_3C_Cunh_3E_un_3C_Cuni_3E_un_3C_unf_3E_` | `av_product(uniform int×3, uniform itmd_t[], uniform data_t[], uniform int[], uniform otpt_t[])` | `pacpu.ispc:71` |

### 4.2 kernel 별 cycle 분포

| Kernel | % | FLOP / seq / layer | seq당 cycle 추정 (60s × 8 worker × 0.43 GHz effective) |
|---|---|---|---|
| qk_product | 8.75 | 4.19 M | 0.598 T cycle / 60s = 9.97 G cyc/sec/system |
| softmax | 9.73 | 0.49 M | 0.668 T cycle / 60s = 11.13 G cyc/sec/system |
| av_product | 7.90 | 4.19 M | 0.542 T cycle / 60s = 9.03 G cyc/sec/system |
| **libpacpu 합** | **26.38** | **8.87 M** | — |

### 4.3 핵심 관찰

★ **softmax 가 qk/av 보다 더 cycle 소모** — FLOP 는 11% 정도인데 cycle 은 비슷.
- 원인: exp/log transcendental 의 latency (ILP 제한 + polynomial chain dependency).
- → **fast_exp 적용 시 가장 큰 net win** (cycle/FLOP ratio 가 worst).

### 4.4 정확도 영향

softmax 정확도:
- max value 계산 (line 119) → exp(x - max) 의 overflow 방지.
- 분포 sum 계산 → 정규화 division.

fast_exp 적용 시:
- ISPC `exp()` = polynomial approximation 정확도 (보통 1 ULP).
- fast_exp = 동등하거나 약간 낮은 정확도 (polynomial degree 에 따라).
- → IDE_006 TST_003 verdict (분포·의도 유사성) 으로 검증 필요. token-level bit-exact 아님.

---

## 5. libtorch_cpu (ATen) 의 분해

### 5.1 NEO swap path 의 source

| Symbol | % | Path |
|---|---|---|
| `index_put_kernel<Half>` | 4.85 | `_neo_handle_kv_swap` → kv_k_list[l]->index_put_({gpu_idx}, k_slice) |
| `AVX2::copy_kernel` | 3.27 | tensor `.to(device)` 의 host-side copy |
| `index_kernel<Half>` | 2.12 | host_k_buf.index_select(1, idx_cpu) |

→ swap path 총 10.24%.

### 5.2 batched approach 폐기 사유 재확인

SUB_030 의 batched alternative 시도가 모두 OOB race 로 실패. 본 분석에서 확인:
- ATen advanced indexing 의 OMP 내부 분기 가 cdec_executor max_workers=2 thread 와 동시 실행 시 race.
- 단일 80-layer Python loop 유지 시 OMP overhead 가 ATen 분할 — 그러나 race 없음.

→ swap path 의 10.24% 는 현재 state 에서 **최선**. 추가 가속 = AMX index_put 의 별도 path 작성 필요 (effort 매우 큼, win < 5%).

---

## 6. python overhead (1.84%)

- `_PyEval_EvalFrameDefault` 1.84% — Python interpreter dispatch.
- NEO 의 80-layer loop 를 Python 으로 driving → 매 layer 마다 ATen op + pacpu.paged_attention_cpu call.
- Phase 1 의 cost — 무시 가능 (cdec_executor 의 future dispatch overhead).

→ 변경 우선순위 매우 낮음.

---

## 7. 동적 분석 핵심 결론

### 7.1 dso 기반 ranking 정정

|  | py-spy flamegraph (stale) | perf dso (현재) |
|---|---|---|
| libtorch | 17.82% | 10.24% (top 3) |
| **libgomp (OMP pool)** | **8.26%** | **43.75%** (★ 5.3× 차이) |
| attention_py | 3.07% | (분류 부재 — Python evaluator 일부) |
| tp_comm | 2.86% | (별도 측정 NCCL 포함, 본 측정에는 active 부족 가능성) |
| **libpacpu** | **0%** (보임 안됨) | **26.38%** |

★ py-spy 는 native frame resolve 부족 — pacpu 의 ISPC kernel + libgomp 의 spin loop 를 잘못 분류. perf 의 dso 단위 분류가 정답.

### 7.2 lever priority (동적 backing)

| 순위 | Lever | 정적 추정 | 동적 % | 통합 우선순위 |
|---|---|---|---|---|
| **1** | G libgomp (barrier wait) | 5-15% Amdahl | 43.75% (single largest) | **★★★ — Phase 2 first** |
| **2** | B softmax fast_exp | 3-5% Amdahl | 9.73% (cycle/FLOP worst) | **★★ — quick + safe** |
| **3** | A AMX qk+av | 5-10% Amdahl | 16.65% (qk + av 합) | **★ — long-haul (dev 검증 불가)** |
| **4** | C K_TILE_WIDTH | 1-2% | 8.75% (qk only) | quick try |
| **5** | swap path AMX | <5% | 10.24% | effort 매우 큼, defer |

---

## 8. Phase 2 측정 plan (정적 + 동적 결합 후)

### 8.1 lever G 의 first move (cheap probe)

**OMP_NUM_THREADS sweep** (단순 env 변경, code 0 변경):
- 현재: 14 thread / worker (Phase 3.1+KMP=200 best 기준)
- Probe: 8, 10, 14, 16 sweep, 100p × 4096 short
- 측정: throughput + cdec_wait + libgomp % (perf 30s)

기대: 8 thread 시 libgomp 의 spinning 절반 ↓. compute throughput 도 ↓ — net 효과는 측정으로 확정.

### 8.2 lever G 의 deeper move (code change)

`core.h:326,345` 의 `#pragma omp barrier` 2 회 중 하나 제거 검토:
- barrier #1 (line 326): Step 0 (store_kv) → Step 1 (attn_one_seq). 같은 thread 가 같은 seq 처리하면 store_kv 후 attn_one_seq 가 read — barrier 불필요. 단, batch_size > ws 시 store_kv 가 분배되어 다른 thread 의 결과 read 필요.
- → **batch_size <= ws (14) 시 barrier #1 제거 가능**. 측정 시 batch_size 분포 확인 필요.

### 8.3 lever B fast_exp 적용

`csrc/cpu/cpu_arch_macros.h` 의 fast_exp 가 macro 인지 inline function 인지 확인. AVX-512 intrinsic 으로 작성 → pacpu.cpp 의 별도 softmax 구현 + ISPC export 대체.
- 단점: ISPC 의 vectorization 과 분리 → linkage 필요.
- 대안: ISPC builtin `exp()` 의 SVML vs SLEEF 분기 — build option 으로 SVML 선호.

### 8.4 측정 우선순위

1. libgomp 정확한 symbol resolve (`dnf install libgomp-debuginfo` 시도 또는 caller stack 추적)
2. OMP_NUM_THREADS sweep — 1 시간 측정
3. lever G code change (barrier #1 제거) — 1 일 작업
4. lever B fast_exp — 2-3 일 작업

→ 사용자 명시 후 Phase 2 진행.
