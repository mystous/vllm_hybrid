# H — SUB_015-Phase 1 최종 lever ranking + Phase 2 진행 plan

> 2026-05-17 KST. branch `feat/neo-amx-apply` HEAD `c698b971c`.
>
> 입력: `H_static_analysis.md` (pacpu source FLOPs/bytes + OMP pattern) + `H_dynamic_analysis.md` (perf dso 분포).
>
> 산출: Phase 2 에서 진행할 단일 lever + 측정 plan.

---

## 1. dso 별 confirmed share (perf 60s, 413K samples)

| Rank | DSO | Cycle % | 영역 | 가속 가능성 |
|---|---|---|---|---|
| 1 | **libgomp.so.1** | **43.75%** | OMP barrier wait spinning | Y (sync 패턴 변경 / thread 수 ↓) |
| 2 | **libpacpu** (softmax) | 9.73% | Transcendental (exp/log) | Y (fast_exp polynomial) |
| 3 | **libpacpu** (qk_product) | 8.75% | FP16 matmul | Y (AMX BF16) |
| 4 | **libpacpu** (av_product) | 7.90% | FP16 matmul | Y (AMX BF16) |
| 5 | **libtorch_cpu** (index_put) | 4.85% | NEO swap path | Defer (race 위험) |
| 6 | **libtorch_cpu** (AVX2 copy) | 3.27% | tensor.to() copy | Defer |
| 7 | **libtorch_cpu** (index) | 2.12% | host_buf.index_select | Defer |
| 8 | python3.12 | 1.84% | Python interpreter | Defer (overhead 작음) |
| — | 나머지 (< 0.5% each) | ~17.79% | misc | Defer |

★ **libgomp 43.75% = 가장 큰 single lever**. py-spy 의 stale 8.26% 가정 정정.

---

## 2. lever 의 정량 정리

### lever G — libgomp barrier wait

| 항목 | 값 | 근거 |
|---|---|---|
| Current cycle share | 43.75% | perf dso |
| Hot offsets | `0x1de60`-`0x1de6f` (16 byte cluster) + `0x1e028` (별개 loop) | perf symbol |
| Mechanism | `core.h:308` `omp parallel` + line 326,345 의 barrier × 2 + implicit fork/join = 4 sync per call | source 분석 |
| Call rate | 80 layer × 72 step/sec × 4 sync = 23,040 sync/sec/worker | step rate (2,157 tps / 30 batch) |
| Probe (code 0 변경) | OMP_NUM_THREADS sweep (8/10/14/16) | env 변경 |
| Probe (code 변경) | barrier #1 제거 (Step 0 → Step 1 사이) | core.h:326 |
| 이론 win | barrier wait 50% 감소 시 → libgomp 22% ↓ → throughput +12% (Amdahl) | 추정 |
| 실효 win (cdec cap 반영) | +5-10% throughput | cdec_executor max_workers=2 cap |
| Effort | 중 (env sweep 1 일 + code change 2-3 일) | 작업량 |
| 위험 | barrier 제거 시 race 가능 (batch_size > ws 시) | 정합 검증 필요 |

### lever B — softmax fast_exp

| 항목 | 값 | 근거 |
|---|---|---|
| Current cycle share | 9.73% | perf symbol (softmax) |
| Mechanism | ISPC exp() 의 polynomial latency (ILP 제한) | static 분석 |
| FLOP 추정 | 0.49 MFLOP / seq / layer (대부분 exp/log) | pacpu.ispc:109-140 분석 |
| 변경 | ISPC `exp()` → `csrc/cpu/cpu_arch_macros.h` 의 fast_exp 또는 SVML 강제 | 자산 재사용 |
| 이론 win | 2-4× kernel speedup → cycle 70% ↓ → throughput +6.8% | Amdahl |
| 실효 win | +3-5% throughput | cdec cap |
| Effort | 중 (intrinsic 작성 또는 SVML link option 추가) | 작업량 |
| 위험 | 정확도 영향 — IDE_006 TST_003 verdict (분포 유사성) 으로 검증 | constraint |

### lever A — AMX qk+av

| 항목 | 값 | 근거 |
|---|---|---|
| Current cycle share | 16.65% (qk 8.75 + av 7.90) | perf symbol |
| AI | 7.11 FLOP/byte | static 분석 |
| Roofline (SPR) | L2-bound 시 355 GFLOP/sec/core (현재 AVX-512 peak 128 의 2.77× ceiling) | SPR spec |
| 변경 | BF16 micro-GEMM tile (`csrc/cpu/micro_gemm/cpu_micro_gemm_amx.hpp` 재사용 후보) | 자산 |
| 변경 cost | FP16→BF16 변환 (vCVTPH2PS + vCVTPS2BF16) | dtype 변환 |
| 정확도 | BF16 mantissa 7 bit vs FP16 10 bit — 3 bit drop | constraint |
| 이론 win | 8× kernel speedup → cycle 87% ↓ → throughput +14.5% (qk+av) | Amdahl |
| 실효 win | +5-10% throughput | L2 BW ceiling + cdec cap |
| Effort | 고 (build 변경 + dtype 변환 + dev 검증 불가 + 정확도 검증) | 작업량 |
| 위험 | dev 머신 SPR 미지원 (Alder Lake) → prod-only 검증. 정확도 영향 검증 필요 | env 제약 |

### lever C — K_TILE_WIDTH ↑ (quick)

| 항목 | 값 | 근거 |
|---|---|---|
| Current cycle share | 8.75% (qk only) | perf symbol |
| 변경 | `pacpu.ispc:4` `#define K_TILE_WIDTH 2` → 4 또는 8 | constant 변경 |
| Effort | 저 (constant + rebuild + measure) | 작업량 |
| 이론 win | 1.05-1.15× qk speedup | reduce_add 횟수 ↓ |
| 실효 win | +1-2% throughput | |
| 위험 | 0 (수치 결과 동일 / register pressure 만 ↑) | |

---

## 3. Phase 2 진행 plan (단일 lever 선택)

### 3.1 권고: **lever G first** + lever C 병행

**근거**:
1. lever G 가 **single largest share** (43.75%) + relatively low effort.
2. lever C 는 **near-zero effort** (constant 변경) + zero accuracy risk.
3. lever B / A 는 정확도 검증 필요 — 시간 cost ↑.
4. lever G 의 OMP_NUM_THREADS sweep 은 **1 시간 측정** 으로 확정 가능 — 가장 cheap discovery.

### 3.2 측정 sequence (사용자 명시 후 진행)

| Step | 작업 | 측정 | 산출 |
|---|---|---|---|
| **G-1** | libgomp 정확한 symbol resolve | torch wheel 의 libgomp build-id 매칭 + addr2line 또는 caller stack 추적 | 0x1de60 = 정확한 함수명 |
| **G-2** | OMP_NUM_THREADS sweep | 8/10/14/16, 100p × 4096 short × 4 run = ~30 min total | throughput + libgomp % 그래프 |
| **G-3** | best thread count 로 500p × 8192 3-run avg | 30-40 min × 3 | net 효과 확정 |
| **C-1** | K_TILE_WIDTH=4 (8 도 검토) | constant 변경 + rebuild + 100p short | qk_product cycle % |
| **C-2** | best K_TILE_WIDTH 으로 500p × 8192 3-run avg | 30-40 min × 3 | net 효과 확정 |

→ 총 약 1 일 측정.

### 3.3 Phase 2 다음 단계 (lever G/C 후)

lever G/C 적용 후 throughput 결과 + cycle 재분포에 따라:
- libgomp 가 여전히 > 25% 면 추가 sync 변경 (barrier #1 제거 또는 `omp for schedule(dynamic)`)
- softmax 가 > 10% 이면 lever B 진행
- AMX 는 lever B 까지 마친 후 — **prod 머신 전용 작업** + 정확도 검증 동반.

---

## 4. D/E 문서 정정 필요 사항

### D_bottleneck_table.md
| 정정 | Stale | 정정 후 |
|---|---|---|
| libgomp share | 8.26% (py-spy) | 43.75% (perf dso) |
| libpacpu share | 0% (invisible) | 26.38% (softmax 9.73 + qk 8.75 + av 7.90) |
| pacpu 가 OMP 호출 안 함 가정 | "OMP pool 은 ATen index_kernel" | "OMP pool 은 pacpu `ispc_attention_tasks` 의 `omp parallel` + barrier × 2 + implicit fork/join" |
| 측정 source | py-spy native unwind | perf record `-e cycles -F 99` |

### E_amx_avx_applicability.md
| 정정 | Stale | 정정 후 |
|---|---|---|
| lever ranking | softmax > qk = av | libgomp > softmax > qk = av |
| libgomp 의 가속 가능성 | 명시 없음 | "OMP_NUM_THREADS sweep + barrier #1 제거 검토 — 가장 cheap lever" |
| AMX ranking | top priority | 3rd priority (B 다음, 정확도 검증 필요) |

→ Phase 2 진행 시점에 정식 반영. 본 H_* 3 파일이 source of truth.

---

## 5. 사용자 명시 대기 사항

다음 결정 사항이 **사용자 명시 후 진행**:

1. **Phase 2 진행 여부** — 본 분석 결과 충분한지?
2. **lever 선택** — 권고 (G first + C 병행) 채택 / 다른 우선순위 / lever 추가 (예: swap path AMX 도 포함) 선택?
3. **측정 launch** — G-1 libgomp symbol resolve 부터 시작 가능 여부.
4. **분석 문서 commit** — H_static_analysis.md / H_dynamic_analysis.md / H_phase1_final_levers.md + D/E 정정 commit 여부.
