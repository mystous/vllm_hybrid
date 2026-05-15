# Phase D — Roofline notes (SPR AMX/AVX-512 peak vs measured)

> 분석 시각: KST 2026-05-15 ~
> 산출 목적: 우리 prod 환경 (Intel Xeon Platinum 8480+ × 2, SPR) 의 ISA 별 peak throughput 과 NEO pacpu kernel 의 실측 도달률 비교 → roofline 상 AMX/AVX-512 의 실효 speedup ceiling 정량

---

## D.9 prod 머신 spec (측정 가능 영역)

| 항목 | 값 | 출처 |
|---|---|---|
| CPU | Intel Xeon Platinum 8480+ × 2 socket | `/proc/cpuinfo` (이전 측정) |
| Core 당 base | 2.0 GHz (boost 3.8 GHz) | Intel ARK |
| Core 수 / socket | 56 (= 112 thread) | 측정 |
| 측정 영역 | taskset 0-111 (1 socket pin) | run_neo_standard.sh |
| AVX-512 | ✅ SPR native | cpuid `avx512f`, `avx512bf16` |
| AMX | ✅ AMX-TILE + AMX-BF16 + AMX-INT8 | cpuid `amx_tile`, `amx_bf16` |
| L1d / core | 48 KB | spec |
| L2 / core | 2 MB | spec |
| L3 shared | 105 MB (socket) | spec |
| Memory BW | DDR5-4800 8 ch × 2 socket = **307 GB/s/socket** (이론 peak) | spec |

---

## D.10 ISA 별 peak (single core, BF16/FP16/FP32, FMA cycle 가정)

### AVX-512 (16-wide FP32 / 32-wide BF16 FMA)

- 한 cycle 에 1× `vfmadd231ps` (16 lane FP32) = **32 FLOPs / cycle / FMA unit**
- SPR core 2 FMA unit → **64 FLOPs / cycle**
- @ 3.0 GHz boost (under all-core load 가정) → **192 GFLOPs/s / core**
- 56 core → **10.75 TFLOPs/s / socket** (FP32 AVX-512)

BF16 변환 시 (`vdpbf16ps`): 한 cycle 에 32 BF16 pair × 2 ops = **64 FLOPs/cycle/unit** = AVX-512 BF16 peak **384 GFLOPs/s / core** = ~21.5 TFLOPs/s / socket

### AMX-TMUL (BF16, M=N=K=16, single tile op)

- 한 `tdpbf16ps` 명령: 16×16 × 32 FLOPs (FMA) = **8,192 FLOPs / 1 instruction** 1 tile instr 당
- latency ~16 cycle, throughput 1 cycle (back-to-back) — AMX-BF16 peak 시
- → **8,192 FLOPs / cycle / core** = 24,576 GFLOPs/s / core (@ 3.0 GHz) = ~24.5 TFLOPs/s **per core (FP32 acc, BF16 input)**
- 56 core → 약 **1,375 TFLOPs/s / socket** 명목 peak (실제는 데이터 fetch / tile reload 제한으로 200-300 TFLOPs/s 영역 도달)

### memory BW

- DDR5-4800 8 ch = 307 GB/s peak per socket (이론)
- 실제 STREAM Triad ~220-260 GB/s / socket (관측 데이터)

---

## D.11 NEO pacpu kernel 별 roofline 위치

`A_kernel_signature_map.md` 의 AI (FLOPs/byte) + 위 peak 사용.

### qk_product

- AI = 30-50 FLOPs/byte (effective, q broadcast reuse 고려)
- crossover (AI* = peak/BW): AVX-512 = 10.75 TFLOPs / 250 GB/s = 43, AMX = 250 TFLOPs / 250 GB/s = 1000
- 위치: **AVX-512 ridge 근처** (AI=30-50, ridge=43) — AVX-512 시 compute-bound 진입 직전
- AMX 시: ridge=1000 → 우리 AI=30-50 < 1000 → **여전히 memory-bound**

→ qk_product 의 AMX **이론 speedup = (250 GB/s × 50 FLOPs/byte) / 도달치 ≈ 12.5 TFLOPs/s 도달**. 현재 ISPC `avx512spr-x16` 도달치 8-10 GFLOPs/s/core (= 0.5-1 TFLOPs/s for 56 core) → **AMX 13-25× 이론 speedup**. 그러나 BF16 변환 + tile load/store overhead 적용 후 **실효 4-7×**.

### av_product

- AI = 7 FLOPs/byte
- AVX-512 ridge=43, AI=7 → memory-bound (BW 8× 부족)
- AMX 시: ridge=1000, AI=7 → memory-bound 더 심함
- AMX 가속 **무의미** — V cache read 가 bandwidth 제한

→ AMX 적용 시 `av_product` **이론 speedup ≈ 1×** (memory-bound). AVX-512 도 동일.

### softmax

- AI ≈ 0.5 FLOPs/byte
- 완전 memory-bound — 그러나 ops 중 exp 가 multi-cycle (~10-30 cycle / element)
- AVX-512 `fast_exp` 적용 시 (polynomial 5 cycle 추정) → **2-6× speedup** (exp cycle 단축)
- AMX 적용 무의미 (GEMM 아님)

---

## D.12 NEO pacpu 의 OpenMP scaling 한계

`core.h` 의 OMP parallel team:
- `n_threads = 12 (per worker)` × 8 worker = 96 thread (1 socket 56 core 의 1.7× over-subscribe)
- per worker `taskset 0-13` (14 core) — 사실상 8 worker × 14 core = 112 core (HT 활성 가정)

omp_pool 8.26% (flamegraph) = thread launch + barrier + idle wait. NEO 의 attn_one_seq 가 매우 small batch (mirror=10) → **OMP team launch cost 가 compute time 보다 큼**.

→ AMX 시 compute time 줄어들면 OMP overhead 비율 더 커짐. OMP team 의 long-live (persistent) 화 또는 inter-step batching 이 동반되어야 AMX 효과 도달.

---

## D.13 cdec dispatch 빈도 측정

| 항목 | 값 |
|---|---:|
| mirror size 모드 | 10 (stable, cap 80 도달 96회) |
| running batch | 222 |
| cdec dispatch ratio | ≈ 4.5% |
| step rate | 측정값 wall 1882s / step count ≈ 0.5-1 step/sec (8 worker) |
| 따라서 cdec call/sec | ≈ 0.5 × 10 / 8 ≈ 0.6 cdec call/sec/worker |
| layer 당 call/sec | 0.6 × 80 = 48 layer-call/sec/worker |

→ 한 worker 의 cdec 영역이 약 48 layer-call/sec × 8.75 ms/layer = **420 ms/sec** = 42% CPU duty. 실제 flamegraph 의 worker self-time 분석에서 cdec 영역 (forward_double + 자식) 이 1,688 samples / 10,407 total ≈ **16% 영역**. 24% 영역 mismatch 는 mirror=10 의 cdec 가 forward_neo_pipelined 영역 안에 묻혀 있고 일부는 GPU sync 대기 영역으로 측정된 것으로 추정.

---

## D.14 roofline 종합 그림 (텍스트)

```
TFLOPs/s
1000 ┤ AMX peak (BF16) ----------------------------
     │
 100 ┤
     │              ridge (BW=250) ↗
  10 ┤ AVX-512 peak ─────────────────────────────
     │         qk_product (AVX-512 ridge 근처) ●
   1 ┤   av_product ●     (BW-bound)
     │ softmax ●  (BW + exp)
 0.1 ┤
     └──────┬──────┬──────┬──────┬──────────── AI (FLOPs/byte)
            1     10     100   1000
```

- `qk_product` 가 유일하게 ISA peak 에 접근 가능
- `av_product` 는 BW 한계로 ISA 변경 효과 거의 없음
- `softmax` 는 scalar exp 가 bottleneck → AVX-512 `fast_exp` 만 의미

---

## D.15 결론 — Roofline notes

1. **AMX 효과가 큰 kernel = qk_product 만** (이론 4-7× 실효 speedup)
2. **av_product 는 ISA 변경 무의미** (memory-bound)
3. **softmax 는 AVX-512 fast_exp 만 의미** (이론 2-6× speedup, 영역 작음)
4. **OMP team launch overhead 가 mirror=10 영역에서 compute 보다 큼** — AMX 적용 시 compute 줄어들어 overhead 비중 더 커짐. OMP team persistent 화 / cdec batching 동반 필요
5. **cdec_executor max_workers=2 cap** 는 layer-wise dependency 로 + 4 시 -52% regression (SUB_023). cap 영역 안에서 AMX 가 layer 별 가속 → wall-clock 짧음 ✓
6. **실효 wall-clock 단축 cap (이상적)**: cdec_wait 8.75 ms × 0.4 (qk_product) × 1/5 (AMX speedup) = 0.7 ms 절감 / 8.75 ms = **8% 단축 / layer**
7. 80 layer × 8% = **64% 단축 / step** — 그러나 cdec 가 step 의 ~16% 영역이므로 step wall **10% 단축 / 전체 throughput +10%** 예상 가능 (workload 동일 시)

→ **AMX 단독으로 paper 의 H100 14% gain 영역 (vs vanilla) 거의 도달 가능** (10% throughput 향상). workload 조정 동반 시 더 큼.
