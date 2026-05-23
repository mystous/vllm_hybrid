# SUB_033 — B3 FlashDecoding++ online softmax 측정 결과 (2026-05-21 KST)

> **parent**: TSK_019 / N 문서 영역 B B3 / O 분석 §7.2 ★★★ 첫 항목
> **plan**: [SUB_033 plan](../../planning/SUB_033_flashdec_plusplus_plan.md)
> **measurement**: HEAD `0d7dc0334`, 100p × 8192, gmu=0.85, A4 ON (numactl --localalloc) + B3 toggle.

---

## 1. 측정 결과

| test | levers | tps | wall (s) | crash |
|---|---|---:|---:|:-:|
| t1 | A4 ON, B3 OFF | 922.3 | 879.5 | 0 ✓ |
| t2 | **A4 ON, B3 ON** | **915.1** | **886.7** | 0 ✓ |
| t3 | A4 ON, B3 OFF (jitter check) | 923.0 | 878.8 | 0 ✓ |

**B3 ON vs OFF avg**: 915.1 vs (922.3+923.0)/2 = 922.65 → **Δ = -7.55 tps = -0.82%**

## 2. 결론 — B3 also noise / 미세 negative

가설 (메모리 traffic 1/3 절감 → +0.5~1.5%) 과 반대 방향. ±2% noise band 안이지만 **방향이 negative 라 default ON 으로 채택 불가**.

## 3. 원인 분석 — 왜 reverse 결과인가

### 3.1 추가 exp 호출이 메모리 절감을 압도

| 영역 | 기존 3-pass | online 2-pass | Δ |
|---|---|---|---|
| 메모리 read+write | seq_len × 3 | seq_len × 2 | -33% |
| exp 호출 | seq_len × 1 (Pass 2) | seq_len × 2 (Pass 1 의 rescaling + Pass 2) | **+100%** |
| foreach (h) barrier | 3 | 2 | -1 |

→ SPR 의 AVX-512 exp 가 매우 빠르지만 (VEXP2PS ~7 cycle), seq_len ≈ 8K 에서 **+8K 추가 exp = 8000 × 7 = 56K cycle ≈ 17.5 μs / 7 GHz / head × 64 head = 1.1 ms / seq**.
→ 메모리 절감은 seq_len × NUM_Q_HEADS × itmd_t = 8K × 64 × 4 = 2 MiB per pass. L2 (60 MiB/socket) 안에 들어가서 HW prefetcher 가 잡고 있는 영역이라 절감 효과 미미.

### 3.2 ISPC vectorize 깨졌을 가능성

기존 3-pass 는 각 pass 가 **순수 element-wise** (loop body 안에 data dependency 없음) → ISPC 가 outer-loop SIMD lane 으로 vectorize.

Online softmax 의 Pass 1:
```c
itmd_t new_max = max(amb[h], s);
asb[h] = asb[h] * exp(amb[h] - new_max) + exp(s - new_max);
amb[h] = new_max;
```
- `amb[h]` 가 loop carried dependency
- `exp` 가 매 iter 호출

→ ISPC outer-SIMD (NUM_Q_HEADS lane) 로는 vectorize 되지만, inner sequential 의 instruction-level parallelism 이 깨짐.

### 3.3 AMX path 에서 softmax 가 critical path 가 아닐 가능성

env-ON baseline 의 측정은 `VLLM_NEO_USE_AMX=1` → AMX path (`attn_one_seq_amx_bf16`).
- qk_amx_bf16 (AMX BF16 matmul) 가 dominant
- softmax 는 sub-step
- softmax 최적화의 wall-time impact 자체가 작음

이 가설이 맞다면, B3 의 ±0.82% 도 **noise 의 분산** 일 수 있음 (실제 영향 < 0.1%).

## 4. 갱신된 결론 — A-tier + B3 모두 noise

| Tier | Lever | 1-run 결과 | 3-run 검증 결과 |
|---|---|---|---|
| A1 | Intel libomp + tournament barrier | +0.4% | (검증 미진행 — 매트릭스 노이즈) |
| A2 | GOMP_SPINCOUNT + KMP_BLOCKTIME | -0.01% | (noise 명백) |
| A3 | core.h `__builtin_prefetch` | -1.1% | (noise 명백, HW prefetcher 잉여) |
| A4 | numactl --localalloc | +1.0% | **noise (SUB_032 3-run, -0.21%)** |
| A5 | FA3 max_num_splits=8 | -0.2% | (workload mismatch) |
| **B3** | **FlashDecoding++ online softmax** | **-0.82% (1-run, OFF vs ON)** | **noise/negative, default OFF 유지** |

→ **모든 lever 가 ±2% noise band 안** 으로 수렴. **본 코드 베이스의 NEO 구현은 측정 noise 영역에 깊이 들어와 있어 single-shot tuning lever 로는 짜낼 여지가 없음**.

## 5. 다음 turn 의 path 선택

### Option A — SUB_034 B1 OmniServe LSE async merge (★★★ 권고 #2)

- 변경 surface: `vllm/model_executor/layers/attention/attention.py` (async cdec depth ≥ 2 + race-safe LSE merge)
- effort: medium (1-2 일)
- 가설: ~5-10 ms wall 단축 (CPU sync wait 제거)
- **risk**: P4 (async cdec depth=1) 가 이미 OOB race 로 한번 unstable → race-safe 재설계 필요. 단 SUB_028 (OOB silent) 후 환경 안정.

### Option B — C1 layer-fusion (C-tier 시작)

- 변경 surface: `csrc/cpu/pacpu/core.h` 의 OMP team launch + cdec executor (cross-layer 묶음)
- effort: large (2-3 일)
- 가설: OMP fork/join 80회 → 40회 또는 더 적게 → libgomp 43.75% 직접 감소
- **risk**: layer 의존성 chain 으로 worker 간 진정한 병렬화 불가 가능성

### Option C — 측정 환경 자체 점검 (회의적 turn)

- 본 측정 system 의 noise floor (±2% / 1-run, ±0.18% / 3-run) 이 너무 큰지 점검
- workload 자체가 throughput-saturated 라 lever 효과가 묻히는지 점검 (100p × 8192 = 818,200 token 인데 baseline 932 tps × 880s = 820K → 거의 ceiling)
- **결정적 질문**: 본 baseline 이 이미 hardware ceiling 에 도달했다면, 더 짜낼 여지 자체가 없음

## 6. 권고

**Option C 부터 우선 (1-2 시간)** — 본 baseline 이 H100×8 + Llama-70B + 8K context 의 hardware ceiling 인지 sanity-check 후, 여유가 있다면 SUB_034 (B1) 진입. ceiling 도달 상태라면 **GPU 단일 측정 (vanilla path) 와 비교해서 NEO 의 net benefit 가 어디까지 가능한지 재정의** 필요.

## 7. raw 자료

| 항목 | 위치 |
|---|---|
| SUMMARY.tsv | `eval/results/20260521_231426_sub033_b3_softmax_100p/SUMMARY.tsv` |
| per-test 결과 (3 dirs) | `eval/results/20260521_231426_sub033_b3_softmax_100p/t1~t3/` |
| launcher | `/tmp/run_sub033_b3_softmax.sh` |
| stdout log | `/tmp/sub033_b3.log` |
| code 변경 | `csrc/cpu/pacpu/pacpu.ispc` (softmax_online), `csrc/cpu/pacpu/amx_kernel.cpp` (env-gated dispatch) |
| build | `csrc/cpu/pacpu/build/libpacpu-llama3_3_70b-tp8.so` (재빌드 ✓) |

## 8. 코드 처리 결정

**env-gated default OFF 유지** — production 영향 없음. `VLLM_NEO_SOFTMAX_ONLINE` 환경 변수로 opt-in. 향후 다른 workload (낮은 seq_len, 또는 ISPC path 만 사용 등) 에서 재시도 가치 있음.
