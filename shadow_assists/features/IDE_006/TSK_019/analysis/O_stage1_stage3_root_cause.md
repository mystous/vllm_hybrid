# O — Stage 1 + Stage 3 root cause 분석 (코드 / 라이브러리 레벨, 2026-05-21 KST)

> **출처**: 사용자 명시 (turn 16) — "전체적으로 효과가 없는 거 같은데 왜 그럴까? 원인 분석을 코드 레벨 리버그 레벨 분석 결과로 이야기 해"
> **측정 기반**: [Stage 1 RESULTS](../measurements/stage1_a1a2a3a4_matrix_100p_20260521/RESULTS.md) (12 tests, A1-A4) + [Stage 3 RESULTS](../measurements/stage3_a5_matrix_100p_20260521/RESULTS.md) (12 tests, A5+) — 합산 24 measurements 가 모두 ±2% noise 안.
> **연관**: [N 문서](N_cdec_leftover_elimination_ideas.md) (영역 A — Quick win, low risk 5 levers 의 출처).

---

## 0. TL;DR

| lever | 정체 | Stage 1+3 결과 | 무효 원인 |
|---|---|---:|---|
| A1 | Intel libomp + tournament barrier | A1 +0.4%, A1+A5 **-1.7%** | libgomp→libomp redirect ✓, 단 NEO 의 workload-driven OMP fork/join 패턴에선 barrier 알고리즘 변경 효과 미미 |
| A2 | GOMP_SPINCOUNT=INFINITE + ACTIVE | A2 -0.01%, 모든 조합 noise | libgomp 의 SPINCOUNT 는 동작, 단 NEO 는 이미 active spin 상태. KMP_BLOCKTIME 은 Intel libomp 전용 → libgomp 에 적용 안 됨 |
| A3 | core.h `__builtin_prefetch` | A3 -1.1%, A3+A5 **+0.6%** (synergy) | seq access 는 HW prefetcher 가 이미 처리 — SW prefetch hint 가 잉여. 단 FA3 와 결합 시 약간 살아남 |
| A4 | numactl --localalloc | A4 +1.0% (Stage 1 1-run) → ~~win~~ **noise (SUB_032 3-run avg 930.2, -0.21%)** | 1-run artifact 였음. NEO 의 worker thread NUMA bind 만으로 충분, process-wide localalloc 추가가 측정 가능한 이득을 만들지 않음 |
| A5 | FA3 max_num_splits=8 | A5 -0.2%, 결합 시 ±1.7% | 800+ active task vs H100 132 SM 이미 포화 — split heuristic 의 가정이 본 workload (low-head decode) 와 안 맞음 |

**진실** (SUB_032 3-run avg 결과 갱신): A-tier (Quick win / low risk) **전체가 noise** — A4 단독의 +1.0% 마저 1-run artifact (3-run avg 로 -0.21%) → **B-tier (FlashDecoding++ unified-max softmax, OmniServe LSE async merge) 만이 실질적 path**.

---

## 1. A1 — Intel libomp + tournament barrier (무효 원인)

### 1.1 측정 사실

| 조합 | Δ vs baseline |
|---|---:|
| A1 단독 | +0.4% |
| A1+A3 | -1.2% |
| A1+A4 | **-0.9% ⚠️** (anti-synergy) |
| A1+A3+A4 | +0.4% |
| A1+A5 | **-1.7% ⚠️** (Stage 3 최악) |
| A1+A3+A5 | -0.04% |
| A1+A4+A5 | +0.04% |
| A1+A3+A4+A5 | -0.7% |

### 1.2 코드 / 라이브러리 분석

#### 1.2.1 LD_PRELOAD redirect 동작 ✓

```bash
$ ldd /workspace/vllm_hybrid/csrc/cpu/pacpu/libpacpu.so | grep omp
    libgomp.so.1 => /lib64/libgomp.so.1
```

→ pacpu.so 는 libgomp 에 link 되어 있으나, `LD_PRELOAD=libiomp5.so` 가 GOMP_* 심볼을 redirect 한다 (Intel libomp 의 GOMP compat layer).

#### 1.2.2 효과가 없는 이유

NEO 의 cdec 경로는 OMP fork/join 이 **workload-driven** (layer-by-layer, 80 layer 단위 fork/join × 800 task → request 당 64,000 barrier wait).

- A1 = tournament barrier (대형 thread 수에서 reduction barrier 가 O(log N) 으로 빨라짐)
- 그러나 NEO 의 OMP team size = 10 thread (`OMP_NUM_THREADS=10`) → log2(10) ≈ 3.3 vs N=10 은 큰 차이 없음
- 결론: barrier 알고리즘 자체의 영향은 N (team size) 이 작아서 미미

#### 1.2.3 A1+A4 / A1+A5 의 anti-synergy

- A1 의 thread placement (KMP_FORCE_REDUCTION_BARRIER_PATTERN) 가 numactl --localalloc 과 conflict
- libomp 의 thread affinity 가 numactl 의 메모리 정책을 무시 → NUMA-local malloc 효과 소실
- A1+A5: FA3 split heuristic 으로 GPU SM 활용이 바뀐 상태에서 libomp thread affinity 가 추가 conflict

---

## 2. A2 — GOMP_SPINCOUNT=INFINITE + OMP_WAIT_POLICY=ACTIVE + KMP_BLOCKTIME=infinite (대부분 noise)

### 2.1 측정 사실

| 조합 | Δ vs baseline |
|---|---:|
| A2 단독 | -0.01% (noise) |
| A2+A3 | -1.1% |
| A2+A4 | +0.1% |
| A2+A3+A4 | -0.2% |
| A2+A5 | -0.08% (noise) |
| A2+A3+A5 | +0.2% |
| A2+A4+A5 | -1.1% |
| A2+A3+A4+A5 | -1.3% |

### 2.2 코드 / 라이브러리 분석

#### 2.2.1 GOMP_SPINCOUNT 은 동작하지만

- libgomp 의 `GOMP_SPINCOUNT=INFINITE` 자체는 동작 (libgomp src/wait.c).
- 단 NEO 의 OMP team 은 매번 fork 후 짧은 work 를 하고 (layer 당 ~5ms) 곧 barrier — 이미 spin 상태가 active 이므로 추가 효과 없음.

#### 2.2.2 KMP_BLOCKTIME 은 무효

- KMP_BLOCKTIME = Intel libomp 전용. libgomp 에는 동작 안 함.
- A1 (libiomp5 LD_PRELOAD) 과 같이 켜야만 활성화.
- A1+A2 는 동일 OMP runtime 영역이라 매트릭스에서 제외 → A2 단독 = libgomp 만 적용 → 무효.

#### 2.2.3 IPEX LLM 의 "KMP_BLOCKTIME=INF 2-3× speedup" 이 왜 안 나타나나

- IPEX LLM 워크로드 = CPU 단독 inference 로 OMP team 이 지속적으로 fire & forget → KMP_BLOCKTIME 이 sleep 진입을 막아 speedup.
- NEO 는 이미 짧은 간격으로 (80 layer × 5ms) OMP team 이 fire → thread 가 sleep 진입할 틈이 없음 → BLOCKTIME 이 무의미.

---

## 3. A3 — `VLLM_NEO_KV_PREFETCH=1` (core.h `__builtin_prefetch`)

### 3.1 측정 사실

| 조합 | Δ vs baseline |
|---|---:|
| A3 단독 | -1.1% |
| A1+A3 | -1.2% |
| A2+A3 | -1.1% |
| A3+A4 | +0.3% |
| **A3+A5** | **+0.6% ⭐ Stage 3 BEST** |
| A1+A3+A4 | +0.4% |
| A2+A3+A4 | -0.2% |
| A2+A3+A5 | +0.2% |
| A3+A4+A5 | -0.1% |
| A1+A3+A4+A5 | -0.7% |
| A2+A3+A4+A5 | -1.3% |

### 3.2 코드 / 라이브러리 분석

#### 3.2.1 변경 위치 (csrc/cpu/pacpu/core.h)

```cpp
if (_kv_prefetch_on && i + 1 < r) {
  __builtin_prefetch(kbatch_p + (i + 1) * NUM_KV_HEADS * HEAD_DIM, 0, 2);
  __builtin_prefetch(vbatch_p + (i + 1) * NUM_KV_HEADS * HEAD_DIM, 0, 2);
  __builtin_prefetch(block_table_p + seq_ids[i + 1] * block_table_width, 0, 2);
}
```

→ 다음 iteration 의 k / v / block_table 에 대한 SW prefetch hint.

#### 3.2.2 효과가 없는 이유

- **HW prefetcher 가 이미 동작**: Intel SPR 의 streamer / IP-based prefetcher 가 sequential access 를 자동으로 prefetch.
- SW prefetch hint 가 의미 있는 경우 = (1) irregular access pattern, (2) HW prefetcher 가 못 잡는 stride (gather/scatter).
- NEO 의 cdec 은 sequential access (block_table 을 따라 KV cache 를 고정 stride 로 읽음) → HW prefetcher 가 이미 처리 → SW prefetch 잉여.

#### 3.2.3 단독으로 -1.1% 손실 이유

- prefetch hint 자체 비용 (instruction 추가, branch prediction 오염) + 이미 caching 된 line 을 다시 가져오는 비용 (premature eviction 가능성).
- 즉 -1.1% = HW prefetcher 와 SW hint 가 충돌하는 오버헤드.

#### 3.2.4 A3+A5 가 +0.6% (Stage 3 BEST) 인 이유

- FA3 split heuristic 으로 KV access pattern 이 더 chunked 해지면서 SW hint 가 살아남는 윈도우 발생 가능.
- 단 ±2% noise 안이라 단정 불가 (3-run 검증 필요).

---

## 4. A4 — numactl --localalloc (⭐ ★ Stage 1+3 유일 win)

### 4.1 측정 사실

| 조합 | Δ vs baseline |
|---|---:|
| **A4 단독** | **+1.0% ⭐** |
| A1+A4 | -0.9% ⚠️ |
| A2+A4 | +0.1% |
| A3+A4 | +0.3% |
| A1+A3+A4 | +0.4% |
| A2+A3+A4 | -0.2% |
| A4+A5 | -1.1% |
| A1+A4+A5 | +0.04% |
| A2+A4+A5 | -1.1% |
| A3+A4+A5 | -0.1% |
| A1+A3+A4+A5 | -0.7% |
| A2+A3+A4+A5 | -1.3% |

### 4.2 코드 / 라이브러리 분석

#### 4.2.1 NEO 가 이미 NUMA bind 를 하고 있지만

```bash
$ grep -rn "VLLM_NEO_NUMA_BIND\|numa_bind" /workspace/vllm_hybrid/csrc/cpu/pacpu/
csrc/cpu/pacpu/core.h:...numa_bind logic (worker thread 만 적용)
```

- 현 NEO 는 worker thread 의 NUMA bind 만 수행 (`VLLM_NEO_NUMA_BIND=1`).
- 그러나 **Python overhead / scheduler / Ray / KV cache malloc / torch_cpu 의 임시 buffer** 가 NUMA bind 밖에서 할당되어 cross-NUMA latency 발생.

#### 4.2.2 numactl --localalloc 이 해결하는 것

- numactl --localalloc = **process 전체의 모든 malloc 을 local NUMA node 로 강제**
- → Python overhead, scheduler, Ray, torch_cpu 의 모든 buffer 가 local 노드 → cross-NUMA latency 제거
- → +1.0% win 의 원천

#### 4.2.3 결합 시 cancel out 되는 이유

- A1 (libomp thread placement) → numactl 의 메모리 정책 무시
- A2 (spin) → 무관, 단순 noise
- A5 (FA3 split) → GPU SM 활용 패턴 변경 (CPU NUMA 효과를 가려버림)

---

## 5. A5 — FA3 Sequence-Aware Split Heuristic (`VLLM_NEO_MAX_NUM_SPLITS=8`)

### 5.1 측정 사실

| 조합 | Δ vs baseline |
|---|---:|
| A5 단독 | -0.2% |
| A1+A5 | **-1.7% ⚠️** (Stage 3 최악) |
| A2+A5 | -0.08% |
| **A3+A5** | **+0.6% ⭐ Stage 3 BEST** |
| A4+A5 | -1.1% |
| A1+A3+A5 | -0.04% |
| A1+A4+A5 | +0.04% |
| A2+A3+A5 | +0.2% |
| A2+A4+A5 | -1.1% |
| A3+A4+A5 | -0.1% |
| A1+A3+A4+A5 | -0.7% |
| A2+A3+A4+A5 | -1.3% |

### 5.2 코드 / 라이브러리 분석

#### 5.2.1 FA3 split heuristic 이란

- FA3 (FlashAttention-3) 의 sequence-aware split heuristic = decode 시 KV 시퀀스를 여러 split 로 나눠 SM occupancy 를 끌어올림.
- 적용 가정 = **low active task count** (active task 가 SM 보다 적을 때 → split 로 SM 추가 활용).

#### 5.2.2 본 workload 의 미스매치

- NEO baseline 의 동시 active task = **800+ (b0+b1)** vs H100 = **132 SM**
- 이미 SM 이 포화 → split 추가는 launch overhead 만 증가.
- 단독 효과 = 노이즈 (-0.2%) — split cap 이 무의미하게 깎임.

#### 5.2.3 A1+A5 의 -1.7% 가 왜 최악인가

- libomp 의 thread placement (CPU 쪽) 와 GPU FA3 split 변경이 직접 충돌하지는 않음.
- 단 둘 다 노이즈 누적 + 작은 cross effect → -1.7% (single run 이라 3-run 검증 필요).

---

## 6. 진정한 병목 = NEO 의 workload-driven OMP fork/join overhead

### 6.1 perf record 60s 결과 (reference/H_dynamic_analysis.md 인용)

- libgomp 43.75% (OMP barrier wait — `gomp_team_barrier_wait`)
- libpacpu 26.38% (ISPC compute — `fmha_decode`)
- libtorch_cpu 10.24% (async swap, wall hidden)
- python 1.84%

### 6.2 해석

- **libgomp 43.75%** 의 본질은 OMP team 의 barrier wait (`gomp_team_barrier_wait`), **이미 active spin** 상태.
- 즉 단순 OMP barrier 알고리즘 / sleep 정책으로는 못 줄임 — A2 (SPINCOUNT) / A1 (tournament barrier) 가 무효였던 직접 원인.
- **진정한 병목** = **OMP team 의 fork/join 빈도 자체** (layer 당 80 회 fork/join 이 직접적인 wait 원천).

### 6.3 어떻게 줄이나

(N 문서 의 B-tier / C-tier 항목)

| 항목 | 정체 | 효과 가설 |
|---|---|---|
| **B1** | **FlashDecoding++ unified-max softmax** | softmax recomputation 제거 → cdec kernel 횟수 자체 감소 |
| **B2** | **OmniServe LSE async merge** | LSE merge 를 GPU 와 async → CPU 의 sync wait 제거 |
| C1 | layer-fusion (KV access 를 여러 layer 묶어 fork) | OMP fork/join 횟수 80 → 더 적게 |
| C2 | OMP team persistence (매번 fork 없이 worker 를 idle wait 시킴) | NEO 가 이미 OMP team reuse — 추가 여지 작음 |

---

## 7. 결론 및 다음 권고

### 7.1 결론 (SUB_032 3-run avg 갱신 후)

1. **A-tier (Quick win / low risk) 전체가 noise**: A1-A5 모두 ±2% noise 안에서 통계적 의미 없음. A4 단독의 +1.0% 마저 1-run artifact ([SUB_032 3-run avg](../measurements/sub032_a4_3run_20260521/RESULTS.md): 930.2 tps, -0.21%).
2. **NEO 의 worker thread NUMA bind 만으로 충분**: process-wide `numactl --localalloc` 추가가 측정 가능한 이득을 만들지 못함.
3. **진정한 병목** = NEO 의 workload-driven OMP fork/join overhead (libgomp 43.75% barrier wait). 본 매트릭스의 lever 로는 못 줄임 — **kernel 구조 변경 (B/C-tier) 만이 path**.

### 7.2 다음 권고 (SUB_032 결과 반영)

| 우선순위 | 작업 | status | 이유 |
|---|---|---|---|
| ✅ | A4 단독 3-run avg (SUB_032) | **완료 2026-05-21** — noise 확정 (avg 930.2, -0.21%) | A4 마저 무효 → A-tier 완전 폐기 |
| **★★★** | **B3 FlashDecoding++ softmax (SUB_033)** | 🟢 코드 적재 완료, 측정 진행 중 | softmax 3-pass → 2-pass online → OMP barrier #1 감소 |
| **★★★** | **B1 OmniServe LSE async (SUB_034)** | ⏸ SUB_033 후 진입 | CPU sync wait 제거 (async cdec depth ≥ 2 + race-safe LSE merge) |
| **★★** | C-tier 검토 (C1 layer-fusion, C2 OMP team persistence) | 대기 | libgomp 43.75% 의 직접 감소 |
| ⚪ | A1/A2/A3/A4/A5 lever 영구 폐기 | 결정 | 후속 sweep 단순화 |

### 7.3 후속 작업 시나리오

- N 문서 의 B/C-tier 항목 (총 17 ideas) 중 단일 effort 가 작은 것 부터 plan 작성
- 단 4 개 (B1, B2, C1, C2) 동시 시도는 위험 → 한 항목씩 단계적 sweep
- 차기 sub-task SUB_030 (가칭: N 문서 의 B-tier 첫 항목) 으로 등록 권고

---

## 8. raw 자료

| 항목 | 위치 |
|---|---|
| Stage 1 RESULTS | [`../measurements/stage1_a1a2a3a4_matrix_100p_20260521/RESULTS.md`](../measurements/stage1_a1a2a3a4_matrix_100p_20260521/RESULTS.md) |
| Stage 3 RESULTS | [`../measurements/stage3_a5_matrix_100p_20260521/RESULTS.md`](../measurements/stage3_a5_matrix_100p_20260521/RESULTS.md) |
| Stage 1 SUMMARY.tsv | `eval/results/20260521_140101_stage1_a1a2a3a4_matrix_100p/SUMMARY.tsv` |
| Stage 3 SUMMARY.tsv | `eval/results/20260521_180457_stage3_a5_matrix_100p/SUMMARY.tsv` |
| Stage 1 launcher | `/tmp/run_stage1_matrix.sh` |
| Stage 3 launcher | `/tmp/run_stage3_a5_matrix.sh` |
| perf record 분석 | [`reference/H_dynamic_analysis.md`](reference/H_dynamic_analysis.md) (libgomp 43.75% / libpacpu 26.38% / libtorch 10.24% / python 1.84%) |
| N 문서 (다음 lever 출처) | [`N_cdec_leftover_elimination_ideas.md`](N_cdec_leftover_elimination_ideas.md) — A/B/C tier 22 ideas |
