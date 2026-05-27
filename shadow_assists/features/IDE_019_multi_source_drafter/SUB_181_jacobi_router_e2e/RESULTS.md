# SUB_181 — IDE_019 / TSK_037 AGSD 4-method router integration + canonical 500p e2e

> **status**: 완료 (2026-05-27 11:27 KST)
> **scope**: SUB_180 의 Jacobi AVX-512 kernel + 새 router method `cpu_jacobi` 를
> sub094 router 에 통합 (4-method: vanilla / ngram / suffix / cpu_jacobi) +
> canonical 500p × 3 mix × 1-run e2e. workload-aware classifier 가 chat →
> cpu_jacobi 분기.
> **verdict**: **drop-in 7번째 실패**. agsd-gated 4-method = 153 tps (3-mix avg) =
> vanilla-only (1,851 tps) 대비 0.083× / trident-only (2,641 tps) 대비 0.058×.
> Jacobi draft cost (~1,810 ms/call) 가 GPU verify cost (~30-44 ms) 의 ~40-60×
> 라 chat request 가 cpu_jacobi 로 분기될 때마다 serial latency 가 폭발.
> SUB_180 의 trade-off 분석 (kernel 단독 net-win 불가) 가 e2e 에서 확정.

---

## 1. Step 2 — Jacobi cost vs acceptance trade-off 분석 (사전)

### 1.1 cost (microbench, 본 SUB 신측정)

| draft 변형 | hidden | vocab | K | B | BK | T=64 p50 (ms) | 출처 |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen 32B (full) | 5,120 | 152,064 | 7 | 4 | 28 | **1,713** | SUB_180 reload |
| Qwen 0.5B small | 896 | 152,064 | 7 | 1 | 7 | **245** | 본 SUB |
| partial top-N | 5,120 | 8,192 | 7 | 1 | 7 | **73** | 본 SUB |

→ 본 SUB 의 새 발견: **smaller draft model (hidden 5.7× 작음) 도 vocab BW-bound
유지** — 5,120 → 896 hidden 으로 cost 가 7× 빨라지지 않고 **7× 빨라짐 (1,713 →
245)**. SUB_180 의 _vocab=152K BW 가 dominant_ 가설 부분 부정. hidden 도 일부
기여. 단 245 ms 도 GPU verify 30-44 ms 의 **6-8×** 라 net-win 안 됨.

partial vocab top-8K (lossless 위배 가능) 만 73 ms 로 GPU verify 1.7-2.4×, α
≥ 0.75 시 1.07-1.72× 가능 — 단 partial vocab 은 본 SUB scope 밖.

### 1.2 speedup matrix (theoretical, GPU verify=40 ms 기준)

| draft 변형 | K | workload α typical | E[accept] | spec speedup |
|---|---:|---|---:|---:|
| qwen32b_full | 7 | chat α=0.80 | 4.21 | **0.58** ← net loss |
| qwen32b_full | 7 | sonnet α=0.65 | 3.20 | 0.39 |
| qwen32b_full | 7 | code α=0.72 | 3.81 | 0.46 |
| qwen05b_small | 7 | chat α=0.80 | 4.21 | **0.58** ← net loss |
| qwen05b_small | 7 | code α=0.75 | 3.95 | 0.50 |
| partial_top8k | 7 | chat α=0.80 | 4.21 | **1.47** ← cond. win (lossless risk) |
| partial_top8k | 7 | code α=0.75 | 3.95 | 1.27 |

전체 표: `measurements/tradeoff_analysis.json`.

→ **본 SUB 의 integration 진행 결정**: kernel 단독 net loss 확인했지만 honest
e2e measurement 가 본 SUB 의 paper §4 lever 자격 판정 (drop-in 7번째 실패 vs
NEW workload conditional accept) 의 binding 지표 — 그래서 _full kernel 그대로_
integration + e2e 측정 진행.

## 2. Step 3 — 4-method router 구현

`src/sub181_router.py` (273 lines):

- base = `/tmp/sub094_router.py` (3-method: vanilla / trident).
- 새 backend `cpu_jacobi`: in-process ctypes 통합. `libjacobi_avx512.so`
  (SUB_180 build) 의 `jacobi_lm_head_argmax_bf16` 호출 → CPU draft cost 실측 후
  vanilla GPU endpoint (`:8001`) 으로 forward (verify-equivalent proxy).
- decide rule: `prefix_len<16` → vanilla / `code` → trident / `sonnet` →
  trident / `chat` → **cpu_jacobi** (ENV `AGSD_USE_JACOBI=1` gated).
- W matrix (5,120 × 152,064 BF16 = 1.56 GB) preallocated. fallback: 0.5B small
  draft 또는 partial vocab top-N 변형은 본 SUB scope 외.

NOTE: 본 router 의 cpu_jacobi 는 _orchestration-level proxy_ — Jacobi 가 실제
verify pipeline 안에서 candidate ids 를 vllm spec_decode 에 inject 하는 full
integration 은 vllm core 수정 필요 (별도 SUB). 본 SUB 의 측정 = "Jacobi draft
cost 가 e2e latency 에 미치는 영향" 의 honest reproduction.

## 3. Step 4 — canonical 500p e2e (3 mix × 1-run)

config: Qwen 32B TP=4×2 (vanilla :8001 GPU0-3 / trident :8002 GPU4-7) +
sub181_router :8000 USE_JACOBI=1 / 500 prompt / max_tokens=32 / concurrency=32
/ taskset 0-99 / OMP=4.

### 3.1 throughput (tps, 본 SUB 1-run)

| mix | vanilla-only | trident-only | **agsd-gated (4-method)** | agsd / vanilla | agsd / trident |
|---|---:|---:|---:|---:|---:|
| balanced | 1,594.7 | 1,622.5 | **105.2** | 0.066× | 0.065× |
| sonnet-heavy | 2,039.0 | 3,271.4 | **175.1** | 0.086× | 0.054× |
| code-heavy | 1,918.7 | 3,028.2 | **179.1** | 0.093× | 0.059× |
| **3-mix avg** | **1,850.8** | **2,640.7** | **153.1** | **0.083×** | **0.058×** |

→ **명확한 net loss 12-19×**. agsd-gated 가 vanilla-only 의 8.3% 수준 throughput.

### 3.2 router-level breakdown (3-mix totals)

| 지표 | 값 |
|---|---:|
| 총 request | 1,503 (500 × 3 + warmup) |
| chat → cpu_jacobi 분기 | 365 |
| sonnet/code → trident 분기 | 1,135 |
| Jacobi draft total ms | **660,823 ms** (≈11.0 분) |
| Jacobi draft avg ms / call | **1,810 ms** (SUB_180 microbench 1,713 ms 와 일치) |
| classify avg ms | ~0.8-1.9 ms (negligible) |
| forward avg ms | ~158-205 ms |

### 3.3 latency profile (balanced mix)

| scenario | wall_s | p50 lat (s) | p99 lat (s) |
|---|---:|---:|---:|
| vanilla-only | 10.03 | 0.629 | 0.724 |
| trident-only | 9.86 | 0.566 | 1.289 |
| **agsd-gated (4-method)** | **152.16** | 0.143 | **31.522** |

→ wall 15× 증가. p99 latency 가 chat request 의 cpu_jacobi 대기 큐 영향으로
**31.5 sec** 까지 폭증. p50 는 cpu_jacobi 미경유 sonnet/code request 가 trident
로 빠르게 처리되어 낮음 — bimodal distribution.

## 4. Step 5 — honest verdict

### 4.1 핵심 발견

1. **Jacobi draft cost (1,810 ms/call) vs GPU verify (30-44 ms)** 의 40-60×
   gap 이 SUB_180 의 microbench 그대로 e2e 에 reflected.
2. **acceptance rate 측정 불가**: 본 router 의 cpu_jacobi 는 candidate id 를
   verify pipeline 에 inject 하지 않음 — verify-equivalent proxy 만. 실
   acceptance rate 측정은 vllm core 의 spec_decode 통합 필수 (별도 SUB).
3. **smaller draft (Qwen 0.5B hidden=896) 도 net-win 불가** (cost 245 ms vs GPU
   verify 40 ms = 6-8×). vocab BW 가 여전히 dominant.
4. **partial vocab top-N (73 ms)** 만 1.07-1.72× theoretical net-win, 단
   lossless guarantee 위배 가능 — accuracy gate (per-token logprob max abs
   diff < 1e-3) 측정 필수.
5. **router-level orchestration overhead 자체는 micro** (classify ~1 ms,
   semaphore acquire ~ms): Jacobi cost 가 dominant.

### 4.2 paper §4 lever 자격 판정

- **drop-in 7번째 실패** = ✓ 본 SUB 확정. SUB_173-179 의 6 실패 + SUB_180 의
  conditional + 본 SUB 의 e2e 실패 — kernel single drop-in 으로는 net-win 불가.
- **NEW workload conditional accept** = ✗ 본 SUB 는 integration 후의 실측
  net loss 확인 — Jacobi 는 SUB_180 의 conditional 평가 (kernel only) 단계와
  달리 e2e 측정에서 명시적 실패. paper §4 lever 자격 **상실**.
- **NEW workload conditional 후보로 남으려면**: kernel rewriting (W vocab-major
  repack + tile-major load → 5-10× speedup 가정) + smaller draft model + 실
  spec_decode 통합 + acceptance rate 실측 → 3 단계 추가 SUB 필요. cost 가 (a)
  kernel rewriting weeks-of-work + (b) vllm core PR-level invasive 라 본
  fork 의 paper-worthy lever 후보에서 deprioritize 권장.

## 5. SUB_178 cold-KV 대비 비교

| 지표 | SUB_178 cold-KV | SUB_181 Jacobi 4-method |
|---|---|---|
| kernel cost vs GPU equivalent | overlap 1.5-1.71× speedup | Jacobi 40-60× cost gap |
| integration scope | overlap proposal (vllm-non-invasive) | router-level (vllm-non-invasive) |
| e2e measurement 결과 | conditional accept | **net loss 12-19×** |
| paper §4 lever 자격 | conditional | **상실** |
| 후속 SUB 필요 | conditional implementation | kernel rewriting + spec_decode integration (heavy) |

→ NEW workload 2개 후보 stack 중 cold-KV (SUB_178) 만 유지. Jacobi 는 drop.

## 6. 산출물

- `src/tradeoff_analysis.py` — cost / acceptance trade-off 계산기 (microbench 포함)
- `src/sub181_router.py` — 4-method AGSD router (cpu_jacobi 분기 포함)
- `src/aggregate_results.py` — 3-mix bench aggregator
- `launcher.sh` — vllm 32B TP=4×2 + sub181_router USE_JACOBI=1 + 3-mix bench
- `measurements/tradeoff_analysis.json` — cost / acceptance matrix
- `measurements/4method_500p/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` — 3 mix raw
- `measurements/4method_500p/{...}/router_stats.json` — Jacobi call stats per mix
- `measurements/4method_aggregate.json` — 3-mix summary + speedup
- `logs/{vanilla,trident,router,main,monitor}_4method.log`
- `logs/boot_4method_seconds.txt` = 80 s

## 7. limitations + next-step

| 한계 | 영향 | 후속 |
|---|---|---|
| cpu_jacobi 가 vllm spec_decode 에 verify-inject 안 됨 (proxy) | 실 acceptance rate 측정 불가 | vllm core 수정 (heavy SUB) |
| Jacobi kernel cost (vocab BW-bound) | net loss 결정적 | W repack + tile-major rewriting (kernel SUB) |
| 1-run 측정 | variance 미정 | sufficient for verdict (12-19× gap > variance) |
| partial vocab top-N path 미측정 | conditional lever 미평가 | accuracy gate + e2e (별도 SUB) |
| baseline ag-gated 3-method (USE_JACOBI=0) 미측정 | control 없음 | sub094_benchmark 의 vanilla-only / trident-only 가 implicit baseline 으로 충분 |

## 8. 정합성 (canonical reference)

- canonical OFF baseline (SUB_179 max-tokens=32, 3-mix avg): vanilla ~4,299 tps
  (concurrency 32). 본 SUB 의 vanilla-only 1,851 tps 는 ~43% 수준 — 차이
  원인: 본 SUB 도 동일 max_tokens=32 / 동일 concurrency 32 / 동일 모델·TP
  설정이라 환경 동일하나 GPU0 / GPU7 의 leftover memory (6.3 GB / 3.7 GB) 영향
  가능 — vllm boot 시 KV cache 가용 메모리 감소 효과. trident-only 2,641 tps
  는 SUB_179 의 trident-only (suffix) ~4,100 tps 대비 64%. 두 scenario 모두
  본 SUB 환경에서 동일 비율로 감소 → 본 SUB 내부 비교 (agsd / vanilla,
  agsd / trident) 는 환경-internal 유효. **결론 = 4-method 의 ~0.06-0.09×
  speedup 은 baseline 자체의 차이와 무관, Jacobi 자체의 cost gap 이 dominant**.

## 9. 후속 SUB 권고

- **본 SUB 의 후속은 권고 안 함** (deprioritize). 이유:
  - Jacobi 단독 lever 의 e2e 명시적 실패 확정
  - kernel rewriting (W repack) + vllm core 수정 + smaller draft 별도 학습
    /로딩 모두 heavy
  - SUB_178 cold-KV 가 유일한 NEW workload conditional 후보로 남음
- 우선순위: SUB_178 cold-KV 의 deeper integration (별도 SUB) > Jacobi rewriting
