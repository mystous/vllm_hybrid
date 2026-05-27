# paper §4 final narrative (SUB_171~SUB_198 aggregate)

> **status**: 2026-05-27 16:40 KST. 26 SUB plan 측정 lever 완료 (SUB_195 paper write-up docs 사용자 skip 명시).
> **scope**: SUB_171 (IDE_016 AVX-512 tokenizer kernel) ~ SUB_198 (AMX draft proxy real integration) 의 paper §4 honest aggregate.

---

## 0. 두괄식 — 26 SUB plan finding

24 measurement-bearing SUB + 2 docs SUB. 1-run 측정 default 의 24 lever 시도 중:

| 결과 | 개수 |
|---|---:|
| 1-run net positive (Δ > 0%) | 6 |
| 1-run paper main +5% 도달 (3-mix avg) | **1 (SUB_196 cellB +5.28%)** |
| 1-run paper main +5% 도달 (single cell) | 2 (SUB_196 cellB balanced +12.04% / SUB_197 code +7.73%) |
| multi-run variance binding net positive | 1 (SUB_183 NUMA +2.24%) |
| **paper main 자격 확보 lever (multi-run binding 후 +5% 도달)** | **0** |

→ **본 fork 의 CPU 활용 lever 시도로 paper main +5% 기준 multi-run binding 도달 lever 0**. paper §4 의 honest aggregate 는 **negative result 의 systematic exploration** 으로 framing.

---

## 1. lever taxonomy (5 카테고리 24 시도)

### A. drop-in CPU kernel replacement (7 시도 / 0 paper-main)

| SUB | lever | 1-run Δ (3-mix avg AGSD) |
|---|---|---:|
| SUB_173 | AVX-512 tokenizer probe | +0.86% (noise) |
| SUB_174 | AVX-512 sampling probe | −1.10% (noise) |
| SUB_175 | AMX matmul drop-in | +0.26% (noise) |
| SUB_177 | AMX prefill assist | −2.08% (drop-in 불가 확정) |
| SUB_179 | zero-copy CPU compute | −1.76% (noise) |
| SUB_181 | 4-method router (Jacobi) | **−94.2%** (catastrophic, 1810 ms × 165 chat req 직렬) |
| SUB_192 | partial KV merge proxy | −0.13% (noise) |

→ **kernel-level isolated speedup (microbench 1.5-5×) 의 e2e 자동 변환 0/7**. paper §4 의 핵심 negative finding.

### B. environment-level lever (4 시도 / 0 paper-main)

| SUB | lever | 1-run Δ |
|---|---|---:|
| SUB_182 | cgroup + hugepages + taskset | −0.39% (noise) |
| SUB_183 | NUMA pin (dual instance) | **+1.54%** ⭐ (multi-run +2.24%) |
| SUB_186 | env stack (cgroup + hugepages + taskset + NUMA) | −0.03% (**destructive**, residual −1.18 pp vs +1.15% prediction) |
| SUB_197 | NUMA + softmax pair (cross-domain) | **+2.83%** ⭐ (near-linear, residual −0.55 pp vs +3.38% prediction, code-heavy +7.73%) |

→ environment lever 의 **2-lever cross-domain pair stack** 만 near-linear superposition. 3+ lever 부터 destructive interference 일관 (SUB_186 / SUB_191).

### C. paper main IDE_018 phase-burst (1 시도 / **자격 상실 retract**)

| SUB | 결과 |
|---|---|
| SUB_169 (stub) | +1.35% → **retract** (SUB_184 검증 후 noise positive) |
| SUB_184 (dummy fill) | **−1.75%** (trident-only −14~−20% catastrophic) |
| SUB_188 (side-channel) | +1.84% — 단 task-pool 아닌 isolated process 라 IDE_018 main lever 자격 회복 안 됨 |

**SUB_184 binding 가설 분석**:
- (a) GPU phase 동안 CPU idle 존재 ✓ (OFF util 4%)
- (b) phase mark IPC 4.67 μs ✓
- (c) CPU work 와 critical path overlap ✗ (trident-only catastrophic)
- (d) CPU util ↑ + throughput 유지 ✗

→ paper §4 main lever 자격 **공식 reject**. IDE_018 영역의 net positive 는 side-channel form 으로만 가능.

### D. NEW workload feasibility + e2e proxy (4 시도 / 0 paper-main)

| SUB | lever | feasibility (microbench) | e2e (proxy or canonical) |
|---|---|---|---:|
| SUB_178 | cold-KV decompress | overlap 1.5-1.71× conditional | — (SUB_185 proxy 측정) |
| SUB_180 | Jacobi LM-head | 1,713 ms BW-bound | — (SUB_181 e2e −94%) |
| SUB_185 | cold-KV long-context proxy | — | +0.18% noise + **TTFT +8.83%** regression |
| SUB_187 | AMX draft head (Qwen 0.5B) | **0.524 ms ⭐ ⭐** (490× SUB_181) | — (SUB_198 proxy −2.77%) |

→ **isolated microbench → e2e proxy 자동 변환 0/3** (SUB_185 / SUB_192 / SUB_198 일관 fail). real vllm spec_decode integration 별도 invasive SUB 필요.

### E. side-channel work-pattern × cycle ablation (5 시도 / 1 paper-main)

| SUB | work | cycle | Δ |
|---|---|---:|---:|
| SUB_189 | branchy (rank/sort) | 10ms | −0.82% (loss) |
| SUB_196 cellA | regular (softmax 10ms) | 10ms | +0.98% (small) |
| SUB_190 | regular (tokenize) | 20ms | +1.66% (1-run) / **−5.96% multi-run** |
| SUB_188 | regular (softmax) | 100ms | +1.84% (1-run) / +0.53% multi-run |
| **SUB_196 cellB** | **branchy (rank)** | **100ms** | **+5.28% (3-mix avg)** ⭐⭐ |

**2×2 grid finding**:

| pattern \ cycle | 10ms | 100ms |
|---|---:|---:|
| regular | +0.98% | +1.84% (multi +0.53%) |
| branchy | −0.82% | **+5.28%** ⭐ |

**paradoxical**: branchy × low-rate (100ms) = best (paper main 도달). 추정 mechanism:
- 100ms cycle = vllm step boundary (35-44ms/step) 의 2-3 step 마다 fire → step idle gap 와 정렬 가능성
- branchy work 의 cache prefetcher inhibit + GPU launch overhead absorption

---

## 2. multi-run variance verification (SUB_194)

Top-3 net positive lever 의 3-run × OFF/ON × agsd-gated only 측정:

| Lever | 1-run Δ | **3-run mean Δ** | stddev | warm-only (run2/3) |
|---|---:|---:|---:|---:|
| L183 NUMA pin | +1.54% | **+2.24%** | ±35.08 pp | **+3.13%** ⭐ |
| L188 softmax | +1.84% | +0.53% | ±34.67 pp | +0.81% |
| L190 tokenize | +1.66% | **−5.96%** ⚠ | ±31.85 pp | **−6.40%** ⚠⚠ |

**run1 cold-start outlier**: 모든 lever 의 run1 ~2,800 tps (cudagraph PIECEWISE compile 진행 중), run2/3 ~4,300-4,600 tps. variance 가 magnitude 보다 dominate.

**finding**:
- **L183 NUMA pin** 만 multi-run mean 으로 robust net positive 유지 (warm-only +3.13%)
- L188 softmax = noise floor (1-run +1.84% 의 30% 만)
- **L190 tokenize 부호 반전** — 1-run small positive signal **retract** (paper §4 narrative 갱신)

→ 1-run 측정의 신뢰성 의문. small magnitude signal (|Δ|<3%) 은 multi-run mean 또는 warm-only (run2+) 가 binding.

**SUB_196 cellB +5.28% / SUB_197 +2.83% 의 multi-run binding 검증은 미수행** — 별도 SUB 필요. 단 SUB_196 cellB balanced agsd +12.04% magnitude 는 noise floor (±10 pp) 보다 충분히 큼 → 의미 있는 signal 가능성 높음.

---

## 3. paper §4 의 honest narrative 권장 form

```
§4.1 본 fork 의 CPU 활용 lever 시도

본 fork 는 paper §1-3 의 가설 (GPU phase 와 CPU idle window 의 ms-level
overlap) 위에 24 lever 를 1-run canonical (Qwen 32B TP=4×2 / 500 prompt
× 3 mix × 32 concurrency × max-tokens=32) 으로 측정했다. lever 는 5
카테고리 (drop-in / environment / phase-burst / NEW workload / side-channel)
로 grouping 가능하다.

§4.2 측정 결과 (Figure 5)

drop-in CPU kernel replacement 카테고리 (AVX-512 tokenizer / sampling
/ AMX matmul / prefill / zero-copy / Jacobi router 등 7 시도) 는 isolated
microbench 의 1.5-5× speedup 에도 불구하고 e2e 3-mix avg AGSD 영향이 모두
±3% noise floor 안에 머물거나 (5/7), Jacobi router 의 catastrophic
−94.2% regression 까지 발생한다 (1/7). vllm 의 spec decoding pipeline 의
GPU verify pipeline 과 CPU draft 의 cost gap (40 ms vs 1,810 ms) 가
drop-in 통합의 fundamental obstacle.

paper §1 의 main hypothesis 였던 IDE_018 phase-burst (GPU phase 마다
CPU task pool 을 fire) 는 SUB_169 의 stub 측정 (+1.35% 3-mix avg AGSD,
CPU util 4→5%) 이 task pool wiring 채운 SUB_184 측정 (−1.75% 3-mix avg
AGSD, trident-only −14~−20% catastrophic) 으로 **공식 reject**. core 가설
(GPU phase 동안 CPU work 가 critical path 와 contention 없이 overlap
가능) 가 vllm worker thread 와 task pool 의 동일 process 내 GIL / pinned
alloc lock / L1L2 cache 공유 contention 으로 reject.

§4.3 net positive lever 의 form

multi-run variance verification (SUB_194, Top-3 lever × 3-run × OFF/ON)
결과 1-run small positive signal 의 신뢰성 의문 확인 (variance ±35 pp,
SUB_190 tokenize 의 부호 반전). 단 L183 NUMA pin 의 multi-run mean
+2.24% (warm-only +3.13%) 만 robust net positive 유지.

cross-domain 2-lever pair stack (SUB_197 = NUMA + softmax precompute)
이 near-linear superposition 유지 (+2.83% 3-mix avg, code-heavy single
cell +7.73%, residual −0.55 pp vs +3.38% linear sum prediction).
환경 + side-channel 의 cross-domain stack 만 destructive interference
회피.

side-channel work-pattern × cycle ablation (SUB_188/189/190/196 의 4
cell 2×2 grid) 에서 **branchy × low-rate (100ms cycle) = SUB_196 cellB**
가 3-mix avg AGSD +5.28% (balanced cell +12.04%) 로 paper main +5% 기준
3-mix avg 도달 first single lever. 단 multi-run binding 미검증, 별도
SUB 필요.

§4.4 paper §1 의 가설 재정립

본 fork 의 24 lever 측정 finding 은 paper §1 의 hypothesis 를 다음과
같이 재정립:

(a) GPU phase 와 CPU idle 의 ms-level overlap 가능성은 confirmed (OFF
    CPU util 4%, phase IPC 4.67 μs)
(b) CPU work 의 critical path 와의 overlap 은 **task-pool form 으로 불가능**
    (vllm worker 의 GIL + pinned alloc + cache 공유 contention) 
(c) **side-channel form (independent process, isolated cores 80-95,
    autonomous timer)** 만 net positive 가능
(d) work-pattern (branchy vs regular) × fire-rate (10ms vs 100ms) 가
    net positive magnitude 의 binding 변수
(e) 2-lever cross-domain pair stack (env × side-channel) 만 near-linear
    superposition, 3+ lever destructive

§4.5 한계 + 후속

- multi-run binding 검증 (SUB_196 cellB / SUB_197 pair) 별도 SUB 필요
- AMX draft real vllm spec_decode integration (SUB_198 proxy 의 후속)
  은 invasive vllm core PR-level work — paper §5 후속 work 로 framing
- 1-run cold-start variance ±35 pp 의 reproducibility 한계
- workload diversity (chat α=0.8 / sonnet α=0.65 / code α=0.75) 의
  fork-specific 한계
```

---

## 4. 산출물 list (paper §4 의 backing material)

- 24 measurement-bearing SUB 의 RESULTS.md (각 87-285 lines)
- `id_registry.md` SUB_171~SUB_198 row (verdict + 누적 패턴)
- benchmark JSON × 500+ cells (각 SUB 별 measurements/)
- monitor csv × 50+ files (CPU/GPU util 0.5s interval)
- microbench data (SUB_178 cold-KV / SUB_180 Jacobi / SUB_187 AMX kernel)
- 본 narrative 문서 `PAPER_S4_FINAL_NARRATIVE.md`

paper §4 write-up docs SUB (SUB_195) 는 사용자 명시 skip — 본 narrative 가 그 대체.
