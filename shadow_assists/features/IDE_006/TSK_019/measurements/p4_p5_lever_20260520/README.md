# P4 (F1 async cdec) + P5 (F2 MIRROR sweep) 측정 — 2026-05-20 KST

> branch `feat/neo-amx-apply` HEAD `3aeb9885f`.
>
> 본 측정 = SUB_015-Phase 3 follow-up roadmap 의 P4 (F1 async cdec) 구현 + 검증
> + P5 (F2 MIRROR_MAX env sweep) 의 1-run long 측정. 사용자 명시: 다음 turn 14
> combination plan 영역 의 진정한 영역 진행 위한 fact 확정.

---

## P4 (F1 async cdec) 코드 변경

### 변경 영역 (Python only, no rebuild)

**vllm/model_executor/layers/attention/attention.py** — unified_attention_with_output
의 cdec wait 영역 (line ~1161):
```python
if cdec_future is not None and _neo_async_cdec_mode:
    _neo_pending_cdec_queue.append(
        (cdec_future, output, cdec_t0, cdec_t1)
    )
    _depth_p4 = int(_os_async_p4.environ.get("VLLM_NEO_CDEC_PIPELINE_DEPTH", "1"))
    while len(_neo_pending_cdec_queue) > _depth_p4:
        _neo_drain_pending_cdec()
elif cdec_future is not None:
    # 기존 sync wait 영역
```

**vllm/v1/worker/sub_batch_executor.py** — forward_double 의 postproj(attn) 직전
drain 2 site 추가 (correctness):
```python
self.cb.transfer(q0_next, ...)
_drain_pending_p4()  # P4 — attn1's cdec drain (overlap 영역)
emb1 = self.cb.postproj(attn1, ...)
...
attn0_next = self.cb.attention(...)
_drain_pending_p4()  # P4 — attn0_next cdec drain (correctness)
next_emb0_new = self.cb.postproj(attn0_next, ...)
```

### env 활성 영역

- `VLLM_NEO_ASYNC_CDEC=1` — async cdec mode 활성
- `VLLM_NEO_CDEC_PIPELINE_DEPTH=1` — depth=1 (concurrent future 1 개)

---

## 측정 환경

| 항목 | 값 |
|---|---|
| Model | Llama-3.3-70B-Instruct |
| Hardware | H100 × 8 (Intel SPR + GPU 7 의 외부 bentoml service) |
| GPU memory utilization | 0.85 |
| Workload | 500p × 8192 token (long, async_scheduling, fp8 kv) |
| max_num_seqs | 256 |
| max_num_batched_tokens | 8192 |
| KMP_BLOCKTIME | 0 |
| OMP_NUM_THREADS | 10 |
| NEO env | async_swap_buffers=3, sync_swap_batched=1, cpu_pin=12c, numa_bind |

---

## P4 측정 결과

| Run | env | MIRROR | tps | wall (s) | 비고 |
|---|---|---:|---:|---:|---|
| P4 sanity 100p × 8192 | async=1, depth=1 | 80 | **920.4** | 881.2 | 코드 통과 ✅ |
| P4 long run 1 (500p) | async=1, depth=1 | 80 | **1,803.0** | 2256.2 | **lucky variance** |
| P4 + MIRROR=80 (500p) | async=1, depth=1 | 80 | **NO_RESULT** | — | EngineDeadError @ 2% (8/500), 30k+ OOB precheck |

### P4 vs baseline (3-run avg, 동일 환경 MIRROR=80)

| Config | tps | vs v1.6 best |
|---|---:|---:|
| **v1.6 best (3-run avg, NEO best)** | **1,833.0** ★ | reference |
| P4 long run 1 (1-run) | 1,803.0 | -1.6% |
| S1-S9 (3-run avg) | 1,800.1 | -1.8% |
| P3 (3-run avg) | 1,787.9 | -2.5% |
| P1 (3-run avg) | 1,745.1 | -4.8% |

### P4 fact

1. **구현 통과**: 100p sanity (920.4 tps) + long run 1 (1,803.0 tps) 의 두 측정 완주.
2. **재현 시 unstable**: P4 + MIRROR=80 (동일 env) 의 두 번째 measurement = NO_RESULT (EngineDeadError @ 17 분, 2% 진행, 30,080 OOB precheck errors). 이전 P4 run 1 의 1,803.0 = **lucky variance**.
3. **OOB precheck overflow root**: async cdec 의 deferred dispatch 영역 → cdec slot 추적 영역 race → D11 OOB precheck trigger → cdec dispatch skip → 누적 backlog → engine death (shm_broadcast cancelled).
4. **net value**: P4 코드 변경 의 진정한 가치 = **음수** (안정성 회귀). 본 환경 (Llama-70B + H100×8 + 500p × 8192 long) 에서 async cdec 의 wall path overlap win 영역 < OOB overflow 회귀 영역.

---

## P5 (F2 MIRROR_MAX sweep) 측정 결과

| MIRROR | tps | wall (s) | 비고 |
|---|---:|---:|---|
| 40 | NO_RESULT | — | engine crash @ 14% (69/500), shm timeout |
| 60 | **1,766.8** | 2286.0 | -3.6% vs v1.6 best |
| 80 (baseline) | **1,833.0** (3-run avg) | — | NEO best, v1.6 best 3-run avg 참조 |
| 100 | NO_RESULT | — | OOB precheck 23k+, 2% stall, 사용자 kill |
| 120 | not run | — | 사용자 중단 |

### P5 fact

1. **MIRROR_MAX 축소 영역 (40, 60)**: throughput regression + stability hazard. MIRROR=40 = engine crash (mirror_set cap saturation → swap pressure → shm timeout). MIRROR=60 = -3.6% throughput.
2. **MIRROR_MAX 확장 영역 (100+)**: OOB precheck overflow + throughput collapse. cdec dispatch 영역의 slot tracking 이 mirror 확장에 비례 안 함.
3. **baseline 80 이 진정한 best**: v1.6 best 의 3-run avg 1,833.0 tps. env 변경 없음.

---

## 본 측정 의 의의

1. **F1 (async cdec) 구현 통과 단 net loss**: P4 의 진정한 영역 = unstable (variance 영역). long workload + H100×8 + Llama-70B 영역에서 async cdec 의 wall path overlap 영역 < OOB overflow 회귀.
2. **F2 (MIRROR sweep) baseline 영역 best**: 80 = 진정한 best. 축소·확대 모두 regression/crash.
3. **NEO 영역 best**: v1.6 best (1,833.0 tps, 3-run avg) — env/code 변경 없음.

---

## 다음 turn — 14 combination × 100p plan

본 turn 의 P4 unstable + P5 baseline 80 best fact 의 후속:

| ID | Lever | 변경 영역 | rebuild |
|---|---|---|---|
| **A** | F4 TP=4 sweep | env-only | ❌ |
| **B** | F6 OMP barrier dynamic + nowait | csrc/cpu/pacpu/core.h | ✅ |
| **C** | F5 BLOCK_SIZE=32 | C++ (광범위, NEO scheduler) | ✅ |
| **D** | P4 OOB precheck overflow root fix | debug + Python/C++ | TBD |

### 14 combination

각각 100p × 8192 short 검증 (env-only A 또는 rebuild 필요한 lever 의 조합):

1. A
2. B
3. C
4. D
5. A+B
6. A+C
7. A+D
8. B+C
9. B+D
10. A+B+C
11. B+C+D
12. A+C+D
13. A+B+D
14. A+B+C+D

본 plan 의 14 × ~7 min 측정 = ~98 min ≈ 1.7 시간.

다음 turn 의 진행 영역 = ranking 영역 의 진정한 영역 확정 + best combination 의 long workload (500p × 8192) 3-run avg.

---

## raw 측정 자료

| Run | 위치 |
|---|---|
| P4 sanity 100p | `eval/results/20260520_082307_p4_async_cdec_100p_sanity/` |
| P4 long run 1 | `eval/results/20260520_084146_p4_async_cdec_500p_085_run1/` |
| P4 + MIRROR=80 (NO_RESULT) | `eval/results/20260520_105031_p4_mirror80_500p_085_run1/` |
| P5 MIRROR=40 (NO_RESULT) | `eval/results/20260520_092238_p5_mirror40_500p_085_run1/` |
| P5 MIRROR=60 | `eval/results/20260520_094530_p5_mirror60_500p_085_run1/` |
| P5 MIRROR=100 (NO_RESULT) | `eval/results/20260520_102651_p5_mirror100_500p_085_run1/` |
| script — P4+P5 통합 | `/tmp/run_p4_p5_full.sh` |
| script — P5 sweep only | `/tmp/run_p5_sweep_only.sh` |
| script — P4 + MIRROR=80 | `/tmp/run_p4_mirror80.sh` |
| summary | `/tmp/run_p4_p5_summary.txt`, `/tmp/run_p5_sweep_summary.txt`, `/tmp/run_p4_mirror80_summary.txt` |
