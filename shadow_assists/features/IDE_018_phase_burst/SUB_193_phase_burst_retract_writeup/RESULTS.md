# SUB_193 — IDE_018 phase-burst SUB_169 retract + SUB_184 main lever reject write-up

> **parent**: IDE_018 / TSK_032 (SUB_169 + SUB_184 사후 docs only).
> **scope**: 2026-05-27 14:20 KST.
> **status**: ✅ 완료 — SUB_169 의 +1.35% 결과 retract 명시 + SUB_184 의 main lever reject 정식 write-up.

---

## 0. 두괄식 — phase-burst main lever **공식 reject**

paper §4 main lever IDE_018 phase-burst 의 **공식 verdict**:

| SUB | 결과 | reinterpretation |
|---|---:|---|
| SUB_169 (stub 측정, 2026-05-27 07:50 KST) | 3-mix avg AGSD **+1.35%** | **retract** — IPC overhead 의 noise positive |
| SUB_184 (dummy fill 측정, 2026-05-27 12:14 KST) | 3-mix avg AGSD **−1.75%** | **binding** — main lever core 가설 reject |

**최종 결정**: paper §4 main lever **자격 상실** (verdict 12:14 KST 확정). IDE_018 task pool 영역의 throughput lever 후보 제외, SUB_188 (side-channel +1.84%) 의 small-positive secondary 형태로만 IDE_018 영역 paper bound 가능.

---

## 1. SUB_169 +1.35% retract 의 이유

SUB_169 측정 시 (2026-05-27 07:50 KST):

- ON 모드: `VLLM_USE_PHASE_BURST=1` + 4 `mark_phase` hooks (execute_model / _model_forward / _sample / _bookkeeping_sync)
- **task pool 의 actual fill 없음** — phase signal IPC 만 fire, 실제 CPU work 없음
- 결과: balanced AGSD +3.54% / sonnet +1.94% / code −0.82% / 3-mix avg **+1.35%**
- CPU util OFF 4.08% → ON 5.33% (+1.25 pp, paper target 30% 멀음)

당시 honest assessment: "task pool wiring stub (tasks_executed=0) — TSK_032/033 actual enqueue wiring 별도 turn"

SUB_184 측정 (2026-05-27 12:14 KST):

- ON 모드: SUB_169 의 stub 위에 **task_pool_dummy_fill.cpp** 추가 (8 worker × 8 iters × 8 count, cpu_base=80)
- 결과: 3-mix avg AGSD **−1.75%**, vs SUB_169 stub Δ **−3.10 pp 악화**
- trident-only catastrophic (−14~−20%) — spec decoding 의 dense GPU phase 와 dummy CPU work contention

**reinterpretation**: SUB_169 의 +1.35% 는 phase-mark IPC overhead 4.67 μs × per-step count = 측정 wall 의 ~0.5% 정도 노이즈 + 1-run variance 의 합. **actual task overlap 효과 없음**. SUB_184 가 dummy work 더할 때 throughput 더 깎이는 패턴 = SUB_169 의 noise positive 였음을 거꾸로 입증.

## 2. SUB_184 main lever reject 의 core 가설 분석

paper §4 main lever IDE_018 phase-burst 의 core 가설 (SUB_184 가 검증):

| 가설 | 결과 |
|---|---|
| (a) GPU phase 동안 CPU 가 idle 한 시간이 존재 | confirmed (OFF CPU util 4% 가 그 증거) |
| (b) phase mark IPC 가 충분히 빠름 | confirmed (4.67 μs) |
| (c) CPU work 를 GPU phase 와 동시 실행하면 critical path 영향 없음 | **rejected** ⚠ |
| (d) 결과적으로 CPU util ↑ + throughput 유지 | **rejected** ⚠ |

핵심 reject 사유:
- **trident-only path 가 −14~−20% 회귀** — spec decoding 의 dense GPU phase boundary + CPU work contention
- CPU work 가 vllm worker 의 GIL / pinned alloc lock / cache line 과 contending
- task pool 의 worker thread 와 vllm worker thread 가 Linux scheduler tick 1000 Hz 안에서 inter-process migration / IPI frequency 증가

## 3. SUB_188 side-channel 의 reinterpretation

SUB_184 reject 직후 시도된 SUB_188 (side-channel batch precompute, +1.84%) 가 **IDE_018 영역 첫 양적 net positive**.

차이점:
| 항목 | SUB_184 (task-pool) | SUB_188 (side-channel) |
|---|---|---|
| Trigger mechanism | phase mark callback (sync) | independent 100ms cycle (async) |
| vllm GIL contact | yes (mark_phase Python binding) | no (independent process) |
| pinned alloc share | yes (vllm worker 와 동일 process) | no |
| cache footprint | vllm worker 와 same L1/L2 | cores 80-95 분리 → L1/L2 분리 |
| schedule coupling | per-step mark fire | autonomous timer |
| 결과 | **−1.75%** ⚠ | **+1.84%** ⭐ |

SUB_188 이 SUB_184 의 fail 원인을 **거꾸로 확인** — task-pool 의 critical path 자원 공유 (GIL + pinned alloc + phase IPC) 가 fail root cause. side-channel 의 fully isolated execution 가 net positive 달성 가능 조건.

## 4. paper §4 honest report 형태

본 SUB 가 paper §4 에 들어갈 honest verdict text:

> "**Phase-burst (IDE_018) 의 main lever 자격**: 우리는 GPU prefill/decode phase 사이 CPU idle window 의 존재를 ms-level granularity 로 측정했고 (SUB_167/168), phase signal IPC 의 latency 가 4.67 μs (SUB_169) 로 microsecond 단위에서 충분히 빠른 것을 확인했다. 그러나 본 phase signal 위에 CPU task pool 의 dummy work 를 채워 GPU phase 와 overlap 시키는 SUB_184 측정은 3-mix avg AGSD −1.75% 의 회귀를 보였다. 특히 spec decoding 의 trident path 는 −14~−20% catastrophic regression — vllm worker 와 task pool 의 동일 process 내 GIL / pinned alloc lock / L1/L2 cache 공유에 의한 contention 이 main lever core 가설 (CPU work overlap 으로 critical path 영향 없이 throughput 유지) 을 reject 한다. 본 fork 에서 phase-burst 의 main lever 자격은 **상실**. 단 동일 cycle granularity 의 CPU work 를 **independent process 의 isolated cores 에서 fire** 하는 side-channel 형태 (SUB_188) 로 fork 의 CPU 활용 lever 의 +1.84% small positive 가 가능하다."

## 5. 변경 사항 (in-place patch list)

본 retract 결과를 반영하기 위해 다음 file 들에 retract 표기 추가 권고:

- `shadow_assists/features/IDE_018_phase_burst/SUB_169_canonical_e2e/RESULTS.md` — 상단에 retract note 추가
- `shadow_assists/features/IDE_018_phase_burst/SUB_184_task_pool_fill/RESULTS.md` — main lever reject 표기 (이미 완료)
- `id_registry.md` SUB_169 row — verdict update (이미 SUB_184 row 에 retract 명시됨)
- `spec_decoding/plan/README.md` — IDE_018 paper §4 lever 자격 section 갱신 권고

## 6. 후속

본 SUB 는 docs only — 코드 변경 없음. 후속 measurement SUB 권고:
- **SUB_188 + SUB_190 fully-isolated core split stack** (cores 80-87 softmax + 88-95 tokenize, SUB_191 이 이미 검증 → destructive)
- **side-channel work-pattern × fire-rate ablation 확장** (regular vs branchy, 10ms vs 100ms cycle)
- **AMX draft head real vllm integration** (SUB_187 microbench feasibility 의 e2e)
