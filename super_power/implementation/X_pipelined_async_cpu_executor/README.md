# X — Pipelined Async CPU Executor

CPU engine 의 `UniProcExecutor` 를 **async pipelined executor** 로 대체하여 main Python thread 와 compute thread 가 **병렬로** 실행되게 하는 구조 수정.

**B2 drill-down 에서 발견됐지만, 적용 범위는 CPU engine 전반** (workload-agnostic).

---

## 왜 필요한가 (원형 분석, draft/ 에 있음)

1. [`../../draft/B2/20260422_094528_claude_b2_longctx_32b_analysis.md`](../../draft/B2/20260422_094528_claude_b2_longctx_32b_analysis.md) — B2 원형 가설 ("heavy workload 에서 hybrid 유용") 을 실측해 실패 확인 (-96% throughput)
2. [`../../draft/B2/20260422_154222_claude_b2_cpu_parallelism_diagnosis.md`](../../draft/B2/20260422_154222_claude_b2_cpu_parallelism_diagnosis.md) — 실패 원인을 flame graph 로 진단. CPU engine 이 single-master pattern (`cpu0` 99%) 이고, 원인은 `UniProcExecutor` 의 **sync 설계** (main thread 가 `model.forward()` 완료까지 block)

→ **X 가 이 구조를 async 로 바꿔 pipeline 을 가능하게 함**. 이로써 비로소 B2 의 원 가설을 공정하게 재측정할 조건이 만들어짐.

---

## Phase 진행 상태

| Phase | 상태 | 문서 | 코드 변경? |
|---|---|---|---|
| **Phase 1** — 의존성 분석 | ✅ | [`02_phase1_dependency_analysis.md`](02_phase1_dependency_analysis.md) | ❌ (분석만) |
| **Phase 2** — 최소 구현 (compute thread 분리) | ✅ | [`03_phase2_3_impl_and_verification.md`](03_phase2_3_impl_and_verification.md) Part A | ✅ vllm 수정 |
| **Phase 3** — Pipeline 활성화 (1-step lookahead) | ✅ (Phase 2 와 통합) | 同上 | core.py 의 `step_with_batch_queue` 재활용 |
| **Phase 3 검증** — Flame graph 에서 worker thread 확인 | ✅ | [`03_...`](03_phase2_3_impl_and_verification.md) Part B | ❌ |
| **Phase 4 tool** — sync vs async 비교 script | ✅ | [`04_phase4_compare_tool.md`](04_phase4_compare_tool.md) | ❌ (tool 만) |
| **Phase 4 실측** — light workload sync/async 비교 | 🟡 대기 | 서버 실행 필요 | — |
| **Phase 5 측정** — 정량 성능 숫자 | 🟡 대기 | Phase 4 실측 결과가 입력 | — |

---

## 전체 설계 — 01_design_and_plan.md

[`01_design_and_plan.md`](01_design_and_plan.md) 가 X 의 **설계 전체 진실 공급원**. 포함 내용:

- **Part I · 맥락** — 이전 분석 요약, sync executor 의 구조적 한계
- **Part II · X 설계** — 아키텍처 / 컴포넌트 / thread 책임 / dependency / interface 호환 / 기대 효과 / 위험 (§3, 본 문서의 핵심)
- **Part III · 실행** — Phase 1~5 plan, 의사결정 포인트, 대안 경로

---

## 구현된 코드

- `vllm/v1/executor/cpu_pipelined_executor.py` (신규) — `PipelinedCPUExecutor(UniProcExecutor)` 클래스. `max_concurrent_batches=2` 로 `EngineCore.step_with_batch_queue` 활성.
- `vllm/v1/engine/hybrid_core.py` (수정) — `HYBRID_CPU_ASYNC_EXECUTOR=1` flag 감지 시 새 executor 사용.
- `vllm/v1/worker/cpu_worker.py` — 변경 없음 (초안 변경은 원복).
- `vllm/v1/engine/core.py` — 무수정 (CLAUDE.md 원칙 유지).

## 실행 tool

- `eval/diagnostics/b2_cpu_parallel/run_compare.sh` — Phase 4 sync/async 비교
- `eval/diagnostics/b2_cpu_parallel/run_all.sh` — Phase 3 snapshot 포함 full 실행
- `eval/diagnostics/b2_cpu_parallel/g0_h100x8_qwen32b_light_trace.env` — light workload env

---

## 다음 단계 (Phase 4 실측)

```bash
rm -rf eval/diagnostics/b2_cpu_parallel/results/compare_*/
pkill -9 -f 'api_server|serve\.sh|CPU_EngineCore|GPU_EngineCore|benchmark_serving' 2>/dev/null; sleep 3
git pull
bash eval/diagnostics/b2_cpu_parallel/run_compare.sh
git add eval/diagnostics/b2_cpu_parallel/results/
git commit -m "X Phase 4: sync vs async 비교"
git push
```

결과가 `COMPARE_REPORT.md` 로 자동 생성됨. speedup 수치에 따라:

- **≥ 40%**: X 완료, B2 heavy workload 재측정으로 이동
- **10~40%**: micro fix (A-1, E-1 등) 스택 가능성 검토
- **< 10%**: MultiProc 방향 재검토

---

## Phase 5 이후

Phase 5 통과 = "X 가 구현되고 light 에서 효과 측정됨". **B2 의 원 질문 "heavy hybrid 가 유용한가" 는 별도 재측정** 필요.

1. X + (선택적) micro fix 스택
2. Heavy workload (16K/16K) 로 B2 재측정
3. 결과에 따라 B1/B2/B3 전략 재정렬
