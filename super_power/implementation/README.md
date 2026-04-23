# Implementation Docs

**실제 vllm 코드 변경을 수반하는 작업의 설계 + 진행 기록.**

원형 가설 / 분석 / 진단 문서는 [`../draft/`](../draft/) 참조. 이 디렉토리는 "**그 분석들을 근거로 실제로 코드를 고치는 일**" 의 문서.

---

## 현재 진행 중인 작업

| 작업 | 상태 | 디렉토리 |
|---|---|---|
| **X — Pipelined Async CPU Executor** | Phase 2+3 완료 · Phase 4+5 실측 대기 | [`X_pipelined_async_cpu_executor/`](X_pipelined_async_cpu_executor/) |

---

## 체계 구분 — 세 가지 축을 혼동하지 말 것

### 축 1 — Breakthrough 가설 (draft/ 에 있음)
시스템 성능 개선의 전략 방향.

| 코드 | 이름 | 상태 |
|---|---|---|
| B1 | Inverted Control Plane (CPU 를 critical path 에서 빼기) | X 로 구조적 부분 달성 중 |
| B2 | Heavy workload CPU decode shadow | 원형 실측 실패 (-96%), X 적용 후 재측정 예정 |
| B3 | Meta-scheduling Gateway | X 와 독립, 미착수 |

→ draft/ 의 094528 / 154222 참조.

### 축 2 — X (이 디렉토리)
B2 실패의 구조적 원인 (sync executor) 을 해결하는 **아키텍처 수정**. B2 가설을 테스트 가능한 조건으로 만들기 위한 선행 작업.

구현 단계는 Phase 0~5 (170451 §4 에서 정의).

### 축 3 — Micro fix (아직 구현 안 됨)
X 위에 쌓는 작은 최적화. Phase 5 측정 결과에 따라 필요 여부 결정.

| 코드 | 내용 | 상태 |
|---|---|---|
| A-1 | CPU engine `--enable-prefix-caching` 비활성화 | 후보 |
| A-2 | `find_longest_cache_hit` C++ 이전 | 후보 |
| B-1 | `CPUModelRunner._update_states` override | 후보 |
| B-2 | `block_ids` append loop C++ | 후보 |
| C-1 | `torch.no_grad` hoisting | 후보 |
| C-2 | `torch.inference_mode()` 교체 | 후보 |
| D-1 | scheduler CPU fast path | 후보 |
| D-2 | scheduler 전체 C++ | 후보 |
| E-1 | `_hybrid_profile_enabled` 캐시 | 후보 |

주의: "B-1" (micro fix) 과 "B1" (가설) 은 **알파벳이 같을 뿐 별개 개념**.

---

## 관련 위치

- `../draft/` — 가설 / 분석 / 진단 (X 를 도출한 intellectual work)
- `../../measurement_results/` — 실측 데이터 (bench 결과, flame graph)
- `../../eval/diagnostics/b2_cpu_parallel/` — 실행 tool 과 raw results
- `../../docs/paper/main.tex` — 최종 논문 (향후 X 결과 반영 예정)
