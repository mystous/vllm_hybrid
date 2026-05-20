# OOB root fix v1 — neo_scheduler_adapter G/H log rate-limit — 2026-05-20 KST

> branch `feat/neo-amx-apply` HEAD `3129d3900` + uncommitted log rate-limit.
>
> 본 fix = 이전 turn 의 combo sweep 의 진정한 OOB root 영역 추적 결과 적용.
> 사용자 명시: OOB precheck root fix.

---

## 진정한 root 영역 발견

이전 combo 4 retry (A+D TP=4) 의 engine.log.stdout 분석:

| 항목 | 값 |
|---|---|
| **총 INFO log lines** | **1,097,219** (1.09M) |
| **stdout file size** | **214 MB** |
| neo_scheduler_adapter.py:1166 (Plan v4 G/H swap_out attach) | **1,092,271** (99.5%) |
| gpu_model_runner.py:6776 (swap-in done) | 4,564 |
| 나머지 | < 350 |

**root**: scheduler 의 `[Plan v4 G/H] swap_out attach` INFO log 가 매 step fire →
stdout pipe saturation → shm_broadcast 영역 worker blocking → engine death.

---

## fix 코드 변경

**vllm/v1/core/sched/neo_scheduler_adapter.py:1166** 의 logger.info 영역에
rate-limit 추가:

```python
# OOB root fix — 본 INFO log 가 매 step fire 시 1M+ lines 누적
# (combo 4 retry fact). stdout pipe saturation → shm_broadcast 영역
# deadlock root. rate-limit:
#  - 첫 5 회 + 매 1000 회 + cap saturation 상태 변경 시만.
_g_h_cnt = getattr(self, "_neo_gh_log_cnt", 0) + 1
self._neo_gh_log_cnt = _g_h_cnt
_cur_cap_full = len(self._neo_cpu_resident_mirror) >= _MIRROR_MAX
_prev_cap_full = getattr(self, "_neo_gh_log_cap_full", False)
self._neo_gh_log_cap_full = _cur_cap_full
_log_now = (
    _g_h_cnt <= 5
    or _g_h_cnt % 1000 == 0
    or _cur_cap_full != _prev_cap_full
)
if _log_now:
    logger.info(...)
```

---

## 검증 측정 — A-only TP=4 D=off 100p × 8192

이전 combo 3 (동일 config) NO_RESULT 였음 (timeout, 12/100 frozen, 30k+ traceback).

| 항목 | 이전 combo 3 | 본 fix 적용 후 |
|---|---|---|
| **swap_out_attach log count** | 1,092,271 | **~328** (22,755× ↓) |
| **stdout size** | 214 MB | **2.89 MB** (74× ↓) |
| **결과** | NO_RESULT (timeout) | **NO_RESULT (timeout)** ← 동일 |
| Progress | 12/100 frozen | 9/100 frozen (similar) |
| **scheduler call rate** | (측정 X) | **190 calls/sec** (286k @ 25min) |
| GPU util | 0% | 0% |

---

## 진정한 root fix 의 부분 효과

### ✅ 차단 영역
- log spam 영역 완전 차단 (1.09M → 328 lines)
- stdout size 74x 감소 (214 MB → 2.89 MB)
- shm_broadcast saturation 영역 risk 영역 mitigation

### ❌ 잔존 root 영역
- **scheduler 무한 spin loop** = 190 calls/sec, mirror cap=80 saturated 영역 에서 매 step 무진행
- 본 영역 fix X — log 영역 만 차단, primary deadlock 영역 root 잔존
- A=TP4 100p 영역 = NO_RESULT 동일 (timeout)

### 본 fix 의 가치
- 본 fix = **secondary cause (log spam)** 차단 만, **primary cause (scheduler spin)** X
- 즉 OOB precheck root 영역 = 2 단계:
  - **secondary**: log spam → stdout saturation → engine death (본 fix 차단 ✓)
  - **primary**: scheduler hot spin at mirror cap → 진정한 GPU compute 진행 X (별도 fix 필요)

---

## 다음 turn — 진정한 primary root fix 후보

| 후보 | 영역 | 복잡도 |
|---|---|---|
| **A. Scheduler 영역 backoff sleep** | scheduler 가 mirror cap saturation detection 시 brief sleep (1-10ms) 추가 → GPU compute 영역 진행 기회 부여 | 낮음 (1-2 일) |
| **B. swap_in 영역 우선순위 boost** | mirror full 시 swap_in 영역 강제 fire → 빠른 slot 회수 | 중 (3-5 일) |
| **C. Mirror cap dynamic 조정** | swap_in/out ratio 기반 cap 동적 조정 | 중 (3-5 일) |
| **D. cdec dispatch 영역 OOB 영역 사전 차단** | scheduler 영역에서 in-flight swap_out 영역 reqs 를 cdec 후보 영역 에서 제외 | 중-높음 (1-2 주) |

---

## raw 측정 자료

| Run | 위치 |
|---|---|
| OOB fix verify A-only TP=4 100p | `eval/results/20260520_152845_oobfix_a_only_tp4_100p/` |
| 이전 combo 4 retry (root 분석 sample) | `eval/results/20260520_141842_combo_a_d_tp4_retry/` |
| script | `/tmp/run_oobfix_verify.sh` |
| summary | `/tmp/run_oobfix_verify_summary.txt` |
