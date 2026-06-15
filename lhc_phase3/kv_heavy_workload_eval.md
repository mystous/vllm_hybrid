# LHC Phase 3 — Task D: KV-heavy workload eval (NEO swap 발화 pilot)

**날짜**: 2026-06-08
**상위**: `lhc_phase3/PHASE3_VERDICT.md`
**산출물**: `lhc_phase3/runs_D/wd1/`, `lhc_phase3/run_kv_heavy_pilot.sh`

---

## 0. TL;DR

W-D1 (Llama-3.1-8B, TP=8, 32K context, 64 prompts × 32 conc, gpu-mem-util=0.18 로 KV 압박 강제) pilot 측정:

| config | NEO swap-out 시도 | swap 성공 | DSA hook 호출 | output tok/s | verdict |
|---|---:|---:|---:|---:|---|
| **vanilla** (NEO off)  | 0   | n/a | 0 | 5759 | KV peak 44.9% — 자연 swap 미발화 (B200 HBM 1.47 TB 여유) |
| **lhc_dsa** (NEO on)   | 8 (8 worker × 1) | **0 (CUDA assert)** | 0 | 0 (crash) | **NEO swap-out → CUDA device-side assert → engine crash** |

→ **Task D FAIL** — NEO scheduler 가 B200 + Llama-3.1-8B TP=8 long-context 워크로드에서 swap-out 시 **CUDA device-side assert** 로 즉시 죽음. DSA hook 은 아예 호출 안 됨 (NEO가 swap을 끝내지 못함).

→ **Task G (통합 sweep) 진행 무의미** — DSA lane 의 consumer (NEO swap) 가 동작 안 함. Phase 3 verdict 에 honest fail report.

---

## 1. W-D1 vanilla pilot 결과

| 메트릭 | 값 |
|---|---:|
| concurrency | 32 |
| num_prompts | 64 |
| sonnet_input_len | 24 000 (target ~28 K KV) |
| sonnet_output_len | 4 096 |
| max_model_len | 32 768 |
| gpu_memory_utilization | 0.18 (B200 184GB × 0.18 ≈ 33 GB KV / GPU) |
| **duration** | 45.1 s |
| **output throughput** | **5 759 tok/s** |
| total_input_tok | 1 405 698 |
| total_output_tok | 259 901 |
| **KV cache usage (peak)** | **44.9%** |
| NEO swap-out (log) | 0 |
| NEO swap-in (log) | 0 |
| DSA `lane ENABLED` workers | 0 |

분석:
- vanilla 모드에서 NEO 자체가 꺼져 있으므로 swap count = 0 은 자명.
- KV peak 44.9% — `gpu-mem-util=0.18` 로 33 GB KV / GPU 로 압박 시켜도 swap 임계치 (보통 100%) 까지 못 감.
- output throughput 5759 tok/s 는 8B 모델 TP=8 + 32 conc 의 정상 수치.

→ **B200 의 1.47 TB HBM 에서 NEO swap 을 자연 발화시키려면 다음 중 하나 필요**:
  - `gpu-mem-util` < 0.10 + max-model-len 65K+ + conc 128+
  - 또는 explicit `--scheduler-config '{"kv_cache_policy":"exclusive"}'` 강제.

본 pilot 은 NEO 가 켜져 있을 때 swap 이 발화하는지 검증이 목적이므로 lhc_dsa 측정에 의존.

---

## 2. W-D1 lhc_dsa (NEO + DSA) pilot 결과 — CRASH

동일 워크로드, `--enable-neo-asymmetric` 추가, `VLLM_LHC_DSA=1` env.

### 2.1 발생한 이벤트
- Application startup complete: 04:33:14
- 첫 request 도착: 04:33:15 (running=6, waiting=26)
- KV usage 6.2% 시점에 NEO scheduler 가 swap-out 시도
- 04:33:24 — **8 worker proc 모두 동시에** `CUDA error: device-side assert triggered` 발생
- request `cmpl-bench-8ecff34b-30-0-be528f27` 의 swap-out 에서 시작 (rollback 시도도 실패)
- engine 전체 abort, bench durations / throughput = 0 (all NaN)

### 2.2 핵심 로그 (`runs_D/wd1/lhc_dsa_boot.log:33:24`)
```
(Worker_TP6 pid=12645) WARNING 06-08 04:33:24 [gpu_model_runner.py:6919]
  [NEO] swap-out: req cmpl-bench-8ecff34b-30-0-be528f27 failed
  (CUDA error: device-side assert triggered

(Worker_TP6 pid=12645) ERROR 06-08 04:33:24 [multiproc_executor.py:977]
  torch.AcceleratorError: CUDA error: device-side assert triggered
```

→ NEO scheduler `swap_out` path 의 **kernel-side 가 sm_100 (B200) + Llama-3.1-8B 의 head_dim=128 / TP=8 분할 KV layout 과 호환 불가**한 것으로 추정. CUDA assert 는 indexing OOB 또는 layout invariant 위반일 가능성.

### 2.3 hook stats
```json
{
  "config": "lhc_dsa",
  "workload": "wd1",
  "neo_swap_out_log_count": 8,       ← 8 worker × 1 시도 (모두 fail)
  "neo_swap_in_log_count": 0,
  "neo_drain_scatter_fail": 0,
  "dsa_lane_enabled_workers": 0,     ← DSA lane 까지 가지도 못함
  "kv_usage_pct_max": 6.2
}
```

DSA lane 은 `dsa_lane_available()` 첫 호출 (= NEO 의 host scatter step 첫 발화) 에서 init 되도록 lazy 디자인. NEO swap-out 이 그 전에 죽었으므로 DSA lane init 자체 미실행 (`dsa_lane_enabled_workers=0` 의 의미).

---

## 3. Task D verdict

| 지표 | 측정 | 게이트 | 결과 |
|---|---:|---:|---|
| NEO swap-out 발화 (lhc_dsa) | 8 attempt | ≥ 10/min | **시도는 됐으나 SUCCESS=0 → FAIL** |
| NEO swap-in (lhc_dsa) | 0 | ≥ 10/min | FAIL (swap-out 단계에서 crash) |
| DSA hook coverage | 0% | ≥ 50% | FAIL |
| engine stability | CUDA assert | no crash | FAIL |

→ **Task D 전체 FAIL — sm_100 (B200) 에서 NEO scheduler swap-out path 가 호환 불가.**

---

## 4. 의미 / 다음 단계

### 4.1 Phase 3 상위 결정
Task D FAIL → **Task G (통합 측정) 진행 안 함**. 이유:
1. lhc_dsa, lhc_amx_c3, lhc_full 모두 NEO scheduler 활성 → 같은 CUDA assert 로 죽을 것 (DSA / AMX 가 problem 의 원인 아님 — NEO swap path 자체).
2. AMX C3 sub-lane (host-side prefix byte scan) 은 NEO 없이도 동작 가능 — 그러나 별도 hook integration 작업 필요 (현 코드는 NEO scheduler 의 cdec_q 안에 fused).
3. Phase 3 의 임무는 **measurement** — Task B/E/C 의 infrastructure 게이트는 통과했고, NEO sm_100 호환성 fix 는 별도 feature 작업 (Phase 4 scope).

### 4.2 Phase 4 권고
1. **TSK_NEO_SM100_FIX**: NEO swap-out CUDA assert 디버그 + sm_100 호환 path 작성
   - `CUDA_LAUNCH_BLOCKING=1` 으로 정확한 assert 지점 추적
   - Llama-3.1-8B head_dim=128 KV scatter kernel 의 index range 검증
   - GH200 (sm_90) 에서 동일 워크로드 PASS 했는지 확인 (Phase 2 가 정확히 그 환경이었음 → 그래서 Phase 2 noise 까지만 갔던 것)
2. **TSK_AMX_C3_STANDALONE**: AMX C3 path 를 NEO 와 무관히 vLLM PrefixCacheBlockHasher 에 직접 hook — Llama-3.1-8B vanilla 환경에서도 측정 가능.
3. Re-attempt Task D + Task G on B200 once NEO sm_100 fix landed.

---

## 5. 산출물
```
lhc_phase3/runs_D/
├── wd1/
│   ├── vanilla.pid + vanilla_boot.log + vanilla_bench.{log,json} + vanilla_hook_stats.json
│   └── lhc_dsa.pid + lhc_dsa_boot.log + lhc_dsa_bench.log + lhc_dsa_hook_stats.json
lhc_phase3/run_kv_heavy_pilot.sh             ← W-D{1,3} pilot launcher (Phase 3 신규)
lhc_phase3/kv_heavy_workload_eval.md         ← 본 문서
```

W-D2 (LoRA churn) 는 NEO 의존 path 에서 동일 crash 예상 → pilot 미실행.

### 5.1 W-D3 (prefix-heavy) 추가 pilot — 확정 reproduce

W-D1 의 CUDA assert 가 sm_100 NEO scheduler universal issue 인지 확인 위해 W-D3 도 lhc_dsa 로 실행. 결과:

| 메트릭 | 값 |
|---|---|
| sonnet_input | 12 000 (shared 8K prefix) |
| concurrency | 32, num_prompts 128 |
| Application startup | OK |
| 첫 NEO swap-out 시도 | 04:38:26 |
| outcome | **동일 CUDA error: device-side assert** (8 worker proc 동시) |
| neo_swap_out log count | 8 (8 worker × 1 attempt) |
| swap success | 0 |
| DSA lane enabled | 0 (NEO 의 hook 까지 도달 못 함) |
| bench output throughput | 0 (crash) |

→ **W-D3 도 동일 fail** — NEO scheduler swap-out path 가 sm_100 + Llama-3.1-8B 에서 발화 자체가 CUDA assert 를 일으킴. 워크로드 종류 (long-context vs prefix-heavy) 와 무관. **kernel-side 인compat 확정**.

NEO swap 미발화 (W-D1 vanilla) 과 swap 발화 즉시 crash (W-D1/W-D3 lhc_dsa) 의 두 시나리오는 **Phase 4 의 LHC_P4_001 (NEO sm_100 fix)** 으로 동시 해결되어야 함.
