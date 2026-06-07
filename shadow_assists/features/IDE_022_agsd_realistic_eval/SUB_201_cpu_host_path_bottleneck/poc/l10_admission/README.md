# L10 — CPU burst-aware admission control

## 위치
- parent: `SUB_201` / `IDE_022`
- sibling levers: `l1_kv_quant`, `l7_oracle_router`, `a2_e2e`, `b1_e2e`, `b3_8gpu_full`, ...
- 본 lever (L10): `poc/l10_admission/`

## 한 줄
FCFS admission 시 짧은 generation 요청이 긴 요청 뒤에 줄서서 막히는 head-of-line blocking 을
**CPU side priority reorder (shortest-job-first, starvation-bounded)** 로 푼다.
GPU 변경 없음, scheduler hot-path 한 곳만 건드린다.

## Patch
`/workspace/host_vllm_hybrid/vllm/v1/core/sched/scheduler.py`
- module-level: `_burst_aware_enabled()`, `_burst_trigger_depth()`, `_burst_head_window()`,
  `_burst_age_cap_s()`  (env-flag cached once)
- `Scheduler._select_waiting_queue_for_scheduling()` 에 burst-aware branch
- `Scheduler._burst_aware_reorder_waiting()` 신규 — FCFS deque head 의 첫 `window`
  candidate 중 `(max_tokens, num_prompt_tokens)` 가 가장 작은 요청을 head 로 swap.
- env flag:
  - `VLLM_BURST_AWARE_ADMISSION=1`  (default off, kill switch)
  - `VLLM_BURST_TRIGGER_DEPTH=4`    (waiting queue size 임계)
  - `VLLM_BURST_HEAD_WINDOW=16`     (검사 window)
  - `VLLM_BURST_AGE_CAP_S=2.0`      (starvation guard: window 안에 이 보다 오래 기다린
    요청이 있으면 그 step 은 strict FCFS 로 폴백)

PRIORITY policy 사용 시에는 reorder 가 트리거되지 않는다(기존 priority 큐가 이미 최적).

## 측정
- HW: B200 1장 (CUDA_VISIBLE_DEVICES=4), TP=1
- 모델: Qwen2.5-7B-Instruct
- workload:
  - sharegpt prompt 400개
  - bimodal max_tokens: 70% short(=64) + 30% long(=2048)
  - 도착 패턴: burst (size 1..32) + Poisson idle gap (mean 0.6s)
- 비교: BASELINE (burst-aware off) vs BURSTAWARE (burst-aware on)
- 지표: TTFT p50/p99, TPOT p50/p99 (overall / short / long 분리)

## 실행
```bash
cd .../poc/l10_admission
./sweep.sh              # 두 case 자동 (baseline → burst-aware)
# 결과 runs/BASELINE.json, runs/BURSTAWARE.json
```

## 산출물
- `unit_test_reorder.py` — head reorder 정확성 단위 테스트 (GPU 불필요)
- `burst_client.py`     — bimodal × burst 클라이언트
- `sweep.sh`            — vLLM boot/bench/stop sweep
- `runs/*.json`         — per-case summary
- `MEASUREMENTS.md`     — 측정 결과 + 결론
