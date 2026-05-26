# CLAUDE.md — IDE_018 구현 시 알아야 할 것

## 0. 핵심 규칙
- **paper main contribution** — 본 IDE 의 measurement 가 paper §4 Figure 5
- vLLM 의 forward path 에 직접 hook — 매우 invasive, careful design
- CUDA event timestamp 의 정확도 (typically μs 단위, 10-50 μs jitter)
- CPU task pool 은 IDE_020 의 cgroup cpuset (cpu 80-99) 에서 실행
- 측정 시간 KST 표시 (`TZ=Asia/Seoul date`)
- 측정 default 1-run
- commit/push 명시 지시 시만

## 1. Phase A 측정 결과 input

| Phase A finding | IDE_018 의 입력 |
|---|---|
| VLLM threads 96% S (SUB_162) | absolute CPU idle window 존재 입증 |
| 10.24 TFLOPS available (SUB_117) | task pool 의 capacity |
| sampler.py 44.3% (SUB_161) | sampling-phase task |
| paper Table 1a/1b (SUB_167/168) | 10 task × 5 phase dispatch matrix |
| DMA 35 μs / 54 GB/s (SUB_166) | linear-phase task data plane |
| VLLM thread default full-mask (SUB_148) | OS-coordination 가능 |

## 2. 통합 위치 (vLLM)

- `vllm/v1/worker/gpu_model_runner.py` — forward 의 phase boundary 에 CUDA event 삽입
- `vllm/v1/engine/core.py` — engine main loop inter-step
- patch 방식: vLLM plugin entry point 또는 monkey-patch
- ENV `VLLM_USE_PHASE_BURST=1` 으로 activate

## 3. risk

| risk | severity | fallback |
|---|---|---|
| CUDA event hook 의 overhead 가 net gain 잠식 | HIGH (paper §6 listed) | per-batch level fallback (per-step → per-batch) |
| phase signal jitter (50 μs target) | medium | granularity batch level fallback |
| CPU task 가 GPU dispatch 와 충돌 (sched) | medium | IDE_020 의 cgroup isolation 필수 |
| task pool 의 starvation (long-running task blocks short task) | medium | task priority queue + preemption |

## 4. 검증 게이트

- per-token logprob max abs diff < 1e-3 (CLAUDE.md 운영 해석)
- paper §4: CPU util 4.1% → **30%+** target (SUB_117 의 16% 위에 추가 14pp)
- paper §4: throughput +10-20% (목표 — TSK_025+026 의 lever combine)
