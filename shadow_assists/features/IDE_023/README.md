# IDE_023 — MoE Expert Offload Hybrid (KTransformers 계열, AMX)

> parent: `PLN_003` (캠페인 플랜: [`PLN_003.md`](PLN_003.md)) / 진행 로그: [`PROGRESS_20260827.md`](PROGRESS_20260827.md)
> 관련: `TSK_043` (feasibility) / `TST_021` (검증 게이트)

## 1. 이론적 배경

### 1.1 왜 이 구조만 이 머신에서 성립하는가

본 저장소의 1~4세대 hybrid 실패 (NinjaGap CPU engine, X async executor, IDE_006 cold-KV partial attention, NEO port) 의 공통 원인은 **GPU가 스스로 할 수 있는 일을 CPU에 옮긴 것**이다. violet-h100-016 의 자원 비율 (메모리 BW 1:44, 연산 1:34~1:100) 에서 이는 구조적으로 손해다.

MoE expert offload 는 반대 방향이다:

- **대상 regime**: 모델 weights + KV 가 HBM 총량 (H100 80GB × 8 = 640GB) 을 **초과**하는 MoE — DeepSeek-R1-0528 FP8 (~688GB), Kimi K2 1T급. 이 클래스는 **GPU-only 서빙 자체가 불가능** → "vanilla 가 항상 이긴다" (SUB_036/042) 결론이 적용되지 않는 유일한 영역.
- **분업**: attention + KV cache + shared experts = GPU (dense, compute/BW-intensive) / sparse routed experts = **CPU 가 자기 DRAM 의 weight 를 AMX GEMM 으로 직접 소비**. 전송이 아니라 in-place 연산이므로 PCIe 병목과 IDE_006 의 Q-dependency dilemma 모두 무관.
- **CPU 활용**: OSDI'26 Expert Deferral 계열은 CPU 활용률 75→~100% 를 보고 — CLAUDE.md "CPU idle 불허" 목표와 정합하는 유일한 검증된 구조.

### 1.2 선행 시스템

| 시스템 | 결과 | 링크 |
|---|---|---|
| KTransformers (SOSP'25) | prefill 4.62–19.74× / decode 1.25–4.09× vs 기존 offload. AMX 특화 kernel | arXiv/ACM 3731569.3764843 |
| SGLang + KT 통합 (2025-10) | hot expert GPU / cold expert CPU + multi-GPU TP | lmsys blog, sglang#11425 |
| 8×L20 + Xeon 실측 | DeepSeek-R1 671B 227 tok/s total | kvcache-ai/ktransformers |
| OSDI'26 local MoE SLO | Expert Deferral 로 CPU util ~100% | arXiv 2606.10493 |

## 2. 구현 방향

Phase 1 (본 TSK_043, fork 수정 0줄): SGLang+KT 컨테이너로 feasibility — 서빙 성립 + tok/s + util.
Phase 2 (조건부): vLLM fork native 통합 — vLLM upstream 에 없는 영역, 독자 기여 후보. Phase 1 수치 확인 후에만 진입.

## 3. 하드웨어 전제 (실측 확인 2026-08-27)

- Xeon 8480+ ×2 (112C, **amx_bf16/amx_int8/avx512_bf16 native**), DDR5 2TB (이론 ~614GB/s), H100 80GB ×8 NVLink full-mesh
- **주의**: `intel_pstate/no_turbo=1` — CPU 2.0GHz base 고정 (turbo unlock 은 사용자 결정 대기)
- GPU 4/4 가 socket 0/1 분할 — expert 연산 NUMA 배치 필요
