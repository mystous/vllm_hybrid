# 특성화 논문 개요 — Why Routing-Side Communication Optimizations Fail in Production MoE Serving

*consolidation, 2026-06-17. 모든 수치 = 본 세션 실측(8×B200, vLLM). 신규 알고리즘이 아닌 systems characterization.*

## Thesis
8×B200에서 대형 MoE(DeepSeek-R1 671B, K-EXAONE-236B)를 프로덕션 스택(vLLM)으로 서빙할 때, **통신이
지배 병목**이지만 **"라우팅을 더 똑똑하게 해서 통신을 줄인다"는 알고리즘 군(軍)은 구조적으로 실패**한다.
그 이유를 측정으로 규명하고, 통신 최적화가 알고리즘이 아니라 **all-to-all 백엔드와 병렬화 config 선택**의
문제임을 보인다.

## 측정된 기여 (전부 실측)
### C1. 프로파일링 방법론: flamegraph ≠ GPU 병목 (SUB_256/257)
- all_reduce가 CPU 스레드 점유 6% vs **GPU 커널시간 31~73%**(config별). 통신 비중은 nsys
  `cuda_gpu_kern_sum`으로만 정확. cudagraph가 launch를 GPU로 내려 CPU 프로파일에 안 보임.

### C2. 지배 통신은 EP all-to-all이 아니라 TP all-reduce (SUB_259/260)
- R1 TP8+EP8 eager nsys: `cross_device_reduce`(attention/dense **TP all-reduce**) = **GPU 72.9%**,
  EP all-to-all(`ep_scatter/gather`+allgather) = **~1%**. 직관과 반대 — MoE인데 통신 병목은 expert
  통신이 아니라 attention의 TP all-reduce. (cudagraph서 31%로 완화되나 여전히 최대 단일 항.)

### C3. 기본 EP all-to-all은 routing-invariant (SUB_260, ★핵심)
- vLLM 기본 백엔드 `allgather_reducescatter`는 **모든 토큰을 all-gather** → 각 rank가 로컬 expert만 계산.
  따라서 inter-GPU 통신량 = (전 토큰 × hidden), **라우팅 결정과 독립**.
- **실측 증명**: top-8 vs top-2 강제 → `ncclAllGather` 시간 **0.95×** (불변). 줄어든 건 local permute
  커널(`ep_gather` 0.59×, `ep_scatter` 0.49×, = compute)뿐.
- **귀결**: 라우팅측 통신 최적화 2종이 모두 무력:
  - **축소-expert(top-k↓)**: 통신 불변 → self-spec의 "통신 동시절감" 주장 불성립.
  - **replica-affinity(nearest replica)**: allgather라 복제본 위치는 load만 바꿈, 통신 불변
    (vLLM 복제본 선택은 `base_router.py:46` random `offs%count`; 토폴로지 무관이나 고쳐도 통신 무이득).
- 통신이 routing-dependent해지려면 **DeepEP true-all2all** 필요. 단 (a) DP+EP config가 NCCL
  "unhandled cuda error"로 부팅 불가(측정4), (b) DeepEP LL slot-padding, (c) 미검증 — 3중 조건.

### C4. fits-on-GPU 임계 + reduced-expert acceptance 곡선 (SUB_257/260)
- 70B FP4(40GB)는 1 GPU 적재 → DP8로 TP all-reduce 통째 제거(+182%, SR-005). 대형(405B/671B)은
  불가 → TP/EP 통신 환원불가.
- R1 reduced-expert draft acceptance(teacher-force argmax 일치): top-1 **0.478**, top-2 **0.804**
  (Mixtral top-1 0.819 대비; fine-grained 256-expert는 top-1 손실 큼). FLOPs-only 속도배율 ~1.06×(top-2)
  = SS-MoE류 비신규 영역.

## 핵심 메시지 (so-what)
MoE 추론의 통신 최적화 문헌(reduced-expert spec, affinity routing 등)이 **프로덕션 스택에서 안 통하는
구조적 이유**: (1) 지배 통신이 expert 통신이 아니라 TP all-reduce, (2) 기본 all-to-all이 routing-invariant.
통신 이득은 알고리즘이 아니라 **백엔드(DeepEP)·config(DP-attention) 선택**에서 나오며, 그 경로조차
프로덕션서 부팅/패딩 문제로 막혀 있다. → "smart routing saves comm"은 dominant 스택에선 미신.

## Max-config baseline (SUB_259, 재현 기준)
R1 EP8 모든 성능 플래그 ON(O3 FaP cudagraph + flashinfer-trtllm AR + deep_gemm + symm_mem):
c64 **2,365 gen_tps**. (함정: `VLLM_USE_FLASHINFER_MOE_FP8`은 R1 block-FP8 미지원 BOOT_FAIL.)

### C5. SPD-on-MoE 실측 — 엄격 게이트 하 NO-GO (SUB_260 #3, faithful 구현)
TP all-reduce(지배 통신)를 drop하는 유일 신규 후보. `deepseek_v2.py` MLA o_proj에 ENV-gated
`reduce_results=False` 구현(8 TP워커 발동 확인). **단일-최후-layer(60) drop조차 게이트 위반**:
max_logprob_diff **1.288**(≤0.5), ppl_rel **0.199**(≤0.1) → FAIL. TP8서 o_proj AR drop = 그 layer
attention의 7/8 손실이라 본질적 lossy → drop 가능 layer 0개. **분포동등을 요구하면 sync-drop류 통신제거는 불가.**
(SPD 논문 "<1% acc"는 본 게이트보다 훨씬 느슨; 정확도-동등 정의가 결과를 가른다.)

## 산출물 (재현)
- nsys: `SUB_257/runs/r1cg_prof.nsys-rep`, `SUB_260/runs/comm_top8.nsys-rep`, `comm_top2.nsys-rep`.
- 측정: `SUB_259/BASELINE.md`, `SUB_260/{B0_FINDINGS,CANDIDATE_TRIAGE}.md`, `accept_bench.py`, `nsys_comm*.sh`.
- hook: `base_router.py` `VLLM_MOE_FORCE_TOPK`(측정용). SR-005(DP8), SR-001~006.
