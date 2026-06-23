# SUB_260 Phase B0 — kill-gate 측정 (분산-EP 축소-expert self-spec)

*2026-06-17, DeepSeek-R1 671B EP8(8×B200), eager. 신규성 사활 판정.*

## 측정 1 — Acceptance a (축소 top-k argmax vs full top-8 argmax)
방법: 동일 고정 텍스트 4종을 teacher-force(`prompt_logprobs=20`)해 각 위치 argmax 수집,
top-8 baseline 서버와 force-topk 서버에서 per-position 일치율. 224 위치.
- ENV-gated hook: `base_router.select_experts`에 `VLLM_MOE_FORCE_TOPK` 추가(routing 후 상위 N만+renorm, 미설정 no-op).

| draft top-k | acceptance a | 비고 |
|---:|---:|---|
| top-1 | **0.478** | Mixtral top-1(0.819)보다 훨씬 낮음 |
| top-2 | **0.804** | spec 성립 가능 수준 |
| top-8 | 1.000 | baseline |

**해석**: R1은 fine-grained 256-expert top-8 + shared-expert라 top-1 축소 시 신호 손실 큼(a=0.478).
top-2라야 Mixtral top-1급 a=0.804. → **draft는 최소 top-2 필요.**

## 측정 2 — FLOPs-only 경제성 (통신 절감 가정 전)
R1 GPU 시간 분해(SUB_257): expert GEMM 20% / all-to-all 통신 31% / quant 13% / 기타.
top-k 축소는 **expert GEMM·통신·quant만** 줄임(attention·norm·dense는 draft도 full).
- top-2 draft (6/8 expert 감소): expert GEMM 0.75×20%=15% 절감. 통신·quant 절감은 **통신 스케일링 가정에 의존**.
  - **통신 절감 無(slot-padding 시)**: c_d ≈ 1−0.15−0.10 = 0.75 → 1/(c+(1−a)) = 1/(0.75+0.196) = **1.06×** (marginal, = 비신규 SS-MoE 영역).
  - **통신 6/8 절감 有(best case)**: c_d ≈ 1−0.15−0.23−0.10 = 0.52 → 1/(0.52+0.196) = **1.40×**.
- top-1 draft: 통신無 0.81×(**net-negative**), 통신有 best 1.04×.

## 판정 (중간)
- **acceptance 게이트**: top-2 PASS(a=0.804), top-1 약함.
- **결정 변수 = 통신 스케일링**(B0 측정 #1 미완): 축소 top-k가 DeepEP all-to-all을 실제로 줄이는가.
  - 줄이면 → top-2에서 ~1.4× 가능, **분산 통신 절감이 신규 기여**.
  - 안 줄이면(LL slot-padding) → ~1.06× marginal = FLOPs만 = SS-MoE 비신규 영역 → **fallback**.
- **다음(필수)**: nsys로 top-8 vs top-2 EP all-to-all(dispatch/combine) 커널시간 직접 측정.
  DeepEP HT(contiguous) vs LL(slot-padding) 모드별. ≥15% 감소면 통신-신규 진행.

## 측정 3 — 통신 스케일링 nsys (★ 결정적, config 의존성 발견)
R1 **TP8+EP8** eager, nsys cuda_gpu_kern_sum (지속 decode 부하, 25s 캡처):
| 커널 | GPU 시간% | 정체 |
|---|---:|---|
| `cross_device_reduce_2stage<bf16,8>` | **72.9%** | **attention/dense의 TP all-reduce** |
| deep_gemm fp8 (expert+dense GEMM 다수) | ~12% | expert/dense GEMM |
| per_token_group_quant | ~2.4% | FP8 quant |
| `_fwd_kernel_ep_scatter/gather` + `_count_expert_num_tokens` | **~1%** | **EP all-to-all dispatch/combine** |
| attention(MLA fmha) | ~1.1% | |
(eager라 TP all-reduce가 cudagraph보다 부풀려짐 — SUB_257 cudagraph 31% 대비. 단 **비율 구조**는 동일.)

**결정적 발견**: TP8+EP8 config에서 **지배 통신 = TP all-reduce(attention/dense, 73%)**, EP all-to-all(experts)은
**~1%뿐**. 축소-expert draft는 EP dispatch만 줄이고 **TP all-reduce는 full로 지불** → 이 config에선 통신 신규성 死.
→ draft가 줄일 수 있는 건 expert GEMM(~소수%)+EP dispatch(~1%)뿐 = FLOPs-only ≈ 비신규.

**그러나 config 의존**: DeepSeek 프로덕션 권장 = **DP-attention + EP**(`--data-parallel-size 8 --enable-expert-parallel`,
TP 없음). 이 config는 attention이 DP(replicated)라 **TP all-reduce 자체가 없고**, 지배 통신 = **EP all-to-all**.
거기선 top-8→top-2 축소가 지배 통신을 직접 줄임(1/4 dispatch volume, slot-padding 제외) → 신규성 부활 가능.

## 판정 (B0 갱신)
- acceptance: top-2 a=0.804 PASS.
- 통신: **TP8+EP8 死**(TP all-reduce 73% 못건드림). **DP-attention+EP에서만 후보 생존** — 미측정.
- **다음(필수)**: R1 `--data-parallel-size 8 --enable-expert-parallel`(DP-attn+EP)로 (1) EP all-to-all이 지배
  통신인지 nsys 확인, (2) top-8 vs top-2 EP all-to-all 커널시간 감소 측정. ≥15%면 신규 진행, 아니면 fallback.

## 측정 4 — DP-attention+EP go/no-go = ⚠️ 환경 블로커 (2026-06-17)
생존 경로(DP-attn+EP, EP all-to-all 지배)에서 top-8 vs top-2 통신 측정 시도. **부팅 실패**:
- R1 `--data-parallel-size 8 --enable-expert-parallel --enforce-eager` → **NCCL "unhandled cuda error"(DP7 rank)** 재현성 있게 실패. all2all-backend(default/naive) 무관, gpu-util 0.88/0.90 무관, GPU clean 상태서도 동일.
- nsys-DP는 추가로 **엔진 프로세스 re-parent로 추적 탈출** + 실패 후 워커가 GPU 125GB×8 점유(SIGKILL 필요).
- 대조: SR-005의 DP8은 **dense 70B(EP 없음)** 라 성공 → **DP+EP(expert-parallel × data-parallel) communicator init**이 이 머신서 실패하는 특정 경로.

**상태**: 생존 경로의 go/no-go **미해결(블로커)**. 통신 신규성은 死도 生도 아닌 측정 불가 상태.
**다음 옵션**: (a) DP+EP NCCL 디버그(NCCL_DEBUG=INFO, NCCL_P2P/CUMEM env, IPC/cudagraph 상호작용), (b) TP+EP에서 top-8 vs top-2 **EP-커널(scatter/gather/allgather) 시간 스케일링** 프록시 측정(부팅 안정, EP comm은 ~1%지만 비율로 volume 스케일 판정), (c) K-EXAONE-236B로 DP+EP 재시도.

## 측정 5 — EP-커널 top-k 스케일링 (★ 결정적, 통신 신규성 NO-GO)
DP+EP 블로커 우회: TP8+EP8(부팅 안정)서 top-8 vs top-2 nsys, EP 커널 시간 비교.
| 커널 | top-8 (ns) | top-2 (ns) | 비율 | 종류 |
|---|---:|---:|---:|---|
| `ncclDevKernel_AllGather_RING` | 345.2M | 326.6M | **0.95×** | **inter-GPU 통신(실제)** |
| `_fwd_kernel_ep_gather` | 647.5M | 383.8M | 0.59× | local permute(compute) |
| `_fwd_kernel_ep_scatter_2` | 692.2M | 339.1M | 0.49× | local permute(compute) |
| `_count_expert_num_tokens` | 237.1M | 211.2M | 0.89× | local |
| `cross_device_reduce`(TP AR) | 125,843M | ~동일 | ~1.0× | TP 통신(지배) |

**결정적**: 실제 **inter-GPU EP 통신(AllGather)은 top_k와 무관(0.95×)**. 이유 = vLLM 기본 all2all 백엔드
`allgather_reducescatter`는 **모든 토큰을 all-gather**(각 rank가 전부 받음) 후 로컬에서 자기 expert만 계산 →
통신량이 활성 expert 수(top_k)와 **독립**. top-k 축소가 줄이는 건 **local permute 커널(0.5×, compute)뿐**, 통신 아님.

**→ 통신 신규성 NO-GO (기본 EP 경로)**. 축소-expert draft는 inter-GPU 통신을 못 줄임.
- 유일 잔존 가능성 = **DeepEP true-all2all 백엔드**(token을 선택 expert 보유 rank로만 dispatch → top_k 비례).
  단 (a) DP+EP config 자체가 NCCL 버그로 부팅 불가(측정 4), (b) DeepEP LL slot-padding이 비례 깰 수 있음(Plan agent),
  (c) DeepEP 설정/가용성 미검증. 조건 3중 스택 → 실현성 낮음.

## B0 최종 판정 (kill-gate)
- acceptance: top-2 a=0.804 PASS / top-1 0.478.
- **통신 신규성(핵심 차별점): NO-GO.** (i) TP+EP는 TP all-reduce 지배(못건드림), (ii) DP+EP는 부팅 블로커,
  (iii) 기본 EP all-to-all(allgather)은 top_k 무관(통신 0.95×). DeepEP 경로만 이론상 잔존하나 3중 조건.
- FLOPs-only 절감(expert GEMM+local permute)은 ~1.06×(top-2) marginal = **비신규 SS-MoE 영역**.
- **결론: 분산-EP 축소-expert self-spec의 "통신 동시절감" 신규 주장은 본 스택서 실현 불가 → 논문 핵심 claim 불성립.**
  → **fallback**: (1) 특성화("MoE EP 통신은 all2all 백엔드에 좌우 — allgather는 top_k 불변, DeepEP만 비례; reduced-expert
  draft의 통신이득 조건 정량화") 또는 (2) 후보#2 replica-affinity routing(출력동등·저위험)/#3 SPD-on-MoE.

## 산출물
- `accept_bench.py`, `nsys_comm.sh`, `runs/argmax_top{8,1,2}.json`, `runs/comm_top8.nsys-rep`, `runs/serve_*.log`.
- hook: `vllm/model_executor/layers/fused_moe/router/base_router.py` (`_FORCE_MOE_TOPK`).
