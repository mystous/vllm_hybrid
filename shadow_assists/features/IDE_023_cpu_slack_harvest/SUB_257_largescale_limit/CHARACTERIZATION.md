# Characterizing Large-Model Serving Limits on 8×B200
*측정 연구 (systems characterization), 2026-06-17. 모든 수치 = 본 세션 실측.*

## Abstract
8×NVIDIA B200(NVLink5/NVSwitch, 183GB/GPU)에서 LLM 서빙의 성능 한계를 dense(70B/405B) 및
대형 MoE(DeepSeek-R1 671B, K-EXAONE-236B)에 대해 측정한다. 세 가지를 보인다: (1) **프로파일링
방법론**: CPU-스레드 프로파일(flamegraph)은 GPU 병목을 심각하게 왜곡한다 — all_reduce가 CPU 스레드
점유율로는 6%지만 GPU 커널시간(nsys)으로는 31–50%다. (2) **fits-on-GPU 임계**: 모델이 1 GPU에
들어가면(70B FP4=40GB) 데이터병렬이 TP 통신을 제거해 +182% throughput; 안 들어가면(405B/671B) TP
통신이 불가피한 한계가 된다. (3) **표준기법 near-optimality**: multimem-PTX all-reduce, EPLB,
aux-loss-free 균형, FP8/FP4가 모두 upstream에 구현돼 있어 대형모델 병목은 이미 고도 최적화돼 있다.
신규 알고리즘 시도(저정밀 통신, FP8 residual, self-speculation)가 왜 막히는지 그 구조적 이유를 정량화한다.

## 1. 측정 방법론 (기여 1): flamegraph ≠ GPU 병목
| 구성(70B FP4 TP8) | all_reduce 비중 | 출처 |
|---|---:|---|
| flamegraph (py-spy, CPU 스레드 점유율) | **6%** | 워커 스레드 |
| **nsys cuda_gpu_kern_sum (GPU 타임라인)** | **50.3%** | GPU 커널시간 |
**교훈**: 통신/병목 비중은 CPU 스레드가 아니라 GPU 커널 타임라인으로 측정해야 한다. 6%로 보고
무시했으면 가장 큰 병목을 놓친다. (flamegraph는 cudagraph가 launch를 GPU로 내려 CPU엔 안 보임.)

## 2. fits-on-GPU 임계 (기여 2)
| 모델 | FP 크기 | 1 GPU(183GB) 적재 | 최적 병렬화 | throughput |
|---|---:|---|---|---|
| 70B | FP4 40GB | ✅ | **DP8 (TP1×8, 통신0)** | TP8 대비 **+181.8%** (5393→15196) |
| 405B | FP8 405GB | ❌ (TP4 최소) | TP4×DP2 (통신 절반) | 특성화 (통신-bound 확정) |
| R1-671B MoE | FP8 671GB | ❌ | EP8 (통신 불가피) | 1725 gen_tps |
| K-EXAONE-236B MoE | FP8 245GB | ❌ | EP8 | 1745 gen_tps |
**핵심**: TP all-reduce는 텐서병렬에 본질적. 모델이 1 GPU에 들어가면 복제(DP)로 통신을 통째 제거
가능(70B +182%)하나, 대형모델은 안 들어가 TP/EP 통신이 환원불가 한계가 된다.

## 3. 대형모델 병목 분해 (nsys, cudagraph)
**R1-671B EP8 decode** (conc32):
| 커널 카테고리 | GPU 시간% |
|---|---:|
| all-reduce (multimem cross_device_reduce) + allgather | **~31%** |
| expert GEMM (FP8 bmm) | ~20% |
| MoE activation/finalize/routing | ~11% |
| FP8 양자화 (per_token_group_quant) | ~13% |
| RMSNorm+quant | ~5% |
| attention (MLA flash) | ~2% |
**K-EXAONE-236B가 일반화**: cross_device_reduce 동일 지배(eager 81.6% vs R1 81.1%). 두 대형 MoE가
동일 구조 = 통신-bound, 균형잡힌 시스템. (eager 모드는 launch 오버헤드로 81%, cudagraph는 31%.)

## 4. 표준기법 near-optimality (왜 novel이 막히나)
| 표준기법 | upstream 위치 | 효과 |
|---|---|---|
| multimem-PTX all-reduce | PyTorch symm_mem (B200 NVLink5 in-fabric) | decode AR 20µs (NCCL 46µs 대비 −44%) |
| EPLB (Expert Parallel Load Balancing) | vllm/distributed/eplb/ | expert 불균형 동적 재배치 |
| aux-loss-free 균형 (noaux_tc) | R1 학습 | expert 부하 설계상 균형 |
| FP8/FP4 텐서코어 | compressed-tensors | byte-level 가속 |
| AR+RMSNorm+FP4quant 융합 | flashinfer allreduce_fusion | 커널 융합 |

## 5. Negative results — 신규 시도가 왜 막히나 (정량)
| 시도 | 결과 | 구조적 이유 |
|---|---|---|
| FP8 all-reduce | 6.4× 느림 | B200 multimem.ld_reduce가 FP8(e4m3) reduce HW 미지원; naive 2-shot은 impl 오버헤드 |
| FP8 residual stream | logit_diff 1.9 (게이트 0.5 초과) | 61층 FP8 누적 정확도 붕괴; quant 13%는 정확도의 대가 |
| self-spec (layer-skip/quant-self) | net-negative | draft가 attention 공유 → c≈0.9, 1/(c+(1−a))<1 |
| **MoE self-spec (reduced-expert)** | **生 1.6× but 발표됨** | MoE memory-bound라 top-1 c=0.445; 단 SS-MoE/MoE-Spec(2026)이 이미 발표 |
| 통신 config (NVLS/SP/async-TP) | +3.6%/노이즈 | 이미 multimem 최적; NVLS 플래그만 소폭 |

## 6. 결론
8×B200에서 대형모델 서빙은 **통신-bound**(all-reduce ~31% GPU시간)이고, 그 통신은 이미
multimem-PTX로 near-optimal이다. fits-on-GPU 모델은 DP로 통신을 제거하지만(+182%), 대형모델은
구조적으로 통신을 못 피한다. 표준기법(EPLB/multimem/FP8)이 모든 주요 병목을 이미 최적화해, novel
알고리즘 여지는 (a)HW 미지원(FP8 AR), (b)정확도 대가(FP8 residual), (c)이미 발표(MoE-spec)로 막혀
있다. **본 연구의 기여는 이 한계를 nsys 직접측정으로 정량화하고, flamegraph가 가리는 GPU 병목을
드러낸 방법론, 그리고 fits-on-GPU 임계가 통신 해결가능성을 가르는 경계임을 보인 것이다.**

## 산출물 (재현)
- nsys 트레이스: `runs/r1cg_prof.nsys-rep`(R1 cudagraph), `runs/kex_prof.nsys-rep`(K-EXAONE)
- 측정 스크립트: `exp/phase1_fp8ar.py`, `exp/probe_fp8_residual.py`, `exp/probe_moe_spec2.py`,
  `exp/moe_spec_econ.py`, `exp/moe_spec_multitoken.py`, `SUB_256/.../sweep_dp.sh`(DP8 +182%)
- 관련 success.md: SR-005(DP8 +182%), SR-001~004.
