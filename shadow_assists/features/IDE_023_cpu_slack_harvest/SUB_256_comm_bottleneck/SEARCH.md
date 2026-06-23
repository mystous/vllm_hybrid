# SUB_256 — TP8 통신(all_reduce) 병목 전면 공략 (2026-06-17~)

> 사용자 지시: 통신 병목 해결, 고전+신규논문+각종 방법 총동원, 통신 다이렉트 측정, **PTX 적극**,
> 방법 찾을때까지 반복, 30분 보고. (KV캐시 이탈 교정 후 복귀)
> 표적: NVFP4 awqgptq 70B + suffix spec + FaP + pad, TP8. flamegraph서 all_reduce ~6% self.

## 핵심 방법론 교정
end-to-end tps A/B는 ±4% 노이즈가 6% 통신을 가림 → **all_reduce 격리 직접 측정** 필수.

## iter1 — NCCL 알고리즘 직접 측정 (`exp/all_reduce_bench.py`, `run_arbench.sh`)
8-GPU bf16 all_reduce µs/call:
| size_MB | default | Ring | Tree | NVLS |
|---|---:|---:|---:|---:|
| ≤1.0 | ~28 | ~28 | ~33 | ~37 |
| **2.75(decode)** | 49.8 | **39.1** | 59.6 | 71.7 |
| 8.0 | 71.6 | 56.8 | 85.6 | 71.6 |
**발견**: (1)소형=latency-bound ~28µs floor(레버=지연,not 대역폭) (2)decode크기서 Ring>default 27%
(3)NVLS 소·중형 최악(큰메시지용). **단 vLLM은 NCCL 아닌 flashinfer custom AR(one-shot IPC)** 사용
→ 진짜 baseline은 그것(~5µs 추정), NCCL은 비교 기준 아님.
→ **방향: decode AR=latency-bound 소형 → multimem-PTX one-shot AR(NVLink 멀티캐스트+리덕션)이 정답.**

## 방법 총동원 목록
| # | 방법 | 분류 | 상태 |
|---|---|---|---|
| 1 | NCCL Ring/Tree/NVLS 비교 | 고전 | ✅ Ring 최선/NVLS 최악(decode) |
| 2 | **symm_mem(multimem-PTX) 2-shot vs NCCL** | 신규/PTX | ✅ **2-shot −44%@decode** (아래) |
| 3 | **flashinfer trtllm AR vs symm_mem 2-shot** | 실 baseline | 대기 (crux) |
| 3b | **custom multimem-PTX 커널** (AR+RMSNorm 융합·decode특화) | 신규/PTX | 대기 |
| 4 | FP8 저정밀 AR (통신량 절반) | 신규 | 대기 |
| 5 | reduce-scatter+all-gather / 2-shot | 고전 | 대기 |
| 6 | async-TP overlap (Flux/CoCoNet) | 논문 | (c)서 노이즈, 재격리측정 |

## 측정 환경
70B hidden=8192, decode AR=168토큰×16KB≈2.75MB. busbw=algbw×2(n-1)/n.

## iter2 결과 ✅ — symm_mem(multimem-PTX) 2-shot all_reduce (`exp/symm_mem_bench.py`)
8-GPU bf16 µs/call:
| size_MB | NCCL | 1-shot | 2-shot | 개선(2shot vs NCCL) |
|---|---:|---:|---:|---:|
| 0.06 | 38.5 | 21.1 | 18.5 | −52% |
| 1.0 | 28.8 | 52.9 | 21.6 | −25% |
| **2.75(decode)** | 45.9 | 108.8 | **25.9** | **−44%** |
| 8.0 | 70.6 | 279.7 | 43.4 | −39% |
**multimem-PTX 2-shot이 NCCL 25-52% 우위.** 1-shot은 큰 크기 폭증(latency-only). PTX(multimem.ld_reduce/st)
in-fabric 리덕션이 latency-bound 소형 AR에 최적 — 사용자 PTX 지시 적중.
**crux**: vLLM은 flashinfer trtllm AR(+일부 symm_mem) 사용 → iter3서 flashinfer AR 직접 측정해 2-shot이
그것도 이기는지 확인 후 vLLM 통합/커널 특화.

## iter3 결과 — vLLM은 이미 multimem-PTX 사용 (near-optimal)
multimem_all_reduce_ µs/call (decode 2.75MB): NCCL 46.2 / 2-shot 24.7 / **multimem 20.0**.
B200(sm10.0)·w8: vLLM은 2.75MB<128MB라 `multimem_all_reduce_`(단일패스 multimem PTX) 사용 = 20µs 최적.
→ NCCL 대비 44% 이득은 vLLM이 이미 capture. 알고리즘 교체 win 없음.
**남은 레버**: (1)FP8 AR(20µs중 bandwidth부분, 12.9µs floor쪽으로) (2)custom 융합 multimem-PTX
커널(AR+RMSNorm+FP4quant 1패스, vLLM symm_mem 버퍼복사2회·별도norm 제거).

## iter4 — FP8 AR 미지원 (`exp/fp8_ar_bench.py`)
PyTorch multimem_all_reduce_ = bf16 only (fp16/fp8e4m3/fp8e5m2 전부 not implemented). FP8 AR엔 custom
PTX 필요 + 8-rank fp8 누적 정밀도손실(게이트 리스크). → FP8 경로 보류.
**종합**: vLLM AR 이미 multimem-PTX(20µs isolated, in-graph 6%) + flashinfer AR-RMSNorm 융합 = 고도최적.
알고리즘교체·FP8 막힘. **남은 단 하나 PTX 레버 = custom 융합 multimem-PTX 커널(AR+RMSNorm+FP4quant
1패스, symm_mem 버퍼복사2회·별도norm 제거)**. iter5=이 커널 실제 작성(multimem.ld_reduce/st + cuMulticast).
**caveat**: 마이크로벤치 20µs는 per-call launch 오버헤드 포함; cudagraph가 그걸 제거하므로 실 in-graph
AR은 작음 → 통신 win 헤드룸 구조적으로 작음(정직).

## iter5 ⭐⭐ 돌파구 — nsys: all_reduce = GPU 시간 50% (flamegraph 6%는 CPU스레드였음!)
nsys cuda_gpu_kern_sum (best 구성, ptok2000 부하):
| 커널 | GPU% |
|---|---:|
| **ncclDevKernel_AllReduce_RING (pynccl)** | **34.4%** |
| **multimem_all_reduce** | **15.9%** |
| FP4 GEMM | 10.4% | RMSNorm+quant | ~11% | attn | 3.6% |
**all_reduce = 50.3% GPU 시간!** 지배 AR=NCCL Ring(맨끝 fallback, 평균 387µs=큰 prefill AR).
**원인**: AR 디스패치(cuda_communicator.py:180) NCCL-symm→flashinfer→custom→multimem→**pynccl Ring**.
`VLLM_USE_NCCL_SYMM_MEM=False`(기본)라 NCCL-symm(NVLS, 큰텐서용) 꺼짐 → 큰 prefill AR이 앞 경로 전부
크기초과로 거부당해 느린 NCCL Ring으로 추락. 주석: "NCCL symm_mem=큰텐서 better bandwidth".
**win 가설**: VLLM_USE_NCCL_SYMM_MEM=1 또는 multimem max_size 상향 → 큰 AR을 빠른 경로로. 측정중.

## iter6 — VLLM_USE_NCCL_SYMM_MEM=1 A/B (prefill-heavy, ptok4000 mtok8)
baseline 15.82 vs nccl_symm 16.39 prefill req/s = **+3.6% 실측 win** (큰 AR NCCL Ring→NVLS).
modest 이유: NVLS가 그 사이즈서 NCCL Ring보다 약간만 빠름. comm 대부분(필수 리덕션) 환원불가.
**신규성 0**: NVLS 플래그·AR+norm+FP4quant 융합(flashinfer allreduce_fusion)·multimem 전부 upstream.
**다음 iter7**: multimem max_size 상향 패치 → 큰 AR을 NVLS 대신 multimem(벤치서 2× 빠름)으로. >+3.6% 가능?

## iter7 — multimem-large 역효과 (복사 오버헤드) → 최종 결론
큰 AR(16-64MB) 격리벤치선 multimem이 NCCL Ring −31~36%(`large_ar_bench.py`)이나, vLLM symm_mem 경로는
multimem 전후 버퍼복사 2회(symm_mem.py:133·154) → 128MB AR이면 256MB 복사 추가. 실측 A/B:
symm512/1024(multimem-large) 14.51 req/s = baseline 15.82 대비 -8%(복사가 이점 잠식).
vLLM이 multimem 128MB 캡한 이유=복사비용. NVLS는 registered symm mem서 복사없이 AR → 최선.

## ★ 최종 통신 결론
realizable comm win = VLLM_USE_NCCL_SYMM_MEM=1 = +3.6%(prefill-heavy, 공짜 플래그, 큰AR NCCL Ring→NVLS).
- multimem-large 역효과(복사), FP8 미구현, overlap/SP 노이즈 → +3.6%가 현실 천장(리덕션 환원불가).
- 신규성 0(NVLS·융합·multimem 전부 upstream). 가치=진단(comm GPU 50%·NCCL Ring 오라우팅 규명)+공짜+3.6%.
- env 패치 역효과라 revert(symm_mem.py 원복).

## iter8 ★★★ DP8 = +181.8% (통신 제거) — 세션 최대 throughput win
nsys가 comm=GPU 50% 규명 → NVFP4 70B=40GB<B200 183GB라 1 GPU 적재 가능 → DP8(데이터병렬, 통신 0):
| 구성 | aggregate gen_tps(conc96) | util |
|---|---:|---:|
| TP8 (통신 50%) | 5393 | 88% |
| DP8 (통신 0, 8 복제본) | **15196** | 90% |
**+181.8% (2.8×).** 통신 추적의 직접 payoff (flamegraph 6%만 봤으면 못 찾음).
**조건/정직**: 고동시성 throughput win (DP는 복제본 8개 포화 필요); 저-conc 레이턴시는 TP가 유리.
모델이 1 GPU에 들어갈 때만(FP4 70B=40GB ✓). 신규성0(DP vs TP는 표준 서빙 통념)이나 측정 win 대형.
