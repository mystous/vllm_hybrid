# SUB_257 — 대형모델 한계 → 논문급 신규 알고리즘 (2026-06-17~)

> 사용자 지시: 논문급 신규 알고리즘이 목표. 현 HW(B200×8) 한계를 대형모델(405B dense, 671B MoE)에서
> 찾고, 신규 알고리즘으로 돌파해 논문 기여. 무한 루프, 30분 보고.
> 교훈(이전): 효과 있는 건 다 upstream. 신규성은 **풀리지 않은 벽**에서만 나온다 — 70B(1GPU 적재)는
> 다 풀렸고, **대형모델의 진짜 벽**(MoE all-to-all·expert 불균형, 405B TP 통신/메모리)이 territory.

## 가용 대형모델
- **DeepSeek-R1 671B MoE** (642GB FP8): 256 routed experts, top-8, 1 shared, 61층, hidden 7168.
  → expert 부하 불균형(hot expert→straggler)·all-to-all = 미해결 active research.
- **Llama-3.1-405B-FP8** (908GB): dense, 1GPU 불가, TP 필수.
- Mixtral-8x7B (소형 MoE 프로토타이핑용).

## 연구 프로토콜 (무한 루프)
1. **한계 측정**: nsys GPU 타임라인으로 지배 병목 규명 (flamegraph 아님 — SR-005 교훈).
2. **미해결 gap 식별**: vLLM/upstream이 이 스케일서 약한 곳.
3. **신규 알고리즘 설계** (gap 타깃, 출력 동등 or 게이트 통과).
4. **구현+측정** (잘 튜닝된 baseline 대비, 정직한 숫자).
5. 신규 win까지 반복 → 논문 기여.

## 반복 기록
| iter | 측정/시도 | 결과 |
|---|---|---|
| 1 | R1-671B nsys GPU 커널 breakdown (한계 규명) | 🔵 프로파일중 |

## iter1 진행 — R1-671B 부팅·서빙 확인 (nsys 타이밍 재시도)
- R1-671B(deepseek_v3, 네이티브 vLLM, trust_remote_code 불요) EP8 부팅 성공. GPU 87GB/8장.
- MoE EP 활성: `MoEPrepareAndFinalizeNoDPEPMonolithic`(all-to-all dispatch/combine).
- **EP8 throughput = 1725 gen_tps** (conc32, 8라운드). 70B FP4 DP8(15196)의 ~1/9 — MoE 대형 한계.
- nsys 1차 캡처 창 빗나감(graph capture 느림). 연속부하+delay840s 재시도중(orch3).
- 신규 알고리즘 2후보: (1)all-to-all expert 통신 (2)expert 부하 불균형(hot expert straggler).

## 두번째 MoE: K-EXAONE-236B-A23B (다운로드중)
LG 독자 대형 MoE 파운데이션. exaone_moe, 128 experts top-8, 48층. vLLM 네이티브 지원(+MTP).
R1(256expert)과 다른 expert 수 → 신규 알고리즘 일반화(논문). FP8 245GB 다운로드중.

## iter1 ★ 결론 — R1-671B는 통신-bound (한계 규명)
nsys GPU 커널 breakdown(EP8 conc32 eager, rep 595MB):
| 커널 | GPU% |
|---|---:|
| cross_device_reduce_2stage (TP all-reduce) | **81.1%** |
| expert GEMM (bmm_E4m3 FP8) | ~6% | MLA attn | 1.9% | multimem AR | 1.1% | MoE routing/finalize | ~2% |
**all-reduce 81%(eager; cudagraph면 ~50%대 추정)**. R1-671B decode = 압도적 통신-bound.
**진짜 한계**: 671B(671GB)>1GPU(183GB) → DP 불가(70B 해법 안통함). TP/EP 통신 불가피 = 대형모델 벽.
**신규 알고리즘 방향**: 통신량 절감 = **error-feedback 저정밀(FP8/FP4) all-reduce**. PyTorch multimem
bf16만(FP8 미구현), naive FP8 8-rank 누적 정밀도붕괴 → 오차보상(stochastic round+error feedback)으로
정밀도 유지하며 통신 2-4×↓. vLLM/upstream 미존재+신규성+TP 대형모델 일반화=논문급. 측정된 81% 직격.
**iter2**: (a)cudagraph 실측 comm% 확인 (b)FP8 error-feedback AR feasibility(정밀도 vs 통신량).

## iter2 — FP8 all-reduce feasibility (신규방향 1차)
`exp/probe_fp8_ar.py` (8-rank, H=7168, Gaussian+outlier): FP8-RTN rel_err ~2.6%/layer, FP8-SR ~3.1%
(SR 단일패스 추론 AR선 RTN 대비 이득 미미, e4m3 ~12.5% step 한계). naive FP8 AR=통신2×↓이나 2.6%오차.
→ 게이트 통과엔 정교한 오차보상 필요(training식 error-feedback은 단일패스라 부적용). end-to-end 측정 필요.
**iter3(robust)**: cudagraph 실측 comm%(eager 81%는 과장 가능) — 진짜 50%+면 저정밀AR 커널 투자가치 확정.
**상태**: 한계=comm-bound 대형모델(671B>1GPU→DP불가) 명확. 신규알고리즘=저정밀AR 오차보상, 큰 구현.

## iter3 ★ cudagraph 실측 — eager 81% 교정 → 통신 31%
R1-671B cudagraph GPU breakdown: all-reduce(multimem)+allgather ~31% / expert GEMM ~20% /
MoE activation+finalize ~11% / FP8 양자화커널 ~13% / RMSNorm+quant ~5% / attn 2%.
**균형잡힌 시스템**(eager 81%는 launch오버헤드 artifact). 단일 2× win 없음.
**신규 후보 재평가**: FP8 AR(통신31%, 2.6%오차 hurdle, 천장~+15%) vs **expert 부하 불균형(straggler-wait가
comm 31%에 숨음, 미측정=진짜 novel)**. → iter4=expert 부하 분포 측정(라우팅 hook). 불균형 크면 적응형
expert 균형 알고리즘(논문), 작으면 FP8 AR 오차보상.

## iter4 — expert-imbalance 각도 차단 (upstream + 설계균형)
1. vLLM에 EPLB(Expert Parallel Load Balancing) 모듈 이미 존재(vllm/distributed/eplb/: policy,
   rebalance_execute, eplb_state, async_worker) — 동적 expert 재배치 upstream 구현.
2. R1 topk_method=noaux_tc = aux-loss-free balancing = 설계상 균형(DeepSeek 학습 bias 균형).
→ expert 불균형 novel 각도 차단. 또 upstream-covered.
## R1-671B 종합: comm31%/expertGEMM20%/MoE오버헤드20%/quant13%. 균형 잘된 고도최적 시스템.
**유일 novel 문**: FP8 AR + 신규 오차보상(vLLM/PyTorch 미존재, 통신31% 직격). 단 2.6%오차 hurdle,
천장~+15%, 수일 커널구현+게이트측정 필요, 불확실. = 논문급 유일 현실 후보.
**철칙 6번째**: 671B 대형MoE도 vLLM 고도최적(EPLB·multimem·DeepEP·FP8). novel은 저정밀통신뿐.

## iter5 — K-EXAONE-236B가 R1 일반화 (대형 MoE 한계 확정)
K-EXAONE breakdown(eager): cross_device_reduce_2stage 81.6%(=R1 81%), group_gemm 5.3%, fp8quant 1.6%.
tps 1745(=R1 1725). 두 대형 MoE 동일구조=통신-bound 고도최적.
## ★ 종합 결론 (70B dense + R1-671B + K-EXAONE-236B)
대형모델 = 통신-bound(all-reduce ~30% cudagraph/~80% eager). fits-on-GPU(70B FP4)→DP로 통신제거
+182%(SR-005). 안들어가면(대형 MoE/dense)→통신 불가피. 표준완화(multimem·EPLB·noaux_tc·FP8·DeepEP)
전부 upstream=철칙 6번째. **논문 경로**: (1)FP8 AR+신규 오차보상(유일 미해결문, 2.6%오차 hurdle, 수일,
천장~+15%, 불확실) (2)특성화 논문(measurement study, publishable) (3)405B dense 추가.

## 다양한 각도 실제-코드 시도 (Phase 접근)
- 각도1 FP8 AR(통신31%): 死 — HW multimem FP8 미지원, naive 6.4×느림(impl 오버헤드). `exp/phase1_fp8ar.py`
- 각도2 Persistent FP8 residual(quant13%): 死 — 8B 32층서 logit_diff 1.9(게이트0.5 초과), 정확도붕괴.
  quant 오버헤드=정확도 대가, 제거불가. `exp/probe_fp8_residual.py`
**일관된 벽**: 주요 병목(comm·quant) 직격은 다 최적화됨/정확도 대가. 남은 novel 여지=덜 최적화 regime
(long-context attention) 또는 다른 패러다임(MoE speculative). 다음: 각도4 MoE-spec or long-context.

## ★★ 각도4 MoE-spec — 첫 생존 lead (실측 1.60×)
gate 절단 top-1 draft vs top-2 full (Mixtral, `exp/probe_moe_spec2.py`,`moe_spec_econ.py`):
- accept=0.819(per-text 0.72-0.92). **단일토큰 100%는 패치버그 artifact였음(검증으로 잡음)**.
- **draft 연산비 c=0.445 실측**(top-1 25.75ms vs top-2 57.83ms decode) — MoE decode memory-bound라
  top-1이 expert 가중치 절반만 로드 → 진짜 쌈. layer-skip(R5)이 죽은 "draft 안쌈" 문제를 깸.
- **속도배율 1/(c+(1-a))=1.60×**. 신규(vLLM reduced-expert self-draft 없음).
**caveat**: 이론값; 실구현~1.2-1.4×, 다중토큰 accept 누적 측정필요, R1(top8) 일반화 확인.
**다음**: 다중토큰 accept / 신규성 문헌 / R1 일반화 / vLLM 구현+end-to-end throughput.

## ★★★ MoE-spec 다중토큰 검증 — 강한 lead
다중토큰(`exp/moe_spec_multitoken.py`, Mixtral 8프롬프트 24토큰): 첫불일치 run-length=8.38(우수),
c=0.445, realistic 속도 ~1.26~1.6×. = 세션 첫 측정 positive novel lead.
**핵심 통찰(논문 angle)**: layer-skip/quant-self/FP8가 죽은 "draft 안싸다" 벽을 MoE의 memory-bound
특성(expert 가중치 로드 지배)이 깸 → top-1 draft가 진짜 0.445×.
**논문**: "Self-Speculative MoE Decoding via Reduced-Expert Drafting". 다음: 문헌신규성/R1 일반화/
vLLM 구현(top-k draft+rejection verify=출력동등)/end-to-end throughput.
