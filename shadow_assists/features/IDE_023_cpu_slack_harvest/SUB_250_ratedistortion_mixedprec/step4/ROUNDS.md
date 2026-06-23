# 10라운드 고전기법 탐색 루프 (2026-06-16~)

> 사용자 지시: "란초스처럼 상관없어 보이는 고전 수학/HPC 기법을 찾아 하드웨어 극대화
> → 빠른 검토 → 유망 후보 실제 구현·측정 → 판정 → 성능/정확도 향상 없으면 반복, 10회."
> 제약: 70B B200 서빙 = GPU-compute-bound → tps는 비트폭만 닿음. 양성 타깃 두 축:
> (A) GPU FLOP/byte 감소, (B) 고정 비트폭 정확도 회복(빠르지만 게이트 FAIL을 PASS로 구제).
> 최우선 표적: **W4A4+spec = bf16 +194%(최고속), 게이트 ppl_rel 0.128>0.1 근소 FAIL**.

## 게이트 기준
분포동등: max_logprob_diff ≤ 0.5 AND ppl_rel ≤ 0.1 (vs bf16 lp_bf16.json).
앵커 tps: bf16 1437 / FP8 1810 / W4A4 2225 / W4A4+spec 4232(FAIL).

## 라운드 기록

| R | 고전 기법 | 응용 가설 | 상태 | 판정 |
|---|---|---|---|---|
| (사전) SUB_250 | rate-distortion water-filling | 게이트-distortion 비트배분 | 완료 | ❌ REFUTED (+34.5%); bump-mixed 70B 실측도 순수W4A4에 dominated(2063/FAIL vs 2273/PASS) |
| (사전) TSK_049 | Lanczos eigen-rotation | NVFP4 회전 보강 | 완료 | ❌ 부적용(group16 micro-scale) |
| **R1** | **Optimal Brain Surgeon→GPTQ** | GPTQ 오차역전파 NVFP4로 +spec 게이트 구제 | ✅ 완료 | ❌ near-miss: +spec ppl_rel 0.128→0.093(처음 통과) 但 max_diff 0.43→0.514(0.5 근소 초과)→FAIL, tps 무이득 |
| **R2** | **AWQ (activation-aware, 채널 saliency 가중)** | 중요 채널 보호로 max_diff(worst-case) 공략 | ✅ 완료 | ❌ FAIL — max_diff 0.306(최저, 설계대로) 但 ppl_rel 0.179 악화. **GPTQ와 직교 상보** |
| **R3** | **AWQ+GPTQ 결합** (saliency 보호 + 오차역전파) | 두 직교 실패축 동시 해결 → +spec 게이트 양축 PASS | ✅ 완료 | ✅ **WIN(SR-004)** — awqgptq+spec 4301tps/PASS(diff0.491 rel0.0667), bf16 +199.3%. W4A4+spec 게이트 구제 성공. 신규성0(upstream) |
| R4 | **Count-Min Sketch/Bloom** (확률적 자료구조) | [B축 spec-draft] sketch draft 가속 (SUB_251) | ✅ 완료 | ❌ no-go: MultiOrder hash accept±0/1.5×↑이나 SUB_225 천장(CPU draft GPU오버랩→70B tps 무영향). 신규지만 향상無 |
| R5 | **layer-skip self-spec** (GPU-direct, 사용자 신규지시) | 부분forward draft+full verify, 출력안전 (SUB_252) | ✅ 완료 | ❌ no-go: 속도배율 1/(c+(1−a)) 전구간<1, draft연산-accept 커플링→training-free 불가. LayerSkip 파인튜닝 필요 |
| (이전 R5) control variates | [B축 verify] spec verify 분산↓ | 보류 | 신규성 의심(verify는 exact rejection) |
| R6 | **contextual activation sparsity** (Deja Vu류) | [GPU-direct] MLP 중간활성 prune FLOP↓ (SUB_253) | ✅ 완료 | ❌ no-go: 30%도 logit_diff 0.78>게이트, 비정형 sparsity TC비호환, predictor=training |
| R7 | **양자화 self-spec** (저비트 self-draft) | [GPU-direct] 저비트 draft+4bit verify, 출력안전 (SUB_253) | ✅ 완료 | ❌ no-go: 2bit accept0/3bit 배율 1.07 낙관상한, 실 break-even 이하 |
| ~~R8~~ Chebyshev softmax | [C축] | — | 기각(신규성無: SFU/FlashAttn) |
| ~~R9~~ Walker Alias·WHT-KV·Monarch | [D/A/E축] | — | 기각(신규성無 또는 출력변경 게이트FAIL) |

> **R4~R7 = GPU-direct 신규 4연속 no-go (각기 다른 구조 이유).** 수렴 임포서빌리티: 70B GPU-bound
> 출력안전 가속 = (a)비트폭[upstream] 또는 (b)싸고정확 draft=별도small/학습head[upstream]뿐.
> training-free self-draft·근사연산은 비용↔정확도 커플링/TC비호환으로 break-even 불가.

> **시야 확대 (사용자 지시 2026-06-16 12:08)**: R1~R3은 모두 (A)가중치정밀도 축. R4부터
> 서브시스템 가로질러 다양화 — B(spec CPU draft, SUB_213 +38% 전례=헤드룸 실재)·C(수치커널
> 정확도중립 속도)·D(샘플러)·A(attn/KV, 포화·대수술 후순위)·E(구조화행렬). tps 직접향상은
> GPU-bound 밖 CPU draft 경로(B)가 현실적 통로 → R4/R5 우선.

## R2 — AWQ-NVFP4 판정 (2026-06-16)
- awq 2217tps/FAIL(diff **0.306** rel 0.179), awq_spec 4195/FAIL(diff 0.512 rel 0.210).
- **핵심**: AWQ가 max_diff를 0.306으로 최저화(RTN0.43·GPTQ0.62 다 이김, 설계목표 적중)했으나
  ppl_rel를 0.179로 악화. **R1 GPTQ(rel0.093 최저/diff0.514 초과)와 정확히 상보** — 각자
  게이트의 다른 축을 고치고 다른 축을 깸. → 결합(R3)이 데이터 주도 다음 수.

## R3 — AWQ+GPTQ 결합 NVFP4
- **동기**: R1(ppl_rel 해결)·R2(max_diff 해결)가 직교. llm-compressor에서 AWQModifier→GPTQModifier
  순차 recipe. AWQ 채널보호로 max_diff↓ + GPTQ 오차역전파로 ppl_rel↓ → 양축 동시 통과 기대.
- **목표**: W4A4+spec에서 max_diff≤0.5 AND ppl_rel≤0.1 → 게이트 PASS = bf16 +183% 실win.
- **구현**: `make_awqgptq.py` (AWQ+GPTQ 결합 recipe). 측정 `awqgptq_sweep`.
- **판정 (2026-06-16) ✅ WIN = SR-004**: awqgptq 2208tps/PASS(diff0.491 rel0.0975),
  **awqgptq_spec 4301tps/PASS(diff0.491 rel0.0667)** — RTN+spec(4195/FAIL 0.128) 구제 성공,
  심지어 더 빠름. **vs bf16 +199.3% / vs FP8 +137.6% / vs RTN-W4A4 +89.2%**. 가설(직교 두 축
  결합) 정확히 적중. 신규성 0(AWQ·GPTQ upstream), 가치=엔지니어링 win+직교성 인사이트.

## R1 — GPTQ-NVFP4 W4A4 (OBS 계열 오차역전파)
- **동기**: RedHat NVFP4는 RTN(round-to-nearest). GPTQ는 Hessian 역행렬 가중으로 컬럼별 양자화
  오차를 남은 가중치에 역전파 → 동일 4bit에서 출력오차↓. **속도 동일**(추론시 그냥 NVFP4),
  정확도만 회복 → W4A4+spec(0.128)을 PASS로 돌릴 가능.
- **구현**: `make_gptq.py` (llm-compressor GPTQModifier, scheme=NVFP4, ultrachat 512 calib). GPU 4-7.
- **측정**: `gptq_sweep.sh` → gptq / gptq_spec, tps + 게이트.
- **판정 (2026-06-16)**: gptq 2247tps/FAIL(diff0.620 rel0.110), gptq_spec 4069tps/FAIL(diff0.514
  rel0.093). 앵커 RTN W4A4 2273/PASS, RTN+spec 4195/FAIL(rel0.128).
  **실패모드 이동**: RTN+spec은 ppl_rel(0.128) FAIL/diff PASS, GPTQ+spec은 **ppl_rel 0.093으로
  처음 통과**하나 **max_diff 0.514로 0.5 근소 초과** FAIL. plain gptq가 RTN보다 나쁜 건 자가
  calib(ultrachat512)이 RedHat 정밀 calib보다 약한 confound. **클린 win 아님(no PASS, tps 무이득)
  but near-miss** — binding 지표 ppl_rel를 +spec에서 처음 통과시킴. 다음 라운드로.
- 산출물: `make_gptq.py`, `gptq_sweep.sh`, `runs/gptq_results.csv`. 체크포인트 `/raid/hf_cache/gptq_nvfp4_70b`.

## R2 — AWQ-NVFP4 (activation-aware weight quant)
- **동기**: R1 GPTQ의 약점=max_diff(단일 채널/토큰 worst-case 스파이크). AWQ는 활성 크기로 채널
  중요도를 추정해 salient 채널의 가중치를 양자화 전 스케일-보호 → worst-case 채널오차↓.
  고전 원리=**중요도/saliency 가중**. GPTQ가 못 막은 max_diff를 정확히 겨냥.
- **목표**: W4A4+spec에서 ppl_rel≤0.1 AND max_diff≤0.5 동시 달성 → 게이트 PASS = bf16 +183% 실win.
- **구현**: `make_awq.py` (llm-compressor AWQModifier, scheme=NVFP4). 측정 `gptq_sweep.sh` 패턴 재사용.
