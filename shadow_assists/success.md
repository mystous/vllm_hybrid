# ✅ Success Register — 성공 사례 기록부

> **규칙** (2026-06-13 제정): 측정으로 확정된 **성공 사례**(성능 향상 또는 채택된
> 설계 규칙)는 본 파일에 등록한다. 항목 형식: ID / 한 줄 성과 / 무엇이 성공했나 /
> 수치 / 재현 방법 / 산출물 링크. 판정 기준·원데이터는 각 SUB 디렉토리가 단일
> 출처이고, 본 파일은 "무엇이 살아남았는가"의 색인이다.

---

## SR-001 · `SUB_213` — Uniform Draft Padding × FULL CudaGraph (2026-06-13)

**한 줄**: suffix spec-decode 의 draft 를 K 로 균일 패딩해 FULL cudagraph 를 적중시켜
**suffix+FaP 대비 70B +38.4% (고정 K6), 8B +16.4%** 의 serving 직접 가속을 확정.

**어떤 부분이 성공했나**:
1. **병목 규명이 실측 기반** — py-spy 워커 프로파일로 "레이어당 Python op 디스패치
   체인 = 워커 CPU 50~75% (FULL replay 는 4.8% 뿐)" 를 확인하고 정확히 그 지점을
   공격. (추측 기반이던 SUB_232/240 의 기각 후 프로파일-주도 재조준의 결실)
2. **출력 무손실 가속** — pad 토큰은 rejection sampling 이 기각 보장 → 분포 등가.
   105+ 셀 전부 100% 성공·0 에러, tpot p50 도 동시 개선 (16.6→12.1 ms).
3. **K 의 regime 의존성 지도 확보** — 역U자 (고정 최적 K6 +38.4%) + corpus 별
   winner (저-accept mbpp→K4 +46% / 고-accept mix→K12 +104%) → per-corpus
   oracle ≈ **+49%**. 적응형 게이트 (TSK_046) 의 상방 +11%p 를 정량 입증.
4. **모델 크기 일반화 (10모델 전수 확정, 2026-06-14)** — 9모델 × K{4,6,8,12} ×
   7corpus = 252셀 (+70B 63셀) 전부 측정. ⑦(best-K pad) 가 **70셀 중 68셀**에서
   기존 6개 설정(①~⑥)을 전부 상회. 모델별 ⑦ vs ⑤(suf+FaP) 기하평균: Q7B +12.5% /
   DS-Q7B +42% / 8B +34% / Q32B +27% / DS-Q32B +25% / Q72B +28% / 70B +33% /
   DS-70B +42% / **671B +157%** (suffix net-neg 를 pad 가 역전) / 405B +24%(vs④).
   regime: Qwen dense→K4, distill→K6~12, 8B→K12, 대형 dense→K6, MoE→K4 (mix 는 큰 K).
5. **측정 방법론** — "셀별 fresh boot" 규칙 (suffix tree 누적학습이 셀 비교를
   최대 +24% 오염 — SUB_214 에서 발견, 본 측정 전체에 적용).

**핵심 수치** (Llama-3.1-70B, 7 corpus, conc=32, 실 trace):

| 설정 | mix tps | 7-corpus 기하평균 (vs suffix+FaP K=32) |
|---|---:|---:|
| suffix+FaP K=32 (기존 최고) | 7,043 | 기준 |
| **K6 + pad (고정 권장값)** | 9,496 | **+38.4%** |
| per-corpus oracle (K4/6/8/12) | — | ≈ +49% |

**재현**:
```bash
VLLM_SUFFIX_PAD_UNIFORM=1 vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.85 --max-model-len 16384 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method":"suffix","num_speculative_tokens":6}'
# 제약: conc×(K+1) ≤ max_cudagraph_capture_size (기본 512)
```

**산출물**: [`features/IDE_023_cpu_slack_harvest/SUB_213_fap_suffix_uniform/`](features/IDE_023_cpu_slack_harvest/SUB_213_fap_suffix_uniform/)
— `MEASUREMENTS_sweep.md` (본판정·K-sweep·8B), `runs*/` (105+ 셀), `run_sub213*.sh`.
근거 프로파일: `features/IDE_026_rdt_guarded_harvest/profiling/worker0_profile.speedscope.json`.
코드: `vllm/v1/spec_decode/suffix_decoding.py` (`VLLM_SUFFIX_PAD_UNIFORM`).
후속: TSK_046 (다중-K capture 인프라 ✅ / 정책 v1 +1.1% 게이트 미달 — SUB_247).
**정확도 게이트 PASS** (2026-06-13, TST_003 D-ii 방식: worst_max_abs_logprob 0.2743 ≤ 0.5,
ppl_rel 0.0730 ≤ 0.1, 32/32) — main 머지 품질 증거 확보.
**E1/E2 확정**: SUB_212 의 +36% = FaP (호스트 DSA 무죄) — confounder 종결.

---

(다음 성공 사례는 SR-002 로 추가)

---

## ⚠ Upstream 중복 경고 (2026-06-15) — SR-001 후속 작업 관련

SR-001(SUB_213 uniform-pad +38%)의 후속으로 검토하던 **적응형 K선택 / cudagraph 정렬은
upstream vLLM 이 이미 하고 있음** (웹조사):
- **동적 K (per-request)**: PR #26504 `DynamicProposer`/`eagle_dynamic` — 수용률 기반 K
  조절, **Open(2026-06-12)**. = TSK_046 옵션 B(α-EMA 게이트)와 사실상 동일.
- **uniform cudagraph 정렬**: PR #23679 — capture size `1+num_spec_tokens` 정렬,
  **Closed(2026-03-13)**. = SUB_213 uniform-pad 메커니즘과 겹침. + Issue #33341, #36657.

→ **AGENTS.md 중복-작업 금지 적용**: fork 에서 동적-K 게이트나 uniform cudagraph 를
새로 구현하면 중복. **권고: upstream 에 rebase 후 +38~49% 가 이미 흡수됐는지 재측정**,
남는 진짜 빈틈(suffix+FaP 특이 상호작용, 워크로드→K LUT)만 upstream 기여. 메모리
`spec-decode-adaptive-k-upstream` 참조.

---

## SR-002 — FP8 가중치 양자화 (바이트단위 B200 FP8 TC) — 검증된 신규 서빙 win (2026-06-15)

70B Llama-3.1, GPU-compute-bound. **FP8 가중치 양자화(2B→1B)** = B200 FP8 텐서코어 가속:
- **단독 +26.5%** (1436.9→1817.3 tps), **+spec +126.9%** (→3259.6), +tp8 +55.8%.
- **분포동등 게이트 PASS** (max_abs_logprob_diff 0.135≤0.5, ppl_rel 0.042≤0.1) — lossy지만 유효.
- 의미: GPU-compute-bound는 커널/스케줄/메모리 레버에 안 닿고 **비트폭(바이트)만 닿음**.
  spec(패스↓)×FP8(연산↓) 곱셈 스택. SUB_248 라운드1-3. FP4(0.5B) online 부팅 확인=다음.

---

## SR-003 — FP4(NVFP4 W4A4) 양자화 — FP8 돌파한 byte-level win (2026-06-16)

70B Llama-3.1, GPU-compute-bound. **사전양자화 NVFP4 W4A4**(RedHatAI, 가중치+활성 4-bit):
- **vs FP8 +23.0%** (1810→2225 tps), **vs bf16 +54.8%**. **분포동등 게이트 PASS**(max_logprob_diff
  0.43≤0.5, ppl_rel 0.068≤0.1).
- **W4A4+spec = bf16 대비 +194.5%**(4232 tps, 최고속) 단 게이트 근소 FAIL(0.128) → rotation 보강 여지.
- byte-level 천장 갱신: bf16(2B)→FP8(1B,+26%)→FP4(0.5B,+55%). 다음=W4A4+spec을 rotation(SpinQuant/
  란초스②, TP=1)으로 게이트 통과시켜 +194% 확정. SUB_248 FP4 sweep.

---

## SR-004 — AWQ+GPTQ calibration 으로 W4A4+spec 게이트 구제 (SR-003 후속 해결) (2026-06-16)

70B Llama-3.1. SR-003 가 남긴 미해결("W4A4+spec 최고속이나 게이트 FAIL 0.128")을 **오프라인
calibration 으로 런타임 비용 0 에 해결**. SUB_250 step4 10-라운드 루프 R3.

- **메커니즘 (직교성 발견)**: 분포동등 게이트의 두 조건이 직교 — **GPTQ**(오차역전파)는 평균
  왜곡 ppl_rel 만 낮추고 worst-case max_diff 는 악화(0.514), **AWQ**(채널 saliency)는 max_diff
  만 낮추고(0.306) ppl_rel 악화(0.179). **둘을 결합**(AWQ 스케일 → GPTQ 오차보정)하면 두 축 동시 통과.
- **수치** (conc24, ptok2000, mtok256, 70B TP4):

| 구성 | tps | vs bf16(1437) | max_diff | ppl_rel | 게이트 |
|---|---:|---:|---:|---:|---|
| RTN W4A4+spec (기존, SR-003) | 4195 | +192% | 0.43 | 0.128 | ❌ FAIL |
| **AWQ+GPTQ W4A4+spec (R3)** | **4301** | **+199.3%** | 0.491 | **0.0667** | ✅ **PASS** |
| AWQ+GPTQ W4A4 (no spec) | 2208 | +53.7% | 0.491 | 0.0975 | ✅ PASS |

- **새 최고 게이트-통과 구성**: vs FP8 +137.6%, vs RTN-W4A4 +89.2%. SR-003(W4A4 +55%) 상회.
- **신규성 정직**: AWQ·GPTQ 모두 upstream(llm-compressor `[AWQModifier, GPTQModifier]` 2줄).
  **알고리즘 신규성 0** — 가치 = 엔지니어링 win(게이트통과 최고속) + **직교성 측정 인사이트**.
- **재현**: `make_awqgptq.py`(AWQ+GPTQ NVFP4, ultrachat512 calib) → 체크포인트 서빙 + ngram spec.
  산출물: `features/IDE_023_cpu_slack_harvest/SUB_250_ratedistortion_mixedprec/step4/`
  (`ROUNDS.md`, `make_{gptq,awq,awqgptq}.py`, `runs/{gptq,awq,awqgptq}_results.csv`).

---

## SR-005 — DP8 (데이터병렬) 통신 제거 = +181.8% throughput (2026-06-17)

70B Llama-3.1 NVFP4(40GB) TP8 서빙의 **all_reduce가 GPU 커널 시간의 50.3%**(nsys 직접측정;
flamegraph 6%는 CPU 스레드 점유율 오해)임을 규명 → 40GB 모델이 B200 1장(183GB)에 들어가므로
**데이터병렬(DP8, 8 독립복제본)로 TP all_reduce를 통째로 제거**.

- **수치** (best 구성 = NVFP4 awqgptq + suffix spec + FaP + uniform pad, conc=96 aggregate gen_tps):

| 병렬화 | aggregate gen_tps | GPU util |
|---|---:|---:|
| TP8 (통신 50% GPU시간) | 5,393 | 88% |
| **DP8 (TP1×8, 통신 0)** | **15,196** | 90% |

- **+181.8% (2.8×)**. 통신 추적(nsys로 comm=50% 규명, VLLM_USE_NCCL_SYMM_MEM=+3.6%/multimem-large
  역효과 거친 끝)의 직접 payoff. flamegraph(CPU 6%)만 봤으면 못 찾았을 win.
- **조건/tradeoff**: 고동시성 throughput win(복제본 8개 포화 필요); 저-conc 레이턴시·초장 컨텍스트(KV
  분할)는 TP 유리. **모델이 1 GPU에 들어갈 때만**(FP4 70B=40GB ✓; bf16 140GB는 TP 필요).
- **신규성 0**: DP vs TP는 표준 서빙 통념(fits-on-GPU면 DP가 throughput 우위). 가치=대형 측정 win +
  "통신 비중은 flamegraph 아닌 nsys로 측정" 방법론 교훈.
- **재현**: `vllm serve <NVFP4-70B> --data-parallel-size 8 --tensor-parallel-size 1 ...`(+spec/FaP/pad).
  산출물: `features/IDE_023_cpu_slack_harvest/SUB_256_comm_bottleneck/` (`sweep_dp.sh`, `runs/dp_results.csv`,
  nsys `runs/decode_prof.nsys-rep`). 메모리 `tp8-allreduce-50pct-nvls-flag`.

---

## SR-006 — DRAM KV-offload: CPU/DRAM tier가 KV-pressured regime서 GPU-only +76.7% (2026-06-17)

프로젝트 코어 질문 "기존 (논문) 분야를 **CPU/서버 하드웨어 극대화**로 GPU-only 동등이상"에 직접 A/B 응답.
KV cache가 GPU에 다 안 들어가는 regime(working-set > GPU cache)을 만들고 idle DRAM을 KV tier로 켬.

- **설정**: Qwen2.5-7B TP1, `--enable-prefix-caching`, GPU KV cache 인위축소(`--num-gpu-blocks-override`).
  워크로드 = distinct 긴 prefix 24개(각 ~3,050 tok) × 4 round 재사용(working-set ≈ 73K tok).
  A=GPU-only(evict→prefill 재계산) vs B=`--kv-offloading-size 60 --kv-offloading-backend native`(evict→DRAM→fetch).
- **수치** (cache=24K tok, 3-pass paired, 변동<1%):

| 구성 | out_tps | TTFT mean | wall |
|---|---:|---:|---:|
| A GPU-only | ~932 | 0.114 s | 3.27 s |
| **B GPU+DRAM** | **~1,650** | **0.036 s** | 1.85 s |

- **+76.7% throughput / −68% TTFT**, 출력 greedy 동등(3051 vs 3042 tok).
- **70B 스케일링**(Llama-3.1-70B-NVFP4, KV 320KB/tok): 고압(ws 180K/cache 40K)서 **B +97.5%**
  (160 vs 81 tps, TTFT 0.337→0.097s), fetch 97.6GB@55GB/s. **모델 클수록 win↑**(recompute-prefill 비용↑).
- **크로스오버 특성화**(GPU cache sweep): 12.8K **+67%** / 24K **+77%** / 48K(경계) **−10%** / 112K(완전적재) **~0%**.
  → harvest는 **KV-pressured일 때만 win**, GPU 여유 시 offload 순오버헤드. **적응 게이트 필요**.
- **메커니즘 검증**: External prefix cache hit 80-86%, CPU→GPU fetch 15.87GB/0.29s ≈ **54GB/s**(pinned DMA),
  fetch(≈ms) ≪ recompute prefill(≈수십 ms)가 win의 물리원인.
- **신규성 0 (정직)**: 메커니즘 = vLLM native CPU KV offload 기존기능(LMCache/Mooncake 동류). 본작업 = harvest
  명제 **실증 + 크로스오버 정량화**. 신규 여지(미구현): (a) **DSA(/dev/dsa) 가속 전송**(현 cudaMemcpy copy-engine
  →DSA), (b) **적응 게이트**, (c) **fetch-vs-recompute 공동스케줄링** — **3개 모두 막힘**(SEARCH.md):
  (a)DSA=아키텍처 dead-end(host DRAM↔DRAM mover, GPU↔host PCIe 못닿음; vLLM 전송 swap_blocks_batch=GPU DMA, CPU memcpy無),
  (b)신규성 약함, (c)decode-bound라 헤드룸0(fetch 1.76s ≪ decode 9s, 이미 critical-path 밖). → 신규 알고리즘 아닌 **harvest 실증/특성화**.
- **재현**: `features/IDE_023_cpu_slack_harvest/SUB_258_dram_kv_offload/` (`bench_kv.py`, `serve.sh`,
  `sweep_blocks.sh`, `runs/results.jsonl`, `runs/sweep_blocks.jsonl`, `MEASUREMENTS.md`).

---

## ⑧⑨ 확장 (2026-06-19) — ⑦ 챔피언 위 byte-width·동적-K (SR-001 후속, 신규 SR 아님)

Llama-3.1-70B, ⑦(suffix+FaP+best-Kpad) 위에 측정. output_tps 기하평균 vs ⑦, DSA-on/TP8/conc32, 7corpus.
- **⑧ = ⑦ + AWQ+GPTQ NVFP4 W4A4 양자화**: **+7.1%** (코드 corpus 큼 mbpp+29%/humaneval+16%, mix −10%). W4A4 standalone +55%(SR-003)가 ⑦ stack 위선 수확체감(spec+pad+FULL이 weight-bw 병목 선점). 게이트 PASS(SR-004 동일 가중치). ⑧은 70B만(양자화 체크포인트 70B 한정).
- **⑨ = ⑦ + 동적-K**(`VLLM_SUFFIX_DYN_K=1` KS={4,6,12}, α-EMA; SUB_247 D3): **−7.9%**(컨트롤러가 oracle best-K 못 따라감, mix −33%/mbpp −15%; naive 고정K6 대비는 ~+1.1%). upstream eagle_dynamic #26504는 fork 미머지+EAGLE-70B draft 부재로 미측정이나, suffix가 이미 native 동적-K라 개념 흡수.
- **결론**: ⑦이 챔피언 유지. ⑧만 +7% 더함. 동적-K(=#26504류)는 ⑦ 못 넘음. 상세=`codesci/{POINT8_RESULT.md,POINT9_RESULT.md}`, 매트릭스=`SUB_212_optimal_dsa_6point/FULL_MATRIX_6point.md §3 ⑧⑨표`.
