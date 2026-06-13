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
4. **모델 크기 일반화** — 이득 ∝ host-bound 정도 (70B +38.4% > 8B +16.4%),
   초고-accept (8B mix acc 0.93) 는 중립 = 게이트 off-조건까지 실측.
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
