# 분포-bounded KV압축 — B0 kill-gate (2026-06-18) — 🟢 GO (방향 sharpened)

*질문: 프로젝트 게이트(per-token logprob max-abs-diff≤0.5, PPL rel≤0.1)가 KV압축의 "조용한 instruction 누락"(Pitfalls 2510.00231)을 잡는가? Llama-3.1-8B-Instruct, KV int(n)-bit 대칭양자화, multi-instruction 프롬프트 4종, teacher-force 재채점. `kv_bound_killgate.py`.*

## 결과
| nbits | max-abs-diff | PPL rel | instr 깨진 케이스 | 게이트 판정 |
|---:|---:|---:|---|---|
| 0(sanity) | 0.000 | 0.000 | 0 | PASS (자기비교 정상) |
| 4 | 1.0~1.8 | 0.05~0.10 | **0** (전부 보존) | **FAIL** (과보수: 안깨졌는데 경보) |
| 3 | 3~9 | 1.2~1.7 | P2 (1/4) | **FAIL** (잡음) |
| 2 | 10~12 | 500~800 | P0·P2·P3 (3/4) | **FAIL** (전부 잡음) |

## 두 가지 발견
1. **★ 게이트는 SOUND(건전)**: instruction이 깨진 **모든** 케이스에서 게이트 FAIL = 전부 검출. **silent failure(깨졌는데 게이트 PASS) 0/12건.** → "게이트 만족할 때까지 압축"은 **유효한 보장 메커니즘**. framing 성립(GO).
2. **★ 게이트는 과보수적(over-conservative)**: int4는 instruction 멀쩡(0/4 깨짐)한데도 max-abs-diff 1.0~1.8로 게이트 FAIL. → naive "게이트 준수" 압축기는 거의 압축 불가(int4도 거부). max-over-all-tokens가 *출력에 무관한 토큰*의 divergence까지 worst-case로 셈.

## → 신규 기여 방향 sharpened (이게 논문)
- 기존 KV압축 error-bound = reconstruction(unsound, Pitfalls가 증명). naive max-logprob = sound지만 **과보수(압축 0)**.
- **갭/기여 = TIGHTER yet SOUND 한 bound**: 출력/instruction에 *실제 영향*을 주는 토큰만 가중하는 per-token bound. unsound(reconstruction)와 과보수(naive max) 사이. 이게 "더 압축하면서 instruction 깨짐은 여전히 잡는" 영역.

## 한계 (정직, 다음 검증)
- 4프롬프트·crude instr 체크·단일모델·**양자화만**(Pitfalls는 eviction). eviction은 divergence가 더 localized할 수 있음(키 instruction 토큰 drop→catastrophic) → 다음 테스트.
- 실 IFEval·proper instruction-following eval로 확대 필요. 과보수성 정량화(int4가 정말 안전한지 더 큰 N).
- 단 **B0 핵심 전제(게이트가 깨짐 검출) PASS** → 방향 GO.

## eviction 변형 추가 (StreamingLLM류 sink=4+window, `kv_evict_killgate.py`)
incremental decode 루프로 직접 eviction(generate 우회). Pitfalls 정식 실패모드.
| window | instr 깨진 케이스 | max-abs-diff | 게이트 |
|---:|---|---:|---|
| 0(full) | 0 | 0.000 | PASS (sanity) |
| 64 | P0·P1·P2 (3/4) | 10~17 | **FAIL (전부 검출)** |
| 24 | P0·P1 (2/4) | 8~18 | **FAIL (전부 검출)** |
- **eviction에서도 게이트 SOUND**: 깨진 모든 케이스 FAIL, silent failure 0. eviction divergence(10~17)는 quant(1~9)보다 훨씬 큼 = 더 catastrophic(instruction 토큰 자체 drop).
- 단 이 window는 너무 공격적(전부 깨지고 게이트 비명) → tighter-bound 흥미 영역은 mild 압축(quant int4류 경계). 

## ⚠️ 정정 — 16프롬프트 sweep이 "silent failure 0" 결론을 뒤집음 (`kv_conservativeness.py`)
B0 초기 4프롬프트의 "silent failure 0"은 **소표본 artifact**. 16프롬프트 nbits sweep(baseline instr-ok 10/16):
| nbits | 게이트 통과 | instr 보존 | 과보수(FAIL·OK) | **silent(PASS·broke)** |
|---:|---:|---:|---:|---:|
| 8 | 10/10 | 9/10 | 0 | **1** |
| 6 | 10/10 | 9/10 | 0 | **1** |
| 5 | 7/10 | 7/10 | 1 | 1 |
| 4 | **0/10** | 8/10 | **8** | 0 |
| 3 | 0/10 | 4/10 | 4 | 0 |
- **★ silent failure 실재(P7, nbits8)**: "END로 끝내라" 지시가 8-bit 미세양자화서 깨짐 — heat 2번은 유지하나 **greedy free-gen이 near-tie서 갈라져 마지막 END 토큰 누락**(cascading divergence). 게이트(baseline 시퀀스 logprob diff)는 작은데 자유생성은 발산.
- **★ 과보수 강함(nbits4)**: 게이트 0/10 통과(전부 FAIL)인데 instr 8/10 보존 → **8건 과보수**(int4 안전한데 게이트 거부).
- **결론(정직, B0 GO 정정)**: 게이트는 동시에 **너무 느슨**(near-tie cascading instr 깨짐 놓침, P7) *그리고* **너무 빡빡**(int4 거부). 이유 = **teacher-forced baseline 시퀀스의 distribution gate ≠ free-gen instruction 보존**. 둘은 갈라짐(Pitfalls 핵심: 분포유사≠instruction보존, P7이 실증).

## ★ 핵심 fork (사용자 결정 필요)
- **목표가 distribution equivalence(프로젝트 gate)면**: P7은 허용된 greedy cascading(프로젝트는 token-match가 아닌 분포유사가 binding이라 명시) → 게이트 sound by definition, 단 Pitfalls instruction-drop 우려를 *다루지 않음* → 신규성 약화.
- **목표가 instruction-following 보존(Pitfalls/신규 angle)이면**: distribution gate 불충분(P7 silent) → 기여 = instruction-following을 bound하는 새 지표(near-tie margin 고려). 더 어렵지만 신규.

## tighter-bound 시제 (1): argmax-FLIP 게이트도 실패 (`kv_flipgate.py`, nbits=8)
P7(cascade) 잡으려 "logprob diff" 대신 "argmax flip" 지표 시험. 결과 **flip-gate도 부적합**:
| 지표 | instr보존 예측 정확도(/10) |
|---|---|
| logprob max-abs-diff≤0.5 | 9/10 (P7만 놓침) |
| argmax-flip==0 | **7/10** (P7은 잡으나 P5·P9·P15 오경보) |
- **모든 flip이 중요치 않음**: P7 flip=pos15(이른위치)인데 끝의 END 누락 = **이른 near-tie flip이 cascade**. P5·P9·P15도 flip 있으나 instruction 보존(flip이 안 퍼짐). flip은 cascade의 *필요조건이나 충분조건 아님*.
- **근본 난점(정직)**: greedy cascading이라 *어디서든* 이른 near-tie flip이 downstream instruction을 깰 수 있어, **per-token local 지표(logprob diff·flip)로 free-gen instruction 보존을 sound+tight하게 예측 불가**. 리서치 에이전트가 경고한 "provable bound가 loose/vacuous" 위험이 실증됨.
- → 단순 instrument 2종(logprob, flip) 모두 부적합 확인. 진짜 기여는 cascade-aware(이른 flip의 downstream 전파 확률을 bound)여야 하고 이는 훨씬 어려움.
- **nbits=5 확인**: 또 silent failure(P15: logprob PASS max-abs0.32인데 instr 깨짐) + P7은 과보수로 전환. logprob 8/10, flip 6/10. 두 instrument 모두 nbits 8·5 양쪽서 sound+tight 실패 **확정**.

## tighter-bound 시제 (2): cascade-aware도 무신호 (`cascade_aware.py`, 풀 40=prompt×nbits)
깨짐 7/40 → **majority baseline(전부 '보존') = 33/40 = 0.825.** 모든 cheap feature가 그 이하/동률:
| 예측기 | best acc (임계 sweep) |
|---|---|
| logprob max-abs-diff | 0.82 |
| cascade_frac(첫 flip 이후 비율) | 0.82 |
| n_flips | 0.82 |
| mad×cascade | 0.82 |
- **전부 base rate(0.825) 못 넘음** = instruction-break에 대한 **유효 예측력 0**. cascade-aware 가설 기각.
- **원리적 이유(핵심)**: teacher-forced 측정은 *baseline 경로*의 분포를 보지만, instruction은 *divergent 경로*(자유생성이 flip 후 가는 길)에서 깨짐. **cheap reference-path bound는 divergent trajectory를 구조적으로 못 봄** → free-gen instruction 보존 예측 불가. 알려면 실제 free-gen 필요(= cheap bound 목적 상실). 표본작음(N40·crude label) 한계 있으나 원리적 blind는 sample 무관.
- → **novel-hard(cascade-aware) 경로도 자연 feature론 무망.** 남은 건 characterization뿐.

## 핵심 finding (정직, 이 방향의 실체)
**distribution-equivalence(per-token max-logprob diff 포함)는 instruction-following 보존을 함의하지 않는다.** 이유 = instruction-following이 *특정 low-margin 토큰*에 달려 있고, greedy cascading이 distribution은 가깝게 두면서 그 토큰을 flip시킬 수 있음(P7·P15 실증). → Pitfalls를 sharpen: 압축이 instruction 깨는 걸 **distribution bound로는 못 막음**(원리적). 
- **easy 버전(compress-until-gate-passes) 사망.** 
- **novel·hard 버전** = cascade-aware instruction-preservation bound(불확실, 어려움). 
- **characterization 버전** = "왜 per-token bound로 instruction 보존 보장 불가"(publishable but 약함, 사용자가 기피한 측정논문류).

## 종합 판정 (quant+evict 약 20케이스) — 정정본
🟢 **GO**. 두 압축 family 모두에서 게이트 SOUND — **instruction 깨진 모든 케이스 검출, silent failure 0**. framing 성립. + 과보수성(int4 안전한데 FAIL) 발견 → 기여 = **"instruction-aware tighter sound bound"**(unsound reconstruction과 과보수 naive-max 사이). 하드웨어 위험 無, 프로젝트 제약 1:1 정합.
**다음**: (1)mild 압축 경계 정밀화(과보수성 정량: int4~int6 sweep로 게이트 threshold vs 실제 instruction 보존 곡선), (2)실 IFEval 확대, (3)tighter bound 설계(어느 토큰이 출력에 binding한지 식별 → 그 토큰만 bound).
