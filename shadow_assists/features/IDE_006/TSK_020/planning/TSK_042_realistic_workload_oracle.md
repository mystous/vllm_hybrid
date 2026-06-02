# TSK_042 — 워크로드 활용 실험 (상세 plan)

> **parent**: `IDE_022` (TSK_020). 선행: `SUB_076`(classifier PoC), `PRE_TSK042_TSK043_prerequisites.md`.
> **status**: 활성 — 데이터·oracle 코어 구현·검증(V1/V2) 완료, full 매트릭스·품질·라우팅 대기.
> **코드**: `vllm_config_perf/gating/realistic_eval/`. 산출: `features/IDE_022_agsd_realistic_eval/TSK_042_realistic_workload_oracle/`
> **핵심 산출물**: `oracle_table.parquet`(throughput) + 품질 표(경로 A/B) + 라우팅 비교 표(llm-d).

---

## 1. 목표 & 산출물
실 trace 를 **모델 × method** 로 돌려:
- (a) **per-prompt oracle throughput** → `oracle_table.parquet` (TSK_043 의 decision-regret ground-truth).
- (b) **출력 품질** — 경로 A(losslessness: spec vs vanilla 동등성) + 경로 B(품질 벤치: pass@k / judge).
- (c) **라우팅 전략 비교** — smart router = **llm-d** vs vanilla-only/trident-only. (AGSD 는 TSK_043.)

핵심 개념: oracle 은 **per-prompt** 측정(workload 라벨 무관). corpus 는 prompt **소스**일 뿐, sonnet/chat/code 라벨은 TSK_043 분류기가 부여 → regret 계산.

## 2. 진행 현황 (2026-06-02)
| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| 데이터 파이프라인 | `corpus_loader.py`, `prompt_sampler.py` | ✅ **V1 검증** (6 corpus, 253p parquet, dedup/length/lang) |
| oracle 측정 코어 | `oracle_runner.py`, `run_oracle_8gpu.sh`, `build_oracle_table.py` | ✅ **V2 검증** (Qwen32B×3method×20p, 60/60 ok) |
| 품질 평가 | `quality_eval.py` | ⬜ 미구현 |
| 라우팅 비교 | `run_routing_compare.sh` | ⬜ (llm-d 인프라 대기) |

**V2 실측**(sharegpt 20p, conc=1): vanilla 177 / suffix 216(+22%) / ngram 193 tps. method spread **16.8%**(<5% 0%) → kill-gate 1차 통과.

## 3. 데이터 파이프라인 (상세)
- **corpus = 소스** (검증된 6종): chat = `lmsys`/`wildchat`/`sharegpt`, code = `humaneval`/`mbpp`/`swebench`. (sonnet/창작은 별도 실 corpus 없음 → lmsys/wildchat 내 분류기가 식별.)
- 추출: 마지막 user turn(chat) / problem_statement(swebench) / prompt(humaneval) / text(mbpp). **streaming 샘플링**(full-cache 안 함).
- 필터: dedup(sha1 normalized), length `32≤tok≤4096`(Qwen tokenizer). **단, mbpp 설명문은 짧아 32 floor 에서 3개만 통과** → code 는 humaneval+swebench 주력, mbpp 는 `--min-tok 16` 로 보강(옵션).
- 메타: langdetect 또는 native_lang(lmsys/wildchat 제공). 실측 분포: 영어 위주 + zh/es/pt/de 등 다국어 자연 포함 → 분류기 비영문 robustness 평가 입력.
- **개선 필요**: `oracle_runner --limit` 가 parquet 앞 N 행(=단일 corpus)만 취함 → 대표성 smoke 위해 `--shuffle`(corpus 교차) 추가, full 은 `LIMIT=0`.
- 샘플 수: smoke 50/corpus → full **500/corpus**(→ 필요시 2000). 6 corpus × 500 = 3,000 prompt.

## 4. oracle 측정 매트릭스 (상세)
**모델 × method × corpus** 격리측정(conc=1, max_tokens=512, temp=0, seed=42, max-model-len 8192).

### 4.1 모델 (3계열 × 크기, B200 1.40TB)
| 티어 | Qwen2.5 | Llama-3 | DeepSeek | 비고 |
|---|---|---|---|---|
| T1 ~7-8B | 7B ✅ | 3.1-8B ✅ | R1-Distill-Qwen-7B ⬇ | vanilla 우세 참조(R≫K) |
| T2 ~32B | 32B ✅ | — | R1-Distill-Qwen-32B ⬇ | gating 유의 |
| T3 ~70B | 72B ✅ | 3.1-70B ✅ | R1-Distill-Llama-70B ⬇ | spec 이득 큼 |
| T4 >70B | — | 3.1-405B(bf16/fp8) ⬇ | V3/R1 671B(fp8) ⬇ | 순차(디스크 874GB) |

### 4.2 method
vanilla / suffix(K=32) / ngram(cap=8) / **eagle**. eagle 은 **모델별 EAGLE head 체크포인트 가용 검사 후**(Llama/Qwen 일부 존재, DeepSeek distill 불확실) — 없으면 해당 모델 3 method.

### 4.3 비용·순서
- 1 phase = 1(model,method) TP=8 부팅(~2분) + oracle(conc=1, 500p × ~1-2s ≈ 8-15분). model당 3-4 method.
- **순서**: T1→T3, corpus open 먼저 → gated. 모델당 **50p smoke 로 wall 외삽** 후 full. method spread<5%(per-model)면 해당 모델 조기종료.
- T4 XL: 405B 받고→측정→정리→671B (순차).

### 4.4 oracle_table 스키마 (long)
`prompt_id, prompt_hash, corpus, lang, n_input_tok, model, method, output_tps, wall_ms, completion_tokens, ok, seed`. 본문 text 로컬 전용. 파생: method spread, model×corpus 평균 tps, oracle-method 분포.

## 5. 품질 평가 (`quality_eval.py` — 상세 설계)
### 경로 A — losslessness (같은 모델, spec vs vanilla)
- 절차: 동일 prompt 셋(subset ~100) 을 temp=0 으로 vanilla / suffix(/eagle) 에 생성, completions `logprobs` 활성.
- 지표: **greedy token exact-match rate**(생성 token id 시퀀스 일치, ≥99% 목표), **per-token logprob max-abs-diff**(BF16<1e-2), KL, **draft acceptance α**, sequence PPL rel-diff(<1%).
- 판정: suffix/ngram 은 lossless 기대(99%+). eagle 은 학습 draft 라 깨질 수 있음 → 경로 B 로 품질 별도 평가.
- 출처: arXiv:2502.05202, vLLM spec-decode docs.

### 경로 B — 품질 벤치 (모델 간 / lossy)
| workload | 벤치/지표 | 도구 | judge |
|---|---|---|---|
| code | **pass@k**(실행) HumanEval + MBPP | `lm-eval` (`lm_eval --tasks humaneval,mbpp`) | 불필요 |
| chat | **win-rate**(pairwise, pos-bias 2-swap) Arena-Hard-Auto / MT-Bench | Arena-Hard-Auto / FastChat llm_judge | **로컬** Qwen2.5-72B 또는 Llama-3.1-70B |
| sonnet/창작 | rubric WildBench/custom | FastChat llm_judge | 로컬 judge |
| reasoning(옵션) | exact-match GSM8K | `lm-eval` | 불필요 |
- 로컬 judge 는 8×B200 에 72B/70B 서빙 → API 비용 0. position-bias 2-swap 의무.
- 출처: Arena-Hard(2406.11939), AlpacaEval LC(2404.04475), LLM-as-judge(2306.05685).

## 6. 라우팅 전략 비교 (`run_routing_compare.sh`)
같은 2-백엔드(vanilla+suffix) 위 **vanilla-only / trident-only / llm-d** 비교(AGSD 는 TSK_043).
- llm-d = K8s-native(Inference Gateway+EPP, KV/prefix-cache-aware + predicted-latency).
- 지표: throughput, TTFT·TPOT p50/p99, KV/prefix cache hit, routing overhead. 6 workload × 3 strategy.
- **인프라 제약**: 이 컨테이너 docker/k8s 없음 → Minikube 는 호스트/별도노드(사용자). **미충족 시 fallback** = llm-d 보류, vanilla/trident 단일백엔드 비교 + oracle/품질만 진행.
- TSK_043(AGSD CPU-optimized) 결과와 cross-reference.

## 7. accept / kill gate
- **kill (TSK_042 단계)**: model×corpus prompt-level method spread `(max−min)/max` < 5% 면 → method 우열 없음 = AGSD/분류기 가치 약함 → 해당 영역 조기종료. (V2: 16.8% → 통과.)
- TSK_043 의 regret/accept 는 oracle_table 을 입력으로 별도 판정.

## 8. SUB 분해 (제안 — ID 부여 시 SUB_199 부터)
| 제안 SUB | 범위 | 상태 |
|---|---|---|
| a. 데이터 파이프라인 | corpus_loader + prompt_sampler + (shuffle/min-tok 보강) | ✅ 구현·V1 |
| b. oracle 코어 | oracle_runner + run_oracle_8gpu + build_oracle_table (단일모델 V2) | ✅ 구현·V2 |
| c. oracle full 매트릭스 | 8+ 모델 × method × 6 corpus full 측정 + T4 XL | ⬜ (모델 다운로드 후) |
| d. 품질 경로 A | losslessness (token match/logprob/KL/α) | ⬜ |
| e. 품질 경로 B | pass@k(HumanEval/MBPP) + judge(Arena-Hard, 로컬 72B) | ⬜ |
| f. 라우팅 비교 | llm-d vs single (Minikube) | ⬜ (인프라) |
> ID 부여는 사용자 지시 시 SUB_199 부터(연구/lever 규칙). a/b 는 이미 구현되어 완료 SUB 로 등록 가능.

## 9. 검증 ladder
V1 corpus(✅) → V2 oracle smoke(✅) → V3 table(✅) → V4 gate dry-run(✅, spread 16.8%) → **V5 oracle full**(8모델×corpus) → **품질 A/B smoke→full** → **V_routing**(llm-d, 인프라 시) → V6 T4 XL.

## 10. 함정 (실측)
실모델명 / `--disable-log-requests` 금지 / oracle 은 max-model-len 8192(출력 512)·full e2e 는 20480 / `kill_pgroup`(pkill 자기매칭 금지) / orphan worker compute-apps PID kill / run_in_background script(포그라운드 sleep·`&` 금지) / mbpp min-tok / `--limit` 단일corpus 편향(→shuffle).
