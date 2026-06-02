# TSK_043 — AGSD: CPU 병렬성 최적화 + 측정 (상세 plan)

> **parent**: `IDE_022` (TSK_020). 선행: `TSK_042`(oracle_table), `PRE_TSK042_TSK043_prerequisites.md`.
> **status**: 활성 (구현 대기). 승인 plan: `/root/.claude/plans/playful-stargazing-fog.md`
> **코드**: `vllm_config_perf/gating/classifiers/` + `features/IDE_022_agsd_realistic_eval/TSK_043_cpu_classifier/src/`
> **불변식**: 라우터 ABI(`classify(str)->str`, ProcessPool pickle) **무변경 호환**.

## 1. 목표
**AGSD**(= CPU-side 분류기 + gating 라우터)의 **CPU 병렬성을 최적화하고 측정**한다. (TSK_042 의 라우팅 비교는 llm-d 가 담당; AGSD 는 본 TSK 에서 다룬다.)
1. **분류기 구현** C0(regex)/C1(ext)/C2(feature-hashing)/C3(MiniLM ONNX INT8).
2. **CPU 병렬성 최적화** §8 R1~R6 (RE2 / mimalloc / feature-hashing / ONNX INT8 / Vectorscan / free-threaded+ThreadPool+Arrow) — AGSD classify/라우팅이 request critical path 위라 저지연·고병렬이 핵심.
3. **측정**: (a) classify **latency p50/p99** + 처리량(req/s), CPU 병렬성 lever ON/OFF, (b) **AGSD routing e2e throughput**(canonical, B200), (c) `oracle_table` 기반 **decision-regret**(분류 품질). 현 regex(C0) 대비 개선폭.

## 2. 신규 코드
### 2.1 분류기 (`vllm_config_perf/gating/classifiers/`)
- `base.py` — `BaseClassifier` Protocol: `predict(str)->WorkloadType`, `predict_batch`, `load()`, `name`.
- `c0_regex.py` — 기존 `workload_classifier.classify` **래핑**(재구현 금지). 회귀 가드.
- `c1_regex_ext.py` — 확장 regex: markdown code-block, CJK/다국어, chat 인용(`User:`/`> `).
- `c2_hashing_lr.py` — feature-hashing(`HashingVectorizer` 2^18, char+word ngram) + LogReg.
- `c3_minilm_onnx.py` — all-MiniLM-L6-v2 + 3-class head, **ONNX INT8**, lazy singleton, batch.
- `_re_backend.py` — R1: `import re2` try/except → `re` fallback. R3: vectorscan.
- 각 모듈 module-level `classify(str)->str`(worker별 lazy singleton) export → 라우터 `AGSD_CLASSIFIER` env 선택. **1차 라우터 코드 변경 0**.

### 2.2 학습·평가 (`TSK_043_cpu_classifier/src/`)
- `labels.py` — oracle_table → best-method label, workload↔method 매핑(라우터 `BACKEND_FOR_WORKLOAD` 정합).
- `train_c2.py`/`train_c3.py`/`export_onnx.py` — auto-label = oracle best. C3 ONNX export + dynamic INT8.
- `regret_eval.py` — **핵심**: oracle_table + 분류기 → regret(mean/p99/CDF/zero·catastrophic율). pyarrow/numpy만 의존 → oracle 전 toy fixture 로 완성.
- `latency_bench.py` — classify p50/p99 (독립). RE2·mimalloc on-off.
- `aggregate.py` — regret_summary → RESULTS.md 표 + §4 accept/kill 판정.

## 3. §8 권고 우선순위 + 의존
R1(re→RE2)·R2(mimalloc LD_PRELOAD) **즉시** → R4(C2 hashing)·R5(C3 ONNX INT8) **중** → R3(Vectorscan)·R6(free-threaded 3.13+ThreadPool+Arrow) **게이트 후**(venv 3.12라 3.13t 별도). 설치(uv, `/workspace/vllm_dev_prj`): `google-re2 scikit-learn onnxruntime optimum[onnxruntime] sentence-transformers transformers`. 전부 try/except optional → 미설치 시 C0/C1+regret(pyarrow/numpy)만 동작.

## 4. regret 정의 / gate
`regret_abs=oracle_tps−picked_tps≥0`, `is_zero=picked==oracle`, `is_catastrophic=picked==argmin`. accept: C0 mean regret>5% 또는 catastrophic>10%. kill: method spread<5%(TSK_042 1차) 또는 C2/C3 가 C0 대비 무개선(>80% 동일).

## 5. 보고 양식 (RESULTS.md)
**(a) 분류 품질·latency**: | corpus | classifier | mean regret% | p99 regret% | zero율 | catastrophic율 | classify p50ms | p99ms | (C0~C3 + C4 oracle ceiling × 5 corpus, + per-language 분해).
**(b) CPU 병렬성 ablation (AGSD)**: §8 lever ON/OFF 별 classify 처리량(req/s)·latency + AGSD routing e2e throughput. | lever | classify req/s | p99 ms | AGSD e2e tps |
| baseline(re, ProcessPool) | … | … | … |
| +R1 RE2 | … | | |
| +R2 mimalloc | … | | |
| +R4 C2 hashing | … | | |
| +R5 C3 ONNX INT8 | … | | |
| (+R6 free-threaded) | … | | |
→ CPU 병렬성 최적화가 AGSD routing throughput 에 주는 net 효과 측정.

## 6. 의존성 / 검증
분류기 구현·latency 는 TSK_042 없이 독립. regret 평가는 oracle_table 의존. 검증: self-test(C0 회귀 가드 + C1 brittle) → pickle smoke → regret toy fixture(C4==0 assert) → oracle 합류 full → latency smoke.

## 7. 함정
라우터 ABI 유지(module-level pickle classify) / optional import try/except / C0 재구현 금지 / 라우터 코드 변경은 R6까지 미룸.
