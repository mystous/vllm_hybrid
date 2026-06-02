# IDE_022 — AGSD Realistic-Workload + Decision-Regret Evaluation

> **parent**: TSK_020 / idea: [`../IDE_006/TSK_020/idea/IDE_022_agsd_realistic_eval.md`](../IDE_006/TSK_020/idea/IDE_022_agsd_realistic_eval.md)
> **자식 TSK**: `TSK_042`(워크로드 활용 실험), `TSK_043`(AGSD CPU 병렬성 최적화)
> **status**: 활성 (계획 완료 → 구현 대기)
> **승인 plan**: `/root/.claude/plans/playful-stargazing-fog.md`

## 이론적 배경 / 동기
현 AGSD 워크로드 분류기 평가의 3대 약점:
1. **regex 분류기 brittle** — 코드블록 없는 코드 요청, chat 인용 속 코드, multi-turn 후반, 비영문에서 오분류.
2. **평가셋이 자기충족적** — sonnet/chat/code 500×3이 분류기 룰에 맞춰 fork 내부 합성 → accuracy 1.000은 in-distribution 자명.
3. **메트릭 misalign** — label accuracy가 아니라 throughput-최선 method 선택이 진짜 목적.

실측 동기: B200 32B 벤치에서 `AGSD × chat −44%`(라우터 `chat→vanilla` 정책 약점) 확인.

## 구현 방향 (2 TSK + 품질 방법론)
> **선행조건**: [`../IDE_006/TSK_020/planning/PRE_TSK042_TSK043_prerequisites.md`](../IDE_006/TSK_020/planning/PRE_TSK042_TSK043_prerequisites.md) (TSK_042/TSK_043 진행 전).
- **TSK_042 워크로드 활용 실험**: 실 trace corpus(LMSYS/WildChat/ShareGPT/LiveCodeBench/SWE-bench) × 모델 매트릭스(Qwen/Llama/DeepSeek, 7B~671B) × method(vanilla/suffix/ngram/eagle)로 (a) per-prompt oracle throughput → `oracle_table.parquet`, (b) 출력 품질(경로 A losslessness / 경로 B 품질벤치), (c) **라우팅 전략 비교 = vanilla/trident/llm-d** (smart router = **llm-d**; AGSD 제외).
- **TSK_043 AGSD CPU 병렬성 최적화 + 측정**: AGSD(분류기 C0~C3 + gating 라우터)의 **CPU 병렬성 최적화**(§8 R1~R6: RE2/mimalloc/feature-hashing/ONNX INT8/Vectorscan/free-threaded) + **측정**(classify latency p50/p99 · AGSD routing e2e throughput · oracle_table 기반 **decision-regret**).

## 출력 품질 비교 방법론 (raw 1:1 비교 대체)
- **경로 A (losslessness)**: 같은 모델, spec vs vanilla / 동일 백엔드 라우팅 → token exact-match·logprob max-abs-diff·KL·acceptance α·PPL rel-diff (출력 "동등" 확인).
- **경로 B (품질 벤치)**: 모델 간 / lossy → code=pass@k(LiveCodeBench/HumanEval, 실행), chat=win-rate/Elo(Arena-Hard-Auto, 로컬 judge), sonnet=rubric(WildBench). BLEU/ROUGE/BERTScore는 보조만.

## 실행 코드 위치
runnable 코드는 라우터 import 호환 위해 `vllm_config_perf/gating/{realistic_eval, classifiers}/`. 측정 산출물은 각 TSK 하위 `measurements/`. 상세 plan은 `../IDE_006/TSK_020/planning/TSK_042_*.md`, `TSK_043_*.md`.
