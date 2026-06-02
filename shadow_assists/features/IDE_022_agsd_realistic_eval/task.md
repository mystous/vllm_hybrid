# task.md — IDE_022 단계별 구현

## 0. 선행 (완료 2026-06-01)
- [x] id_registry TSK_042/TSK_043 등록(다음 TSK_044)
- [x] README Trace tree IDE_022→TSK_042/TSK_043 노드
- [x] feature dir + TSK 하위 + per-TSK planning 문서 dump

## TSK_042 — 워크로드 활용 실험 (선행)
- [ ] `vllm_config_perf/gating/realistic_eval/corpus_loader.py` (open default + gated 옵션)
- [ ] `prompt_sampler.py` (dedup/length/lang/stratified → sampled_prompts.parquet)
- [ ] `oracle_runner.py` (concurrency=1 per-prompt, 실모델명)
- [ ] `run_oracle_8gpu.sh` (모델×method phase; run_full_8gpu.sh 변형; ngram/eagle 추가)
- [ ] `build_oracle_table.py` → `oracle_table.parquet` + RESULTS(method spread, kill-gate 1차)
- [ ] `quality_eval.py` (경로 A losslessness + 경로 B 품질벤치)
- [ ] `run_routing_compare.sh` (vanilla/trident/**llm-d** Minikube; AGSD 제외)
- [ ] smoke(open 50p) → full(8모델×corpus, gated) → T4 XL(405B·671B 순차)

## TSK_043 — AGSD: CPU 병렬성 최적화 + 측정 (TSK_042 oracle_table 의존)
- [ ] `vllm_config_perf/gating/classifiers/{base,c0_regex,c1_regex_ext,c2_hashing_lr,c3_minilm_onnx,_re_backend}.py`
- [ ] `TSK_043.../src/{train_c2,train_c3,export_onnx,regret_eval,latency_bench,labels,aggregate}.py`
- [ ] §8 CPU 병렬성 R1(RE2)·R2(mimalloc) 즉시 → R4(C2)·R5(C3 ONNX INT8) → R3/R6 게이트 후
- [ ] 측정: classify latency p50/p99 + **AGSD routing e2e throughput**(lever ON/OFF) + regret(toy→oracle 합류) + aggregate accept/kill

## 의존
TSK_043 분류기 구현·CPU 최적화·latency·AGSD routing 은 독립, regret 평가는 TSK_042 oracle_table 의존. 라우팅 비교(llm-d)는 TSK_042, AGSD CPU 최적화는 TSK_043.
