# test.md — IDE_022 테스트/검증

## TSK_042 검증 단계
- **V1 corpus smoke**: `prompt_sampler --corpora open --n 50` → corpus_stats(dedup/length/lang) 합리성. GPU 불필요.
- **V2 oracle smoke**: Qwen32B × {vanilla,suffix,ngram} × 50p, max_tokens=512, conc=1 → per_request_raw 150행.
- **V3 table**: build_oracle_table → oracle_table.parquet 스키마, method 3개 존재.
- **V4 gate dry-run**: method spread `(max−min)/max` 분포 → kill(method<5%) 판정.
- **V_routing**: Minikube + llm-d-deployer 기동 → 1 workload(TTFT/TPOT/throughput + cache hit) smoke.
- **V5 full**: 8모델 × corpus(gated 포함). 품질 경로A(token match·logprob) + 경로B(LiveCodeBench pass@1 + Arena-Hard-Auto judge) smoke→full.
- **V6 T4 XL**: Llama-405B·DeepSeek-671B 순차 다운로드→측정→정리.

## TSK_044 검증 단계
- **분류기 self-test**: 각 모듈 (prompt,expected) + brittle 케이스. C0 회귀 가드(`c0.predict == workload_classifier.classify`). C1 brittle 개선.
- **pickle smoke**: `ProcessPoolExecutor(2)`로 각 `classify` 호출 가능.
- **regret toy fixture**: 10 prompt×4 method toy parquet → C4 regret==0, regret_abs≥0, is_catastrophic 정의 일치 assert.
- **oracle 합류 full**: regret_eval → corpus×classifier×metric 표.
- **latency smoke**: classify p50/p99, RE2/mimalloc on-off.

## 정확도 게이트
분류기 변경은 라우팅만 → 경로 A 동등성(token match≥99%)으로 재확인. CLAUDE.md 정확도 제약 준수.
