# IDE_024 — CLAUDE.md (구현 시 알아야 할 것)

## 핵심 제약 (코드에서 확정된 사실)

1. `uniform_decode_query_len = 1 + num_spec_tokens` 는 **init 고정**
   (`vllm/v1/worker/gpu_model_runner.py:908`, `vllm/v1/cudagraph_dispatcher.py:37`).
   → 스텝별 K 변경으로는 FULL cudagraph 를 못 탄다. 적응형은 **고정 K + pad ON/OFF** 만.
2. FULL graph 적중 조건 (`_is_uniform_decode`, gpu_model_runner.py:3993):
   `max_num_scheduled_tokens == 1+K` **AND** `num_tokens == max × num_reqs` — 전 요청 동일 길이.
3. pad ON/OFF 의 단일 제어점 = `vllm/v1/spec_decode/suffix_decoding.py:424` (SUB_213 lever).
   env `VLLM_SUFFIX_PAD_UNIFORM` + flag file `/tmp/vllm_l3_VLLM_SUFFIX_PAD_UNIFORM.flag`
   (EngineCore spawn 이 env 못 받는 경우 대비 — 양쪽 모두 세팅/제거할 것).
4. 출력 등가 논거: pad 토큰은 rejection sampler 가 항상 기각. 정확도 게이트는
   CLAUDE.md(root) Constraint 운영 해석 (분포 유사성 binding, token 일치 informational).

## 측정 수칙 (SUB_212 confounder 교훈)

- 모든 A/B 는 **cudagraph_mode 를 명시** 부팅 (`--compilation-config '{"cudagraph_mode":...}'`)
  — 묵시 기본값 차이가 +36% 급 confounder 였음.
- boot log 에서 `cudagraph_mode` 라인 grep 으로 실제 적용 확인 후 bench.
- 호스트 상태 (DSA WQ clients, HugePages, uptime) 를 측정 전 기록.
- 사전 예측을 측정 **전** 문서에 commit (IDE_023 P0 프로토콜).

## 환경

- harness env: `ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0`
- PY=/workspace/vllm_dev_prj/bin/python, VBIN=/workspace/vllm_dev_prj/bin/vllm
- 기준 corpus: `vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet` (mix, limit 500)
- 기준점: van+PW 8,850 / van+FaP 12,089 / suf K32+PW 27,851 / suf K32+FaP 24,407 (Llama-8B mix conc=32)

## 재사용 인프라

- regime_detector (`vllm/v1/lhc/regime_detector.py`) — 런타임 신호 (KV%, swap rate)
- L3 instrumentation (`suffix_decoding.py` `_l3_record`) — per-req accept_len 부기 (EMA 소스)
- aggregate.py (SUB_212) — oracle 테이블 생성기 베이스
- sweep harness 패턴: `SUB_212_optimal_dsa_6point/sweep_corpus.sh` (wait_ready/kill_pgroup/wait_gpu_free)
