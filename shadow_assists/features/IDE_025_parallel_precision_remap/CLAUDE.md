# IDE_025 — CLAUDE.md (구현 시 알아야 할 것)

## 확인된 플래그 (본 빌드 v1.7.dev16107 실재, 2026-06-11 코드 확인)

- `-dp N` / `--data-parallel-hybrid-lb` / `--data-parallel-external-lb` (`arg_utils.py:934-980`)
- `--enable-dbo`, `--dbo-decode-token-threshold` (기본 32), `--dbo-prefill-token-threshold` (512)
  — `config/parallel.py:193-203`, 구현 `v1/worker/ubatching.py`
- `--compilation-config '{"pass_config":{"enable_sp":true,"fuse_gemm_comms":true}}'`
  — `config/compilation.py:129,1418` (TP>1 필요)
- `--kv-cache-dtype fp8` (+ `--kv-cache-dtype-skip-layers`)
- `--decode-context-parallel-size N` (dcp_comm_backend=ag_rs)
- NIXL: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`

## 주의

1. **DP 와 spec-decode 병행**: suffix proposer 는 EngineCore 별 독립 suffix cache —
   DP=8 이면 cache 가 8 분할되어 α 가 떨어질 수 있음. 1차 sweep 은 **vanilla 로 N1 단독 판정**
   후 spec 을 얹을 것 (교란 분리).
2. **DP=8 의 메모리**: replica 마다 weight 사본 — 8B×8=128GB (GPU 당 16GB) 문제없음.
   KV 용량은 replica 당 1/1 (192GB−16GB) — 오히려 단일 TP8 보다 per-replica KV 풍부.
3. **prefix cache 분산**: DP 는 replica 간 prefix 공유 불가 — mix corpus 는 prefix 재사용이
   낮아 영향 적지만, 결과 해석 시 명기.
4. **conc 스케일**: throughput_runner `--concurrency` 를 셀별로 32×DP 로. RAW/summ 파일명에
   conc 포함해 구분.
5. cudagraph_mode 명시 부팅 (SUB_212 confounder 교훈). FaP 기본 권장 (vanilla 우세 기측정).
6. port 충돌: DP 는 단일 serve 프로세스가 내부 LB — port 1개면 됨 (`-dp` 네이티브).
   external-lb 모드만 다중 port.

## 기준점 (Llama-8B mix conc=32)

van+PW 8,850 / van+FaP 12,089 / suf K32+PW 27,851 — per-GPU 3,481 (suf 기준)

## 환경

PY=/workspace/vllm_dev_prj/bin/python, VBIN=/workspace/vllm_dev_prj/bin/vllm,
corpus=`vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet`,
harness env 는 SUB_212 와 동일 (ARCTIC_INFERENCE_ENABLED=0 등).
