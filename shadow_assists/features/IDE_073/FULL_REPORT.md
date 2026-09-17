# IDE_073 FULL_REPORT — 2모델 × (GPU-only 8장 / KT 공식 기본 4장 / 최고 구성 4장) 비교·구간 계측 (측정 사실만)

생성 2026-09-17T11:07:43.259637+09:00. campaign_id IDE_073_20260917 · 지시서 cpu_offload_no_04 · 브랜치 feat/cpu-offload-two-models-20260917 · 상태 COMPLETED_WITH_FAILURES

## 1. 모델 자격 검사와 실행 상태

| 모델 | revision | tensor 합 | GPU8 참조 | GPU4 GPU-only 자격 | 4장 하이브리드 |
|---|---|---|---|---|---|
| Qwen3-Coder-480B-A35B-Instruct-FP8 | 003f183a… | 482.1 GB | Q-GPU8: FAILED_REQUESTS | 482 GB > 320 GB → 산정상 불가 (기동 생략; IDE_068 TP4 OOM 기록) | BASIC4 FAILED_BOOT / OPT4 COMPLETED,COMPLETED |
| GLM-4.7-FP8 | 7b3b5f81… | 362.1 GB (93 shard) | G-GPU8: COMPLETED | 362 GB > 320 GB → 산정상 불가 | BASIC4 TIMED_OUT_DURING_QUALITY / OPT4 NOT_RUN,NOT_RUN |

GLM OPT4-TRANSFER 준비 상태: {'status': 'OPT_TRANSFER_BLOCKED_FORMAT'} (state/glm_prepare_state.json)

## 2. 공식 원문 설정·자원 정규화·최적화 설정 diff

BASELINE_PROVENANCE.md, CONFIG_DIFF.md 참조 (본 저장소 동일 디렉터리). 설치본 미지원 옵션: --kt-enable-dynamic-expert-update, --kt-gpu-prefill-token-threshold, --kt-expert-placement-strategy (UNSUPPORTED_OPTION).

## 3. 실행 환경·모델 revision·모든 명령·effective feature

```
violet-h100-016
5.14.0-427.124.1.el9_4.x86_64
no_turbo=1 governor=performance
index, uuid, pci.bus_id, driver_version, memory.total [MiB], power.limit [W], power.max_limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz], clocks.applications.graphics [MHz], compute_mode, ecc.mode.current
0, GPU-3b84c06a-611d-8e07-2726-44a868f646fe, 00000000:0A:00.0, 580.126.20, 81559 MiB, 700.00 W, 700.00 W, 1980 MHz, 2619 MHz, 1980 MHz, Default, Enabled
1, GPU-8a39f0ec-5884-45d9-f056-53ace5e57df8, 00000000:18:00.0, 580.126.20, 81559 MiB, 700.00 W, 700.00 W, 1980 MHz, 2619 MHz, 1980 MHz, Default, Enabled
2, GPU-3f7fea22-2c53-a28e-59a2-07cc6682cad0, 00000000:3A:00.0, 580.126.20, 81559 MiB, 700.00 W, 700.00 W, 1980 MHz, 2619 MHz, 1980 MHz, Default, Enabled
3, GPU-5cfe0e43-68b1-5c29-9082-c085abacf446, 00000000:43:00.0, 580.126.20, 81559 MiB, 700.00 W, 700.00 W, 1980 MHz, 2619 MHz, 1980 MHz, Default, Enabled
4, GPU-43a78ae4-f89f-ca72-2ff2-27e8a8a27f74, 00000000:87:00.0, 580.126.20, 81559 MiB, 700.00 W, 700.00 W, 1980 MHz, 2619 MHz, 1980 MHz, Default, Enabled
5, GPU-11f36bf0-2679-fa1a-3940-f938f0b6a05d, 00000000:90:00.0, 580.126.20, 81559 MiB, 700.00 W, 700.00 W, 1980 MHz, 2619 MHz, 1980 MHz, Default, Enabled
6, GPU-b73567b4-d441-96fe-2b36-39ff827022b2, 00000000:B8:00.0, 580.126.20, 81559 MiB, 700.00 W, 700.00 W, 1980 MHz, 2619 MHz, 1980 MHz, Default, Enabled
7, GPU-f57f9621-9754-17cd-b327-841e495b1b25, 00000000:C1:00.0, 580.126.20, 81559 MiB, 700.00 W, 700.00 W, 1980 MHz, 2619 MHz, 1980 MHz, Default, Enabled
```
- 소프트웨어: torch 2.13.0+cu130 · sglang 0.5.18 · flashinfer 0.6.17 · triton 3.7.1 · cuda 13.0 · nccl (2, 29, 7)
- kt_kernel_ext.so e29357f786a53030296f75beaf3db2745baa730835ddc7d98aca88dd014aa46d
- sglang 로컬 tree 71de97b + 10파일; 공식 worktree /sgl-workspace/sglang-upstream (diff 0)

| cell | model | profile | attempt | 상태 | boot s | mismatch | per_layer 합 | cf 행 | capture bs | pin 이동 | kt 코어 | env | launch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G-GPU8 | glm | GPU8 | a1 | COMPLETED | 328 | [] | None | 0 | [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64] | None | 0 | `{}` | ` CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server --host 127.0.0.1 --port 30000 --tp 8 --context-length 32768 --trust-remote-code --kv-cache-dtype auto --max-total-tokens 40960 --mem-fraction-static 0.9 --max-running-requests 64 --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --model-path /models/hub/models--zai-org--GLM-4.7-FP8/snapshots/7b3b5f81eee81be12a6f8da2710eac4bafb0166a --served-model-name glm47 --attention-backend flashinfer --fp8-gemm-backend triton --enable-p2p-check --disable-shared-experts-fusion --enable-mixed-chunk --chunked-prefill-size 4096 --ep-size 8` |
| G-KT-BASIC4 | glm | BASIC4 | a1 | TIMED_OUT_DURING_QUALITY | 243 | [] | None | 0 | [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64] | None | 100 | `{}` | ` CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --host 127.0.0.1 --port 30000 --tp 4 --context-length 32768 --trust-remote-code --kv-cache-dtype auto --max-total-tokens 40960 --mem-fraction-static 0.95 --max-running-requests 64 --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --model-path /models/hub/models--zai-org--GLM-4.7-FP8/snapshots/7b3b5f81eee81be12a6f8da2710eac4bafb0166a --served-model-name glm47 --attention-backend flashinfer --fp8-gemm-backend triton --enable-p2p-check --disable-shared-experts-fusion --enable-mixed-chunk --chunked-prefill-size 4096 --kt-weight-path /models/hub/models--zai-org--GLM-4.7-FP8/snapshots/7b3b5f81eee81be12a6f8da2710eac4bafb0166a --kt-method FP8_PERCHANNEL --kt-cpuinfer 100 --kt-threadpool-count 2 --kt-num-gpu-experts 80 --kt-max-deferred-experts-per-token 2` |
| Q-GPU8 | qwen | GPU8 | a1 | FAILED_REQUESTS | 77 | [] | None | 0 | [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64] | None | 0 | `{}` | ` CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server --host 127.0.0.1 --port 30000 --tp 8 --context-length 32768 --trust-remote-code --kv-cache-dtype auto --max-total-tokens 40960 --mem-fraction-static 0.9 --max-running-requests 64 --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name qwen480 --ep-size 8 --attention-backend triton --chunked-prefill-size 8192` |
| Q-GPU8 | qwen | GPU8 | a2 | COMPLETED | 77 | [] | None | 0 | [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64] | None | 0 | `{}` | ` CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server --host 127.0.0.1 --port 30000 --tp 8 --context-length 32768 --trust-remote-code --kv-cache-dtype auto --max-total-tokens 40960 --mem-fraction-static 0.9 --max-running-requests 64 --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name qwen480 --ep-size 8 --attention-backend triton --chunked-prefill-size 8192` |
| Q-KT-BASIC4 | qwen | BASIC4 | a1 | FAILED_BOOT | 42 | None | None | None | null | None | 0 | `{"PYTHONPATH": "/sgl-workspace/sglang-upstream/python"}` | `PYTHONPATH=/sgl-workspace/sglang-upstream/python CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --host 127.0.0.1 --port 30000 --tp 4 --context-length 32768 --trust-remote-code --kv-cache-dtype auto --max-total-tokens 40960 --mem-fraction-static 0.95 --max-running-requests 64 --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name qwen480 --attention-backend triton --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 112 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --kt-max-deferred-experts-per-token 2 --chunked-prefill-size 8192` |
| Q-KT-BASIC4 | qwen | BASIC4 | a2 | FAILED_BOOT | 36 | None | None | None | null | None | 0 | `{"PYTHONPATH": "/sgl-workspace/sglang-upstream/python"}` | `PYTHONPATH=/sgl-workspace/sglang-upstream/python CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --host 127.0.0.1 --port 30000 --tp 4 --context-length 32768 --trust-remote-code --kv-cache-dtype auto --max-total-tokens 40960 --mem-fraction-static 0.95 --max-running-requests 64 --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name qwen480 --attention-backend triton --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 112 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --kt-max-deferred-experts-per-token 2 --chunked-prefill-size 8192` |
| Q-KT-BASIC4 | qwen | BASIC4 | a3 | FAILED_BOOT | 36 | None | None | None | null | None | 0 | `{"PYTHONPATH": "/sgl-workspace/sglang-upstream/python"}` | `PYTHONPATH=/sgl-workspace/sglang-upstream/python CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --host 127.0.0.1 --port 30000 --tp 4 --context-length 32768 --trust-remote-code --kv-cache-dtype auto --max-total-tokens 40960 --mem-fraction-static 0.95 --max-running-requests 64 --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name qwen480 --attention-backend triton --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 112 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --kt-max-deferred-experts-per-token 2 --chunked-prefill-size 8192` |
| Q-KT-BASIC4 | qwen | BASIC4 | a4 | COMPLETED | 117 | [] | None | 0 | [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64] | None | 112 | `{}` | ` CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --host 127.0.0.1 --port 30000 --tp 4 --context-length 32768 --trust-remote-code --kv-cache-dtype auto --max-total-tokens 40960 --mem-fraction-static 0.95 --max-running-requests 64 --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name qwen480 --attention-backend triton --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 112 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --kt-max-deferred-experts-per-token 2 --chunked-prefill-size 8192` |
| Q-OPT4-b1 | qwen | OPT4 | a1 | COMPLETED | 122 | [] | 5952 | 1 | [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64] | 1197 | 96 | `{"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1", "KT_GPU_EXPERTS_PER_LAYER": "/models/kt/ide070/layer_budget_5952.json", "KT_AVX_RB": "0"}` | `KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json KT_AVX_RB=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --host 127.0.0.1 --port 30000 --tp 4 --context-length 32768 --trust-remote-code --kv-cache-dtype auto --max-total-tokens 40960 --mem-fraction-static 0.95 --max-running-requests 64 --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name qwen480 --attention-backend triton --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide070/hotmap_v2.json --kt-max-deferred-experts-per-token 8 --ep-dispatch-algorithm dynamic --chunked-prefill-size 8192` |
| Q-OPT4-b2 | qwen | OPT4 | a1 | COMPLETED | 114 | [] | 5952 | 1 | [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64] | 1197 | 96 | `{"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1", "KT_GPU_EXPERTS_PER_LAYER": "/models/kt/ide070/layer_budget_5952.json", "KT_AVX_RB": "0"}` | `KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json KT_AVX_RB=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --host 127.0.0.1 --port 30000 --tp 4 --context-length 32768 --trust-remote-code --kv-cache-dtype auto --max-total-tokens 40960 --mem-fraction-static 0.95 --max-running-requests 64 --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name qwen480 --attention-backend triton --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide070/hotmap_v2.json --kt-max-deferred-experts-per-token 8 --ep-dispatch-algorithm dynamic --chunked-prefill-size 8192` |

bench 명령: 각 rep 디렉터리 bench_cmd.sh. config/<model>/<profile>/ 에 argv.json·effective_config.json·effective_feature_paths.json·affinity.

## 4. 모든 반복의 성공·실패·입출력 token·duration·처리량·지연

| model | profile | cell | attempt | rep | workload | C | n | 성공/실패 | dur s | in tok | out tok | out tok/s | total tok/s | TTFT p50/p95/p99 | TPOT p50/p95/p99 | ITL p50/p95 | E2EL p50/p95 | pooled TTFT p95 / TPOT p95 | valid | measure_start | flush |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glm | GPU8 | G-GPU8 | a1 | M073_GLM_MAIN_SHORT_C64_rep1 | M073_GLM_MAIN_SHORT | 64 | 256 | 256/0 | 26.5 | 134904 | 32768 | 1238.03 | 6334.9 | 801/2379/2625 | 41.6/44.0/45.9 | 29.1/245.3 | 6116/7665 | 2379 / 44.0 | True | 2026-09-17T09:12:32.613830+09:00 | Cache flushed.
Please check ba |
| glm | GPU8 | G-GPU8 | a1 | M073_GLM_MAIN_SHORT_C64_rep2 | M073_GLM_MAIN_SHORT | 64 | 256 | 256/0 | 26.5 | 134904 | 32768 | 1238.75 | 6338.6 | 797/2367/2620 | 41.8/43.8/45.9 | 29.1/251.4 | 6108/7668 | 2367 / 43.8 | True | 2026-09-17T09:13:19.052984+09:00 | Cache flushed.
Please check ba |
| glm | GPU8 | G-GPU8 | a1 | M073_GLM_MAIN_SHORT_C64_rep3 | M073_GLM_MAIN_SHORT | 64 | 256 | 256/0 | 26.6 | 134904 | 32768 | 1231.53 | 6301.6 | 807/2408/2659 | 42.1/43.8/46.1 | 29.1/252.9 | 6165/7770 | 2408 / 43.8 | True | 2026-09-17T09:14:05.329463+09:00 | Cache flushed.
Please check ba |
| glm | GPU8 | G-GPU8 | a1 | M073_GLM_LOW_CONCURRENCY_C1_rep1 | M073_GLM_LOW_CONCURRENCY | 1 | 16 | 16/0 | 25.8 | 8426 | 2048 | 79.36 | 405.9 | 255/289/350 | 10.6/10.7/10.7 | 10.6/10.8 | 1605/1639 | 264 / 10.7 | True | 2026-09-17T09:14:51.816351+09:00 | Cache flushed.
Please check ba |
| glm | GPU8 | G-GPU8 | a1 | M073_GLM_LONGER_PREFILL_C8_rep1 | M073_GLM_LONGER_PREFILL | 8 | 32 | 32/0 | 17.9 | 131283 | 4096 | 228.78 | 7561.6 | 1294/2203/2349 | 25.1/28.5/28.8 | 17.1/18.5 | 4453/5336 | 2087 / 28.4 | True | 2026-09-17T09:15:36.883197+09:00 | Cache flushed.
Please check ba |
| glm | BASIC4 | G-KT-BASIC4 | a1 | M073_GLM_MAIN_SHORT_C64_rep1 | M073_GLM_MAIN_SHORT | 64 | 256 | 256/0 | 705.5 | 134904 | 32768 | 46.45 | 237.7 | 45840/120958/125747 | 1010.3/1137.0/1287.9 | 389.7/2019.5 | 173261/249693 | 120958 / 1137.0 | True | 2026-09-17T09:23:20.681354+09:00 | Cache flushed.
Please check ba |
| glm | BASIC4 | G-KT-BASIC4 | a1 | M073_GLM_MAIN_SHORT_C64_rep2 | M073_GLM_MAIN_SHORT | 64 | 256 | 256/0 | 705.2 | 134904 | 32768 | 46.46 | 237.8 | 45669/120843/125665 | 1005.3/1133.1/1287.5 | 390.1/2178.2 | 173123/249339 | 120843 / 1133.1 | True | 2026-09-17T09:35:26.018113+09:00 | Cache flushed.
Please check ba |
| glm | BASIC4 | G-KT-BASIC4 | a1 | M073_GLM_MAIN_SHORT_C64_rep3 | M073_GLM_MAIN_SHORT | 64 | 256 | 256/0 | 705.7 | 134904 | 32768 | 46.44 | 237.6 | 45823/120589/125358 | 1006.5/1136.6/1286.1 | 389.7/2022.3 | 173453/249025 | 120589 / 1136.6 | True | 2026-09-17T09:47:31.159270+09:00 | Cache flushed.
Please check ba |
| glm | BASIC4 | G-KT-BASIC4 | a1 | M073_GLM_LOW_CONCURRENCY_C1_rep1 | M073_GLM_LOW_CONCURRENCY | 1 | 16 | 16/0 | 117.0 | 8426 | 2048 | 17.51 | 89.6 | 2075/2096/2109 | 41.0/42.1/42.8 | 41.0/43.3 | 7293/7428 | 2091 / 41.8 | True | 2026-09-17T09:59:36.778397+09:00 | Cache flushed.
Please check ba |
| glm | BASIC4 | G-KT-BASIC4 | a1 | M073_GLM_LONGER_PREFILL_C8_rep1 | M073_GLM_LONGER_PREFILL | 8 | 32 | 32/0 | 550.1 | 131283 | 4096 | 7.45 | 246.1 | 59937/118182/120192 | 610.4/816.2/817.5 | 138.5/146.3 | 137576/200654 | 116572 / 815.6 | True | 2026-09-17T10:01:53.358916+09:00 | Cache flushed.
Please check ba |
| qwen | GPU8 | Q-GPU8 | a1 | M073_QWEN_MAIN_SHORT_C64_rep1 | M073_QWEN_MAIN_SHORT | 64 | 256 | null/null | null | null | null | null | null | null/null/null | null/null/null | null/null | null/null | null / null | False | 2026-09-17T08:07:52.734859+09:00 | Cache flushed.
Please check ba |
| qwen | GPU8 | Q-GPU8 | a1 | M073_QWEN_MAIN_SHORT_C64_rep2 | M073_QWEN_MAIN_SHORT | 64 | 256 | null/null | null | null | null | null | null | null/null/null | null/null/null | null/null | null/null | null / null | False | 2026-09-17T08:08:12.868957+09:00 | Cache flushed.
Please check ba |
| qwen | GPU8 | Q-GPU8 | a1 | M073_QWEN_MAIN_SHORT_C64_rep3 | M073_QWEN_MAIN_SHORT | 64 | 256 | null/null | null | null | null | null | null | null/null/null | null/null/null | null/null | null/null | null / null | False | 2026-09-17T08:08:32.844518+09:00 | Cache flushed.
Please check ba |
| qwen | GPU8 | Q-GPU8 | a1 | M073_QWEN_LOW_CONCURRENCY_C1_rep1 | M073_QWEN_LOW_CONCURRENCY | 1 | 16 | null/null | null | null | null | null | null | null/null/null | null/null/null | null/null | null/null | null / null | False | 2026-09-17T08:08:52.920145+09:00 | Cache flushed.
Please check ba |
| qwen | GPU8 | Q-GPU8 | a1 | M073_QWEN_LONGER_PREFILL_C8_rep1 | M073_QWEN_LONGER_PREFILL | 8 | 32 | null/null | null | null | null | null | null | null/null/null | null/null/null | null/null | null/null | null / null | False | 2026-09-17T08:09:12.848471+09:00 | Cache flushed.
Please check ba |
| qwen | GPU8 | Q-GPU8 | a2 | M073_QWEN_MAIN_SHORT_C64_rep1 | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 21.1 | 131786 | 32768 | 1555.36 | 7810.7 | 1013/1489/5061 | 29.7/32.4/34.5 | 26.5/27.7 | 4741/4813 | 1489 / 32.4 | True | 2026-09-17T08:14:34.063353+09:00 | Cache flushed.
Please check ba |
| qwen | GPU8 | Q-GPU8 | a2 | M073_QWEN_MAIN_SHORT_C64_rep2 | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 21.1 | 131786 | 32768 | 1553.60 | 7801.8 | 1007/1482/5075 | 29.6/32.3/34.4 | 26.5/27.9 | 4758/4803 | 1482 / 32.3 | True | 2026-09-17T08:15:14.924408+09:00 | Cache flushed.
Please check ba |
| qwen | GPU8 | Q-GPU8 | a2 | M073_QWEN_MAIN_SHORT_C64_rep3 | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 21.1 | 131786 | 32768 | 1555.67 | 7812.2 | 1012/1486/5085 | 29.6/32.3/34.3 | 26.6/27.8 | 4747/4783 | 1486 / 32.3 | True | 2026-09-17T08:15:56.441097+09:00 | Cache flushed.
Please check ba |
| qwen | GPU8 | Q-GPU8 | a2 | M073_QWEN_LOW_CONCURRENCY_C1_rep1 | M073_QWEN_LOW_CONCURRENCY | 1 | 16 | 16/0 | 24.8 | 8234 | 2048 | 82.50 | 414.2 | 188/196/197 | 10.7/10.8/10.8 | 10.7/10.9 | 1551/1559 | 196 / 10.8 | True | 2026-09-17T08:16:37.830159+09:00 | Cache flushed.
Please check ba |
| qwen | GPU8 | Q-GPU8 | a2 | M073_QWEN_LONGER_PREFILL_C8_rep1 | M073_QWEN_LONGER_PREFILL | 8 | 32 | 32/0 | 15.3 | 131147 | 4096 | 267.39 | 8828.8 | 1046/1460/1478 | 21.9/27.0/27.0 | 18.9/19.7 | 3824/3853 | 1446 / 27.0 | True | 2026-09-17T08:17:22.376144+09:00 | Cache flushed.
Please check ba |
| qwen | BASIC4 | Q-KT-BASIC4 | a4 | M073_QWEN_MAIN_SHORT_C64_rep1 | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 238.4 | 131786 | 32768 | 137.43 | 690.1 | 15903/20470/61561 | 327.1/388.5/388.5 | 290.4/305.4 | 57496/57781 | 20470 / 388.5 | True | 2026-09-17T08:35:23.415830+09:00 | Cache flushed.
Please check ba |
| qwen | BASIC4 | Q-KT-BASIC4 | a4 | M073_QWEN_MAIN_SHORT_C64_rep2 | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 231.9 | 131786 | 32768 | 141.31 | 709.6 | 15657/20557/36619 | 315.6/377.8/377.8 | 281.3/291.7 | 56174/56263 | 20557 / 377.8 | True | 2026-09-17T08:39:41.623534+09:00 | Cache flushed.
Please check ba |
| qwen | BASIC4 | Q-KT-BASIC4 | a4 | M073_QWEN_MAIN_SHORT_C64_rep3 | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 229.7 | 131786 | 32768 | 142.66 | 716.4 | 15503/19839/59592 | 312.9/371.1/371.1 | 280.7/287.5 | 55298/55334 | 19839 / 371.1 | True | 2026-09-17T08:43:53.301635+09:00 | Cache flushed.
Please check ba |
| qwen | BASIC4 | Q-KT-BASIC4 | a4 | M073_QWEN_LOW_CONCURRENCY_C1_rep1 | M073_QWEN_LOW_CONCURRENCY | 1 | 16 | 16/0 | 67.4 | 8234 | 2048 | 30.39 | 152.6 | 469/480/480 | 29.4/29.9/30.3 | 29.4/31.0 | 4204/4268 | 480 / 29.8 | True | 2026-09-17T08:48:02.711118+09:00 | Cache flushed.
Please check ba |
| qwen | BASIC4 | Q-KT-BASIC4 | a4 | M073_QWEN_LONGER_PREFILL_C8_rep1 | M073_QWEN_LONGER_PREFILL | 8 | 32 | 32/0 | 124.8 | 131147 | 4096 | 32.82 | 1083.7 | 15393/19889/19942 | 123.9/202.1/202.9 | 90.6/97.7 | 31254/31309 | 19853 / 201.6 | True | 2026-09-17T08:49:29.686779+09:00 | Cache flushed.
Please check ba |
| qwen | OPT4 | Q-OPT4-b1 | a1 | M073_QWEN_MAIN_SHORT_C64_rep1 | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 47.5 | 131786 | 32768 | 689.43 | 3462.2 | 2204/3615/10875 | 72.0/83.2/86.1 | 63.6/75.4 | 11375/11978 | 3615 / 83.1 | True | 2026-09-17T08:22:46.510423+09:00 | Cache flushed.
Please check ba |
| qwen | OPT4 | Q-OPT4-b1 | a1 | M073_QWEN_LOW_CONCURRENCY_C1_rep1 | M073_QWEN_LOW_CONCURRENCY | 1 | 16 | 16/0 | 32.4 | 8234 | 2048 | 63.13 | 316.9 | 252/301/304 | 13.9/14.0/14.0 | 13.9/14.2 | 2021/2068 | 300 / 14.0 | True | 2026-09-17T08:23:53.682687+09:00 | Cache flushed.
Please check ba |
| qwen | OPT4 | Q-OPT4-b1 | a1 | M073_QWEN_LONGER_PREFILL_C8_rep1 | M073_QWEN_LONGER_PREFILL | 8 | 32 | 32/0 | 25.2 | 131147 | 4096 | 162.26 | 5357.5 | 2501/3253/3349 | 30.2/42.6/42.7 | 24.2/26.0 | 6305/6434 | 3236 / 42.5 | True | 2026-09-17T08:24:45.449890+09:00 | Cache flushed.
Please check ba |
| qwen | OPT4 | Q-OPT4-b2 | a1 | M073_QWEN_MAIN_SHORT_C64_rep1 | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 46.4 | 131786 | 32768 | 705.73 | 3544.0 | 2282/3253/11297 | 68.9/80.5/80.8 | 63.9/72.7 | 11017/11259 | 3253 / 80.5 | True | 2026-09-17T08:29:44.851231+09:00 | Cache flushed.
Please check ba |
| qwen | OPT4 | Q-OPT4-b2 | a1 | M073_QWEN_MAIN_SHORT_C64_rep2 | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 43.7 | 131786 | 32768 | 749.68 | 3764.7 | 2199/3123/6499 | 60.4/71.7/74.6 | 57.1/66.9 | 10253/10412 | 3123 / 70.9 | True | 2026-09-17T08:30:51.227367+09:00 | Cache flushed.
Please check ba |

### 통계 (valid 반복만)

| model | profile | workload | C | n_valid | out tok/s 원값 | 평균 | 중앙값 | 최소 | 최대 | sd(ddof=1) | mean_of_rep TTFT p95 | mean_of_rep TPOT p95 | boots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glm | GPU8 | M073_GLM_MAIN_SHORT | 64 | 3 | 1238.03, 1238.75, 1231.53 | 1236.10 | 1238.03 | 1231.53 | 1238.75 | 3.98 | 2385 | 43.9 | ['G-GPU8'] |
| glm | GPU8 | M073_GLM_LOW_CONCURRENCY | 1 | 1 | 79.36 | 79.36 | 79.36 | 79.36 | 79.36 | null | 289 | 10.7 | ['G-GPU8'] |
| glm | GPU8 | M073_GLM_LONGER_PREFILL | 8 | 1 | 228.78 | 228.78 | 228.78 | 228.78 | 228.78 | null | 2203 | 28.5 | ['G-GPU8'] |
| glm | BASIC4 | M073_GLM_MAIN_SHORT | 64 | 3 | 46.45, 46.46, 46.44 | 46.45 | 46.45 | 46.44 | 46.46 | 0.01 | 120797 | 1135.6 | ['G-KT-BASIC4'] |
| glm | BASIC4 | M073_GLM_LOW_CONCURRENCY | 1 | 1 | 17.51 | 17.51 | 17.51 | 17.51 | 17.51 | null | 2096 | 42.1 | ['G-KT-BASIC4'] |
| glm | BASIC4 | M073_GLM_LONGER_PREFILL | 8 | 1 | 7.45 | 7.45 | 7.45 | 7.45 | 7.45 | null | 118182 | 816.2 | ['G-KT-BASIC4'] |
| qwen | GPU8 | M073_QWEN_MAIN_SHORT | 64 | 3 | 1555.36, 1553.60, 1555.67 | 1554.87 | 1555.36 | 1553.60 | 1555.67 | 1.12 | 1486 | 32.3 | ['Q-GPU8'] |
| qwen | GPU8 | M073_QWEN_LOW_CONCURRENCY | 1 | 1 | 82.50 | 82.50 | 82.50 | 82.50 | 82.50 | null | 196 | 10.8 | ['Q-GPU8'] |
| qwen | GPU8 | M073_QWEN_LONGER_PREFILL | 8 | 1 | 267.39 | 267.39 | 267.39 | 267.39 | 267.39 | null | 1460 | 27.0 | ['Q-GPU8'] |
| qwen | BASIC4 | M073_QWEN_MAIN_SHORT | 64 | 3 | 137.43, 141.31, 142.66 | 140.47 | 141.31 | 137.43 | 142.66 | 2.72 | 20289 | 379.1 | ['Q-KT-BASIC4'] |
| qwen | BASIC4 | M073_QWEN_LOW_CONCURRENCY | 1 | 1 | 30.39 | 30.39 | 30.39 | 30.39 | 30.39 | null | 480 | 29.9 | ['Q-KT-BASIC4'] |
| qwen | BASIC4 | M073_QWEN_LONGER_PREFILL | 8 | 1 | 32.82 | 32.82 | 32.82 | 32.82 | 32.82 | null | 19889 | 202.1 | ['Q-KT-BASIC4'] |
| qwen | OPT4 | M073_QWEN_MAIN_SHORT | 64 | 3 | 689.43, 705.73, 749.68 | 714.95 | 705.73 | 689.43 | 749.68 | 31.16 | 3330 | 78.5 | ['Q-OPT4-b1', 'Q-OPT4-b2'] |
| qwen | OPT4 | M073_QWEN_LOW_CONCURRENCY | 1 | 1 | 63.13 | 63.13 | 63.13 | 63.13 | 63.13 | null | 301 | 14.0 | ['Q-OPT4-b1'] |
| qwen | OPT4 | M073_QWEN_LONGER_PREFILL | 8 | 1 | 162.26 | 162.26 | 162.26 | 162.26 | 162.26 | null | 3253 | 42.6 | ['Q-OPT4-b1'] |

### REPLAY (OPT4 b1, 워밍업과 같은 생성 인자 C16·32 재전송)

| cell | 완료/실패 | out tok/s | healthy_after | 입력 비고 |
|---|---|---|---|---|
| Q-OPT4-b1 | 32/0 | 482.15 | True | ORIGINAL_FAILURE_INPUT_UNAVAILABLE: IDE_071 load1024/a1 워밍업 프롬프트 원문 미보존 → 같은 생성 인자(sonnet seed, 32 요청, C16) 로 재생성; token 동일성은 seed 만으로 단정하지 않음 |

## 5. layer·rank·stage 구간 시간, 노출 대기·overlap·critical path

기록: qwen D2 는 2회 실행. 1차(bootA_retry, 10:21) 트레이스는 프로브 코드가 /start_profile activities 를 ['CPU','CUDA'] 로 보내(SGLang 키는 'GPU') GPU 활동 0건 → INVALID_NO_GPU_ACTIVITY(원장 status_correction). 2차(bootA_retry2, 10:56, activities ['CPU','GPU']) 트레이스를 §5·profiles/ 에 사용. 2차 트레이스 span 40.96 s 중 첫 16.4 s 는 프로파일 시작(10:56:01)부터 첫 요청 도착 전(벤치 클라이언트 기동) 구간이며 busy 비율·gap max 에 그대로 포함됨(보정 없음).

### qwen 프로브 세션

| 세션 | rc | out tok/s | duration s | throughput_ratio | elapsed_ratio | 비고 |
|---|---|---|---|---|---|---|
| D0 | 0 | 676.13 | 24.2 | null | null |  |
| D2 | 0 | 686.33 | 23.9 | null | null | trace [['D2_1789610161-TP-0.trace.json.gz', 66462764], ['D2_1789610161-TP-1.trace.json.gz', 33290536], ['D2_1789610161-TP-2.trace.json.gz', 33314637], ['D2_1789610161-TP-3.trace.json.gz', 33245802]] |
| D3 | 0 | 686.76 | 23.9 | 1.016 | 0.985 |  |
| D1 | 0 | 703.75 | 23.3 | 1.041 | 0.961 | cpu_jobs rows 754 |

# layer/stage aggregate — qwen (프로브 산출의 기계적 집계)

## CPU expert 작업 (D1, KT_PHASE_PROF 64회당 1회 표본, µs; numa 서브풀별 1행 = 층 작업의 절반)

표본 754 행

| qlen 구간 | n | activated_expert p50 | total p50/p95/p99 | up_gate p50 | down p50 | q_input p50 | cpy_input p50 | weight p50 |
|---|---|---|---|---|---|---|---|---|
| decode(qlen≤80) | 732 | 8 | 652/1224/1451 | 353 | 214 | 18 | 15 | 17 |
| prefill(qlen>80) | 22 | 33 | 5494/6968/7500 | 1544 | 1355 | 135 | 769 | 1197 |

activated_expert 별 total µs 중앙값 (decode): 0:10(n=180), 1:120(n=54), 2:168(n=12), 3:270(n=4), 4:348(n=16), 5:434(n=9), 6:485(n=28), 7:561(n=26), 8:622(n=45), 9:680(n=42), 10:774(n=32), 11:815(n=54), 12:880(n=50), 13:952(n=33), 14:1008(n=50), 15:1078(n=31), 16:1144(n=18), 17:1192(n=20), 18:1286(n=13), 19:1376(n=2), 20:1402(n=4), 21:1475(n=5), 22:1554(n=2), 23:1570(n=2)

[kt-tq] TaskQueue 집계 (마지막 3행):

```
Profiling Results (numa[1]): activated_expert: 1, prepare: 3 us, cpy_input: 1 us, q_input: 4 us, up_gate: 70 us, act:[kt-tq] n=2048 exec_us p50=24 p90=144 p99=283 mean=41 | hist(<5,<30,<100,<300,<600,<1k,<2k,<4k,>=4k)=1016,376,276,368,11,1,0,0,0 | wait_us p50=0 p90=203 p99=223 mean=76 | busy=0.35
[kt-tq] n=2048 exec_us p50=24 p90=143 p99=296 mean=43 | hist(<5,<30,<100,<300,<600,<1k,<2k,<4k,>=4k)=1016,319,305,388,19,1,0,0,0 | wait_us p50=0 p90=207 p99=222 mean=75 | busy=0.37
Profiling Results (numa[1]): activated_expert: 0, prepare: 1 us, cpy_input: 0 us, q_input: 0 us, up_gate: 2 us, act: 0 us, q_down: 1 us, down: 1 us, weight: 5 us, total: 11 us, max_local_n[kt-tq] n=2048 exec_us p50=24 p90=138 p99=201 mean=31 | hist(<5,<30,<100,<300,<600,<1k,<2k,<4k,>=4k)=1016,445,33
```

## GPU 타임라인 (D2, SGLang torch profiler CPU+CUDA activity; rank 별 트레이스)

| rank trace | events | GPU ops | GPU busy s | span s | busy 비율 | 커널 간 유휴 gap: n / p50 / p95 / p99 / max µs / 합 s | MoE 커널 수 | MoE 커널 사이 유휴 p50/p95/p99/max µs |
|---|---|---|---|---|---|---|---|---|
| D2_1789610161-TP-0.trace.json.gz | 1523341 | 825802 | 16.438 | 40.961 | 0.401 | 774666 / 0.5126953125 / 10.462890625 / 354.33203125 / 16411161.338867188 / 24.888 | 114266 | 1.087890625/470.2685546875/848.150390625/408809.0439453125 |
| D2_1789610161-TP-1.trace.json.gz | 1363766 | 604262 | 31.758 | 40.962 | 0.775 | 549947 / 0.4482421875 / 1.1826171875 / 2.431640625 / 16411838.08203125 / 17.526 | 114266 | 1.4072265625/6.4326171875/17.3447265625/411287.5341796875 |
| D2_1789610161-TP-2.trace.json.gz | 1364053 | 604262 | 32.151 | 40.961 | 0.785 | 550522 / 0.4482421875 / 1.15234375 / 2.0791015625 / 16411531.216796875 / 17.261 | 114266 | 1.376953125/6.4326171875/7.392578125/408457.1513671875 |
| D2_1789610161-TP-3.trace.json.gz | 1363992 | 604262 | 32.149 | 40.962 | 0.785 | 549089 / 0.4482421875 / 1.18359375 / 2.1435546875 / 16412551.061523438 / 17.289 | 114266 | 1.4072265625/6.43359375/7.6494140625/411893.7744140625 |

### 상위 GPU 커널 (첫 rank 트레이스, 이름 원문, 호출 수, 합계 µs)

| cat | kernel | n | sum µs |
|---|---|---|---|
| gpu_user_annotation | `scheduler.run_batch` | 397 | 24101270.5 |
| gpu_user_annotation | `step[DECODE bs=63]` | 256 | 15083939.9 |
| kernel | `fused_moe_kernel` | 49228 | 7134431.7 |
| gpu_user_annotation | `step[EXTEND bs=17 toks=8192]` | 5 | 3570243.0 |
| gpu_user_annotation | `step[DECODE bs=2]` | 128 | 2007244.0 |
| gpu_memcpy | `Memcpy DtoH (Device -> Pinned)` | 98853 | 1870112.6 |
| gpu_memcpy | `Memcpy HtoD (Pinned -> Device)` | 24716 | 1774908.8 |
| kernel | `_w8a8_block_fp8_matmul` | 49228 | 1618872.5 |
| kernel | `void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lampor` | 48375 | 846696.5 |
| gpu_user_annotation | `step[EXTEND bs=16 toks=8192]` | 1 | 810709.5 |
| gpu_user_annotation | `step[EXTEND bs=15 toks=7171]` | 1 | 689915.1 |
| gpu_user_annotation | `step[EXTEND bs=11 toks=5620]` | 1 | 579748.4 |
| kernel | `void sglang::per_token_group_quant_flat_kernel<sglang::QuantTrait<__nv_bfloat16,` | 98456 | 555873.0 |
| gpu_user_annotation | `nccl:all_reduce` | 1000 | 476082.4 |
| kernel | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1000 | 476080.4 |
| kernel | `_fwd_grouped_kernel_stage1` | 23870 | 420042.3 |
| gpu_user_annotation | `step[EXTEND bs=4 toks=2013]` | 1 | 362605.4 |
| gpu_user_annotation | `step[EXTEND bs=2 toks=1018]` | 1 | 324876.4 |
| gpu_user_annotation | `step[EXTEND bs=1 toks=510]` | 1 | 266003.5 |
| gpu_user_annotation | `step[EXTEND bs=1 toks=1]` | 1 | 265506.3 |
| kernel | `void moe_sum_reduce_warp_per_token_vec_kernel<8>(c10::BFloat16 const*, c10::BFlo` | 682 | 146664.3 |
| kernel | `_fwd_kernel` | 744 | 134922.4 |
| gpu_user_annotation | `scheduler.get_next_batch_to_run` | 397 | 134557.8 |
| kernel | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKerne` | 1381 | 131617.0 |
| kernel | `_fwd_kernel_stage2` | 23870 | 96552.8 |
| kernel | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind)0, true, ` | 24614 | 92160.1 |
| kernel | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c1` | 24614 | 86285.0 |
| kernel | `nvjet_sm90_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 15872 | 78632.5 |
| kernel | `_router_triton_kernel` | 24614 | 73605.4 |
| kernel | `void sglang::fused_rope_store_kernel<true, 128l, true, __nv_bfloat16, __nv_bfloa` | 24614 | 69225.1 |
| kernel | `triton_poi_fused__to_copy_arange_floor_divide_index_remainder_view_0` | 24614 | 68633.3 |
| kernel | `void moe_sum_reduce_kernel<c10::BFloat16, 8>(c10::BFloat16 const*, c10::BFloat16` | 15872 | 66908.5 |
| kernel | `void sglang::fused_qknorm_warp<128l, true, __nv_bfloat16>(sglang::QKNormParams)` | 24614 | 59994.7 |
| kernel | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda` | 27026 | 57994.2 |
| kernel | `void moe_align_block_size_kernel<int>(int const*, int*, int*, int*, int, int, un` | 16554 | 51146.5 |
| gpu_user_annotation | `scheduler.process_batch_result` | 15 | 45184.6 |
| kernel | `nvjet_sm90_tst_288x64_64x4_2x1_v_bz_TNN` | 256 | 44205.6 |
| kernel | `void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_twoshot_sync<(` | 250 | 37745.4 |
| kernel | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __n` | 23994 | 37587.4 |
| kernel | `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>,` | 24614 | 37485.7 |

의존 대기(cold_exposed_wait 등): 트레이스에 CPU→GPU done 신호(memop)·H2D 복사 완료 이벤트가 커널과 별도 항목으로 식별되지 않아 `PARTIAL_DEPENDENCY_MEASUREMENT` — 위 'MoE 커널 사이 GPU 유휴' 가 결합 전 대기의 상한 관측치이며, hot/cold 어느 경로가 늦었는지는 이 트레이스만으로 분리되지 않음.

## CPU 타임라인 (D3, perf record 99 Hz 20 s, 스케줄러 4 프로세스; 상위 25 심볼)

```
     2.87%     0.00%  sglang::schedul  [unknown]                                                                                      [.] 0000000000000000
     2.68%     0.00%  sglang::schedul  [unknown]                                                                                      [k] 0x8b485355fc894900
     2.68%     0.00%  sglang::schedul  libtorch_python.so                                                                             [.] torch::PyWarningHandle
     2.63%     0.00%  sglang::schedul  libtorch_python.so                                                                             [.] THCPEvent_synchronize(
     2.63%     0.00%  sglang::schedul  libcudart.so.13                                                                                [.] cudaEventSynchronize
     2.63%     0.00%  sglang::schedul  libcuda.so.580.126.20                                                                          [.] cuEventSynchronize
     1.44%     1.27%  sglang::schedul  kt_kernel_ext.cpython-312-x86_64-linux-gnu.so                                                  [.] std::thread::_State_im
     1.15%     0.37%  sglang::schedul  python3.12                                                                                     [.] _PyEval_EvalFrameDefau
     1.06%     0.00%  sglang::schedul  python3.12                                                                                     [.] PyEval_EvalCode
     1.03%     0.00%  sglang::schedul  python3.12                                                                                     [.] _start
     1.03%     0.00%  sglang::schedul  libc.so.6                                                                                      [.] __libc_start_main
     1.03%     0.00%  sglang::schedul  python3.12                                                                                     [.] Py_BytesMain
     1.03%     0.00%  sglang::schedul  python3.12                                                                                     [.] Py_RunMain
     1.03%     0.00%  sglang::schedul  python3.12                                                                                     [.] PyRun_SimpleStringFlag
     1.03%     0.00%  sglang::schedul  python3.12                                                                                     [.] PyRun_StringFlags
     1.01%     0.01%  pt_gloo_runloop  [kernel.kallsyms]                                                                              [k] entry_SYSCALL_64_after
     0.99%     0.01%  pt_gloo_runloop  [kernel.kallsyms]                                                                              [k] do_syscall_64
```

perf stat 20 s:

```
 Performance counter stats for process id '1181181,1181306,1181374,1181489':
         41,652.16 msec task-clock                       #    2.082 CPUs utilized
    81,688,853,588      cycles                           #    1.961 GHz
   152,425,034,373      instructions                     #    1.87  insn per cycle
           433,946      context-switches                 #   10.418 K/sec
             5,870      cpu-migrations                   #  140.929 /sec
      20.004523646 seconds time elapsed
```

pcm-memory (2 s, MB/s 단위는 pcm-memory 헤더 정의 'System Read/Write'): 샘플 10, 읽기 평균 2592 MB/s, 쓰기 평균 1732 MB/s, 읽기 최대 5892 MB/s



## 6. expert hit/Cold 선택·rows 분포·deferred 통계

D1 LIGHT_PROBE 의 cpu_jobs.csv (KT_PHASE_PROF, 64회당 1회 표본: numa, activated_expert, prepare/cpy_input/q_input/up_gate/act/q_down/down/weight/total µs, max_local_num, qlen). 층 id·step id·expert id 는 이 계측이 출력하지 않음 → NOT_COLLECTED(layer/expert id). deferred produced/consumed/stale 카운터: NOT_COLLECTED (계측 없음).

## 7. CPU·GPU·DRAM·NUMA 시계열

각 rep: cpu_timeseries.csv (1 s, 코어별 busy, all/phys/smt/kt), gpu_timeseries.csv (2 s), memory_timeseries.csv (5 s). D3: perf_report_top.txt, perf_stat_20s.txt, pcm_memory.csv (2 s, MB/s 정의는 pcm-memory 헤더). 경계: timestamps.json (measure_start/end).

## 8. 계측 ON/OFF 간섭과 누락

§5 표의 throughput_ratio/elapsed_ratio. 미수집: S00 토큰화·스케줄러 대기(요청별 TTFT 만), S02 router 시간(트레이스 커널명 수준), S03/S06 복사 bytes(트레이스 memcpy 크기), S08 의존 대기(PARTIAL_DEPENDENCY_MEASUREMENT), S12 동적 배치(해당 없음).

## 9. 동일 문항별 출력·기계적 채점·짝비교

quality/paired_outputs.md (문항별 전문 포함). 요약:

| cell | model | profile | exact | unscored | truncated | errors |
|---|---|---|---|---|---|---|
| G-GPU8 | glm | GPU8 | 19/20 | 0 | 0 | 0 |
| G-KT-BASIC4 | glm | BASIC4 | 0/20 | 8 | 12 | 8 |
| Q-GPU8 | qwen | GPU8 | 19/20 | 0 | 0 | 0 |
| Q-KT-BASIC4 | qwen | BASIC4 | 19/20 | 0 | 0 | 0 |
| Q-OPT4-b1 | qwen | OPT4 | 20/20 | 0 | 0 | 0 |

## 10. 모든 오류·재시도·미실행 항목

failures/README.md. manifest 상태:

| cell | 계획 | 상태 |
|---|---|---|
| Q-GPU8 | GPU-only 8장 참조: 자격(smoke)·MAIN×3·LOW·LONG·품질20 (1 부팅) | COMPLETED |
| Q-KT-BASIC4 | 공식 기본 4장: MAIN×3·LOW·LONG·품질20 (1 부팅, 공식 checkout PYTHONPATH) | COMPLETED |
| Q-OPT4-b1 | 최고 구성 4장 첫 부팅: MAIN×1·LOW·LONG·품질20·REPLAY | COMPLETED |
| Q-OPT4-b2 | 최고 구성 새 부팅: MAIN×2 (재기동 재현) | COMPLETED |
| G-GPU8 | GPU-only 8장 참조: 자격(smoke)·MAIN×3·LOW·LONG·품질20 (1 부팅) | COMPLETED |
| G-KT-BASIC4 | 성능 5세션 유효, 품질 20문항 부분(INVALID_TIMED_OUT), 품질 예산 120/120 소진으로 재실행 없음 | TIMED_OUT_DURING_QUALITY |
| G-OPT4-b1 | 최고 구성 4장 첫 부팅: MAIN×1·LOW·LONG·품질20·REPLAY | NOT_RUN_DEPENDENCY |
| G-OPT4-b2 | 최고 구성 새 부팅: MAIN×2 (재기동 재현) | NOT_RUN_DEPENDENCY |

원장 재계산: PERF 25/30 · DIAG 6/8 · 재시도·조건부 2/6 · 세션 33/44 · 부팅 14/20 · 품질 120/120

## 11. 30분 보고 이력 전체

| # | 예정 | 실제 | 경과 s | 단계 | 현재 | 항목 수 | 최근 완료 | GLM bytes | HBM | 오류 | 전달 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-09-17T07:45:00+0900 | 2026-09-17T07:55:33+0900 | 633 | PREPARING | GLM-4.7-FP8 다운로드 + 환경 동결 + 하네스 준비 | {'registered': 0} | None | 43963106757 | 0, 0 1, 0 2, 0 3, 0 4, 0 5, 0 6, 0 7, 0 | None | saved |
| 1 | 2026-09-17T08:15:00+0900 | 2026-09-17T08:15:15+0900 | 1815 | qwen/GPU8 | {'cell': 'Q-GPU8', 'model': 'qwen', 'profile': 'GPU8', 'atte | {'completed': 0, 'failed': 1, 'blocked': 0, 'registered': 1} | ["Q-GPU8:FAILED_REQUESTS reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 0), ('M073_QWEN_MAIN_SHORT_C64_rep2', 0), ('M073_QWEN_M | 177181242161 | 0, 63752 1, 63848 2, 63848 3, 63848 4, 63848 5, 63848 6, 638 | ['Q-GPU8: rc=1 completed=None failed=None n=16', 'Q-GPU8: rc=1 completed=None fa | saved |
| 2 | 2026-09-17T08:15:00+0900 | 2026-09-17T08:19:13+0900 | 81253 | None | None | None | None | None | None | None | delivered |
| 3 | 2026-09-17T08:45:00+0900 | 2026-09-17T08:25:05+0900 | 81605 | None | None | None | None | None | None | None | delivered |
| 4 | 2026-09-17T08:45:00+0900 | 2026-09-17T08:45:17+0900 | 3617 | qwen/BASIC4 | {'cell': 'Q-KT-BASIC4', 'model': 'qwen', 'profile': 'BASIC4' | {'completed': 3, 'failed': 1, 'blocked': 0, 'registered': 4} | ["Q-GPU8:FAILED_REQUESTS reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 0), ('M073_QWEN_MAIN_SHORT_C64_rep2', 0), ('M073_QWEN_M | 362091875343 | 0, 79432 1, 79342 2, 79342 3, 78862 4, 0 5, 0 6, 0 7, 0 | ['Q-KT-BASIC4: DIED'] | saved |
| 5 | 2026-09-17T08:45:00+0900 | 2026-09-17T08:53:54+0900 | 83334 | None | None | None | None | None | None | None | delivered |
| 6 | 2026-09-17T09:15:00+0900 | 2026-09-17T08:55:03+0900 | 83403 | None | None | None | None | None | None | None | delivered |
| 7 | 2026-09-17T09:15:00+0900 | 2026-09-17T09:15:19+0900 | 5419 | glm/GPU8 | {'cell': 'G-GPU8', 'model': 'glm', 'profile': 'GPU8', 'attem | {'completed': 4, 'failed': 0, 'blocked': 0, 'registered': 4} | ["Q-GPU8:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 1555.4), ('M073_QWEN_MAIN_SHORT_C64_rep2', 1553.6), ('M073_QW | 362091875343 | 0, 50070 1, 50166 2, 50166 3, 50166 4, 50166 5, 50166 6, 501 | ['Q-KT-BASIC4: DIED'] | saved |
| 8 | 2026-09-17T09:15:00+0900 | 2026-09-17T09:25:06+0900 | 85206 | None | None | None | None | None | None | None | delivered |
| 9 | 2026-09-17T09:45:00+0900 | 2026-09-17T09:45:02+0900 | 7202 | glm/BASIC4 | {'cell': 'G-KT-BASIC4', 'model': 'glm', 'profile': 'BASIC4', | {'completed': 5, 'failed': 0, 'blocked': 0, 'registered': 5} | ['Q-KT-BASIC4:FAILED_BOOT reps=[]', 'Q-KT-BASIC4:FAILED_BOOT reps=[]', "Q-OPT4-b1:COMPLETED reps=[('M073_QWEN_MAIN_SHORT | 362091875343 | 0, 54898 1, 54444 2, 54444 3, 53964 4, 0 5, 0 6, 0 7, 0 | ['Q-KT-BASIC4: DIED'] | saved |
| 10 | 2026-09-17T09:45:00+0900 | 2026-09-17T09:55:13+0900 | 87013 | None | None | None | None | None | None | None | delivered |
| 11 | 2026-09-17T10:15:00+0900 | 2026-09-17T10:15:04+0900 | 9004 | glm/BASIC4 | {'cell': 'G-KT-BASIC4', 'model': 'glm', 'profile': 'BASIC4', | {'completed': 5, 'failed': 0, 'blocked': 0, 'registered': 5} | ['Q-KT-BASIC4:FAILED_BOOT reps=[]', 'Q-KT-BASIC4:FAILED_BOOT reps=[]', "Q-OPT4-b1:COMPLETED reps=[('M073_QWEN_MAIN_SHORT | 362091875343 | 0, 54836 1, 54678 2, 54678 3, 54198 4, 0 5, 0 6, 0 7, 0 | ['Q-KT-BASIC4: DIED'] | saved |
| 12 | 2026-09-17T10:15:00+0900 | 2026-09-17T10:18:45+0900 | 88425 | None | None | None | None | None | None | None | delivered |
| 13 | 2026-09-17T10:45:00+0900 | 2026-09-17T10:45:06+0900 | 10806 | glm/BASIC4 | None | {'completed': 5, 'failed': 0, 'blocked': 0, 'registered': 6} | ['Q-KT-BASIC4:FAILED_BOOT reps=[]', "Q-OPT4-b1:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 689.4), ('M073_QWEN_LOW | 362091875343 | 0, 0 1, 0 2, 0 3, 0 4, 0 5, 0 6, 0 7, 0 | ['Q-KT-BASIC4: DIED'] | saved |
| 14 | 2026-09-17T10:45:00+0900 | 2026-09-17T10:48:43+0900 | 90223 | None | None | None | None | None | None | None | delivered |
| 15 | 2026-09-17T11:15:00+0900 | 2026-09-17T11:07:28+0900 | 12148 | glm/BASIC4 | None | {'completed': 5, 'failed': 0, 'blocked': 0, 'registered': 6} | ['Q-KT-BASIC4:FAILED_BOOT reps=[]', "Q-OPT4-b1:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 689.4), ('M073_QWEN_LOW | 362091875343 | 0, 0 1, 0 2, 0 3, 0 4, 0 5, 0 6, 0 7, 0 | ['Q-KT-BASIC4: DIED'] | saved |

## 12. 파일 색인·Git 게시·다운로드

- ARTIFACT_INDEX.csv: 690 파일, 0.28 GB (5 MB 초과·profiles 원본은 서버 보존). SHA256SUMS.txt.
- PUBLISH_RECEIPT.md (게시 후 작성). FULL_RAW_DATA.md + raw_md/part-*.md.

