# INPUT_SOURCES — IDE_071

| ID | 자료 | 위치 | SHA256 (2026-09-16 00:35 KST 계산) |
|---|---|---|---|
| 지시서 | `cpu_offload_no_02` (사용자 업로드 `9a1575df-cpu_offload_no_02_________.md`) | `shadow_assists/features/IDE_071/cpu_offload_no_02_실험계획_지시서.md` | evidence/software_manifest.json `files.instruction` |
| S1 | IDE_069 `FULL_REPORT.md` | `shadow_assists/features/IDE_069/FULL_REPORT.md` | evidence/software_manifest.json `files.ide069_full_report` (지시서 기재값 542723c4…) |
| S2 | IDE_070 `RESULT.md` | `shadow_assists/features/IDE_070/RESULT.md` | evidence/software_manifest.json `files.ide070_result` (지시서 기재값 07144c37…) |
| hotmap v1 | IDE_069 prompt 트레이스 hotmap | `~/.cache/huggingface/kt/ide069/hotmap.json` (컨테이너 `/models/kt/ide069/hotmap.json`) | `files.hotmap_v1` |
| hotmap v2 | IDE_070 1차 측정 트레이스 hotmap | `~/.cache/huggingface/kt/ide070/hotmap_v2.json` | `files.hotmap_v2` |
| layer budget 5,952 | IDE_070 `build_layer_budget.py` 산출 | `~/.cache/huggingface/kt/ide070/layer_budget_5952.json` | `files.layer_budget_5952` |
| hotmap_mixed_0.25 | IDE_034 (2026-09-09) `build_hotmap_mixed.py` α=0.25 | `~/.cache/huggingface/kt/ide070/hotmap_mixed_0.25.json` (원본 `eval/results/20260909_014147_ide034_mixed_hotmap/`) | `files.hotmap_mixed_0.25` |
| 모델 | Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 revision 003f183a92fbe5b9a8325aaa8b2ae797c91dd90f | `~/.cache/huggingface/hub/models--Qwen--…-FP8/snapshots/003f183a…` | `files.model_snapshot` (파일 목록·크기) |
| CPU 가중치 | kt INT4 (`kt quant … -m int4 -i fp8`, IDE_069) | `~/.cache/huggingface/kt/qwen3-480b-int4` | `files.kt_weights` |
| AWQ 체크포인트 | QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ revision 9ce3eaa67fe88609afec235117e97eb03d9b3cda (awq 4-bit g128) | `~/.cache/huggingface/hub/models--QuantTrio--…-AWQ/snapshots/9ce3eaa…` | P8 W00 단계에서 manifest 생성 |
| CODE_MIX | openai/openai_humaneval (MIT) 164 문항 | `~/.cache/huggingface/kt/ide071/datasets/code_mix.jsonl` | `manifests/code_mix_manifest.json` (행별 sha256, jsonl sha256) |
| MIXED_SERVICE | sonnet 행 조합 + HumanEval, 비율 SHORT 60/CODE 20/PREFILL_HEAVY 10/DECODE_HEAVY 10, seed 20260922, 300 요청 | `~/.cache/huggingface/kt/ide071/datasets/mixed_service.jsonl` | `manifests/mixed_service_manifest.json` |
| sonnet | vllm 벤치 동봉 `sonnet.txt` (517행) | 컨테이너 vllm-h100 `/tmp/sonnet.txt`, 사본 `datasets/sonnet.txt` | — |
| GSM8K | openai/gsm8k main test (1,319) revision 740312add88f781978c0658806c59bc2815b9866 | HF cache datasets--openai--gsm8k | — |
| 공개 문서 W1–W9 | 지시서 §21.2 | 로컬 설치본 `--help`·소스로 확인 (evidence/sglang_launch_server_help.txt, option_registry.json) | — |

컨테이너·소프트웨어 식별 (evidence/software_manifest.json): sgl-kt = lmsysorg/sglang:latest (nerdctl, image digest 미제공 — `RepoDigests` 없음), SGLang 0.5.18 git 71de97b2 + 로컬 수정 10 파일, ktransformers git 6d460cc1 + 로컬 수정 10 파일 (kt-kernel 0.7.0.post1 소스 빌드, `kt_kernel_ext…so` sha256 e29357f7…), torch 2.13.0+cu130, flashinfer 0.6.17, triton 3.7.1, NCCL 2.29.7, CUDA 13.0; vllm-h100 = vllm/vllm-openai:nightly-a9a17e70 (vllm 0.26.1rc1.dev1177).
