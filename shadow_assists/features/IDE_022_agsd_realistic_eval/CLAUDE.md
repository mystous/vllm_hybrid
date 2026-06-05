# CLAUDE.md — IDE_022 구현 시 알아야 할 것

## 환경 (이번 세션 실측)
- HW = DGX B200 8×(1.40TB HBM), CUDA 툴킷 12.8(컨테이너 내부), 드라이버 580. vllm는 **sm_100 재빌드 완료**(`host_vllm_hybrid`, editable via `/workspace/vllm_dev_prj`).
- venv = `/workspace/vllm_dev_prj/bin/python` (CPython 3.12). vllm 실행 = `/workspace/vllm_dev_prj/bin/vllm`.
- 캐시 모델: Qwen2.5-0.5/7/32/72B, Llama-3.1-8/70B. DeepSeek·405B·671B는 다운로드 필요. 디스크 ~874GB 여유 → 대형은 순차.
- 빌드/환경 함정: 메모리 `b200-vllm-build`, 직전 벤치 `agsd-32b-benchmark`.

## 반드시 지킬 함정 (실측)
- completions 요청에 **실제 모델명** 전달(`"model":"model"` → 404).
- `--disable-log-requests` **미지원** → 쓰지 말 것.
- `--max-model-len 20480` (input ~8.4k + output 8.2k 가 16640 초과해 400 에러).
- **`pkill -f "vllm serve"` 자기-매칭 금지** — PID·process-group kill(`kill_pgroup`).
- orphan TP worker(`VLLM::Worker`)는 `nvidia-smi --query-compute-apps=pid` PID 직접 kill.
- 포그라운드 Bash `sleep`·`&` 막힘 → 장시간은 **run_in_background 스크립트**.
- gated HF: 웹 라이선스 동의 + `hf auth login`(또는 상위 HF_TOKEN, 로컬 전용).

## 재사용 자산 (`vllm_config_perf/gating/`)
- `run_full_8gpu.sh`: 3-phase 백엔드 기동 — `start_one`/`wait_ready`/`kill_pgroup`/`wait_gpu_free`.
- `benchmark_workloads.py`: `_request_one`(실모델명), `_load_resources`/`_ntok`.
- `finalize_results.py`: run dir → 표/CSV 집계.
- `workload_classifier.py`: C0 regex(재사용·래핑, 재구현 금지). `agsd_router.py`: `classify` import, ProcessPool — ABI 무변경.

## ID 규칙
SUB는 연구/lever 탐색에만 부여. TSK_042/TSK_043/TSK_044 등록됨, 다음 TSK_045. (TSK_043=Host-Side Slack Reclamation, TSK_044=AGSD CPU 분류기; 2026-06-05 기존 043↔신규 재번호). id_registry 단일 출처 + README Trace tree.
