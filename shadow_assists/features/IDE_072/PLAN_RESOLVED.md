# PLAN_RESOLVED — IDE_072 (지시서 cpu_offload_no_03_compact_verification)

## 식별·입력
- campaign_id `IDE_072_20260916` · 브랜치 `feat/cpu-offload-ide071` (IDE_071 과 같은 승인 브랜치, 새 디렉터리) · 노드 violet-h100-016 · 하네스 `eval/ide071/` 재사용 (+ `run_verification.py`, `gsm20_paired.py`, `report_verification.py`)
- 결과: `eval/results/IDE_072_20260916/<cell>/<attempt>/…` · 원장 `state/execution_events.jsonl` · 분기 `state/branch_trace.jsonl`
- 입력 자료 SHA256 (실행 환경에서 계산): 지시서 사본 2a339320…, [S1] IDE_071 FULL_REPORT.md ed93206e… (지시서 기재값과 일치), [S3] compact 지시서 3e5077e0… (일치), [S2] IDE_070 RESULT.md 07144c37… (evidence/software_manifest.json)

## S2 원본 대조 (SOURCE_CONFIG_CONFLICT 검사)
- 원본 `eval/results/IDE_071_20260916/compact/S2_b3_v2_nu5952__confirm/a1/launch_cmd.sh`: `KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path …/003f183a… --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide070/hotmap_v2.json --kt-max-deferred-experts-per-token 8 --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled --mem-fraction-static 0.95 --max-total-tokens 40960` — 지시서 §2.2 명세와 동일.
- 명령 외부 절차: per-layer 패치 (`eval/ide070/patch_per_layer_experts.sh`, 부팅 전 적용·종료 후 원복), 비-kt 스레드 pinning (kt 워커 cpu 0–47, 56–103 실측 → 비-kt 스레드 1,197개를 48–55,104–111,160–167,216–223 으로) — run_cell.py 가 동일 절차 수행.
- 실효값 (`effective_config.json`): kv_cache_dtype auto → 서버 로그 `KV Cache is allocated. dtype: torch.bfloat16, #tokens: 40960`; chunked_prefill_size 8192 (기본값); page_size 1; max_running_requests None; per_layer 합 5,952; `[kt-cf] callback-free handoff ready` 1행.
- **SOURCE_CONFIG_CONFLICT (기록)**: `/get_server_info` 의 `cuda_graph_max_bs` = None, `cuda_graph_bs` = None (요청 `--cuda-graph-max-bs 64`, 서버가 '--cuda-graph-max-bs is deprecated, use --cuda-graph-max-bs-decode' 경고). 서버 로그 `server_args=ServerArgs(...)`: `cuda_graph_max_bs_decode=64`, `cuda_graph_bs_decode=None`, `cuda_graph_padding=False`; capture 로그 `Capture target decode CUDA graph begin. backend=full, bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]`, 4 rank 각 13.5 s. 세 출처(요청/서버정보/로그)를 모두 보존; 이번 실행도 같은 인자를 사용하고 runtime_proofs.json 에 ServerArgs 행·capture 행을 저장.

## 구성 (env 는 명시적 whitelist; docker exec 새 셸이라 상속 없음)
| ID | env | server_args (R0 대비) | pin | patch | 세션 |
|---|---|---|---|---|---|
| V0 | KT_CALLBACK_FREE=1, KT_CF_SKIP_EMPTY_IMM=1, KT_GPU_EXPERTS_PER_LAYER=…/layer_budget_5952.json | hotmap_v2, deferred 8 | ON | per_layer | 첫 부팅: smoke·warmup·SHORT_COLD 1·REPLAY; 확인 부팅: SHORT_COLD 3·ALT 1·GSM20 |
| V1 | V0 + `KT_AVX_RB=0` | 동일 | ON | per_layer | 동일 |
| V2 | KT_CALLBACK_FREE=1, KT_GPU_EXPERTS_PER_LAYER (skip-empty 변수 unset) | 동일 | ON | per_layer | 첫 부팅: smoke·warmup·SHORT_COLD 1·REPLAY·GSM20 |
| Q0 | V2 env | deferred 0 | ON | per_layer | smoke·warmup·GSM20 만 |
| D0 | 오류 관측 첫 구성 | + `--disable-cuda-graph` | 동일 | 동일 | 조건부, 32 요청 |

- `KT_AVX_RB`: 소스 `amx_kernels.hpp:1769` `getenv("KT_AVX_RB") != nullptr` → V0(미설정) = 레지스터 블로킹 경로 미사용, V1(문자열 "0") = 사용. 문자열 값이 아니라 존재 여부로 결정됨을 그대로 기록; V1 ≠ V0 실효 경로 (DUPLICATE 아님).
- `KT_CF_SKIP_EMPTY_IMM`: `experts_base.py:853` `bool(os.environ.get(...))` → OFF 는 unset 만 (빈 문자열·"0" 도 truthy). V2/Q0 는 unset.
- deferral 0: `select_deferred_experts` protected_k = topk − max_deferred = 8 → deferred 없음 (KT_COLD_DEFER 미설정). Q0 는 정확성 기준선이 아님.

## 워크로드
- SHORT_COLD: sonnet 512/128, prefix 0, seed 20260916, C64·256, ignore_eos, 매 반복 전 flush (+3 s), 워밍업(C16·32, 같은 seed) 후에도 flush. IDE_071 과 같은 생성 인자 (input 129,237 / output 32,768 tokens 기대; 실측 비교 기록).
- REPLAY (WARM_SERVER_REPLAY): 주 벤치 후 같은 서버에서 워밍업과 같은 생성 인자(C16·32) 재전송. 원본 실패 입력(IDE_071 load1024/a1 워밍업 프롬프트 원문) 미보존 → `ORIGINAL_FAILURE_INPUT_UNAVAILABLE`, seed 동일성만으로 입력 동일을 주장하지 않음.
- COMPACT_ALT (seed 20261002, prefix 100), COMPACT_LOAD (seed 20261003, 1,024 요청) — IDE_071 정의 그대로. `CALIBRATION_OVERLAP_UNVERIFIED`.
- GSM20_PAIRED: GSM8K test 행 0..39 (원본 GSM40 = `gsm_eval.py` head(40)) 정렬 후 앞 20 (index 0..19); greedy, 같은 user prompt template, max_tokens 1024, 동시성 4. 과거 GSM40 (768 tokens, 40문항) 과 분모·한도가 다름.

## 예산·watchdog
- PERF 11 / REPLAY 3 / LOAD 1 / DIAG 1 / RETRY 2 / 세션 18 / 부팅 10 / GSM 80문항. 원장 `execution_events.jsonl` 에서 재계산.
- 부팅 600 s, 일반·REPLAY 300 s, LOAD·GSM 600 s. 자동 연장 없음.
- 30분 보고: `report_verification.py` (파일) + 세션 cron (사용자 전달, delivery_log).
