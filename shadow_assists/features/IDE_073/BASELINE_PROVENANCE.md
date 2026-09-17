# BASELINE_PROVENANCE — IDE_073 (지시서 §3)

설정 출처 분류: `OFFICIAL_MODEL_RECIPE` / `OFFICIAL_GENERAL_GUIDE` / `HARDWARE_WORKLOAD_ADAPTATION` / `LOCAL_OPTIMIZATION`. 설치본 = SGLang 0.5.18 @71de97b (sglang-kt), kt-kernel 0.7.0.post1. 공식 문서 snapshot: `evidence/sources/` (W3 native tutorial = GitHub raw main, 2026-09-17 07:49 KST; ktransformers.net W1/W2/W12 는 curl 응답 0 byte → `SOURCE_SNAPSHOT_UNAVAILABLE`, 지시서 §14 요약을 근거로 사용).

## 설치본에서 지원되지 않는 공식 예 옵션 (→ 기본군 명세에서 제외·기록)
| 옵션 | 설치본 `--help` | 처리 |
|---|---|---|
| `--kt-expert-placement-strategy uniform` | 없음 | 미지정 = `--init-expert-location` 없음 → 물리 id 0..N-1 = 논리 0..N-1 (uniform) 로 동작 (kt_ep_wrapper: `_gpu_mask[:num_gpu_experts]`). 동일 의미 |
| `--kt-enable-dynamic-expert-update` | 없음 | `UNSUPPORTED_OPTION` — GLM 기본군의 runtime dynamic update ON 은 이 설치본에서 실행 불가. OFF 로 실행하고 명세와의 차이를 CONFIG_DIFF 에 기록 |
| `--kt-gpu-prefill-token-threshold 2048` | 없음 | `UNSUPPORTED_OPTION` — layerwise GPU prefill 분기 없음 (kt_kernel `run_layerwise_fp8_batch` 는 block-FP8 전용) |
| `--fp8-gemm-backend triton`, `--enable-p2p-check`, `--disable-shared-experts-fusion`, `--enable-mixed-chunk` | 있음 | 사용 |
| `--kt-method FP8_PERCHANNEL` | 있음 (kt_kernel INFERENCE_METHODS 포함, `AMXFP8PerChannel_MOE` 확장 존재 여부는 부팅으로 확인) | 사용 |

## Q-KT-BASIC4 (Qwen3-Coder-480B FP8, GPU 0–3, TP4)
| 설정 | 값 | 분류 |
|---|---|---|
| GPU 체크포인트 | 003f183a… (FP8, block-quantized) | OFFICIAL_MODEL_RECIPE (모델 카드) |
| CPU backend / weights | AMXINT4 / `/models/kt/qwen3-480b-int4` (같은 revision 에서 `kt quant -m int4 -i fp8`, IDE_069) | OFFICIAL_GENERAL_GUIDE (AMX 변환 경로) |
| kt-cpuinfer | 112 (물리 코어 수) | OFFICIAL_GENERAL_GUIDE (물리 코어 기준) |
| threadpool | 2 (NUMA 2) | OFFICIAL_GENERAL_GUIDE |
| kt-num-gpu-experts | 96 (uniform, hotmap 없음) | HARDWARE_WORKLOAD_ADAPTATION (H100 4장 적재 기록) |
| deferred / token | 2 | OFFICIAL_GENERAL_GUIDE (보수적 권고 범위) |
| KT_* 로컬 env | 모두 unset | — |
| non-KT pinning | 없음 (upstream 기본 affinity) | — |
| attention / dispatch | triton / dynamic (설치본 기본값 확인) | HARDWARE_WORKLOAD_ADAPTATION |
| KV / mem / graph / running / chunk | auto(bf16 확인) 40,960 / 0.95 / decode ≤64, prefill disabled / 64 / **8192** (기존 최고 구성의 실효값, effective_config.json 확인) | HARDWARE_WORKLOAD_ADAPTATION |
| 소프트웨어 | sglang 공식 checkout worktree `/sgl-workspace/sglang-upstream` (71de97b, 로컬 diff 0) 을 `PYTHONPATH` 로 사용; kt_kernel .so 는 로컬 패치 빌드 (env-gated 기능 전부 unset) → `SOFTWARE_BASE_DIFFERS(kt_kernel binary)` | — |

## G-KT-BASIC4 (GLM-4.7-FP8 7b3b5f81…, GPU 0–3, TP4)
| 설정 | 값 | 분류 |
|---|---|---|
| GPU/CPU 체크포인트 | 같은 per-channel FP8 revision | OFFICIAL_MODEL_RECIPE (W3 native 예) |
| CPU backend | FP8_PERCHANNEL | OFFICIAL_MODEL_RECIPE |
| kt-cpuinfer / pool | 100 / 2 | OFFICIAL_MODEL_RECIPE |
| kt-num-gpu-experts | 80 (uniform) | OFFICIAL_MODEL_RECIPE |
| dynamic expert update | 명세 ON → 설치본 미지원 → OFF 실행 (`UNSUPPORTED_OPTION`) | — |
| deferred / token | 2 | OFFICIAL_GENERAL_GUIDE |
| attention / fp8 gemm / p2p / shared fusion / mixed chunk | flashinfer / triton / on / disabled / on | OFFICIAL_MODEL_RECIPE |
| layerwise prefill threshold | 명세 2048 → 미지원 (`UNSUPPORTED_OPTION`) | — |
| KV / mem / graph / running / chunk | auto 40,960 / 0.95 / ≤64, prefill disabled / 64 / 4096 | HARDWARE_WORKLOAD_ADAPTATION (공식 예의 TP8·0.75·chunk16384·running4·KV100000 을 정규화; 원문·diff 는 CONFIG_DIFF.md) |
| MoE 층 | config 92층 중 dense 3 (0–2), routed 3–91 (89층), MTP 층 92 는 speculative OFF 로 미적재 | 모델 config |

## OPT4 (LOCAL_OPTIMIZATION 포함)
- Q-OPT4: IDE_072 V1 복원 — KT_CALLBACK_FREE=1, KT_CF_SKIP_EMPTY_IMM=1, KT_GPU_EXPERTS_PER_LAYER=layer_budget_5952.json, KT_AVX_RB="0" (소스 `amx_kernels.hpp:1769` getenv!=NULL → ON), hotmap_v2, deferred 8, cpuinfer 96, per-layer 패치, 비-kt pinning, 로컬 패치 sglang tree.
- G-OPT4-TRANSFER: GLM INT4 변환 (`kt quant -m int4 -i fp8`, loader 가 per-channel `weight_scale` 자동 감지: kt_kernel/utils/loader.py:341-441) 시도 → 성공 시 AMXINT4 + 같은 메커니즘 + GLM calibration hotmap/예산(89층 × 80 = 7,120 슬롯); 실패 시 `OPT_TRANSFER_BLOCKED_FORMAT`.
