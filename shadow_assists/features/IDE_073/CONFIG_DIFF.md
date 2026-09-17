# CONFIG_DIFF — IDE_073

## 1. GLM 공식 native 예 (W3, GitHub main 2026-09-17 snapshot) → G-KT-BASIC4 정규화 diff
| 항목 | 공식 예 원문 | 이번 실행 | 사유 |
|---|---|---|---|
| tp | 8 | 4 | 지시서 §3.3 자원 정규화 (GPU 4장) |
| mem-fraction-static | 0.75 | 0.95 | 공통 정규화 |
| chunked-prefill-size | 16384 | 4096 | 공통 정규화 |
| max-running-requests | 4 | 64 | 공통 정규화 |
| max-total-tokens | 100000 | 40960 | 공통 KV 예산 |
| kt-expert-placement-strategy uniform | 있음 | 옵션 없음 → uniform 기본 동작 | 설치본 미지원 (의미 동일) |
| kt-enable-dynamic-expert-update | 있음 | 실행 불가 | 설치본 미지원 → UNSUPPORTED_OPTION |
| kt-gpu-prefill-token-threshold 2048 | 있음 | 실행 불가 | 설치본 미지원 → UNSUPPORTED_OPTION |
| context-length | (예 참조) | 32768 | 공통 |
| cuda graph | (예 참조) | decode ≤64, prefill disabled | 공통 |

원문 명령 (W3 snapshot 발췌) 은 `evidence/sources/W3.html` 에 보존.

## 2. Q-KT-BASIC4 ↔ Q-OPT4 전체 diff
| 항목 | BASIC4 | OPT4 | 분류 |
|---|---|---|---|
| sglang tree | `/sgl-workspace/sglang-upstream` (71de97b, diff 0) | `/sgl-workspace/sglang` (71de97b + 로컬 10파일) | SOFTWARE |
| kt-cpuinfer | 112 | 96 | 공식 물리코어 vs 로컬 |
| GPU expert | uniform 96 (hotmap 없음) | hotmap_v2 + 층별 75~142 (합 5,952) | LOCAL_OPTIMIZATION |
| deferred | 2 | 8 | LOCAL |
| KT_CALLBACK_FREE / KT_CF_SKIP_EMPTY_IMM / KT_AVX_RB | unset | 1 / 1 / "0"(ON) | LOCAL |
| pinning | 없음 | 비-kt 스레드 재배치 | LOCAL |
| chunked-prefill | 8192 (OPT 실효값과 동일) | 8192 (기본값) | 공통 |
| KV / mem / graph / running | 40,960 bf16 / 0.95 / ≤64 / 64 | 동일 | 공통 |

## 3. GPU-only 8장 (Q-GPU8, G-GPU8)
| 항목 | 값 |
|---|---|
| Qwen | tp 8, ep-size 8, triton, KV 40,960 (IDE_069 참조의 131,072 와 다름), mem 0.90, graph ≤64, running 64, chunk 8192 |
| GLM | tp 8 (+EP 지원 시 ep-size 8; 미지원이면 TP8 만 기록), flashinfer, fp8-gemm triton, p2p-check, shared fusion disabled, mixed chunk, KV 40,960, mem 0.90, graph ≤64, running 64, chunk 4096 |

## 4. 기본군 소프트웨어 경로 변경 (08:26 결정, SOFTWARE_BASE_DIFFERS)
- 시도: 공식 checkout worktree `/sgl-workspace/sglang-upstream` (71de97b, 로컬 diff 0) 을 `PYTHONPATH` 로 사용 → Q-KT-BASIC4 a2·a3 부팅 실패: `TypeError: KTMoEWrapper.__new__() missing 1 required positional argument: 'gpu_experts_mask'` (upstream sglang kt_ep_wrapper ↔ 설치된 kt_kernel 0.7.0.post1 API 불일치). 로그 `eval/results/IDE_073_20260917/Q-KT-BASIC4/a2/server.full.log.gz`.
- 결정: 기본군(Q/G-KT-BASIC4)은 로컬 sglang tree(71de97b + 로컬 10파일, 그중 kt_ep_wrapper 의 gpu_experts_mask 호환 패치 포함)로 실행. LOCAL_OPTIMIZATION 은 전부 비활성: KT_* env unset, `--init-expert-location` 없음(uniform), per-layer 패치 미적용(서버 로그 per-layer 행 0 확인), 비-kt pinning 없음, deferred 2, cpuinfer 112(Qwen)/100(GLM).
- 로컬 tree 의 나머지 diff(SGL_INTERLEAVE·SGL_DEBUG_REMAP 분기, qwen3_moe.py dispatch-info 전달, model_runner 인터리브 훅, 프로브 로그)는 env 미설정·hotmap 미지정 시 기본 경로와 동일하게 동작하는 것으로 코드상 확인했으나, 실행 바이너리 수준의 upstream 동일성은 보장하지 않음 → 기본군 vs 최적화군은 '전체 구성 비교'.
