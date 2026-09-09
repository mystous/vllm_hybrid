# IDE_046 — prefill CPU 커널 효율 (AMX INT4 GEMM B 타일 언팩 캐시 + A 양자화 병렬화) — 진행 중

동기: IDE_045 에서 prefix cache 전량 적중 시 C64 처리량 766→1093 (+43%) → prefill 이 처리량의 ~30%. prefill 은 층마다 cold expert 64개에 각 50~400행이 들어가는 CPU AMX GEMM.
코드 판독: `GemmKernel224Int4::amx_kernel` 은 32행 M 블록마다 같은 INT4 B 타일을 AVX-512 로 INT8 언팩 (mask/shift → 스택 → tileload). m>32 에서 M/32 배 중복.

## 패치 (환경변수로 켬/끔, 기본 경로 불변)
| env | 내용 | 파일 |
|---|---|---|
| `KT_AMX_BCACHE=1` | 첫 M 블록에서 언팩 결과를 스레드 로컬 L2 캐시 (448 KB) 에 저장, 이후 M 블록은 tileload 만 | `la/amx_kernels.hpp` `amx_kernel_bcached`, `integer_mat_mul` 분기 |
| `KT_AMX_A_LOADD=1` | A 타일 `_tile_stream_loadd` → `_tile_loadd` | 같은 파일 |
| `KT_QA_PAR=1` | 입력/다운 A 양자화 (`from_mat`, expert 당 1 스레드) → expert × 32행 블록 task | `la/amx_buffers.hpp` `from_mat_block`, `moe_base.hpp` q_input/q_down |

수치 동등성: 4개 형상 (T=8/40/200 n_cold 8, T=1024 n_cold 64) 에서 원본 대비 **max abs diff 0** (bit 동일).

## rows 스윕 (모두 AMX 경로 `KT_AMX_MIN_QLEN=0`, 층 1회 호출, µs)
| n_cold / rows | base | bcache | bcache+A loadd |
|---|---|---|---|
| 8 / 128 | 2351 | 2218 | 2202 |
| 8 / 256 | 4342 | 4363 | 4115 |
| 64 / 32 | 7347 | 7409 | 7302 |
| 64 / 128 | 17803 | 17151 | 18054 |
| 64 / 256 | 32177 | 30831 | 30956 |
→ −4~6%, 사전 등록 (−30%) 미달. 언팩 중복은 주 비용 아님.

## 위상 분해 (`KT_PHASE_PROF`, 소켓 0, 64 expert × 128행, µs)
| 위상 | base | bcache | qapar | qapar+bcache | 192 스레드 (HT) |
|---|---|---|---|---|---|
| cpy_input | 754 | 736 | 769 | 788 | 529 |
| q_input | 1520~1587 | 1767 | 1280 | 1647 | 2016 |
| up_gate GEMM | 6448~6464 | 6080 | 6578 | 6209 | 6390 |
| act | 444 | 459 | 448 | 469 | 410 |
| q_down | 186~244 | 280 | 176 | 249 | 318 |
| down GEMM | 6442~6490 | 5844 | 6524 | 5883 | 5018 |
| weight | 689 | 751 | 682 | 715 | 791 |
| **total** | **16562~16661** | **15961** | **16489** | **15995** | **15525** |

해석:
- GEMM 2개 = 77%. task 당 k-step 쌍 (타일 적재 8 KB + tdpbssd 8회 = 128 cycle) 에 ~527 cycle → **AMX 활용률 ~24%**, 2×2 타일 블로킹에서 A·B 타일이 L2 에서 오는 적재 대역폭 (≈20 B/cycle) 한계. 언팩 캐시·A load 방식·HT 모두 이 한계를 바꾸지 못함.
- q_input 은 병렬화해도 −16% 뿐 → gather 복사본 (96 MB/소켓) 을 DRAM 에서 다시 읽는 메모리 트래픽 한계. gather+양자화 융합이면 −1 ms 가능.
- 남은 큰 지렛대는 GEMM 의 L1 상주 블로킹 재작성 (C 타일 트래픽과 교환, 기대 ≤1.5×).

## 서빙 A/B (진행 중) — `KT_QA_PAR=1 KT_AMX_BCACHE=1` 켬/끔, 최선 구성 (hot-96, FP8 KV 110k, graph 160)
(결과 대기)

## CPU-bound 판별 (진행 중) — `--kt-cpuinfer 48` (CPU 시간 ≈2배) 에서 C64 TTFT / C160
(결과 대기)
