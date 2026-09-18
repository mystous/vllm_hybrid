# BASELINE_PROVENANCE — IDE_076 (2026-09-17 22:50 KST, O0)

## 실행 환경 (읽기 확인 결과)
- host violet-h100-016, GPU H100 80GB ×8 (본 캠페인은 0~3 TP4), 작업 시작 시 GPU 사용 0, 활성 서버 없음(sgl-kt 안 launch_server 0). 디스크 / 1.6 TB 여유.
- 컨테이너: `sgl-kt` (SGLang 0.5.18, HEAD 71de97b + 로컬 10 파일; kt-kernel upstream 6d460cc + 로컬 11 파일 + kt_evt.h), `vllm-h100` (vllm 0.26.1rc1 nightly a9a17e7; 클라이언트).
- 저장소: `/home/mystous/projects/vllm_hybrid` HEAD edfb8bb99 (feat/cpu-moe-bottleneck-followup-20260917 게시 완료) → 새 브랜치 `feat/cpu-moe-a1a2b-20260917`. 기존 사용자 수정은 유지(리셋·clean 안 함). 미커밋 파일 64 (IDE_074/073 잔여·대용량 산출물; 본 캠페인과 무관, 건드리지 않음).

## 수정 바이너리의 출처 (PLAN.md §3.2)
- 설치 `.so` SHA256 `12926df2c30b724694171108635637e8cda14064ae9a643c03b114d6c9d6b926` (8,376,352 B, 2026-09-17 18:11 빌드) = IDE_075 v4: 기록기 v3 + `moe_base.hpp` FP8/BF16 dispatch 수정(TSK_060). S29(689.7 tok/s CORR)·FOCUS2 S28(703.5 OFF) 에서 사용된 바이너리와 동일.
- 소스: `/sgl-workspace/ktransformers/kt-kernel` (upstream 6d460cc, dirty). 파일별 SHA256·diff SHA 는 `SOURCE_MANIFEST.json`. 컴파일: g++ 13.3.0, CMake Release `-O3 -DNDEBUG`, `-march=native`.
- Python: `kt_kernel/experts_base.py` (v3 recorder 패치 fc44bf94…), `sglang/srt/layers/moe/kt_ep_wrapper.py` (rsf 패치 1bc65851…; GLM 전용 분기, Qwen 경로 영향 없음 — 회귀 검사 §13 대상).
- 이 바이너리를 **reference_binary** 로 삼는다. A1/A2 를 담은 신규 빌드는 플래그 OFF 로 이 바이너리와 먼저 대조한다(§3.3).

## 고정 구성 (PLAN.md §2.1 확인)
| 항목 | 확인값 |
|---|---|
| 모델 | Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 rev 003f183a…; 62층·160 expert·top-k 8·hidden 6,144·intermediate 2,560 (config.json) |
| CPU 가중치 | /models/kt/qwen3-480b-int4 (AMXINT4, 67 파일 232 GB), index/config/tokenizer SHA 는 manifest |
| H0 | hotmap_v2.json 7ce02cad… / layer_budget_5952.json c8ba4247… (합 5,952) |
| 실행 기능 | KT_CALLBACK_FREE=1, KT_CF_SKIP_EMPTY_IMM=1, KT_GPU_EXPERTS_PER_LAYER=layer_budget_5952, KT_AVX_RB=0(ON), deferred 8, CPU 96 / NUMA pool 2 (rc73.cfg("qwen","OPT4")) |
| 실제 kernel shape | NUMA 서브풀 2 → 서브풀별 intermediate 1,280 (moe-tp 균등 분할), hidden 6,144. gate/up: K=6,144·N=1,280 (task N_BLOCK 128 → nth 10); down: K=1,280·N=6,144 (nth 48). INT4 tile M_STEP 32·N_STEP 32·K_STEP 64·K_BLOCK 3,584 |
| bs=63 Cold job 경로 | qlen=64 → `forward_prefill`; 64 ≤ KT_AMX_MIN_QLEN(80) → `vec_mul` → `integer_mat_mul<Int4,false>` → RB `avx_kernel_rb` (R=1: `avx_rb_rows<1,4>`, R=2: `<2,2>`, R=4: `<4,1>`, R≥8: `<8,1>`) |
| 워크로드 | MAIN_SHORT(64/256/128), PROBE128(64/128), C1(1/16), LONG(8/32) — 입력 jsonl SHA 는 manifest; 클라이언트 vllm bench / vllmw(동등 확인 IDE_075 OFF_Y) |
| 스케줄러 | max running 64, decode graph ≤64, prefill graph off, chunk 8,192, mem 0.95, KV 40,960 (launch argv 는 부팅 시 `launch_cmd.sh`) |

## 비교에 쓰지 않는 자료 (PLAN.md §1.3)
IDE_074 MAIN 753 tok/s(역사), S29 689.7(CORR), 서로 다른 .so 의 OFF 평균, probe 클라이언트 세션, S22 clock tail, S23/S27 TP 시차, S26/S27 C1/LONG(workload_valid=None).

## 새 무계측 기준선 R0 (§3.4)
- 1차: reference_binary(v4) 로 `R0_REF` 부팅 1회 — MAIN_SHORT OFF ×3(같은 부팅), C1, LONG. smoke 4 + warmup 32 후 실행.
- 2차(신규 빌드 완성 후): 신규 바이너리 A1=OFF/A2=OFF/H0 로 같은 계획 → reference 와 대조 (§3.3). 이후 부팅 블록 교차(R0→후보→후보→R0).
