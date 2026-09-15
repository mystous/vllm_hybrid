# IDE_070 진행 로그

## 2026-09-15 19:46 — 착수
- 사용자 문서 `cpu_offload_no_01.md` 수령 → IDE_070/TSK_054~056/TST_027 발급, 브랜치 feat/cpu-offload-diag.
- 기준선 = IDE_069 최종 (TP4 hot96 def4 KV40k, C64 642±7). 순서: 측정 → A 시리즈(코드 무변경) → layer 별 비균일 배치.

## 2026-09-15 20:10 — TSK_054 1차 측정 결과 (`eval/results/20260915_194755_ide070_measure_baseline/`, 요약 `MEASURE_SUMMARY.md`)
기준선 TP4 hot96 def4 KV40k, C64 N256, turbo OFF. 벤치 3회: **633.18 / 634.48 / 638.97 tok/s**, TPOT p50 77.4~78.1 ms, p95 86.1~86.6 ms.
- pcm (부하 구간 26 샘플×2 s): 코어 IPC 1.25 (socket0 1.21 / socket1 1.29), **Backend bound 67 %**, Retiring 24 %, L3 hit 25 %, L3 MPI 0.0018.
- ~~DRAM 대역폭 시스템 428 GB/s~~ **정정 (20:25)**: pcm 의 READ/WRITE 열은 2 s 간격당 GB 누적값이라 ÷2 가 필요. 실제 **시스템 읽기 183 + 쓰기 31 = 214 GB/s, 소켓당 ~107 GB/s** (pcm-memory 채널별 실측과 일치: 소켓0 읽기 97.3 + 쓰기 14.9 = 112 GB/s, 소켓1 110 GB/s, 채널 8개 × 12.2 GB/s 균등, 피크 샘플 245 GB/s). IDE_030 실측 천장 ~430 GB/s 의 **약 50 %**. → 대역폭 포화가 아님. 20:12 보고의 '대역폭 한계' 판단은 오류.
- pcm-numa: 원격 DRAM 접근 비율 시스템 2.0 % (socket0 코어 3.5 %, socket1 코어 0.4 %). UPI 링크 이용률 socket0 수신 5.8 %, socket1 0.7 %. → NUMA 교차 트래픽은 작음 (kt 의 numa 서브풀이 이미 국소화). B 시리즈(NUMA) 의 기대 이득 작음.
- turbostat: Bzy_MHz 1,990 (turbo OFF 확인), Busy% 44.1.
- mpstat: 물리 코어 socket0 81.8 % / socket1 81.5 %, HT 형제 7.6 % / 2.6 %. ≥90 % 코어 95개 (= kt 스레드 96개가 물리 코어에 고정), 112개 코어 <10 %.
- vmstat: **컨텍스트 스위치 240 k/s, 인터럽트 265 k/s** (시스템 전체, 부하 구간). 코어당 ~1 k/s.
- recorder (hotmap96=IDE_069 기준): **cold 선택 7.30 / 토큰** (496 선택 중 1.47 %), 층별 cold 비율 0.43 % (층 29) ~ **8.44 % (층 0)**. 지시서 추정 6.76 과 근사.
- 에너지: package 669 W, DRAM 130 W (부하 구간 평균).
- 결함 (재측정 중, `run_measure2.sh`): perf stat `-p` 대상이 HTTP 서버 프로세스였음 (스레드 최다 기준 오판; 스케줄러는 comm `sglang::scheduler_TP*`) · perf 창 400 s 가 부하 52 s + 유휴 혼합 · py-spy `--native` 실패 (UNW_EBADREG) · numastat -p 권한. 시스템 perf 는 비율만 유효: IPC 1.19/1.18, cache-miss 64/66 %, branch-miss 0.45/0.51 %, 주요 페이지 폴트 0.
- 층별 예산 계산 (`build_layer_budget.py`, 슬롯 5,952 고정): 층별 75~142 (층 0 = 142). 같은 트레이스 기준 cold 선택/토큰 **uniform 7.04 → 비균일 5.96 (−15 %)**, coverage 최소 91.6 → 98.1 %. hotmap_v2 (이 트레이스로 재생성) + `layer_budget_5952.json` 을 `/models/kt/ide070/` 에 준비. 주의: 예산·hotmap 모두 같은 sonnet seed42 부하의 트레이스 (in-sample) — IDE_069 hotmap 과 동일한 조건.

## 2026-09-15 20:35 — TSK_054 2차 측정 (`eval/results/20260915_200733_ide070_measure2/`)
벤치: rep1 633.06 (perf stat 창 정렬 시도) / rep2 517.21 (py-spy+perf record 부하 — 프로파일 오버헤드, 기준선 제외) / rep3 632.81 (pcm-memory+turbostat).
- **스케줄러 프로세스 특정**: TP0 = comm `sglang::scheduler_TP0` (nvidia-smi compute-apps 로 GPU0 의 pid), 스레드 277. 프로세스 affinity 0-55,112-167 (socket0) 이나 kt 워커 스레드는 개별 고정: `numa_0_t_*` → cpu 0..55, `numa_1_t_*` → cpu 56..103 (socket1 물리 코어 40개), `numa_*_m_0` 마스터. HT 형제 미사용.
- **perf record (TP0, 20 s, 298 k 샘플) 분류** — kt 워커 스레드가 샘플의 79 %:
  - kt_kernel_ext `0x75xxx–0x78xxx` 영역 **36 %** — 역어셈블 결과 `vpdpbusd`/`vpsignb`/`vmovdqa32` = **AVX-512 VNNI int8 커널** (AMX `tdpb*` 아님). decode (expert 당 M 작음) 는 kt 가 VNNI 경로를 쓰는 것으로 보임 → AMX 는 decode 처리량에 관여하지 않음.
  - `[vdso] clock_gettime` + `chrono::now` **29 %** + `0x443–446xxx` (lock xadd 스핀 루프, `unique_lock::unlock` 근방) **6 %** + std sync 2 % = **약 37 % 가 대기·스핀**. 즉 kt 워커는 부하 중에도 시간의 ~40 % 를 다음 작업/동기화 대기에 씀.
  - perf 자체 오버헤드 7 % (`perf_adjust_freq_unthr_context`; 다음엔 `-c` 고정 주기 사용).
- pcm-memory (rep3): 소켓0 읽기 97.3 + 쓰기 14.9 GB/s, 소켓1 95.4 + 14.6 GB/s, 채널 8/12 장착·8개 균등 (12.2 GB/s/채널). 시스템 ~222 GB/s, 피크 샘플 245 GB/s.
- numastat (TP0): private 450 GB = node0 338 GB + node1 113 GB (kt 가중치 NUMA 분할이 3:1 로 비대칭. 대역폭은 소켓 대칭이므로 접근은 균형).
- 컨텍스트 스위치 출처 (pidstat 라이브 샘플, A 시리즈 벤치 중): 스케줄러 메인 스레드 4.2 k/s × 4 rank, `gloo_tcp_loop` 4.7 k/s, `pt_gloo_runloop` 2.1 k/s × 2 — TP rank 간 CPU 측 동기화가 gloo(TCP) 로 층마다 일어남. kt 워커 스레드는 컨텍스트 스위치 없음 (스핀).
- py-spy: 277 스레드에서 200 Hz 샘플링이 28 s 지연 → 무효. perf stat SIGINT 종료는 출력 없이 끝남 → `run_measure3.sh` (perf stat 이 벤치 명령을 감쌈) 로 대기열 마지막에 재측정.
- **turbo**: `no_turbo` 쓰기가 root 로도 `Operation not permitted` — IDE_030 (8-30) 에 기록된 BIOS 잠금 재확인. **A1 (turbo ON) 은 이 노드에서 실행 불가**, A 시리즈의 a1 셀은 a0 반복으로 취급.

### 병목 분류 (지시서 §1 기준, 1·2차 측정 종합)
- DRAM 대역폭: 소켓당 ~107 GB/s = 천장(~215/소켓) 의 50 % → 포화 아님.
- 코어: 물리 코어 82 % busy 이나 그중 ~40 % 는 스핀 대기, IPC 1.25, Backend bound 67 % (메모리 지연) → 연산 포화도 아님.
- NUMA 원격 2 % → 문제 아님. 주파수 2.0 GHz 고정 (BIOS).
- → **층 단위 동기 호출의 지연·동기화 한계**: 층마다 작은 cold 작업(≈7 distinct expert × 64 토큰) 을 96 스레드에 쪼개고 gloo 로 rank 동기화 → 스레드당 일감이 작아 대기가 절반. 유효한 개선 방향은 (a) 층당 CPU 로 가는 일 자체를 줄이기 (층별 예산: cold −15 %), (b) 층당 동기화 횟수·대기 줄이기 (deferral 확대는 IDE_069 에서 8 이 4 보다 나빴음), (c) 워커 수 축소로 스레드당 일감 키우기 (A4 80 이 답을 줌).

## 2026-09-15 20:48 — 하네스 결함 2건과 대기열 재구성
- `run_aseries.sh`: `local name=$1 … out=$BASE/$name` 한 줄 선언에서 `$name` 미정의 → `set -u` 로 즉시 종료 (20:16). 수정: 선언 분리. a1 (turbo ON) 은 BIOS 잠금으로 기본 목록 제외.
- `lib.sh bench()`: rep 디렉터리를 만들지 않아 `run_tsk056.sh` u96v2 셀의 벤치가 리다이렉션 실패로 실행되지 않음 (20:19~20:40 GPU 공회전). 수정: `bench()` 첫 줄 `mkdir -p`. 무효 결과 디렉터리 2개 삭제.
- 재구성 대기열 (20:48 시작): TSK_056 (u96v2, nu5952) → measure3 (perf stat 창 = 벤치) → A 시리즈 (a0, a4_80, a4_112, a6).

## 2026-09-15 21:05 — 사용자 지시: 지시서 전 항목 끝까지 진행, 불가 항목은 스킵, 완료 후 평가 없이 결과만 정리·push·md 제공
- TSK_056 1차 실행 결과: u96v2 (hotmap_v2, uniform 96) 677.58 / 645.81 / 666.87. nu5952 셀은 **무효** — `patch_per_layer_experts.sh` 의 `docker exec` 에 `-i` 가 없어 heredoc 이 전달되지 않아 패치가 적용되지 않음 (로그에 "patched" 부재, 서버 로그에 per-layer 라인 0). 수정 후 컨테이너에서 팩토리 단위 검증하고 대기열 끝에 재실행.
- B 시리즈 (TSK_057): b1 interleave / b2 membind0 / b3 threadpool 1 / b4 socket0 전용 (cpuinfer 56) — A 시리즈 뒤 대기열.
- E 시리즈 (W4 GPU expert): 이 모델의 compressed-tensors W4A16 체크포인트는 없고 AWQ/GPTQ/AutoRound 만 존재. `QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ` (252 GB, awq 4-bit g128) 다운로드 시작 (~50 MB/s, ≈85 분). GPU expert 경로는 KTEPWrapperMethod 가 gpu_method 에 `num_gpu_experts` 를 주입하므로 awq 계열도 시도 가능 여부를 부팅으로 판정.
- D (비동기 파이프라인): kt_ep_wrapper 는 submit_forward(스트림 전달) → GPU expert → sync_forward 구조로 층 내 overlap 은 이미 있음. CPU expert 계산은 **TP rank 0 만** 수행 (`if self.tp_rank == 0`). 복사 버퍼(pinned 여부)·스트림은 kt_kernel C++ (`/sgl-workspace/ktransformers/kt-kernel`) 확인 후 판단.

## 2026-09-15 21:32 — A·B 시리즈 완료, C nu5952 재실행 시작, AWQ 다운로드 완료 (252 GB)
- A (`eval/results/20260915_203843_ide070_aseries/`): a0 642.14/620.00/627.63 · a4_cpu80 617.99/588.19/598.14 · a4_cpu112 652.82/629.89/639.51 · a6_gpu0145 635.54/617.94/634.16.
- B (`eval/results/20260915_210401_ide070_bseries/`): b1_interleave 666.52/654.57/647.99 · b2_membind0 663.90/641.00/644.12 · b3_pool1 446.56/451.79/452.14 (TPOT 119) · b4_local(socket0, cpuinfer 56) 444.10/426.02/418.69 (TPOT 122~127).
- nu5952 재실행: 패치 적용 확인 ("patched"). 이어서 measure3 → D (d1~d4) → E (e1~e3).

## 2026-09-15 22:30 — C·D·E 결과
- C nu5952 (`eval/results/20260915_213137_ide070_tsk056/`): 717.70 / 700.56 / 671.91, TPOT p50 65.8/67.8/71.8, p95 78.7/78.0/80.6.
- measure3 (`20260915_213804_ide070_measure3/`): 664.93 / 648.13 / 636.48 (perf stat 창 = 벤치, rep1 시스템·rep2 스케줄러 4개·rep3 TP0).
- D (`20260915_214431_ide070_dseries/` 계열): d1_cf 678.39/660.30/669.98 · d2_cf_skip8 rep1 52.67 (128/256 실패, 스케줄러 프로세스 크래시 — 서버 로그 꼬리에 Python fatal 덤프·py-spy 실패만 남음, 원인 미특정) rep2/3 0 · d3_cf_skip8_pin 792.22/778.11/773.44 · d4_epoch 813.62/852.97/781.52, C224 1,089.19.
- E (`20260915_221142_ide070_eseries/`): e1/e2/e3 (AWQ hot 96/128/144) 모두 부팅 실패 — cuda graph 캡처 중 `activation.cuh:208 illegal memory access`, e2 는 추가로 `moe_sum_reduce CUDA kernel (small-token) launch failed`. eager (`--disable-cuda-graph`) 로 e1/e2 재시도 중.

## 2026-09-15 22:40 — 종결
- E eager 재시도 (`20260915_222941_ide070_eseries/`): e1_awq_h96_nograph DIED 100 s (스케줄러 프로세스 종료, py-spy 덤프 실패 메시지), e2_awq_h128_nograph DIED (`moe_sum_reduce CUDA kernel (small-token) launch failed`, `activation.cuh:208 illegal memory access`). E 시리즈 전 셀 부팅 불가로 종료.
- 지시서 항목 처리 현황: 측정(1·2·3차) 완료 / A0·A4·A6 완료, A1 BIOS 잠금 스킵, A2 측정으로 대체 / B1~B4 완료 / C u96v2·nu5952 완료 / D1·D3·D4 완료, D2 크래시 기록 / E 3셀 + eager 2셀 부팅 실패 기록.
- 결과 정리: `RESULT.md` (`eval/ide070/make_result.py` 생성, 측정값만).
