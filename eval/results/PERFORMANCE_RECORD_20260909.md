# 성능 기록 — CPU·GPU 하이브리드 MoE 서빙 (2026-09-09 마감)

- 모델: Qwen3-Coder-480B-A35B-Instruct FP8 (GPU 측), expert INT4 (CPU 측, KTransformers AMXINT4)
- 노드: violet-h100-016 — NVIDIA H100 80GB × 4 (TP=4, GPU 0~3) + Intel Xeon Platinum 8480+ × 2 (96 워커 스레드, 2 풀, AMX, turbo OFF 2.0 GHz) + DDR5-4400 8ch × 2DPC × 2 소켓 (이론 282 GB/s/소켓)
- 스택: SGLang 0.5.18 + kt-kernel 0.7.0.post1 (소스 패치, 컨테이너 `sgl-kt`)
- 벤치: `vllm bench serve` sonnet 입력 512 / 출력 128 / prefix 100, 프롬프트 수 = 3 × 동시성, seed 42. 정확도: GSM8K test 앞 N문항 greedy (`gsm_eval.py`).
- 기준선 (캠페인 시작 2026-08-29~30): hybrid 성립 44.5 → 56.3 tok/s (C32), GSM8K(40) 85.0.

## 1. 심층조사 보고서 4개 제안 기준 결과

보고서(`shadow_assists/brainstorming/deepresearch_report_20260909.md`)의 목표 지표를 그대로 적용.

| 제안 | 보고서 목표 | 수행 | 측정 결과 | 판정 |
|---|---|---|---|---|
| **EPOCH** (가중치 1회 읽기당 행 수 ↑) | 읽기당 행 2~4×, DDR bytes/token ↓, 처리량 +15%, TPOT p99 +5% 이내, 출력 동일 | 트레이스 오라클 (IDE_037) → 스케줄러 형태 불가 (자기회귀). 실현형 = 동시성 확대: FP8 KV (IDE_039) + CUDA graph 버킷 축소 (IDE_049) 로 KV 37k → 127k 토큰, 동시성 64 → 224 | 읽기당 행 1.1 → 1.34 (+20%). DDR bytes/token 3.75 → 3.1 MB (−17%). **처리량 769 → 1110 tok/s (+44%)**. TPOT 61 → 149 ms (다른 운영점). 출력 동일 (logprob 동등성) | 처리량 목표 초과, 행 배수·TPOT 기준 미달 (운영점 이동) |
| EPOCH 변형: 투기 디코딩 (IDE_047) | 같음 | EAGLE3 draft 로 검증 행 4× | C160 706 (−27%), C64 461 (−38%); cold slot 이 서로 다른 expert 에 분산 | 기각 |
| **SHWG** (호스트 동기화 노출 ↓) | 층당 고정비 77 µs → e2e 2% 미만 | 위상 분해 → gather·양자화 융합 (IDE_051) 으로 장벽 1개 제거 | +3% (C224 1067 → 1098). 남은 고정비 ~130 µs/층 ≈ 8% | 부분 (GPU 측 persistent graph 미구현) |
| **TriX** (GPU 가 호스트 메모리를 직접 읽는 제3 경로) | zero-copy 교차점 존재 | PCIe H100 (C2C 없음) 에서 H2D 복사 경로만 측정 (IDE_053, 사용자 중단) | 모든 m 에서 ~800 µs/expert > CPU 66~500 µs; 단 측정 경로가 pageable 스테이징 (7.3 GB/s) 이라 상한 아님 | 보류 |
| **JIT-EO** (학습 optimizer JIT) | step 시간 ↓, 파라미터 동일 | 미착수 (IDE_054, 사용자 중단) | — | 보류 |

## 2. 최종 구성 처리량 (버스트, 무손실)

공통: hot-96 (α=0.25 phase 가중 hot set `/tmp/hotmap_mixed_0.25.json`) + callback-free 핸드오프 + τ=0.25 deferral + 비-워커 코어 재배치 + expert 별 AMX 선택 + AVX 레지스터 블로킹·prefetch + gather·양자화 융합 + FP8 KV (mem 0.94).

| 운영점 | 동시성 | tok/s | TPOT ms | TTFT s | 정확도 |
|---|---|---|---|---|---|
| **B 최대** (graph 224, 버킷 32·64·96·128·160·192·224, 청크 4096, KV 127,309) | **C224** | **1110.3** (재현 1098.2) | 149.0 | 6.76 | GSM100 **97.0** |
| B | C192 | 1071.5 | 133.1 | 6.02 | |
| **A 균형** (graph 192, 버킷 6종, 청크 8192, KV 122,880) | C192 | 1048.7 | 131.1 | 6.77 | GSM100 96.0 (동일 구성 노이즈 이내) |
| A | C160 | 1012.2 (융합 켬 1038.7) | 114.0 | 5.73 | GSM40 97.5 |
| A | C128 | 961.9 | 96.0 | 4.83 | |
| A | C64 | 796.6 | 57.6 | 2.97 | |
| A | C32 (fresh) | 578.0 | 37.2 | 2.36 | |

캠페인 시작 (56.3) 대비 **19.7×**, 전날 최종 (C64 769) 대비 +44%, 당일 정오 (C160 982.4) 대비 +13%.

## 3. 정확도·동등성

| 항목 | 값 |
|---|---|
| GSM100 (최종 구성 B) | 97.0 (97/100) — 캠페인 기준선 97.0 과 동일 |
| GSM100 (혼합 forward 켬, 3회) | 94 / 96 / 97 (평균 95.7) |
| per-token logprob (GSM8K 30문항 gold 풀이 teacher-forcing, 5,590 토큰): 최종 A vs 정오 구성 | max abs 4.50 nats, PPL 상대차 평균 1.2% (worst 10.4%) |
| 같은 구성 재부팅 노이즈 | max abs 6.75 nats, PPL 상대차 평균 2.1% (worst 10.4%) → 구성 차이는 노이즈 이내 |
| 커널 패치 (AVX 레지스터 블로킹·prefetch·AMX B 캐시) | 원본 대비 bit 동일 (4개 형상 max abs diff 0) |
| 위상 융합 | int8 양자화 값의 0.016% 가 ±1 (fast-math 역수 1-ulp 로 tie ±63↔±64 만 갈림) |

## 4. 도착 체제 (Poisson, 최종 구성 B)

| 구성 | rate 5: tok/s / TPOT / TTFT p99 | rate 7 | rate 9 | 버스트 C224 (tok/s / TPOT / TTFT p50) |
|---|---|---|---|---|
| 기본 (배칭·혼합 없음) | 532.6 / 299.0 / 1.44 s | 716.0 / 227.2 / 3.18 s | 871.7 / 179.7 / 7.22 s | 1110.3 / 149.0 / 6.76 s |
| prefill 배칭 delayer, 게이트 128 | 536.1 / 199.8 / 1.37 s | 715.3 / 231.3 / 2.10 s | 846.0 / 196.7 / 4.10 s | 1095.5 / 151.0 / 7.11 s |
| delayer, 게이트 225 | 541.5 / 171.0 / 1.42 s | 694.6 / 225.0 / 1.97 s | 781.1 / 210.6 / 2.09 s | 1094.5 / 151.9 / 6.88 s |
| delayer 16패스·1000 ms | 542.2 / 155.9 / 1.88 s | 679.2 / 228.0 / 2.54 s | 773.4 / 209.9 / 3.13 s | 1031.6 / 158.4 / 6.64 s |
| **혼합 forward (`--enable-mixed-chunk`)** | 543.3 / **184.1** / **0.92 s** | 706.7 / 225.1 / **1.01 s** | **814.4** / 203.2 / **1.27 s** | 1096.2 / 173.4 / **1.75 s** |
| 혼합 + delayer 225 | 545.8 / **141.7** / 1.55 s | 702.6 / **190.5** / 2.29 s | 790.7 / 202.5 / 2.64 s | 1036.6 / 176.5 / 2.81 s |

운영 권고: 기본 = 혼합 forward (TTFT 최저, 처리량 −7%); TPOT 중심이면 혼합 + delayer. 최고 처리량 벤치는 배칭·혼합 없이.

## 5. 기법별 기여 (누적)

| 순서 | 기법 (ID) | CPU 측 인과 | 이득 |
|---|---|---|---|
| 1 | hot expert GPU 배치 hot-64 → 80 → 96 + decode CUDA graph (IDE_030) | CPU 가 스트리밍할 cold 바이트 축소 | 56.3 → 459 (C32) |
| 2 | callback-free CPU↔GPU 핸드오프 + 빈 immediate 제거 + 비-워커 코어 재배치 (IDE_033) | 층당 고정비 126 → 77 µs, AMX 워커의 HT 형제 간섭 제거 | 505 → 600 (C32) |
| 3 | phase 가중 hot set α=0.25 (IDE_034) | decode 커버리지 96 → 97.8% | 600 → 661 (C32), C64 884 |
| 4 | τ=0.25 가중치 임계 deferral (IDE_035) | 확률 낮은 cold expert 를 다음 층과 겹침, 무손실 | C32 572 / C64 769, GSM100 97.0 |
| 5 | expert 별 행 수 기준 AMX/AVX 선택 (IDE_038-b) | kt 기본 규칙 (qlen>80 전부 AMX) 의 1~2행 타일 손해 회피 | C96 +8% |
| 6 | FP8 KV + graph 160 (IDE_039) | 실KV 37k → 110k: 같은 스트리밍으로 2.5× 토큰 | C160 982.4 (+28%) |
| 7 | 스트리밍-인지 prefill 배칭 (IDE_043-c) | prefill forward 고정비 (cold 전량 스트리밍 ≈250 ms) 상각 | 경부하 TPOT −27~43%, 과부하 TTFT p99 −43~76% |
| 8 | AVX-512 INT4 vec 커널 레지스터 블로킹 + prefetch (IDE_048/050) | 누산기 체인 해소, 1행 스트리밍 75 → 66 µs (−13%), 소켓당 ~190 GB/s | C64 782~797 |
| 9 | CUDA graph 버킷 축소 → KV 123~127k → graph 192/224 (IDE_049) | 같은 스트리밍으로 토큰 +20~40% | C192 1049, C224 1067 |
| 10 | gather·양자화 위상 융합 (IDE_051) | 장벽 1개·bf16 복사 제거 | C224 1098 (+3%), C160 +3.3% |
| 11 | prefill·decode 혼합 forward (IDE_052) | decode 행을 prefill 의 cold 스트리밍에 얹음 | 경부하 TPOT −38%, 과부하 TTFT p99 −82% |

## 6. 기각 항목과 근거

| ID | 내용 | 결과 |
|---|---|---|
| IDE_036 | 폴러 수준 빈 deferred 생략 | 효과 0 |
| IDE_037 | EPOCH 스케줄러 | 읽기당 행 중앙값 1, 패스 병합 자기회귀상 불가 |
| IDE_038 | 전역 AMX 전환 | C32/C64 −7% (1~2행 타일 패딩) |
| IDE_040 | hot-88/80 + C192/256 | C256 933 / 779 < hot-96 (CPU 바이트 증가 > KV 이득) |
| IDE_042 | prefill 청크 16k/32k | CUDA OOM |
| IDE_044 | hot set 재도출 (C64 트레이스) | C160 918 / 965 < 982 (커버리지 포화) |
| IDE_045 | expert 버퍼 2MB huge page | 적용 확인 (133→245 GB) 했으나 처리량 동일 |
| IDE_046 | prefill AMX GEMM B 타일 캐시·양자화 병렬 | 층당 −3~6%, 서빙 동일 (GEMM 은 L2 타일 적재 한계, AMX 24%) |
| IDE_047 | EAGLE3 투기 디코딩 | C160 706 / C64 461 |
| 탐침 | 풀 스레드 96 → 112 | −2% (노이즈) |

## 7. 물리 상한 (실측)

- decode 층 시간 (cold expert 24개, 1행): GEMM 스트리밍 88% (소켓당 214 GB/s = DDR 피크 76%), 소위상 11%.
- prefill 임계 경로의 CPU 몫 ≈ 1/4 (스레드 절반 → TTFT +27%); decode 는 CPU 노출 큼 (스레드 절반 → TPOT +24~26%).
- GPU 메모리 78.9 / 80 GB (hot-96 expert 70 GB + KV + graph) → 동시성 상한 224.

## 8. 재현 명령 (운영점 B, 최고 처리량)

```
KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25 \
KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m sglang.launch_server --model-path <Qwen3-Coder-480B-A35B-Instruct-FP8> --tp 4 --attention-backend triton \
  --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 \
  --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json \
  --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --kv-cache-dtype fp8_e5m2 --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 --max-total-tokens 143360 --cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224
# 부팅 후 pin_nonkt.sh 실행 (비-워커 스레드를 코어 48-55/104-111 로). 운영 시 --enable-mixed-chunk 추가.
```

## 9. 파일 색인

- 요약: `eval/results/SUMMARY_20260909.md` (§11 추가 트랙), 이력: `shadow_assists/CPU_UTILIZATION_HISTORY_20260909.md`
- 최종 재측정: `eval/results/20260909_170510_final_sweep/`, 도착률: `20260909_182418_final_arrival/`, 혼합 forward: `20260909_190222_ide052_mixed_chunk/`
- 기법별: `20260909_1110*_ide038*`, `1130*_ide039*`, `122541_ide039i_graph160`, `142515_ide043c_running_gate`, `145919_ide045_hugepage`, `152125_ide046_amx_bcache`, `155921_ide047_eagle3`, `162155_ide048_avx_rb`, `163853_ide049_graph_buckets`, `170007_ide050_avx_prefetch`, `174350_ide051_fuse_qin`, `174147_probe_decode_phase_threads`, `194156_probe_longctx`, `201104_ide053_trix_pcie`
- 패치·하네스: `eval/harness/20260909_mechanism/`, `eval/harness/20260909_afternoon/`
- ID 상태: `shadow_assists/id_registry.md`
