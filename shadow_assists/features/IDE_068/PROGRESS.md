# IDE_068 진행 로그

## 2026-09-15 09:22 — 착수
- 노드 확인: violet-h100-016 (H100×8 / 8480+×2 / 2 TB / turbo OFF). GPU 전부 유휴.
- 자원: 480B FP8 450 GB + kt INT4 232 GB 보유. R1-0528 원본 스냅샷은 삭제됨(kt INT4 328 GB 만 잔존) → R1 은 실험 A 후보에서 제외.
- 실험 A 후보 = Kimi-K2-Instruct 1,029 GB (HF API). 대역폭 실측 39.5 MB/s. `/data/hf` (23 TB 여유) 로 `hf download --max-workers 16` 시작.
- ID 발급: IDE_068 / PLN_010 / TSK_050 / TSK_051 / TST_024 / TST_025. 브랜치 `feat/moe-offload-limits`.
- 30분 보고 cron 등록 (23a2a6c1).

## 2026-09-15 09:30 — 실험 B 착수, b0 기준선 부팅 성공
- `eval/results/20260915_092544_ide068_expB_480b/` 시작 (turbo OFF 확인: no_turbo=1, 2.0 GHz).
- **b0 GPU-only TP8+EP8**: HEALTH OK **140초**. TSK_047 에서 순수 TP8 이 FP8 block 제약으로 부팅 불가였던 것을 `--ep-size 8` 로 우회 — 480B 의 GPU-only 기준선 최초 확보. HBM 73.7~74.1 GiB/장.
- b0 greedy 4문항 **전부 정상** (Paris / fibonacci 재귀 / 5050 / `s[::-1]`). C16 64req 벤치 진행 중.
- Kimi-K2 다운로드: `hf download` 재시작(setsid, 인증). 81개 HTTPS 연결 수립됐으나 `/data/hf/hub` 에 blob 이 안 쌓임 — xet 청크 캐시 경로 확인 중.

## 2026-09-15 09:38 — b0 기준선 확정, 캐시 경로 정정
- **b0 GPU-only TP8+EP8 (C16, 64req)**: 64/64 완료, **701.60 out tok/s**, 전체 3,453 tok/s, TTFT p50 335 ms / p95 455 ms, TPOT p50 20.14 ms / p95 21.10 ms, 벤치 11.68 s. CPU busy 평균 5.8 %. greedy 4/4 정상.
- b0a GPU-only TP4 진행 중 (OOM 예상).
- 캐시 경로 정정: 셸 환경에 `HF_HOME=/data/mystous/.cache/huggingface` 가 설정돼 있어 `hf download` 는 그쪽으로 받는다 (`~/.cache/huggingface` 도 같은 곳). 따라서 Kimi-K2 는 별도 마운트 없이 `sgl-kt` 컨테이너의 `/models/hub/` 에서 보인다. `/data/hf` 시도는 무효 (소형 파일 3개만 남음).
- Kimi-K2 수신: net RX **118 MB/s**, 45 GB 도착 (16 blob 진행 중). 잔여 ~985 GB → 약 2.4 시간 (≈ 12:00 완료 예상).

## 2026-09-15 09:40 — b0a OOM 재실증, b1 하이브리드 TP4 성립 (TSK_047 재현)
- **b0a GPU-only TP4**: 40초 만에 사망. `torch.OutOfMemoryError` — device 3 이 78.91 GiB 점유 상태에서 600 MiB 할당 실패. TSK_047 재현.
- **b1 하이브리드 TP4 (expert 전량 CPU)**: HEALTH OK **70초**. 64/64 완료. **43.39 out tok/s** (TSK_047 44.5 와 일치), 전체 213.6 tok/s, TTFT p50 8,622 ms / p95 11,318 ms, TPOT p50 299.1 ms / p95 303.0 ms, 벤치 188.8 s. greedy 4/4 정상. **HBM 17.0~17.5 GiB/장** (GPU 4장만), DRAM 사용 315 GB.
- 기준선 대비: 처리량 701.60 → 43.39 (**6.2 %**), TPOT 20.14 → 299.1 ms (14.9×). GPU 8장 → 4장.
- b2 하이브리드 TP2: HEALTH OK 70초, 벤치 진행 중.

## 2026-09-15 09:41 — 30분 점검
- b2 하이브리드 TP2: greedy Q1·Q2 정상 확인, C16 벤치 17/64 진행 (서버 로그 gen throughput 32.8~54.5 tok/s, #running-req 16, cuda graph False). HBM 20.0~20.5 GiB/장 (GPU 2장).
- Kimi-K2: net RX 118 MB/s 유지, blob 진행 16개(합 96 GB)·완료 17개, 디렉터리 117~131 GB. 디스크 반영 속도는 RX 의 약 절반(≈55~60 MB/s) — xet 청크 처리로 추정. 완료 예상 13:30~14:00 로 수정.
- B2(GPU expert 배치 16/40/96) 는 expB 종료 직후 자동 연결됨.

## 2026-09-15 09:46 — b2 TP2 성립, b3 TP1 부팅 성공
- **b2 하이브리드 TP2**: 64/64 완료, **43.33 out tok/s**, TTFT p50 8,643 ms / p95 11,075 ms, TPOT p50 300.1 ms / p95 302.0 ms, 벤치 189.1 s. greedy 4/4 정상. HBM 20.0~20.5 GiB/장 (2장). **TP4 (43.39) 와 동일** — 처리량이 GPU 장수와 무관 = CPU expert 구간이 지배.
- **b3 하이브리드 TP1**: HEALTH OK **60초**, HBM **24.6 GiB (1장)**, DRAM 307 GB. smoke·벤치 진행 중.

## 2026-09-15 09:50 — 실험 B 본 셀 완료: 480B 가 GPU 1장에서 성립
- **b3 하이브리드 TP1**: 64/64 완료, **43.24 out tok/s**, TTFT p50 8,694 ms / p95 11,008 ms, TPOT p50 300.1 ms / p95 304.4 ms, 벤치 189.5 s. greedy 4/4 정상. **HBM 24.0 GiB (1장)**, DRAM 307 GB. CPU busy 평균 43.9 %.
- TP4 / TP2 / TP1 = 43.39 / 43.33 / 43.24 tok/s — GPU 장수와 무관. 답: **최소 1장** (GPU-only 는 8장 필요·TP4 OOM).
- B2 자동 착수 (`20260915_094613_ide068_expB2_480b_gpuexperts`): b4 TP1 + GPU expert 16개 부팅 중.

## 2026-09-15 09:55 — B2 b4: TP1 + GPU expert 16개
- **b4 TP1 + `--kt-num-gpu-experts 16`** (물리 id 0~15, hotmap 없음): HEALTH OK 70초, 64/64, **46.81 out tok/s** (b3 43.24 대비 **+8.3 %**), TTFT p50 8,014 ms, TPOT p50 274.0 ms (300.1 → 274.0, −8.7 %), 벤치 175.0 s. greedy 4/4. **HBM 65.7 GiB (1장)** — expert 16개×62층 FP8 ≈ 46 GB 가 올라감. CPU busy 43.7 %.
- b5 TP2 + 40개 부팅 중.

## 2026-09-15 10:02 — B2 완료 (b5, b6)
- **b5 TP2 + GPU expert 40**: HEALTH OK 90초, 64/64, **49.90 tok/s** (+15.2 % vs 전량 CPU), TTFT p50 6,891 ms, TPOT 268.0 ms, HBM 70.0 GiB ×2. greedy 4/4.
- **b6 TP4 + GPU expert 96**: HEALTH OK 100초, 64/64, **51.55 tok/s** (+18.8 %), TTFT p50 4,548 ms, TPOT 273.4 ms, HBM 75.7~76.2 GiB ×4. greedy 4/4.
- hotmap 없는 물리 id 배치의 이득은 +8~19 % 에 그친다. IDE_030 의 hotmap 96개 = 490.9 tok/s 와의 차이가 빈도 기반 배치의 가치.
- GPU 전부 유휴. Kimi 다운로드 대기 중 → 그 사이 TP1 에서 cuda graph 를 켠 변형(B3)을 돌려 최소 장수 구성의 현실적 처리량을 본다.

## 2026-09-15 10:09 — 30분 점검
- B3 b7 (TP1, cuda graph ON, expert 전량 CPU): HEALTH OK 70초, 벤치 중 — 서버 로그 `cuda graph: True`, gen throughput 51~57 tok/s (graph OFF 였던 b3 의 서버 로그 33~55 대비 상승). 완료 후 집계.
- Kimi-K2: 287 GB (blob 완료 33·진행 16). 09:55 → 10:09 사이 100 GB — 실효 ≈ 120 MB/s. 완료 예상 11:50 전후.

## 2026-09-15 10:14 — 사용자 요청: GPU 3·5·6·7 장 추가
- B4 (`run_expB4.sh`) 를 B3 종료 직후 자동 연결. 하이브리드 TP3/5/6/7 + GPU-only TP6/TP7(EP). 480B 는 attention head 96 / KV head 8 이라 TP5·TP7 은 head 분할이 안 되고 TP3·TP6 은 KV head 가 안 나뉠 가능성이 있음 — 추측으로 빼지 않고 전부 부팅 시도, 실패 시 엔진 오류를 기록.

## 2026-09-15 10:16 — B3 완료: TP1 cuda graph 효과 없음
- b7 TP1 expert 0 graph ON: 43.53 tok/s (OFF 43.24), TPOT 297.6 ms. b8 TP1 expert 16 graph ON: 46.10 (OFF 46.81), TPOT 281.4 ms. 모두 greedy 4/4. ±1 % 로 편차 수준 — CPU expert 구간이 지배라 GPU launch overhead 감소가 드러나지 않음.
- B4 착수: b9 하이브리드 TP3 부팅 시도 중.

## 2026-09-15 10:22 — B4 완료: GPU 3·5·6·7 장은 TP 로 불가 (6/6 엔진 거부)
- 하이브리드 TP3/5/6/7: 전부 40초 내 `AssertionError: 151936 is not divisible by N` — vocab 151,936 = 2⁷×1187 (소수) 이라 TP 는 2 의 거듭제곱만 가능.
- GPU-only TP6+EP6 / TP7+EP7: `assert num_physical_experts % ep_size == 0` — expert 160 이 6·7 로 안 나뉨.
- 결론: 480B 의 TP 가능 장수 = {1, 2, 4, 8}. 3·5·6·7 은 DP(복제)로만 사용 가능 → B5 에서 DP3×TP1 1점 측정 중.

## 2026-09-15 10:31 — B5(DP3) 사용자 지시로 중단, 실험 A 로 전환
- 사용자: "굳이 필요 없어. 멈추고 실험 A 진행하자" → DP3×TP1 셀 중단 (부팅·greedy 는 성립했으나 벤치 미완, 결과 미기록). 실험 B 는 B4 까지로 종결.
- 실험 A 자동 연결: 다운로드 완료(hf 종료 + incomplete 0 + safetensors 61) 감지 → `run_expA.sh all` (kt quant int4 → 하이브리드 TP8 부팅 → greedy 4문항 → C8 벤치).
- Kimi-K2 398 GB / 1,029 GB (blob 완료 41, 스냅샷 safetensors 25/61). 완료 예상 12:00 전후, 변환 ~1.7 h → 서빙 결과 14:00 전후.

## 2026-09-15 10:39 — 30분 점검
- 실험 B 종결 (b0~b14). 실험 A 대기: Kimi-K2 **453 GB / 1,029 GB (44 %)**, RX 105 MB/s, blob 완료 41·진행 16, 스냅샷 safetensors 25/61. 완료 예상 12:10 전후. GPU 8장 유휴, DRAM used 80 GB.
- 자동 연결(`chain_expA.sh`) 대기 중 — 완료 감지 즉시 `kt quant int4` 착수.

## 2026-09-15 11:09 — 30분 점검
- Kimi-K2 **621 GB / 1,029 GB (60 %)**, RX 104 MB/s, blob 완료 49·진행 16, 스냅샷 safetensors 33/61. 30분간 +168 GB (≈93 MB/s 실효). 잔여 408 GB → 완료 예상 12:20 전후. 자동 연결 대기 중, GPU 유휴.

## 2026-09-15 11:39 — 30분 점검
- Kimi-K2 **784 GB / 1,029 GB (76 %)**, RX 117 MB/s, blob 완료 68·진행 14, 스냅샷 safetensors 49/61. 30분간 +163 GB. 잔여 245 GB → 완료 예상 12:20 전후. 자동 연결 대기 중, GPU 유휴.

## 2026-09-15 12:09 — Kimi-K2 다운로드 완료, INT4 변환 착수
- 다운로드 완료 12:08 (시작 09:23 → 2 h 45 min). 디스크 959 GiB (= 1,030 GB), blob 82 완료, 스냅샷 safetensors 62. 평균 실효 ≈ 104 MB/s.
- `kt quant /models/hub/.../Kimi-K2-Instruct -m int4 -i fp8 -o /models/kt/kimi-k2-int4 --cpu-threads 96 --numa-nodes 2` 12:08:30 시작. DRAM used 76 GB / buff·cache 1,574 GB (page cache 에 원본 적재됨). R1 642 GB 가 65 분이었으므로 1,030 GB 는 ~105 분 → 13:55 전후 완료 예상, 이어서 TP8 부팅 자동.

## 2026-09-15 12:39 — 30분 점검: INT4 변환 절반
- kt quant: **layer 31 / 61 완료** (12:08:30 → 12:39, 층당 ≈ 1 분, 마지막 층 11.73 s 저장). 출력 244 GB (30 shards) → 최종 ≈ 480 GB 예상. DRAM used 379 GB, page cache 1,535 GB.
- 잔여 30 층 → **13:10 전후 완료** (당초 13:55 예상보다 빠름). 완료 즉시 TP8 하이브리드 부팅 자동.

## 2026-09-15 13:09 — 30분 점검: INT4 변환 55/61
- kt quant **layer 55 / 61** (마지막 층 65.9 s — 초반 12 s 대비 느려짐, DRAM free 22 GB·page cache 1,432 GB 로 캐시 압박 추정). 출력 435 GB → 최종 ≈480 GB. 완료 예상 **13:20 전후**, 즉시 TP8 부팅 자동.

## 2026-09-15 13:17 — Kimi-K2 INT4 변환 완료, TP8 부팅 착수
- `kt quant -m int4 -i fp8` **완료 13:17:04, 소요 4,114 s (68.6 min)**, QUANT_EXIT=0. 출력 **488 GB, 64 files** (`/models/kt/kimi-k2-int4`). 원본 1,030 GB → 0.47×.
- a1 하이브리드 TP8 부팅 시작 13:17:16 (`--kt-cpuinfer 96 --kt-num-gpu-experts 0 --disable-cuda-graph --max-total-tokens 65536`). tiktoken tokenizer 로드 확인 (#words 163,840).
