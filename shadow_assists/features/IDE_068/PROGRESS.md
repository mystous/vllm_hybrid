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
