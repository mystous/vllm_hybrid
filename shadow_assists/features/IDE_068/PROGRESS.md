# IDE_068 진행 로그

## 2026-09-15 09:22 — 착수
- 노드 확인: violet-h100-016 (H100×8 / 8480+×2 / 2 TB / turbo OFF). GPU 전부 유휴.
- 자원: 480B FP8 450 GB + kt INT4 232 GB 보유. R1-0528 원본 스냅샷은 삭제됨(kt INT4 328 GB 만 잔존) → R1 은 실험 A 후보에서 제외.
- 실험 A 후보 = Kimi-K2-Instruct 1,029 GB (HF API). 대역폭 실측 39.5 MB/s. `/data/hf` (23 TB 여유) 로 `hf download --max-workers 16` 시작.
- ID 발급: IDE_068 / PLN_010 / TSK_050 / TSK_051 / TST_024 / TST_025. 브랜치 `feat/moe-offload-limits`.
- 30분 보고 cron 등록 (23a2a6c1).
