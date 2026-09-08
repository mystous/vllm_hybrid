# 새 메커니즘 (CF+skip+pin, α=0.25, hot-96 N=8) 프로파일 + prefill 법칙 재측정 (2026-09-09 02:33~02:47)
- 프로파일: 8스텝 캡처했으나 트레이스가 Python 함수 이벤트 1,138만 건 + GPU 커널 이벤트 0 (activities CPU/GPU 지정에도 커널 미기록) → 스텝 분해 불가. 비-층 고정비는 모델 v3 에서 **가정 2.0ms** 로 두고 12셀로 검증.
- prefill 법칙 (입력 512/출력 1, α=0.25 hot set): C1 268.8ms/req, C8 73.8, C32 17.0 (구 메커니즘 prompt map: 322 / 71.1 / 19.3) — 거의 동일. TTFT(out=1, C32) = 544ms = 32 × 17.0 → 파도식 admission 확인.
- 모델 v3 확정 파라미터: D_c 비율 0.88 (in-situ/트레이스), 고정비 2.0ms, prefill b_p=min(C,32) 파도, TTFT = 2.95 × 파도 prefill × cold 계수 (3점 인자 2.9~3.0), tok/s = B·out/(TTFT + (out−1)·TPOT). 기측정 6셀 재예측 TPOT +0.6~+6.0%.
