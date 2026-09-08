# IDE_035 τ 스윕 1차 시도 (CF + KT_COLD_DEFER) — 부팅 실패 (2026-09-09 03:50)
τ=0.25: CUDA graph 캡처 실패 `cudaErrorStreamCaptureInvalidated` (`/tmp/sgl_tau0.25.log`). callback-free 경로 + 가중치 임계 deferral (immediate 작업 비어 있지 않음) 조합은 캡처 중 오류 — CF 는 N=8 (빈 immediate) 에서만 검증됨. 정확도 질문에는 CF 가 필요 없으므로 host 콜백 경로로 τ 스윕 재실행 (`…_ide035_tau_sweep_hostcb/`). CF + 비어있지 않은 immediate 의 캡처 오류는 별도 결함으로 기록.

**정정 (03:56)**: host 콜백 경로에서도 동일 실패 → CF 무관. 근본 원인 = `select_deferred_experts` 의 `gmask.to(device)` (pinned 마스크의 H2D) 가 CUDA graph 캡처 중 실행 → `cudaErrorStreamCaptureUnsupported`. (IDE_032 당시엔 캡처 전 eager 호출이 있어 통과했던 것으로 추정.) 수정: `_gpu_experts_mask_gpu` 를 __init__ 에서 생성·캐시. τ 스윕 재실행 (`…_ide035_tau_sweep_hostcb/`).
