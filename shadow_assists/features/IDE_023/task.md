# IDE_023 / TSK_043 — 단계별 작업

## Phase 1 — feasibility (fork 수정 0줄)

| 단계 | 내용 | 상태 |
|---|---|---|
| 1.1 | R1-0528 다운로드 | ✅ (642GB) |
| 1.2 | SGLang 이미지 + kt-kernel | ✅ (0.7.0.post2 no-deps + 패치 3건) |
| 1.3 | Qwen3-30B-A3B smoke | ✅ (t2 96.1 tok/s, CPU 43.1% 포화, 출력 정상 — 우회 4건 문서화) |
| 1.4 | R1-0528 GPU-only OOM 실증 | ✅ (`torch.OutOfMemoryError`, r0_gpu_only_oom/) |
| 1.5 | R1-0528 hybrid 서빙 | ✅ 서빙 성립 (AMXINT4 변환 65분 후 19.67 tok/s, CPU 42.7%, 기동 160s) — 단 출력 품질 결함 → `SUB_167` |
| 1.6 | TST_021 판정 | ✅ 부분 통과 (OOM 실증·서빙 성립·CPU 활용 ✓ / 품질 ❌) — **Phase 2 진입은 SUB_167 통과 후** |

## Phase 2 — vLLM fork native 통합 (조건부, Phase 1 게이트 통과 시)

별도 TSK 로 분리 발급 예정. 본 문서에서는 scope 만 기록: expert-granular weight placement + AMX GEMM 경로 + expert deferral 스케줄.
