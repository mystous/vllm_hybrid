# IDE_047 — EAGLE3 투기적 디코딩 × 하이브리드 — 진행 중

draft: `lmsys/SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI` (0.9B, 1층). 플래그: `--speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 160` (미지정 시 SGLang 이 48 로 자동 제한). env `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.

## 1차 (graph 160, KV 강제 110k → 실제 87k): 부팅 OK, 첫 prefill 에서 CUDA OOM (여유 1.0 GB, fused_moe 중간 버퍼 768 MB) → 서버 사망.
## 2차 (mem 0.92, graph 128, 청크 4096, KV 자동 = 35,944 토큰): 동작. admission 제한 (≈56 동시).
| | tok/s | TPOT | TTFT | accept len | GSM40 |
|---|---|---|---|---|---|
| C160 (실효 ~60) | 486 | 129 | 21.4s (대기열) | 1.97~2.19 (sonnet) | |
| C64 (실효 ~56) | 512 | 108 | 1.39s | 1.99~2.27 | 95.0 (38/40) |
| 꼬리 13 running | 649 (step 44ms) | | | 2.19 | |
| GSM 8 running | | | | 2.3~2.8 (수학) | |

관찰: 13 running 에서 step 44 ms 는 CPU 모델 (52 검증 행 → cold ~8개 → 0.7 ms/층) 과 일치. 56~66 running 에서는 step 220~275 ms 로 모델 (126~136 ms) 의 약 2배 → 검증 행 수 (4×) 에 비례하는 추가 비용 = AVX vec 경로의 행당 18 µs (IDE_037) 가 노출됨 → **IDE_048** (vec 커널 레지스터 블로킹) 파생. prefix cache 규칙: C64 r2 는 무효.
GSM40 95.0 vs 97.5: greedy 검증은 이론상 동일하나 배치 구성 차이로 인한 BF16 비결합성 토큰 분기 (운영 해석상 허용 범위). 최종 구성에서 GSM100 재확인 예정.

## 3차 (KV 명시 81,920 / 65,536, KT_PHASE_PROF=1): 진행 중
