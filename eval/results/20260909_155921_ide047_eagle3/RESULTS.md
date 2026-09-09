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

## 3차 (mem 0.92 + KV 명시 81,920): KV 여전히 35,944 (메모리 부족) → 중단
## 4차 (mem 0.94, , KV 요청 98k → 실제 87,449, 청크 4096, KT_PHASE_PROF=1)
| | tok/s | TPOT | TTFT | accept len | GSM40 |
|---|---|---|---|---|---|
| C160 (실효 ~136, fresh) | **706.3** | 190.5 | 2.68s | 1.99~2.23 | |
| C64 | **461.1** | 111.2 | 1.39s | 1.99~2.27 | 97.5 |
| 기준 (비투기) | 965~985 / 728~766 | 120 / 61 | 5.6~5.9s / 3.0~3.2s | — | 97.5 |
→ C160 −27%, C64 −38%. **기각**.

### 검증 스텝 CPU 위상 (qlen 640 = 136×4 + prefill 혼합, 소켓 0, 층당)
activated cold expert 40~48개, up_gate 1.75~2.3 ms + down 1.5~1.9 ms, **총 3.6~4.7 ms/층** (→ 스텝당 CPU ≈ 250 ms). expert당 ≈ 95 µs (1행 71 µs + 평균 2.7행의 추가 행 × 18 µs). 측정 스텝 385 ms 중 나머지 ~135 ms 는 GPU 측 투기 오버헤드 (draft 3회 + extend + 검증 forward).

### 구조적 결론
hot-96 에서 cold slot 은 2.2% 이고 각 slot 이 서로 다른 cold expert 에 떨어질 확률이 높아, 검증 행이 4배가 되면 스트리밍할 cold expert 수가 C160 에서 22 → 42~48 (2배), C64 에서 10 → 32 (3배) 로 늘어난다. 즉 "같은 스트리밍으로 더 많은 행" 이 아니라 "행에 비례해 스트리밍 증가" 가 되어, 수락 길이 2.0 으로는 스텝 비용 증가 (CPU 2배 + 투기 오버헤드 135 ms) 를 넘지 못한다. 손익분기: 수락 길이 ≥ ~2.7 (코드 워크로드) 이면서 행당 비용 (IDE_048) 제거 시 C160 에서 +10% 내외. 부산물: 행당 18 µs 비용이 서빙에서 정량 확인 → IDE_048.
