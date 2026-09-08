# IDE_032 배치-인지 deferral — kill test (2026-09-08 21:05, hot-96, C=32 sonnet)

사전 등록 기준: N=8(cold 전량 deferral) C=32 ≥600 tok/s AND GSM40 ≥95%.

| 구성 | tok/s | TPOT | GSM40 | 비고 |
|---|---|---|---|---|
| N=0 (기준, 09-08 09:04) | 458.9 | 54.8 | 95.0 | |
| N=4 (최종 구성) | 490.9 | 51.5 | 97.0 (100문항) | |
| **N=8 (cold 전량 deferral)** | **505.4** | **48.3** | **97.5** | 처리량 기준 미달, 정확도 기준 충족 |
| N=6 | 484.3 | 50.6 | 97.5 | N=4·8 사이 (잡음 수준) |

- **처리량 판정: 미달** (+3% vs N=4, 예측 +40%). N=0→4→8 이 각 −3.3ms 로 선형 소폭 → deferral 로 숨겨지는 양이 스텝당 ~3~4ms 로 캡.
- **정확도: 통과** — cold expert 기여를 전량 한 층 늦게 더해도 GSM 97.5%. 알고리즘의 정확도 전제 성립 (KT 는 37% 질량 deferral 에 −0.5%; 여기선 sonnet 기준 1.4% 질량. 단 topk 덤프의 에세이 프롬프트에선 cold 질량 40% — 워크로드 의존, τ 임계의 존재 이유).
- **캡의 가설**: 잔여 유휴 ~8ms/스텝은 CPU 대기가 아니라 층당 3~4개 host 함수 노드 (submit imm/def, sync) 의 CUDA 콜백 디스패치 지연 (62×3×~40µs ≈ 7ms) — deferral 로 제거 불가한 바닥. 판별 = N=8 정상 상태 프로파일 (갭 크기 분포). 참이면 다음 알고리즘: **host 노드 제거 (양방향 mapped-memory 폴링 핸드오프)** — GPU→CPU 는 memop write + CPU 스핀, CPU→GPU 는 기존 신호 memop. 예측 이득 ≈ 7ms/스텝 (+18%) + deferral 겹침 실현분.

## 21:20 갱신 — host 노드 바닥 마이크로벤치 (유휴 GPU4, graph 재생, `mb_hostnode_floor.py`)
- host sync 노드 (큐 빈 상태): **~20µs/노드**; memop wait+write 쌍: **~9µs/노드**. 층당 3 host 노드 × 62층 = 스텝당 **~3.7ms 바닥** (memop 대체 시 ~1.7ms → 절약 ~2ms/스텝 ≈ 5%).
- 이걸로 hot-96 N=8 이 설명됨: 27.7 (GPU 커널) + 1.9 (복사) + 3.7 (host 바닥) + ~3 (잔여) ≈ 36ms decode → TPOT 48 (prefill ~12 포함) ✓. **hot-96 에선 deferral 이 이미 CPU 를 다 숨겼고, 남은 병목은 GPU 자체 (hot expert 17ms)**. 캡의 원인은 "deferral 실패" 가 아니라 "숨길 CPU 가 더 없음".
- 따라서 알고리즘의 결정 무대 = **hot-80 (CPU 병목: 노출 35ms, GPU 26)**: cold 전량 deferral 로 CPU 35 가 GPU 26 밑에 숨으면 스텝 66→~40ms (+65%), 게다가 GPU 메모리 여유로 KV 131K (hot-96 의 2.7×) → C=64 에서 thrash 없음. hot-80 N=8 kill test (C32/C64 + GSM40) 를 체인 끝에 예약.
- 사전 예측: hot-80 N=8 C32 TPOT ≈ 40+12 = 52ms (~480 tok/s, N=4 의 332 대비 +45%); C64 ≈ 60ms (~520 tok/s, hot-96 의 408 대비 +27%).
