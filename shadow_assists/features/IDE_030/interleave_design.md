# 교차 실행 (interleaved dual micro-batch) 설계 노트 — 2026-09-07

## 문제 (실측)
327 tok/s 체제 스텝 66ms = GPU 26ms + **CPU cold-expert 대기 40ms** (GPU 유휴). CPU busy 37%, GPU busy 33% — 서로 기다리며 둘 다 놂.

## 막힌 길 (오늘 판정)
- x>80: HBM 한계 / CPU 가속: 대역폭 한계 / 커널 튜닝: −4.6% 기각
- TBO(기성): dp-attention 강제 → hot-80 불가 (mem 0.858), hot-48 후퇴 시에도 캡처 TypeError (마이크로배치 크기 미전파) — 조합 상한 낮음, 접음

## 설계
배치 C를 A/B 절반으로. layer 단위 교차:
```
A: attn(L) → submit_cold(L,A) → gpu_hot(L,A) ─┐            ┌ sync(L,A) → ...
B:                       attn(L) → submit_cold(L,B) → gpu_hot(L,B) → sync(L,B)
```
핵심: sync(L,A) 를 B 의 attn/gpu_hot 뒤로 지연 — A 의 CPU 계산이 B 의 GPU 시간과 겹침. 수학 무손실 (각 마이크로배치는 완전한 forward, 순서만 교차).

## 구현 지점
1. graph 유지가 관건: 교차 구조 전체를 하나의 decode graph 로 캡처 (forward 를 (A,B) 교차 luoop 로 재작성한 runner) — sglang `decode_cuda_graph_runner` 의 run_once 를 2-마이크로배치 버전으로
2. kt 쪽: KExpertsCPUBuffer 슬롯을 (마이크로배치×layer) 로 확장 (buffer_depth 2 → 4), submit/sync 짝 재배선
3. 예상 지점별 난관: attention backend 의 batch 분할 메타데이터 / graph 캡처 시 두 배치 자리표 / TP allreduce 순서

## 판정 기준 (사전 등록)
- 정확도: greedy 4문항 + GSM 40 ≥ 기준−5%p
- 성능: C=32 에서 ≥ 360 (오늘기준 327 대비 +10%) — 이론 여지 +60% 의 하한선
- 실패 조건: 2일 내 부하 생존 불가 또는 +5% 미만이면 중단 보고
