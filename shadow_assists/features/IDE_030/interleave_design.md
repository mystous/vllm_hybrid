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

## 구현 일지 (2026-09-07 밤)

구현 완료분 (각각 실측으로 필요성 입증):
1. kt 고정 버퍼에 채널 축 추가 (experts_base.py) — 세트 간 버퍼 공유 차단. 단독 검증: 세트 1개+부하 생존 ✓ 무해
2. 세트 B 를 전용 CUDA graph 메모리 풀로 캡처 (full_cuda_graph_backend + 훅) — 전역 풀 공유 시 A 재생 즉사 실측
3. 세트 B 에 전용 attention 처리기 인스턴스 (model_runner 훅) — 공유 시 init_cuda_graph_state 재할당으로 A 재생 즉사 실측
4. 교차 실행 본체 (_il_slice_fb / _interleaved_execute / _il_merge_logits) + 실제 초기화 경로(init_cuda_graphs)에 훅

미해결 (오늘 종착점):
- 교차 발동 시 **교착**: rank 전원이 세트 A 첫 재생 안에서 정지, GPU 100% 공회전 (py-spy 로 위치 특정)
- 해석: kt 의 spinning 설계 — graph 내 GPU 커널이 CPU 완료 플래그를 폴링. CPU 태스크가 안 돌고 있음
- 유력 가설: kt CPUInfer 의 재생-시-태스크-재실행 등록이 세트별 2벌이 아니라 전역 1벌 → 마지막 캡처(B)가 A 의 등록을 대체
- 다음 수: kt C++ 바인딩의 submit_with_cuda_stream 재생 트리거 메커니즘 분석 → 세트별 등록 분리

원상 복구 방법: SGL_INTERLEAVE 미설정 시 모든 훅 비활성 — 기존 327 구성 그대로 동작 (분리 검증 완료)

## 2일차 진행 (2026-09-07 심야)

- kt C++ 소스 판독으로 대기 방식 확정: submit=cudaLaunchHostFunc(작업 큐 삽입), sync=host 함수가 **전역 큐 잔량 0까지 블로킹 대기** (`task_queue_->sync`)
- 수리 4: CPU 작업 큐를 세트별 2개로 분리 (CPUInfer 인스턴스 2개, 스레드 48+48) — 적용했으나 **행업 재발, 같은 지점**
- 최종 원인 규명: **NCCL 통신 순서 불일치** — rank0 의 graph 에만 CPU 대기 노드가 있어 rank 간 A/B 진행 속도가 갈리고, rank0 의 A 통신과 rank1-3 의 B 통신이 매칭 → 교착. 두 스트림이 같은 통신 채널을 공유하는 한 구조적
- 다음 후보 (미착수): ① 세트 B 에 별도 NCCL 커뮤니케이터 (sglang 에 스트림별 자원 그룹 개념 존재 — decode_attn_backend_group/get_current_stream_idx 경로 조사) ② rank 전체 동기 배리어로 A/B 재생 경계 고정 (겹침 이득 일부 상실) ③ TP allreduce 를 graph 밖으로 빼는 재구성 (대공사)
- 현재 안전 상태: SGL_INTERLEAVE 미설정 시 기존 327 구성 무손상

## 결론 (2026-09-07 심야 2차)

판별 실험 최종 결과:
- 순차 재생 (A→B, 스트림 1개): **생존** — 격리 수리 5건 (kt 채널 버퍼 / 전용 graph 풀 / 전용 attention / 세트별 CPU 큐 / 세트별 TP 통신그룹[pdmux 인프라 재사용]) 전부 유효 입증
- 동시 재생 (스트림 2개): 행업 / **절반-겹침 (B만 별도 스트림): 역시 행업**

**1차 결론**: 두 graph 의 GPU 동시 존재가 kt 의 대기 방식과 구조적으로 충돌한다.
기전: kt 는 GPU 폴링 커널이 CPU 완료 신호를 기다리는 설계 (KT 논문의 spinning). 세트 B 의 폴링 커널이 SM 을 점유한 채 신호를 기다리는데, 그 신호를 만들 host 함수 실행이 CUDA 의 host 함수 직렬 실행 규약 아래 세트 A 의 블로킹 대기(sync: 큐 잔량 폴링)에 막힘 → 상호 잠금. 파이썬/통합 수준의 어떤 격리로도 이 층은 못 푼다 (5중 격리 후에도 재현).

**따라서 교차 실행의 이론 이득 (+60%대) 실현에는 kt C++ 수정이 필요**:
- 방향 A: 폴링 커널을 SM-비점유 대기로 (cudaStreamWaitValue32 기반, 또는 그리드 축소+nanosleep)
- 방향 B: sync host 함수의 블로킹 제거 (완료 이벤트를 GPU 가 기다리게 하고 host 함수는 즉시 반환)
- 요구 사항: kt-kernel 소스 빌드 체인 (repo + cmake + AMX 툴체인) 구축 후 커널 수정·재빌드 — 다음 단계

부산물: 격리 수리 5건은 kt×sglang 의 다중-graph-세트 사용 일반(예: pdmux 와 kt 병용)에 필요한 수리로, 자체 가치 있음.
