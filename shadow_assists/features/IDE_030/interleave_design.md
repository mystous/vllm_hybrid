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

## 3일차: KTransformers C++ 수정 (2026-09-08 자정 무렵)

- 소스 빌드 체인 구축 완료 (repo + 서브모듈 + AMX 툴체인, 무수정 wheel 빌드 성공)
- 수정 구현 (방향 B):
  1. `cpuinfer.h`: 완료-신호 슬롯 512개 (호스트-GPU 공유 핀 메모리), `submit_signal_with_cuda_stream`(비블로킹: 큐에 "플래그=1 쓰기" 작업만 추가), `wait_signal_on_stream`(cuStreamWaitValue32 로 GPU 스트림이 플래그 대기 → cuStreamWriteValue32 로 0 재설정 — 재생 반복 안전)
  2. `ext_bindings.cpp`: 파이썬 바인딩 2개 추가
  3. `CMakeLists.txt`: CUDA driver 라이브러리 링크 추가
- 남은 순서: 재빌드 → wheel 교체 설치 → 파이썬 패치(채널 버퍼/채널 큐) 소스판 재적용 → sync_forward 를 신호 방식으로 교체 → 동시 재생 재시험

## 4일차: 환경 사고와 복구 (2026-09-08 00:00~00:30)

빌드·설치 자체는 완료된 상태에서 부팅이 환경 문제 2건으로 연쇄 실패, 순서대로 복구:

1. `deep_gemm.utils.layout` ModuleNotFoundError — 어제 만든 libnvrtc 심링크 때문에 그동안 실패하던 `import deep_gemm` 이 성공하게 되며 구버전 패키지 구조 불일치가 노출. 조치: `deep_gemm` 디렉토리를 `deep_gemm.disabled` 로 개명하여 sglang 의 optional import 가드 복원.
2. `ImportError: cannot import name 'fp8_scaled_mm' from 'sgl_kernel' (unknown location)` — 진단 결과 `dist-packages/sgl_kernel/` 에서 `__init__.py`, `ops/`, 주 연산 .so 대부분이 삭제되어 있었음 (디렉토리 변경 시각 9-07 15:00 = kt-kernel wheel 작업 시간대. pip 재설치 과정의 부작용으로 추정). `__file__=None` 인 네임스페이스 패키지로 전락한 상태.
   - 복구: 같은 이미지(lmsysorg/sglang:latest)로 임시 컨테이너를 띄워 원본 `sgl_kernel` 1.2GB 를 호스트로 복사 후 sgl-kt 에 재주입. `from sgl_kernel import fp8_scaled_mm` 정상 확인.
   - 부기: nerdctl stdout 파이프로 GB 단위 tar 스트림을 뽑으면 1.19GB 지점에서 절단됨 — bind mount 방식으로 우회 (violet-h100-016 노드 특이사항에 추가할 것).

복구 후 결정적 시험 재개: SGL_INTERLEAVE=1 부팅 → 단건 품질 → 24 동시 버스트 생존 → C=32 벤치 (사전 등록 기준: ≥360 tok/s 채택 / +5% 미만 중단).

## 4일차 계속: 신호 코드 버그 #6 — host 노드 인자 use-after-free (2026-09-08 00:30~01:00)

- 환경 복구 후 SGL_INTERLEAVE=1 부팅 재시도 2회:
  1. kt_kernel 재설치본에서 채널 버퍼 패치의 옛 버그(출력 버퍼 크기 인자 tuple) 재발 — 패치 스크립트가 미수정 상태여서 재적용 때 같이 들어감. 설치본·스크립트 양쪽 수정.
  2. 로드·캡처 통과 후 **decode graph 재생 중 rank 1개 segfault** (faulthandler 스택: decode_cuda_graph_runner.execute → full_cuda_graph_backend.replay → torch graph replay).
- 원인 (코드 판독으로 확정): `signal_enqueue_` host 함수가 인자(SignalArgs*)를 heap 할당 후 실행 시 `delete`. CUDA graph 에 캡처된 host 노드는 재생 때마다 **같은 포인터로 재호출**되므로 첫 재생에서 해제 → 이후 재생에서 use-after-free. 관측된 "재생 중 segfault" 와 일치.
- 수리: CPUInfer 멤버로 슬롯별 고정 인자 배열 `SignalArgs signal_args_[512]` 를 두고 submit 시 `&signal_args_[slot]` 전달, delete 제거. 교훈: **graph 에 캡처되는 host 노드의 인자는 영속 저장소여야 한다** (기존 sync_ 경로는 매 재생마다 new 만 하고 delete 하지 않는 누수 설계였는데, 그것이 사실상 replay-safe 조건이었던 것).
- wheel 재빌드 진행 중. 병행 회귀 확인: **SGL_INTERLEAVE 미설정 기본 구성, 새 소스빌드 wheel 에서 정상** — health OK, greedy 품질 정상("Paris"), 24 동시 버스트 24/24 (4.2초). 기존 327 구성의 기능 무손상 확인.

## 4일차 계속: 버그 #7 — deferred 층은 신호 경로 미적용이었음 (01:00~01:20)

- UAF 수리 후 5차 부팅: **로드·캡처·단건 품질 통과** (greedy 정상: Paris / fibonacci 코드). 단, 단건 생성 속도 3.67 tok/s 로 비정상 저속 (원인 미규명, 아래 수리 후 재측정 예정). 24 동시 버스트에서 전건 타임아웃 — 행업 재발.
- py-spy 진단: rank0 만 `_interleaved_execute` 의 세트 A graph 재생 호출에서 블록, rank1-3 은 이미 다음 스텝의 요청 수신 대기. GPU 4장 100% (NCCL 스핀 커널로 추정).
- 원인: `allow_pending=1` (deferred expert 가 걸린 층 — hot-80/cold-80 구성에선 사실상 모든 층) 은 신호 경로 조건 (`allow_pending == 0`) 에서 제외되어 **여전히 블로킹 host 함수 sync 사용**. 신호 방식이 대부분의 층에서 미적용이었으니 교착 구조가 그대로였던 것.
- 수리 (설계 개선): 신호 큐잉을 sync 시점이 아니라 **submit_forward 안, immediate 작업 직후 · deferred 작업 직전**으로 이동. 큐 순서상 flag=1 은 "immediate 완료" 시점에 서므로 기존 sync(allow_pending=1) 과 동일 의미이면서 deferred 의 다음-층 겹침은 보존. sync_forward 는 allow_pending 무관하게 wait_signal 만 수행 (블로킹 host 함수 완전 제거).
- 5차 결정적 시험 재기동.

## 4일차 계속: 판별 — 신호 코드 정상, 동시성 특유 경합으로 좁힘 (01:20~01:45)

- 순차 모드 (SGL_IL_SEQ=1) + 새 신호 설계 (전 층 memop 대기): **24/24 버스트 생존, 3.4초**. 신호 코드의 할당·캡처·재생은 전부 정상.
- 따라서 크래시 (CPU segfault 2건 + GPU Xid 31 MMU fault) 는 **A/B 동시 재생에서만** 발생. 유력 기전: 채널별 CPU 작업 큐 2개 (수리 4) 가 동시에 돌며 **같은 층의 AMX MoE 객체 forward 를 두 워커풀이 동시 실행** — 객체 내부 per-call 상태 경합. (순차·구식 블로킹 설계에선 큐가 하나씩만 돌아 불가능했던 상황이라 이제야 노출.)
- 재설계: 블로킹 sync 가 사라졌으므로 채널 분리의 존재 이유가 소멸. **단일 CPUInfer (96 스레드) 를 두 채널이 공유** — 단일 큐는 작업을 순차 실행하므로 같은 층 동시 실행 원천 차단 + CPU 스레드 온전히 사용 + 신호 슬롯은 채널별 영역이라 무충돌. 채널 간 결합 (A 의 신호가 B 의 앞선 작업도 대기) 은 보수적 순서화일 뿐 교착은 불가 (블로킹 host 함수 없음).
- 6차 결정적 시험 (동시 재생 + 단일 큐) 진행 중. C++ 계측 빌드 (오류 검사) 병행.

## 4일차 계속: 7차 시험 — 여전히 동시 재생 행업, 진단 심화 (01:50~02:10)

- 슬롯에 캡처 축 추가 (eager 128-255 / captured 0-127, 채널 B 는 +256) 후 7차: 단건 통과 (4.1 tok/s), 24 버스트 행업 재발. 크래시는 안 남 (단일 큐 유효 지속).
- 심화 진단 (계측 빌드 + /proc/mem + gdb 전 스레드):
  - `[kt-signal] slots ready` 가 **rank0 에만** 출력 — CPU expert 계산과 신호 노드는 rank0 graph 에만 존재 (기존 관찰 재확인).
  - rank0 cuGraphLaunch 가 driver 내부 cond wait 에서 블록 = **같은 exec 의 직전 재생 미완료**.
  - 신호 플래그 512개 전부 0 + CPU idle 98.8% + CUDA 콜백 스레드 전원 대기 = 걸린 graph 재생이 **첫 host 노드에 도달하기도 전에** GPU 안쪽에서 정지. 층0 초입의 TP collective 로 추정.
  - 종합: 통신그룹을 A/B 로 분리해도 (수리 5), 두 그룹의 collective kernel 이 4 rank GPU 에 동시 상주하는 한 rank 간 진행 격차 (rank0 만 CPU 게이팅) 로 spin-대기 상호 점유가 발생하는 것으로 판단. 스트림 2개 동시 재생의 구조적 장벽 3층째.
- 방침 전환: **순차 재생 모드 (SGL_IL_SEQ=1) 는 생존 입증 완료** (24/24, 3.4초). 순차도 A 의 deferred CPU 작업이 B 의 GPU 시간과 겹치는 실질 overlap 을 가지므로, 동시 재생 논쟁을 지속하는 대신 순차 모드의 C=32 처리량을 측정하여 사전 등록 기준 (≥360 채택 / <+5% 중단) 으로 판정한다. 측정 진행 중.

## 최종 결론 (2026-09-08 02:15) — 교차 실행 공사 종결: 기각

측정 (같은 밤, 같은 조건, sonnet 512/128, C=32, num-prompts 128):
- 기준 구성 (SGL_INTERLEAVE 미설정): **319.9 tok/s** (TPOT 83.3ms) — 어제 324~327 과 오차 범위. 새 wheel·패치 전체가 플래그 미설정 시 무손상임을 처리량 수준에서 확인.
- 순차 교차 실행 (SGL_IL_SEQ): **254.5 tok/s** (TPOT 109.4ms) = **-20.4%**. 배치 이분할로 층별 고정 비용 (kernel 기동·통신·신호) 이 2배가 되는데 순차 재생엔 이를 상쇄할 동시성이 없음.
- 동시 재생: 미생존. 사전 등록 기준 "생존 불가 또는 +5% 미만 → 중단" 충족.

동시 재생 불가의 3층 구조 (각각 실측·수리·재발로 입증):
1. KTransformers 블로킹 host 함수 sync — CUDA host 콜백 직렬성과 결합해 교착. **수리 완료** (신호식 대기: 큐 내 flag 작업 + cuStreamWaitValue32. 버그 #6 UAF, #7 deferred 층 미적용, #8 eager/captured 슬롯 충돌까지 수리).
2. 채널별 CPU 큐 2개가 같은 층 AMX MoE 객체를 동시 실행 → 객체 내부 상태 경합 segfault. **수리 완료** (단일 큐 공유 — 순차 실행이 경합 원천 차단).
3. rank0 에만 CPU 게이팅이 있는 비대칭 구조에서, A/B 통신그룹 분리에도 불구하고 두 그룹의 collective kernel 이 동시 상주하면 rank 간 진행 격차로 spin 상호 점유 → GPU 수준 정지 (플래그 전부 0, CPU idle, 콜백 스레드 대기, 걸린 graph 는 첫 host 노드 이전에서 정지). **애플리케이션 코드 수준에서 해소 불가로 판단, 미해결**.

부산물 (자체 가치, 보존):
- 신호식 대기 C++ 코드 (비블로킹 sync): kt 의 host-콜백 직렬화 교착 계급을 제거. 순차 이중-graph 재생을 가능하게 함. graph 캡처 host 노드 인자의 영속성 요구 등 upstream 제보 후보.
- 격리 수리 5건 (채널 버퍼 / 전용 graph 풀 / 전용 attention / 채널 큐→단일 큐 / 전용 TP 그룹): kt×sglang 다중-graph-세트 병용 일반에 필요.

안전 상태: SGL_INTERLEAVE 미설정 기본 구성 (hot-80, 319.9~327 tok/s, GSM 97.5%) 무손상. 서버는 기준 구성으로 기동된 채 종료.

## 5일차: 기각 철회 — 층 사다리 (layer ladder) 공사 착수 (2026-09-08 02:30~)

사용자 지적: "안된다는 소리만" — 진단이 가리키는 다음 설계를 시도하지 않고 접은 것이 잘못. 3층 장벽의 원인이 "두 graph 의 collective 순서가 rank 간 비결정"이라면, 순서를 결정적으로 고정하는 설계가 곧 다음 공사다.

설계 — 층 사다리 (SGL_IL_LADDER):
- 캡처 중 각 decoder 층 경계에 rank-로컬 memop 격자 기록. A graph: 층 L 종료 시 flag[384+L]=1 (`write_flag_on_stream`, C++ 신규 5줄). B graph: 층 L 진입 시 flag[384+L]≥1 대기 후 0 소비 (`wait_signal_on_stream` 재사용). A 의 층0 진입 시 사다리 전체 0 리셋 (스텝 간 오염 차단).
- 구현: model_runner 에 forward_pre/post hook 을 캡처 전 등록, 본체는 `is_current_stream_capturing()` 일 때만 동작 (eager 시 완전 무효) — 결합 캡처 없이 두 graph 에 대응 격자를 새김.
- 효과: 모든 rank 가 같은 격자 → B_L 은 어느 rank 든 A_L 완료 후에만 시작 → collective 전역 순서 A_L→B_L→A_L+1 로 고정 → 3층 장벽 원인 제거. 겹침 보존: A 가 층 L+1 CPU 대기로 GPU 를 비울 때 B 의 층 L GPU 계산이 채움.
- 부수 규칙: ladder 모드는 decode 를 항상 이중 재생 (홀수 bs 는 ceil/floor 분할, min_bs=2). 단독 A 재생은 사다리 플래그를 쓰고 소비자가 없어 금지 → bs=1 등 부적격은 eager 센티널로 처리.
- 판정 기준 유지: C=32 ≥360 (기준 319.9).

### 사다리 1차 (08:11~08:20): 행업 — 원인 = rank1-3 에 사다리 무효
- hook 62층 전 rank 등록 확인, 부팅·품질 통과, 24 버스트 행업 (GPU 4장 100%).
- 플래그 직독: rank0 신호·사다리 전부 0, rank0 의 진행 중 A graph 가 층0 조차 못 끝냄 (첫 ladder write 이전). `slots ready` 는 rank0 1건뿐 → **CPU MoE 가 rank0 에만 있어 rank1-3 엔 공유 CPUInfer 가 없고 hook 본체가 no-op** 였음. 사다리가 rank0 에만 있으면 순서 고정이 안 되고, 오히려 rank0 에서 B_0→A_0 의존이 생겨 다른 rank 의 B 커널 (SM 점유) → rank0 B_0 → rank0 A_0 → 다른 rank A_0 (SM 못 얻음) 의 순환 완성.
- 수리: 모든 rank 에 사다리 전용 소형 CPUInfer(스레드 2) 를 생성, 플래그 배열을 rank 공통 격자로. 2차 시험 진행 중.
- 사다리 2차 (08:20): 부팅 사망 `[kt-signal] cudaHostAlloc FAILED err=900` (= cudaErrorStreamCaptureUnsupported). 사다리 인스턴스의 플래그 배열 할당이 첫 hook 호출 = 캡처 중에 일어남. **부수 확인**: 5·6차의 GPU Xid 31 MMU fault 도 같은 기전 (B 채널 전용 CPUInfer 의 할당이 캡처 중 조용히 실패 → NULL 주소 memop) 으로 설명됨 — 계측 abort 가 이를 드러냄. 수리: 인스턴스 생성 직후 eager 에서 더미 memop 으로 선할당 + synchronize. 3차 진행 중.

### 사다리 3차 (08:23~08:35): 행업 지속 — 격자 판독과 정량 재평가
- 전 rank 사다리 = `00111111000…`: A 는 층 7 완료·층 8 내부, B 는 층 1 내부, **양쪽 모두 collective 에서 정지**. 사다리 (A_L→B_L) 는 층이 다른 A_8 / B_1 의 상대 순서를 못 고정 → 두 통신그룹 커널의 상호 SM 점유 순환은 남음. 단일 그룹 + 완전 순서 (층당 4간선) 는 겹침을 거의 소거하므로 실익 없음.
- **통신 백엔드 확인**: TP allreduce = FlashInfer AllReduce Fusion (symmetric memory·multicast, SM90 자동 활성). 세트 B 그룹도 별도 SymmDeviceMemory 획득. 이 fusion 커널은 큰 grid 로 signal pad 를 spin 하므로 두 그룹 동시 상주 시 상호 점유 교착이 구조적으로 가능 → `--enforce-disable-flashinfer-allreduce-fusion` 으로 소형 custom AR 전환 시험 진행 중.
- **정량 재평가** (8-29 라우팅 트레이스 18,776 토큰 재분석): hot-80 커버리지 97.0% (min 85.7). bs=32 에서 층당 기대 distinct cold expert **6.7 개** (bs=16: 3.6, bs=64: 12.1). 스텝당 cold 바이트 ≈ 9.8 GB → CPU 구간 40ms 기준 246 GB/s = DDR5 실효의 ~절반. **CPU 구간은 대역폭 한계가 아니라 층당 고정비 (~650µs/층) 지배**. 시사점 ① 절반 분할 시 cold expert 3.6+3.6 ≈ 7.2 로 CPU 바이트 페널티 거의 없음 (교차 실행 상한 회복: GPU 측 1.45× 부담만) ② 층당 고정비 절반 감축 시 CPU 40→~20ms, 스텝 66→~45ms (+45%) — 교차 실행보다 크고 순수 호스트 측 공사. 두 방향은 합성 가능.
- kt-kernel CPU 측 판독: 워커는 50ms spin 후 sleep (정상 decode 에선 wake 지연 아님). qlen>1 forward = NUMA 풀당 5 fork/join 구간 (48스레드 동기화 ~10회/층). 나머지는 GPU↔CPU 연결층 (host 함수 3~4개/층, D2H/H2D 복사, 큐) 몫으로 추정 — 마이크로벤치 (`mb480_cpu_layer.py`) 와 nsys 로 분해 예정.

### 사다리 4차 (08:42~08:52): AR fusion 비활성 — 여전히 행업. 우선순위 전환
- `--enforce-disable-flashinfer-allreduce-fusion` 적용 확인 (로그 "forcibly disabled"). 그러나 allreduce 자체는 `custom_all_reduce_v2` (symmetric memory + multicast) 로 동일. 격자: 전 rank 에서 A 층 7~8 내부 / B **층 0 내부** 정지 — 세트 B 의 첫 collective 부터 완료 안 됨. 두 그룹의 symm-mem AR 커널 동시 사용 자체가 미완료로 귀결 (작업공간·signal 시퀀스 공유 여부 미확인).
- **정량적 판정 (분할 자체의 비용)**: CPU 층당 비용이 고정비 (~650µs/층) 지배이므로 마이크로배치 2분할 = 층당 CPU 호출 2회 = CPU 시간 ~2배 (40→~80ms). 순차 모드 +26ms (83→109) 가 이 현상 (deferred 겹침으로 일부 상쇄). **고정비를 먼저 줄이지 않으면 어떤 형태의 교차 실행도 이길 수 없다.** 반대로 고정비 감축 (650→~250µs) 은 단독으로 스텝 66→~45ms (+40%대).
- 순서 전환: ① CPU 층당 고정비 분해 (480B 마이크로벤치 `mb480_cpu_layer.py`: 순수 CPU 호출 비용 vs n_cold / nsys: 통합층 host 함수·복사 지연) → 감축 공사 (호스트 측) → ② 그 후 교차 실행 재평가 (교착 미해결 항목: 두 그룹 symm-mem AR 동시 사용).

### CPU 층 호출 마이크로벤치 (480B AMXINT4, 96스레드/2풀, 08:53) — `eval/results/20260908_085300_ide030_mb480_cpu_layer/`
| T(토큰) | n_cold 1 | 2 | 4 | 7 | 12 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| 32 | 8735µs | 5227 | 4858 | 4849 | 4529 | 4823 | 5297 |
| 16 | 4357 | 2632 | 2509 | 2535 | 2406 | 2783 | 3727 |
| 1 | 1567 | 812 | 435 | 454 | 440 | 443 | 467 |

해석:
- T=1 (쌍 8개, distinct ≤8): n_cold≥4 에서 ~440µs 평평 = 8 expert × 23.6MB = 189MB / 0.44ms ≈ **430 GB/s — DDR 스트리밍 한계 구간**.
- T=32, n_cold=7: 같은 165MB 인데 4.85ms → 쌍(토큰×expert) 당 ~19µs 의 **vec_mul 경로 계산 한계** (qlen ≤ 4·E/k=80 이면 AMX 타일 대신 벡터 곱). 시간 ∝ T.
- n_cold=1 이 유난히 느림 (8쌍이 한 expert 에 몰리면 1.5ms): expert 단위 병렬 분할 (nth) 이 expert 수가 적으면 스레드를 못 채움.
- 서빙 (bs=32, hot-80): 층당 cold 쌍 ≈ 8, distinct ≈ 6.7 → 스트리밍 ~350µs + 계산 ~150µs ≈ 500µs × 62 = 31ms. 실측 CPU 구간 40ms 와의 차 ~9ms = 통합층 (host 함수·D2H/H2D·큐) 추정.
- **결론: 마이크로배치 분할은 층당 호출 2회 = 스트리밍·고정비 2배 → 구조적으로 불리. CPU 구간 감축 경로 4개**: ① hot expert 80→96 (cold 선택 절반 → CPU 바이트 −45%, 스텝 −18ms 추정, GPU 여유 14GB 내 가능 여부가 관건) ② DDR 활용 430→550 GB/s (NUMA·프리페치) ③ vec_mul 경로 (M 패딩 후 AMX 타일) ④ 통합층 9ms. ①+DDR 상한 측정 진행 중.

### DDR 상한·deferred 겹침 (09:00~)
- DDR 실측 (torch, 96스레드, 2.0GHz): 읽기 ~390 GB/s, 복사 ~250 GB/s. CPU MoE 의 430 GB/s 는 이미 천장 → 경로 ② 제외. 경로 ③ (vec_mul) 도 서빙 구간 (expert 당 행 ≈1.2) 에선 무의미 → 제외. 남는 호스트 측 경로: ① cold 바이트 감축 (hot-96: 부팅 실패, 원인 확인 중) ④ 통합층 ~9ms ⑤ **deferred 겹침**.
- ⑤ `--kt-max-deferred-experts-per-token N` 이 현재 **미설정(0)** = CPU 구간 전부 노출. 의미: 토큰별 가중치 하위 N개 expert 의 CPU 계산을 다음 층 GPU 시간과 겹치고 그 기여를 **한 층 늦게** 더함 (근사 → 정확도 재게이트 필수). N=2,4 스윕 진행 중 (C=32 벤치 + greedy 품질, 이후 GSM 40).
- deferred 스윕 결과 (hot-80, C=32): N=2 **330.5** tok/s (TPOT 80.1) / N=4 **332.3** (TPOT 78.9) — 기준 319.9 대비 +3.3% / +3.9%, greedy 정상. 이득이 작은 이유: deferral 은 토큰의 8개 선택 중 가중치 하위 N개를 고르는데 선택의 97% 가 hot(GPU) expert 라 cold 가 걸릴 확률이 낮고, 겹침 창도 다음 층 attention (~0.2ms) 뿐. (`eval/results/20260908_0856*_ide030_hot80_def2_c32`, `_def4_c32`)
- hot-96 1차: 가중치 후 여유 8.45 GB, KV 부족으로 부팅 실패 (mf ≥0.892 요구) → mf 0.92 / max-total-tokens 24576 / graph bs≤64 로 재시도 중.

### hot-96 (09:05): **458.9 tok/s @C=32 (TPOT 54.8ms) = 기준 319.9 대비 +43.4%** — 사전 등록 기준 (≥360) 통과
- 구성: hot-80 대비 `--kt-num-gpu-experts 96 --mem-fraction-static 0.92 --max-total-tokens 24576 --cuda-graph-max-bs 64` (캡처 시 여유 4.84 GB). `eval/results/20260908_090*_ide030_hot96_mf92_c32/`
- 기전 (마이크로벤치 예측과 일치): 라우팅 트레이스상 hot-96 커버리지 ≈98.5% → cold 선택 절반 → 층당 cold expert 스트리밍 바이트 −45% → CPU 구간 40→~26ms.
- 품질 게이트 진행 중 (greedy Q1 이 " Paris." 로 짧게 끝남 — EOS 조기 출력 여부 확인 필요): GSM8K 40 + 부하 중 torch profiler (통합층 분해용).
- 다음: hot-96 + deferred N=4 합성, C=64, hot-104 (메모리 한계: +5.9 GB 필요, 현 여유 4.8 GB → mf 0.95 시도 가치 검토).
- hot-96 품질 게이트: **GSM8K 40문항 95.0% (38/40)** — 기존 97.5% (39/40) 와 1문항 차 (표본 잡음 범위). 통과 판정, 최종 구성에서 100문항 재확인 예정. 프로파일 트레이스 4 rank 수집 (`eval/results/20260908_090741_ide030_gate_hot96/`).
- 프로파일 (hot-96, C=32 램프업 구간 decode 3스텝, TP0): GPU 창 26~32ms = 커널 15~18ms (fused_moe 7.9, fp8 GEMM 2.9, AR fusion 1.3, attention ~1.2) + 복사 2ms (H2D 19µs×60, D2H 3µs×181 — 통합층 복사는 사소) + **유휴 10~12ms (CPU 대기 memop 갭, 층당 ~0.17ms)**. 실측 TPOT 54.8 과의 차이는 스텝 사이 호스트 구간 의심 → 분석 중.
- **hot-96 + deferred N=4 (09:13): C=32 483.2 tok/s (TPOT 51.0ms) = 기준 319.9 대비 +51.0%**. 최종 검증 (GSM 100 / 정상 상태 프로파일 / C16·32·64) 예약. hot-104 (mf 0.95) 시도 진행 중.
- hot-96+def4 @C=64: 412.3 tok/s (TPOT 86.0) — C=32 (483) 보다 낮음: max-total-tokens 24576 에 64×640≈41K 토큰이 안 들어가 KV 압박 (hot-80 의 C=64 343 대비로는 +20%). hot expert 수 ↔ KV 용량 (동시성 상한) 교환 관계 확인. hot-104 (mf 0.95): 부팅 실패.

## 최종 (2026-09-08 09:30): hot-96 + deferred 4 — C=32 **490.9 tok/s (+53.5%)**, GSM8K 100문항 97.0%
- C=16 358.6 / C=32 490.9 (프로파일 중 462.1) / C=64 408.2. 정상 상태 분해: 스텝 39ms = GPU 커널 27.7 + 복사 1.9 + CPU 대기 9.9 (hot-80 의 CPU 대기 40 → 9.9). 표·근거·남은 손잡이: `FINAL_RESULT_20260908.md`.
