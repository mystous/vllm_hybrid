# IDE_033 (설계 초안, 미등록) — Host-노드 없는 CPU↔GPU 핸드오프 (Callback-free Expert Offload Handoff)

## 동기 (IDE_032 kill test 잔여 캡)
- cold 전량 deferral(N=8) 로도 스텝당 ~3ms 만 줄고 유휴 ~8ms 가 남음. 가설: KTransformers 경로는 층당 host 함수 노드 3개 (submit immediate / submit deferred / sync = `cudaLaunchHostFunc`) → 62층 × 3 × 콜백 디스패치 지연(~30~50µs) ≈ 6~9ms/스텝 의 GPU 정지. 이는 CPU 작업량과 무관한 **동기화 프리미티브의 바닥**이라 deferral·hot 확대로 제거 불가.
- 판별 실험: N=8 정상 상태 프로파일에서 유휴 갭의 분포 — (a) 층당 여러 개의 30~100µs 갭이면 host 노드 지연, (b) 소수의 큰 갭이면 CPU 대기.

## 설계
GPU→CPU (작업 제출) 과 CPU→GPU (완료 통지) 를 모두 **mapped pinned memory 플래그 + 스트림 memop** 으로 바꿔 CUDA 호스트 콜백 스레드를 경로에서 완전히 제거한다.
1. **작업 디스크립터 사전 등록**: 층·슬롯별 CPU 작업의 인자 (입력/ids/weights/출력 포인터, bsz 포인터) 는 정적 버퍼라 캡처 시점에 고정 → `arm_task(slot, task)` 로 CPUInfer 에 등록 (graph 재생마다 동일).
2. **GPU→CPU 제출**: graph 안에 `cuStreamWriteValue32(go_flag[slot], seq)` memop (host 노드 대신). seq 는 재생마다 증가해야 하나 graph 상수라 불가 → 대신 **토글 비트 + CPU 측 소비 리셋**: GPU 는 1 을 쓰고 CPU 폴러가 읽은 뒤 0 으로 되돌림 (기존 신호 슬롯의 역방향).
3. **CPU 폴러 스레드** (1개, spin + `_mm_pause`): go_flag 배열을 폴링해 세트된 슬롯의 등록 작업을 TaskQueue 에 enqueue 후 플래그 리셋. 지연 ~1µs.
4. **CPU→GPU 완료**: 기존 `submit_signal`/`wait_signal_on_stream` (memop) — 이미 구현·검증됨 (IDE_030 부산물). 단 `submit_signal` 의 host 노드도 폴러가 대신 (작업 완료 시 CPU 가 직접 done_flag 세트).
5. 결과: 층당 host 노드 0개. GPU 스트림은 memop wait 에서만 정지 (SM 비점유). deferral 겹침이 실제 CPU 시간에만 의존하게 됨.

## 예측 (모델 v2 + 프로파일)
- hot-96 C32: 유휴 9.9 → (host 노드 바닥 제거) ~3 → 스텝 39 → ~32ms (+22%); 여기에 N=8 deferral 의 실제 겹침이 살아나면 CPU 노출 → ~0 → 스텝 ~28ms (+40%).
- 사전 등록 kill test: 프로파일이 (a) 로 판별되면 착수; 구현 후 hot-96 N=8 C32 ≥580 tok/s (+15% vs 505) 면 진입.

## 선행 (K1 기준)
- KTransformers: CUDA graph 내 "CPU 스핀" 으로 launch 오버헤드 제거 (논문 §3) — 그러나 제출/동기화는 host 콜백. 우리 = 콜백 자체 제거 (양방향 폴링). MoE-Lightning·Fiddler 는 PCIe 전송 기반이라 해당 없음. 정독 필요: NVIDIA "stream memops" 활용 선례 (NCCL LL 프로토콜의 플래그 폴링과 유사 — 인용).
