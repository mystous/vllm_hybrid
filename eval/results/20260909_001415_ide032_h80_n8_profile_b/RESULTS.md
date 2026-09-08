# hot-80 N=8 (cold 전량 deferral) 정상 상태 프로파일 — 겹침 붕괴 기전 (2026-09-09 00:20)
정상 스텝 87.2ms = 커널 26.5 + 복사 2.4 + **유휴 58.8**. 갭 126개/스텝, 그중 **sync 자리 (combine → H2D) 60개 = 53ms (각 ~0.88ms)**, submit 자리 62개 = 3.1ms.

기전 (task_queue.cpp 판독 + 수치):
- KT TaskQueue = 워커 1개 순차 실행, `sync(allow)` = pending ≤ allow 대기. 층 L+1 의 immediate 작업은 단일 FIFO 에서 층 L 의 deferred **뒤**에 줄서므로, sync(L+1) 은 def(L) 이 끝나야 풀림. 따라서 숨김 창 = GPU 한 층 (~0.43ms) 이고 노출 = max(0, def − 0.43).
- hot-96: def ≈ 0.3ms < 0.43 → 완전 은닉 (유휴 = host 노드 바닥만). hot-80: 노출 0.88 → **def ≈ 1.3ms/층 = immediate 경로 (N=0: 0.65ms/층) 의 2배**. deferral 이 hot-80 에서 무효인 이유는 "겹침 실패" 라기보다 **deferred 작업 자체가 2배 느린 것** (원인 미확인: incremental 누적 경로·출력 슬롯 처리·rows 중복 계산 등). 커널 26.5ms 도 겹치지 않는 이유 = def 가 GPU 층보다 훨씬 길어 매 층 GPU 가 sync 에서 대기.
- 시사: ① IDE_033 (host 노드 제거) 은 hot-96 체제 (+18% 예측) 에 유효, hot-80 체제엔 미미. ② hot-80 체제의 실제 손잡이 = deferred 작업 지연 2× 의 원인 제거 → 성공 시 hot-80 스텝 87 → ~60 (+45%) 이고 hot-80 은 KV 131K 로 C64 이상에서 hot-96 을 이김 → 큰 조합 이득. ③ 검증 = TaskQueue 작업별 소요시간 계측 (다음 빌드) 으로 imm/def/signal 각각의 층당 시간 실측.
- 원본: `gaps_TP0.txt`, 트레이스 4 rank.
