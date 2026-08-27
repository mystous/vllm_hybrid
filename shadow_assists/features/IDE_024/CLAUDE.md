# IDE_024 — Claude 작업 메모

- vLLM (TP=8) 의 worker 프로세스는 GPU affinity 에 따라 socket 0/1 에 걸침 — BG pin 은 `lscpu` 의 NUMA CPU 목록과 vLLM 프로세스의 실제 affinity (`taskset -pc <pid>`) 를 확인 후 잔여 코어로.
- HT sibling (0-55↔112-167, 56-111↔168-223) 공유 금지 — physical core 단위로 분리.
- 판정 지표는 서버 합산 (GPU tps 불변 + BG 처리량) — CPU busy% 단독은 성공 아님.
