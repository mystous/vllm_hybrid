# IDE_029 — Claude 작업 메모

- 순환 적합 금지가 이 후보의 생명선: 모델 파라미터 ← 스펙+마이크로벤치만. end-to-end 수치로 조정 금지. 예측은 측정 전 커밋 (해시 고정).
- 컨테이너: sgl-kt / sgl-kt2 재사용. 신규 필요 시 `features/IDE_023/scripts/setup_kt_container.sh` (패치 4건 포함). 다중 인스턴스는 cgroup cpuset 필수 (kt 절대-pin 결함).
- P2 (`--kt-num-gpu-experts` > 0) 와 P4 (TP=2) 는 **사전 smoke 필수** — 지금까지 안 써본 경로.
- 벤치 클라이언트 = vllm-h100 컨테이너, sonnet 512/128 seed 42, warm-vs-warm.
- 운영 수칙: pkill bracket / 이중 그리드 금지 / nvidia-smi 유휴 확인 / turbo OFF 명시.
