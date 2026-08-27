# IDE_023 — Claude 작업 메모

- 이 노드는 k8s 워커 (containerd). **Docker Engine 없음** — `~/bin/docker` = sudo nerdctl 셔임. 모든 컨테이너 `--net host`.
- HF cache: `HF_HOME=~/hetero-exp/models` (→ `~/.cache/huggingface` 심볼릭). R1-0528 은 여기로 다운로드.
- GPU 8장 전부 유휴 확인 후 사용. k8s 가 GPU 파드를 스케줄하면 충돌 가능 — 기동 전 `nvidia-smi` 재확인.
- CPU turbo OFF (2.0GHz 고정) — 성능 수치 해석 시 명시. turbo unlock 은 사용자 결정 사항.
- 측정 원칙 (legacy 운영 규칙 승계): "CPU became busy" 는 성공 아님. net throughput 만 binding.
- 선행 기각 맥락: dense 모델 CPU offload 는 본 노드 클래스에서 재실증 기각 (SUB_036/040/041/042) — 본 IDE 는 그 결론과 충돌하지 않는 유일 regime (GPU-only 불가능 모델) 만 다룬다.
