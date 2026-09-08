# PLN_008 M2 — HiCache 공유 prefix 이득·간섭 (2026-09-08 20:09)
워크로드: random, 입력 1600 / 공유 prefix 1500 / 출력 256, C=32, 32요청, 2회 반복(w1=cold, w2=warm). hot-96+def4, GPU KV 49152 토큰.

| 모드 | pass | TTFT | TPOT | tput(tok/s) |
|---|---|---|---|---|
| gpu (radix만) | w1 | 31992 | 128.1 | 97.6 |
| gpu | w2 | 28746 | 128.1 | 101.6 |
| hicache (host 64GB) | w1(cold) | 33423 | 123.1 | 98.2 |
| hicache | **w2(warm)** | **13320** | **112.6** | **148.2** |

발견:
- **공유 prefix 워크로드에서 HiCache w2(warm)는 TTFT 33s→13s(−60%), 처리량 98→148 tok/s(+51%)**. GPU radix(w2)는 개선 없음(29s) — GPU KV 49152 가 32×1600=51200 작업셋+prefix 를 못 담아 prefix 가 evict, HiCache(host 64GB)만 prefix 를 보존해 재사용.
- TPOT 도 소폭 개선(128→112.6): warm 패스는 prefill 재계산이 줄어 decode 스텝 사이 prefill 간섭이 감소.
- **정책 함의**: GPU KV 가 작아 prefix 가 evict 되는 멀티턴/공유-prefix 워크로드에서 HiCache 는 실이득. 이는 정책의 KV 축을 "실행 중 KV offload" 가 아니라 "prefix host 계층 + 그로 인한 GPU KV 여유" 로 재정의한 근거(M2 정정)와 정합.
- MoE-Lens 의 "DDR 경합 비-binding" 대조: 여기서 warm HiCache 가 이득을 낸 건 DDR 경합보다 prefill 재계산 절약이 커서. cold 스트리밍과의 DDR 간섭이 이득을 잠식하는 임계(더 큰 C·긴 prefix)는 추가 실측 대상.
