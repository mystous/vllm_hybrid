# IDE_025 — Claude 작업 메모

- 컨테이너의 vLLM 버전에서 CPU offloading 활성 flag 를 먼저 확인할 것 (버전에 따라 `--kv-transfer-config '{"kv_connector":"OffloadingConnector",...}'` 형식 상이).
- 압박 emulation 은 pool 축소로 하되, 보고서에는 "실제 long-context 의 대리" 임을 명시.
- pinned memory 대량 할당 시 NUMA locality (`TSK_004` 의 numa_aware 경험 참조).
- 측정은 반드시 hit 카운터와 함께 — merged/dropped 0% 인데 이득을 주장했던 실수 반복 금지.
