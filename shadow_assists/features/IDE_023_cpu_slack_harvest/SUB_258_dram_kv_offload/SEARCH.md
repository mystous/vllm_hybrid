# SUB_258 — 신규성 탐색 로그 (harvest win 위에 신규 기여 쌓기)

본 win(SR-006, +76.7%)의 메커니즘은 vLLM native CPU offload = 신규 아님. 그 위에 신규 기여가
가능한지 후보별로 탐색.

## 후보 (a) DSA(/dev/dsa) 가속 전송 — ❌ 아키텍처적 dead-end (2026-06-17)
**가설**: SR-006의 GPU↔DRAM 전송(54GB/s)을 DSA로 내려 SM/copy-engine 비점유 → decode와 진짜 overlap.

**검증**:
- DSA 가용: `dsa0`(numa0)/`dsa1` enabled, `wq0.1` 666 권한 사용가능, `clients:0`, op_cap memmove 지원.
- **그러나 전송 경로 사실**: vLLM offload 전송 = `vllm/v1/kv_offload/worker/cpu_gpu.py:267`
  `ops.swap_blocks_batch(batch_src, batch_dst, batch_sizes)` **CUDA 스트림 위에서 실행** =
  GPU 엔진이 PCIe로 끄는 async DMA(cudaMemcpy류). **CPU memcpy 없음, 전송 중 CPU 코어 완전 idle.**
- **DSA 도메인 = host DRAM↔DRAM/fill/compare.** GPU VRAM↔host DRAM(PCIe)은 DSA 사정권 밖
  (DSA는 호스트 메모리 mover; GPU BAR로의 MMIO write는 비표준·저성능·어느 프레임워크도 미지원).

**결론(3중)**: (1) DSA는 GPU↔DRAM PCIe 링크에 닿지 못함. (2) vLLM 전송은 이미 GPU DMA(최적, CPU 0).
(3) 그 경로에 DSA가 offload할 CPU memcpy가 **애초에 없음** — harvest 목표(CPU idle 회피)는 이 op에선
이미 달성돼 있고, DSA로 더 개선할 여지 0. **DSA 가속 전송 = 신규지만 부적용(死).**

**DSA가 닿을 수 있는 유일 지점**(미채택): DRAM↔DRAM 작업 — 다단 tier(DRAM→NVMe staging),
NUMA migration, KV 압축 등. 단 vLLM native offload는 단일 tier(GPU↔DRAM 직결)라 DRAM-DRAM 단계
자체가 없음 → 현 경로엔 진입점 없음. (다단 tier 설계 시 재검토 가능하나 그건 별도 큰 작업.)

## 후보 (b) 적응 게이트 (KV 압력 기반 offload on/off) — 🟡 구현가능, 신규성 약함
크로스오버서 cache≥working-set이면 offload −10%(순오버헤드). 압력(eviction rate/GPU prefix hit)을
측정해 offload store를 동적 토글하면 −10% 회피. vLLM은 현재 enable 시 무조건 store(정적). 구현은
가능하나 "adaptive cache admission"은 캐싱 문헌 통념 — 신규성 약함. 보류.

## 후보 (c) fetch-vs-recompute 공동 스케줄링 — 🔵 신규 가능성, 단 헤드룸은 대형/롱컨텍스트 한정
재사용 prefix-block을 PCIe-fetch(DRAM) vs GPU-recompute 중 **자원 병목에 따라 per-block 라우팅**:
decode가 memory-bound라 GPU compute idle → recompute "공짜", PCIe idle → fetch "공짜". 둘을
동시 포화(neither idle)하면 pure-fetch(B)·pure-recompute(A) 둘 다 상회 가능. vLLM은 all-or-nothing
(connector가 fetch만) — 미구현 = 신규 후보.
**그러나 헤드룸 측정**: SR-006의 B는 fetch 0.29s / wall 1.85s = 전송이 병목 아님(decode-bound).
7B 구성선 arbitration 이득 거의 0. **전송이 실제 병목이 되는 regime(대형 모델·롱 컨텍스트, KV
bytes/token 큼 + PCIe 포화)에서만 헤드룸** → 다음 검증은 70B/롱컨텍스트로 fetch 병목화 후 측정.

### (c) 검증 완료 — ❌ 헤드룸 0 (2026-06-17, 70B 실측)
70B-NVFP4 고압(working-set 180K/cache 40K)서 측정: B=+97.5%(160 vs 81 tps), fetch 97.6GB/1.76s=55GB/s.
**B 병목 = decode(wall 9.0s) ≫ fetch(1.76s, async 오버랩)** → fetch는 이미 critical-path 밖.
fetch↔recompute 재배분은 9s decode floor 못 움직임 + fetch가 recompute보다 3.5× 싸(TTFT 0.097 vs 0.337)
이전이 손해. vLLM엔 `recompute_kv_load_failures`(실패 fallback)만 있고 능동 arbitration은 신규지만
**이득 0이라 무의미.** → 후보 (c) 死.

## 종합 결론
3개 신규 후보 (a)DSA/(b)적응게이트/(c)fetch-vs-recompute 전부 막힘(아키텍처/약신규성/헤드룸0).
**SR-006은 신규 알고리즘이 아니라 harvest 명제의 강력한 실증·특성화**(7B +77%/70B +97.5%, 모델 클수록
win↑). CERES 논문에 evidence로 기여. 다음 신규성은 SUB_258 밖에서 찾아야(단일tier→다단tier DSA,
또는 전혀 다른 축). [[largemodel-b200-comm-bound-characterization]] "작동하면 이미 upstream" 패턴 재확인.

## 다른 축 탐색 (2026-06-17, "다른 축 지금 진행해")
세 후보 recon → 전부 upstream 또는 이미 fork 구현:
- **(1) 다단 tier(NVMe/GDS)**: lmcache/mooncake/hf3fs(NVMe)/nixl connector 다수 upstream. DSA-DRAM staging은 단일tier엔 진입점 없음(기확인). = 비신규.
- **(2) disaggregated prefill/decode**: nixl/mooncake/p2p_nccl/moriio/kv_producer·consumer 완비 upstream. = 비신규.
- **(3) 구조화출력 CPU 경로**: 이 fork에 **이미 SUB_201 L5(multi-thread grammar advance, `VLLM_GRAMMAR_MT_*`) + parallel bitmask fill 구현**. SUB_208(xgrammar jump-forward)은 **net-positive 미달**(JF fire 0). = 이미 시도.
  - 빠른 실측(Qwen2.5-7B, conc200, max-num-seqs256): plain 172 req/s vs **structured(JSON schema) 136 req/s(−21%)**. 구조화 제약의 CPU grammar+bitmask 오버헤드는 실재(GPU 비점유 순수 CPU work)이나, **그 지점이 바로 L5가 이미 다루는 곳** → 추가 이득은 비신규 레버(L5/AMX-bitmask) 위 incremental.
**결론**: "다른 축"도 신규성 막힘(upstream or fork-구현됨). 세션 전반 패턴 재확인 — GPU-bound LLM 서빙에서 CPU 신규 기여 여지는 측정으로 작동 검증되는 순간 이미 존재.
