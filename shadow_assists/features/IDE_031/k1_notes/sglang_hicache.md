# SGLang HiCache (공식 문서·LMSYS 블로그) — K1 정독 노트

- **서지**: LMSYS 블로그 "SGLang HiCache: Fast Hierarchical KV Caching with Your Favorite Storage Backends" (2025-09-10); SGLang 문서 "HiCache System Design and Optimization", "SGLang HiCache Best Practices" (docs.sglang.ai / docs.sglang.io); Mooncake 문서 "Mooncake x SGLang HiCache System Design"; NVIDIA Dynamo "Using HiCache". 논문 아님.
- **전문 접근 여부**: **불가** (docs.sglang.ai, lmsys.org 접속 타임아웃; 로컬 sglang 소스도 이 dev 머신에 없음 — `find / -name hiradix_cache.py` 결과 없음). WebSearch 스니펫만 확보. 인용은 스니펫 문장.

## 구조·결정 변수
- 3계층: GPU HBM = L1, host memory = L2, 분산 스토리지 = L3 (Mooncake, HF3FS, NIXL, Tair 등). RadixAttention 을 HiRadixTree 로 확장 (페이지 테이블 역할). Cache controller 가 load/backup 자동 관리.
- 스니펫 인용: "HiCache organizes GPU memory as L1, host memory as L2, and distributed storage as L3."
- 운영자가 정하는 knob (문서 확인): `--hicache-ratio` (host pool / device pool 크기 비, 예 2 = host 2배), `--hicache-size` (GB), `--hicache-write-policy` = write_through / write_through_selective (hit-count 임계 초과분만 백업) / write_back, `--hicache-storage-prefetch-policy` = best_effort / wait_complete / timeout, `--hicache-mem-layout` = layer_first / page_first / page_first_direct, `--hicache-io-backend`, `--hicache-storage-backend`.
- 스니펫 인용: "hicache-ratio: The ratio of the size of host KV cache memory pool to the size of device pool."
- 자동 결정은 캐시 교체·prefetch 시점·write-back 여부뿐. **host 크기 (ratio) 는 사람이 정함** — 어떤 값이 좋은지 고르는 모델 없음.

## 비용 모델 파라미터 출처와 검증
- 성능 모델·예측기 **없음**. 대역폭에 대한 언급은 정성적: 스니펫 인용 (Netpreme 협업 블로그) "Host DRAM-based offload increases capacity, but its PCIe-level bandwidth can become the limiting factor for KV-intensive inference, especially when TTFT SLO is tight." 즉 병목은 **PCIe** 로 서술, DRAM 읽기 대역폭 항은 없음.
- 최적화는 커널 수준: GPU-assisted I/O 커널 (CPU–GPU 전송 최대 3×), prefill 시 layer N+1 KV 로드를 layer N 계산과 겹침, L3 zero-copy.
- 보고 수치: Qwen3-Coder-480B 코딩 에이전트 시나리오 TTFT −56%, 처리량 2×, hit rate 40→80%; cache hit 시 TTFT −84%; 전체 최대 6× 처리량, TTFT −80%.

## expert·KV 경합 취급
- MoE expert offload 와 HiCache 동시 사용에 대한 문서 없음. kt-kernel (CPU expert) + SGLang 튜토리얼과 HiCache 문서는 별개; 둘이 host 메모리·DDR 대역폭을 공유할 때의 간섭은 어디에도 모델링 안 됨 (스니펫 범위, 추측 포함). 관련 이슈 #12826 "HiCache for Hybrid and Sparse LLMs" 는 하이브리드 attention (Mamba 류) 대상이지 expert offload 아님.
- **호스트 DRAM 대역폭 모델링**: **아니오**.

## D1/D2/D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | KV 티어링만. expert 상주는 범위 밖. host 크기도 수동 |
| D2 | **아니오** | 예측 모델 없음 |
| D3 | **아니오** | PCIe 를 병목으로 정성 언급할 뿐, 전이점 분석 없음 |
| 호스트 DRAM BW | 아니오 | PCIe 만 |

## 우리와의 delta
- HiCache 는 우리 M2 의 **도구** (KV 를 DRAM 으로 내리는 경로). HiCache 자체는 "얼마나 내릴지" 를 정하지 않으며, cold expert 스트리밍과의 DDR 대역폭 공유를 모른다. 우리 분할 정책이 HiCache 의 `--hicache-size` / `max-total-tokens` 를 출력으로 내는 구조가 성립. 위협도 **하**. 단 kt × HiCache 동시 사용은 문서상 미검증 → PLN_008 의 "생존 시험 먼저" 판단 유지.
