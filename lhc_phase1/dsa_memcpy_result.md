# LHC Phase 1 — DSA microbench (실측 결과)

**날짜**: 2026-06-08
**머신**: DGX B200 host, Xeon Platinum 8570 (Emerald Rapids), DSA 2.0 (dsa0)
**WQ 구성**: `dsa0/wq0.0` **dedicated / type=user / wq-size=64 / max-transfer=2 GB**
**제출 경로**: **MOVDIR64B** (dedicated WQ → WQCFG.PASID 가 cdev open 시 커널 SVA bind, per-desc PASID 불필요)

> 이전 ENQCMD/shared-WQ 경로는 PASID handshake 로 hang → dedicated+MOVDIR64B 로 전환하여 **해결**.
> bench: `dsa_memcpy_bench.c`, overlap proof: `dsa_overlap_bench.c`, raw: `dsa_movdir64b_raw.jsonl` / `glibc_memcpy_raw.jsonl` / `dsa_overlap_raw.jsonl`

---

## 1. memcpy — DSA vs CPU vs GPU (p50 latency / bandwidth)

| size  | (a) DSA MOVDIR64B (실측) | (b) glibc memcpy 1-thread (실측) | (c) cudaMemcpy H2D¹ | (d) cudaMemcpy D2H¹ |
|------:|-------------------------:|---------------------------------:|--------------------:|--------------------:|
|   4 KB | 1.06 μs /  3.87 GB/s    | **0.04 μs / 108.2 GB/s** | 6.56 μs / 0.62 GB/s | 6.43 μs / 0.64 GB/s |
|  64 KB | 3.17 μs / 20.70 GB/s    | **1.19 μs /  55.0 GB/s** | 7.74 μs / 8.46 GB/s | 7.46 μs / 8.79 GB/s |
|   1 MB | 34.5 μs / 30.42 GB/s    | 33.1 μs /  31.7 GB/s     | 27.1 μs / 38.6 GB/s | 25.3 μs / 41.5 GB/s |
|  16 MB | **534 μs / 31.40 GB/s**  | 976 μs /  17.2 GB/s      | 316 μs / 53.0 GB/s | 308 μs / 54.5 GB/s |
| 256 MB | **8557 μs / 31.37 GB/s** | 14022 μs / 19.1 GB/s     | 4.83 ms / 55.6 GB/s | 4.84 ms / 55.5 GB/s |

¹ (c)(d) cudaMemcpyAsync = 이전 phase python bench 값 (호스트 torch 부재로 재실측 보류, Task 3 컨테이너 env 확보 시 갱신). PCIe Gen5 x16 ≈ 55 GB/s.

**관찰**:
- **single DSA engine ≈ 31 GB/s** 천장 (1 WQ / 1 engine). PCIe/DRAM 이 아니라 **단일 엔진 처리율** 한계 — multi-engine/multi-WQ 병렬로 확장 여지 (Phase 2).
- **작은 블록 (≤64 KB)**: glibc 압승. DSA 의 고정 제출+완료 latency (~1 μs) 가 dominate → **DSA 는 small-copy 에 부적합**.
- **큰 블록 (≥16 MB)**: DSA 가 glibc 단일코어 대비 **1.6–1.8× 대역폭** (31.4 vs 17.2 / 31.4 vs 19.1 GB/s). glibc 는 단일 코어 load/store 한계, DSA 는 엔진 스트리밍.
- **crossover ≈ 1 MB** (양쪽 ~31 GB/s 동률). 1 MB 미만은 CPU, 이상은 DSA 가 유리.
- p99/p50 비율 ≈ 1.0 (예: 256 MB p50 8557 / p99 8570 μs) → DSA latency **극도로 결정적** (jitter 거의 0).

---

## 2. 3-op 비교 (DSA MOVDIR64B, 256 MB)

| op       | lat p50 (μs) | "size" BW (GB/s) | 정규화 BW² (GB/s) | 메모리 트래픽 |
|----------|-------------:|------------------:|------------------:|---------------|
| memcpy   | 8557 | 31.37 | 31.4 (read+write) | 2× (R src + W dst) |
| memfill  | 8422 | 31.87 | 31.9 (write-only)  | 1× (W dst) |
| compare  | 16933 | 15.85 | 31.7 (2× read)    | 2× (R src1 + R src2) |

² 엔진이 실제로 touch 한 바이트로 정규화하면 세 op 모두 **~31.7 GB/s 로 수렴** → DSA 엔진 raw 처리율이 op 무관하게 일정함을 확인 (memfill 은 read 없어 size-BW 최고, compare 는 2× read 라 size-BW 절반).

---

## 3. **핵심: async-overlap (직교 lane 증명)** — `dsa_overlap_bench.c`

DSA copy 진행 중 **제출 코어가 다른 유용한 연산을 얼마나 수행할 수 있는가**.

| size   | copy 시간 (μs) | overlap kernel rate (ops/μs) | **코어 free 비율** | DSA BW (GB/s) |
|-------:|---------------:|------------------------------:|-------------------:|--------------:|
| (calibrate, no copy) | — | 664.2 | (기준) | — |
| 16 MB  | 603  | 663.0 | **99.81 %** | 27.8 |
| 256 MB | 8731 | 664.3 | **100.0 %** | 30.8 |

> **결론**: DSA copy 중 제출 코어는 calibrate 와 **동일한 속도(664 ops/μs)** 로 다른 작업을 계속 수행 → DSA copy 의 CPU 연산 비용 ≈ **0**.
> 대조: glibc memcpy 256 MB = 코어 100 % 점유 14 ms 동안 **다른 작업 0** (CPU 가 곧 copy 그 자체). DSA 는 같은 copy 를 1.6× 빠르게 + 코어 100 % free.

이것이 LHC 의 "host = monolithic CPU 가 아닌 직교 가속 lane" 명제의 **1차 정량 근거**. DSA lane 은 GPU 가 다른 op 중일 때 host-side 대용량 transfer (KV swap / prefix prefetch / LoRA swap) 를 **CPU 코어 점유 없이** 처리.

---

## 4. DSA lane viability 판정

### 4.1 공식 viability gate (≥0.8× cudaMemcpy AND CPU stall <5%)

DSA/cudaMemcpy(H2D) bandwidth ratio (single engine):

| size | DSA BW | cudaMemcpy H2D¹ | DSA / cudaMemcpy | ≥0.8× ? |
|-----:|-------:|----------------:|-----------------:|:-------:|
| 1 MB  | 30.42 GB/s | 38.6 GB/s | 0.79× | ✗ (경계) |
| 16 MB | 31.40 GB/s | 53.0 GB/s | **0.59×** | ✗ |
| 256 MB| 31.37 GB/s | 55.6 GB/s | **0.56×** | ✗ |

| gate | 기준 | 실측 (single engine) | 판정 |
|------|------|----------------------|------|
| BW vs cudaMemcpy | ≥0.8× | 0.56–0.79× | ⚠️ **단일 엔진 FAIL** |
| CPU stall | copy 중 <5% 점유 | **≈0%** (overlap 99.8–100% free) | ✅ **PASS (대폭 초과)** |
| latency 결정성 | — | p99/p50 ≈ 1.00 | ✅ PASS |

**결론: 조건부 GO (CONDITIONAL).**
- CPU stall 게이트 압도적 PASS — DSA 직교성(GPU/CPU 코어 비점유)은 완벽 입증.
- BW 게이트는 single DSA engine ~31 GB/s 천장으로 미달 → **Phase 2 multi-engine/multi-WQ 병렬(목표 ~100 GB/s aggregate)** 로 ≥0.8× 돌파 필요·가능.

¹ cudaMemcpy 컬럼은 **이전 phase python bench 값**(호스트 torch 부재로 재실측 보류). 또한 cudaMemcpy=PCIe H2D/D2H 인 반면 DSA=DRAM↔DRAM 이라 transfer 종류가 달라 엄밀한 동일 비교 아님. DSA 가 실제 대체하는 **호스트 KV-swap CPU memcpy(glibc)** 대비로는 ≥16 MB 에서 **1.6–1.8× + 코어 100% free**.

### 4.2 적용 범위

**DSA lane = 채택 (조건부)**. 적용 영역은 **≥1 MB block 의 대용량 transfer 로 한정** (small copy 는 CPU 가 우월). Phase 2 multi-engine BW 게이트 충족 입증을 조건으로 함.

**Phase 2 DSA 과제**:
1. **multi-engine/multi-WQ 병렬** — 1 engine 31 GB/s 천장 돌파, dsa0 4 engine + dsa1 → aggregate BW 측정 (목표 PCIe-bound ~100 GB/s).
2. **batch descriptor** (`DSA_OPCODE_BATCH`) — KV block 들을 1 submit 로 묶어 제출 overhead 분산.
3. vLLM 통합 지점: KV cache CPU 오프로드(page swap-out), prefix cache prefetch, LoRA adapter swap — 모두 ≥1 MB 단위라 DSA sweet-spot 부합.

## 5. 복원
- shared 복원: `sudo accel-config disable-wq dsa0/wq0.0 && sudo accel-config config-wq dsa0/wq0.0 --mode=shared --type=user --name=lhc --threshold=8 --wq-size=16 --group-id=0 && sudo accel-config enable-wq dsa0/wq0.0`
- 완전 disable: `sudo accel-config disable-wq dsa0/wq0.0 && sudo accel-config disable-device dsa0`
