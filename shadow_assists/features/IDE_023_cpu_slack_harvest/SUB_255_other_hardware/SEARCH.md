# SUB_255 — 다른 하드웨어로 GPU 서빙 최적화 탐색 (2026-06-17~)

> 사용자 지시: 다른 하드웨어 사용해 GPU 서빙 최적화 방법 찾을 때까지 반복, 30분 보고.
> CLAUDE.md 본 목표(CPU/하드웨어 활용률 극대화로 GPU 서버 전체 가속)와 직결.
> 표적 best 구성: NVFP4 awqgptq 70B + suffix spec + FaP + pad, TP8 (~4700 tps, GPU util 91-100%).

## 가용 하드웨어 인벤토리 (이 B200 호스트, 2026-06-17)
| HW | 상태 | 비고 |
|---|---|---|
| CPU Xeon Platinum 8570 + **AMX** (amx_bf16/int8/tile), AVX-512 | ✅ | **이 호스트에 AMX 있음** (prod급, dev머신과 다름) |
| **DSA ×2** (/dev/dsa wq 구성) | ✅ | 데이터 스트리밍 (memcpy 오프로드, FERRY) |
| DRAM 2TB (NUMA 2×1TB), NVMe ×10 | ✅ | 대용량 KV/prefix 캐시 후보 |
| IAA | ❌ | PCI 부재 (SUB_246) |
| NVSwitch fabric + NCCL 2.28.9 (NVLS 가능) | ✅ | 단 vLLM은 trtllm custom AR 사용 중 |

## 핵심 제약 (전 세션 누적)
GPU compute-bound(util 91-100%) → **다른 하드웨어를 "연산"으로 쓰는 건 GPU가 너무 빨라 불가**.
남은 길 = 연산 아닌 것(데이터이동·통신·용량)으로 GPU를 *덜 기다리게* 하거나 *덜 일하게*.

## 반복 기록
| iter | 다른 HW 레버 | 결과 |
|---|---|---|
| **1** | **CPU-AMX draft 모델** (spec draft를 CPU AMX로) | ❌ **死** — 1B=28.6ms/tok(40.8×), 3B=67.9ms(97×) vs GPU step 0.7ms. CPU가 토큰당 40-97× 느려 draft=critical-path. CPU-연산 길 확정 사망. |
| **2** | **NVLS/SHARP** (NVSwitch in-network all-reduce, reduction을 GPU SM→스위치 오프로드) | 🔵 A/B중 (예측: 무이득 — decode AR은 작은 메시지라 custom one-shot 우위, NVLS는 큰메시지용 + (c)서 comm 비한계 확인) |
| **2 결과** | NVLS A/B | ❌ nccl_nvls BOOT_FAIL(NCCL NVLS+custom-AR-off 비호환), custom_ar 4543(baseline). NVLS 적용불가+무이득, 종결. |
| **3 ⭐** | **사용자 축: CPU/DRAM 미리계산 KV → GPUDirect 주입** | ✅ **feasibility 양성** — 아래 |
| 4 (대기) | DSA KV tiering / 비-matmul GPU op 오프로드 | KV 압력 無·op 작음 → EV 낮음 |

## iter3 ⭐ — GPUDirect-KV 주입 (사용자 통찰: CPU 미리계산 + GPUDirect 주입)
**가용 HW 확인**: GPUDirect Storage(`/dev/nvidia-fs0~15`, nvidia_fs)·nvidia_peermem·mlx5 RDMA·
GPU P2P 전체 OK·FERRY(pinned+DSA, SUB_239) 기존. **모두 갖춰짐.**
**feasibility probe** (`exp/probe_gpudirect_kv.py`, 70B KV=L80×2×KVH8×HD128):
| | KV 크기 | 주입(56GB/s pinned) | vs prefill 재연산 |
|---|---:|---:|---|
| P2000 fp16 | 655MB | 11.8ms | prefill ~100-400ms → **10-40× 저렴** |
| P2000 fp8 | 328MB | 5.9ms | → **17-68× 저렴** |
| P8000 fp8 | 1.3GB | 23.5ms | 긴 프리픽스일수록 이득↑ |
→ **양성**: 캐시 prefix의 KV를 GPUDirect 주입 = prefill 90-97% 회피. CPU matmul 아닌 데이터이동이라
iter1 함정(40-97× 느림) 우회. **사용자 통찰이 정확.**
**단서**: (1)공유-프리픽스 워크로드 한정(유니크 프롬프트 무효) (2)신규성=vLLM 기존 prefix-cache/
CPU-offload(LMCache/NIXL) 대비 GPUDirect-Storage(NVMe 35TB tier)·DSA 경로가 신규인지 점검 필요
(3)56GB/s는 보수치, 실 peer-DMA/GDS는 더 높음.
**다음**: (a)vLLM 기존 KV-offload/prefix 경로 점검(신규성) (b)공유-프리픽스 워크로드 실측 throughput.

## NEO 차별점 (사용자 요구) + 전체 계획 (사용자: 모든 축 사용, 반드시 win)
| | NEO (기각) | 이 축 (정적 prefix KV 재사용) |
|---|---|---|
| 대상 | 활성 요청 KV | 여러 요청 공유 정적 prefix KV |
| 목적 | 용량(swap) | 연산절감(prefill skip) |
| 70B서 | KV 압력 無→무효 | 압력 무관, 연산 직접 절감 |
**차별 핵심**: NEO=활성KV를 용량 위해 swap(무효), 이 축=공유 prefix KV 재사용해 prefill 연산 건너뜀(실이득).

### iter3 실측 결과 ✅ WIN (`sweep_prefixcache.sh`, `prefixcache_results.csv`)
공유-프리픽스 워크로드(공유 prefix 3500토큰+유니크 suffix, 192req, best TP8):
| | wall | throughput |
|---|---:|---:|
| APC OFF | 11.24s | 17.1 req/s |
| **APC ON** | **1.96s** | **98.2 req/s (+475%)** |
**측정된 +475% throughput.** 핵심 재프레이밍: "GPU-bound 아무것도안통함"(R4~R7)은 유니크-프롬프트
(최악·비현실) 벤치 산물. 현실 서빙(공유 시스템프롬프트/RAG/few-shot)은 KV재사용 헤드룸 큼.
NEO 차별 확정. 단 메커니즘=vanilla APC(신규성無). DRAM tier는 이 B200(GPU mem 1.46TB) 무의미.

**계획 (모든 축)**: (1)APC 재사용 win 실증[✅ +475%] → (2)DRAM 2TB tier
확장(offloading 커넥터, GPU캐시 초과분 보관→적중률↑=vanilla-APC 상회 "다른HW" 기여) → (3)GDS NVMe
35TB tier 직접주입(신규 sliver). vLLM 기존(LMCache/NIXL/offloading)은 DRAM tier까지 커버=신규성은
GDS-for-KV·DSA경로에 한정이나, **사용자 목표=측정 win**이므로 effective deploy도 산출.
산출물: `exp/probe_gpudirect_kv.py`, `exp/bench_sharedprefix.py`, `exp/sweep_prefixcache.sh`.

## iter1 산출물
`exp/probe_amx_draft_latency.py` — CPU bf16(AMX) 1B/3B decode 지연 측정.
