# DSA-KV kill-gate (2026-06-18) — 🔴 대부분 NO-GO (구조적)

*Intel DSA로 KV-cache offload를 가속하는 novel 갭 검증. [[dram-kv-offload-harvest-win]]가 플래그한 "DSA 전송" 신규여지. raw movdir64b microbench + vLLM 코드 검사. GPU 서빙 미점유.*

## 측정 1 — DSA 구동 가능성 ✅
- wq0.1 = **dedicated WQ, type=user, world-rw**(`crw-rw-rw-`). DML 라이브러리 없이 **raw 64B 디스크립터 + movdir64b**로 구동 성공.
- 단, mmap 포털에 **CAP_SYS_RAWIO(root) 필요**(커널 user_submission_safe 게이트). sudo 무암호 가능해서 진행. → 프로덕션 통합 시 권한 이슈.
- memmove 정확성 검증 OK. BOF(block-on-fault) 플래그는 wq 미지원→제거(페이지 사전 폴트인).

## 측정 2 — 대역폭 (`dsa_copy.c`, `dsa_async.c`)
| 전송크기 | DSA 단일 | CPU memcpy(cold, 1코어) | DSA/1core |
|---:|---:|---:|---:|
| 16 KB | 10.1 GB/s | (캐시히트 시 빠름) | 작은전송 死(제출지연 ~1.6us) |
| 256 KB | 27~31 GB/s | 8~11 GB/s | **~3.0×** |
| 2 MB | 31 GB/s | 11 GB/s | **~2.9×** |
- DSA 집계 대역폭 **~31 GB/s 천장**(단일 wq0.1, depth 1→16 무관 — 멀티엔진 집계 안 됨, WQ 1개 한계).
- cold(캐시우회) 실데이터 기준 **DSA가 CPU 1코어의 ~3×**, 그리고 **CPU를 비움**(제출만). 큰 cold 전송에서만 유효(작은전송은 제출지연에 死).

## 측정 3 (★결정적) — vLLM KV offload 경로엔 DSA 자리가 없음
`vllm/v1/simple_kv_offload/{copy_backend.py,cuda_mem_ops.py}`:
- offload = **GPU 캐시 ↔ pinned CPU 캐시를 `cuMemcpyBatchAsync`(CUDA batch memcpy, PCIe DMA)로 직접 복사**.
- **CPU-side DRAM↔DRAM 복사 단계 없음**(pinned 목적지로 GPU DMA 직행, repack 無).
- **DSA는 DRAM↔DRAM만 가능, GPU↔host PCIe 전송 불가** → 지배 경로(GPU↔host)에 끼어들 수 없음.

## kill-gate 판정 — 🔴 NO-GO (지배 경로), 🟡 narrow niche만
- 지배 KV-offload 비용 = GPU↔host PCIe(cudaMemcpyBatchAsync). **DSA 구조적으로 불가**. + CPU-side 복사 단계가 없어 offload할 대상도 없음.
- DSA가 살 수 있는 유일 niche = **DRAM-tier 내 이동**(pinned↔pageable, NUMA 마이그레이션, CXL/far-memory tiering — `kv_dram_tiering.py`). 좁고 2차적, ~31 GB/s 천장, 임팩트 약함.
- **결론**: "DSA로 KV offload 가속" 신규주장 불성립(지배 경로 구조적 불가). DSA 자체는 동작·cold 3×/core지만 **적용처가 KV 핫패스에 없음**.

## 종합 (정직)
- reduced-expert self-spec: NO-GO(B0, allgather routing-불변). DSA-KV: NO-GO(PCIe 경로, DSA는 DRAM↔DRAM).
- **남은 두 novel 갭이 둘 다 구조적으로 막힘**(NVSHMEM 부재·PCIe 경로). 이 머신/스택의 novel-win 탐색공간 사실상 소진.
- 산출물: `newidea/microbench/{dsa_copy.c,dsa_async.c}`.
