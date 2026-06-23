# SUB_259 — Max-Config Baseline (모든 성능 플래그 ON) — 매크로 플랜 Phase A

*측정 2026-06-17, 8×B200 EP8. 이후 모든 신규 알고리즘 향상의 단일 기준선.*

## Config (serve_maxcfg.sh, tier=max)
DeepSeek-R1 671B (256 exp top-8, block-FP8), EP8:
- CLI: `--tensor-parallel-size 8 --enable-expert-parallel --enable-chunked-prefill --enable-prefix-caching
  --max-num-batched-tokens 32768 --max-num-seqs 512 --max-model-len 8192 --gpu-memory-utilization 0.90
  --optimization-level 3 --enable-flashinfer-autotune --performance-mode throughput`
- ENV: `VLLM_USE_DEEP_GEMM=1 VLLM_ALLREDUCE_USE_SYMM_MEM=1 VLLM_ALLREDUCE_USE_FLASHINFER=1
  VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE=1`
- 활성 최적화(부팅 로그 확인): performance mode=throughput, **async scheduling ON**, custom fusions
  **norm_quant + act_quant + allreduce_rms**, O3 (FULL_AND_PIECEWISE cudagraph), flashinfer autotune.
- MoE 백엔드 = auto(deep_gemm). **주의**: `VLLM_USE_FLASHINFER_MOE_FP8=1`은 R1 block-FP8(128×128)
  미지원으로 BOOT_FAIL → 강제 금지(deep_gemm이 block-FP8 처리). 부팅 ~225s, 167GB/GPU.

## Throughput (plain decode, max_tokens=120, gen_tps aggregate)
| concurrency | out_tps | req/s | per-req tok/s |
|---:|---:|---:|---:|
| 1 | 51.9 | 0.43 | 51.9 |
| 4 | 198.4 | 1.65 | 49.6 |
| 16 | 699.5 | 5.83 | 43.7 |
| 64 | **2,365.4** | 19.71 | 37.0 |
- SUB_257 baseline(1,725 @ conc32, 비-max config) 대비 max-config가 상향. c64에서 2,365 tps.
- per-req tok/s가 conc↑서 완만히 감소(43.7→37.0) = decode가 batch에서 잘 묶임(통신/메모리 공유).

## 잔여 병목 (nsys 분해 — SUB_257 트레이스 기준, max-config 재측정은 B0서 통신 직접)
SUB_257 `runs/r1cg_prof.nsys-rep`(R1 EP8 cudagraph decode) 분해:
| 커널 카테고리 | GPU 시간% |
|---|---:|
| **all-reduce + all-to-all (cross_device_reduce / dispatch·combine)** | **~31%** |
| expert GEMM (FP8 bmm) | ~20% |
| MoE activation/finalize/routing | ~11% |
| FP8 양자화 (per_token_group_quant) | ~13% |
| RMSNorm+quant | ~5% |
| attention (MLA flash) | ~2% |
→ **지배적 잔여 병목 = EP 통신 ~31%**. K-EXAONE-236B 동일 구조(일반화). 이게 신규 알고리즘 타깃.

## 판정
max-config로 모든 표준 성능 레버를 켠 상태에서도 대형 MoE는 **통신-bound(~31%)**. 통신을 줄이는
유일 미답 알고리즘 = 분산-EP 축소-expert self-spec(draft가 FLOPs+통신 동시 절감). → Phase B.

## 다음 (Phase B0 kill-gate)
B0에서 본 max-config 위에 top-k override hook(forward_context+base_router)을 얹고 동일 config로:
(1) top-8 vs top-1/2 DeepEP dispatch/combine 통신 nsys 직접 측정(이게 위 31%의 max-config 재측정도 겸함),
(2) top-1/2 강제 acceptance(a) 측정. 둘 다 PASS면 통신-신규 진행.

## 산출물
- `serve_maxcfg.sh`(core|max tier), `runs/tput_r1_max.jsonl`, `runs/serve_r1_max.log`.
- bench: `../SUB_258_dram_kv_offload/bench_struct.py`(plain 경로 재사용).
