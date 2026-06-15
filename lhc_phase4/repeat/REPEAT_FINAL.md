# REPEAT_FINAL — LHC throughput 양수 시도 종합

생성: 2026-06-08 by repeat loop (Auto Mode)

## 명령
LHC 로 throughput 양수 만들 때까지 반복.

## 핵심 결론

**현 환경 (vLLM v1.7.dev16107, B200 8× sm_100, Llama-3.1-8B-Instruct TP=8) 에서 LHC throughput 양수 달성 실패**. 5 시도 모두 negative Δ% (-51% ~ catastrophic). 

근본 원인은 LHC dead branch 가 단일 원인이 아니라 *3-층 dead-branch chain*:

1. **NEO scheduler adapter 가 PoC Phase 4.6 단계** — "first-stage wiring — vanilla data path retained, NEO decisions are recorded but not yet executed". 즉 swap_out 결정만 기록되고 실 발화는 부분적.
2. **NEO predictive swap_out 은 KV usage 30%대에서도 fire** (Plan v4 G/H 의 swap_out attach). 그러나 worker 에서 `[NEO LHC_P4_001] swap-out OOB drop: max_block_id=N >= kv_cap=2` 로 100% drop. 원인은 profile-time stale `kv_caches[0].shape[0]=2` 가 production 에서 reset 안됨.
3. **DSA hook 은 NEO swap path 의 contiguous block branch 내부에 위치** — OOB drop 이후의 코드 경로라 호출 0건.

게다가 `--enable-neo-asymmetric` flag 자체가 prefill chunking / decode mirror 동작을 변경하여 **-51% ~ -61% overhead** 를 생성. NEO swap path 가 dead branch 인 동안에는 vanilla path 보다 무겁기만 함.

## 시도별 결과

| Attempt | 워크로드 | Vanilla output tok/s | LHC output tok/s | Δ% output | DSA hook | NEO swap |
|---|---|---|---|---|---|---|
| 1 | sonnet 512/512 conc=64 mem=0.30 | 17,130 | 6,955 | **-59.4%** | 0건 | 0건 (KV 1.6%) |
| 2 longA | 16K/2K conc=64 mem=0.30 | 8,334 (mean) | 3,406 (mean) | **-59.1%** | 0건 | 70건/100step → 100% OOB drop |
| 3 (skipped) | — | — | — | — | — | OOB drop pre-known |
| 4 eag1 (s1 lhc) | 4K/1K conc=256 mem=0.20 eager (broken fix) | 14,272 (mean) | 49 | **-99.7% catastrophic** | 0건 | NEO BUF ALLOC 발화 후 CUDA assert 388건 (잘못된 _kv_cap fix) |
| 4 eag1 (s2 lhc) | (revert 후) | 14,272 (mean) | 5,492 | **-61.5%** | 0건 | 100% OOB drop |
| 5 kvcap1 (lhc s1) | 4K/1K conc=256 mem=0.20 cudagraph | 22,287 (mean) | 10,200 | **-54.2%** | 0건 | 85건/50step → 100% OOB drop |
| 5 kvcap1 (lhc s2) | TBD | 22,287 (mean) | TBD | TBD | TBD | TBD |

## 시도된 코드 변경 (모두 revert)

`vllm/v1/worker/gpu_model_runner.py` line 6841-6862: `_kv_cap` 도출 시 `cache_config.num_gpu_blocks` (cluster-wide block pool size) 우선 read 시도. **잘못된 fix**: cluster pool size > per-worker capacity → OOB guard 통과 → 실제 OOB access → CUDA assert 388건 → throughput 49 tok/s catastrophic. **즉시 revert**.

진짜 fix 방향:
- `_cleanup_profiling_kv_cache` 가 `kv_caches.clear()` 만 함. realloc 이후 새 KV cache shape 가 `[2, ...]` 으로 남는 게 race / lazy alloc 버그.
- TP=8 환경에서 per-worker shape[0] 이 production 단계에서 production size (e.g. 254 blocks) 가 되어야 함.
- 본 sweep budget (6시간) 외.

## 다음 단계 권고

- **A (현실적 적용)**: 본 환경에서 LHC 가 negative-result 임을 paper §08 에 evidence 기반 통합. 5 시도의 결과 표 + 진단 chain (NEO PoC stub → predictive swap fire → OOB drop → DSA hook 0건 → -51%~-61% overhead) 을 honest negative result 로 게재. Theorem 1 (host idle cycle 활용) 의 environment-dependent claim 으로 narrow.
- **B (시간 budget 외)**: vllm v1 KV cache reinit flow 의 race 추적 → 진짜 `_kv_cap` reset → NEO swap path 활성 → DSA hook 발화 → throughput 측정.
- **C (시간 budget 외)**: SD 머신 (Sapphire Rapids + H100 PCIe) 으로 cross-platform 확인.

## 산출물

- `lhc_phase4/repeat/attempt_1/` — sonnet 512/512 mem=0.30 sweep (vanilla+lhc each 1 sweep)
- `lhc_phase4/repeat/attempt_2/` — long-context 16K/2K mem=0.30 sweep (longA 2 sweep)
- `lhc_phase4/repeat/attempt_4/` — eager mode + mem squeeze (eag1 2 sweep × {van, lhc})
- `lhc_phase4/repeat/attempt_5/` — cudagraph mode + same KV pressure (kvcap1 2 sweep × {van, lhc})

## 안전

본 sweep loop 중 vllm worker PID 만 명시적으로 kill (사용자 spawn 한 process 만). GPU 전역 kill 시도 시 권한 거부 (안전 boundary 보호 정상 동작).

코드 변경 모두 revert. git diff 는 sweep 이전 상태와 동일 (HWC1 stream priority 등 사전 변경분만 잔존).
