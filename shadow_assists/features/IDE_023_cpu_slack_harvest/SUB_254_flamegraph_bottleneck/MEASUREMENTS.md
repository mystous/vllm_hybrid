# SUB_254 — best config TP8 flamegraph 병목 분석 (2026-06-17)

> 구성: NVFP4 awqgptq 70B(SR-004) + suffix spec K6 + FaP(FULL_AND_PIECEWISE) + uniform pad,
> TP8. py-spy(sudo, ptrace_scope=1) 워커 TP0 메인스레드 20초·1999샘플. (= SR-001 레버 × SR-003/004 양자화 스택)

## 콜트리 (TOTAL-time)
worker_busy_loop 98.6% → execute_model 78% → **model forward 64.3%** (cudagraph replay+FP4 GEMM).
→ 워커 시간 ~64%가 GPU forward(환원불가), 나머지 ~34%가 비-forward(spec draft·샘플·통신·디스패치).

## SELF-time 병목 순위
| 비중 | 지점 | 분류 |
|---:|---|---|
| **~12.5%** | arctic suffix draft: `add_active_response`(4.2+3.6=7.8%)+`speculate`(1.6+1.5)+`start_request`(1.6) | **spec CPU 유지비** |
| ~10% | `torch/_ops.py __call__`(6.8+2.8) | op 디스패치 |
| ~9% | flashinfer GEMM `forward`(7.5)+`mm_fp4`(1.4) | FP4 matmul launch/대기 |
| ~6% | `symm_mem all_reduce`(3.7+1.5+1.0) | **TP8 통신** |
| ~2% | rejection_sampler.parse_output(1.2)+copy_to_gpu(1.0) | spec verify·H2D |

## 핵심 발견
1. **GPU forward 64% = 천장 대부분** (FP4 GEMM, cudagraph로 launch 이미 최소화).
2. **최대 비-GPU 병목 = arctic suffix tree 유지(`add_active_response` 7.8% self)** — 매 스텝
   샘플토큰 suffix tree 삽입. spec draft 단일 최대 CPU 핫스팟 (cache.py:209/215).
3. **TP8 all_reduce ~6%** — NVFP4 70B=40GB는 TP8 과분할 → 통신/연산비 ↑. TP4가 통신비 측 유리할 수.
4. **torch op 디스패치 ~10%** — FaP가 forward는 graph화했으나 spec/샘플 경로 op은 eager 잔존.

## (a)+(b) 직접 검증 — 플레임그래프 가설 2개 모두 기각 (`sweep_ab.sh`, `runs/ab_results.csv`)
### (a) TP4 vs TP8 (best 구성, 동일)
| 구성 | best_tps | GPU util |
|---|---:|---:|
| **TP8** | **4726** | **100%** |
| TP4 | 4182 | 84% |
→ **TP8 +13% 빠름. "40GB TP8 과분할 통신비 손해" 가설 기각.** TP8이 GPU 100% 포화(병렬연산이
allreduce 6% 압도), TP4는 util 84%로 미포화(bubble). **best = TP8.**

### (b) suffix draft critical-path (TP4, propose 지연 주입)
| 지연 | tps |
|---|---:|
| 0µs | 4182 |
| 300µs | 4144 |
| 800µs | 4169 |
→ **평탄(800µs에도 0.3%↓=노이즈). suffix draft는 GPU forward에 완전 오버랩 = critical-path 아님.**
SUB_225 천장 직접 재확인. `add_active_response` 최적화해도 tps 무변.
(계측: `suffix_decoding.py propose()` 에 env-gated `VLLM_SUFFIX_PROBE_DELAY_US` 추가, 기본 no-op.)

## (c) TP8 all_reduce(6%) 감소 레버 A/B (`sweep_comm.sh`, `runs/comm_results.csv`)
부팅 config 확인: 현행이 이미 **flashinfer trtllm AR + RMSNorm 융합**(`fuse_allreduce_rms:True`,
auto-selected trtllm) 사용 = 최상위 AR 경로. 꺼져있던 레버 A/B:
| 구성 | best tps | vs base | util |
|---|---:|---:|---:|
| baseline (trtllm AR+rms 융합) | 4655 | — | 91% |
| +sequence_parallel | 4722 | +1.4% | 91% |
| +async-TP(fuse_gemm_comms) | 4682 | +0.6% | 99% |
| +both | 4689 | +0.7% | 96% |
→ **전부 노이즈 범위(±4% r1/r2 편차)내 = 무의미.** 통신은 vLLM 기본 융합으로 이미 거의 최적,
SP/async-TP 추가 이득 없음. all_reduce 6%는 NVLink5+융합으로 환원 불가, util 91-99% 포화라 숨길 여유 無.

## 종합 결론
플레임그래프 워커-스레드 점유율(suffix 12.5%·TP8 comm 6%)은 **둘 다 비-병목** 확정: suffix=오버랩(b),
TP8 comm=그래도 TP8 우위(a). **유일 실제 병목 = GPU forward 64%(=비트폭, upstream)** —
R4~R7 임포서빌리티 결론과 일치. 교훈: **플레임그래프 점유율 ≠ critical-path** (주입 검정 필수).
산출물: `runs/flame_best.svg/.speedscope.json`, `profile_best.sh`, `sweep_ab.sh`, `runs/ab_results.csv`.
