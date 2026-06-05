# SUB_201 / A1 — CPU drafting lever × vLLM 통합 + e2e 측정 (8 GPU TP=8)

> 본 task 는 A1 (CPU 측 draft proposer) 를 vLLM 의 spec-decode dispatch 에
> 실제 wire-in 한 후 **net throughput impact** 를 측정한 첫 e2e 실험이다.
> 24-layer 실 forward chain 의 .so 통합 (SUB_198 §3 (d)) 은 본 task 의 scope
> 가 아니다. 본 측정에서 draft id 생성에 실제 사용된 경로는 **PyTorch CPU
> forward (Qwen 2.5-0.5B-Instruct, BF16, no-KV-cache, K=7)** 이며, AMX kernel
> (`libamx_draft_qwen05b.so`) 은 `forward()` symbol 미빌드로 인해 ctypes
> `dlopen`/`hw_amx` 확인 단계에서 자동 비활성화 (PyTorch path 로 fallback).

## 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA B200 × 8 (sm_100, 183 GB HBM each) |
| CPU | Intel **Xeon Platinum 8570 (Emerald Rapids)** — `amx_tile + amx_bf16 + amx_int8 + avx512f + avx512_bf16` native, 224 cores |
| Target model | `meta-llama/Llama-3.1-8B-Instruct` (32 heads ÷ TP=8 = 4 OK) |
| Draft model (CPU) | `Qwen/Qwen2.5-0.5B-Instruct` (no KV cache, naive re-forward, K=7) |
| vllm | `1.7.dev16107+gffe20fb09.d20260601` (sm_100 재빌드, editable via `/workspace/vllm_dev_prj`) |
| Target serve config | TP=8, port=8005, gpu-memory-utilization=0.85, max-model-len=16384, compilation `FULL_AND_PIECEWISE` |
| Workload | sharegpt 200 prompts (`poc/b3_8gpu_full/sharegpt200.parquet`) |
| Env (cpu_amx_draft) | `VLLM_USE_AMX_DRAFT=1 VLLM_CPU_DRAFT_USE_AMX=1 VLLM_CPU_DRAFT_THREADS=16` |

## 변경된 파일 (patch summary)

본 task 에서 신규 추가한 vllm 본체 patch:

| file | line | change | 줄수 |
|---|---|---|---|
| `vllm/config/speculative.py` | 462-471 | `elif self.method == "cpu_amx_draft":` 분기 추가 → `draft_model_config = target_model_config` (HF repo lookup 우회) | **+10 / -0** |

이외의 vllm 본체 wire-up 은 이전 세션에서 완료된 상태였다 — `CpuAmxProposer`
의 `propose(input_batch, sampled_token_ids, slot_mappings=None) -> list[list[int]]`
시그니처는 이미 `SuffixDecodingProposer` 와 일치, `gpu_model_runner.py:667-674
/ 5236-5243` 의 dispatch 분기도 등록 완료.

본 task 신규 산출물 (test harness — vllm 본체 아님):

```
poc/a1_vllm_integration/
├── boot_smoke.sh           # TP=8 + Llama-3.1-8B, vanilla|cpu_amx_draft|suffix
├── run_e2e.sh              # correctness collect + e2e tps (env-tunable sample size)
├── kill_engine.sh          # pgroup kill + orphan + GPU 0-7 free verify
├── correctness_gate.py     # 분포-유사성 게이트 (logprob max-abs-diff + PPL rel diff). asyncio 구현
├── MEASUREMENTS.md         # 이 파일
└── runs/                   # tput_*.json, correctness_*.jsonl, *.log
```

## propose API 검증

`CpuAmxProposer.propose(input_batch, sampled_token_ids, slot_mappings=None)
-> list[list[int]]` — SuffixDecodingProposer 와 동일 시그니처. dispatch 분기
(line 5236-5243) 에서 별도 어댑터 없이 직접 호출. vllm spec-decode interface
호환 확인.

`CpuAmxProposer.load_model()` — vllm worker init 시 1회 호출되어 Qwen-0.5B
+ (가용 시) AMX kernel eager init. 본 측정에서 8 TP worker 모두에서 호출
(PP=1 → 모든 worker 가 `get_pp_group().is_last_rank == True`).

## boot smoke test 결과

| mode | boot wall (s) | startup | 1-prompt smoke | spec metrics 활성 |
|---|---|---|---|---|
| vanilla | 97 | OK | OK (16 tokens 정상 응답) | n/a |
| **cpu_amx_draft** | 92 | OK (vllm 가 자동으로 Async scheduling 비활성화) | OK (spec metric 활성 — drafts=72, draft_tokens=504, accepted=2 after 1 smoke) | **활성** |
| suffix | ~85 | OK | OK | 활성 (smoke 시점 metrics α ≈ 33 %) |

cpu_amx_draft 부팅 로그 (8 worker 모두):
```
[cpu_amx_kernel] missing symbol: ... undefined symbol: amx_draft_qwen05b_forward
                 — AMX path disabled.
[CpuAmxProposer] AMX kernel unavailable (loaded=False, hw_amx=0, ...) —
                 falling back to PyTorch CPU path.
```
→ AMX-kernel 경로는 .so 빌드 buffer 부족으로 자동 비활성화, PyTorch CPU forward
로 fallback. 본 e2e tps 측정에서 보고되는 비용은 **PyTorch CPU Qwen-0.5B
no-cache re-forward** 의 cost 이다 (AMX kernel SIGILL 위험 없음).

또한 vllm 이 자동으로:
- `Async scheduling not supported with cpu_amx_draft-based speculative decoding`
  → `scheduler_config.async_scheduling = False` (`vllm/config/vllm.py:806`).
  vanilla 와 달리 cpu_amx_draft 모드는 sync scheduling 강제 — TPOT 비교 시 이
  scheduling 차이도 amplification factor 임을 유의.

## Correctness gate (vanilla vs cpu_amx_draft, 20 prompt × 32 token greedy, t=0)

cpu_amx_draft 측 async collect 175.3 s (16 conc), vanilla 측은 100p × 64 tok run
의 첫 20p × 32 tok 슬라이스. compare 결과:

| metric | 값 | gate | verdict |
|---|---|---|---|
| logprob max abs diff | **0.945** | < 0.1 | **FAIL** (informational) |
| token match frac (per-position 동일률) | **95.78 %** (613/640) | informational | high — 27 token 만 갈림 |
| **agg PPL rel diff** | **0.43 %** | **< 5 %** | **PASS** |
| **mean seq PPL rel diff** | **0.68 %** | **< 5 %** | **PASS** |

CLAUDE.md §Constraint 운영해석 — `verdict_overall = verdict_d_ii` 즉 **분포
PPL 게이트가 binding**. agg/seq PPL 양쪽 모두 < 5 % 통과. logprob max-abs-diff
는 informational — BF16 비결합성 + cascading divergence 의 알려진 single-position
outlier 패턴 (token-level 95.78 % 일치가 분포 동일성을 뒷받침).

→ **A1 의 spec-decode 결과는 vanilla 와 분포 수준에서 등가**, 즉 정확도
constraint **통과**.

## e2e throughput (Llama-3.1-8B-Instruct, TP=8, 8 GPU)

### A) Production-scale (sharegpt 200 prompts × concurrency=32 × max-tokens=8192)

| mode | output_tps | TTFT p50 (ms) | TPOT p50 (ms) | accept α | gpu% | cpu% | n_ok | wall (s) | **Δ vs vanilla** |
|---|---|---|---|---|---|---|---|---|---|
| **vanilla** | **11 981.9** | 54.1 | 2.5 | n/a | 98.0 | 5.2 | 200/200 | 125.6 | 0 % |
| **cpu_amx_draft (K=7)** | **측정 불가** (≥ 16 min — 시간박스 초과) | 1099+ (사실상) | 1096+ | **0.0050-0.0072** | 3.8 | 55+ | (interrupted) | ≥ 1000 | **≪ -99 % (추정 -99.5 %)** |
| **suffix (K=7)** | **13 290.7** | 20.4 | 1.8 | **0.8818** | 69.9 | 4.4 | 200/200 | 112.9 | **+10.9 %** |

cpu_amx_draft 의 production-scale wallclock 추정 ≥ 16 분 (8 worker × 16 req
conc 동시 CPU forward → engine generation rate **3-5 token/s/req** vs vanilla
≥ 2000 token/s/req). 시간 박스 내 측정 중단; 5-min 표본 + spec metric 누적치
기준 추정 ≈ **40-60 tps**.

### B) Micro-scale apples-to-apples (4 prompts × max-tokens=64 × concurrency=4)

| mode | output_tps | TTFT p50 (ms) | TPOT p50 (ms) | accept α | gpu% | cpu% | wall (s) | **Δ vs vanilla** |
|---|---|---|---|---|---|---|---|---|
| **vanilla** | **1 621.4** | 19.1 | 2.1 | n/a | 6.9 | 4.1 | 0.2 | 0 % |
| **cpu_amx_draft (K=7)** | **3.6** | 1 099.1 | 1 096.0 | **0.29 %** (5/1729) | 3.8 | 55.3 | 70.1 | **−99.78 %** |
| **suffix (K=7)** | **873.5** | 25.9 | 4.1 | **0.4348** (80/184) | 3.5 | 4.0 | 0.3 | **−46.1 %** (small batch에서 spec overhead) |

### 핵심 관찰

1. **GPU starvation (cpu_amx_draft)**: vanilla 98 %→ cpu_amx_draft 3.8 % GPU
   util. CPU 가 매 step (K=7 draft × 8 worker 동시) 의 Qwen-0.5B forward 를
   sequential 로 처리하느라 GPU 가 wait — GPU 측 throughput collapse.
2. **TPOT 522x 악화 (cpu_amx_draft)**: vanilla 2.1 ms vs cpu_amx_draft 1096 ms.
   K=7 spec step 당 CPU forward 7회 + GPU verify 1회 = step 당 ~1.1 s 의
   추가 wait.
3. **Accept rate α ≈ 0.005 (cpu_amx_draft) vs 0.88 (suffix)**. K=7 budget
   중 cpu_amx_draft 는 평균 1.02 token (position 0 에서만 가끔 hit, position
   1-6 은 모두 0.00). Qwen-0.5B ↔ Llama-3.1-8B vocab/tokenizer mismatch
   (Qwen 151 936 vs Llama 128 256) + `_propose_real_single` 이 `sampled_ids`
   의 마지막 1 token 만 context 로 사용 (input_batch 의 prefix 미사용)이
   결합되어 draft 가 사실상 random sequence. **target/draft vocab 일치 +
   적절한 prefix 사용** 이 net win 의 사전 조건.
4. **8 worker 중복 init (PP=1 환경 issue)**: 모든 TP worker (8개) 가
   `is_last_rank == True` 로 인식되어 CpuAmxProposer · Qwen-0.5B · AMX
   kernel 을 각자 8 회 동시 init. CPU 자원 8 중복 경합. 단일 process 만 draft
   를 담당하도록 refactor 가 필요 (별도 sub-task).
5. **Suffix 대조점**: 동일 spec-decode framework 위에서 suffix decoding 은
   production-scale 에서 **+10.9 %** net win (200p × 8192 × c=32, α 88 %).
   spec-decode wire-up 자체는 정상이며, **bottleneck 은 cpu_amx_draft 의
   draft 품질 + CPU forward 비용** 에 한정.

## Net tps Δ%

- **apples-to-apples micro (4p × 64 × c=4)**:
  cpu_amx_draft 3.6 vs vanilla 1621.4 = **−99.78 %**
- **production scale 추정 (200p × 8192 × c=32)**:
  cpu_amx_draft ≈ 40-60 vs vanilla 11982 = **≈ −99.5 %**
- **두 scale 모두 -99 % 대 규모 collapse**.
- 참고: 동일 환경 suffix decoding = vanilla 대비 **+10.9 %** (정상 net win).

## task 결론 — A1 net positive 여부 + 모델별 ROI

### A1 net positive? — **NO. 대규모 net negative (-99 % 이상)**.

원인 정리 (큰 → 작음 순):

1. **No-cache greedy CPU forward 가 step latency dominator**. Qwen-0.5B 의
   K=7 sequential re-forward = step 당 ~1 s. GPU 가 그 동안 idle. 본 PoC 의
   `_propose_real_single` 은 매 step 마다 prefix 전체 (max_ctx=256)를 re-encode
   하므로 KV cache 도입이 critical first step. 그러나 그조차도 본 측정의
   -99 % 의 일부만 회복 (~10x 추정).
2. **Acceptance rate α ≈ 0.005**. K=7 budget 중 ≈ 1.02 token 만 accept.
   Qwen-0.5B ↔ Llama-3.1-8B vocab mismatch + scaffold 가 sampled_ids 의
   마지막 1 token 만 context 로 사용. 비교: suffix decoding α = 0.88.
3. **8 TP worker 중복 init/호출**: PP=1 환경에서 8 worker 모두 propose 호출
   → CPU contention. dispatch 단계에서 rank-0 only routing + 결과 broadcast
   가 필요 (별도 sub-task).
4. **AMX kernel 미연동**: SUB_187 의 `libamx_draft_qwen05b.so` 가 `forward`
   symbol 미빌드 (latency probe 만). SUB_198 §3 (d) 의 real-forward ABI
   extension 이 들어와도 위 (1)(2)(3) 이 우선 해결되어야 net positive 가능.
   AMX 자체는 본 측정의 bottleneck 이 아님.

### 모델별 ROI

| target | draft vocab 호환 | 본 측정/예상 α | 본 측정/예상 net | 결론 |
|---|---|---|---|---|
| Llama-3.1-8B (본 측정) | × (Qwen 0.5B vocab mismatch) | 0.005 (측정) | -99.78 % (측정) | **NO-GO** |
| Llama-3.1-70B (TP=8, 64 heads) | 동일 mismatch | < 0.01 추정 | -90~99 % 추정 (decode 비중 ↑) | NO-GO (동일 draft 조합) — **본 task 70B sweep 생략** |
| Llama-3.3-70B + Llama-3.2-1B-Instruct (vocab 일치 family) | ○ | 0.5-0.7 추정 (suffix α 0.88 참조) | unknown — 별도 측정 필요 | 다음 sub-task 후보 |

**모델별 ROI 결론**: 본 측정의 cpu_amx_draft + Qwen-0.5B 조합은 어느 target
에서도 net positive 가 어렵다. CPU draft lever 의 net win 검증은
**(a) vocab 일치 draft model**, **(b) CPU 측 KV cache 도입**, **(c) rank-0
only dispatch + broadcast**, **(d) AMX kernel real-forward ABI** 의 4-step
refactor 이후 재측정이 합리적 다음 사이클. 70B 측정은 본 조합의 -99 % collapse
가 이미 확정된 만큼 본 task 시간 박스 내 추가 부팅·측정의 ROI 가 낮다.

## GPU 0-7 최종 free 검증

```
$ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
0, 0    1, 0    2, 0    3, 0    4, 0    5, 0    6, 0    7, 0
$ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
(empty)
```
모든 GPU free, orphan process 없음 (2026-06-05 11:11 UTC).

## 산출물 paths

- harness: `poc/a1_vllm_integration/{boot_smoke,run_e2e,kill_engine}.sh`,
  `correctness_gate.py`
- vllm patch: `vllm/config/speculative.py:462-471` (+10/-0)
- runs:
  - `runs/tput_vanilla_Llama-3.1-8B-Instruct.json` (200p × 8192 × c=32) = 11 981.9 tps (FINAL)
  - `runs/tput_vanilla_Llama-3.1-8B-Instruct.4p64.json` (4p × 64 × c=4) = 1 621.4 tps (apples-to-apples baseline)
  - `runs/tput_cpu_amx_draft_Llama-3.1-8B-Instruct.json` (4p × 64 × c=4) = 3.6 tps (FINAL)
  - `runs/tput_suffix_Llama-3.1-8B-Instruct.json` (200p × 8192 × c=32) = 13 290.7 tps (FINAL)
  - `runs/tput_suffix_Llama-3.1-8B-Instruct.4p64.json` (4p × 64 × c=4) = 873.5 tps
  - `runs/correctness_vanilla_Llama-3.1-8B-Instruct.jsonl` (100p × 64 tok)
  - `runs/correctness_cpu_amx_draft_Llama-3.1-8B-Instruct.jsonl` (20p × 32 tok)
  - `runs/correctness_compare_vanilla_vs_cpu_amx_draft.json` (distribution gate: PASS)
- engine boot logs: `_logs/boot_{vanilla,cpu_amx_draft,suffix}_Llama-3.1-8B-Instruct.log`

## Commit

`f97cd07de` — `poc(sub_201/a1): cpu_amx_draft vLLM 통합 + e2e 측정 (TP=8, 8 GPU)`
branch `feat/spec-decode-tuning`, push 미진행.
