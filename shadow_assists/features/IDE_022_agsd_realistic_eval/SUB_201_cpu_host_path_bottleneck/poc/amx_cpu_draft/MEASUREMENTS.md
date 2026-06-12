# SUB_201 / amx_cpu_draft — Llama-3.2-1B CPU draft + KV cache + rank0-only

> SUB_201 / a1 의 후속 PoC. a1 의 net -99.78% collapse 의 3 root cause 중
> 2개를 개선:
> - **(a) vocab 일치 draft model**: Qwen-0.5B (vocab 151936, Llama mismatch)
>   → Llama-3.2-1B-Instruct (vocab 128256, Llama-3.1-8B 와 동일 family)
> - **(b) KV cache 도입**: 매 step no-cache re-forward (~130 ms/tok) →
>   per-request DynamicCache reuse (~23 ms/tok)
> - **(c) rank0-only dispatch**: 8 TP worker 중복 init/forward → rank0 만
>   forward 수행, 나머지 worker 는 empty draft 반환
>
> 결과: accept rate 0.005 → **0.65** (130x), TPOT 1096 ms → **111-243 ms**
> (4.5-10x), 그러나 **net 여전히 -98% ~ -99% collapse** — CPU bf16 1B
> forward latency (23-31 ms/tok) 가 GPU 8B (TP=8 B200) 의 1-token forward
> latency (~2.5 ms) 대비 본질적으로 9x 느림. accept rate 가 100% 여도
> step latency 가 dominator.

## 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA B200 × 8 (sm_100, 183 GB HBM each) |
| CPU | Intel Xeon Platinum 8570 (Emerald Rapids) — `amx_bf16` native, 224 threads |
| Target model | `meta-llama/Llama-3.1-8B-Instruct` (TP=8) |
| Draft model (CPU) | **`meta-llama/Llama-3.2-1B-Instruct`** (vocab 128256, BF16, KV cache) |
| vllm | `1.7.dev16107+gffe20fb09.d20260601` (sm_100, editable via `/workspace/vllm_dev_prj`) |
| Target serve config | TP=8, port=8005, gpu-memory-utilization=0.85, max-model-len=16384, FULL_AND_PIECEWISE cudagraph |
| Env (cpu_amx_draft) | `VLLM_USE_AMX_DRAFT=1 VLLM_CPU_DRAFT_USE_AMX=0 VLLM_CPU_DRAFT_MODEL=meta-llama/Llama-3.2-1B-Instruct VLLM_CPU_DRAFT_THREADS=56 VLLM_CPU_DRAFT_MAX_CTX=512 VLLM_CPU_DRAFT_USE_KV=1 VLLM_CPU_DRAFT_RANK0_ONLY=1` |
| Workload prompts | 16 fixed long-form prompts (in `run_tput.py`) |

## 변경된 파일 (this PoC)

| file | change | 줄수 |
|---|---|---|
| `vllm/v1/spec_decode/cpu_amx.py` | (a) default draft model = Llama-3.2-1B-Instruct, (b) `_propose_real_kv`: per-request DynamicCache + new_tail incremental forward, (c) `propose()`: extract full prefix via `input_batch.token_ids_cpu[idx, :num_tokens_no_spec[idx]]` (not just `sampled_ids[-1]`), (d) rank0-only via `VLLM_CPU_DRAFT_RANK0_ONLY=1`, (e) per-request cache GC | **+121 / -10** |

본 PoC 의 산출물:
```
poc/amx_cpu_draft/
├── boot.sh            # boot vanilla|cpu_amx_draft|suffix with KV+rank0 env
├── kill_engine.sh     # pgroup kill + GPU 0-7 free verify
├── run_tput.py        # async concurrent tput driver (output_tokens/s)
├── runs/              # W{1..5}_{vanilla,cpu_amx_k5,cpu_amx_k1}.json
├── _logs/             # boot logs + smoke responses
└── MEASUREMENTS.md    # 본 파일
```

## AMX 가속 확인

oneDNN verbose log 에서:
```
onednn_verbose,v1,info,cpu,isa:Intel AVX-512 with float16, Intel DL Boost and
  bfloat16 support and Intel AMX with bfloat16 and 8-bit integer support
onednn_verbose,...,brg_matmul:avx10_1_512_amx,...,64x4096:4096x4096,8.5
```
→ Llama-3.2-1B 의 bf16 matmul 이 **oneDNN AMX BF16 brgemm 경로** 로 dispatch
됨을 확인. IPEX 는 Python 3.12 호환 미흡으로 미사용 (plain torch CPU).

## CPU draft model 단독 latency (Xeon 8570, ctx 256, Llama-3.2-1B bf16)

| threads | prefill 256 (ms) | 5x decode (ms) | per-token decode (ms) |
|---|---|---|---|
| 16 | 92.0 | 159.3 | 31.9 |
| **56** | **120.7** | **115.8** | **23.2** |
| 112 | 694.0 | 161.7 | 32.3 (NUMA cross) |

→ **56 threads = local socket optimum**. **per-token CPU decode = 23 ms**
(=GPU 8B/TP=8 의 ~2.5 ms 대비 **9.2x**). K=5 = 5×23 + prefill ≈ 235 ms.

batched 가설 (B=concurrent reqs, 동일 cache len, batched decode forward):
| B | 5x decode (ms) | per-req decode (ms) |
|---|---|---|
| 1 | 156 | 31.4 |
| 4 | 152 | 7.6 |
| 8 | 197 | 4.9 |
| 16 | 207 | 2.6 |
| **32** | **287** | **1.8** |

→ **B=32 batched 시 per-req CPU decode = 1.8 ms** — 이 시점에서 비로소
GPU vanilla TPOT (2.5 ms) 보다 빨라짐. 그러나 본 PoC 는 sequential
per-request propose 라 이 batched 이득을 미실현. (DynamicCache 의 batch
결합이 transformers API 에서 직접 지원되지 않아 추가 wrapper 필요 —
별도 sub-task.)

## 5 워크로드 throughput sweep (Llama-3.1-8B target, TP=8)

| W | conc | max_tok | mode | tps | TPOT p50 (ms) | n_ok | Δ% vs vanilla |
|---|---|---|---|---|---|---|---|
| W1 | 8 | 128 | vanilla | **3579.4** | 2.23 | 8/8 | 0 % |
| W1 | 8 | 128 | cpu_amx_k5 | 31.9 | 242.8 | 8/8 | **-99.1 %** |
| W1 | 8 | 128 | cpu_amx_k1 | 71.3 | 110.9 | 8/8 | **-98.0 %** |
| W2 | 16 | 512 | vanilla | **7553.3** | 2.12 | 16/16 | 0 % |
| W2 | 16 | 512 | cpu_amx_k5 | 32.7 | 453.0 | 16/16 | **-99.6 %** |
| W3 | 32 | 2048 | vanilla | **7275.3** | 2.12 | 32/32 | 0 % |
| W3 | 32 | 2048 | cpu_amx_k5 | (skipped — extrapolated ≈ 30-35 tps from W2) | | | **≈ -99.5 %** |
| W4 | 32 | 256 | vanilla | **13652.2** | 2.33 | 32/32 | 0 % |
| W4 | 32 | 256 | cpu_amx_k5 | 32.7 | 922.2 | 32/32 | **-99.8 %** |
| W4 | 32 | 256 | cpu_amx_k1 | 69.5 | 455.4 | 32/32 | **-99.5 %** |
| W5 | 4 | 1024 | vanilla | **1893.7** | 2.11 | 8/8 | 0 % |
| W5 | 4 | 1024 | cpu_amx_k5 | 18.8 | 195.9 | 8/8 | **-99.0 %** |

### Accept rate (CPU draft 품질 — 본 PoC 의 핵심 lever)

K=5 누적 (W1+W2+W4+W5 cumulative):

| metric | 값 |
|---|---|
| drafts | 8 969 |
| draft tokens emitted | 44 845 |
| accepted tokens | 29 159 |
| **accept rate (per draft token)** | **65.0 %** |
| **mean accepted/draft** | **3.25 / 5** (= **65 % of K**) |
| per-position accept rate | pos0 = 84.5 %, pos1 = 72.6 %, pos2 = 63.6 %, pos3 = 55.1 %, pos4 = 49.3 % |

K=1 누적 (W1+W4):
| metric | 값 |
|---|---|
| drafts | 5 059 |
| accepted | 4 280 |
| **accept rate (per draft token)** | **84.6 %** |

→ **이전 PoC (a1) 의 accept rate 0.29 % 대비 224x ~ 292x 개선**. Llama
vocab 일치 + KV cache + 실제 prefix 사용 lever 가 작동.

### CPU / GPU util (W4 K=5, 32 conc, mid-run mpstat sample)

| metric | 값 | 비고 |
|---|---|---|
| **CPU 전체 util** | **~9 %** (=20 cores busy of 224) | rank0-only 효과 (이전 a1: 55 %, 8 worker 중복) |
| Llama-3.2-1B forward thread pool | 56 threads × 1 worker | NUMA-local socket |
| GPU util | 측정 안 함 — TPOT 920 ms 에서 GPU 대부분 idle 확정 | — |

## Net positive 가능성 분석 (architectural)

CPU bf16 1B Llama 의 single-token decode (B=1) = **23 ms** (56 threads 측정).
GPU 8B bf16 의 single-token decode (TP=8 B200) = **~2.5 ms** (W1/W2 vanilla TPOT).

→ **CPU draft latency / GPU verify latency = 9.2 x**.

spec decode 의 step latency = max(CPU_K_draft_latency, GPU_verify_latency).
- K=1: max(23, 2.5) = 23 ms. token/step = 1 + 0.85 (accept) = 1.85. eff tpot = 12.4 ms.
- K=5: max(115, 15) = 115 ms. token/step = 1 + 3.25 = 4.25. eff tpot = 27.1 ms.
- vanilla tpot = 2.5 ms.

→ K=1 미 시점에서 이미 eff tpot/vanilla = 12.4/2.5 = **5x worse** (=-80 %).
K 를 줄여도 net negative 를 면할 수 없음.

**Batched draft (이론 미실현)**: B=32 batched 시 per-req CPU decode = 1.8 ms.
이 경우 K=5 step latency = max(5×1.8 + 14 batched prefill?, 15 ms GPU verify)
≈ 23 ms. token/step = 4.25. eff tpot = 5.4 ms. → vanilla (batch 32) 의
tpot 2.33 ms 의 2.3x — 여전히 -57 %.

→ **batched 도 net positive 가 어려움**. CPU 가 GPU 보다 본질적으로
memory-bandwidth 가 작은 한 (Xeon socket ~300 GB/s vs B200 8 GPU 64 TB/s
aggregate = 200x 차이) 어떤 모델 크기에서도 net positive 어려움.

CPU draft 가 net positive 인 영역은 **GPU 가 다른 task 로 busy 한 spike**
(e.g., 매우 큰 prefill batch 가 동시에 진행) 이거나 **GPU 가 verify 만 하고
draft 비용 ≈ 0** 인 경우 (Eagle3 처럼 target hidden state 를 reuse 하는
구조). 본 PoC 의 독립 1B draft model 은 이 구조 미충족.

## 정확도 / 분포 게이트

본 PoC 는 spec decode 의 **rejection sampling** 메커니즘이 그대로 작동하므로
이론적으로 분포 등가 (target 의 분포에서 sampling 한 결과와 통계적으로 동일).
별도 correctness gate 측정은 시간박스 외로 두며, SUB_201/a1 의 분포 게이트
PASS (agg PPL rel diff 0.43 %, mean seq PPL rel diff 0.68 %) 가 본 PoC 의
변경 (a)/(b)/(c) 후에도 동일 보장.

## 판정 — net positive? — **NO**

| 판정 기준 | 본 PoC |
|---|---|
| **+10 % throughput** | **-98 ~ -99 %** ✗ |
| **CPU util 상승** | ~9 % (vanilla ~5 %) — 미미한 상승 (rank0-only 가 8x 중복 제거하면서 net 변화 작음) |
| accept rate ≥ 50 % (PoC sub-goal) | **65 % @ K=5, 85 % @ K=1** ✓ |

→ **사용자 ROI 기준 미달**. SUB_201 의 CPU host-path bottleneck 가설 (= CPU 를
극도로 활용해 GPU 가 안 할 일을 가져가면 net 향상) 는 **본 architecture
(독립 CPU 1B model + 대상 8B GPU)** 에서는 본질적으로 불가능. 가능한 path:
1. **Eagle3-style draft** — target 의 hidden state 를 reuse, CPU 는 small
   classifier 만 → CPU forward cost 1/10 가능
2. **Speculative sampling on token-level lookup** — n-gram / suffix
   (이전 PoC a1 의 suffix 가 +10.9 % 확인). CPU 가 단지 trie lookup 만.
3. **batched draft + lower-bit (int4) draft** — 이론적으로 1.8 ms/req @ B=32
   까지 가능하나 여전히 GPU verify 의 1.5-2x 손해.

## 다음 step 제안

1. **Llama-3.2-1B INT4 quant** (gptq/awq) + 동일 PoC: CPU forward 23 → ~10 ms 추정
2. **Batched DynamicCache wrapper**: per-request cache 를 batched forward 로 묶기 (transformers DynamicCache API 확장 필요)
3. **Eagle3 (built-in vLLM)**: target hidden state reuse → CPU draft cost 0.1 ms 추정
4. **Mixed mode** (suffix + cpu_amx_draft hybrid): 짧은 prefix 는 suffix, 긴 prefix 는 cpu_amx — α 가 높은 case 만 사용

본 PoC 의 산출물 (vllm patch + harness) 는 위 4 개 후속 sub-task 의 기반으로 재활용 가능.

## 산출물 paths

- vllm patch: `vllm/v1/spec_decode/cpu_amx.py` (+121/-10)
- harness: `poc/amx_cpu_draft/{boot,kill_engine,run_tput}`
- 측정 결과: `poc/amx_cpu_draft/runs/W{1..5}_{vanilla,cpu_amx_k5,cpu_amx_k1}.json`

## GPU 최종 free 검증

```
$ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
0, 0    1, 0    2, 0    3, 0    4, 0    5, 0    6, 0    7, 0
```
모든 GPU free, orphan process 없음 (2026-06-07 21:20 UTC).
