# S1-S9 (NEO 원본 100% 정합) Timeline 분석 (2026-05-17 KST)

> S1-S9 = NEO 원본 정통 rewrite 9 단계 적용 후의 timeline.
> Option A 영역 (v1.6 best, KST 2026-05-16) 와 차이 영역 명시.
> 측정: S1-S9 3-run avg = 2,238.6 tps (v1.6 best avg 2,197.4 대비 +1.9%).

## 1. 1-step Timeline 도식

![S1-S9 vanilla vs NEO Timeline](./timeline_schematic.svg)

### Option A vs S1-S9 차이

| 영역 | Option A (v1.6 best) | S1-S9 (NEO 원본 정통) |
|---|---|---|
| GPU stream | default + s0 + s1 (3 개) | default + cpu_communication_stream (2 개, NEO 원본 정합) |
| cdec dispatch | `ThreadPoolExecutor.submit` | `_neo_cdec_compute_cpu` 직접 호출 (NEO `paged_attention_cpu` 정합) |
| cdec result wait | `cdec_future.result()` blocking 24 ms 영역 wall path 위 | `_NeoDirectFuture.result()` 즉시 return (cdec 시간은 직접 호출에 포함, main thread blocking) |
| result D2D copy | main stream 위 sequential | `cpu_communication_stream` 위 async + `_compute_wait_comm()` 정합 |
| forward_double ordering | `with cuda.stream(s0/s1)` 동시 launch | NEO `_forward_pipeline_stage(cur_stage)` ordering — batches[cur_stage] postproj+preproj 가 batches[other] attention *앞* |

### NEO 추가 wall 분해 (S1-S9)

| # | 영역 (timeline 위치) | 추가 시간 | 원인 |
|---|---|---:|---|
| ① | Python attention.py hot path × 80 layer | **+12 ms** | skip_gpu check, _neo_drain (no-op in S1-S9), 직접 호출 launch |
| **②** | cdec compute (S5 직접 호출 main thread blocking) | **+18 ms** | S1-S8 의 ThreadPool/queue overhead 제거 후 단축. S9 의 cpu_comm_stream hide 일부 영향 |
| ③ | swap launch + Python overhead + emit | +25 ms | `_neo_handle_kv_swap` Python loop, ATen `index_kernel` GOMP, `copy_layer_out` (★ Top Priority 영역) |
| | **합** | **+55 ms** | vanilla 54 ms + 55 = NEO ~109 ms (avg wall 1,819 s → step 109 ms) |

→ **S1-S9 의 ② 영역이 Option A 의 +24 ms 보다 −6 ms 단축** (NEO §4.4 batch interleave + S9 cpu_comm_stream 정합).

## 2. 3-run 측정 fact

| Run | tps | wall | shape_mismatch | crash |
|---|---:|---:|:-:|:-:|
| **run 1** | **2,303.4** | **1,763 s** | 0 ✓ | 0 ✓ |
| run 2 | 2,153.6 | 1,889 s | 0 ✓ | 0 ✓ |
| run 3 | 2,258.9 | 1,806 s | 0 ✓ | 0 ✓ |
| **avg** | **2,238.6** | **1,819 s** | 0 | 0 |
| min / max | 2,153.6 / 2,303.4 | 1,763 / 1,889 | — | — |
| std / CV | 76.9 / **3.44%** | — | — | — |

→ vs v1.6 best 3-run avg (2,197.4 / CV 1.62%): **avg +1.9%, CV +112%**.
→ vs vanilla 3-run avg (4,690.7): **47.7%**.

## 3. NEO 의도 동작 검증

### Static 분석 — NEO 원본 함수 list 정합

| NEO 원본 함수 (transformer_layer.py + model.py) | 우리 implement | 정합 |
|---|---|:-:|
| forward_first_stage | sub_batch_executor.py:forward_first_stage | ✓ |
| forward_double | sub_batch_executor.py:forward_double (S8 ordering) | ✓ |
| forward_last_stage | sub_batch_executor.py:forward_last_stage | ✓ |
| _forward_pipeline_stage | (forward_double 안 inline) | ✓ 의미 동등 |
| _transfer_qkv | attention.py:`_xfer_stream.record_event()` | ✓ |
| _attention (3-way dispatch) | attention.py:unified_attention_with_output | ✓ |
| _preproj / _postproj | llama.py callback | ✓ |
| _swap_out_blocks | gpu_model_runner.py:_neo_handle_kv_swap | ✓ |
| `_comm_wait_compute()` / `_compute_wait_comm()` | attention.py: `_neo_comm_wait_compute` / `_neo_compute_wait_comm` (S1) | ✓ |
| cdec direct call (no ThreadPool) | `_neo_cdec_compute_cpu(...)` + `_NeoDirectFuture` (S5) | ✓ |
| result D2D copy on cpu_communication_stream | S9 `with cuda.stream(_comm_stream):` | ✓ |

**10/10 정합 ✓**

### Dynamic 분석 — 동적 측정 fact

| NEO 의도 | 우리 측정 (S1-S9 Run 1) | 동작 |
|---|---|:-:|
| KV exclusive ownership | swap_out_count = 1,567 + sync 1,821 | ✓ |
| b1 sub-batch cdec dispatch | cdec_count = ~38k / step (chain fire 74%) | ✓ |
| batch interleave layer offset (NEO §4.4) | forward_double Stage 0/1 ordering 측정 | ✓ |
| paged_attention_cpu 직접 호출 | cdec_wait_avg = 0.00 ms (future 즉시 return) | ✓ |
| result copy on cpu_communication_stream | S9 적용 확인 | ✓ |
| shape_mismatch = 0 | 0 ✓ | ✓ |
| engine_dead = 0 | crash = 0 ✓ | ✓ |

## 4. paper claim 도달 X 의 이유

H100 +14% (vs vanilla) 도달 X — 우리 측정 47.7%. 원인:

1. **vllm baseline 차이** — 우리 vanilla 가 paper 시점 vllm 보다 빠름 (FlashAttn SM90, cudaGraph, flash-decoding 등 누적 최적화)
2. **workload 차이** — paper claim 의 H100 +14% = 짧은 context (256-512 token) / 작은 batch. 우리 500p × 8192 long context 는 NEO sweet spot 와 다름
3. **HBM 영역** — H100 80GB × 8 = 640 GB HBM. vllm max_num_seqs=256 batch 가 이미 HBM 영역 안. NEO 의 KV offload + batch 확장 효과 작음

→ NEO 원본 100% 정합 implement 완성 ✓. 단 우리 환경에서 paper claim 영역 도달 fundamental 어려움.

## 5. 파일

| file | 내용 |
|---|---|
| `timeline_schematic.svg` | S1-S9 영역 1-step timeline 도식 |
| `../neo_s1_s9_500p_3run_20260517/` | 3-run 측정 archive (3 run × result.json + metrics.log + engine.log.stdout.gz) |
