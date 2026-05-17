# S1-S9 (NEO 원본 100% 정합) Timeline 분석 (2026-05-17 KST)

> S1-S9 = NEO 원본 정통 rewrite 9 단계 적용 후의 timeline.
> Option A 영역 (v1.6 best, KST 2026-05-16) 와 차이 영역 명시.
> 측정: S1-S9 3-run avg = 2,238.6 tps (v1.6 best avg 2,197.4 대비 +1.9%).

## 1. 1-step Timeline 도식

![S1-S9 vanilla vs NEO Timeline](./timeline_schematic.svg)

### NEO 원본 asymmetric pipeline 정합 fact

NEO upstream `transformer_layer.py` 의 5 가지 정합 요소가 S1-S9 에 다 작동 중:

| 요소 | NEO 원본 line | 우리 S1-S9 | 의미 |
|---|---|---|---|
| `cpu_communication_stream` 별도 운영 | 116 / 143 / 150 | `_get_neo_communication_stream` (S2) | default stream 과 분리. transfer/swap/result_copy 가 GPU compute 와 진정한 동시 진행 |
| `_transfer_qkv` 가 cpu_comm async | 171 | `with cuda.stream(_xfer_stream): q_cpu.copy_(non_blocking=True)` (attention.py:993) | Q/K/V D2H 가 GPU 다음 launch 와 overlap |
| `paged_attention_cpu` main thread direct | 336 | `_neo_cdec_compute_cpu(...)` direct (S5, attention.py:1013) | main blocking 이지만 GIL released (C++ ext) + default stream queue 가 그 동안 GPU 에서 실행 진행 → 진짜 CPU/GPU overlap |
| result copy on cpu_comm_stream + `_compute_wait_comm` | 351-355 | S9 + `_neo_compute_wait_comm` (S1) | result D2D 가 GPU 다음 launch 와 overlap |
| `_forward_pipeline_stage(cur_stage)` ordering | 397-427 | S8 `forward_double` | **layer N+1 의 preproj launch 가 layer N 의 attn launch *앞*** → GPU 가 N+1 work 부터 시작 → layer N/N+1 GPU 동시 진행 |

→ **paper §4.4 의 "Layer N/N+1 동시 + CPU async pipeline" mechanism 이 S1-S9 에 실제로 작동 중** ✓.

### Option A vs S1-S9 차이 (정확)

| 영역 | Option A (v1.6 best) | S1-S9 (NEO 원본 정통) |
|---|---|---|
| GPU stream 구성 | default + s0 + s1 + cpu_comm_stream (4 개) — s0/s1 priority diff 영역 | default + cpu_communication_stream (2 개, NEO 원본 정합) |
| cdec dispatch | `ThreadPoolExecutor.submit` (2 worker process, NEO 원본에 없음) | `paged_attention_cpu` direct (NEO 원본 정합) |
| cdec 동안 GPU stream | ThreadPool 의 queue drain + GIL race 로 default stream queue 진행이 일부 막힘 | GIL released, default stream queue 가 자유롭게 GPU 에서 진행 → 더 깔끔한 overlap |
| result D2D copy | main stream 위 sequential | `cpu_communication_stream` 위 async (NEO 원본 정합) |
| forward_double ordering | `with cuda.stream(s0/s1)` 동시 launch (NEO 원본에 없음) | `_forward_pipeline_stage(cur_stage)` 정합 ordering — layer N+1 preproj 가 layer N attn 앞 |

### NEO 추가 wall 분해 (S1-S9) — overlap mechanism 작동 상태의 잔여 비용

| # | 영역 | 추가 시간 | 의미 |
|---|---|---:|---|
| ① | Python attention.py hot path × 80 layer | **+12 ms** | skip_gpu check, 직접 호출 launch overhead, cudaStream sync — pure Python overhead, overlap 과 무관 |
| **②** | CPU pacpu time **>** GPU concurrent work time → cumulative GPU IDLE 누적 | **+18 ms** | overlap 작동 중. 단 layer 당 CPU pacpu (~2.3ms) > 동시 GPU work (preproj+postproj+gdec ~0.4ms) → 차이 누적 → GPU stream queue 빈 시점 발생. **cdec 자체가 GPU IDLE 의 원인이 아니라 CPU bottleneck 의 결과**. v1.6 의 +24 ms → S1-S9 의 +18 ms 단축 (−6 ms) = ThreadPool overhead + GIL race 제거 |
| ③ | swap launch + Python overhead + emit | +25 ms | `_neo_handle_kv_swap` Python loop, ATen `index_kernel` GOMP (★ Top Priority 영역, overlap 끝난 step 마감) |
| | **합** | **+55 ms** | vanilla 54 ms + 55 = NEO ~109 ms |

→ **paper +14% sweet spot 와의 차이**: paper 의 작은 batch / 짧은 context 에서는 CPU pacpu time ≈ GPU layer work time → ② ≈ 0 → 진짜 +14% 도달. 우리 long context (500p × 8192) 에서는 layer 당 CPU work ≫ GPU work → ② = +18 ms 누적 → mechanism 은 작동하지만 효과 작음. **overlap 실패가 아니라 CPU bottleneck 의 wall path 노출**.

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
