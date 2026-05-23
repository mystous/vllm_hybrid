# 병목 분석 — Run-A/B 중간 결과 (2026-05-12 KST)

## Run-E (smoke): Phase 3 sync 제거 검증
- `seq_lens_attr_none=0` ✅ → flash_attn.py `_seq_lens_cpu` field 정상 동작
- `.cpu()` fallback 0건 확인
- crash 0 (assert/cuda/segv/dead/name 모두 0)

## Run-A (py-spy flamegraph 9 프로세스): 22항목 발현 검증

### Worker 측 (raw_tp{0..7}.txt, 60s @ 100Hz)
| 항목 | keyword | 샘플 수 |
|---|---|---|
| #2,#10,#16 | `_neo_cdec_compute_cpu` | 1,259 |
| #3 | `forward_double` | 14,906 |
| #4 | `forward_neo_pipelined` | 15,230 |
| #7 | `unified_attention_with_output` | 3,657 |
| #9 | `ispc_attention_tasks` (--native) | 45 |
| #13 | `forward_pipeline` | 15,222 |
| #12,#14 | `_neo_handle_kv_swap` | 921 |
| #21 | `ensure_capacity` | 13 |

### EngineCore 측 (raw_enginecore.txt)
| 항목 | keyword | 샘플 수 | 비고 |
|---|---|---|---|
| #5,#6,#11,#20 | `neo_scheduler_adapter.py` | 7 | NeoSchedulerAdapter 활성 |
| #1,#8 | `_handle_neo_swaps` | 0 | SWAP_OUT 빈도 낮음 — 미샘플 (인지필요) |
| #22 | `neo_swap_in_alloc` | 0 | swap-in alloc 미샘플 |

### EngineCore 시간 분포
- Total samples: 84 (MainThread)
- schedule (neo_scheduler_adapter): 7 (8%)
- spin-wait (acquire_read/sched_yield): **82.1%**
- → EngineCore가 시간 대부분 Worker 응답 대기 (직렬 동기 대기)

### Worker top function 비율
- `execute_model`: 49,947
- `decorate_context`: 33,479 (PyTorch dispatch)
- `_model_forward`: 15,294
- `forward_neo_pipelined`: 15,230
- `forward_pipeline`: 15,222
- `forward_double`: 14,906
- `attention`: 9,542 / `neo_attention`: 9,461
- `_neo_cdec_compute_cpu`: 1,259 (**2.5%** of execute_model)

## Run-B (NEO_PROFILE): cdec/GPU/SWAP 컴포넌트 시간

### PER-LAYER (TP0~TP7 평균)
| 지표 | 값 |
|---|---|
| cdec_count | 13,100/worker |
| cdec_wait_avg | **8.75 ms/layer** |
| cdec_wait_max | 112-132 ms |
| gpu_avg | 0.09-0.10 ms/layer |
| gpu_max | 4-9 ms |
| **cdec/gpu ratio** | **89-94×** |
| skip_gpu | 12,220 (93%) |
| b0_avg | 3,836 token |
| b1_avg | 0 |

### Per-step 환산 (× 80 layer)
- **cdec_wait total: ~700 ms/step**
- GPU forward total: ~7 ms/step
- → CPU attention 대기가 step time의 압도적 dominant

### SWAP 측정
| 지표 | SWAP_OUT | SWAP_IN |
|---|---|---|
| 이벤트 수 | 542 | 468 |
| avg ms | **74.44** | **54.77** |
| min/max | 54/168 | 44/128 |
| bytes/call | 211.88 MiB | 211.88 MiB |
| effective BW | ~2.85 GiB/s | ~3.87 GiB/s |

### Throughput
- Last 5 (ramp-up): 188.5 → 202.9 → 212.0 → 228.1 → 275.1 tps
- crash: 0 ✅

## 병목 우선순위 (잠정)

| 순위 | 영역 | 비용 | 출처 |
|---|---|---|---|
| 1 | cdec_wait (CPU attention 대기) | **~700 ms/step** | Run-B PER-LAYER |
| 2 | EngineCore spin-wait | 82% time | Run-A flamegraph |
| 3 | SWAP_OUT 74 ms × 542 = 40.2 s 누적 | per-call dominant | Run-B SWAP |
| 4 | SWAP_IN 55 ms × 468 = 25.7 s 누적 | per-call | Run-B SWAP |
| 5 | NCCL allreduce | (Run-C 측정 예정) | torch.profiler |

