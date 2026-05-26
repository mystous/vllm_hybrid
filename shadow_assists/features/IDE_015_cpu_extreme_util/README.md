# IDE_015 — Sub-Layer Profile + CPU Idle Gap Mapping + CPU Extreme Utilization

> **scope**: Phase A foundation — vllm_hybrid fork 의 CPU 극도 활용 측정 + N curve 비단조 valley + mechanism 규명.
> **parent**: TSK_020 (Spec decode tuning) 후속. id_registry IDE_015.
> **plan doc (full)**: [`/spec_decoding/plan/README.md`](../../../spec_decoding/plan/README.md) — IDE_015~021 전체 hierarchy.
> **last update**: 2026-05-26 KST — Phase A 1차 PoC 완료.

---

## 1. 이론적 배경

### 1.1 D4 GPU util paradox (motivation)

SUB_093/097 측정 (TSK_020) — Trident core (suffix + cudagraph PIECEWISE) 가 vanilla 대비 **throughput +52% 증가** 와 **GPU util −20.5pp 감소** 동시 발생. **CPU util 5.3% 만 유지**. 본 feature 는 두 idle gap (GPU 20pp + CPU 95pp) 의 정량과 fill 가능성 확인.

### 1.2 본 feature 의 기여 (paper-worthy)

| 발견 | 정량 | source SUB |
|---|---|---|
| **AMX BF16 22.05 TFLOPS peak** | Qwen 7B B=256, 20.79× vs FP32 | SUB_106 |
| **Naive worker fill 회귀** | 16-worker unpinned → AGSD −9% | SUB_108 |
| **N curve 비단조 valley** | N=4/8 +3.6%, **N=16 −14.35%**, N=32 +3.9% | SUB_112+116 |
| **VLLM 은 worker affinity pin 없음** | 모든 thread default full-mask 224 CPU | SUB_148 |
| **pinning mechanism = vllm thread eviction** | N=32 가 OS 를 강제로 다른 NUMA 사용 시킴 | SUB_148 |
| **container env cgroup partition invalid** | Podman 내부 cpuset 직접 manipulation 불가 | SUB_114 |
| **CPU util 4.1% → 16% achievable** | N=32 with 10.24 TFLOPS active | SUB_117 |
| **GPU NUMA affinity 확정** | GPU 0-3↔NUMA0, GPU 4-7↔NUMA1 | SUB_113 |

---

## 2. 구현 방향

### 2.1 본 feature 의 lever — physical-core pinning (SUB_112 ⭐⭐)

```python
# /tmp/sub112_cpu_fill_pinned.py 핵심
os.sched_setaffinity(0, {cpu_pin})  # 각 worker physical core 1개에 pin
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# BF16 matmul tight loop
y = A @ W  # K×N = 5120×27648, batch=128 for Qwen 32B MLP shape
```

**결과**: 사용자 100 core max constraint 반영 시 N=20 pinned (CPU 80-99) → ~6.4 TFLOPS gross CPU compute 활용 가능 (Phase A SUB_117 의 N=32 10.24 TFLOPS 비례 scaling).

### 2.2 OS-level 추가 lever (production deploy)

| 단계 | doc | 본 container | production host |
|---|---|---|---|
| `sched_setaffinity` task-pin | SUB_112 protocol | ✅ 사용 중 | ✅ |
| `isolcpus=80-99` boot param | SUB_165 boot_config | ❌ container ext | ✅ |
| cgroup v2 cpuset systemd slice | SUB_165 cgroup_yaml | ❌ partition invalid (SUB_114) | ✅ |
| 2 MB hugepages × 4096 | SUB_165 sysctl conf | ❌ host config 영향 | ✅ |
| IRQ smp_affinity reroute | SUB_165 irq_setup.sh | ⚠ root only | ✅ |

### 2.3 paper §4 main figure 후보 — N curve plot

x-axis: N (CPU fill worker count, log scale)
y-axis: AGSD throughput Δ vs N=0 baseline (%)

```
+5%  |       ⭐                       ⭐
+3%  |   ⭐  N=4   N=8                N=32
   0 |───────────────────────────────────
-5%  |
-10% |                          
-15% |                ⚠ N=16 valley
     +---+---+---+---+---+---+---+---+---+
       1   2   4   8   16  32  64  100
              (N CPU fill workers)
```

---

## 3. Phase A 완료 SUB (시간순)

| 순서 | SUB | 핵심 결과 | RESULTS |
|---:|---|---|---|
| 1 | [SUB_098](SUB_098_baseline_util/RESULTS.md) | canonical baseline lock-in | ✅ |
| 2 | [SUB_099](SUB_099_extended_baseline/RESULTS.md) | 3-run extended baseline | ✅ (mod data) |
| 3 | [SUB_100](SUB_100_tp8_single_util/RESULTS.md) | TP=8 single-instance util | ✅ |
| 4 | [SUB_106](SUB_106_amx_microbench/RESULTS.md) ⭐ | **AMX 22 TFLOPS peak** | ✅ |
| 5 | [SUB_107](SUB_107_cpu_fill_canonical/) | cpu fill v1 (segfault) | ⚠ no RESULTS |
| 6 | [SUB_108](SUB_108_cpu_fill_v2/RESULTS.md) | naive 16W → **−9%** | ✅ |
| 7 | [SUB_109](SUB_109_bisect_workers/RESULTS.md) | qwen7b unpinned N=2 +3.5% | ✅ |
| 8 | [SUB_110](SUB_110_bisect_qwen32b/RESULTS.md) | qwen32b unpinned N=2 +2.8% | ✅ |
| 9 | [SUB_111](SUB_111_sweet_spot_3mix/RESULTS.md) | unpinned 3-mix ceiling +0.07% | ✅ |
| 10 | [SUB_112](SUB_112_pinned_bisect/RESULTS.md) ⭐⭐ | **pinned N=32 +3.9%** core PoC | ✅ |
| 11 | [SUB_113](SUB_113_numa_audit/RESULTS.md) | NUMA + GPU PCIe affinity | ✅ |
| 12 | [SUB_114](SUB_114_irq_cgroup_audit/RESULTS.md) | IRQ + container cgroup 제약 | ✅ |
| 13 | [SUB_116](SUB_116_n16_variance/RESULTS.md) ⚠ | **N=16 valley −14.35%** | ✅ |
| 14 | [SUB_117](SUB_117_per_worker_util/RESULTS.md) | **N=32 10.24 TFLOPS** active | ✅ |
| 15 | [SUB_148](SUB_148_trident_thread_placement/RESULTS.md) ⭐ | **VLLM thread mechanism** 규명 | ✅ |
| 16 | [SUB_160](SUB_160_stability_500p/) | 500p baseline + 1h N=32 stability | 🔵 진행 중 |
| 17 | [SUB_165](SUB_165_cgroup_isolcpus_doc/RESULTS.md) | production cgroup + isolcpus doc | ✅ doc-only |

---

## 4. 다음 step (Phase B 이후)

| Phase | 작업 | 본 fork 자율 가능 여부 |
|---|---|---|
| **Phase A 잔여** | SUB_161 (ncu+py-spy 대체 sublayer profile), SUB_162 (/proc+py-spy CPU state), SUB_163/164 (분석) | ✅ 현재 세션 가능 |
| **Phase 2** (IDE_016/017 kernel) | AVX-512 + AMX C++ intrinsics + DMA primitive | ⚠ 별도 turn (수 주 작업) |
| **Phase 3** (IDE_018 ★ paper main) | Phase-burst scheduler + CUDA event hooks + e2e | ⚠ 별도 turn (paper main) |
| **Phase 4** (IDE_020 production) | Production host에서 SUB_165 doc 적용 + 재측정 | ⚠ host root + orchestrator |

---

## 5. Constraint (memory rules)

- **물리 코어만 사용** (CPU 0-111), HT 시블링 (112-223) 금지 — [feedback_cpu_no_ht](../../../../../root/.claude/projects/-workspace-vllm-hybrid/memory/feedback_cpu_no_ht.md)
- **최대 100 physical core** 활용 (12 core kernel 여유 보존) — kernel panic 회피
- 측정 시간은 KST (UTC+9), `TZ=Asia/Seoul date` 사용
- **commit/push 사용자 명시 지시 시에만**

---

## 6. 관련 file path

| 영역 | path |
|---|---|
| Full plan | [`/spec_decoding/plan/README.md`](../../../spec_decoding/plan/README.md) §IDE_015 + §1.4 |
| canonical AGSD launch | `/tmp/run_sub098_baseline_util.sh` |
| CPU fill worker (pinned) | `/tmp/sub112_cpu_fill_pinned.py` ⭐ |
| benchmark | `/tmp/sub094_benchmark.py` |
| router (FastAPI) | `/tmp/sub094_router.py` |
| monitor | [`/eval/monitor.py`](../../../eval/monitor.py) |
| per-worker util script | `/tmp/sub117_per_worker_util.py` |
| thread state sampler | `/tmp/sub162_thread_state_sampler.py` |
| RESUME doc | [`/spec_decoding/plan/RESUME_AFTER_RESTART.md`](../../../spec_decoding/plan/RESUME_AFTER_RESTART.md) |
