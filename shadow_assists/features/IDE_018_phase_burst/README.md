# IDE_018 — Sub-Layer Phase-Aware CPU Burst ★★★ core paper

> **scope**: GPU sublayer phase (attention vs linear) 별 CPU task switching — paper main contribution.
> **paper angle**: 직접 대응 논문 없음 — sublayer-granular phase detection + CPU task pool 의 OS-coordination + production e2e.
> **status**: ✅ design + skeleton 작성 완료 / ⚠ vLLM CUDA event hook + scheduler runtime 별도 turn.

---

## 1. 이론적 배경 (paper §3 input)

### 1.1 thesis

> **"Spec decoding 의 GPU util −20.5pp 와 CPU util −95pp 두 idle gap 을 sublayer-phase-aware CPU task burst 으로 동시에 fill 한다."**

### 1.2 Phase A 측정 lever 정량 입증

| Phase A finding | IDE_018 의 입력 |
|---|---|
| **VLLM threads 96-100% S** (SUB_162) | CPU 자원 거의 100% idle — task pool 의 absolute capacity |
| **10.24 TFLOPS available** (SUB_117) | NUMA 1 N=32 가용 CPU compute |
| **sampler.py 44.3% CPU** (SUB_161) | sampling-phase task pool 의 main item |
| **VLLM 은 worker affinity pin 없음** (SUB_148) | phase-burst scheduler 가 OS sched 와 협조 가능 |
| **DMA 1 MB crossover / 54 GB/s asymptotic** (SUB_166) | linear-phase task (KV prefetch) 의 data plane |
| **paper Table 1a/1b matrix** (SUB_167/168) | 10 task × 5 phase 의 dispatch policy |

### 1.3 4 sub-task

| TSK | 영역 | scope | priority |
|---|---|---|---|
| TSK_031 | Phase detection mechanism (CUDA event hooks) | per-step phase boundary signal < 50 μs | ★★★ critical |
| TSK_032 | Attention-phase CPU task pool | schedule / detokenize / grammar / classify | ★★ |
| TSK_033 | Linear-phase CPU task pool | KV prefetch / AMX draft / cold-KV decompress | ★★ |
| **TSK_034** | **Integration + measurement ★ paper main result** | full e2e + CPU util 5%→30%+ + throughput delta | **★★★ paper main** |

---

## 2. 구현 방향

### 2.1 Phase detection via CUDA event hooks (TSK_031)

```cpp
// src/phase_burst/phase_detector.cpp
// vLLM 의 forward path 에 CUDA event 삽입:
//   attention_enter / attention_exit / linear_enter / linear_exit / sample_enter / sample_exit
//
// IPC: shared atomic counter + eventfd notify CPU task pool
struct PhaseSignal {
    std::atomic<uint64_t> step_id;
    std::atomic<uint8_t>  phase;     // 0=attn, 1=linear, 2=sample, 3=tp_allreduce, 4=idle
    std::atomic<uint64_t> phase_start_ns;
};
```

### 2.2 Phase-burst scheduler (TSK_034)

```cpp
// src/phase_burst/scheduler.cpp
class PhaseBurstScheduler {
public:
    void on_phase_signal(uint8_t phase);

    // dispatch tasks from pool based on phase
    void run_attention_phase_tasks();   // task A,B,C,D (SUB_168 Table 1b)
    void run_linear_phase_tasks();       // task E,F,G
    void run_sample_phase_tasks();       // task H,I
    void run_idle_tasks();               // any task — inter-step wait

private:
    TaskPool attn_pool_;
    TaskPool linear_pool_;
    TaskPool sample_pool_;
};
```

### 2.3 vLLM integration points

- `vllm/v1/worker/gpu_model_runner.py` — forward path 의 phase boundary
- `vllm/v1/engine/core.py` — engine main loop 의 inter-step idle
- patch 방식: monkey-patch via plugin entry point

---

## 3. paper main figure 예상

```
            attention      linear       sample      idle      attention   linear   ...
GPU:        ▓▓▓▓▓░░░░░     ▓▓▓▓▓▓▓▓▓    ▓░░░░░░░    ░░░░░░    ▓▓▓▓▓░░░░░  ...
CPU baseline: ░░░░░░░░░░    ░░░░░░░░░░   ░░░░░░░░    ░░░░░░    ░░░░░░░░░░  ...   (4.1%)
CPU IDE_018: ▓▓▓ A/B/C/D    ▓▓▓ E/F/G   ▓▓▓ H/I     ▓▓▓ any   ▓▓▓ A/B/C/D ...   (30%+ target)
```

→ paper §4 Figure 5 후보 — CPU util 5%→30%+ + throughput delta.

---

## 4. 검증 (test.md 참조)

- TSK_031: phase signal latency target ≤ 50 μs (CUDA event timestamp 정확도)
- TSK_034: canonical AGSD-gated balanced 500p, CPU util 측정, throughput delta
- target: CPU util 5%→**30%+**, throughput +10-20% (paper main result)

---

## 5. dependencies

- IDE_016 (AVX-512 sampling kernel, AMX draft head) — task H,I,F 의 implementation
- IDE_017 (DMA + zero-copy) — task E (KV prefetch), task G (cold-KV)
- IDE_019 (multi-source drafter) — task F 의 specific implementation
- IDE_020 (production isolation) — task pool 의 dedicated CPU cores
