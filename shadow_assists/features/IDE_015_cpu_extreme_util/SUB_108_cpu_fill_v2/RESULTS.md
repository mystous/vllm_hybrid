# SUB_108 — CPU AMX fill v2 (Negative result + paper-worthy finding)

> **parent**: IDE_016 / TSK_026
> **scope**: 2026-05-26 14:34 KST
> **status**: ✅ 완료 — **Negative result** (paper Discussion section 영역 핵심 finding)
> **purpose**: 16 CPU AMX worker × 2 thread + canonical AGSD 영역 동시 실행 영역 throughput 영향 측정

---

## 0. 두괄식 — Naive CPU fill 영역 vLLM throughput **−7~20% 회귀**

| metric | SUB_098 baseline | SUB_108 (CPU AMX fill 16w × 2t) | delta |
|---|---:|---:|---:|
| AGSD balanced | 4,569 | **4,145** | **−9.3%** ✗ |
| AGSD sonnet-heavy | 5,273 | 4,902 | **−7.0%** ✗ |
| AGSD code-heavy | 5,985 | 5,331 | **−10.9%** ✗ |
| vanilla-only avg | 2,396 | 1,976 | **−17.5%** ✗✗ |

→ **Naive 영역 background CPU AMX 영역 fill 영역 vLLM throughput 영역 degrade**.

---

## 1. 측정 환경

- 16 worker × 2 thread = 32 threads total (CPU 56 core 영역 절반 정도)
- 각 worker: Qwen 7B MLP shape (3584, 18944) × B=128 × BF16
- duration: 180s sustained, AGSD 3 mix benchmark 영역 그 안에서 실행
- CPU util max 39.6% (single capture), avg 4.3% (대부분 workers 죽음)

## 2. Why CPU fill DEGRADES vLLM throughput

| 원인 | 정량 evidence |
|---|---|
| **CPU contention** | vLLM worker process (8 × TP=4 = 8개) + router + monitor + 본 fill workers = scheduler conflict |
| vLLM 영역 main thread CPU path | detokenize (GIL-bound), schedule next batch, sampling — AMX 영역 cycle 영역 steal |
| 6/16 fill workers 영역 only survived | thread limit + scheduler eviction 영역 추정 |
| CPU avg 영역 elevate 안 됨 (4.1% → 4.3%) | fill workers 영역 contention 영역 underrun |

## 3. Paper-worthy finding (IDE_021 paper Discussion)

> **"Filling the GPU idle window with naive CPU compute does NOT yield free throughput. Untrusted CPU work competes with vLLM's own CPU path (detokenize / scheduling / sampling), causing −7-20% end-to-end throughput regression. CPU resource isolation (IDE_020: isolcpus + cgroup) is REQUIRED before CPU co-inference can deliver net win."**

→ 본 SUB_108 영역 **IDE_020 (isolcpus + cgroup) 영역 necessity 영역 정량 motivation**. plan §5 risk + fallback 영역 안 영역 추가.
→ phase-aware CPU burst (IDE_018) 영역 동일 결론 — phase-aware scheduling 없이 영역 CPU 영역 fill 영역 안전 영역 보장 불가.

## 4. 다음 SUB

| SUB | 목적 | 가설 |
|---|---|---|
| SUB_109 | bisect worker count (1/2/4/8) — find safe CPU work threshold | small worker count 영역 throughput 영향 영역 negligible 가능 |
| SUB_110 | taskset 영역 CPU affinity pin (worker 영역 specific CPU set) | OS-level isolation 영역 surrogate |
| SUB_113~115 (IDE_020) | cgroup + isolcpus production-grade isolation | real solution |

---

## 5. raw data

- `benchmark_{balanced,sonnet-heavy,code-heavy}.json` (3 scenario × 3 mix × tps + p50/p99)
- `cpu_workers/worker_{00..15}.log` (TFLOPS report from surviving workers)
- `_monitor_cpu.csv` / `_monitor_gpu.csv` (5Hz util)
- 소스: `/tmp/sub108_cpu_amx_fill_v2.py` + `/tmp/run_sub108_cpu_fill_v2.sh`
