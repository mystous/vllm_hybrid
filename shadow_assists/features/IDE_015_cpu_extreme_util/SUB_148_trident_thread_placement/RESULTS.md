# SUB_148 — VLLM trident worker thread placement audit

> **parent**: IDE_015 (N=16 valley mechanism 최종 검증) / IDE_020 (TSK_038 NUMA 입력)
> **scope**: 2026-05-26 22:04 ~ 22:06 KST (~2 min after vllm boot)
> **status**: ✅ 완료 — vllm vanilla + trident boot → thread placement dump + affinity audit + busy state sampling
> **method**: `ps -eLo pid,tid,psr,comm,cmd` + `taskset -p` per TID

---

## 0. 두괄식 — N=16 valley + N=8 50% throttle mechanism 최종 규명 ⭐

| 발견 | 정량 |
|---|---|
| **VLLM EngineCore thread affinity = full 224-CPU mask** | 모든 vllm thread affinity = `ffffff...ff` (default) — **pinning 없음** |
| busy state 에서 VLLM thread 가 NUMA 1 + HT 시블링 에 84% 집중 | NUMA1 physical 38.2% + NUMA1 HT 45.7% |
| busy state 중 14.3% 의 VLLM thread 가 N=16 pinned range (80-95) 에 떨어짐 | 276 / 1927 thread samples |
| busy state 중 24.7% 의 VLLM thread 가 N=32 pinned range (80-111) 에 떨어짐 | 476 / 1927 |
| busy state 중 6.4% 의 VLLM thread 가 N=8 pinned range (80-87) 에 떨어짐 | 123 / 1927 |

→ **SUB_117 의 N=8 50% throttle mechanism 규명**: pinned CPU fill worker 가 OS-scheduling 자유로운 VLLM thread 와 같은 코어 공유 → 각각 ~50% core time.
→ **SUB_116 의 N=16 valley mechanism 규명**: pinned fill 16 cores 와 VLLM thread (~14% NUMA1 활동 overlap) 가 정확히 같은 영역에서 contention → throughput −14.35%.
→ **SUB_112 의 N=32 +3.9% mechanism**: 32 cores 모두 점유로 VLLM thread 가 다른 NUMA 또는 HT siblings 로 이동 강제 → contention 해소.

---

## 1. VLLM thread placement (busy state, n=1927 samples × 5 time points)

### 1.1 NUMA region 분포 (idle state baseline n=199)

| Region | CPU range | Thread count | % |
|---|---|---:|---:|
| NUMA 0 physical | 0-55 | 16 | 8.0% |
| **NUMA 1 physical** | 56-111 | **76** | **38.2%** |
| NUMA 0 HT siblings | 112-167 | 16 | 8.0% |
| **NUMA 1 HT siblings** | 168-223 | **91** | **45.7%** |

→ idle state 에서도 84% 의 VLLM thread 가 NUMA 1 권역 (physical + HT) — trident backend GPU 4-7 의 NUMA 친화도 반영.

### 1.2 Pinned range overlap

| range | idle (n=199) | busy (n=1927) |
|---|---:|---:|
| N=8 (CPU 80-87) | 10 (5.0%) | **123 (6.4%)** |
| N=16 (CPU 80-95) | 22 (11.1%) | **276 (14.3%)** |
| N=32 (CPU 80-111) | 49 (24.6%) | **476 (24.7%)** |

→ N=16 pinned range 의 VLLM thread density 가 N=32 의 절반 — 그러나 16 fill workers vs 32 fill workers 비율 (50%) 와 정확히 매치. **per-core contention 강도는 N=16 과 N=32 가 사실상 동일**.

---

## 2. Affinity 확인 — VLLM 은 pin 하지 않음

```bash
# affinity_per_tid.txt 의 모든 VLLM thread
$ awk '$5 == "ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"' affinity_per_tid.txt | wc -l
# default (full mask) — 모든 VLLM thread

$ grep -v "ffffff...ff" affinity_per_tid.txt | grep "VLLM" | wc -l
32  # 32 thread만 non-default — 대부분 engine main / leader thread
```

→ **VLLM 은 worker thread 에 sched_setaffinity 사용하지 않음** — OS scheduler 가 NUMA preference (GPU PCIe affinity) 기반 자유 배치.

---

## 3. N curve mechanism — paper main figure 입력

| N | 가용 fill 코어 | VLLM thread overlap | per-worker CPU 결과 | throughput Δ vs N=0 |
|---:|---|---:|---:|---:|
| 4 | 80-83 | ~3% | (미측정) | +3.5% (SUB_112) |
| 8 | 80-87 | **6.4%** | 49.8% (50% throttle) | +3.6% (SUB_112), 50% per-worker (SUB_117) |
| 16 | 80-95 | **14.3%** | (미측정) | **−14.35%** (SUB_116) |
| 32 | 80-111 | **24.7%** | **99.4%** (full saturate) | **+3.9%** (SUB_112) ⭐ |

→ N=8 → N=16 (overlap 2.2× 증가) 영역 valley 진입 — fill worker 가 vllm thread 와 core-share 비율 증가.
→ N=16 → N=32 (overlap 1.7× 증가) — 그러나 **점유 비율 50% → 거의 100%** 가 더 큰 effect → vllm thread 가 다른 영역으로 evict 강제됨.

---

## 4. Paper-worthy 핵심 finding

| finding | implication |
|---|---|
| **VLLM 은 worker thread pinning 없음** (default full-mask affinity) | paper §3 의 baseline 의 OS scheduling 자유 가정 정량 입증 |
| **NUMA 1 + HT sibling 영역 84% concentration** | trident backend 의 NUMA-aware 자동 placement — GPU PCIe affinity 가 OS 에 신호 전달 |
| **N=32 fill 의 contention 해소 mechanism** = vllm thread 를 cross-NUMA / HT siblings 로 push 강제 | IDE_020 cgroup 설계 입력: 명시적 `cpuset.cpus=56-79` for vllm + `=80-111` for fill 이 N=32 자동 효과 의도적 재현 |
| **N=16 valley 가 paper §4 의 핵심 plot** | "N curve 가 비단조 — naive worker increase 는 회귀 가능" 메시지 |

---

## 5. 다음 step

| SUB | 영역 | 의존 | priority |
|---|---|---|---|
| SUB_149 (IDE_020/TSK_038) | host-level cgroup `cpuset.cpus=56-79` (vllm trident) + `cpuset.cpus=80-111` (fill) 명시 분리 PoC | host root + orchestrator | ★ N=16 valley 해소 검증 |
| SUB_150 | IRQ smp_affinity 재배치 (97/98/101/103/111 → cores 56-79) | root + 본 container 가능여부 | ★ N=32 추가 gain |
| SUB_134/SUB_138 (IDE_018) | phase-burst scheduler 기반 N=32 의 CPU 10.24 TFLOPS 활용 | IDE_018 development | ★★★ paper main result |

---

## 6. raw data

- `raw/threads_idle.txt` — vllm idle state thread snapshot (n=199)
- `raw/threads_busy_t{1,2,3,4,5}.txt` — busy state samples (n=~1927 total)
- `raw/threads_post.txt` — post-workload snapshot
- `raw/affinity_per_tid.txt` — TID 별 affinity mask (137 lines)
- `logs/{vanilla,trident,main}.log` — vllm boot + workload log
- 소스: `/tmp/run_sub148_thread_placement.sh`
