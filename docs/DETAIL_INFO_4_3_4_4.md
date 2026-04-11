# §4.3 CapacityAwareRouter & §4.4 Automatic CPU Configuration — 상세 설명

> **마지막 업데이트**: 2026-04-11

## §4.3 CapacityAwareRouter

### 핵심 아이디어

들어오는 추론 요청을 **GPU로 보낼지, CPU로 보낼지** 결정하는 라우터. 이 라우터의 설계는 3가지 HPC 스케줄링 기법에 뿌리를 두고 있다.

---

### HPC 스케줄링 계보

| 전략 | HPC 원조 기법 | 핵심 원리 |
|------|-------------|----------|
| **capacity** | Work Stealing (Blumofe & Leiserson, 1999) | 놀고 있는 프로세서가 바쁜 프로세서의 작업을 "훔쳐온다" |
| **length-aware** | HEFT (Topcuoglu et al., 2002) | 각 작업을 **가장 빨리 끝낼 수 있는** 프로세서에 배치 |
| **throughput-adaptive** | StarPU (Augonnet et al., 2011) | **실측 성능**을 기반으로 실시간으로 배치를 조정 |

#### Work Stealing
- 원래 멀티스레드 환경에서 나온 개념. 스레드 A가 할 일이 없으면 스레드 B의 작업 큐에서 가져옴.
- 우리 시스템에서는 CPU가 "놀고 있는 프로세서"이고, GPU에 몰리는 요청을 CPU가 빈 슬롯이 있을 때 가져가는 것이 work stealing과 동일.

#### HEFT (Heterogeneous Earliest Finish Time)
- 이기종 시스템에서 DAG 작업 스케줄링할 때 각 노드를 "Earliest Finish Time이 최소인 프로세서"에 배치.
- 우리의 length-aware 전략은 이 원리를 적용: 긴 프롬프트(compute-intensive) → GPU가 빨리 끝냄, 짧은 프롬프트(memory-bound) → CPU가 즉시 처리 가능.

#### StarPU
- GPU/CPU 이기종 시스템의 런타임 스케줄러로, 각 디바이스에서 실제로 측정된 실행 시간을 기반으로 다음 작업 배치를 결정.
- 우리의 throughput-adaptive 전략이 EMA로 실측 처리량을 추적하는 것이 이와 동일.

---

### Algorithm 1: Capacity-Based Strategy

```
입력: 요청 r, 현재 CPU 처리중 수 C, CPU 최대 슬롯 N
출력: GPU 또는 CPU

if C < N:        ← CPU에 빈 슬롯이 있으면
    C = C + 1    ← 슬롯 하나 차지
    return CPU   ← CPU로 보냄
return GPU       ← 슬롯이 꽉 차면 GPU로 보냄
```

알고리즘 자체는 극도로 단순. 이것이 의도적인 설계. 복잡한 비용 함수나 예측 모델 없이, **"CPU에 자리가 있나?"**라는 단 하나의 질문으로 라우팅을 결정. 이 단순함 덕분에 라우팅 오버헤드가 O(1)이고, §4.2에서 보인 11μs 라우팅 비용이 가능해짐.

---

### 3가지 Property

#### Property 1 — Self-Regulation (자기 조절)
- 외부에서 "CPU가 바쁘다/한가하다"를 알려주는 피드백 신호가 **불필요**
- C 자체가 자기 조절 메트릭: CPU가 요청을 완료하면 C가 자동 감소 → 새 요청이 CPU로 감 → C 다시 증가
- 별도의 모니터링 시스템이나 heartbeat 없이 동작

#### Property 2 — GPU Non-Interference (GPU 무간섭)
- GPU가 **기본(default)** 타겟. CPU는 여분이 있을 때만 받음
- CPU가 크래시하거나 먹통이 되면? → C가 N 아래로 절대 안 내려감 → 모든 요청이 GPU로 감
- 즉, CPU 장애 시 자동으로 **GPU-only 모드로 graceful degradation**

#### Property 3 — Maximum CPU Utilization (CPU 최대 활용)
- 부하가 높을 때(λ > T_GPU), CPU 요청이 하나 완료될 때마다 슬롯이 비고, 대기 중인 다음 요청이 즉시 그 슬롯을 채움
- 결과적으로 CPU 활용률이 100%에 수렴 → §4.2의 α → 1과 직접 연결

**3가지 property의 의미:** Property 1은 "관리 비용 제로", Property 2는 "장애 내성", Property 3은 "성능 극대화"를 보장. 이 세 가지가 결합되어, 운영자가 별도 모니터링이나 수동 조정 없이도 시스템이 알아서 최적 상태를 유지.

---

### Algorithm 2: Length-Aware Strategy

```
입력: 요청 r, 프롬프트 길이 |r|, 임계값 τ, CPU 처리중 수 C, 최대 슬롯 N
출력: GPU 또는 CPU

if |r| > τ:      ← 프롬프트가 τ 토큰 초과면
    return GPU   ← 무조건 GPU (HEFT 원리: GPU가 긴 prefill 더 빨리 끝냄)
if C < N:        ← CPU에 빈 슬롯 있으면
    C = C + 1
    return CPU   ← 짧은 프롬프트는 CPU로 (work stealing)
return GPU
```

#### 왜 프롬프트 길이가 중요한가?
- **Prefill 단계** (처음 프롬프트를 처리하는 단계)는 **compute-intensive**. 프롬프트의 모든 토큰을 동시에 attention 계산해야 하므로, 토큰이 많을수록 연산량이 폭증 (O(n²) attention).
- GPU는 compute가 강하므로 긴 prefill에 유리. CPU는 memory bandwidth는 있지만 compute가 약하므로 긴 prefill에서 극도로 느려짐.
- **Decode 단계** (한 토큰씩 생성)는 memory-bandwidth-bound이므로 CPU도 잘 처리.
- 따라서 τ (기본 512 토큰)을 기준으로: 긴 프롬프트 → GPU, 짧은 프롬프트 → CPU 슬롯 있으면 CPU.

---

### Throughput-Adaptive Strategy

Algorithm으로 제시하지 않고 텍스트로 설명:

1. **EMA(지수이동평균)** 로 GPU와 CPU의 실제 완료율(tokens/s)을 지속적으로 추적
2. 관측된 처리량 비율에 비례하여 **CPU 슬롯 수 N을 동적 조정**
3. **Warmup 단계:** 처음 W개 요청(기본 10개)은 양쪽 엔진에 균등 배분하여 초기 EMA를 구함
4. **StarPU와의 차이:** StarPU는 매 작업마다 디바이스를 선택하지만, 우리는 **단일 글로벌 파라미터 N** 하나만 조정 → 캘리브레이션 오버헤드를 수많은 요청에 걸쳐 분산(amortize)

#### 왜 이 전략이 필요한가?
- capacity 전략은 N 이 고정 (`num_cpu_engines × cpu_max_num_seqs = num_numa_nodes × 1`). 실제로 CPU 성능은 모델 크기, 양자화 방식, 입력 특성에 따라 달라진다.
- 고정 N 이면 어떤 모델에서는 CPU 가 밀리고 (queue 쌓임), 다른 모델에서는 CPU 가 논다 (slot 낭비).
- throughput-adaptive 는 이 문제를 실측 데이터를 사용해 동적 threshold (length-aware 의 τ) 또는 effective N 을 EMA 비율로 조정하여 완화한다.
- 주의: 본 프로젝트의 기본 N 은 hardware topology 에서 유도되며, 변경 시 runtime 재시작이 필요하다. throughput-adaptive 는 "라우팅 결정 임계"를 조정하지 per-engine `cpu_max_num_seqs` 자체를 건드리지는 않는다.

---

## §4.4 Automatic CPU Configuration

### 핵심 문제

이기종 배포의 **가장 큰 장벽은 설정 복잡성**. CPU 추론을 제대로 실행하려면:
- 스레드 수는 몇 개?
- 어느 NUMA 노드에 바인딩?
- KV cache 메모리는 얼마?
- AVX-512? AMX? VNNI? 어떤 ISA를 사용?
- OpenMP affinity는 어떻게 설정?

이걸 수동으로 하면 전문 지식이 필요하고, 하드웨어마다 다르게 설정해야 함. 이 시스템은 **모든 설정을 0(auto)으로 두면 하드웨어에서 자동 감지**.

---

### Auto Parameter Derivation

| 파라미터 | 자동 규칙 | 이유 |
|---------|----------|------|
| `num_cpu_engines` | NUMA 노드 수 | 노드 당 1 개의 독립 CPU EngineCore 프로세스 — strict NUMA bind 로 remote memory access 제거 |
| `cpu_threads` (per engine) | 해당 NUMA 노드의 물리 코어 수 전체 | Hyper-Threading 경합 회피 |
| `cpu_max_num_seqs` (per engine) | **1 고정** | 1 시퀀스가 NUMA 의 모든 물리 코어를 OMP + BLAS matmul 병렬로 사용. 배치를 만들지 않음 |
| `kv_cache_gb` | `clamp(eff_mem × 0.4, 32, 512)` | OS / 모델 가중치 / torch 버퍼 확보 후 나머지를 KV cache 로 |
| `batch_tokens` | `cpu_max_num_seqs × 256` (= 256) | 짧은 decode 배치 버퍼 |

#### 각 규칙의 근거

- **num_cpu_engines = NUMA 노드 수:** 2 소켓 = 2 NUMA 시스템이면 CPU EngineCore 프로세스를 2 개 띄우고, 각각 자기 NUMA 노드의 물리 코어와 DRAM 에 strict bind. 프로세스 간 GIL / 메모리 allocator / OpenMP pool 을 완전히 분리한다. `vllm/v1/engine/hybrid_core.py :: _resolve_num_cpu_engines` 가 `NUMAAllocator.num_nodes` 로 자동 감지.
- **cpu_threads = NUMA 물리 코어:** 예를 들어 2 소켓 × 56 코어 시스템이면 한 NUMA 노드의 56 물리 코어만 사용. 112 논리 코어 (HT 포함) 를 다 쓰면 SMT 경합으로 성능이 떨어진다 (memory-bandwidth-bound + 캐시 경합).
- **cpu_max_num_seqs = 1 고정 (per engine):** 예전 draft 에서는 `max(4, ⌊cores/4⌋)` 로 기술했으나, 이는 **잘못된 규칙** 이었다. 실측 결과 per-engine 1 시퀀스로 고정하고 NUMA 의 모든 물리 코어를 해당 1 시퀀스의 matmul 에 쏟아붓는 편이 wall-clock 기준 가장 빠르다. 여러 시퀀스를 배치로 묶으면 per-seq OMP pool 이 잘게 쪼개져 NUMA/cache 이점이 사라진다. 총 동시 CPU 시퀀스 = `num_cpu_engines × 1 = num_numa_nodes`. 이 원칙은 `_resolve_cpu_params` 에 강제되며, 사용자가 `cpu_max_num_seqs ≠ 1` 로 override 하면 경고 로그가 출력된다.
- **kv_cache_gb:** `clamp(eff_mem × 0.4, 32, 512)` — 가용 메모리의 40% 를 CPU KV cache 로 할당하되 하한 32 GB / 상한 512 GB. 나머지는 OS, 모델 가중치, PyTorch 버퍼 등에 필요.
- **batch_tokens = seqs × 256:** per-engine seqs=1 기준으로 256 토큰. 짧은 decode 배치 버퍼.

---

### hwloc 계층 모델

하드웨어 토폴로지를 **hwloc 프레임워크**(Broquedis et al., 2010)의 계층 구조로 탐색:

```
Machine (전체 시스템)
  └─ NUMANode (NUMA 노드)
       └─ Package (물리 소켓/CPU 패키지)
            └─ L3Cache (L3 캐시 공유 단위)
                 └─ Core (물리 코어)
                      └─ PU (Processing Unit = 논리 코어/HT 스레드)
```

각 계층에서 읽어오는 정보:

| 계층 | 감지 내용 | 왜 필요한가 |
|------|----------|-----------|
| **NUMANode** | 스레드 수, 메모리 바인딩 | 같은 NUMA 노드의 코어와 메모리를 매칭해야 remote access 페널티(~30-40% 느림)를 피함 |
| **Package** | CPU 엔진을 실행할 소켓 선택 | 2소켓 시스템에서 GPU가 소켓 0에 연결되어 있으면 CPU 엔진은 소켓 1 사용 (PCIe 경합 회피) |
| **Core vs PU** | 물리 코어와 HT 스레드 구분 | 물리 56코어 vs 논리 112코어. HT 스레드는 같은 실행 유닛을 공유하므로 메모리 바운드 워크로드에서 이득 없음 |
| **ISA 감지** | AVX-512, VNNI, AMX | `/proc/cpuinfo`에서 읽어서 최적 커널 자동 선택 |

#### 왜 hwloc이 중요한가?
- 단순히 "코어 몇 개인가"만 알면 안 됨. **어떤 코어가 어떤 메모리에 가까운가**가 성능을 크게 좌우.
- NUMA 시스템에서 "remote memory access"(다른 소켓의 메모리 접근)는 local에 비해 1.3~1.5배 느림. 이 때문에 스레드와 메모리를 같은 NUMA 노드에 바인딩하는 것이 필수.
- hwloc의 6단계 계층 모델은 이 정보를 체계적으로 제공. 우리 시스템은 OS의 sysfs (`/sys/devices/system/node/`, `/proc/cpuinfo`)에서 이 정보를 직접 읽어 hwloc 없이도 동작하되, 개념적으로 hwloc의 계층 구조를 따름.

---

### Intel 최적화 설정

감지된 ISA 기반으로 자동 설정되는 항목:

| 설정 | 값 | 목적 |
|------|-----|------|
| `KMP_AFFINITY` | `granularity=fine,compact,1,0` | OpenMP 스레드를 물리 코어에 1:1 바인딩 |
| `ONEDNN_MAX_CPU_ISA` | `AVX512_CORE_AMX` | oneDNN에서 AMX 커널 활성화 |
| `MKL_ENABLE_INSTRUCTIONS` | `AVX512` | MKL에서 AVX-512 사용 |
| `KMP_BLOCKTIME` | `1` | 스레드가 idle 후 1ms만 spinning 후 sleep (전력 절약) |

---

### Graceful Fallback

```
IPEX 없음 → PyTorch 기본 CPU 백엔드 사용
NUMA 라이브러리 없음 → 표준 메모리 할당 사용
AMX 없음 → AVX-512로 fallback
AVX-512 없음 → AVX2로 fallback
```

이것이 "zero-configuration" 철학의 핵심. 고성능 Xeon + 다중 NUMA 환경에서는 AMX / VNNI / NUMA 분리가 모두 자동 활성화되고, AVX2 밖에 없는 개발자 노트북에서도 가용한 기능만으로 동작한다. 사용자는 `--hybrid-mode parallel-batch` 한 줄만 추가하면 된다.

---

## §4.3 + §4.4 요약

**§4.3**은 "요청을 **어디로** 보내는가"에 대한 답이고, **§4.4**는 "CPU 엔진을 **어떻게** 최적 설정하는가"에 대한 답.

- §4.3: 3단계 라우팅 전략 (capacity → length-aware → throughput-adaptive)으로 단순함에서 정교함으로 확장. 각 전략이 HPC 스케줄링 계보에 이론적 근거를 둠.
- §4.4: hwloc 계층 구조를 따라 하드웨어를 자동 감지하고, 사용자 개입 없이 최적 설정을 도출. 배포 장벽을 제거하는 실용적 기여.
