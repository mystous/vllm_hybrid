# §4.3 CapacityAwareRouter & §4.4 Automatic CPU Configuration — 상세 설명

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
- capacity 전략은 N이 고정. 하지만 실제로 CPU 성능은 모델 크기, 양자화 방식, 입력 특성에 따라 달라짐.
- 고정 N=28이면 어떤 모델에서는 CPU가 밀리고, 다른 모델에서는 CPU가 놀 수 있음.
- throughput-adaptive는 이 문제를 해결: 실측 데이터를 보고 N을 늘리거나 줄임.
- 예: CPU가 예상보다 느리면 N을 줄여 CPU 요청을 줄이고, 빠르면 N을 늘려 더 많이 활용.

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
| `cpu_threads` | NUMA 노드의 물리 코어 수 | Hyper-Threading 경합 회피 |
| `max_seqs` | ⌊코어 수 / 4⌋ | 시퀀스당 4스레드 할당 |
| `kv_cache_gb` | 총 메모리 × 0.4 | OS/모델 가중치용 메모리 확보 |
| `batch_tokens` | max_seqs × 256 | 일반적인 디코드 길이 |

#### 각 규칙의 근거

- **cpu_threads = NUMA 물리 코어:** 예를 들어 2소켓 × 56코어 시스템에서, 한 NUMA 노드의 56개 물리 코어만 사용. 112개 논리 코어(HT 포함)를 다 쓰면 오히려 **SMT 경합**으로 성능이 떨어짐. 추론은 memory-bandwidth-bound이므로 HT의 이점이 없고, 캐시 경합만 발생.
- **max_seqs = 코어/4:** 시퀀스 하나당 attention, FFN 등에서 병렬 처리할 스레드가 필요. 실험적으로 시퀀스당 4개 스레드가 최적 균형점. 56코어 → 14 시퀀스 동시 처리.
- **kv_cache_gb = 메모리×0.4:** 2TB 서버에서 800GB를 KV cache로 할당. 나머지는 OS(~수십 GB), 모델 가중치(~수십~수백 GB), PyTorch 버퍼 등에 필요.
- **batch_tokens = seqs×256:** decode 시 한 시퀀스가 평균 ~256 토큰을 생성한다고 가정하여 배치 버퍼 크기 설정.

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

이것이 "zero-configuration" 철학의 핵심. H100 + Xeon 8480+ 최적 환경에서는 모든 기능이 자동 활성화되고, 개발자의 i9 노트북에서도 가용한 기능만으로 동작. 사용자는 `--hybrid-mode parallel-batch` 한 줄만 추가하면 됨.

---

## §4.3 + §4.4 요약

**§4.3**은 "요청을 **어디로** 보내는가"에 대한 답이고, **§4.4**는 "CPU 엔진을 **어떻게** 최적 설정하는가"에 대한 답.

- §4.3: 3단계 라우팅 전략 (capacity → length-aware → throughput-adaptive)으로 단순함에서 정교함으로 확장. 각 전략이 HPC 스케줄링 계보에 이론적 근거를 둠.
- §4.4: hwloc 계층 구조를 따라 하드웨어를 자동 감지하고, 사용자 개입 없이 최적 설정을 도출. 배포 장벽을 제거하는 실용적 기여.
