# §4.2 Throughput Model — 심층 해설

> 논문 §4.2 에 사용된 HPC 개념 (LogGP, Roofline, α) 의 배경 조사 및 상세 해설
>
> **마지막 업데이트**: 2026-04-11
>
> 이 문서의 수식/정의는 `docs/paper/main.tex` 에 이미 반영되어 있다. 본 문서는 그
> 배경 해설이며 Roofline 계산의 수치 예시에는 실제 평가 대상인 NVIDIA H100 +
> Intel Xeon Platinum 8480+ 를 사용하지만, 이는 **하나의 예시 환경**이고 구현
> 자체는 어떤 x86_64 + CUDA GPU 에서도 런타임 감지 후 자동 동작한다.

---

## 목차

1. [Definition 1: Hybrid Throughput](#1-definition-1-hybrid-throughput)
2. [Theorem 1: Additive Throughput](#2-theorem-1-additive-throughput)
3. [Corollary 1: GPU Latency Preservation — LogGP 모델](#3-corollary-1-gpu-latency-preservation--loggp-모델)
4. [Corollary 2: Energy Efficiency](#4-corollary-2-energy-efficiency)
5. [Corollary 3: Roofline-Bounded CPU Contribution](#5-corollary-3-roofline-bounded-cpu-contribution)
6. [참고 문헌](#6-참고-문헌)

---

## 1. Definition 1: Hybrid Throughput

```
T_hybrid = T_GPU + α · T_CPU     (α ∈ [0, 1])
```

### α(알파)란?

α는 **CPU의 실효 활용률**을 나타내는 무차원 계수이다.

- α=0 → CPU가 놀고 있음 → GPU-only와 동일
- α=1 → CPU가 이론적 최대 처리량을 달성 → 풀가동
- α=0.7 → CPU 최대 용량의 70%만 사용 중

### α가 1 미만인 경우

- 부하가 낮아서 GPU만으로 충분히 소화 → CPU에 요청이 거의 안 감
- 요청이 간헐적으로 와서 CPU 슬롯이 빈 채로 있는 시간이 많음
- 라우팅이 비효율적으로 CPU를 과소 활용

### α가 1에 가까운 경우

- 높은 부하: 요청이 충분히 많아 CPU 슬롯이 항상 꽉 참
- 효율적 라우팅: 요청이 슬롯 반환 즉시 새로 할당됨
- 균일한 요청 크기: 각 요청의 처리 시간 변동이 적어 슬롯 회전이 균등

### α가 1을 초과할 수 없는 이유

물리적으로 CPU 가 자신의 최대 처리량을 초과할 수 없다. 슬롯 수는 유한하고
(`num_cpu_engines × cpu_max_num_seqs = num_numa_nodes × 1`, 현재 구현 원칙), 각
슬롯의 처리 시간에는 하한이 있다. 본 프로젝트는 per-engine `cpu_max_num_seqs = 1`
을 고정하여 각 CPU engine 이 자기 NUMA 노드의 모든 물리 코어를 OMP 로 점유하게 한다.

### "+"가 성립하는 이유

§4.1의 HBSP에서 설명됨 — GPU와 CPU가 독립 프로세스이므로 처리량이 **더해진다**. 서로 간섭하지 않으므로 빼기나 감소가 없다.

---

## 2. Theorem 1: Additive Throughput (α → 1 수렴)

### 주장

요청 도착률 λ가 GPU 처리량을 넘으면 α → 1

### 증명의 논리

```
1. λ > T_GPU  (요청이 GPU 혼자 감당할 수 있는 것보다 많이 옴)
2. → GPU 큐가 쌓임
3. → 라우터가 CPU 슬롯(C < N)을 확인하고 CPU로 보냄
4. → λ가 계속 높으면 CPU 슬롯이 점점 찬다 (C → N)
5. → CPU 활용률 100% → α → 1
```

### CapacityAwareRouter의 메커니즘

```python
def route(self, request):
    if self.cpu_in_flight < self.cpu_max_num_seqs:  # C < N
        self.cpu_in_flight += 1                      # C++
        return "cpu"
    return "gpu"
```

**빈 슬롯이 있으면 무조건 채움** → 부하가 충분하면 슬롯이 항상 꽉 참 → α=1

이것은 HPC에서 말하는 **work-conserving 스케줄러** — 자원이 비어있고 할 일이 있으면 절대 쉬지 않는다.

### RequestRouter(고정 비율)와의 비교

| 속성 | RequestRouter (고정 비율) | CapacityAwareRouter (슬롯 기반) |
|------|---------------------------|----------------------------------|
| 라우팅 결정 | N번째마다 CPU | CPU 슬롯 비었으면 즉시 CPU |
| 낮은 부하 시 | CPU 일부만 활용 | 여유 있는 한 CPU 우선 |
| 높은 부하 시 | 비율 고정으로 CPU 과부하/과소 가능 | 슬롯 꽉 차면 자동 GPU 전환 |
| α 상한 | cpu_ratio에 의해 제한 | **부하 충분 시 1에 수렴** |

---

## 3. Corollary 1: GPU Latency Preservation — LogGP 모델

### 질문

"CPU 엔진을 추가하면 GPU 성능이 떨어지지 않나?"

### 답

떨어지지 않는다. 유일한 공유 지점은 ZMQ 라우팅인데, 이 오버헤드가 무시 가능하다.

---

### 3.1 LogP 모델 (Culler et al., PPoPP 1993)

David Culler, Richard Karp, David Patterson 등이 1993년 PPoPP 학회에서 발표한 병렬 머신 통신 모델. 이름 "LogP"는 4개 파라미터의 머리글자이다.

#### 4개 파라미터

| 파라미터 | 이름 | 정의 | 특징 |
|----------|------|------|------|
| **L** (Latency) | 지연시간 | 메시지가 통신 링크에 진입하여 수신자에게 도착할 때까지의 시간 | 이 동안 프로세서는 **자유** (다른 일 가능) |
| **o** (overhead) | 오버헤드 | 송신 또는 수신 시 프로세서가 통신 활동에 **점유되어** 다른 작업을 할 수 없는 시간. 커널 시스템콜, 버퍼 복사, 프로토콜 처리 등 포함 | 확장 모델에서 `o_s`(송신)와 `o_r`(수신)으로 분리 |
| **g** (gap) | 갭 | 한 프로세서에서 연속적인 두 메시지 송신(또는 수신) 사이의 **최소 시간 간격**. 1/g는 통신 채널의 대역폭으로 해석 | 짧은 메시지의 대역폭 결정 |
| **P** (Processors) | 프로세서 수 | 병렬 시스템의 프로세서/메모리 모듈 수 | |

#### 기본 통신 비용

```
T_short = o_s + L + o_r
```

#### 핵심 제약

- 프로세서는 o 시간 동안 점유(busy)되어 계산이나 다른 통신 불가
- L 시간 동안에는 프로세서가 자유(free) — 계산 가능
- g는 메시지 주입률(injection rate)의 상한을 설정

---

### 3.2 LogGP 확장 (Alexandrov et al., SPAA 1995)

#### 문제: LogP는 짧은 메시지만 정확하게 모델링

실제 병렬 머신은 **긴 메시지에 대해 특수한 DMA 전송 지원**을 가지고 있어, 짧은 메시지보다 훨씬 높은 대역폭을 달성한다. LogP의 고정 크기 메시지 가정으로는 이를 포착할 수 없었다.

#### G 파라미터 (Gap per Byte)

| 파라미터 | 정의 |
|----------|------|
| **G** (Gap per byte) | 긴 메시지의 벌크 전송 시 **바이트당 시간**. 프로세서는 이 시간 동안 자유(free). |

G는 긴 메시지의 실효 대역폭을 모델링한다: `BW_long = 1/G` bytes/sec.

#### LogGP 점대점 메시지 비용 공식

```
T_msg(m) = o_s + L + (m - 1) × G + o_r
```

여기서 m은 메시지 크기(바이트).

---

### 3.3 짧은 메시지 vs 긴 메시지: 어느 파라미터가 지배하는가

**짧은 메시지 (제어 신호, 라우팅 결정):**

```
T_short = o_s + L + o_r        (m이 작으므로 (m-1)×G ≈ 0)
```

- `o` (오버헤드)가 지배적: 커널 시스템콜, ZMQ 프레이밍, 버퍼 복사 등이 대부분의 시간
- `L` (지연시간): IPC의 경우 매우 작음 (같은 머신에서 수백 ns)
- `(m-1)×G`: 페이로드가 작으므로 무시 가능

**긴 메시지 (데이터 전송):**

```
T_long ≈ m × G    (m >> 1일 때, o와 L은 상수 오버헤드로 무시)
```

- `(m-1)×G`가 지배적: 벌크 데이터 전송 대역폭이 총 시간의 대부분
- G = 1/BW이므로, `T_long ≈ m / BW` — 순수 대역폭 제한

---

### 3.4 우리 시스템에 대입: ZMQ IPC

vLLM Hybrid에서 HybridAsyncMPClient는 ZMQ IPC(`ipc://` 프로토콜)로 GPU/CPU EngineCoreProc과 통신한다. 이것은 Unix domain socket 기반이다.

#### Unix Domain Socket 레이턴시 실측값

벤치마크(Kamal Marhubi, 2015) 기준:

| 백분위 | 왕복 지연시간 |
|---------|-------------|
| 50th (중위) | 1,439 ns (~1.4 μs) |
| 75th | 1,621 ns |
| 99th | 1,898 ns |
| 99.9th | 2,681 ns |

#### ZMQ IPC 레이턴시

ZeroMQ Performance 문서 및 libzmq 이슈 기준:
- 연속 전송(back-to-back): ~40 μs (1-byte 메시지)
- 단일 메시지: ZMQ 프레이밍 오버헤드 ~15 μs + 네트워크 스택 ~25 μs
- 간헐적 전송(timeout 사이): 150-400 μs (ZMQ 내부 배치 최적화가 꺼짐)

#### LogGP 파라미터 추정

| 파라미터 | 추정값 | 근거 |
|----------|--------|------|
| **L** | ~0.5-1 μs | Unix domain socket 커널 경로 지연 |
| **o_s** | ~5-20 μs | ZMQ 송신 오버헤드: zmq_msg_init, 프레이밍, sendmsg |
| **o_r** | ~5-20 μs | ZMQ 수신 오버헤드: zmq_msg_recv, 디프레이밍, 복사 |
| **g** | ~1-5 μs | 연속 짧은 메시지 간 최소 간격 |
| **G** | ~1-2 ns/byte | IPC 벌크 전송 대역폭 ~1-2 GB/s (memcpy 속도에 근접) |

#### 논문에서의 계산

```
T_route = o_s + L + o_r

논문 값: L ≈ 5μs, o_s ≈ 3μs, o_r ≈ 3μs
→ T_route ≈ 11μs

비교: T_decode ≈ 30~100ms (decode 1 step)

비율: 11μs / 30ms = 0.037% < 0.04%
→ 3자릿수(1000배) 차이 → "negligible by three orders of magnitude"
```

---

### 3.5 LogGP의 학술적 가치: "그냥 레이턴시 측정"과의 차이

1. **분해(Decomposition)**: 총 비용을 o_s, L, G, o_r로 분해 → "어떤 구성요소가 병목인가?" 진단 가능
2. **중첩(Overlap) 모델링**: o 동안 프로세서 점유, L 동안 자유 → 계산-통신 중첩 분석에 필수. 단순 레이턴시 측정으로는 이 구분 불가
3. **스케일링 예측**: 메시지 크기 변화 시 비용 변화를 해석적으로 예측 (`T(m) = 2o + L + (m-1)G`)
4. **시스템 비교**: 서로 다른 통신 백엔드(ZMQ IPC vs shared memory vs TCP)를 파라미터 수준에서 비교 — "왜 빠른가?(o가 작은가? L이 작은가?)"
5. **알고리즘 설계 지침**: 짧은 메시지 多 → o가 중요, 긴 메시지 少 → G가 중요. 트레이드오프 정량화

---

## 4. Corollary 2: Energy Efficiency

```
η = (T_hybrid / P_hybrid) / (T_GPU / P_GPU)
  = (1 + α·T_CPU/T_GPU) / (1 + ΔP/P_GPU)
```

- 분자: 처리량 증가율 (1 + CPU 기여분)
- 분모: 전력 증가율 (1 + CPU 추가 전력)
- ΔP (CPU idle→active 추가 전력) ≈ 175W
- P_GPU (H100 ×8 + 시스템) > 5,600W
- → 분모 ≈ 1.03, 분자 > 1.03이면 η > 1 (에너지 효율 개선)

---

## 5. Corollary 3: Roofline-Bounded CPU Contribution

### 핵심 질문

"CPU가 진짜 의미 있는 기여를 하나? 얼마나?"

---

### 5.1 Roofline 모델이란? (Williams, Waterman, Patterson, CACM 2009)

시각적 성능 분석 프레임워크로, 하드웨어의 두 가지 한계를 2D 그래프에 표현한다.

#### Operational Intensity (연산 강도, OI)

```
OI = 수행한 연산량(FLOPs) / 이동한 데이터량(Bytes)    [단위: FLOP/Byte]
```

어떤 커널이 데이터 1바이트를 읽을 때마다 몇 번의 연산을 하는가?

#### 두 개의 "지붕(roof)"

```
달성 가능 성능 = min(π, β × OI)
```

- **π** (Peak Performance): 하드웨어 최대 연산 처리량 (FLOP/s)
- **β** (Peak Bandwidth): 메모리 대역폭 (Byte/s)

log-log 그래프:

```
성능(GFLOPS)
    │
    │         _______________  ← 연산 지붕 (compute roof): π
    │        /
    │       /
    │      /  ← 대역폭 지붕 (bandwidth roof): 기울기 = β
    │     /
    │    /
    │   /
    └──────────────────── OI (FLOP/Byte)
              ↑
          능선점(ridge point): OI_ridge = π / β
```

- **능선점 왼쪽** (OI < OI_ridge): **memory-bound** — 메모리가 데이터를 공급하는 속도가 병목
- **능선점 오른쪽** (OI > OI_ridge): **compute-bound** — 연산 유닛의 처리 속도가 병목

---

### 5.2 LLM Decode가 극단적으로 memory-bound인 이유

Decode 1 step에서 일어나는 일:

```
토큰 1개 생성 = 모델 전체 가중치를 메모리에서 읽기 + 행렬-벡터 곱 1회
```

행렬-벡터 곱 `y = W × x` (W: [d, d], x: [d, 1]):
- **연산량:** 2d² FLOPs (곱셈 d² + 덧셈 d²)
- **데이터 이동:** d² × sizeof(element) Bytes (W 전체 로드)
- **OI = 2d² / (d² × sizeof) = 2/sizeof**

| 정밀도 | sizeof | OI |
|--------|--------|-----|
| FP32 | 4 bytes | 0.5 FLOP/byte |
| BF16 | 2 bytes | 1.0 FLOP/byte |
| INT8 | 1 byte | 2.0 FLOP/byte |

**H100의 능선점:**

```
π = 989 TFLOPS (BF16 Tensor Core)
β = 3.35 TB/s (HBM3)
OI_ridge = 989 / 3.35 = 295 FLOP/byte
```

**LLM decode의 OI(~1) vs 능선점(295) → 약 300배 차이!**

→ Decode는 극단적으로 memory-bound. H100의 거대한 연산 능력(989 TFLOPS)의 0.3%도 활용 못함.

**핵심 통찰:** 연산 능력이 아무리 높아도 의미 없고, **메모리 대역폭만이 decode 성능을 결정**한다. 따라서 CPU의 "약한 연산 능력"은 문제가 되지 않는다 — 대역폭만 있으면 된다.

---

### 5.3 T_decode 공식 유도

Decode가 순수 memory-bound이면, 성능은 오직 메모리 대역폭에 의해 결정:

```
1 토큰 생성 시간 = 모델 크기(bytes) / 메모리 대역폭
                = (P × b) / B_mem

역수를 취하면 초당 토큰 수:
T_decode = B_mem / (P × b)
```

여기서:
- P: 모델 파라미터 수 (예: 7B = 7 × 10⁹)
- b: 파라미터당 바이트 수 (BF16 = 2, INT8 = 1)
- B_mem: 메모리 대역폭 (bytes/sec)

**검증 예시:**
- 7B 파라미터, BF16, A10 GPU (600 GB/s)
- T = (7e9 × 2) / 600e9 = 14/600 = 23.3 ms/token → 실측과 일치 (Baseten 블로그)

---

### 5.4 T_CPU/T_GPU = B_CPU/B_GPU 유도

동일 모델(P, b가 같음)에 대해 CPU와 GPU를 비교:

```
T_CPU = B_CPU / (P × b)
T_GPU = B_GPU / (P × b)

T_CPU / T_GPU = [B_CPU / (P×b)] / [B_GPU / (P×b)]
              = B_CPU / B_GPU
```

**P×b가 소거된다!** 모델 크기에 무관하게 처리량 비율 = 대역폭 비율.

이것이 Roofline의 핵심 결과: memory-bound 영역에서는 성능 비율이 순수하게 대역폭 비율에 의해 결정된다.

---

### 5.5 구체적 수치: 하드웨어 스펙

#### Xeon 8480+ 메모리 대역폭

Intel 공식 사양 기준:

```
DDR5-4800:
  전송률 = 4800 MT/s (Mega Transfers/sec)
  채널 폭 = 64 bits = 8 bytes
  채널 수 = 8 (per socket)

소켓당 이론 대역폭 = 4800 × 10⁶ × 8 bytes × 8 channels = 307.2 GB/s

2소켓 시스템 이론: 307.2 × 2 = 614.4 GB/s
```

실측(STREAM Benchmark, 교토대학 평가):
- 2소켓 112코어 기준: ~490 GB/s (이론의 ~80%)

#### H100 HBM3 대역폭

NVIDIA 공식 사양:
- H100 SXM5: 80GB HBM3, **3.35 TB/s** (3,350 GB/s)
- 7B BF16 모델(14GB) 전체를 ~4.2ms에 읽을 수 있음
- 이론적 decode: ~238 tokens/second (batch=1)

---

### 5.6 단일 GPU 대비 비교

```
B_CPU = 307 GB/s   (Xeon 8480+ 1소켓)
B_GPU = 3,350 GB/s (H100 SXM5)

T_CPU / T_GPU = 307 / 3350 = 9.2%
```

→ "CPU가 GPU의 9.2%밖에 안 되는데 의미가 있나?"

---

### 5.7 TP=8 환경에서의 반전

H100 8장이 Tensor Parallelism으로 **하나의 모델을 함께 서빙**할 때:

```
모델을 8등분 → 각 GPU가 가중치의 1/8만 보유
→ 8장이 협력하여 1개 요청을 처리
→ 8장 전체가 1개 요청에 묶임
```

**"GPU 1장이 시스템에 기여하는 실효 대역폭":**

```
B_GPU_effective = B_GPU_single / TP = 3350 / 8 = 419 GB/s
```

왜 나누는가? 8장이 합쳐서 1개 요청을 처리하므로, 시스템 관점에서 GPU 1장의 "독립 스트림 기여분"은 1/8이다.

**반면 CPU는 완전히 독립적으로 별도 요청을 처리한다:**

```
CPU 대 GPU 1장 등가 = B_CPU / B_GPU_effective = 307 / 419 ≈ 73%
```

**의미:**
- CPU 하나가 GPU 시스템에서 "GPU 1장분"의 73%에 해당하는 **추가** 처리량을 제공
- 전체 시스템 관점: 73% × (1/8) = 약 9.1%의 총 처리량 증가

**왜 이 비교가 유효한가:**
- GPU 8장은 TP로 묶여서 **한 팀으로** 움직임 — 1개 요청에 8장 전부 점유
- CPU는 **독립적**으로 다른 요청을 처리 — GPU 리소스를 전혀 소비 안 함
- 따라서 CPU의 기여는 순수 **additive** (더해지기만 함, 빼지지 않음)

---

## 6. 참고 문헌

### LogP / LogGP

- Culler, Karp, Patterson et al., "LogP: towards a realistic model of parallel computation," Proc. 4th ACM PPoPP, pp. 1-12, 1993. [ACM DL](https://dl.acm.org/doi/10.1145/173284.155333)
- Alexandrov, Ionescu, Schauser, Scheiman, "LogGP: Incorporating Long Messages into the LogP Model for Parallel Computation," Proc. 7th ACM SPAA, pp. 95-105, 1995. [ACM DL](https://dl.acm.org/doi/10.1145/215399.215427)
- Kielmann et al., "Fast Measurement of LogP Parameters for Message Passing Platforms," Springer LNCS. [Link](https://link.springer.com/chapter/10.1007/3-540-45591-4_162)
- ETH Zurich DPHPC Lecture 12: LogGP Communication Models. [Link](https://spcl.inf.ethz.ch/Teaching/2015-dphpc/lecture/lecture12-loggp)

### Roofline Model

- Williams, Waterman, Patterson, "Roofline: an insightful visual performance model for multicore architectures," Communications of the ACM 52(4), pp. 65-76, 2009. [ACM DL](https://dl.acm.org/doi/abs/10.1145/1498765.1498785)
- NERSC Roofline Model Documentation. [Link](https://docs.nersc.gov/tools/performance/roofline/)
- Yuan et al., "LLM Inference Unveiled: Survey and Roofline Model Insights," arXiv:2402.16363. [arXiv](https://arxiv.org/html/2402.16363v4)

### LLM Inference Performance

- Baseten, "A guide to LLM inference and performance." [Link](https://www.baseten.co/blog/llm-transformer-inference-guide/)
- NVIDIA, "Mastering LLM Techniques: Inference Optimization." [Link](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- AMD ROCm, "Analyzing Impact of Tensor Parallelism." [Link](https://rocm.blogs.amd.com/artificial-intelligence/tensor-parallelism/README.html)

### 하드웨어 사양

- Intel Xeon Platinum 8480+ Specifications. [Intel](https://www.intel.com/content/www/us/en/products/sku/231746/intel-xeon-platinum-8480-processor-105m-cache-2-00-ghz/specifications.html)
- NVIDIA H100 GPU Product Page. [NVIDIA](https://www.nvidia.com/en-us/data-center/h100/)
- NVIDIA Hopper Architecture In-Depth. [NVIDIA Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- Kyoto Univ: Performance Evaluation of 4th Gen Xeon. [ACM DL](https://dl.acm.org/doi/fullHtml/10.1145/3636480.3637218)

### IPC / ZMQ Benchmarks

- Kamal Marhubi, "Early Linux IPC Latency Data," 2015. [Blog](https://kamalmarhubi.com/blog/2015/06/10/some-early-linux-ipc-latency-data/)
- ZeroMQ Performance Wiki. [Link](http://wiki.zeromq.org/whitepapers:measuring-performance)
- libzmq Issue #4673: High latency with intermittent messages. [GitHub](https://github.com/zeromq/libzmq/issues/4673)
- rigtorp/ipc-bench: Unix IPC Latency Benchmarks. [GitHub](https://github.com/rigtorp/ipc-bench)
