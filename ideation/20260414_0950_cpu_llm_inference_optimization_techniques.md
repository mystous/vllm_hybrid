# CPU LLM 추론 최적화 기법 상세 조사

**작성일**: 2026-04-13  
**대상 하드웨어**: Intel Xeon 8480+ (Sapphire Rapids, 96 vCPU, AMX BF16/INT8, AVX-512 VNNI/BF16, DDR5-4800 8ch)  
**참조 프레임워크**: vLLM (IPEX 백엔드), KTransformers, SGLang CPU 백엔드, llama.cpp, OpenVINO

---

## 1. 하드웨어 연산 유닛별 특성과 최적 활용 조건

### 1.1 AMX (Advanced Matrix Extensions)

Sapphire Rapids에 탑재된 AMX는 **타일 기반 행렬 곱셈 가속기**이다. 8개의 1KB 타일 레지스터(각 16행 × 64바이트)와 TMUL(Tile Matrix Multiply Unit) 연산기로 구성된다.

**지원 데이터 타입:**
- BF16: 16×32 × 32×16 = 16×16 결과 (1 tdpbf16ps 명령에 512 FMA)
- INT8: 16×64 × 64×16 = 16×16 결과 (1 tdpbssd 명령에 1024 MAC)

**이론 피크 (Xeon 8480+, 56코어 단일 소켓, ~2.0GHz AMX 주파수):**
- BF16: ~112 TFLOPS
- INT8: ~224 TOPS

**실측 성능 (주요 출처 종합):**

| 환경 | BF16 TFLOPS | INT8 TOPS | 피크 대비 | 출처 |
|------|-----------|----------|---------|------|
| KTransformers (8452Y, 36코어) | 21.3 | 35 | 19% / 16% | SOSP'25 |
| SGLang CPU (6980P, 128코어×2) | ~51 (추정) | — | ~23% | LMSYS 블로그 |
| SparAMX (6430L, 32코어) | — | ~15 (추정) | — | arXiv 2502.12444 |
| LIA 마이크로벤치마크 (SPR) | — | — | GEMM: A100의 11% | ISCA'25 |

**AMX가 피크에 도달하지 못하는 5가지 원인:**

1. **메모리 대역폭 병목**: AMX 연산 속도(112 TFLOPS)가 DDR5 데이터 공급 속도(~250GB/s)를 압도. 균형점(Balance Point) = 112T / 0.25T = **448 Ops/Byte**인데, LLM 디코드는 1~2 Ops/Byte에 불과
2. **주파수 스로틀링**: AMX 워크로드 시 터보 3.8GHz → ~2.0~2.5GHz로 다운클럭. 이는 전력 제한(TDP 350W)에 의한 것
3. **타일 레지스터 제약**: 8개 타일(총 8KB)만 사용 가능. L1 캐시(48KB)와 L2 캐시(코어당 2MB) 사이의 데이터 재사용이 제한적
4. **AVX-512 ↔ AMX 전환 불가**: zmm 레지스터와 타일 레지스터 간 직접 데이터 전송이 없어, 메모리를 경유해야 함
5. **코어 간 동기화 오버헤드**: OMP 배리어, 캐시 코히어런스 비용

**핵심 통찰 — AMX vs AVX-512 동적 전환:**

KTransformers는 산술 강도가 낮은 디코드 단계에서 AVX-512 커널로 동적 전환하여 AMX 대비 최대 2.22× 속도 향상을 보고했다. 이는 디코드(배치=1)에서 AMX의 높은 초기화 비용과 주파수 스로틀링이 오히려 성능을 저하시키기 때문이다.

| 추론 단계 | 산술 강도 | 최적 ISA | 이유 |
|----------|---------|---------|------|
| 프리필 (긴 시퀀스) | 50~500 Ops/Byte | **AMX BF16** | 연산 병목, 타일 곱셈 유리 |
| 디코드 (배치=1) | 1~2 Ops/Byte | **AVX-512** | 메모리 병목, 낮은 레이턴시 우선 |
| 디코드 (배치≥8) | 8~32 Ops/Byte | **AMX INT8** | 배치가 커지면 연산 비중 증가 |

### 1.2 AVX-512 VNNI / BF16

AVX-512는 512비트(64바이트) 벡터 레지스터 32개를 사용하며, VNNI(Vector Neural Network Instructions)로 INT8 dot-product를 가속한다.

**LLM 추론에서의 역할:**
- **디코드 단계(배치=1)**: AMX보다 효율적. 타일 초기화 오버헤드 없음, 주파수 스로틀링 없음
- **어텐션 커널**: IPEX의 paged attention이 AVX-512로 구현 (`custom_avx` 경로)
- **비벡터 연산**: RMSNorm, SiLU, Softmax 등 원소별 연산은 AVX-512가 담당

LIA 논문에서 측정한 GEMV(행렬-벡터 곱) 처리량을 보면, SPR에서 AMX와 AVX-512의 성능 차이가 10% 미만이다. 이는 GEMV가 완전 메모리 병목이며, 두 ISA가 동일한 메모리 대역폭(260GB/s)을 공유하기 때문이다.

### 1.3 DDR5 메모리 대역폭

**이론 대역폭**: DDR5-4800 × 8채널 = **307.2 GB/s** (단일 소켓)

**실측 대역폭:**

| 측정 도구 | 결과 | 피크 대비 |
|----------|------|---------|
| Intel MLC (8452Y) | 220 GB/s | 72% |
| STREAM Triad (8480+) | ~200 GB/s | 65% |
| 교차 소켓 (2-socket) | 125 GB/s | 41% |
| SGLang MoE 커널 (6980P, DDR5-12800) | **85% 메모리 대역폭 효율** 달성 | 85% |

**실효 달성 가능 대역폭**: ~200~260 GB/s (단일 소켓, 최적화 시)

---

## 2. 가중치 양자화 — CPU 추론의 가장 직접적인 가속

### 2.1 메모리 병목과 양자화의 관계

LLM 디코드는 **완전 메모리 대역폭 병목**이다. 매 토큰 생성 시 전체 모델 가중치를 메모리에서 읽어야 하므로:

```
이론 tok/s = 메모리 대역폭 / 모델 가중치 크기
```

| 모델 | BF16 크기 | INT8 크기 | INT4 크기 | BF16 tok/s | INT8 tok/s | INT4 tok/s |
|------|---------|---------|---------|-----------|-----------|-----------|
| 7B | 14 GB | 7 GB | 3.5 GB | 18 | **36** | **71** |
| 32B | 64 GB | 32 GB | 16 GB | 3.9 | **7.8** | **15.6** |
| 70B | 140 GB | 70 GB | 35 GB | 1.8 | **3.6** | **7.1** |

*250 GB/s 실효 대역폭 기준, KV 캐시/어텐션 오버헤드 미포함*

### 2.2 INT8 WoQ (Weight-Only Quantization)

**IPEX LLM 경로 (현재 vLLM Hybrid에서 가장 빠른 적용):**

```python
from intel_extension_for_pytorch.quantization import WoqWeightDtype
qconfig = ipex.quantization.default_weight_only_quant_qconfig_mapping(
    weight_dtype=WoqWeightDtype.INT8
)
model = ipex.llm.optimize(model, dtype=torch.bfloat16, quantization_config=qconfig)
```

- 가중치만 INT8로 저장, 연산은 BF16으로 수행 (역양자화 후 GEMM)
- AMX INT8 타일 연산 직접 활용 가능 (`brg_matmul:avx10_1_512_amx` INT8 경로)
- 정확도 열화: 일반적으로 PPL < +0.5 (Llama-2-7B 기준)

### 2.3 INT4 양자화

AMX는 **네이티브 INT4를 지원하지 않는다**. INT4 가중치는 메모리에서 INT4로 저장되지만 연산 전에 INT8로 캐스트해야 한다.

**접근법별 비교:**

| 방법 | 저장 정밀도 | 연산 정밀도 | 역양자화 위치 | 성능 영향 |
|------|-----------|-----------|-------------|---------|
| IPEX WoQ INT4 | INT4 | BF16 | CPU 커널 내 | 대역폭 2× 절감, 역양자화 오버헤드 |
| llama.cpp Q4_0 | INT4 | FP32 | GEMM 내 융합 | **가장 최적화된 경로** |
| GPTQ INT4 | INT4 | FP16 | 그룹별 역양자화 | 그룹 크기에 따라 다름 |
| GGUF Q4_K_M | 4.5-bit 혼합 | FP32 | 블록별 | 정확도-속도 균형 최적 |

llama.cpp에서 듀얼 6980P(128코어×2)로 DeepSeek-R1 671B Q4_K_M을 실행 시 약 6.2~10.5 tok/s가 보고되었다.

### 2.4 MXFP4 / FP8 — 차세대 양자화

KTransformers는 2025년 4월에 AMX-INT8, AMX-BF16 지원을 추가했고, FP8 per-channel 정밀도도 지원한다. Granite Rapids에서는 FP8 AMX 지원이 예상되며, 이는 INT4 대비 역양자화 오버헤드 없이 2× 메모리 절감을 달성할 수 있다.

---

## 3. 메모리 계층 최적화 — 대역폭을 최대로 끌어내기

### 3.1 NUMA 인식 메모리 배치

**문제**: 교차 NUMA 접근 시 대역폭이 220GB/s → 125GB/s로 **43% 감소**.

**최적화 전략:**

1. **소켓별 텐서 분할**: KTransformers는 모든 전문가의 가중치 행렬을 소켓 간 열/행 기준으로 분할하여, 각 소켓이 로컬 메모리의 자기 슬라이스만 접근하도록 한다. 이 접근으로 듀얼 소켓에서 디코딩 처리량이 최대 1.63× 향상되었다.

2. **SGLang의 공유 메모리 기반 통신**: SGLang은 torch.distributed를 사용하지 않고 공유 메모리 기반 all-reduce/all-gather를 구현하여, 통신 오버헤드를 전체 시간의 3%로 줄였다.

3. **현재 vLLM Hybrid의 NUMA 처리**: `_get_autobind_cpu_ids`가 NUMA 노드별 코어를 선택하고, `numa_set_membind + numa_set_strict(1)`로 로컬 메모리 바인딩을 수행한다. 단, H100x4 KVM은 단일 NUMA이므로 현재 효과 없음.

### 3.2 Huge Pages

**문제**: 70B INT4(~35GB) 기준 4KB 페이지 → ~900만 TLB 엔트리 필요 → 심각한 TLB 미스.

| 페이지 크기 | 70B INT4 페이지 수 | TLB 압력 |
|-----------|----------------|---------|
| 4 KB | ~9,175,040 | 극심 |
| 2 MB | ~17,920 | 낮음 |
| 1 GB | ~35 | **무시 가능** |

**적용 방법:**

```bash
# 부트 타임 1GB 거대 페이지 사전 할당
sudo grubby --update-kernel=ALL --args="hugepagesz=1G hugepages=40 default_hugepagesz=1G"

# 런타임 2MB 거대 페이지
echo 20000 > /proc/sys/vm/nr_hugepages

# Python에서 활용 (mmap)
import mmap
fd = os.open("/dev/hugepages/model_weights", os.O_RDWR | os.O_CREAT)
mm = mmap.mmap(fd, model_size, flags=mmap.MAP_HUGETLB | mmap.MAP_HUGE_1GB)
```

**예상 효과**: 5~15% 처리량 향상 (TLB 미스 감소에 의한 메모리 접근 지연 개선)

### 3.3 소프트웨어 프리페칭

**KTransformers의 프리페치 전략:**

1. **가중치 블록 프리페치**: 현재 레이어의 GEMM 실행 중 다음 레이어의 가중치 블록을 L2 캐시로 프리페치
2. **지그재그 순회 패턴**: 가중치 블록을 순차가 아닌 타일링 패턴으로 접근하여 L2 캐시 활용 극대화
3. **KV 캐시 LUT 프리페치**: 어텐션 계산 시 다음에 접근할 KV 블록 주소를 LUT에서 미리 로드

**Intel 하드웨어 프리페처 설정:**

```bash
# DCU 프리페처와 Adjacent Line 프리페처 활성화 확인
rdmsr 0x1A4  # bit 0: DCU HW, bit 1: DCU IP, bit 2: Adjacent Line, bit 3: L2 HW
# 모두 0이면 전부 활성 (기본값)
```

### 3.4 캐시 라인 정렬

AMX 타일 데이터는 **64바이트 캐시 라인 정렬**이 필수이다.

KTransformers의 AMX 타일링 인식 메모리 레이아웃은 블록별 양자화, 64바이트 정렬, 타일링 인식 부분행렬 접근 패턴을 포함하며, 캐시 지역성과 메모리 대역폭 요구를 크게 개선한다.

**WOQ Aware Cache Blocking 패턴:**
- INT4 가중치를 64바이트 정렬된 블록으로 패킹
- 역양자화 후 BF16 타일을 L2 캐시(코어당 2MB)에 캐싱
- 캐시된 BF16 타일을 AMX TMUL이 반복 참조

---

## 4. 연산자 융합 — 커널 호출과 메모리 왕복 줄이기

### 4.1 GEMM + 역양자화 융합

양자화된 가중치의 역양자화와 GEMM을 단일 커널로 융합하면 중간 BF16 텐서의 메모리 기록/읽기를 제거한다.

```
[기존] INT4 로드 → INT8 캐스트 → BF16 역양자화 → 메모리 저장 → BF16 로드 → GEMM
[융합] INT4 로드 → 레지스터 내 역양자화 → 즉시 GEMM (메모리 왕복 0회)
```

**KTransformers의 구현**: oneDNN의 `brgemm` 커널이 INT8→BF16 역양자화와 타일 곱셈을 융합

### 4.2 SiLU + Gate + Up Projection 융합

SwiGLU MLP는 세 단계로 구성된다:
```
gate = W_gate × x
up = W_up × x
output = SiLU(gate) * up
```

**개별 실행 시** 각 단계마다 입력 `x`를 메모리에서 재로드. **융합 시** `x`를 한 번 로드하여 세 연산을 연속 수행.

SGLang은 SiLU + up_proj GEMM 융합으로 중간 로드/스토어를 제거하고, KV 버퍼 세팅 융합으로 12% 개선을 달성했다.

### 4.3 Flash Attention CPU 변형

**IPEX의 CPU Flash Attention** (`flash_attn_varlen_func`):
- 프리필 시 사용 (청크 프리필)
- Q, K, V를 타일 단위로 분할하여 L2 캐시에 맞춤
- Softmax 스케일링과 출력 누적을 단일 패스로 수행

**NEO의 PACPU 어텐션 커널**:
- Intel ISPC(Implicit SPMD Program Compiler) 기반
- AVX-512로 컴파일되어 디코드 어텐션을 CPU에서 수행
- vLLM의 PagedAttention과 동일한 블록 테이블 인터페이스

### 4.4 RMSNorm + Residual 융합

```
[기존] x = x + attention_output → 메모리 저장 → 메모리 로드 → RMSNorm(x) → 메모리 저장
[융합] fused_add_rms_norm(x, attention_output) → 한 번의 읽기/쓰기
```

vLLM의 `_custom_ops.py`에서 CPU fallback 구현 (`fused_add_rms_norm`)이 존재한다.

---

## 5. 스레딩과 동기화 최적화

### 5.1 OMP 스레드 바인딩

**현재 vLLM Hybrid의 구현:**

```cpp
// csrc/cpu/utils.cpp::init_cpu_threads_env
#pragma omp parallel for schedule(static, 1)
for (int i = 0; i < num_threads; i++) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpuids[i], &mask);
    sched_setaffinity(0, sizeof(mask), &mask);  // 1:1 핀
}
```

이것이 올바른 접근이다. 핵심 원칙:

- **물리 코어만 사용**: SMT(HyperThreading)의 논리 코어는 AMX 타일을 공유하므로, 물리 코어만 사용하는 것이 AMX 성능에 유리
- **1:1 핀**: OS 스케줄러의 코어 마이그레이션을 방지하여 L2 캐시 워밍 유지
- **static 스케줄링**: 동적 스케줄링의 부하 분산은 불필요 (모든 코어가 동일 작업)

### 5.2 OMP 환경변수 최적화

```bash
# 필수 설정
export OMP_NUM_THREADS=48          # 물리 코어 수 (96 vCPU / 2 = 48 코어)
export OMP_PROC_BIND=close         # 인접 코어에 스레드 바인딩
export OMP_PLACES=cores            # 물리 코어 단위 배치

# 선택적 성능 튜닝
export OMP_SCHEDULE=static         # 정적 작업 분배
export OMP_WAIT_POLICY=active      # 스핀 웨이트 (지연 최소화, 전력 증가)
export KMP_BLOCKTIME=1             # Intel OpenMP: 배리어 후 1ms 스핀 후 슬립
export KMP_AFFINITY=granularity=fine,compact,1,0  # Intel 전용 세밀 바인딩
```

### 5.3 interop 스레드와 intra-op 스레드 분리

```python
torch.set_num_threads(48)           # intra-op: GEMM 등 단일 연산 내 병렬화
torch.set_num_interop_threads(2)    # inter-op: 독립 연산 간 병렬화
```

**주의**: `set_num_interop_threads`는 첫 연산 실행 전에만 호출 가능. 이후 호출 시 `RuntimeError`. 현재 vLLM Hybrid에서 `try/except`로 감싸져 있음.

---

## 6. ISA별 커널 최적화 — 프레임워크별 비교

### 6.1 KTransformers의 AMX 커널 (SOSP'25)

KTransformers는 AMX 최적화 커널로 단일 Xeon 소켓에서 최대 21.3 TFLOPS의 지속 처리량을 달성했으며, 이는 PyTorch 네이티브 구현 대비 3.9× 빠르다.

**핵심 최적화 기법:**

1. **AMX 타일링 인식 메모리 레이아웃**: 가중치를 AMX 타일(16×64 바이트) 크기에 맞춰 사전 패킹. 런타임 재배치 오버헤드 제거
2. **산술 강도 기반 ISA 동적 전환**: 프리필(높은 ARI) → AMX, 디코드(낮은 ARI) → AVX-512
3. **CUDA Graph 연동**: GPU-CPU 간 커널 실행 동기화를 CUDA Graph로 관리, 런칭 오버헤드를 거의 0으로 축소

### 6.2 SGLang CPU 백엔드 (Intel 기여)

SGLang은 BF16, INT8, FP8을 Dense FFN과 Sparse FFN(MoE) 모두에서 지원하며, TTFT 6~14× 가속, TPOT 2~4× 가속을 llama.cpp 대비 달성했다.

**핵심 최적화:**
- 공유 메모리 기반 NUMA 간 all-reduce (torch.distributed 회피)
- MoE 커널의 85% 메모리 대역폭 효율
- SiLU + up_proj GEMM 융합

### 6.3 llama.cpp AMX 지원

llama.cpp의 AMX 지원은 PR #6341에서 시작되어, 현재 `ggml-cpu/amx/` 디렉토리에 구현되어 있다.

- Q4_0, Q8_0 등 GGUF 양자화 형식에 대해 AMX INT8 GEMM 경로 제공
- `GGML_OP_MUL_MAT` 시 AMX 플랫폼 감지 후 자동 dispatch
- **한계**: 현재 16×64 바이트 타일만 사용, 최적 128×128 블록 타일링 미구현

### 6.4 OpenVINO CPU 최적화

OpenVINO는 AMX 가속 희소 INT8 실행을 지원하며, `brgemm_avx512_amx_sparse_I8` 실행 타입으로 희소 가중치 분해와 AMX GEMM을 결합한다.

- INT8 모델 + 구조적 희소성 시 AMX 자동 dispatch
- 비구조적 희소성은 미지원 → SparAMX가 이를 보완

### 6.5 SparAMX (Intel Labs, arXiv 2502.12444)

SparAMX는 비구조적 희소성을 AMX 타일에 적용하여, Llama-3 8B에서 리니어 레이어 1.42× 속도 향상, 어텐션 1.14× 속도 향상을 달성했다.

**핵심 기법**: 가중치의 비구조적 희소성을 마스크로 인코딩하고, AVX-512에서 마스크를 확장한 후 메모리에 기록, AMX가 이를 읽어 희소 GEMM을 수행. 현재 한계는 AVX→AMX 직접 전송이 불가능하여 메모리 경유가 필요한 점.

---

## 7. 어텐션 커널 최적화 — CPU 특화

### 7.1 IPEX PagedAttention

현재 vLLM Hybrid CPU 경로의 기본 어텐션 커널이다.

**디코드 커널 선택 체인:**
```
1순위: custom_avx (AVX-512F 빌드 시) → batch16 AVX-512 paged attn
2순위: IPEX oneDNN → ipex_modules.PagedAttention.single_query_cached_kv_attention
3순위: torch SDPA batched
4순위: 순차 Python 루프
```

H100x4 (SPR 8480+)에서는 100% `ipex` 경로로 실행됨이 실측 확인됨 (`path=ipex`, 3500회).

### 7.2 CPU 어텐션의 대역폭 한계

디코드 어텐션의 연산량 분석 (7B, GQA 8 KV heads, batch=1):

```
레이어당 KV 접근량 = 2 × seq_len × num_kv_heads × head_dim × sizeof(BF16)
                    = 2 × 2048 × 8 × 128 × 2 = 8.4 MB

32 레이어 합계 = 268 MB

DDR5 250 GB/s 기준 어텐션 시간 = 268 MB / 250 GB/s ≈ 1.1 ms
```

이 시간은 전체 디코드 스텝에서 리니어 GEMM(가중치 14GB / 250GB/s ≈ 56ms) 대비 2%에 불과. **어텐션은 CPU 디코드의 병목이 아니다** — 리니어(MLP/FFN) 연산이 95%+ 시간을 차지.

### 7.3 ScoutAttention의 CPU 어텐션 처리량 발견

ScoutAttention은 CPU 어텐션 처리량(~100GB/s)이 PCIe 전송 처리량(~15GB/s)보다 높다는 핵심 관찰을 보고했으며, GPU가 유휴인 시간이 HGCA에서 57%에 달하는 반면 ScoutAttention에서는 5% 미만으로 줄었다.

이 발견은 **KV 캐시를 GPU로 전송하는 대신 CPU에서 직접 어텐션을 계산**하는 것이 더 효율적임을 의미한다. 96코어 Xeon 8480+에서는 ~150~200 GB/s의 실효 어텐션 처리량이 기대된다.

---

## 8. 모델 로딩 및 초기화 최적화

### 8.1 가중치 로드 병렬화

70B 모델의 가중치(~35GB INT4)를 CPU DRAM에서 로드하는 시간:

- 순차 로드: ~35GB / 2GB/s(디스크) ≈ 17.5초
- 병렬 로드 (mmap + 4 스레드): ~4.4초
- **NVMe SSD 직접 로드**: ~35GB / 7GB/s(NVMe Gen4) ≈ 5초

### 8.2 모델 사전 패킹

KTransformers는 모델 가중치를 AMX 타일 레이아웃에 맞춰 **사전 변환**하여 저장한다:

```bash
# 사전 변환 (한 번만 실행)
python convert_weights.py --model Qwen2.5-32B --output-format amx_int4 --output-path ./amx_weights/

# 추론 시 변환 없이 즉시 로드
--kt-amx-weight-path ./amx_weights/
```

이렇게 하면 런타임 재배치 오버헤드가 완전히 제거된다.

---

## 9. 프리필 단계 특화 최적화

프리필은 디코드와 달리 **연산 집약적**이다 (산술 강도 50~500 Ops/Byte). 여기서 AMX BF16이 진가를 발휘한다.

### 9.1 Chunked Prefill

긴 프롬프트를 청크(예: 512 토큰)로 나누어 처리한다:
- 각 청크가 L2 캐시에 맞는 크기로 분할
- 청크 간 KV 캐시가 점진적으로 누적
- GPU에서는 디코드 요청과 프리필 청크가 같은 배치에 공존 가능 (Sarathi-Serve)

### 9.2 AMX 프리필 성능

KTransformers는 프리필 단계에서 AMX 최적화 CPU 커널로 llama.cpp 대비 최대 20× 속도 향상을 달성했다.

이는 프리필의 높은 산술 강도에서 AMX의 타일 곱셈이 효과를 발휘하기 때문이다. 디코드 단계(1~2 Ops/Byte)와 달리 프리필(50+ Ops/Byte)에서는 AMX의 112 TFLOPS 피크에 훨씬 가까이 접근할 수 있다.

---

## 10. 종합: Dense 모델 CPU 추론 최적화 체크리스트

현재 vLLM Hybrid의 CPU EngineCore에 적용 가능한 최적화를 우선순위별로 정리한다:

| 우선순위 | 기법 | 구현 비용 | 예상 효과 | 비고 |
|---------|------|---------|---------|------|
| **1** | INT8 WoQ (IPEX 한 줄) | 2~3일 | 디코드 **2× 가속** | 가장 즉각적 |
| **2** | OMP 환경변수 최적화 | 1일 | 5~15% | OMP_PROC_BIND=close, OMP_PLACES=cores |
| **3** | ISA 동적 전환 (AMX↔AVX-512) | 2주 | 디코드 **1.5~2.2×** | KTransformers 참조 |
| **4** | 2MB Huge Pages | 1일 | 5~15% | /proc/sys/vm/nr_hugepages |
| **5** | SiLU+Gate+Up 융합 | 2주 | 10~15% | IPEX/oneDNN 커널 |
| **6** | 가중치 AMX 사전 패킹 | 1주 | 10~20% | 런타임 재배치 제거 |
| **7** | INT4 WoQ | 2주 | 추가 **2× 가속** | AMX→INT8 캐스트 오버헤드 |
| **8** | NUMA 인식 텐서 분할 | 3주 | 듀얼소켓 **1.6×** | 2-socket 환경 전용 |
| **9** | SparAMX 희소 GEMM | 4주 | 추가 **1.4×** | 가중치 프루닝 필요 |
| **10** | KV 캐시 프리페치 | 2주 | 5~10% | 어텐션 비중 작으므로 효과 제한적 |

**1~4를 모두 적용하면 BF16 대비 ~4~6× CPU 디코드 가속이 가능**하며, 이는 GPU 대비 CPU 속도 격차를 18× → 3~5× 수준으로 줄인다. 여전히 GPU보다 느리지만, GPU 큐가 포화되는 고부하 시나리오에서 Property 2 gate가 일부 요청을 CPU로 보내기 시작하는 영역에 진입한다.
