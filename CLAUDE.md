Answer to me always Korean
Write your task on CLAUDE.md file
# vLLM Hybrid 프로젝트 컨텍스트

## 프로젝트 개요
vLLM의 CPU/GPU 이기종(Heterogeneous) 실행 최적화 포크.
Intel Xeon + NVIDIA GPU 하이브리드 추론 환경에 최적화.

## 타겟 하드웨어
- **GPU**: NVIDIA H100
- **CPU**: Intel Xeon Platinum 8480+ (Sapphire Rapids)
- **RAM**: 2TB DDR5
- **특징**: AVX-512, AMX-BF16, NUMA 멀티소켓

---

## 현재 진행 중인 작업

### Intel CPU 최적화 (2026-02-03 완료)

#### 1. NUMA-aware KVCache 할당
- `vllm/platforms/intel_cpu_utils.py`: NUMA 유틸리티 모듈 신규 생성
- `vllm/v1/worker/cpu_model_runner.py`: `_allocate_kv_cache_tensors()` 오버라이드
- 효과: 2TB RAM의 NUMA 노드별 최적 메모리 배치

#### 2. AVX-512 / Inductor 최적화
- `vllm/platforms/cpu.py`: AVX-512, VNNI, BF16, AMX 자동 감지
- Inductor 설정: `simdlen=16`, `epilogue_fusion`, `max_autotune`
- Intel OpenMP 환경변수 최적화 (KMP_AFFINITY, KMP_BLOCKTIME 등)

#### 3. IPEX 자동 활성화
- `_use_ipex` 플래그로 자동 감지 (cpu_attn.py)
- `optimize_model_with_ipex()` 함수로 모델 최적화
- IPEX PagedAttention 커널 자동 선택

#### 4. 스레드 어피니티 개선
- NUMA 노드별 스레드 바인딩
- `_configure_threads_for_numa()`: 로컬 NUMA 코어 수 기반 설정
- OpenMP/MKL 최적 설정

---

## 주요 파일 구조

```
vllm/platforms/
├── intel_cpu_utils.py    # [신규] Intel CPU 최적화 유틸리티
├── cpu.py                # CPU 플랫폼 (AVX-512 감지 추가)
└── heterogeneous.py      # 이기종 플랫폼 통신

vllm/v1/worker/
├── cpu_worker.py         # CPU 워커 (NUMA/Intel 최적화 통합)
├── cpu_model_runner.py   # CPU 모델 러너 (NUMA KVCache 할당)
└── gpu_worker.py         # GPU 워커

vllm/v1/attention/backends/
└── cpu_attn.py           # CPU Attention (IPEX PagedAttention)
```

---

## 기술적 결정사항

| 항목 | 결정 | 이유 |
|------|------|------|
| Inductor 컴파일 | DYNAMO_ONCE | 안정성 + 성능 균형 |
| KVCache 할당 | NUMA-aware | 메모리 대역폭 최적화 |
| 스레드 수 | NUMA 노드별 코어 수 | 캐시 지역성 |
| IPEX | 자동 활성화 | 최적화된 커널 사용 |
| Gloo 백엔드 | CPU↔GPU 통신 | NCCL 미지원 환경 |

---

## 이기종 실행 병목점 (분석 완료)

### 핵심 병목점
1. **KV Layer 블로킹** - attention/layer.py (파이프라이닝 필요)
2. **RPC Collective 동기화** - multiproc_executor.py (병렬화 필요)
3. **Gloo 디바이스 전송** - CPU 경유 오버헤드

### 향후 최적화 방향
- KV 전송 파이프라이닝
- GPU/CPU 병렬 실행
- NCCL 선택적 사용

---

## 환경 변수 설정 (권장)

```bash
# Intel MKL/OpenMP
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export KMP_TPAUSE=0
export MKL_ENABLE_INSTRUCTIONS=AVX512

# vLLM CPU KVCache (예: 512GB)
export VLLM_CPU_KVCACHE_SPACE=512

# NUMA 바인딩 (자동)
export VLLM_CPU_OMP_THREADS_BIND=auto
```

---

## 테스트 방법

```bash
# 모듈 테스트 (CPU 기능 자동 감지)
python -c "
from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features, setup_intel_cpu_environment
features = detect_intel_cpu_features()
print(f'CPU: {features.model_name}')
print(f'AVX2: {features.avx2}, AVX-512: {features.avx512f}, AMX: {features.amx_bf16}')
"

# IPEX 설치 확인
python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"

# CPU 기능 확인 (Linux)
cat /proc/cpuinfo | grep -E "avx512|amx|avx2"

# NUMA 토폴로지 확인
numactl --hardware
```

---

## 호환성

| 환경 | AVX | NUMA | IPEX | 상태 |
|------|-----|------|------|------|
| H100 서버 (Xeon 8480+) | AVX-512/AMX | 멀티노드 | 지원 | 최적화 |
| 개발 머신 (i9-12900KF) | AVX2/VNNI | 단일노드 | 미설치 | 테스트됨 |
| 일반 x86_64 | AVX2 | - | - | 호환 |

코드는 모든 환경에서 graceful fallback으로 동작함.

---

## 변경 이력

### 2026-02-03: CPU PagedAttention 토큰-시퀀스 불일치 수정

#### 문제
`forward_decode()`에서 `num_tokens (42) != num_seqs (41)` 불일치로 인한 텐서 크기 오류

#### 해결
- `context_lens.shape[0]`으로 실제 시퀀스 수 확인
- `num_tokens != num_seqs`일 때 loop 기반 fallback 구현
- 자세한 내용: [`docs/HETEROGENEOUS_CPU_OPTIMIZATIONS.md`](docs/HETEROGENEOUS_CPU_OPTIMIZATIONS.md)

### 2026-02-03: Heterogeneous 플랫폼 수정

#### 완료된 수정
1. **플랫폼 감지 순서 수정** (`vllm/platforms/__init__.py`)
   - `heterogeneous` 플러그인을 맨 앞으로 이동
   - `VLLM_HETEROGENEOUS_PLATFORM=1` 환경변수로 활성화

2. **V1 엔진 지원** (`vllm/platforms/heterogeneous.py`)
   - `supports_v1()` 메서드 추가
   - `get_device_capability()` 메서드 추가 (Flash Attention 버전 감지용)
   - Lazy 초기화로 조기 CUDA 초기화 방지

3. **CPU Attention 직접 호출 구현** (`vllm/attention/layer.py`)
   - 런타임: `query.device.type == "cpu"` 체크 추가
   - CPU 텐서일 경우 CUDA 전용 커스텀 op (`vllm::unified_attention`) 우회

4. **IPEX 오류 처리 개선**
   - 모든 파일에서 예외 처리 강화
   - import 실패 시 안전한 fallback

#### 테스트 명령어
```bash
VLLM_HETEROGENEOUS_PLATFORM=1 python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-6.7b \
  --device heterogeneous \
  --pipeline-parallel-size 2
```

### 2026-02-03: 호환성 개선
- AVX-512 없는 환경 지원 (AVX2 fallback)
- libnuma 미설치 환경 지원
- IPEX 선택적 사용 (미설치시 표준 PyTorch)
- Intel Core i9-12900KF에서 테스트 완료

### 2026-02-03: Intel CPU 최적화
- NUMA-aware KVCache 할당
- AVX-512/Inductor 최적화
- IPEX 자동 활성화
- 스레드 어피니티 개선

---

*마지막 업데이트: 2026-02-03*
*작업자: Claude (Heterogeneous 플랫폼 수정)*
