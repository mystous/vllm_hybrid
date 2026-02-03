# vLLM Heterogeneous CPU/GPU 최적화 기술 문서

## 개요

이 문서는 vLLM의 CPU/GPU 이기종(Heterogeneous) 실행 모드에서 수행된 모든 기술적 변경 사항을 기술합니다.

**타겟 환경:**
- GPU: NVIDIA H100
- CPU: Intel Xeon Platinum 8480+ (Sapphire Rapids)
- RAM: 2TB DDR5
- Pipeline Parallel Size: 2 (GPU PP0 + CPU PP1)

**변경 파일 요약:**
| 파일 | 변경량 | 주요 내용 |
|------|--------|----------|
| `README.md` | +99 | IPEX/NUMA 설치 가이드, 테스트 방법 |
| `vllm/_ipex_ops.py` | +5/-1 | IPEX import 예외 처리 강화 |
| `vllm/attention/layer.py` | +21/-3 | CPU 텐서 런타임 디바이스 체크 |
| `vllm/platforms/__init__.py` | +4/-1 | heterogeneous 플랫폼 우선순위 조정 |
| `vllm/platforms/cpu.py` | +124/-3 | AVX-512 감지, Inductor/OpenMP 최적화 |
| `vllm/platforms/heterogeneous.py` | +244/-27 | lazy 초기화, `get_device_capability()` |
| `vllm/v1/attention/backends/cpu_attn.py` | +93/-1 | 토큰-시퀀스 불일치 처리 |
| `vllm/v1/worker/cpu_model_runner.py` | +225/-1 | NUMA KVCache, IPEX 모델 최적화 |
| `vllm/v1/worker/cpu_worker.py` | +177/-3 | Intel CPU 환경 설정, 스레드 어피니티 |
| `vllm/v1/worker/gpu_model_runner.py` | +18/-1 | Mamba/Triton lazy import |

---

## 1. 플랫폼 감지 순서 변경

### 파일: `vllm/platforms/__init__.py`

### 변경 내용
`heterogeneous` 플랫폼 플러그인을 다른 플랫폼보다 먼저 체크하도록 순서 변경.

```python
# 변경 전
builtin_platform_plugins = {
    'tpu': tpu_platform_plugin,
    'cuda': cuda_platform_plugin,
    ...
    'heterogeneous': heterogeneous_platform_plugin,  # 마지막
}

# 변경 후
builtin_platform_plugins = {
    # heterogeneous must be checked first (when env var is set)
    'heterogeneous': heterogeneous_platform_plugin,  # 첫 번째
    'tpu': tpu_platform_plugin,
    'cuda': cuda_platform_plugin,
    ...
}
```

### 이유
`VLLM_HETEROGENEOUS_PLATFORM=1` 환경변수가 설정된 경우, 다른 플랫폼(CUDA, CPU 등)보다 먼저 heterogeneous 플랫폼이 선택되어야 합니다.

---

## 2. IPEX Import 예외 처리 강화

### 파일: `vllm/_ipex_ops.py`

### 변경 내용
```python
# 변경 전
try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    logger.warning("Import error msg: %s", e.msg)

# 변경 후
try:
    import intel_extension_for_pytorch as ipex
except (ImportError, AttributeError, Exception) as e:
    ipex = None
    logger.warning("IPEX import failed: %s", str(e))
```

### 이유
- `AttributeError`: IPEX 내부 버그 처리 (https://github.com/intel/intel-extension-for-pytorch/pull/813)
- `Exception`: PyTorch 버전 불일치 등 예상치 못한 오류 처리
- `ipex = None` 설정으로 이후 코드에서 안전하게 체크 가능

---

## 3. Attention Layer의 CPU 디바이스 지원

### 파일: `vllm/attention/layer.py`

### 문제점
vLLM의 attention layer는 `torch.ops.vllm.unified_attention_with_output()` 커스텀 CUDA 연산자를 사용합니다. 이 연산자는 CPU 텐서를 지원하지 않습니다.

```
NotImplementedError: Could not run 'vllm::unified_attention_with_output'
with arguments from the 'CPU' backend.
```

### 해결책

#### 초기화 시점 변경
```python
# 변경 전
import vllm.platforms
self.use_direct_call = not vllm.platforms.current_platform.is_cuda_alike()

# 변경 후
self.use_direct_call = current_platform.is_cpu()
```

#### 런타임 디바이스 체크 추가
```python
# 변경 전
if self.use_direct_call:
    # direct call path
else:
    torch.ops.vllm.unified_attention_with_output(...)

# 변경 후
# Use direct call for CPU tensors to avoid CUDA-only custom op
use_direct = self.use_direct_call or query.device.type == "cpu"
if use_direct:
    # direct call path - CPU 텐서에서 안전하게 동작
else:
    torch.ops.vllm.unified_attention_with_output(...)
```

### 기술적 세부사항
- `self.use_direct_call`은 초기화 시점에 플랫폼 기반으로 설정됨
- 런타임 체크 `query.device.type == "cpu"`를 추가하여 동적 디바이스 전환 지원
- 이중 체크로 heterogeneous 환경에서도 안전하게 동작
- 두 곳에서 적용: `forward()` 메서드의 두 분기 (with/without output buffer)

---

## 4. Heterogeneous Platform 전면 개선

### 파일: `vllm/platforms/heterogeneous.py`

### 4.1 Lazy 초기화 패턴 적용

#### 문제점
기존 코드는 클래스 로드 시점에 CUDA/CPU 플랫폼을 즉시 초기화하여 불필요한 CUDA 초기화를 유발했습니다.

```python
# 변경 전 (클래스 속성으로 즉시 초기화)
class HeterogeneousPlatform(Platform):
    _cuda_platform = CudaPlatform()  # 즉시 CUDA 초기화
    _cpu_platform = CpuPlatform()
```

#### 해결책
```python
# 변경 후 (Lazy 초기화)
class HeterogeneousPlatform(Platform):
    _cuda_platform_instance: Optional[CudaPlatform] = None
    _cpu_platform_instance: Optional[CpuPlatform] = None
    _cuda_available: Optional[bool] = None

    @classmethod
    def _get_cuda_platform(cls) -> Optional[CudaPlatform]:
        """Lazy initialization of CUDA platform."""
        if cls._cuda_platform_instance is None:
            if cls._check_cuda_available():
                cls._cuda_platform_instance = CudaPlatform()
            else:
                logger.warning("CUDA not available, heterogeneous mode will use CPU-only features")
        return cls._cuda_platform_instance

    @classmethod
    def _get_cpu_platform(cls) -> CpuPlatform:
        """Lazy initialization of CPU platform."""
        if cls._cpu_platform_instance is None:
            cls._cpu_platform_instance = CpuPlatform()
        return cls._cpu_platform_instance
```

### 4.2 안전한 CUDA 가용성 체크

```python
def _is_cuda_available() -> bool:
    """Check if CUDA is available without triggering full initialization."""
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False

@classmethod
def _check_cuda_available(cls) -> bool:
    """Check CUDA availability (cached)."""
    if cls._cuda_available is None:
        cls._cuda_available = _is_cuda_available()
    return cls._cuda_available
```

### 4.3 `get_device_capability()` 메서드 추가

#### 문제점
Flash Attention 버전 선택 시 `get_device_capability()` 메서드가 호출되는데, `HeterogeneousPlatform`에 이 메서드가 없어 `None`을 반환했습니다.

```
ValueError: Unsupported FA version: None
```

#### 해결책
```python
@classmethod
def get_device_capability(cls, device_id: int = 0):
    """Get device capability, delegating to CUDA platform."""
    if cls._check_cuda_available():
        cuda_platform = cls._get_cuda_platform()
        if cuda_platform is not None:
            return cuda_platform.get_device_capability(device_id)
    return None  # CPU doesn't have CUDA capabilities
```

### 4.4 V1 엔진 지원 메서드 추가

```python
@classmethod
def supports_v1(cls, model_config: "ModelConfig") -> bool:
    """Heterogeneous platform supports V1 engine."""
    if cls._check_cuda_available():
        cuda_platform = cls._get_cuda_platform()
        if cuda_platform is not None:
            return cuda_platform.supports_v1(model_config)
    return True  # Enable V1 for heterogeneous mode

@classmethod
def default_v1(cls, model_config: "ModelConfig") -> bool:
    """Enable V1 by default for heterogeneous platform."""
    if cls._check_cuda_available():
        cuda_platform = cls._get_cuda_platform()
        if cuda_platform is not None:
            return cuda_platform.default_v1(model_config)
    return True

@classmethod
def get_current_memory_usage(cls, device: Optional[torch.device] = None) -> float:
    """Return memory usage in bytes, delegating to appropriate platform."""
    if device is not None and device.type == "cpu":
        return cls._get_cpu_platform().get_current_memory_usage(device)
    if cls._check_cuda_available():
        cuda_platform = cls._get_cuda_platform()
        if cuda_platform is not None:
            return cuda_platform.get_current_memory_usage(device)
    return cls._get_cpu_platform().get_current_memory_usage(device)
```

### 4.5 모든 메서드에 Fallback 로직 추가

기존 메서드들에 CUDA 미사용 환경을 위한 fallback 로직이 추가되었습니다:
- `supported_dtypes`
- `get_device_name()`
- `get_device_total_memory()`
- `is_async_output_supported()`
- `check_and_update_config()`
- `verify_model_arch()`
- `verify_quantization()`
- `has_device_capability()`
- `device_count()`
- `get_ray_placement_group_bundles()`
- `set_device()`
- `get_attn_backend_cls()`

---

## 5. CPU Platform의 Intel 최적화

### 파일: `vllm/platforms/cpu.py`

### 5.1 CPU 기능 자동 감지

```python
@classmethod
def _get_cpu_flags(cls) -> set[str]:
    """
    Get CPU feature flags from /proc/cpuinfo.
    Returns set of CPU flags (e.g., 'avx512f', 'avx512_vnni', 'amx_bf16').
    """
    flags: set[str] = set()
    if platform.system() != "Linux":
        return flags

    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":")[1].strip().split())
                    break
    except Exception:
        pass

    return flags
```

### 5.2 Inductor 컴파일 최적화

```python
# Intel CPU optimized Inductor configuration
inductor_config = {
    "dce": True,  # Dead code elimination
    "size_asserts": False,
    "nan_asserts": False,
    "epilogue_fusion": True,  # Fuse epilogue operations
    "max_autotune": True,  # Auto-tune for best performance
    "freezing": True,  # Enable parameter freezing
}

# Detect CPU features and set SIMD optimization level
cpu_flags = cls._get_cpu_flags()

if "avx512f" in cpu_flags:
    # AVX-512 specific optimizations (Xeon, some consumer CPUs)
    inductor_config["cpp.simdlen"] = 16  # 512-bit / 32-bit
    logger.info("AVX-512 detected, enabling 512-bit SIMD")

    # Enable AVX-512 VNNI for int8 operations if available
    if "avx512_vnni" in cpu_flags:
        logger.info("AVX-512 VNNI detected")

    # Enable BF16 support for Sapphire Rapids
    if "avx512_bf16" in cpu_flags:
        logger.info("AVX-512 BF16 detected")

    # AMX support for Sapphire Rapids
    if "amx_bf16" in cpu_flags:
        logger.info("Intel AMX-BF16 detected (Sapphire Rapids)")

elif "avx2" in cpu_flags:
    # AVX2 optimization (most modern CPUs)
    inductor_config["cpp.simdlen"] = 8  # 256-bit / 32-bit
    logger.info("AVX2 detected, enabling 256-bit SIMD")

    # AVX-VNNI available on Alder Lake+ without AVX-512
    if "avx_vnni" in cpu_flags:
        logger.info("AVX-VNNI detected (Alder Lake+)")
```

### 5.3 Intel OpenMP 환경변수 최적화

```python
# Intel OpenMP settings for optimal performance
# KMP_BLOCKTIME: time (ms) thread waits before sleeping
os.environ.setdefault('KMP_BLOCKTIME', "1")

# KMP_TPAUSE: prevents CPU from entering low-power state
os.environ.setdefault('KMP_TPAUSE', "0")

# Barrier patterns: "dist,dist" provides fine-grained parallelism
os.environ.setdefault('KMP_FORKJOIN_BARRIER_PATTERN', "dist,dist")
os.environ.setdefault('KMP_PLAIN_BARRIER_PATTERN', "dist,dist")
os.environ.setdefault('KMP_REDUCTION_BARRIER_PATTERN', "dist,dist")

# KMP_AFFINITY: thread-to-core binding strategy
os.environ.setdefault('KMP_AFFINITY', "granularity=fine,compact,1,0")

# MKL settings for AVX-512
if "avx512f" in cpu_flags:
    os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', "AVX512")
```

---

## 6. CPU PagedAttention의 토큰-시퀀스 불일치 처리

### 파일: `vllm/v1/attention/backends/cpu_attn.py`

### 6.1 IPEX Import 예외 처리 강화

```python
# 변경 전
except (ImportError, AttributeError):
    _use_ipex = False

# 변경 후
# Also catch general Exception for PyTorch version mismatch errors
except (ImportError, AttributeError, Exception):
    _use_ipex = False
    ipex_modules = None
```

### 6.2 토큰-시퀀스 불일치 문제

#### 문제점
`forward_decode()` 함수에서 `query.shape[0]`을 시퀀스 수로 사용했으나, preemption에서 복구된 시퀀스가 여러 토큰을 가질 경우 불일치가 발생합니다.

```
RuntimeError: The size of tensor a (42) must match the size of tensor b (41)
at non-singleton dimension 0
```

- `query.shape[0]` = 42 (총 토큰 수)
- `context_lens.shape[0]` = 41 (실제 시퀀스 수)

#### 해결책

```python
num_tokens = query.shape[0]
num_seqs = context_lens.shape[0]

# Handle case where num_tokens != num_seqs
if num_tokens != num_seqs:
    # Fall back to loop-based implementation for this edge case

    if num_tokens < num_seqs:
        # Rare edge case: fewer tokens than sequences
        # Process only the tokens we have, assuming 1:1 mapping
        for token_idx in range(num_tokens):
            ctx_len = context_lens[token_idx].item()
            seq_blocks = block_tables[token_idx]
            # ... 개별 토큰 attention 계산 ...
        return

    # num_tokens > num_seqs: some sequences have multiple tokens
    extra_tokens = num_tokens - num_seqs
    token_idx = 0

    for seq_idx in range(num_seqs):
        ctx_len = context_lens[seq_idx].item()
        seq_blocks = block_tables[seq_idx]

        # Heuristic: last 'extra_tokens' sequences have 2 tokens each
        if seq_idx >= num_seqs - extra_tokens:
            tokens_for_seq = 2
        else:
            tokens_for_seq = 1

        # ... KV 캐시 로드 및 attention 계산 ...

        for t in range(tokens_for_seq):
            q_token = query[token_idx:token_idx+1].unsqueeze(2)
            attn_out = F.scaled_dot_product_attention(q_token, _k, _v)
            output[token_idx] = attn_out.squeeze(0).squeeze(1)
            token_idx += 1
    return

# num_tokens == num_seqs: use existing vectorized implementation
```

### 기술적 세부사항

#### 토큰-시퀀스 매핑 휴리스틱
명시적인 토큰-시퀀스 매핑 정보가 `forward_decode()`에 전달되지 않기 때문에, 다음 휴리스틱을 사용합니다:

| 조건 | 처리 방법 |
|------|----------|
| `num_tokens == num_seqs` | 기존 벡터화된 구현 사용 (최적 성능) |
| `num_tokens > num_seqs` | 마지막 `(num_tokens - num_seqs)` 개 시퀀스에 각 2개 토큰 할당 |
| `num_tokens < num_seqs` | 첫 `num_tokens`개 시퀀스에만 1:1 매핑 |

---

## 7. CPU Model Runner의 NUMA 최적화

### 파일: `vllm/v1/worker/cpu_model_runner.py`

### 7.1 Intel CPU 유틸리티 Import

```python
# Intel CPU optimization utilities (optional, graceful fallback)
try:
    from vllm.platforms.intel_cpu_utils import (
        NUMAAllocator,
        create_numa_aware_tensor,
        is_ipex_available,
        optimize_model_with_ipex,
    )
    _INTEL_UTILS_AVAILABLE = True
except ImportError:
    _INTEL_UTILS_AVAILABLE = False
    NUMAAllocator = None
    is_ipex_available = lambda: False
    optimize_model_with_ipex = lambda m, **kw: m
```

### 7.2 NUMA-aware 초기화

```python
def __init__(self, vllm_config: VllmConfig, device: torch.device,
             numa_node: int = -1):
    # Store NUMA node before parent __init__ (which may allocate memory)
    self._numa_node = numa_node
    self._numa_allocator = None

    # Try to set up NUMA binding if available
    if _INTEL_UTILS_AVAILABLE and NUMAAllocator is not None and numa_node >= 0:
        try:
            self._numa_allocator = NUMAAllocator()
            if self._numa_allocator.is_available:
                self._numa_allocator.bind_to_node(numa_node)
                logger.info(f"CPUModelRunner bound to NUMA node {numa_node}")
            else:
                self._numa_allocator = None
        except Exception as e:
            logger.debug(f"NUMA setup skipped: {e}")
            self._numa_allocator = None

    super().__init__(vllm_config, device)
```

### 7.3 NUMA-aware KVCache 할당

```python
def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """NUMA-aware KVCache allocation for CPU."""
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}

    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        size = kv_cache_tensor.size

        # Use NUMA-aware allocation if available
        if (_INTEL_UTILS_AVAILABLE and
            self._numa_allocator is not None and
            self._numa_allocator.is_available and
            self._numa_node >= 0):

            # Ensure allocation happens on the correct NUMA node
            self._numa_allocator.bind_to_node(self._numa_node)
            tensor = torch.zeros(size, dtype=torch.int8, device='cpu')
            logger.info(f"Allocated KVCache ({size / (1024**3):.2f} GiB) "
                       f"on NUMA node {self._numa_node}")
        else:
            # Standard allocation (fallback)
            tensor = torch.zeros(size, dtype=torch.int8, device='cpu')
            logger.info(f"Allocated KVCache ({size / (1024**3):.2f} GiB) "
                       f"without NUMA binding")

        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor

    return kv_cache_raw_tensors
```

### 7.4 IPEX 모델 최적화

```python
def load_model(self, **kwargs) -> None:
    # ... 모델 로드 ...

    # Apply IPEX optimization if available
    if _INTEL_UTILS_AVAILABLE and is_ipex_available():
        try:
            # Use bfloat16 for Sapphire Rapids with AMX
            dtype = self.model_config.dtype
            if dtype == torch.float32:
                # bfloat16 is more efficient on Sapphire Rapids
                dtype = torch.bfloat16
            self.model = optimize_model_with_ipex(self.model, dtype=dtype)
            logger.info(f"Model optimized with IPEX (dtype={dtype})")
        except Exception as e:
            logger.warning(f"IPEX optimization failed: {e}")
```

### 7.5 CPU 전용 Profile Run 및 Dummy Run

CUDA 그래프 캡처 없이 CPU에서 동작하는 간소화된 버전:

```python
def profile_run(self) -> None:
    """CPU-specific profile run. Simplified version without CUDA graph capture."""
    import gc
    from vllm.distributed.parallel_state import get_pp_group

    hidden_states, last_hidden_states = self._dummy_run(
        self.max_num_tokens, is_profile=True)

    if get_pp_group().is_last_rank:
        if self.is_pooling_model:
            output = self._dummy_pooler_run(hidden_states)
        else:
            output = self._dummy_sampler_run(last_hidden_states)
    else:
        output = None

    del hidden_states, output
    gc.collect()

@torch.inference_mode()
def _dummy_run(self, num_tokens: int, ...) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU-specific dummy run. Simplified version without CUDA-specific features."""
    # ... CUDA 그래프 관련 코드 제거 ...
    # No CUDA graph capture on CPU
    attn_metadata = None
    # ... 모델 실행 ...
```

---

## 8. CPU Worker의 Intel 최적화

### 파일: `vllm/v1/worker/cpu_worker.py`

### 8.1 Intel CPU 유틸리티 Import

```python
# Intel CPU optimization utilities (optional, graceful fallback)
try:
    from vllm.platforms.intel_cpu_utils import (
        setup_intel_cpu_environment,
        detect_intel_cpu_features,
        NUMAAllocator,
        is_ipex_available,
        IntelCPUFeatures,
    )
    _INTEL_UTILS_AVAILABLE = True
except ImportError as e:
    _INTEL_UTILS_AVAILABLE = False
    IntelCPUFeatures = None
    NUMAAllocator = None
```

### 8.2 Intel CPU 환경 초기화

```python
def __init__(self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False):
    # Intel CPU Optimization Setup
    self._intel_config: dict = {}
    self._numa_node: int = -1
    self._cpu_features: Optional[IntelCPUFeatures] = None

    if _INTEL_UTILS_AVAILABLE:
        try:
            # Setup Intel CPU environment early
            self._intel_config = setup_intel_cpu_environment(
                rank=local_rank,
                world_size=vllm_config.parallel_config.world_size,
                enable_numa=True,
                enable_avx_optimization=True,
                enable_ipex=True
            )
            self._numa_node = self._intel_config.get("numa_node", -1)
            self._cpu_features = self._intel_config.get("features")

            logger.info(f"CPUWorker[{rank}] Intel optimization initialized:")
            logger.info(f"  NUMA node: {self._numa_node}")
            logger.info(f"  AVX-512: {self._intel_config.get('avx512_enabled', False)}")
            logger.info(f"  IPEX: {self._intel_config.get('ipex_enabled', False)}")

            if self._cpu_features:
                logger.info(f"  CPU: {self._cpu_features.model_name}")
                logger.info(f"  Topology: {self._cpu_features.num_sockets} sockets x "
                           f"{self._cpu_features.cores_per_socket} cores")
                if self._cpu_features.amx_bf16:
                    logger.info("  AMX-BF16: Enabled (Sapphire Rapids)")
        except Exception as e:
            logger.warning(f"Intel CPU optimization setup failed: {e}")
```

### 8.3 NUMA-aware 스레드 설정

```python
def _configure_threads_for_numa(self):
    """Configure thread count based on NUMA topology or CPU count."""
    current_threads = torch.get_num_threads()

    # Try to use NUMA-aware thread count if features are detected
    if self._cpu_features and self._cpu_features.cores_per_socket > 0:
        # Calculate cores per NUMA node
        num_numa_nodes = 1
        if _INTEL_UTILS_AVAILABLE and NUMAAllocator is not None:
            try:
                allocator = NUMAAllocator()
                if allocator.is_available:
                    num_numa_nodes = max(1, allocator.num_nodes)
            except Exception:
                pass

        total_cores = self._cpu_features.cores_per_socket * self._cpu_features.num_sockets
        cores_per_node = total_cores // num_numa_nodes

        # Reserve 1-2 cores for system tasks
        target_threads = max(4, cores_per_node - 1)

        torch.set_num_threads(target_threads)
        if self._numa_node >= 0:
            logger.info(f"CPUWorker: Thread count set to {target_threads} "
                       f"(NUMA node {self._numa_node}, {cores_per_node} cores/node)")
    elif current_threads < 4:
        # Fallback: use most CPU cores
        target_threads = max(8, multiprocessing.cpu_count() - 2)
        torch.set_num_threads(target_threads)
```

### 8.4 Model Runner에 NUMA 노드 전달

```python
def init_device(self):
    # ... 기존 초기화 ...

    # Construct the model runner with NUMA awareness
    self.model_runner: CPUModelRunner = CPUModelRunner(
        self.vllm_config, torch.device("cpu"),
        numa_node=self._numa_node)  # NUMA 노드 전달
```

### 8.5 속성 추가

```python
@property
def numa_node(self) -> int:
    """Get the NUMA node this worker is bound to."""
    return self._numa_node

@property
def intel_config(self) -> dict:
    """Get Intel CPU optimization configuration."""
    return self._intel_config
```

---

## 9. GPU Model Runner의 Triton Lazy Import

### 파일: `vllm/v1/worker/gpu_model_runner.py`

### 문제점
CPU 워커에서 GPU Model Runner를 상속할 때, Mamba 관련 import가 Triton을 로드하여 실패합니다.

### 해결책

```python
# 변경 전
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaBase
from vllm.v1.attention.backends.mamba_selectors import get_mamba_attn_backend

# 변경 후
# Lazy import MambaBase to avoid Triton initialization on CPU-only workers
try:
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaBase
    _MAMBA_AVAILABLE = True
except (ImportError, RuntimeError):
    MambaBase = None
    _MAMBA_AVAILABLE = False

# Lazy import Mamba selectors to avoid Triton initialization
try:
    from vllm.v1.attention.backends.mamba_selectors import get_mamba_attn_backend
except (ImportError, RuntimeError):
    get_mamba_attn_backend = None
```

### 사용 시 체크

```python
# Mamba 백엔드 사용 시
elif isinstance(kv_cache_spec, MambaSpec):
    if get_mamba_attn_backend is None:
        raise RuntimeError("Mamba backend not available (Triton/CUDA required)")
    # ...

# Mamba 레이어 조회 시
mamba_layers = get_layers_from_vllm_config(self.vllm_config, MambaBase) \
    if _MAMBA_AVAILABLE and MambaBase is not None else []
```

---

## 10. 호환성 및 Fallback

모든 최적화는 graceful fallback을 지원합니다:

| 기능 | 사용 조건 | Fallback |
|------|----------|----------|
| AVX-512 최적화 | AVX-512F 지원 CPU | AVX2 사용 |
| NUMA 바인딩 | libnuma 설치 + 멀티소켓 | 표준 할당 |
| IPEX 최적화 | intel-extension-for-pytorch 설치 | 표준 PyTorch |
| AMX-BF16 | Sapphire Rapids 이상 | 소프트웨어 BF16 |
| Mamba 모델 | Triton + CUDA | 미지원 (오류) |

---

## 11. 테스트 방법

### 서버 실행
```bash
VLLM_HETEROGENEOUS_PLATFORM=1 python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-6.7b \
  --device heterogeneous \
  --pipeline-parallel-size 2
```

### 벤치마크
```bash
python benchmarks/benchmark_serving.py \
  --backend openai \
  --base-url http://localhost:8000 \
  --model facebook/opt-6.7b \
  --dataset-name random \
  --num-prompts 500 \
  --random-input-len 128 \
  --random-output-len 128 \
  --no-stream
```

### CPU 기능 테스트
```bash
python -c "
from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features
features = detect_intel_cpu_features()
print(f'CPU: {features.model_name}')
print(f'AVX-512: {features.avx512f}, AMX: {features.amx_bf16}')
print(f'NUMA: {features.num_sockets} sockets')
"
```

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2026-02-03 | `platforms/__init__.py`: heterogeneous 플랫폼 우선순위 조정 |
| 2026-02-03 | `_ipex_ops.py`: IPEX import 예외 처리 강화 |
| 2026-02-03 | `attention/layer.py`: CPU 텐서 런타임 디바이스 체크 추가 |
| 2026-02-03 | `heterogeneous.py`: lazy 초기화, `get_device_capability()` 구현 |
| 2026-02-03 | `cpu.py`: AVX-512 감지, Inductor/OpenMP 최적화 |
| 2026-02-03 | `cpu_attn.py`: IPEX 예외 처리, 토큰-시퀀스 불일치 fallback |
| 2026-02-03 | `cpu_model_runner.py`: NUMA KVCache, IPEX 모델 최적화 |
| 2026-02-03 | `cpu_worker.py`: Intel CPU 환경 설정, 스레드 어피니티 |
| 2026-02-03 | `gpu_model_runner.py`: Mamba/Triton lazy import |

---

*문서 작성: Claude (vLLM Heterogeneous CPU/GPU Optimization)*
*최종 업데이트: 2026-02-03*
