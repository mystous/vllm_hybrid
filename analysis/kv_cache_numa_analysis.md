# KVCache NUMA 최적화 및 할당 원리 분석

이기종(Heterogeneous) 환경에서 CPU 워커의성능을 좌우하는 핵심 요소 중 하나인 KVCache가 NUMA(Non-Unified Memory Access) 구조에서 어떻게 할당되고 관리되는지 분석한 결과입니다냥!

## 1. KVCache 할당 메커니즘 분석

vLLM의 KVCache 할당은 `vllm/worker/cache_engine.py`의 `CacheEngine` 클래스에서 담당합니다냥.

### 핵심 코드 (`CacheEngine._allocate_kv_cache`)

```python
# vllm/worker/cache_engine.py
def _allocate_kv_cache(self, num_blocks: int, device: str) -> List[torch.Tensor]:
    # ... 중략 ...
    for _ in range(self.num_attention_layers):
        layer_kv_cache = torch.zeros(
            kv_cache_allocation_shape,
            dtype=self.dtype,
            pin_memory=pin_memory,
            device=device).permute(*kv_cache_stride_order)
        kv_cache.append(layer_kv_cache)
    return kv_cache
```

* **할당 방식**: `torch.zeros`를 사용하여 CPU(`device="cpu"`)에 메모리를 할당합니다냥.
* **특징**: 별도의 `libnuma`나 `mbind` 같은 명시적인 시스템 콜을 직접 호출하지는 않지만, PyTorch의 CPU 텐서 할당은 OS의 기본 메모리 할당 정책을 따릅니다냥.

## 2. NUMA Pinning과의 연계 원리

우리가 구현한 `Worker.init_device`에서의 CPU Pinning 로직이 KVCache 성능을 극대화하는 핵심 원리는 다음과 같습니다냥:

### 초기화 순서

1. **`Worker.init_device()` 실행**: `os.sched_setaffinity`를 통해 프로세스를 특정 NUMA 노드(예: `node0`)에 고정합니다냥.
2. **`Worker.initialize_cache()` 실행**: 이후 `CacheEngine`이 생성되면서 `torch.zeros`로 KVCache를 할당합니다냥.

### OS의 Local Allocation 정책

* Linux 커널은 기본적으로 **"메모리를 요청한 스레드가 현재 실행 중인 NUMA 노드의 메모리를 우선적으로 할당"**하는 정책(Default/Local Policy)을 가집니다냥.
* 워커 프로세스가 이미 특정 노드에 고정되어 있으므로, `torch.zeros` 호출 시 해당 노드의 로컬 메모리 뱅크에서 페이지가 할당됩니다냥.
* 결과적으로 CPU 워커가 자기 옆에 있는 가장 빠른 메모리를 사용하게 되어, 노드 간 통신(Cross-node traffic)에 따른 지연 시간이 사라집니다냥!

## 3. KVCache 활용 방식

* **읽기/쓰기**: 추론 시 CPU 워커는 고정된 노드의 로컬 메모리에 있는 KVCache에 직접 접근하여 연산합니다냥.
* **스왑(Swap)**: GPU와 CPU 간의 `swap_in`, `swap_out` 작업 시에도 로컬 NUMA 노드의 대역폭을 최대로 활용하여 전송 속도가 가장 빠릅니다냥.

## 4. KVCache 할당 크기 상세 분석

KVCache의 전체 크기는 **[블록 개수] × [블록당 크기]**로 결정됩니다냥.

### 1) 블록당 크기 (Block Size) 계산 수식

`CacheEngine.get_cache_block_size`에서 정의된 수식은 다음과 같습니다냥:

> **Size per Block = Layers × Heads × HeadSize × 2 (K/V) × TokensPerBlock × DtypeSize**

* **Layers**: 해당 워커가 담당하는 **어텐션 레이어 수** (Pipeline Parallelism 분할 결과).
* **Heads**: 해당 워커가 담당하는 **KV 헤드 수** (Tensor Parallelism 분할 결과).
* **HeadSize**: 모델의 헤드 차원 (예: 128).
* **2 (K/V)**: Key와 Value 텐서를 각각 저장하기 위한 계수.
* **TokensPerBlock**: 블록당 토큰 수 (vLLM 기본값: **16**).
* **DtypeSize**: 자료형 크기 (BF16/FP16은 **2바이트**, FP8은 1바이트).

#### 예시 (Qwen2.5-7B, TP=2, 1개 레이어 기준)

* `Layers = 1`, `Heads = 2` (총 4개 중 TP로 나눔), `HeadSize = 128`, `TokensPerBlock = 16`, `Dtype = 2byte`
* **크기** = 1 × 2 × 128 × 2 × 16 × 2 = **16,384 bytes (16 KiB)**

### 2) 전체 할당 크기 (Total Allocation)

* **GPU 워커**: 가용 VRAM의 약 90%(`gpu_memory_utilization`)를 블록 개수(`num_gpu_blocks`)로 변환하여 할당합니다냥. (보통 수만 개 단위)
* **CPU 워커 (이기종 모드)**:
    * 현재 코드(`worker.py`)에서 `is_cpu_worker`인 경우 `num_gpu_blocks`가 **0**으로 설정됩니다냥.
    * 대신, `--swap-space` 설정(기본값 **4 GiB**)을 기반으로 계산된 `num_cpu_blocks`만큼 CPU 메모리에 KVCache가 할당됩니다냥.
    * **할당량** = `num_cpu_blocks` × `Block Size`. (예: 4 GiB 설정 시 약 4 GiB가 로컬 NUMA 노드에 할당됨)

## 5. 결론 및 최적화 확인

현재 vLLM의 구조는 **"프로세스 레벨의 Pinning → 자동 로컬 메모리 할당"** 구조를 취하고 있어, 추가적인 로우 레벨 최적화 없이도 NUMA 이점을 충분히 누리고 있습니다냥! 특히 CPU 워커는 지정된 `swap_space` 크기만큼 고정된 NUMA 노드에 집중적으로 메모리를 할당하므로 매우 효율적입니다냥!

> [!TIP]
> CPU 워커의 KVCache 용량을 늘리고 싶다면 실행 시 `--swap-space 16` (GiB 단위)과 같이 설정하면 된다냥! 🐈냥!
