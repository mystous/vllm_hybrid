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

## 4. 결론 및 최적화 확인

현재 vLLM의 구조는 **"프로세스 레벨의 Pinning → 자동 로컬 메모리 할당"** 구조를 취하고 있어, 추가적인 로우 레벨 최적화 없이도 NUMA 이점을 충분히 누리고 있습니다냥!

> [!TIP]
> 더 극단적인 성능이 필요하다면 `numactl --membind`와 같은 명령어로 프로세스 전체의 메모리 정책을 강제할 수도 있지만, 현재 구현된 `worker.py` 내부의 Pinning 로직만으로도 KVCache는 최적의 위치에 자리 잡게 됩니다냥! 🐈냥!
