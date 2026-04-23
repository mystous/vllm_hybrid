# 이기종(Heterogeneous) 플랫폼 검증 가이드

이 문서는 vLLM의 새로운 이기종 플랫폼 지원 기능을 검증하는 방법을 안내합니다. 이 기능을 통해 단일 분산 vLLM 인스턴스에서 CPU와 GPU 자원을 동시에 활용할 수 있습니다.

## 변경 사항 요약

1.  **새로운 플랫폼**: `vllm/platforms/heterogeneous.py` - 상황에 따라 `CudaPlatform` 또는 `CpuPlatform`으로 위임하는 `HeterogeneousPlatform`을 구현했습니다.
2.  **CLI 인자**: `vllm/engine/arg_utils.py` - `--device=heterogeneous` 인자를 추가했습니다.
3.  **워커 초기화**: `vllm/worker/worker.py` - `local_rank`와 가용 GPU 수를 비교하여 CUDA 또는 CPU(NUMA 바인딩 포함) 환경을 명시적으로 초기화하도록 `init_device`를 업데이트했습니다.
4.  **분산 상태**: `vllm/distributed/parallel_state.py` - 혼합 장치 그룹을 처리하고 장치 커뮤니케이터를 조건부로 인스턴스화하도록 `GroupCoordinator`를 업데이트했습니다.

## 검증 단계

에이전트 환경은 분산 GPU 워크로드를 실행하는 데 제약이 있으므로, 사용자의 환경에서 다음 검증 단계를 수행해 주십시오.

### 1. 기본 임포트 확인
새 플랫폼이 등록되어 있고 임포트 가능한지 확인합니다.
```bash
python -c "import vllm.platforms.heterogeneous; print('Heterogeneous Platform Module Import Successful')"
```

### 2. 이기종 실행 (수동 검증)
`heterogeneous` 장치로 vLLM을 실행합니다. 이기종 로직(Rank 0 -> GPU, Rank > GPU_COUNT -> CPU)을 검증하려면 멀티 GPU 환경 또는 최소 1개의 GPU가 필요합니다.

**명령어:**
```bash
# 2 GPU 시스템 예시 (총 4개 Rank: 2 GPU + 2 CPU)
python -m vllm.entrypoints.api_server --model facebook/opt-125m --device heterogeneous --tensor-parallel-size 4
```

**예상 동작:**
- **Rank 0 & 1**: CUDA 워커로 초기화됩니다.
- **Rank 2**: CPU 워커로 초기화되며 "Process bound to NUMA node 0" 로그가 출력됩니다. (GPU 수 + 0 번째 Rank)
- **Rank 3**: CPU 워커로 초기화되며 "Process bound to NUMA node 1" 로그가 출력됩니다. (GPU 수 + 1 번째 Rank)
- **참고**: `tensor-parallel-size`가 가용 GPU 수를 초과하면 표준 vLLM은 불평할 수 있지만, `HeterogeneousPlatform`은 이를 허용하도록 설계되었습니다. "World size > Device count" 오류가 발생하면 `heterogeneous.py`의 `device_count()` 로직(현재 `cuda_device_count() + 2`로 설정됨) 조정이 필요함을 확인하는 것입니다.

### 3. 환경 변수 확인
`--device=heterogeneous` 사용 시 코드는 자동으로 `VLLM_HETEROGENEOUS_PLATFORM=1`을 설정합니다. 디버그 출력을 추가하여 프로세스 환경 변수에서 이를 확인할 수 있습니다.

## 문제 해결

- **임포트 오류**: `vllm/platforms/heterogeneous.py`가 존재하고 문법 오류가 없는지 확인하십시오.
- **장치 불일치**: "Expected all tensors to be on the same device" 오류가 보이면 `GroupCoordinator` 또는 `Worker`의 장치 할당 로직이 일관되지 않은 것입니다.
- **통신 멈춤(Hang)**: 혼합 CPU-GPU 그룹은 Gloo에 의존합니다. 멈춤 현상이 발생하면 이기종 그룹에 대해 `dist_backend="gloo"`가 유효한지 확인하십시오.
