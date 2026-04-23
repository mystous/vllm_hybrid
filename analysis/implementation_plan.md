# [구현 완료] 이기종(Heterogeneous) 플랫폼 지원

이 문서는 vLLM이 8 GPU + 2 CPU (총 10 Rank)와 같은 이기종 환경에서 동작하도록 지원하는 최종 구현 내역을 기술합니다. 초기 계획에서 발견된 이슈들을 해결하고 안정성을 확보하기 위한 수정 사항들이 모두 반영되었습니다.

## 목표 설명
vLLM의 실행 환경을 확장하여 GPU와 CPU 워커가 공존하는 이기종 클러스터를 지원합니다. 이를 통해 GPU의 연산 능력과 CPU의 유연성을 결합한 하이브리드 추론이 가능해집니다.

## 주요 변경 사항

### 1. 플랫폼 추상화 및 설정 (`vllm/platforms`)
이기종 환경을 정의하고 관리하는 새로운 플랫폼 클래스를 도입했습니다.

#### [NEW] `vllm/platforms/heterogeneous.py`
*   **`HeterogeneousPlatform` 클래스**:
    *   `device_type="heterogeneous"` 정의.
    *   `device_count()`: `torch.cuda.device_count() + 2`를 반환하여 물리적 GPU 외에 2개의 추가 CPU 워커가 있음을 알림.
    *   GPU/CPU 플랫폼 인스턴스를 내부적으로 보유하고 상황에 따라 위임.

#### [MODIFY] `vllm/platforms/__init__.py`
*   `VLLM_HETEROGENEOUS_PLATFORM` 환경 변수가 설정되면 `HeterogeneousPlatform`을 로드하도록 플러그인 등록.

### 2. 엔진 및 인자 처리 (`vllm/engine`)
사용자가 CLI를 통해 이기종 모드를 쉽게 활성화할 수 있도록 수정했습니다.

#### [MODIFY] `vllm/engine/arg_utils.py`
*   `--device` 인자에 `heterogeneous` 옵션 추가.
*   해당 옵션 선택 시 `VLLM_HETEROGENEOUS_PLATFORM="1"` 환경 변수 자동 설정.

### 3. 워커 초기화 및 실행 (`vllm/worker`)
CPU 워커가 GPU 자원을 점유하지 않고, 효율적으로 동작하도록 핵심 로직을 변경했습니다.

#### [MODIFY] `vllm/worker/worker.py`
*   **동적 장치 초기화 (`init_device`)**:
    *   `local_rank < GPU 수`: CUDA 장치 사용.
    *   `local_rank >= GPU 수`: CPU 장치 사용.
    *   **NUMA 바인딩**: CPU 워커를 각각 NUMA Node 0, 1에 바인딩(`_bind_to_numa_node`)하여 메모리 접근 성능 최적화.
*   **ModelRunner 선택 (`__init__`)**:
    *   CPU 워커일 경우 `v1.worker.cpu_model_runner.CPUModelRunner`를 사용하도록 동적 로딩.
    *   `CPUModelRunnerAdapter`를 구현하여 시그니처 불일치 해결.
*   **메모리 할당 차단 (`determine_num_available_blocks`)**:
    *   **CRITICAL FIX**: CPU 워커는 `num_gpu_blocks`를 강제로 0으로 설정하여 GPU 메모리 할당 시도 차단.
    *   GPU 메모리 프로파일링 단계 우회.
*   **검증 로직 예외 처리**:
    *   `_assert_memory_footprint_increased_during_profiling`에서 CPU 워커는 검사 제외.

### 4. 분산 통신 (`vllm/distributed`)
GPU(NCCL)와 CPU(Gloo)가 혼재된 환경에서 안전하고 효율적인 집합 통신(Collective Communication)을 구현했습니다.

#### [MODIFY] `vllm/distributed/parallel_state.py`
*   **그룹 분리 (`GroupCoordinator`)**:
    *   `device_group`: GPU 랭크들만 포함하는 NCCL 그룹 (CPU 랭크는 None).
    *   `cpu_group`: 모든 랭크를 포함하는 Gloo 그룹 (제어 및 데이터 교환용).
*   **계층적 통신 구현 (`all_reduce`, `broadcast`, `all_gather`)**:
    *   **AllReduce**: [GPU 집계 (NCCL)] -> [리더가 CPU로 복사] -> [전체 집계 (Gloo)] -> [GPU 복원]의 3단계 파이프라인 적용.
    *   **Broadcast/AllGather**: GPU 텐서를 CPU로 이동시킨 후 Gloo 그룹을 통해 안전하게 전파하고 다시 복구하는 `Bridge` 패턴 적용. 이를 통해 CPU 워커의 크래시 방지.

## 검증 계획

### 수동 검증
제공된 `walkthrough.md`에 따라 다음 항목을 확인합니다:
1.  **로그 확인**: "Initializing CPUModelRunner...", "Process bound to NUMA node...", "Initializing Heterogeneous CPU worker" 등의 로그가 출력되는지 확인.
2.  **실행 확인**: 8 GPU 서버에서 10개의 프로세스가 정상적으로 뜨고, 추론 결과가 반환되는지 확인.
3.  **메모리 확인**: `nvidia-smi`를 통해 불필요한 GPU 메모리 점유가 없는지, CPU 랭크(8, 9)가 GPU를 건드리지 않는지 확인.
