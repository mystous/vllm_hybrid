# vLLM 이종(Heterogeneous) 실행 모드 수정 사항 요약

본 문서는 vLLM의 Multiprocessing (MP) 백엔드에서 GPU와 CPU를 혼합하여 사용하는 **이종(Heterogeneous) 실행 모드**를 활성화하기 위해 수행된 코드 수정 사항과 그 의도를 정리합니다.

## 1. 초기화 행(Hang) 및 순환 참조 해결

가장 먼저 발생한 문제는 작업자 프로세스(Worker Process)들이 초기화 단계에서 멈추는(Hang) 현상이었습니다. 이는 모듈 간의 순환 참조(Circular Import)로 인한 데드락이 원인이었습니다.

### `vllm/platforms/interface.py`

- **수정 내용**: `from vllm.inputs import ...` 구문을 `TYPE_CHECKING` 블록 내부로 이동하고 `from __future__ import annotations`를 추가했습니다.
- **수정 의도**: `vllm.config` -> `vllm.platforms` -> `vllm.platforms.interface` -> `vllm.inputs` -> `vllm.config`로 이어지는 참조 고리를 끊어, 초기화 시 데드락을 방지했습니다.

### `vllm/platforms/__init__.py`

- **수정 내용**: 잘못된 임포트 경로(`vllm.utils.load_plugins_by_group`) 등을 수정했습니다.
- **수정 의도**: 플랫폼 플러그인 로딩 과정에서 발생하는 `ImportError`를 해결하여 정상적인 플랫폼 감지가 이루어지도록 했습니다.

## 2. 분산 환경 초기화 및 통신 (Distributed Environment)

GPU와 CPU가 공존하는 환경에서는 통신 백엔드와 그룹 초기화 로직이 기존의 GPU 전용 로직과 달라야 했습니다.

### `vllm/distributed/parallel_state.py`

- **수정 내용**:
    1. **`_is_heterogeneous_environment` 헬퍼 추가 및 캐싱**: 현재 설정이 이종 모드인지 확인하는 함수를 추가했습니다. 특히, `TorchDynamo` 컴파일 중 로깅 부작용(Logger side-effect)으로 인한 충돌을 방지하기 위해, `init_distributed_environment` 호출 시 이종 모드 여부를 전역 변수 `_IS_HETEROGENEOUS_MODE`에 캐싱하도록 수정했습니다.
    2. **`Config` 캐싱**: `get_current_vllm_config` 호출 시 경고 로그가 발생하는 문제를 피하기 위해 초기화 시점에 상태를 확정했습니다.
    3. **Gloo 백엔드 강제**: 이종 환경에서는 `nccl` 대신 `gloo` 백엔드를 사용하도록 강제했습니다 (NCCL은 P2P 통신에서 GPU-CPU 혼합을 직접 지원하지 않음).
    4. **`GroupCoordinator` 수정**: CPU 워커에서는 `device_communicator`(NCCL 기반)를 생성하지 않도록 예외 처리했습니다.
    5. **`init_model_parallel_group` 수정**: CPU 워커가 NCCL 그룹 생성(new_group)에 참여할 때 `gpu_ranks`가 비어있어 발생하는 `TypeError`를 수정하기 위해, 빈 리스트가 전달될 경우 `new_group`을 호출하지 않거나 예외 처리를 하도록 변경했습니다.

## 3. CPU 워커 (CPU Worker)

GPU 워커와 달리 CPU 워커는 CUDA Graph를 사용하지 않으며, 특정 라이브러리(NUMA 등) 의존성 문제가 있었습니다. 하지만 분산 처리 시 동기화를 위해 GPU 워커와 동일한 횟수의 통신(Collective op)을 수행해야 합니다.

### `vllm/v1/worker/cpu_worker.py`

- **수정 내용**:
    1. **`compile_or_warm_up_model` 재작성**: GPU 워커는 웜업 및 CUDA Graph 캡처를 위해 수십 번의 더미 실행(Dummy run)을 수행합니다. CPU 워커가 단 한 번만 실행하고 대기하면 **Deadlock**이 발생하므로, CPU 워커도 GPU 워커의 실행 횟수와 패턴(Warmup sizes + Capture sizes)을 그대로 따라하며 빈 실행(`skip_eplb=True`)을 하도록 로직을 동기화했습니다.
    2. **NUMA 초기화 예외 처리**: `torch.ops._C_utils.init_cpu_threads_env` 호출 실패 시 프로세스가 죽지 않고 로그만 남기도록 `try-except` 블록을 추가했습니다.
    3. **블록 수 계산 구현**: `determine_num_available_blocks` 메서드가 구현되어 있지 않아 추가했습니다.

### `vllm/v1/worker/cpu_model_runner.py`

- **수정 내용**:
    1. **`load_model` 시그니처 수정**: `GPUWorker`에서 상속받은 호출 규약(`**kwargs`)을 따르도록 수정하여 `TypeError`를 방지했습니다.
    2. **모델 로딩 장치 수정**: `get_model` 함수가 `vllm_config.device_config.device`를 참조하여 무조건 CUDA로 로딩하려던 문제를 해결하기 위해, `load_model` 메서드 내에서 임시로 `vllm_config`의 디바이스를 `cpu`로 오버라이딩(Override)하고 모델을 로드한 뒤 복구하도록 수정했습니다.

## 4. 워커 할당 및 실행 (Worker Setup & Execution)

### `vllm/worker/worker_base.py`

- **수정 내용**: **`init_worker`** 메서드에 이종 모드 감지 로직을 추가했습니다. 할당된 `rank`가 시스템의 물리적 GPU 개수보다 크거나, `device_type`이 `heterogeneous`인 경우, 해당 랭크의 워커 클래스를 강제로 `vllm.v1.worker.cpu_worker.CPUWorker`로 전환하도록 수정했습니다.

### `vllm/v1/executor/multiproc_executor.py`

- **수정 내용**:
    1. **Spawn 방식 강제**: 리눅스에서 기본 `fork` 방식을 사용하면 CUDA 컨텍스트 복제 등의 문제로 교착 상태에 빠질 수 있어, 모든 프로세스 생성 방식을 `spawn`으로 명시했습니다.
    2. **이종 환경 감지**: Executor 초기화 시에도 이종 환경 설정을 감지하여 적절한 환경 변수 설정을 돕도록 했습니다.

## 5. 요약

이 수정들의 핵심 목표는 **"GPU 중심의 기존 vLLM 구조에 CPU 워커를 억지로 끼워 맞추는 것"**이었습니다.

1. **초기화 단계**: 순환 참조를 끊고 올바른 클래스(`CPUWorker`)를 로드하게 함.
2. **통신 단계**: `NCCL` 대신 `Gloo`를 쓰게 하고, 불필요한 CUDA 통신 그룹 생성을 막음.
3. **실행 단계**: GPU 워커가 수행하는 수많은 웜업/캡처 루프에 CPU 워커가 보조를 맞추도록 하여(`dummy_run` 반복), 분산 통신 장벽(Barrier)에서 멈추지 않게 함.
