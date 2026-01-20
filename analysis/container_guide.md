# vLLM Hybrid 컨테이너화 가이드

이 문서는 `vllm_hybrid` 프로젝트를 도커(Docker) 컨테이너 환경에서 빌드하고 실행하기 위한 단계별 가이드입니다. NVIDIA GPU와 CPU를 모두 활용하는 하이브리드 환경을 구성하며, `uv`를 사용한 가상 환경 구성 및 수동 컴파일 단계를 포함합니다.

## 1. 사전 요구 사항 (Prerequisites)

호스트 머신에 다음 소프트웨어들이 설치되어 있어야 합니다.

1. **NVIDIA Driver**: GPU를 사용하기 위해 적절한 버전의 드라이버 설치.
2. **Docker Engine**: 컨테이너 런타임.
3. **NVIDIA Container Toolkit**: 컨테이너 내부 GPU 접근용.

    * 설치 확인:

        ```bash
        sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
        ```

## 2. Dockerfile 생성

개발 및 빌드 환경 구성을 위한 `Dockerfile.dev`를 생성합니다.

```dockerfile
# Base image: CUDA 12.8, Ubuntu 22.04
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    vim \
    sudo \
    build-essential \
    ninja-build \
    ccache \
    pkg-config \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# pip 및 빌드 도구 설치 (get-pip.py 사용)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install packaging setuptools wheel uv

# 작업 디렉토리 설정
WORKDIR /workspace

# 빌드 환경 변수
ENV MAX_JOBS=8
ENV NVCC_THREADS=8

CMD ["/bin/bash"]
```

## 3. 컨테이너 이미지 빌드

```bash
sudo docker build -t vllm_hybrid:dev -f Dockerfile.dev .
```

## 4. 컨테이너 실행

로컬 디렉토리를 마운트하지 않고 독립적인 환경으로 실행합니다.

```bash
sudo docker run -it \
    --gpus all \
    --shm-size=16g \
    --name vllm_container \
    vllm_hybrid:dev
```

## 5. vLLM 환경 빌드 및 실행 (상세 절차)

컨테이너 내부(`root@container_id:/workspace#`)에서 다음 단계를 순서대로 진행합니다.

### 5.1. 가상 환경 생성 및 활성화

`uv`를 사용하여 독립적인 가상 환경을 생성합니다.

가상 환경 생성 (Python 3.12)

```bash
uv venv vllm_dev_prj --python 3.12 --seed
```

가상 환경 활성화

```bash
source vllm_dev_prj/bin/activate
```

### 5.2. 소스 코드 다운로드

GitHub에서 소스 클론

```bash
git clone https://github.com/mystous/vllm_hybrid.git
```

프로젝트 디렉토리로 이동

```bash
cd vllm_hybrid
```

### 5.3. 의존성 설치 및 사전 설정

빌드 시간 단축을 위해 `VLLM_USE_PRECOMPILED=1` 옵션을 사용하고, 의존성을 설치합니다.

기본 의존성 및 torch 설치 (Precompiled 사용)

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
```

빌드 의존성 추가 설치

```bash
uv pip install -r requirements/build.txt --torch-backend=auto
```

### 5.4. CMake 설정 및 NVTX 패치

최신 CUDA Toolkit과 PyTorch 간의 NVTX 헤더 충돌을 해결하기 위해 `CMakeLists.txt`를 수정해야 합니다.

1. `CMakeLists.txt` 파일 열기:

    ```bash
    vi CMakeLists.txt
    ```

2. 약 73번째 줄 (`find_program(NVCC_EXECUTABLE nvcc)` 아래) 근처에 다음 코드를 삽입합니다.

    ```cmake
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Workaround for PyTorch NVTX headers issue with newer CUDA Toolkits
    # Assumes find_package(CUDAToolkit) was already done
    message(STATUS "Applying custom PyTorch NVTX headers workaround...")
    if(NOT TARGET CUDA::nvToolsExt)    
        message(STATUS "--> nvToolsExt Not found, looking for nvtx3.")
        if (NOT TARGET CUDA::nvtx3)
            message(STATUS "--> nvtx3 not found, adding library.")
            add_library(CUDA::nvtx3 INTERFACE IMPORTED)
            target_include_directories(CUDA::nvtx3 SYSTEM INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
            target_link_libraries(CUDA::nvtx3 INTERFACE ${CMAKE_DL_LIBS})
        endif()
        if (TARGET CUDA::nvtx3)
            add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
            target_compile_definitions(
                CUDA::nvToolsExt INTERFACE
                TORCH_CUDA_USE_NVTX3
            )
            target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
            message(STATUS "--> Workaround applied. Created CUDA::nvToolsExt target linked to CUDA::nvtx3.")
        else()
            message(STATUS "--> nvtx3 not found.")
        endif()
    else()
        message(STATUS "--> Workaround not needed or conditions not met.")
    endif()
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ```

### 5.5. CMake Preset 생성 및 빌드

이제 수동으로 CMake를 구성하고 빌드합니다.

CMake Preset 생성

```bash
python tools/generate_cmake_presets.py
```

(선택 사항) 생성된 Preset 확인 또는 수정

```bash
vi CMakeUserPresets.json
```

Release 프리셋으로 CMake 구성

```bash
cmake --preset release
```

빌드 및 설치 실행

```bash
cmake --build --preset release --target install
```

위 과정이 `pip install` 과정에서 내부적으로 수행하는 빌드 단계를 직접 제어하여 실행하는 절차입니다.

### 5.6. 실행 테스트

빌드 완료 후 정상 동작을 확인합니다.

텍스트 생성 테스트

```bash
python examples/offline_inference.py
```

## 요약 프로세스

1. **호스트**: `docker build ...` & `docker run ...`
2. **컨테이너**: `uv venv ...` & `source ...`
3. **컨테이너**: `git clone ...` & `cd vllm_hybrid`
4. **컨테이너**: `VLLM_USE_PRECOMPILED=1 uv pip install -e .`
5. **컨테이너**: `CMakeLists.txt` 수정 (NVTX 패치 적용)
6. **컨테이너**: `python tools/generate_cmake_presets.py`
7. **컨테이너**: `cmake --preset release`
8. **컨테이너**: `cmake --build --preset release --target install`
