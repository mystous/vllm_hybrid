# vLLM V1 프로세스 및 쓰레드 계층 구조 (Process & Thread Hierarchy)

이 다이어그램은 vLLM V1 아키텍처(특히 `MultiprocExecutor`를 사용하는 `AsyncLLM`)의 프로세스, 쓰레드, 그리고 핵심 컴포넌트 간의 포함 관계를 보여줍니다.

```mermaid
graph TB
    subgraph "Web Server Process (FastAPI / Uvicorn)"
        style WebServer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
        WebServerLabel[FastAPI Application]
        
        subgraph "Main Thread Loop"
            API_Router[API Router / Endpoints]
            
            subgraph "AsyncLLM Object"
                AsyncLLM_Obj[AsyncLLM Instance]
                
                subgraph "AsyncMPClient (Engine Client)"
                    MPClient[AsyncMPClient]
                    OutputTask[AsyncIO Task: OutputQueueLoop]
                    MonitorThread[Thread: MPClientEngineMonitor]
                end
            end
        end
    end

    subgraph "EngineCore Process (Child Process)"
        style EngineProc fill:#ffecb3,stroke:#ff6f00,stroke-width:2px
        EngineCoreProc_Cls[EngineCoreProc Object]
        
        subgraph "Internal Threads"
            MainThread[Main Thread: run_busy_loop]
            InputThread[Thread: process_input_sockets]
            OutputThread[Thread: process_output_sockets]
        end
        
        subgraph "MultiprocExecutor Object"
            Executor[MultiprocExecutor Component]
            WorkerMonitor[Thread: MultiprocWorkerMonitor]
            IOThreadPool[ThreadPoolExecutor: mp_exec_io]
        end
    end

    subgraph "Worker Process 0 (Grandchild Process)"
        style WorkerProc0 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
        WorkerProc0_Cls[WorkerProc Object]
        Worker0[Worker Instance]
        GPUModelRunner0[GPUModelRunner]
    end

    subgraph "Worker Process 1 (Grandchild Process)"
        style WorkerProc1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
        WorkerProc1_Cls[WorkerProc Object]
        Worker1[Worker Instance]
        GPUModelRunner1[GPUModelRunner]
    end

    %% Process Spawning Relationships
    ServerSocket[Uvicorn Socket] --> API_Router
    API_Router -- "calls" --> AsyncLLM_Obj
    AsyncLLM_Obj -- "spawns (multiprocessing)" --> EngineProc
    EngineProc -- "spawns (via Executor)" --> WorkerProc0
    EngineProc -- "spawns (via Executor)" --> WorkerProc1

    %% IPC Connections
    MPClient <== "ZMQ (TCP/IPC)" ==> InputThread & OutputThread
    Executor <== "Shared Memory + ZMQ" ==> WorkerProc0_Cls
    Executor <== "Shared Memory + ZMQ" ==> WorkerProc1_Cls
```

## 핵심 계층 구조 설명

1.  **Web Server Process (Main)**: 사용자가 실행하는 메인 OS 프로세스입니다 (`vllm serve` 등). **FastAPI** 애플리케이션과 **Uvicorn** 비동기 이벤트 루프가 여기서 실행됩니다.
    *   **AsyncLLM**: 이 프로세스 내에 파이썬 객체로 존재합니다.
    *   **API Router**: HTTP 요청을 받아 `AsyncLLM.generate()`를 호출합니다.
2.  **EngineCore Process (Child)**: 무거운 연산 로직(`busy_loop`, 스케줄링, 모델 실행 조율)을 담당합니다. 웹 서버의 HTTP 처리와 격리하기 위해 별도 프로세스로 생성됩니다.
3.  **Worker Processes (Grandchild)**: **EngineCore 프로세스의 자식 프로세스** (Web Server의 손자)입니다. 각 워커는 GPU를 관리하며, 빠른 속도를 위해 Shared Memory를 통해 EngineCore와 통신합니다.
4.  **Threads (쓰레드)**:
    *   **Web Server**: 요청 처리와 I/O를 위해 비동기 태스크(AsyncIO Task, 예: `OutputQueueLoop`)를 사용하며, EngineCore 상태 감시를 위한 별도 백그라운드 쓰레드(`MPClientEngineMonitor`)를 하나 둡니다.
    *   **EngineCore**: 메인 연산 루프(`run_busy_loop`)가 멈추지 않도록, ZMQ 통신(입/출력)을 위한 별도 쓰레드들과 워커 감시용 쓰레드를 사용합니다.
