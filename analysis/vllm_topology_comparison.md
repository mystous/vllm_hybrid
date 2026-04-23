# vLLM 프로세스 토폴로지 비교 (Single Node, 8 GPUs)

이 문서는 8-GPU 단일 노드 스펙에서 가능한 6가지 핵심 구성(TP, DP, PP 조합)에 따른 런타임 구조 변화를 시각화합니다.

> **용어 정리**
> *   **TP (Tensor Parallelism)**: 레이어(Layer)를 쪼개어 병렬 연산 (단일 오퍼레이션 가속).
> *   **DP (Data Parallelism)**: 모델 전체를 복제하여 처리량 증대 (배치 병렬 처리, 여러 EngineCore 생성).
> *   **PP (Pipeline Parallelism)**: 레이어(Layer)를 그룹(Stage)으로 나누어 순차 처리 (메모리 분산, 초대형 모델 수용).

---

## Case 1: TP=8, DP=1, PP=1 (초대형 모델 가속)

*   **구조**: 1개 엔진, 8개 워커가 하나의 거대한 모델 인스턴스 구동.
*   **특징**: 모델이 너무 커서 GPU 1개에 안 들어갈 때 사용. 워커들이 좌우로 배치되어 하나의 그룹을 형성.

```mermaid
graph TB
    subgraph "Web Server Layer"
        AsyncLLM[AsyncLLM]
    end

    subgraph "EngineCore Process (Global Scheduler)"
        EngineCore[EngineCoreProc]
    end

    subgraph "Workers (Single Model Instance)"
        direction LR
        W0["W0 (GPU0)"]
        W1["W1 (GPU1)"]
        W2["W2 (GPU2)"]
        W3["W3 (GPU3)"]
        W4["W4 (GPU4)"]
        W5["W5 (GPU5)"]
        W6["W6 (GPU6)"]
        W7["W7 (GPU7)"]
    end

    AsyncLLM --> EngineCore
    EngineCore -- "TP Group (AllReduce)" --> W0 & W1 & W2 & W3 & W4 & W5 & W6 & W7
```

---

## Case 2: TP=4, DP=2, PP=1 (밸런스형)

*   **구조**: **2개의 독립 엔진**. 각 엔진은 4개 GPU를 사용하여 모델 1개를 돌림.
*   **특징**: `AsyncLLM`이 들어오는 요청을 두 엔진으로 로드 밸런싱. 각 엔진 내 워커들은 좌우로 배치.

```mermaid
graph TB
    subgraph "Web Server"
        LB[Load Balancer]
    end

    subgraph "Replica 1 (DP Rank 0)"
        Engine1[EngineCore 1]
        subgraph "TP Group A"
            direction LR
            W0["W0 (GPU0)"]
            W1["W1 (GPU1)"]
            W2["W2 (GPU2)"]
            W3["W3 (GPU3)"]
        end
    end

    subgraph "Replica 2 (DP Rank 1)"
        Engine2[EngineCore 2]
        subgraph "TP Group B"
            direction LR
            W4["W4 (GPU4)"]
            W5["W5 (GPU5)"]
            W6["W6 (GPU6)"]
            W7["W7 (GPU7)"]
        end
    end

    LB --> Engine1
    LB --> Engine2
    Engine1 --> W0 & W1 & W2 & W3
    Engine2 --> W4 & W5 & W6 & W7
```

---

## Case 3: TP=4, DP=1, PP=2 (파이프라인 + 텐서 혼합)

*   **구조**: 1개 엔진. 모델을 앞부분(Stage 0)과 뒷부분(Stage 1)으로 나눔. 각 Stage는 4개 GPU로 TP 구동.
*   **특징**: 각 Stage의 TP 그룹은 좌우로 배치. Stage 0의 출력이 Stage 1로 전달됨.

```mermaid

graph TB
direction LR
    subgraph "Web Server"
        AsyncLLM
    end

    subgraph "EngineCore Process"
        EngineCore
    end

    subgraph "Pipeline Stage 0 (Layers 1~N/2)"
        subgraph "TP Group A"
            direction LR
            W0["W0 (GPU0)"] 
            W1["W1 (GPU1)"]
            W2["W2 (GPU2)"]
            W3["W3 (GPU3)"]
        end
    end

    subgraph "Pipeline Stage 1 (Layers N/2~N)"
        subgraph "TP Group B"
            direction LR
            W4["W4 (GPU4)"]
            W5["W5 (GPU5)"]
            W6["W6 (GPU6)"]
            W7["W7 (GPU7)"]
        end
    end

    AsyncLLM --> EngineCore
    EngineCore --> W0
    W3 == "P2P (Activation Passing)" ==> W4
    W0 ==> W1 ==> W2 ==> W3
```

---

## Case 4: TP=2, DP=2, PP=2 (복합 구성)

*   **구조**: **2개의 독립 엔진**. 각 엔진은 2-Stage 파이프라인(각 Stage는 TP=2)을 가짐.
*   **특징**: TP 그룹 내부 워커들이 좌우로 배치됨.

```mermaid
graph TB
    subgraph "Web Server (LB)"
        LB[Load Balancer]
    end

    subgraph "Replica 1 (DP Rank 0)"
        Engine1[EngineCore 1]
        subgraph "Stage 0 (TP=2)"
            direction LR
            W0["W0"]
            W1["W1"]
        end
        subgraph "Stage 1 (TP=2)"
            direction LR
            W2["W2"]
            W3["W3"]
        end
        W1 -.-> W2
    end

    subgraph "Replica 2 (DP Rank 1)"
        Engine2[EngineCore 2]
        subgraph "Stage 0 (TP=2)"
            direction LR
            W4["W4"]
            W5["W5"]
        end
        subgraph "Stage 1 (TP=2)"
            direction LR
            W6["W6"]
            W7["W7"]
        end
        W5 -.-> W6
    end

    LB --> Engine1 & Engine2
```

---

## Case 5: TP=1, DP=8, PP=1 (처리량 극대화)

*   **구조**: **8개의 독립 엔진**. 각각 GPU 1개씩 사용.
*   **특징**: 8개의 (엔진-워커) 쌍이 좌우로 나란히 배치되어 독립적으로 동작함을 나타냄.

```mermaid
graph TB
    subgraph "Web Server (LB)"
        LB[Load Balancer]
    end

    subgraph "Engines (Independent Replicas)"
        direction LR
        subgraph Pair0
            E0[E0] --- W0["W0(G0)"]
        end
        subgraph Pair1
            E1[E1] --- W1["W1(G1)"]
        end
        subgraph Pair2
            E2[E2] --- W2["W2(G2)"]
        end
        subgraph Pair3
            E3[E3] --- W3["W3(G3)"]
        end
        subgraph Pair4
            E4[E4] --- W4["W4(G4)"]
        end
        subgraph Pair5
            E5[E5] --- W5["W5(G5)"]
        end
        subgraph Pair6
            E6[E6] --- W6["W6(G6)"]
        end
        subgraph Pair7
            E7[E7] --- W7["W7(G7)"]
        end
    end

    LB --> E0 & E1 & E2 & E3 & E4 & E5 & E6 & E7
```

---

## Case 6: TP=1, DP=1, PP=8 (순수 파이프라인)

*   **구조**: 1개 엔진. 8개 GPU가 길게 직렬 연결.
*   **특징**: 좌우(Left-to-Right)로 순차 연결된 파이프라인 흐름.

```mermaid
graph LR
    subgraph "EngineCore"
        EC[Scheduler]
    end

    W0["W0 (Stage0)"] --> W1["W1 (Stage1)"] --> W2["W2 (Stage2)"] --> W3["W3 (Stage3)"] --> W4["W4 (Stage4)"] --> W5["W5 (Stage5)"] --> W6["W6 (Stage6)"] --> W7["W7 (Stage7)"]

    EC -- "Input" --> W0
    W7 -- "Output" --> EC
```
