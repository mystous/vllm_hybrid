# **업로드된 프로젝트 로그 기반 vLLM Hybrid 추론 성능 최적화 방안**

## **서론: vLLM Hybrid 프로젝트의 현황 및 병목 분석**

2026년 4월 11일 기준, H100x4 및 Xeon 8480+ 기반의 프로덕션 환경과 RTX 3090 및 i9-12900KF 기반의 개발 환경에서 진행 중인 vLLM Hybrid 프로젝트는 CPU와 GPU를 동시에 활용하는 아키텍처를 성공적으로 구현하여 검증을 마쳤다.\[8, 8\] 그러나 현재의 하이브리드 구성은 '성능 천장(Performance Ceiling)'에 직면해 있다. 개발 환경에서는 순수 GPU 모드 대비 2.4배에서 2.8배의 성능 패널티가 발생하며, 프로덕션 환경에서는 32B 모델까지도 H100 GPU의 활용률이 14%\~43%에 머물러 CPU로의 유효한 작업 오버플로우(Overflow)가 발생하지 않고 있다.1

이는 개별 요청(Request-level) 단위의 처리 시간이 ![][image1] 공식에 종속되기 때문이다.1 7B 모델 기준 CPU의 토큰당 지연 시간은 약 400ms로 GPU(약 25ms) 대비 10\~20배 느리기 때문에, 단순한 용량 기반 라우팅(Capacity-based routing)만으로는 GPU 단독 구동 성능을 능가할 수 없다.1 따라서 향후 최적화의 방향은 CPU와 GPU에 근본적으로 '다른 역할'을 부여하는 \*\*자원 차별화(Resource Differentiation)\*\*와 최적화된 마이크로 커널 도입으로 요약되며, 본 문서는 프로젝트 TODO.md에 명시된 핵심 과제(Ninja Gap)를 중심으로 성능 최적화 방안을 구체화한다.\[8, 8\]

## **1\. 기반 아키텍처: 다중 프로세스 및 C++ 기반 스레드 핀닝 최적화**

하이브리드 성능 극대화의 기본 전제는 Python 인터프리터의 GIL(Global Interpreter Lock) 병목과 운영체제 스케줄러의 간섭을 완전히 배제하는 것이다. 최근 CUDA 13.0 마이그레이션과 함께 vLLM 코어에 적용된 최적화들이 이 기반을 형성한다.1

### **1.1 하드웨어 친화적 NUMA 바인딩과 스레드 핀닝**

멀티 소켓 및 다중 NUMA 토폴로지 환경(예: H100x8 환경)에서 메모리 접근 지연을 막기 위해서는 CPU 엔진 프로세스의 정밀한 물리 코어 할당이 필수적이다.

* **스레드 핀닝 (Thread Pinning):** CUDA 빌드 환경에서도 CPU 스레드를 고정하기 위해 vllm.\_C\_utils C++ 확장 모듈을 도입하였다.1 init\_cpu\_threads\_env 함수를 통해 VLLM\_CPU\_OMP\_THREADS\_BIND='auto' 설정 하에서 OpenMP 워커 스레드들을 물리 코어에 1:1로 강제 고정하여 CPU 활용률 저하 문제를 해결했다.\[8, 8\]  
* **NUMA 인식 자동화:** \_resolve\_num\_cpu\_engines()를 통해 시스템의 NUMA 노드 개수를 자동 탐지하고 엔진을 할당한다.1 메모리 버스 경합을 막기 위해 엔진당 cpu\_max\_num\_seqs를 1개로 엄격히 제한하고, OMP\_PROC\_BIND=close 및 OMP\_PLACES=cores 환경 변수를 적용해 로컬 메모리 접근 속도를 극대화해야 한다.\[8, 8\]

### **1.2 IPEX 기반 Python 병목 우회 (Native Execution)**

CPU 연산이 의미 있는 처리량을 확보하려면 순수 파이썬 루프(sdpa\_loop)를 배제해야 한다.1 cpu\_attn.py에서 \_IPEXPagedAttention 백엔드를 우선적으로 호출하게 함으로써, 어텐션 연산을 C++ 기반의 oneDNN 커널(ipex\_modules.PagedAttention)로 직접 위임하여 파이썬 스택의 오버헤드를 완벽히 제거하였다.1

## **2\. \[A4\] AMX-INT8 양자화를 통한 CPU 마이크로 커널 가속**

이전까지 CPU 대역폭의 한계로 인해 7B 모델 기준 2.3 tokens/sec가 한계라고 가정했으나, 인텔 Xeon 8480+의 이론적 한계는 14 tokens/sec이며 실제 llama.cpp 역시 12\~18 tokens/sec를 달성하고 있다.1 이 격차를 메우기 위해 CPU 네이티브 양자화 연산기의 활성화가 필수적이다.1

* **VNNI 및 AMX 마이크로 커널 디스패치:** PyTorch 2.9 이상 환경에서 IPEX WoqWeightDtype.INT8 구성을 활성화하여 가중치 전용 양자화(WOQ)를 적용한다.\[8, 8\] Python 런타임에 인텔 VNNI 및 AMX 명령어 세트에 최적화된 C++ 마이크로 커널(gemm\_vnni.cpp)을 직접 연결(Dispatch)한다.1  
* **성능 목표:** 이 디스패치 경로가 활성화되면 순수 FP32/BF16 연산 대비 연산 처리량을 2배 이상(6\~12 tok/s 수준) 끌어올릴 수 있어, CPU 병목을 크게 완화할 수 있다.\[8, 8\]

## **3\. \[A2\] InfiniGen 스타일의 예측 기반 KV 캐시 계층화**

기존의 단순한 용량 기반 CPU 오프로드(--cpu-offload-gb)는 PCIe 지연 시간으로 인해 심각한 성능 저하를 유발한다.1 뜨거운 데이터(HBM)와 차가운 데이터(DRAM)를 단순히 나누는 방식을 폐기하고, 지능적인 사전 인출(Prefetching) 메커니즘을 도입해야 한다.\[8, 8\]

* **동적 예측 및 사전 인출:** InfiniGen(OSDI'24) 논문의 접근 방식을 차용하여, ![][image2] 번째 레이어의 어텐션 점수(Scores)를 바탕으로 ![][image3] 번째 레이어에서 필요할 상위 ![][image4] 의 중요 블록을 예측한다.1  
* **비동기 DMA 오버랩:** 예측된 필수 데이터만을 cudaMemcpyAsync 통신 경로를 통해 CPU DRAM에서 GPU HBM으로 비동기 사전 인출한다.1 동시에 GPU의 PagedAttention block\_table에는 슬라이딩 윈도우(최근 토큰 \+ 어텐션 싱크) 방식의 LRU(Least Recently Used) 기반 축출 정책을 적용하여 HBM 점유율을 억제한다.\[8, 8\]  
* **적용 대상:** 이 방식은 배치 크기가 1500을 초과하는 대규모 처리나 70B 모델 환경에서 처리량(Throughput)을 2\~3배 향상시킬 수 있는 핵심 기술이다.\[8, 8\]

## **결론 및 향후 검증 과제**

프로젝트 문서의 흐름에 따른 하이브리드 vLLM의 최적화는 '모든 것을 GPU에서 처리하고 남는 것을 CPU에 버리는' 방식이 아니라, 하드웨어 특성에 맞게 자원을 차별화하여 활용하는 데 있다.1 특히 C++ 기반의 네이티브 실행 환경 구축과 AMX-INT8 마이크로 커널을 통한 CPU 자체의 연산력 극대화, 그리고 InfiniGen 스타일의 지능적 KV 캐시 계층화를 통한 I/O 병목 제거가 핵심적인 역할을 수행할 것이다.1

단기적으로는 "Property 2 expected-finish gate"와 같은 라우팅 게이트의 정량적 공식을 보완하여 유휴 상태 오판을 막고1, \_update\_adaptive\_slots의 하드코딩 문제를 해결해야 한다.1 이후 ShareGPT 데이터셋을 활용해 Capacity, Round-robin, Length-aware 라우팅 전략을 벤치마킹하고, Intel RAPL 카운터를 통한 전력 대비 성능(Energy Efficiency) 검증이 병행된다면 완벽한 이기종 하이브리드 추론 엔진을 완성할 수 있을 것이다.1

#### **참고 자료**

1. 20260411\_154523\_hybrid\_optimization\_literature\_survey.md

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAR0AAAAYCAYAAADH0Cl8AAAJSUlEQVR4Xu2cd6gcVRTGP7FgL7FHJbFFxd67BjE2jF1UDIq9K1HsCk9FEkVFjd0EjSExFlCxgqLPghVs2LCAivqHoqIo2PX8cuayd+/u7M7ui++9Te4HH9m9M3fm7jnnnjbzImVkZGRkZGRkZGRkZGRkZGRkZGQMByxsXCYdzMgYBCxoXM64QHogY94FzuZ24zbpgYyMQQDO5hzjxOJzKXYzfmb8siJ392kZwwwLGW82nhyNLWacoUYdlnG2cYk5MzNSZFlWA5n2PcaD0wMBeKObjPcZRxffwVTjP8a9iu+kTbsaPzduXYxlDC+MM/arvrTayPix8TDjosXYpsafjU/JNxJgzjTjTLWJUPMxsiyrA1m9YlwjPQBWNt5vXCkaoyZ7Q+5gVovGl5R7+tWjsYzhAQwewz89GT9V9ZkPONL4r/H8ZHy88aJkLKOGLMvqIOu+19iXjM8BpRL1V4zgvR+UTw7AGd1oXCoayxge2NL4iXGDaIxoTLkVBw5AFvuncadknA10UDKW4ciy7BzIgOQFv1GHQ41jkrEy77288UTNvyljABkf2d4e8k1ODbuV8QDV0klkhFwPKc6hPI3BnF2M58kjKOfGcsXIV42I7AkAZKTxWJhzkvFF1QeEkcajVX/dsiwWsP7YaWXUkGXZOUhevlDFdkyZ985wYFBfyx3zHfKSE0d9sfFX4/HF+NnydJxz6ZuFrJFS6FHj68Z15VnKm8Zb5c4IbGZ81fiH/D6kqhg2Rs73H4xXq9ZbmF6wHcqy2IzOkWXZGgRGnE7bjK+V9+5VXKrGJwutSG9kxJyZ5SDT+cr4obwvBsgyyDZ+N+5cjIEr5cJHCYBGIw6FJ4ZhLg37v437Fd8DtpUbNhkR2dIk44XF5wAyr375fdqhLIvN6BxZlq0R7LKtfIi6ROsy772n8TuVHy/D5ca/5GVAVWxiPFyN5Rybm8j/m4YuGwtenB5XQBByWuYg9NjpAM6FAUHuqYL47TgcMhuOkTGFbCigqnK51l0a/llstzY2mOgFWQ61HCsHwyrem43W6ngzkEE9r84UdIq8h9QM1MvPGVdIDwwSgtOJ5RCEDGOH0szpUGLRy3m74GPyVxSayTWUYz/JHXGKqk6nl7LYbmxsMNErshxKOQa7pF1Tiiremws9ofLjZaD+JQOYW06COrGqB2fNcVO2HWnWpo3fFANxOvz7mtxoRxdjZZkOILOhX0NmR6+o20ynXRY7XNCtjQ0mekGWQy3HSnZZxXuTYXDOGcZH5O8ihE2wRTG2d/Gd3gXlFDcng3pB3vy81riKcU3jLHkJdaz8Lca15G9xXma8U/UvutFHYS7jRH4atVWwsfwpUlWy/vCiVxkG4nSQSfziJYidDrJizYBAcG5BnjQSECYUxwIweoy/ZURRtSyW+/FUjeY4JfFY4+LyJzd3y/+8YrI8gqJDel9TinOxA+ajz6pPbnYwPiDXKfoFrWwMBDvgvpQQ2CoyICu+zbij3DmTSQaHsJ7c9sgoxxVjKQg0BJzQnG+FKrJkzcjiIfl60X8vyxH57CNfPwz3KUPwJy33aRXvTaPzE/lTFzbWw/J5KIsnNQfL325mgbsbn5Q7EYQbojRZChnVmfLHzh/IG6bvyAV9oPzpDXO3lwMlUZ6hIBzTpxo6Dw4QOI3k+AWwVk7ne9UMCKeDwcZNY2QXjBgiU2R4nPFpeY8IQ6Sn86Ma/7YK+T6u8g1TJYsF6A+DQ084xbfk5+OIZ6pRh5TABJuXjaPkmSxlb5XNsqHxFrmtYfwEHlBmY2Bz1ewAHXwkXx/j+8rXeIXxBPmaCFqslU3HZ+TIcWwyBbJGB/QLy+wfVJElMuK9niPkMkGO6LhX5ch6+D0EP34/j8E51gpcl4cl+IE6jJE/rqVRicADf5E/mQmLDMC4g3fHM/IUBsHgqfkBCJAoANiQV6qxn8NxvlNysQGZwwbjPP5FSNvJNxtjbCSixelMlt/vFdWe/Aw2MJ7wKBvy6HuC6mXIZ8ZYZxhjDnMx/tly5zFd7qT5bSj1d3nEIjrE+sBY+b0oMYx9rFpGhIG9p8bylciJc+TJWHy9b+T3i8tI3jGivxTriQA00ri2XB/BHmIdcm/easfoMUacH5u7HTBG7IwMd6xqUbjMxrg+DqGZHRD0sOWX5GvEZthoyON943XyNeNQ4wwzxnh5CYuMQ1YaoxNZEjyfk6+B8eXl1+xVOXKPb+X2i4OcqvaZDnaE7Ll31+DmcY2I8h5VrRTh4i/KMxEWyzE2S+okEAYMESM4qRh9BQHKele1l4yIDmyGVtGoF4A8+W1xdhJ/7gRrqhatugW6ih0XOiIoAAwQQ+RY0FszHXI+41XA5iAT6Je/y0QJ1MrGUjuINylgjSFQBTCGXAhkVYD8p2jgAY1gQsBN0atyxHkR+INDq4I+tc8a2wJnglPBuSAkUn2EE2pPFvuY3DvjBfGszMGLs2AEgTJfkKdxCJ7rcTwG4yiGbOc0+Ybqlwsr3JdM4BgN3DjmFSAX0mFkw+dugIFiqOiJyIxu4qyHlJtrkxERRNAh52LMGDDHOIdzKVVio00xVp6poVPsBeNEz61sjF5TfzGnmR1wHhE4BnZJJhoyF+ZtofKNwBquUfnxqmAd2G4AVcAo9a4cGZsefSdbXz/6ngL7eVb176t1BSJh8IiQlLVP/le3LJ6FcPwseYnwjLx2xjteVYzPUK3RjLPByJkXgwUzl03EorkXny+RR49Z8qgwUd1vsHkROHpSemr8boCeaLYiWzbAW6qPyERH+nD0I3Yr5nCMOv96+f+jgqFRwhxVHC8DzoBIfojxBvkG41qtbIz1BTvAnihRYztgXWyqGIyz5snye5DFEBybgTIIhzMuPdAFtpIHWtZDlnCBvCHfq3Jkj06Tl1c4U+aNqDujHhNU6w0OCFxgkeg7C4wbcvQX8N545kmq7+xzbujZBPAZRTQD90qbfcGBhc/Z4TSCBjPGljryTnGQaiUseiPdZrMSTWMdBnA/dINOQtOba/A2eMqj5XbEdQgw8fXa2RgoswNsqdnaAJnEsulgAiL//pp7dpX+vm7kuLQ8WKcyhCELTe8D/g85cpz7tGsBrCN/CERm97+CBtSb8mbc3DL8jO5A9MRQOwHGhP6o/dEbTUxqfbCtvFResfie0R3mBzniKKlGyLqb4j+PN2vqjiaM1gAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAYCAYAAACIhL/AAAABIklEQVR4Xu3VL0tDURiA8VdkoDiGOBEEy4qLBi3CwoKClhlnXNvW3PIEo0UwGGRJmzY/gmA0+wEE+4oYBNHn9Sjce+TcneDOBT0P/GDcczfe3b8isVjsbzWHHlbthQDNooV5a3uqPt7Rthcm1AKauMAIj1hO7mC3iDoK1vZJpQPuYQNX4jFgnl1KxoDTWEft63MeOQfU03mKQ9zjJL0cLOeA2xighFtcSz5H0TlgF2vYxAv208s/mhHzIz7KmDJfG5tzwO+O8ISKtd1uF0NPxzLmuZYoc0B9vDzgTPz/8W+XOeAOXrGFqpijGbrMAc/F3MF6zejbRG+c0OmAeomt2AtaQ8ziDQ4k3Glewh2exbxi1ZuYWTqJ/T4riv8FHYv9mz4AYDE2A4ed800AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAXCAYAAADHhFVIAAAAjUlEQVR4XmNgGMqAG4gLgVgNXQIEioD4PxCno0uAgAgQOwAxK5o4bsAMxMZAbANlwwHIiAlAXAvEp4G4F1kyE4j1gdgSiL8BcQSyJAw0APETIFZEEwe78ioQTwFiRjQ5Bg8g/gXELkCszgAxBQ5mMEAcI8wACQhXZEk/Boh9G4C4gAGL0TxALIAuSG8AAE3IEZ6ptvLvAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAAYCAYAAAALQIb7AAACB0lEQVR4Xu2VPShGURjHH6HIR6R8hOQ7WUiUMlIkJSyilJKF8jGQBYPBRwal2MjAIFIMWMQki8EgpSjZGIwU/n/nnPeee7y31yr+9YvnnHPvOc//Oc99Rf71V5QI0kG0O2GJc0nuoK0okAZS3AlLbeAMTIAdkOmf/hI3Ggcj7oTRLHgFHxK8KBdcgjodd4NnMAWKQRZoBEfgUJQDgeKp38R7masmcA9ydFwAFkE+6ABdoFVUxpV6TaCWwB3IdsaNxkRtxgwo/l0Xfwb9EuxMSCwma3EA4pw5I2Zub8YMV8RbXw42JIJ9VBl4EnV6ipelBDSAeD1WCq5BtY5Zn2H9P9esyQ/so+j3O6gHsWAazIM98WzhAUbBKegF++LVj2si2mdk6lUIJkGFqIfD3c4i0AKSdcxstsSzjw7MgU5RB/cpFVyAK7AqylKKNtKmBB2HEzfgRsa+GnCu4z4wJMqRkEy9XkRlRwvNqSPJti8GbIIZHbOOvEC+2233Fy26AbsSfCuNXPsywK14l4waBFVW/K2/2Du0lfY2gx49bsu1j2JLsDXszQZArQn40AnYFmUDxc0Y04YFUV8IV7SODWzL9KrdPnze3AHJA4+iTmDUDh5EfXY47iuwqGzWxOs/W9yIJeAtZElYs9A6voh2uT8XzDjc1591XBb1onDii/lRZ4bHolroX79Un973VclDKqqxAAAAAElFTkSuQmCC>