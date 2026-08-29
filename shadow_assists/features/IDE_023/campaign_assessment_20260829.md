# 캠페인 정직 평가 — "지금 논문을 쓸 내용이 있는가?" (2026-08-29)

> 사용자 질문 "지금 논문을 쓸 내용이 있긴 하니?"에 대한 공식 답변의 기록.
> **결론: 지금 당장 톱티어 논문 한 편이 되는 내용은 없다.** 아래에 근거와 도달 가능한 선택지를 남긴다.

## 1. 실험 환경의 정확한 기술 (480B×2 실험 기준)

**하드웨어** — violet-h100-016 한 대:
- NVIDIA H100 80GB × 8 (NVLink full-mesh, 인스턴스당 4장 분할)
- Intel Xeon 8480+ × 2소켓 (112C/224T, AMX-BF16/INT8), DDR5 2TB (2 NUMA, 인스턴스당 1소켓)
- CPU turbo OFF (2.0GHz 고정) — 모든 CPU 수치는 하한값

**소프트웨어 스택** — vLLM이 아니라 **SGLang + kt-kernel(KTransformers)**:
- SGLang 0.5.18 + kt-kernel 0.7.0.post2 (CPU-AMX expert 연산)
- 컨테이너 2개, 각각 cgroup cpuset(`--cpuset-cpus`/`--cpuset-mems`)으로 소켓 격리
- vLLM 컨테이너는 벤치 클라이언트로만 사용
- vLLM을 쓰지 않은 이유: vLLM의 CPU offload는 전송(swap) 방식이라 본 노드 클래스에서 기각됨 (8-27 계열). expert-상주-연산 경로는 kt-kernel이 유일한 실행 가능 구현

**기여 귀속** (논문 가치 판정의 핵심):

| 누가 만든 것 | 내용 |
|---|---|
| KTransformers 팀 (기존, SOSP'25) | 핵심 메커니즘 — expert 가중치를 DRAM에 두고 CPU AMX로 계산, GPU는 attention/KV. INT4 변환 도구 |
| 우리 | 이미지 호환 패치 4건, **kt pin 결함 발견 + cgroup 격리 해법** (없으면 다중 인스턴스 115× 붕괴), 4+4 NUMA-정합 분할 설계, 전 측정·판정 |

## 2. 보유 자산의 실제 등급

| 자산 | 실제 등급 |
|---|---|
| 480B GPU-only 불가 실증 (TP4 OOM / TP8 FP8-block 불가) + hybrid 서빙 성립·품질 4/4 | 능력 시연. 메커니즘은 남의 것 — 우리는 잘 쓴 것 |
| 분할>통합 전 부하 구간 +2~12% (K3, 교차점 없음, 전환 83~86초) | 견고한 측정 1건. AlpaServe(OSDI'23) 틀 안의 데이터 포인트 |
| kt CPUInfer 절대-pin 결함 + cgroup 해법 / DeepSeek 변환 결함(SUB_167) | 좋은 **버그 리포트 2건** — 논문 아님 |
| E1 knee (expert 배치 시 토큰당 비용 43~53× 하락), spec +45~55%, C1 부정 결과 ("가지=표", logprob 역신호) | 확인적 측정 + 소규모 부정 결과 |
| 기각 후보 4개의 사인 기록 (SCED·C1·Footprint·4라운드 축들) | 방법론 corpus — 후속 탐색의 사전값 |

## 3. experience 논문 권고의 자기 정정

직전 권고(experience/measurement 논문)도 냉정 기준으로는 **부족**: ATC/NSDI experience track은 수개월~수년 production 운영 교훈을 요구. 우리 corpus는 머신 1대 × 3일 × 남의 스택 위 측정.

## 4. 도달 가능한 선택지

1. **characterization study로 확장** (SIGMETRICS/IISWC/MLSys 계열): "CPU-expert 오프로드는 언제 이기는가"를 모델 크기(30B~600B) × 부하 × spec × 복제 구성 전체로 체계 측정. 보류된 IDE_027(regime atlas)이 골격. 필요 투자: 수 주 + prod 머신 교차 검증
2. **지금 수준 마감**: workshop 논문/기술 보고서 + upstream 이슈 2건(kt pin, DeepSeek 변환) + 측정 데이터 공개
3. **novelty 탐색 계속**: 단, 4라운드 연속 격추 패턴(모든 후보가 2023~2026 선행의 1~2보 이내)이 데이터로 반대함

## 5. 총평

지금까지의 성과 = **논문 반 편 분량의 측정 + 버그 리포트 2건 + 기각 사유가 정리된 후보 4개**. 이를 논문으로 포장하지 않는다. 방향 결정은 사용자 몫으로 남긴다.
