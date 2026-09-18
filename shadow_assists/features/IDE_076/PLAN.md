# CPU MoE 최적화 A1·A2·B 구현·검증 실행 지시서

> 문서 ID: `CPU_MOE_A1_A2_B_IMPLEMENTATION_V1`  
> 작성 기준일: 2026-09-17  
> 문서 상태: **구현 실행 지시서. 문서 작성 과정에서 서버 코드 변경·빌드·추론·성능 측정은 수행하지 않았다.**  
> 주 대상: **Qwen3-Coder-480B OPT4, 동일 물리 서버의 H100 80GB 4장 + CPU 2소켓**  
> 설계 원문: `CPU_MoE_병목분석_및_최적화설계.docx` 9~17장, 그림 6~9  
> 실측 근거: `FULL_REPORT-6.md`의 IDE_075 결과, 특히 S29 및 expert 표본  
> 실행 원칙: **A1 → A2 적용성 확인·구현 → 비용 재측정 → B → 개별·통합 검증 → 채택 또는 원복**  
> 실행 횟수 정책: **세션·부팅·재시도·품질 검증 횟수에 임의의 누적 상한을 두지 않는다.**

---

## 0. 실행자에게 전달할 핵심 지시

이 지시서는 기존 설계 보고서를 다시 요약하는 작업이 아니라, **설계된 A1·A2·B를 실제 소스에서 구현하고 성능·정상성·부작용을 검증하기 위한 작업 명세**다. 세 방안 모두 이번 작업의 평가 범위에 포함한다. A1에서 이득을 얻었다는 이유, 과거의 ‘최대 두 변경’ 규정, 또는 실행 횟수를 이유로 A2·B를 미검토 상태로 종료하지 않는다.

다만 **세 기능을 무조건 켠 구성을 최종 채택하라는 뜻은 아니다.** 설계 원문의 조건을 유지한다. A2는 같은 Cold job 내부에 줄일 수 있는 병렬 호출·작업 분할 비용이 확인될 때 구현·적용한다. 이미 동일한 최적화가 존재하거나 적용할 비용이 없으면 코드와 측정 근거를 남긴다. B도 실제 비용과 정상성·HBM 조건을 확보한 후 적용한다. 성능이나 정상성을 악화시키는 기능은 기본 활성화하지 않는다.

**필수 작업 흐름**

1. 현재 실행 소스·수정 바이너리·모델·입력·클라이언트를 확인하고 정상적인 무계측 기준선을 만든다.
2. **A1:** 기존 RB에 없는 잔여 비용을 찾아 실제 expert rows 1·2 및 실제 shard shape의 실행 경로를 특수화한다.
3. **A2:** 상위 FIFO와 job 순서는 그대로 두고, 같은 Cold job 안의 descriptor 묶음 실행을 구현·검증한다. 적용할 잔여 비용이 없다는 결론도 증거로 판정한다.
4. 선택한 CPU 실행 구성을 고정하고 Cold 서비스·GPU Hot 비용을 다시 측정한다.
5. **B:** calibration 자료로 Cold→Hot 승격과 Hot→Cold 강등을 짝지어 정적 Hot 배치를 만든다. 총 5,952슬롯, KV 및 GPU별 실제 메모리 제약을 유지한다.
6. 개별 효과와 조합 효과를 분리하여 무계측 MAIN, C1, LONG, 독립 입력, 수치·품질·수명·안정성을 검증한다.
7. 모든 시도, 실패, 미적용 근거, 최종 선택, 복구 방법과 원자료를 저장한다. 최종 활성 구성과 구현만 남기고 비활성화한 기능을 구분한다.

**이미 적용된 기능은 기준 구성이다.** callback-free, skip-empty, deferred8, AVX register blocking(RB), 비균일 Hotmap, CPU96·NUMA2, non-KT pinning을 다시 켠 것을 신규 성과로 보고하지 않는다. `[D §9~12; H073 §4, §7]`

**이번에 하지 않을 일:** GPU 수 변경, 전문가 전량 CPU 배치의 신규 주 비교군, 모델·프레임워크 전면 교체, AMX threshold 전수 탐색, CPU 워커·NUMA·affinity 전수 탐색, 전역 FIFO 교체, speculative decoding 추가, 정밀도 변환, 요청·출력 길이 축소. 별도 근거 없이 A1·A2·B와 무관한 문제로 확장하지 않는다.

### 0.1 완료 기준과 중단 사유

종료 기준은 정해진 실행 횟수가 아니라 **필요한 구현·검증·판정과 결과 인계를 완료했는가**다. 정상성 실패, 실제 접근 권한 부재, 저장 공간 부족, 다른 작업 침해, 복구가 필요한 교착은 영향을 받는 실행의 중단 사유다. ‘세션 소진’, ‘부팅 소진’, ‘품질 문항 소진’은 중단 사유가 아니다.

반복해도 새 정보가 나오지 않으면 같은 실행을 더 쌓기 전에 질문·대조·표본·계측 위치를 재설계한다. 필요한 검증을 미완료로 남긴 채 임의로 성공 또는 완료 처리하지 않는다. 반대로 A1·A2·B 판정과 무관한 모든 하드웨어 현상을 설명하려고 작업을 지연하지 않는다. `[P2 §0.3, §14~17]`

### 0.2 근거와 신규 실행 계약의 구분

| 표기 | 자료·의미 | 사용 원칙 |
|---|---|---|
| `[D]` | 첨부 최적화 설계 DOCX | A1·A2·B의 목적·범위·보존 조건을 계승한다. |
| `[R075]` | `FULL_REPORT-6.md` | 원 측정값과 실제 실행 이력이다. |
| `[V075]` | `IDE075_results_review.md` | 측정 해석, 유효성 제한, 개선 착수 판단이다. |
| `[P2]` | `IDE074_additional_measurement_plan_v2.md` | 지표·수명·비교 계약과 실행 횟수 상한 폐지 원칙이다. |
| `[H073]` | `CPU_MoE_optimization_session_handoff-2.md` | 모델·환경·기존 최적화·워크로드의 계승 자료다. 옛 실행 상한은 폐기한다. |
| **신규 실행 계약** | 이 문서의 작업 ID·파일명·필드·기능 플래그·의사코드·검증 절차 | 설계를 실행 가능하게 구체화한 요구다. 이미 저장소에 구현되어 있거나 성능이 입증됐다는 뜻이 아니다. |

실제 커널 전체 소스, 최종 런타임의 현재 HEAD, 새로운 성능 임계값은 첨부만으로 확정되지 않는다. 구현자는 실행 소스에서 확인하고 `PLAN_RESOLVED.md`에 고정한다. 첨부에 이름만 나온 `READINESS.md`, `OPTIMIZATION_HANDOFF.md`, 원시 trace는 내용을 확보하기 전까지 읽고 검증한 근거로 취급하지 않는다.

현재 지시서는 이전 **측정만 수행하라는 단계의 변경 금지 조건 중 A1·A2·B에 필요한 부분을 대체**한다. 나머지 정상성·재현성·안전 조건은 유지한다. 최신 사용자 지시와 충돌하는 ‘최대 두 후보’, ‘잔여 세션’ 조건은 적용하지 않는다.

### 0.3 빠른 탐색

| 구간 | 내용 |
|---|---|
| 1~3장 | 근거 수치, 고정 구성, 기준선과 코드 확인 |
| 4~6장 | 기능 분리, replay·정상성 공통 기반, A1 구현 |
| 7~9장 | A2 구현, B 비용 모델·배치 생성, 조합 비교 |
| 10~13장 | 성능·통계·원인 검증, 회귀·안정성, GLM 보호 |
| 14~18장 | 실행 순서, 상태·계속 조건, 결과·복구·완료 |
| 부록 A~D | 스키마, 구현해야 할 도구 계약, 작업 점검표, 출처 해시 |

---

## 1. 최적화의 출발 근거

### 1.1 대표 측정과 직접적인 구현 방향

아래 수치는 IDE_075 **S29 / DECODE bs=63 / cold_present_nonempty / n=15,600**이다. 시간 단위는 µs다. 서로 다른 열의 p50을 더하거나 빼서 대표 timeline 또는 기대 speedup을 만들지 않는다. `[R075 §5 S29; D §4, 부록 A]`

| 지표 | p50 | p95 | 구현에 반영할 의미 |
|---|---:|---:|---|
| CPU 완료 게시 지연 | 335.0 | 822.0 | CPU 생산 완료 경로가 Hot 완료보다 늦는 경우가 많다. |
| GPU pre-H2D gap | 348.5 | 833.0 | Hot 완료 후 결과 전송 시작을 기다리는 간격이 있다. |
| 위 간격 안 GPU idle union | 348.5 | 832.0 | 간격과 실제 GPU 유휴를 구분하여 비교한다. |
| deferred enqueue→start | 673.0 | 1,186.0 | 이 전체를 전달 비용으로 취급하지 않는다. |
| 대기창 내 선행 task 실행 union | 672.0 | 1,185.2 | 대기가 선행 Cold 서비스와 연결된다. |
| 미귀속 queue gap | 0.8 | 1.5 | 전역 큐 교체가 현재의 주 개선 대상은 아니다. |
| deferred 서비스 span | 921.0 | 1,420.8 | A1·A2가 줄일 직접 대상이다. 순수 산술 시간만은 아니다. |
| 생산자 완료→게시 | 0.5 | 1.0 | 현재 대표값에서 큰 전달 공백은 아니다. |
| H2D 전송 | 26.8 | 32.2 | 서비스 시간과 구분하여 보존·측정한다. |
| NUMA 완료 시차 | 34.8 | 87.5 | job 내부 변경 후 불균형 악화를 점검한다. |

Cold 완료 게시가 늦은 사건 비율은 0µs 기준 91.1%, 5µs 기준 90.8%다. 이는 **해당 사건의 비율**이지 전체 시간의 비율이나 제거 가능한 지연의 비율이 아니다.

expert 표본에서 **rows≤2의 비율은 89.9%**다. slot별 1/16 표본이며, FLOP·바이트·서비스 시간 비중이 아니다. 전체 batch가 크다고 expert별 R까지 큰 것은 아니므로 실제 descriptor로 분기해야 한다. `[R075 §8b; D §5]`

선택 생산층 사례에서 NUMA0의 `up_gate/down` p50은 L20 796/504µs, L21 698/429µs, L42 665/423µs, L13 662/410µs다. 특정 네 층만 수정 대상으로 하드코딩하지 말고 calibration의 전체 유효 분포에서 적용 shape를 선정한다. `[R075 §7 S29; D §5]`

### 1.2 경로를 잘못 해석하지 않는다

소비층 L이 기다리는 Cold 결과는 이 구현에서 생산층 L−1의 deferred 결과와 연결된다. **소비층의 지연을 생산층의 비용에 연결**하고, 상위 FIFO에 있는 선행 작업도 추적한다. 이 계약은 현재 구현의 설명이며 일반 MoE와의 수학적 동등성이 자동 입증됐다는 의미가 아니다. `[D §3; V075 §3]`

TP0가 post-MoE collective의 마지막 도착 rank인 사건은 S29의 해당 cohort 15,872건 모두다. 도착 시차 p50은 435.8µs다. 다른 rank의 긴 커널 duration을 순수 NVLink 전송 비용으로 대체하지 않는다. 이 지표는 CPU 서비스 개선의 후속 영향 확인에 사용한다. `[R075 §8b; D §6]`

### 1.3 비교에 사용할 수 없는 자료

| 자료 | 사용 제한 |
|---|---|
| IDE_074 MAIN 753.16 tok/s | 역사적 성능이다. 신규 바이너리 개선율의 분모로 사용하지 않는다. |
| S29의 689.69 tok/s | CORR 계측 실행이다. 신규 무계측 기준선이 아니다. |
| 서로 다른 `.so`의 OFF 평균 | 같은 구성 반복처럼 통합하지 않는다. |
| CORE2/CORE3의 `probe` | 비동등 클라이언트 간섭 하 자료다. vllm/vllmw 대표 성능·비용 모델에 섞지 않는다. |
| S22의 교차 시계 tail | `clock_valid_by_pair=False`. CPU↔GPU lateness 및 Hot 배치 점수에서 제외한다. |
| S23/S27의 큰 TP 시차 | rank 간 대응을 별도 검증하기 전 대표 인과 근거로 쓰지 않는다. |
| S26/S27의 C1/LONG | `workload_valid=None` 제한이 있다. 새 회귀 게이트를 대체하지 않는다. |

S29에서도 GPU layer 23,870건 중 23,808건이 연결되고 62건이 남는다. 새 집계는 누락의 위치·원인을 보존하고, 단일 `valid=True` 또는 항등식 residual 0으로 모든 대응을 통과시키지 않는다. `[R075 §2~5, §8b; V075 §2, §7, §9; D §8]`

---

## 2. 고정할 구성과 허용 변경

### 2.1 공통 불변 조건

아래는 첨부의 **기준값**이다. 현재 서버에서 동일함을 확인해야 하며 기록 없이 조정하지 않는다. `[H073 §3~5; D §1]`

| 구분 | 고정 내용 |
|---|---|
| 실행 위치 | CPU와 GPU는 동일 물리 서버다. |
| 서버 식별 | 계승 host `violet-h100-016`. 접속 후 실제 확인한다. |
| GPU | H100 80GB 4장, 기존 GPU UUID·TP4 구성 유지 |
| CPU | 2소켓, KT worker 96, NUMA pool 2, 기존 TID affinity 및 non-KT 배치 유지 |
| CPU 정책 | no_turbo=1, performance governor 등 기준 정책 유지 |
| 모델 | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` |
| 모델 revision | `003f183a92fbe5b9a8325aaa8b2ae797c91dd90f` |
| CPU 가중치 | 기존 `/models/kt/qwen3-480b-int4`; 파일·index·scale 해시 고정 |
| 정밀도 | GPU block FP8 / CPU KT INT4. 새로운 재양자화·형식 변경 없음 |
| 논리 구조 | 62층, routed expert 160/층, top-k8, hidden 6,144, intermediate 2,560 |
| 실제 kernel shape | 위 전체 차원이 아니라 런타임의 NUMA shard·tile descriptor로 확인 |
| 실행 기능 | callback-free + skip-empty + deferred8 + 기존 RB |
| 초기 배치 H0 | hotmap v2 + layer budget 5,952슬롯 |
| 메모리 | KV dtype·capacity 40,960 tokens, context 32,768, 기존 workspace·메모리 정책 |
| 스케줄러 | max running 64, decode graph 최대 bs64, prefill graph disabled, chunk8192 등 실제 baseline argv 고정 |
| 데이터 | 고정 custom JSONL, tokenizer revision, token IDs, seed, 요청 순서와 생성 설정 |
| 클라이언트 | 기존 vllm 또는 해당 workload에서 동등성을 확인한 vllmw |

**A1·A2에서는 Hotmap·층별 예산을 모두 고정한다. B에서만 배치 집합과 그에 따른 층별 배분을 바꿀 수 있다.** B에서도 총 논리 Hot 슬롯 5,952, top-k8, CPU96·NUMA2, deferred8, KV, 출력 길이는 유지한다. 로그가 rank별로 반복되어 23,808로 보이는 것을 5,952슬롯과 혼동하지 않는다. 실제 rank별 복제·shard 관계는 loader에서 확인한다.

### 2.2 기존 환경변수의 의미

```bash
# 첨부에 기록된 기존 최적화. 신규 A1/A2 플래그가 아니다.
KT_CALLBACK_FREE=1
KT_CF_SKIP_EMPTY_IMM=1
KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json
KT_AVX_RB=0
```

해당 기존 빌드에서는 `KT_AVX_RB=0`이 ON이다. 기존 RB의 OFF 진단이 필요할 때는 실제 parser를 확인하고 unset 대조를 만든다. **본 A1의 정식 대조군은 RB OFF가 아니라 기존 RB ON**이다. `[H073 §4; D §1, §9]`

‘계측 OFF’는 새 이벤트·profiler를 끈 상태다. 기존 최적화나 A1·A2의 성능 기능까지 모두 끈 상태와 다르다. 기능 활성 상태와 계측 모드를 별도 필드로 저장한다.

### 2.3 허용되는 변경의 경계

| 방안 | 허용 | 금지 |
|---|---|---|
| A1 | 실제 R=1/2·layout·shape에 맞는 기존 RB 내부 경로 특수화, 검증된 load/unpack·주소 계산·입력 접근 중복 감소 | 새로운 정밀도·scale 규칙, 다른 expert의 W 공유, 다음 step 대기로 R 증가, 전체 worker·threshold sweep |
| A2 | 동일 Cold job·NUMA-local pool 안의 descriptor 구성·묶음 크기·출력 tile 소유권 조정 | 상위 FIFO 교체·재정렬, job/step 간 batch, NUMA 간 stealing, 새 OS worker pool, 무검증 K축 부분합 분할 |
| B | 부팅 시 적용하는 정적 Hot 집합 교환, 검증된 범위의 층별 슬롯 재배분 | 모델 실행 중 mask·capture 포인터 교체, 매 토큰 weight migration, KV 축소·GPU 추가로 용량 우회 |

---

## 3. O0 — 현재 기준선·소스·실행 정책을 먼저 고정한다

### 3.1 현재 상태 수집

아래는 읽기 확인의 시작점이다. 실제 호스트·경로를 확인한 실행 환경에서 수행한다. 이 문서 작성 환경에서 실행한 결과가 아니다.

```bash
cd /home/mystous/projects/vllm_hybrid
hostname
git status --short
git rev-parse HEAD
git branch --show-current
git remote -v
```

현재 사용자 수정·활성 서버·컨테이너·마운트·GPU 작업·디스크 여유를 확인한다. 기존 수정은 `reset --hard`, `clean -fd` 등으로 제거하지 않는다. 작업별 고유 branch/worktree와 결과 디렉터리를 만들고 기존 파일을 덮어쓰지 않는다.

계승한 점검 위치는 다음과 같다. **현재 존재 여부·실제 import·로드 경로를 먼저 확인한다.**

| 위치 | 확인할 내용 |
|---|---|
| `/sgl-workspace/ktransformers/kt-kernel/` | 현재 C++ 소스, build 설정, 실제 `.so`와의 대응 |
| `operators/amx/moe_base.hpp` | INT4·FP8 등 타입 분기, RB 진입, gather·quant·gate/up·down·reduction |
| `operators/moe-tp.hpp` 및 실제 NUMA wrapper | shard 분할 축, pool 호출, 합류·merge, result buffer 소유권 |
| `cpu_backend/cpuinfer.h`, 실제 task queue | 상위 FIFO 순서, done 게시, producer/consumer 관계 |
| 실제 `experts_base.py`, `kt_ep_wrapper.py` | Hot/Cold mask, logical map, H2D·결합, graph capture·slot 수명 |
| `eval/ide075/` 및 현재 runner | source/so 기록, client·manifest, v2 계측·수명 검사, 횟수 guard |
| `eval/ide071/configs/workloads.json` 및 원 manifest | MAIN/PROBE/C1/LONG의 데이터·토큰·옵션 |

과거 함수명·행 번호는 탐색 단서다. 현재 소스를 확인하지 않고 문자열 치환 패치를 적용하지 않는다. 새 파일의 실제 위치와 함수는 `CODE_MAP.md`에 기록한다. `[D §17; H073 §13]`

### 3.2 수정 바이너리의 출처

S29의 `.so` 접두사는 `12926df2c30b…`다. 접두사만 맞는 것으로 검증 완료 처리하지 않는다. 현재 사용할 정상성 수정본의 전체 SHA256, source commit+dirty diff, compiler·flags, 빌드 산출물 경로, Python `__file__`, 확장 모듈 실제 로드 경로를 수집한다.

Qwen 최적화의 baseline은 관련 정상성 수정을 포함한 기준으로 고정한다. 이후 공통 FP8 dispatch 또는 scaling 결함을 수정해야 한다면 **정상성 수정 전후와 성능 최적화 효과를 분리**하고 기준선을 다시 만든다. 이전 결함을 둔 바이너리와 후보를 비교해 A1 성능 이득이라고 보고하지 않는다. `[R075 §1~2, §8c; D §13~16]`

### 3.3 기능이 모두 꺼진 신규 빌드의 기준 경로 확인

가능하면 A1·A2를 모두 포함하되 기능 플래그로 기존 경로를 선택할 수 있는 하나의 시험 바이너리를 만든다. 먼저 **원 기준 바이너리 ↔ 신규 바이너리 A1=OFF/A2=OFF/H0**를 비교하여 정상성·기존 경로·성능에 설명되지 않은 변화가 없는지 확인한다.

같은 바이너리를 사용할 수 없으면 분리 빌드의 변경점을 기록하고 compiler·flags·공통 수정 차이를 통제한다. 서로 다른 `.so`를 자동으로 같은 대조군이라고 취급하지 않는다. 플래그가 capture 또는 부팅 때 고정된다면 변형마다 재부팅·재capture한다. 실행 도중 값을 바꿔 live slot이나 graph를 재사용하지 않는다.

### 3.4 새 무계측 기준선

`R0`는 신규 비교의 기준이다. 다음 조건을 만족해야 한다.

- 적용 바이너리, H0 map·budget, 실제 모델·가중치·tokenizer·client, 요청 manifest가 모두 고정되어 있다.
- 부팅 후 정상성 smoke와 동일한 warmup이 통과한다. warmup 실패 뒤 본 측정을 실행하지 않는다.
- MAIN_SHORT의 무계측 원값을 시간·부팅 블록으로 확보한다. 한 번의 가장 높은 값을 기준으로 고르지 않는다.
- PROBE128은 원인 분석용으로 따로 측정한다. MAIN 처리량을 대신하지 않는다.
- C1·LONG의 manifest·token totals·클라이언트 동등성도 확인하여 `workload_valid=None`을 그대로 계승하지 않는다.
- baseline 출력 변동, tensor/logit 비교 방법, 품질 입력과 종료 정책을 보존한다.

자료가 충분한 항목은 재실행하지 않되, **현재 수정 바이너리에 없는 OFF 대조**는 실제로 확보한다. `[D §8, §15~16; V075 §7]`

### 3.5 횟수 guard 제거를 실제로 검증한다

현재 runner·planner·reporter에서 `BUDGET`, 누적 `SESSIONS`/`BOOT`, retry cap, quality cap, 시간 예산으로 종료하는 분기를 조사한다. 필요 실행을 차단하는 횟수 guard를 제거하거나 명시적인 무제한 정책으로 비활성화한다. 숫자를 매우 크게 바꾸는 방식으로 대신하지 않는다.

합성 원장이 과거 상한을 넘었어도 유효한 미해결 작업은 실행 가능해야 한다. 안전·권한·실제 메모리/디스크·다른 작업 보호·무진행 감지는 유지한다. 과거 원장은 수정하지 않고 새 `execution_policy`와 계획 해시를 기록한다. `[P2 §14.7]`

**O0 산출물:** `BASELINE_PROVENANCE.md`, `SOURCE_MANIFEST.json`, `CODE_MAP.md`, `R0/effective_config.json`, `R0/correctness/`, `R0/e2e/`, `EXECUTION_POLICY_TESTS.md`, `PLAN_RESOLVED.md`.

---

## 4. O1 — A1·A2·B의 기능 분리와 실행 증빙

### 4.1 제안 기능 인터페이스 — 구현 전까지는 존재하는 옵션이 아니다

아래는 **신규 실행 계약**이다. 동일 기능의 기존 설정이 있으면 재사용하고 대응표를 남긴다. 미구현 환경변수를 설정한 뒤 기능이 적용됐다고 보고하지 않는다.

| 항목 | 제안 설정 | 의미 |
|---|---|---|
| A1 | `KT_OPT_A1_ENABLE=0 또는 1` | INT4의 지원되는 실제 R/shape에 한해 신규 특수화 사용 |
| A2 | `KT_OPT_A2_ENABLE=0 또는 1` | 지원되는 같은 job의 descriptor 묶음 실행 사용 |
| A1 세부 구성 | 해시가 고정된 kernel policy 파일 | 지원 R/K/N/layout/tile, 구현 revision, fallback 조건 |
| A2 세부 구성 | 해시가 고정된 job policy 파일 | descriptor 구성·분할 단위·소유권·stage별 정책 |
| B | 기존 Hotmap 인자와 layer budget 파일 | H0 또는 검증한 HB를 부팅 시 로드 |
| 계측 모드 | `OFF / CORR / RESOURCE / FOCUS`와 실제 parser 대응 | 기능 활성 상태와 독립적으로 기록 |

신규 boolean은 문자열 존재 여부가 아니라 명시적 0/1로 해석한다. 미지정은 OFF, 잘못된 값은 부팅 전 오류로 처리한다. 기존 `KT_AVX_RB`의 특수한 의미는 변경하지 않는다. 어떤 변형에서도 parse 결과·적용 파일 hash·실제 분기 카운터를 함께 저장한다.

### 4.2 실제 활성화 증거

기능 사용을 증명할 최소 필드는 다음과 같다. 이름은 신규 계약이며 실제 카운터와 일치하도록 구현한다.

```text
variant_id, build_sha256, reference_build_sha256
requested_a1, effective_a1, a1_policy_sha256
requested_a2, effective_a2, a2_policy_sha256
requested_hotmap_sha256, loaded_hotmap_sha256
requested_layer_budget_sha256, loaded_layer_budget_sha256
capture_generation, model_revision, client_sha256

A1: eligible_calls, r1_calls, r2_calls, fallback_calls_by_reason
A2: eligible_jobs, bundled_jobs, descriptors_by_stage,
    internal_dispatch_count, task_tiles_completed, fallback_jobs_by_reason
B : logical_hot_count, hot_count_by_layer, rank_map_digest,
    hot_assignments, cold_assignments, hbm_peak_by_rank, kv_capacity
```

`eligible_calls>0`인데 신규 branch가 0회면 적용 경로를 조사한다. 해당 workload에 지원 shape가 전혀 없으면 `NOT_EXERCISED_FOR_WORKLOAD`로 기록하고 필요한 replay로 기능을 검증한다. environment 문자열과 부팅 로그 한 줄만으로 최적화 활성화를 인정하지 않는다.

성능용 OFF에서는 고빈도 출력·매 호출 문자열 포맷·추가 task를 끈다. 분기 사용 증거는 낮은 간섭의 진단 실행에서 얻고, 실제 성능 바이너리와의 동일성을 보존한다. 디버그 카운터 때문에 기존 task 수·실행 순서가 바뀌면 해당 관측은 무효다.

### 4.3 독립 실행과 fallback

A1=OFF/A2=ON 조합도 동작하도록 설계한다. A2의 descriptor executor는 A1 사용 여부와 별개로 기존 RB 또는 A1 커널을 호출할 수 있어야 한다. B 역시 CPU 실행 정책과 독립적인 정적 배치 입력이다.

지원하지 않는 정상 descriptor는 **기존 정상 경로**로 보낸다. 잘못된 shape·범위·map·scale 포인터 등 데이터 계약 위반을 정상 fallback으로 숨기지 않는다. 부분 출력을 쓴 뒤 기존 경로를 중복 실행하는 fallback은 금지한다. 자격 판단은 출력에 쓰기 전에 끝낸다.

---

## 5. O2 — 실제 작업 replay와 정상성 공통 기반

### 5.1 자료를 다시 사용할 때의 단위

A1·A2의 첫 비교는 **동일 입력과 동일 expert 작업**에서 수행한다. 모델의 생성 token이 바뀌어 다음 routing이 달라지는 효과와 커널 자체의 실행 비용을 구분하기 위해서다. 이후 서비스 E2E는 자연스럽게 실행하여 별도로 검증한다. `[D §10.2, §12.2, §15]`

최소 replay 단위는 다음과 같다.

| 단위 | 보존할 정보 |
|---|---|
| tile/microkernel | dtype, R/K/N, stride, layout, scale axis·group, padding, 실제 NUMA shard, 입력·가중치 해시 |
| expert | logical expert ID, 같은 expert의 모든 실제 rows, gate/up/down, activation·양자화·routing weights |
| Cold job | producer layer·job ID, expert 집합·순서·rows, 두 NUMA 작업, 결과 합류·최종 reduction |
| 연속 job sequence | 실제 순서·shape 변화·가중치 working set·buffer generation, 선행 FIFO 관계 |
| 모델 fixed-token 진단 | 입력 및 고정 token sequence, step별 route IDs·weights, 중간 결과·logits |

요청 입력이나 tensor 원문은 프로젝트가 허용한 저장 위치에만 보존한다. 원본 경로 대신 hash/reference를 사용할 수 있지만, 실제로 재생에 필요한 자료가 누락되어서는 안 된다. 이미 저장된 값이 없으면 해당 질문에 필요한 자료를 새로 수집한다. 평균값으로 입력 tensor나 누락 shape를 만들어 실제 replay라고 부르지 않는다.

### 5.2 cache-hot 시험과 실제 작업 순서 시험

단일 W를 반복 호출하는 microbench는 커널 구조를 확인하는 시험으로 분류한다. 이것만으로 서버의 메모리 접근이나 E2E 이득을 주장하지 않는다. 별도로 실제 expert ID·rows·layer 순서를 재생하여 여러 가중치를 접근하는 환경에서 비교한다.

NUMA-local 메모리의 실제 배치와 서브풀 shard를 그대로 유지한다. 원 replay의 두 소켓 tensor를 임의로 한 소켓으로 모으거나 가중치 사본을 새로 복제해 비용을 바꾸지 않는다. 준비·로딩 시간과 steady-state 연산 시간을 분리하고, 실행 중 발생한 준비 비용은 job 또는 E2E 비용에서 누락하지 않는다.

### 5.3 수치 기준은 후보 측정 전에 고정한다

첨부는 A1·A2·B의 새 `atol/rtol`이나 성공 이득 임계값을 확정하지 않았다. 값을 이미 합의한 것처럼 만들지 않는다. 실행자는 기존 테스트 계약·기준 경로의 반복 오차·dtype별 정상 범위를 확인한 뒤 `ACCEPTANCE_CRITERIA.yaml`에 다음을 등록한다.

```yaml
# 신규 계약 예시. null은 미확정이며, 최종 확인 실행 전 반드시 해소한다.
numerics:
  int4_same_path:
    compare_reference: current_rb
    atol: null
    rtol: null
    max_normalized_error: null
    reduction_order_preserved: true
  unchanged_fp8_bf16_paths:
    compare_reference: corrected_native_path
    atol: null
    rtol: null
  placement_int4_to_fp8:
    compare_reference: frozen_hybrid_and_available_reference
    tensor_logit_limits: null
    quality_noninferiority_rule: null
lifecycle:
  missing_production_allowed: 0
  duplicate_consumption_allowed: 0
  stale_generation_allowed: 0
  premature_publish_allowed: 0
```

A1·A2는 동일 정밀도·scale·누산 계약 보존이 기본이다. 기존 정상 경로가 결정적이고 연산 순서가 같다면 bitwise 비교도 수행한다. 일반적인 ‘부동소수점 차이’라는 설명만으로 큰 오차를 통과시키지 않는다. 허용치를 결과에 맞춰 늘리지 않는다.

B는 CPU INT4와 GPU FP8 사이의 실행 위치를 바꾸므로 bitwise 동등성을 자동 요구하거나 자동 가정하지 않는다. 승인할 정상성·품질 범위를 사전에 정의하고 tensor, logits, tokens, quality를 각각 보고한다. 필요한 기준이 아직 없으면 B의 자료 수집·구현은 진행할 수 있으나 최종 채택은 미루고 기준을 확정한다.

### 5.4 필수 경계 시험

`R=0/1/2`, 관측된 `R≥3`, 현재 AVX/AMX 분기 경계 양쪽, row padding, 비정렬 입력, tail tile, 정상 stride 조합, scale 경계, zero/negative/큰 크기의 정상 입력을 포함한다. 실제 지원 범위를 넘어서는 descriptor는 명시적 입력 오류 또는 정상 fallback 중 올바른 계약을 따르게 한다.

empty Cold, 일부 expert가 empty인 job, 두 NUMA pool, 연속 slot 재사용, decode capture batch 전환, EXTEND↔DECODE 전환, 마지막 요청 drain을 검증한다. `capture rows=64`와 실제 `bs=63`을 혼동해 padding 행을 실제 routing 수량에 더하지 않는다.

**O2 산출물:** `replay_manifest.json`, `descriptors/`, `reference_outputs/`, `NUMERICAL_CONTRACT.md`, `ACCEPTANCE_CRITERIA.yaml`, `replay_coverage.csv`, 초기 타입·수명 테스트 결과.

---

## 6. O3 — A1: 기존 RB 내부의 실제 R=1·2 경로 특수화

### 6.1 목표와 기존 기능 조사

A1의 목표는 기존 RB ON 경로에서 **아직 남아 있는 동적 row loop, 반복 load/unpack, 주소 계산, 입력 재접근, 불필요한 spill·부분합 저장**을 줄이는 것이다. ‘register blocking 추가’ 자체는 새 방안이 아니다. 현재 RB가 이미 R=1·2 특수화와 W 재사용을 수행한다면 같은 기능을 복제하지 않고 실제 잔여 비용을 대상으로 한다. `[D §10~10.2, 그림 6]`

구현 전에 다음 대응표를 작성한다.

| 확인 대상 | 남길 증거 |
|---|---|
| RB 진입 조건·실제 symbol | 현재 source 위치, compile 분기, branch proof |
| blocking 축 | row/output/K 축과 register 사용 방식 |
| 현재 row batching | R=1·2 처리가 정적/동적인지, padding 처리 |
| W load·unpack·scale | 어느 루프에서 반복되는지, 재사용 범위 |
| gate/up 실행 | 서로 다른 W/scale의 독립성과 같은 입력 접근 |
| down 실행 | activation 이후 실제 양자화 입력 및 scale 경계 |
| NUMA shard | full model 차원과 실제 K/N/stride의 대응 |
| asm 또는 compiler evidence | 실제로 남은 분기·주소 계산·load·spill 등 확인 가능한 증거 |

`A1_GAP_ANALYSIS.md`에 기존 기능, 새로 줄일 비용, 변경할 코드 범위, 기대하는 경로 변화, 반증 조건을 적는다. 소스만으로 줄일 비용을 특정하기 어려우면 해당 shape의 좁은 진단을 수행한다. 근거 없이 tile 크기와 compiler 옵션을 전수 탐색하지 않는다.

### 6.2 신규 dispatch의 안전 구조

아래는 실제 ABI가 아니라 **구현 구조 예시**다. 기존 타입·함수·descriptor에 맞게 옮긴다.

```cpp
template <class BufferType>
void run_tile(const Descriptor& d, const Policy& policy) {
    validate_descriptor_contract(d);

    if constexpr (supports_a1_int4_path<BufferType>) {
        if (policy.a1_enabled && supported_layout_and_shard(d)) {
            if (d.actual_rows == 1) {
                specialized_r1(d);
                return;
            }
            if (d.actual_rows == 2) {
                specialized_r2(d);
                return;
            }
        }
    }

    existing_rb_or_corrected_native_path<BufferType>(d);
}
```

이 구조의 핵심은 특수 경로를 지원하지 않는 FP8/BF16 계열에서도 기존 정상 경로가 남는다는 점이다. `else`가 내부 `if`에 결합하여 `if constexpr(false)`일 때 원래 연산까지 제거되는 회귀를 막는다. 모든 조건부 블록에 중괄호를 사용하고 해당 빌드에서 지원되는 타입을 실제 컴파일·실행해 출력까지 검증한다. `[D §14; R075 §8c]`

`R`은 실제 유효 rows이며 padding이나 요청 concurrency가 아니다. 일부 row만 지원된다고 전체 tensor를 조용히 잘라 실행하지 않는다. invalid descriptor는 early validation에서 검출하고, 정상적인 미지원 shape만 기존 경로로 보낸다.

### 6.3 R=1 경로

지원 shape에서 고정 row 수를 활용해 동적 row 반복·주소 계산을 제거하거나 단순화한다. 기존 weight pack/scale 접근과 K축 누산 순서를 보존한다. output tile의 유효 범위와 tail mask를 유지한다.

A1 전후에 같은 tile·입력·가중치로 출력을 비교하고, 실제 코드에서 제거한 비용을 확인한다. 이름만 `r1`인 wrapper가 기존 경로를 그대로 호출하는 경우에는 기능 추가로 계수하지 않는다. dispatcher 자체의 비용까지 포함한 호출 시간과 상위 expert/job 시간을 별도로 기록한다.

### 6.4 R=2 경로

같은 expert의 두 입력 행은 같은 W tile을 사용한다. 현재 RB에 반복 load/unpack이 남아 있는 경우 한 번 준비한 W tile을 두 row의 독립 누산에 재사용한다. `[D §10~10.2]`

다음 조건을 반드시 보존한다.

- 두 row의 입력·scale·출력 accumulator를 분리한다. row0의 scale 또는 출력이 row1에 섞이지 않는다.
- 실제 group/scale 경계별 연산 순서와 rounding·중간 dtype을 유지한다.
- 서로 다른 expert 두 개를 ‘R=2’로 합쳐 W를 공유하지 않는다.
- register pressure·spill·instruction 증가 때문에 잔여 비용이 늘면 tile 구조를 재검토하거나 기존 경로로 보낸다.
- weight 재사용이 실제 DRAM traffic 감소를 의미한다고 사전 선언하지 않는다. 확인된 load/unpack·service 변화만 보고한다.

### 6.5 gate/up 입력 접근 공유

gate와 up은 같은 입력을 사용할 수 있지만 **W와 scale은 서로 다르다.** 현재 구현에 반복 입력 접근이 남고, 각 연산의 수치 계약을 그대로 유지할 수 있을 때 입력 접근 공유를 별도 A1 subvariant로 시험한다.

W/scale 병합, 양자화 경계 이동, activation을 gate/up 완료 이전에 수행하는 변경은 포함하지 않는다. 기존 경로가 이미 입력 접근을 공유하면 중복 구현하지 않는다. R 특수화와 입력 공유를 한 번에 바꿔 효과를 분리할 수 없게 만들지 말고, 개발 중 subvariant와 결과를 연결한다.

### 6.6 output tile·누산 소유권

한 tile 출력의 쓰기 주체를 명확히 한다. 초기 A1에서는 K축 부분합을 여러 worker로 나누어 새로운 reduction을 도입하지 않는다. reduction 순서·alignment·stride를 유지하면서 불필요한 부분합 store/reload를 줄일 수 있는 범위만 다룬다.

두 NUMA 서브풀의 shard 관계가 불명확하면 전체 expert 차원을 가정해서 tile을 생성하지 않는다. 코드 지도와 실제 descriptor를 먼저 연결한다. `hidden=6144`, `intermediate=2560`을 무조건 NUMA-local K/N으로 사용하지 않는다.

### 6.7 A1 구현·검증 순서

| 순서 | 작업 | 통과 증거 |
|---|---|---|
| A1-0 | RB 잔여 비용·지원 descriptor 계약 확정 | gap analysis와 source/symbol map |
| A1-1 | 신규 경로 OFF 및 모든 native fallback 검증 | 기준과 동일한 출력·분기, 정상성 수정 보존 |
| A1-2 | R=1 구현·시험 | 경계 시험, 실제 branch 실행, 오차·시간 원값 |
| A1-3 | R=2 구현·시험 | 두 row 독립성, 실제 W 재사용 범위, spill·오차 확인 |
| A1-4 | 필요한 gate/up 입력 접근 공유 subvariant | 기존 대비 실제 신규 diff와 단계 시간 |
| A1-5 | 전체 expert·Cold job·연속 작업 replay | phase·NUMA별 service p50/p95/p99와 출력 |
| A1-6 | 같은 모드 원인 비교 및 무계측 MAIN | 서비스→대기→사용자 지표 변화 |
| A1-7 | C1/LONG·품질·독립 확인 | 회귀 조건과 재현성 통과 |

기능이 이미 존재해 해당 subvariant가 불필요하면 이유를 남기고 다른 실제 잔여 비용을 확인한다. 단, 새 이득이 없다는 사실을 숨기거나 기존 RB 성능을 A1 성과로 재명명하지 않는다.

### 6.8 A1 채택·기각

최소한 **정상성 통과 + 실제 경로 사용 + 대표 작업에서 서비스 개선 + 무계측 MAIN의 판정 가능한 개선 + 회귀 기준 통과**가 필요하다. microbench만 빠르면 `KERNEL_ONLY_GAIN`으로 남기고 기본 활성화하지 않는다.

R=1만 이득이고 R=2가 악화되면 선택 dispatch를 정식 subvariant로 기록할 수 있다. 둘 다 적용한 것처럼 보고하지 않는다. performance threshold와 noise 정밀도는 10장의 사전 계약으로 판정한다.

**A1 산출물:** 소스 diff·빌드, `A1_GAP_ANALYSIS.md`, `a1_policy.json`, `A1_BRANCH_PROOF.json`, `a1_tile_results.csv`, `a1_job_replay.csv`, 수치·lifecycle 결과, 무계측·관측 실행, `A1_DECISION.md`.

---

## 7. O4 — A2: 동일 Cold job 내부 descriptor 묶음 실행

### 7.1 먼저 A2의 적용 조건을 검증한다

A2는 전역 FIFO를 교체하거나 큐 대기 673µs를 직접 지우는 방안이 아니다. **서비스 span 내부의 반복 parallel 호출·작업 분할·장벽 및 작은 descriptor 처리 비용**을 줄이는 조건부 설계다. `[D §11, 그림 7]`

A1 적용성·정상성을 판정한 후 기준 CPU 정책을 고정하여 내부 구조를 확인한다. A1이 채택되지 않았다면 기존 RB를 기준으로 한다. A1의 개발 결과가 아직 모호하다는 이유로 A2의 source 조사를 누락하지 않는다.

`A2_APPLICABILITY.md`에 다음을 기록한다.

| 질문 | 확인 방법·증거 |
|---|---|
| 현재 어느 단위로 pool 작업을 공급하는가? | job/expert/tile/stage 호출 트리, 실제 source 위치 |
| 반복 parallel 호출이 존재하는가? | job별 호출 수, 같은 입력 replay의 경과 시간 |
| 장벽이 필요한 데이터 의존 때문인가? | stage·buffer readiness DAG, 각 장벽의 선행 조건 |
| 줄일 비용이 서비스 경과 시간에 노출되는가? | parent span과 worker 구간·idle/tail의 연결 |
| 현재 이미 충분히 묶여 있는가? | descriptor 개수·분할·pool 호출 횟수와 코드 |
| 묶음화로 새로운 불균형이 생길 수 있는가? | descriptor별 비용·worker별 완료 시각 |

호출 횟수만 많다고 비용이 크다고 판단하지 않는다. worker time 합계를 parent critical span에 그대로 더하지 않는다. source와 좁은 replay에서 잔여 비용이 없거나 현재 설계와 동등하면 **`NOT_APPLICABLE_WITH_EVIDENCE` 또는 `ALREADY_IMPLEMENTED`**로 판정한다. ‘시간이 없다’, ‘횟수를 다 썼다’, ‘A1이 이미 빨라졌다’는 근거가 아니다.

### 7.2 A2가 유지해야 할 상위 계약

```text
상위 FIFO task 순서                         유지
producer job → consumer layer 연결         유지
done 게시·H2D·결합 순서                      유지
동일 job에 이미 있는 expert/rows             유지
NUMA-local shard 및 기존 worker pool         유지

변경: 현재 job의 stage별 descriptor 구성과 공급 단위
```

다음 요청이나 다음 decode step을 기다려 묶음 크기를 늘리지 않는다. job별 deadline을 재정의하거나 expert 누락·처리 순서 변경으로 처리량만 높이지 않는다. 빈 Cold 경로의 완료·buffer 초기화 규칙도 보존한다.

### 7.3 descriptor 구조와 독점 소유권

다음은 신규 논리 descriptor 명세다. 현재 실제 타입과 연결한다.

```text
job_id, producer_layer, slot_epoch, buffer_generation
numa_id, logical_expert_id, stage
actual_row_begin, actual_row_end
output_tile_begin, output_tile_end
local_K, local_N, strides, scale_layout_id
input_ref, weight_ref, output_ref
owner_assignment, dependency_generation, kernel_policy_id
```

descriptor의 결과 쓰기 범위는 중복되지 않아야 한다. 하나의 stage에서 `(job, numa, expert, row_range, output_tile_range)`의 유효 출력은 한 worker만 소유한다. 다른 worker가 같은 결과를 atomic add로 경쟁해 더하는 구조는 초기안에서 제외한다.

현재 output tiling이 이미 이 원칙을 만족한다면 소유권을 바꾸기보다 descriptor 공급 단위만 최소 수정한다. packing·layout·수치 순서를 바꾸지 않아도 줄일 수 있는 부분부터 구현한다.

### 7.4 stage별 묶음 실행

아래는 구조 예시이며 현재 코드에 없는 함수명을 구현 완료로 해석하지 않는다.

```text
execute_cold_job(job):
    validate_job_and_generation(job)
    for each existing NUMA-local subjob in the existing fork/join structure:
        build_or_reuse_descriptors_for_this_subjob()
        execute_gate_up_descriptors_in_existing_pool()
        wait_for_the_required_gate_up_outputs()
        execute_existing_activation_and_quantization_contract()
        wait_for_the_full_input_domain_required_by_down()
        execute_down_descriptors_in_existing_pool()
        complete_existing_local_reduction()
    join_existing_NUMA_subjobs()
    complete_existing_merge_and_producer_end()
    # done publication stays in its original upper-level path
```

**장벽을 줄이는 것과 필요한 의존성을 제거하는 것은 다르다.** down이 요구하는 K축 입력 또는 양자화 scale 산출 범위가 모두 준비되기 전에 down을 시작하지 않는다. 원본이 expert별 의존성을 사용하면 그 단위를 보존하며, 검증 없이 모든 장벽을 하나로 합치거나 제거하지 않는다.

기존 pool에 이미 있는 worker를 사용한다. nested parallel 호출로 worker가 같은 pool의 미완료 작업을 기다리는 교착이 생기지 않는지 실제 pool의 계약을 확인한다. stage 묶음을 공급하는 부모 실행과 worker 실행 관계를 도식 또는 호출 트리로 저장한다.

### 7.5 묶음 분할 정책

초기 정책은 이미 존재하는 같은 job·같은 NUMA-local descriptor만 대상으로 한다. 단순히 전체 expert를 하나의 거대 작업으로 묶어 병렬성을 줄이지 않는다. 반대로 모든 작은 tile을 독립 OS 작업으로 만드는 구조를 추가하지 않는다.

묶음의 판단 기준은 실제 descriptor 비용·output tile 수·worker별 완료 분포다. 필요하면 적은 수의 근거 있는 정책을 순차 시험하되, 묶음 크기×worker 수×NUMA×커널 종류의 전수 조합으로 확장하지 않는다. 정책 결정 자료·후보 ID·반증 조건을 남긴다.

서로 다른 expert의 descriptor를 같은 공급 묶음에 담는 것과 **그 expert들의 W를 공유하는 것**은 전혀 다르다. 각 descriptor의 logical expert, W, scale, 출력 범위를 그대로 유지한다. NUMA 간 work stealing과 W 복제는 포함하지 않는다.

### 7.6 메모리와 실행 간섭

descriptor 저장소와 scratch는 가능한 기존 job 수명 안에서 재사용한다. 새 준비·정렬·할당 비용이 발생하면 job 시간에 포함한다. 크기 증가가 필요한 경우 현재 buffer capacity·안전한 경계·fallback을 검증한다. 입력이나 expert 수를 임의로 잘라 capacity에 맞추지 않는다.

A2 이후 worker TID 수와 affinity가 기준과 같아야 한다. 동일한 고정 job replay에서는 상위 FIFO task와 producer/consumer 관계가 보존되어야 하며, 내부 descriptor 수 변경과 혼동하지 않는다. 자연스러운 E2E 실행에서는 처리 속도에 따라 scheduler의 실제 batch 구성과 job 수가 달라질 수 있으므로, 총 task 수의 무조건적인 동일성을 요구하지 않고 actual phase/batch와 논리 작업 대응으로 검증한다.

### 7.7 A2 필수 검증

| 항목 | 통과 조건 |
|---|---|
| tile coverage | 유효 출력 범위 누락·중복 0, padding은 명시 처리 |
| stage readiness | activation/quant/down/reduction 선행 조건 위반 0 |
| 수치 | 동일 job 기존 executor와 사전 계약 통과 |
| 상위 순서 | FIFO·done·H2D·consumer 세대 관계 불변 |
| 자원 | 새 OS worker·NUMA 간 stealing·무단 W 복제 없음 |
| 비용 | descriptor 준비를 포함한 service span·tail에서 한계 이득 확인 |
| 독립성 | A1 OFF/ON 각각 올바른 커널을 호출하며 fallback도 정상 |
| 서비스 | 같은 CPU 기준에 A2만 더한 OFF MAIN, C1, LONG 비교 |

서비스 p50이 줄어도 load imbalance로 p95/p99나 사용자 tail이 악화되면 채택 기준에 따라 기각한다. 더 큰 묶음이 항상 좋다는 전제를 두지 않는다.

**A2 산출물:** `A2_APPLICABILITY.md`, 소스 diff·빌드, `a2_job_policy.json`, `descriptor_coverage.csv`, `worker_stage_timeline.csv`, `A2_BRANCH_PROOF.json`, replay·E2E·회귀 원값, `A2_DECISION.md`.

---

## 8. O5~O6 — B: 생산자 비용 기반 정적 Hot 배치

### 8.1 B에 들어가기 전에 CPU 비용을 다시 고정한다

B는 사용 빈도가 높은 expert를 다시 정렬하는 작업이 아니다. **Hot 승격·강등으로 남은 Cold job의 서비스와 GPU Hot 경로가 어떻게 바뀌고, 그 결과 다음 진행 시점이 얼마나 달라지는가**를 평가한다. `[D §12~12.2, 그림 8]`

A1·A2 중 검증된 CPU 구성에 `R_CPU`라는 식별자를 부여한다. 예를 들어 둘 다 통과하면 A1+A2, A2가 적용 대상이 아니면 A1, 둘 다 미채택이면 기존 RB다. 이 선택과 hash를 기록한 뒤 calibration 비용을 다시 수집한다.

**A1 적용 전 서비스 시간으로 A1 적용 후 B를 설계하지 않는다.** 커널 또는 job executor가 바뀌면 기존 B 비용표는 이력으로 남기고 후보 선별에 필요한 비용만 갱신한다. GPU Hot 경로가 고정되어도 placement로 active experts·rows·padding이 바뀔 수 있으므로 GPU 비용도 대응 자료가 필요하다.

### 8.2 calibration과 평가 입력 분리

다음 분리는 **신규 실행 계약**이며 설계 원문의 calibration/held-out 구분을 구체화한다.

| 집합 | 용도 | 사용 제한 |
|---|---|---|
| CALIBRATION | expert routing·rows·CPU/GPU 비용 추정 | 최종 성능·품질 결과로 보고하지 않는다. |
| SELECTION | 후보 배치의 실측 선별·오차 확인 | 한 번 본 결과를 다시 ‘미사용 held-out’으로 부르지 않는다. |
| CONFIRMATION | 동결한 후보의 독립 확인 | 결과를 본 뒤 같은 집합을 사용해 후보를 재튜닝하지 않는다. |
| 고정 MAIN/C1/LONG | 기존 서비스 프로토콜과의 비교·회귀 | 기존에 알려진 입력임을 기록한다. 새로운 독립 입력과 구분한다. |

집합별 prompt hash·token IDs·tokenizer revision·생성 설정·중복 여부를 저장한다. calibration과 고정 평가 manifest를 섞지 않는다. 별도 독립 입력을 사용하면 결과는 고정 MAIN과 합산하지 않고 따로 보고한다.

### 8.3 logical expert와 Hot 집합의 계약

```text
x[l,e] ∈ {0,1}
H = {(l,e) | x[l,e] = 1}
Σ(l,e) x[l,e] = 5952
0 ≤ hot_count[l] ≤ 160
```

위 합은 **논리 expert 슬롯 수**다. 실제 loader의 rank별 복제·shard를 확인하여 HBM 바이트를 별도로 검증한다. 변경 전 H0의 logical↔physical map, 각 layer의 160개 expert 매핑, GPU mask와 CPU 경로의 연결을 보존한다.

후보는 다음 형태의 교환으로 만든다.

```text
promote P: 현재 Cold인 logical expert 집합
 demote D: 현재 Hot인 logical expert 집합
H' = (H \ D) ∪ P
|P| = |D|, P ∩ H = ∅, D ⊆ H
```

같은 층 내부 교환은 층별 슬롯 수를 유지한다. 층 간 교환은 층별 budget 파일을 함께 변경한다. 둘을 실험 기록에서 구분한다. 단순 검증 단계에서 작은 교환을 먼저 확인할 수 있으나, 교환 수에 임의의 총상한을 설정해 필요한 개선 검증을 중단하지 않는다.

서로 다른 층의 같은 expert 번호는 서로 다른 논리 expert다. 물리 슬롯 번호만으로 비용과 배치를 연결하지 않는다. `layer_count=62`, `experts_per_layer=160`, ID 범위·중복·빠진 매핑·Hot count·rank별 digest를 검증한다.

### 8.4 GPU별 메모리 제약

각 rank r에서 다음을 확인한다.

```text
fixed_model_bytes[r]
+ hot_expert_bytes[r, H']
+ kv_bytes[r]
+ workspace_and_graph_bytes[r, H']
+ reserve_bytes[r]
<= usable_hbm_bytes[r]
```

기준선과 같은 KV dtype·capacity를 유지한다. workspace·alignment·padding·graph allocation·일시적 peak까지 확인한다. 메모리 항목은 중복되지 않는 정의로 집계하고 allocator allocated/reserved와 장치 전체 사용량을 별도로 기록한다. 같은 바이트를 여러 통계에 중복 합산하지 않는다. 슬롯 수가 같다는 이유로 메모리가 같다고 선언하지 않는다. OOM을 피하려고 KV, context, running requests, 출력 길이를 낮추는 것은 B의 동일 조건 비교가 아니다.

CPU INT4 backing copy의 존재와 실제 계산 위치를 구분한다. demote한 expert를 CPU가 올바른 가중치·scale로 계산할 수 있는지 확인한다. 필요한 사본이 없으면 원본 형식·loader 계약을 검증하여 준비한다. 파일 일부를 임의 복사하거나 FP8 scale을 INT4 계약으로 잘못 해석해 우회하지 않는다.

### 8.5 비용표 수집 — 빈도와 실제 서비스 비용을 구분한다

최소한 다음 정보를 생산 job 기준으로 연결한다.

```text
calibration_id, manifest_sha256, cpu_policy_id, hotmap_sha256
phase, actual_batch, capture_rows, step_id, producer_job_id
producer_layer, consumer_layer, previous_fifo_task_id
logical_expert_id, hot_or_cold, expert_rows
local_shape, actual_kernel_branch, numa_id
service_span, stage_spans, subpool_completion_times
hot_span, hot_ready, cold_pub, pre_h2d_gap, h2d, combine
mapping_valid, clock_valid, lifecycle_valid, sampling_probability
```

현재 Cold expert만 수집하면 Hot→Cold 강등 비용을 알 수 없다. 강등 후보도 동일 입력·route에 대응하는 CPU replay가 가능해야 한다. 원자료에 필요한 입력·IDs·scale이 없으면 그 후보의 한계 비용은 `UNMEASURED`로 남기고 필요한 자료를 추가 수집한다. 누락 값을 평균 expert 비용으로 확정해 채워 넣지 않는다.

host PCM 대역폭을 expert 수로 나누거나, layer 지연 p50을 expert 하나의 이득으로 사용하지 않는다. 예를 들어 L20의 884.5µs는 그 소비 관계의 관측 지연이지 특정 expert 승격의 절감량이 아니다. `[D §12; V075 §11]`

### 8.6 한계 비용은 남은 Cold 집합 전체로 측정한다

생산 job j에서 다음과 같이 정의한다.

```text
Cj(H)  = Hot 집합 H에서 실제 남은 Cold job의 서비스 비용
Gj(H)  = H에 대응하는 실제 GPU Hot 경로 비용
ΔCj    = Cj(H) - Cj(H')
ΔGj    = Gj(H') - Gj(H)
```

`Cj`는 expert별 시간의 단순 합이 아니다. NUMA 병렬 실행, 같은 job의 descriptor 묶음, kernel branch, empty 상태, prepare·merge가 포함된 실제 service span이다. 후보가 A2의 묶음 구성까지 바꾸면 해당 정책으로 전체 job을 재생한다.

가능하면 동일 입력과 route로 H/H'의 두 job을 교차 재생한다. 전체 후보를 매번 서버 부팅하기 전에 대표 shape·job으로 비용을 선별하되, 계측한 부분만 근거로 사용한다. 추정 비용에는 `measured / interpolated / unavailable` 등 출처와 coverage를 남긴다. 추정값만 있는 경우 최종 채택 전에 해당 비용을 검증한다.

Hot 증가 비용도 GPU의 실제 expert 집합·rows·현재 backend로 측정한다. 현재 Hot 커널 시간에 expert 개수 비례식을 적용해 확정 비용으로 쓰지 않는다. GPU Hot이 늦어져 Cold 대기만 작아진 경우는 실제 개선이 아닐 수 있다.

### 8.7 FIFO·GPU 의존 replay는 후보 선별용이다

설계 원문의 screening 관계를 다음과 같이 사용한다. `[D §12]`

```text
for each task in the preserved FIFO order:
    task_start = max(task_release, previous_task_end)
    task_end   = task_start + task_service_under_candidate

for each validated producer-consumer edge:
    publication is constrained by producer completion,
        the actual done task order and consumer signal readiness
    next GPU readiness also depends on GPU Hot, H2D and other inputs
```

실제 FIFO에 deferred 외의 done·control task가 있으면 그 순서도 반영한다. 모든 job에 `pub=end`를 가정하거나 현재 관측된 queue wait를 고정해서 더하지 않는다. 서비스가 바뀌었을 때 queue wait를 재계산해야 하며, 선행 서비스 감소와 뒤의 queue 감소를 독립 비용으로 중복 합산하지 않는다.

기존 release 시각을 고정한 replay는 **순위 선별용**이다. 실제 실행에서는 GPU 진행과 다음 요청·job의 제출 시각도 달라진다. screening의 출력은 예상 경로 변화·후보 순위·불확실성으로 표시하고 실제 tok/s 또는 보장 speedup으로 보고하지 않는다.

최소한 H0를 재생했을 때 원자료의 동일 사건 관계를 설명하는지 검사한다. H0 자체를 재현하지 못하는 비용 모델로 후보의 정밀 순위를 확정하지 않는다. 모델이 과도하게 복잡해져 새로운 연구 과제로 확장되면 직접 측정 가능한 작은 교환의 비교로 돌아간다.

### 8.8 후보 생성과 순차 검증

1. 전체 유효 calibration에서 **생산 비용이 크고 소비 지연과 연결된** 층·expert 집합을 찾는다. 빈도는 보조 feature로 사용한다.
2. 현재 Hot 중 강등 시 증가할 CPU 비용이 작은 후보를 실제 자료로 확인한다.
3. 승격/강등 쌍 또는 집합을 생성한다. 각 후보에 변경 expert 목록, map diff, slot 수, 비용 출처, 예상 trade-off를 기록한다.
4. map·HBM·backing weight·supported branch를 검증한다.
5. H/H'의 동일 job replay로 Cold 서비스와 Hot 증가 비용을 확인한다.
6. SELECTION에서 정상 부팅·graph 재생성·수치·E2E를 비교한다. 초기 cost model과 다르면 실측으로 갱신한다.
7. CPU 구성과 HB를 동결하고 CONFIRMATION 및 고정 MAIN/C1/LONG을 평가한다.
8. 실제 한계 이득이 없거나 회귀가 있으면 H0 또는 마지막으로 통과한 배치로 원복한다.

각 단계는 실행 횟수의 상한이 아니다. 다만 이미 효과가 없다고 확인된 동일 교환을 근거 없이 반복하지 않는다. 다음 후보는 갱신된 실측 비용이나 새로운 정보로 정한다. 전체 조합의 전역 최적해를 찾았다고 주장하지 않는다.

### 8.9 B의 적용은 새 graph와 함께 부팅 시 수행한다

실행 순서는 다음과 같다.

```text
현재 변형 요청 drain 확인
→ 소유한 서버 프로세스만 정상 종료
→ 후보 H/H' 파일과 전체 해시 검증
→ 실제 loader 형식으로 Hotmap·layer budget 입력
→ logical/physical map·mask·weights·scales 로드 확인
→ 새 graph capture generation 생성
→ smoke·수명·정상성 검사
→ 동일 warmup
→ 성능·진단 실행
```

운영 중인 capture의 포인터나 mask를 고쳐 후보를 바꾸지 않는다. 빈도 기반 초기 파일과 B 파일은 별도 경로로 저장하고 원본을 덮어쓰지 않는다. 기동 옵션을 바꾼 뒤 실제 적용된 layer budget이 다르면 성능을 유효 처리하지 않는다.

### 8.10 B의 수치·deferred 의미 검증

Hot 승격은 CPU INT4 대신 GPU FP8 계산을 선택할 수 있으며 deferred 경로의 참여 집합도 바꾼다. **top-k가 같다는 사실만으로 무손실 배치라고 주장하지 않는다.** `[D §12.2, §14]`

동일 입력·token sequence에서 route IDs/weights, Hot·Cold 기여분, 합성 layer 출력, logits, greedy tokens를 비교한다. 차이가 최초로 나타나는 producer layer·consumer step을 기록한다. logical map 변경으로 다른 expert 가중치를 읽거나 기여분을 두 번 더한 오류와 정상적인 정밀도 차이를 구분한다.

Cold가 사라진 job과 새로 생긴 job, layer 경계, graph 재capture, 마지막 요청·drain의 정확히 한 번 생산·소비를 확인한다. 단순 처리량과 제한된 정답 문항만으로 이 검증을 대체하지 않는다.

**B 산출물:** `CALIBRATION_MANIFEST.json`, `expert_cost_table.csv`, `marginal_job_costs.csv`, `candidate_swaps.jsonl`, `placement_model_validation.json`, `HB/hotmap.json`, `HB/layer_budget.json`, `HB/memory_proof.json`, replay·수치·E2E 결과, `B_DECISION.md`.

---

## 9. O7 — 개별 효과와 A1·A2·B 통합 효과 분리

### 9.1 비교 행렬

아래는 **기능별 조건의 정의**이며 실행 횟수 한도가 아니다. 적용 가능하고 정상성 검사를 통과한 기능은 개별·통합 조건을 비교한다. A2가 증거로 비적용 판정된 경우 관련 행은 이유를 기록하고 제외하되, A1·B의 독립 및 통합 비교는 수행한다.

| variant | A1 | A2 | Hot 배치 | 확인할 내용 |
|---|---|---|---|---|
| R0 | OFF | OFF | H0 | 신규 바이너리의 기존 최적 경로 기준 |
| A1 | ON | OFF | H0 | A1 단독 효과 |
| A2 | OFF | ON | H0 | A2 단독 효과·A1과의 독립성 |
| A12 | ON | ON | H0 | CPU 두 변경의 상호작용 |
| B | OFF | OFF | HB | 고정한 HB의 기존 CPU 경로 대비 효과 |
| A1B | ON | OFF | HB | A1과 HB의 상호작용 |
| A2B | OFF | ON | HB | A2와 HB의 상호작용 |
| A12B | ON | ON | HB | 세 변경이 적용된 통합 조건 |

HB는 `R_CPU`에서 비용을 다시 측정하여 선택한 **동일한 고정 map**이다. 비교 행마다 map을 다시 최적화하면 기능별 효과와 배치 차이가 섞인다. 다른 CPU 정책 전용으로 다시 만든 map을 시험할 경우 `HB_for_R0`, `HB_for_A12` 등 별도 실험군으로 기록하고 위 고정-map 비교와 구분한다.

이 행렬은 모든 불리한 후보에 전체 장시간 검증을 강행하라는 뜻이 아니다. 정상성 실패 조건은 E2E를 차단하고, 단독 성능이 나쁜 구성도 정상성이 통과했다면 조합 효과를 확인할 명확한 질문이 있는 경우에만 필요한 통합 비교를 수행한다. 최종 조합은 개별·통합 결과를 바탕으로 결정한다.

### 9.2 두 종류의 기준선

**총 개선 기준:** 최종 선택 F와 R0의 동일 프로토콜 비교.

```text
total_throughput_gain = throughput(F) / throughput(R0) - 1
```

**추가 변경의 한계 이득:** 예를 들어 A12+HB와 A12+H0의 비교.

```text
B_incremental_gain_on_A12 = throughput(A12B) / throughput(A12) - 1
A2_incremental_gain_on_A1 = throughput(A12) / throughput(A1) - 1
```

모든 비율은 유효한 같은 workload·client·정상성 기준의 짝 또는 블록 비교로 계산한다. 각 기능의 퍼센트를 단순 합산하여 통합 개선율을 만들지 않는다. 소스·공통 수정·compiler가 달라지면 비교 기반을 별도 고정한다.

### 9.3 기능적 통합 검증과 최종 성능 채택은 다르다

세 기능이 각각 정상 동작해도 조합에서 service tail, GPU Hot 경로, workspace, graph pointer, 품질이 바뀔 수 있다. A12B는 자체적인 수치·수명·메모리·성능 검증이 필요하다.

최종 선택이 A1B라면 ‘A1·A2·B 구현/검토 완료, A2는 기각 또는 비적용, 최종 활성 A1+B’로 정확하게 기록한다. 세 항목을 지시받았다는 이유로 효과가 없는 기능을 켜거나, 비활성 기능까지 적용 성과로 포함하지 않는다.

---

## 10. O8 — 성능·통계·비교 계약

### 10.1 워크로드와 목적

다음은 `[H073 §5; D §1, §16]`의 워크로드를 계승한 값이다. 실제 manifest·tokenizer·token IDs와 일치하는지 확인한다.

| workload | concurrency | 요청 수 | 출력/요청 | 입력·출력 합계 기준 | 목적 |
|---|---:|---:|---:|---|---|
| `M073_QWEN_MAIN_SHORT` | 64 | 256 | 128 | 131,786 / 32,768 | 주 성능·최종 채택 |
| `M073_QWEN_PROBE128` | 64 | 128 | 128 | 65,886 / 16,384 | 원인 경로 비교 |
| `M073_QWEN_LOW_CONCURRENCY` | 1 | 16 | 128 | 8,234 / 2,048 | 작은 batch 지연·회귀 |
| `M073_QWEN_LONGER_PREFILL` | 8 | 32 | 128 | 131,147 / 4,096 | 긴 입력·EXTEND와 decode 회귀 |

표의 요청 수와 출력 길이는 **한 workload의 정의**이지 전체 반복 횟수 상한이 아니다. 표와 실제 token 합계가 다르면 입력·tokenizer·실행 옵션을 확인하고 mismatch를 해결한다. 값을 표에 맞추기 위해 실제 로그를 수정하지 않는다.

기존 custom JSONL 대신 generator를 다시 돌려 동일 데이터라고 가정하지 않는다. 성능용 `ignore_eos`·고정 출력128과 품질용 종료 설정을 분리한다. API, prompt template, temperature, top_p, cache flush, warmup을 R0와 후보에 동일하게 적용한다.

### 10.2 무계측 성능과 관측 실행의 분리

최종 처리량·TTFT·TPOT는 **OFF 실행**으로 판정한다. CORR·RESOURCE·FOCUS는 원인 설명용이다. OFF를 만든다고 기존 RB/CF를 해제하거나 compiler에서 다른 연산 경로를 선택하지 않는다.

배경 다운로드·다른 모델 부팅·변환·무거운 profiler·분석기 실행을 성능 창과 겹치지 않게 한다. 모니터가 필요하면 기준·후보에 같은 낮은 간섭 모드로 적용하고 실제 동작을 기록한다. 여러 모델 세션을 별도 프로세스라는 이유로 동시에 실행해 호스트 메모리·I/O 간섭을 허용하지 않는다.

### 10.3 시간·부팅 블록의 교차 비교

기준과 후보를 한쪽 시간대에 몰지 않는다. startup 고정 기능이면 부팅 순서를 균형화하고 블록을 기록한다. 예를 들어 `R0 → 후보 → 후보 → R0`와 반대 순서를 사용할 수 있다. 이는 구현 환경에 맞춰 정할 **신규 설계 예시**이며 필수 부팅 횟수가 아니다.

같은 부팅 반복과 다른 부팅 반복을 분리해 저장한다. 실제로 동일 부팅에서도 변동이 있었으므로 모든 변동을 boot effect로 설명하지 않는다. 각 블록에 실행 순서, 온도·클럭·background state, warmup, graph 상태, client를 남긴다. `[R075 §2~3; D §8, §16]`

### 10.4 반복 수는 정밀도와 필요한 검증으로 결정한다

초기 baseline 자료로 변동 수준과 의미 있는 최소 개선을 정한다. 그에 맞춘 반복·부팅 블록·확인 실행 계획을 후보 결과를 보기 전에 등록한다. 계획 반복 수는 **현재 검증 라운드의 설계**이지 누적 실행 상한이 아니다.

단발 측정에는 표본 SD나 입증된 CI를 붙이지 않는다. 수만 건의 layer 사건을 독립된 서버 반복 수로 세지 않는다. boot/session/step 군집을 유지하고, 재표본화가 필요하면 무엇을 단위로 했는지 명시한다.

유리한 결과가 나올 때까지 돌리다 통과 순간 멈추지 않는다. 탐색 후 후보·설정·평가 입력을 동결하여 독립 확인한다. 정밀도가 부족하면 원인을 확인해 추가 라운드를 설계하고, 관측을 늘려도 질문이 해결되지 않으면 대조·측정 방법을 바꾼다. `[P2 §15; D §16]`

### 10.5 성능 채택 기준의 사전 등록

첨부의 3% 처리량·5% p95 기준은 **계측 간섭의 기술적 분류**다. 최적화 성공 기준으로 자동 전용하지 않는다. 다음 항목을 현재 기준선과 목표에 따라 `ACCEPTANCE_CRITERIA.yaml`에 확정한다. `[D §16]`

```yaml
# 신규 실행 계약. 값은 최초 후보 확인 실행 전에 확정한다.
performance:
  primary_workload: M073_QWEN_MAIN_SHORT
  metric: output_tokens_per_second
  minimum_meaningful_gain: null
  confidence_or_precision_rule: null
  allowed_main_ttft_p95_regression: null
  allowed_main_tpot_p95_regression: null
  allowed_c1_latency_regression: null
  allowed_long_ttft_tpot_regression: null
  tail_and_stability_rule: null
quality:
  fixed_dataset_and_generation_sha256: null
  acceptance_rule: null
memory:
  kv_capacity_must_equal_baseline: true
  logical_hot_slots: 5952
  per_rank_reserve_rule: null
```

정확한 수치가 첨부에 없다는 이유로 필요한 확인을 생략하지 않는다. 실행 가능한 기존 계약을 먼저 확인하고 없으면 baseline 근거로 정한다. 채택 기준 변경이 필요하면 이유·시점을 기록하고 후보를 새 계약에서 다시 평가한다. 이전 결과를 소급하여 통과 처리하지 않는다.

### 10.6 보고해야 할 통계

각 조건별로 반복 원값, 유효/무효 사유, 평균·중앙값·표본 SD(`ddof=1`)·min/max, boot 수, session 수, 입력/출력 총토큰, 완료/실패 수, TTFT/TPOT/ITL/E2EL p50/p95/p99를 남긴다.

반복별 p95의 평균과 pooled p95를 다른 열로 저장한다. 처리량 평균과 전체 토큰/전체 duration으로 계산한 pooled throughput도 구분한다. failed/no-server/0-request 시도를 유효 tok/s=0으로 평균내지 않는다. 실패 자체는 원장과 완료율에서 반드시 보존한다.

---

## 11. 원인 경로 검증 — ‘어디가 줄었는가’를 입증한다

### 11.1 공통 timestamp와 지표

현재 v2 이름과 정의를 보존하고, 새로운 이벤트가 필요하면 schema version을 올린다. 다음 timestamp는 **같은 producer/consumer 사건에 연결된 값**이어야 한다. `[P2 §5, §8~12; D §3]`

```text
A = t_enqueue - t_go
Q = t_start - t_enqueue
C = t_producer_end - t_start
D = t_pub - t_producer_end
O = t_hot_ready - t_go

signed_cpu_lateness = A + Q + C + D - O
cold_pub_lateness   = max(0, t_pub - t_hot_ready)
gpu_pre_h2d_gap     = t_h2d_start - t_hot_ready
h2d_duration       = t_h2d_end - t_h2d_start
```

여기서 O는 기존 문서의 `overlap_budget`에 해당하는 **동시 실행 가능 시간창**이다. 실행 횟수 예산과 전혀 다른 개념이다. A1·A2·B 방안 이름이나 배치 집합 B와 혼동하지 않도록 분석 코드에서도 field를 구분한다.

`cold_pub_lateness`는 CPU 게시 지연이다. GPU가 결과를 실제 소비할 수 있는 H2D 완료와 다르다. H2D가 Hot 이후의 같은 stream에서 시작되는 구조라면 `cold_gpu_ready_late_share=1`만으로 CPU가 항상 병목이라고 판단하지 않는다. GPU idle과 CPU 게시 관계를 함께 본다.

### 11.2 방안별 인과 확인

| 방안 | 직접 비용 | 다음 단계 영향 | 최종 검증 |
|---|---|---|---|
| A1 | 동일 R/shape의 gate/up·down·expert/job service | 선행 FIFO 점유, 게시 lateness, GPU idle | OFF MAIN 처리량·TPOT 및 회귀 |
| A2 | descriptor 준비·내부 dispatch·장벽·worker tail을 포함한 service | parent 완료·FIFO 잔여 점유·GPU idle | A1 상태가 같은 대조군의 OFF MAIN |
| B | 남은 Cold 집합의 service와 증가한 GPU Hot 비용 | 생산/소비 지연·Hot/Cold balance·TP0 도착 | 고정 HB의 OFF MAIN·품질·HBM |

서비스가 줄어도 현장에서 원래 가려져 있던 시간만 줄었다면 E2E 이득이 작을 수 있다. 그 결과는 실패를 숨길 이유가 아니라 최적화의 적용 범위를 설명하는 근거다. 반대로 처리량만 올랐는데 직접 경로 지표가 맞지 않으면 client·batch·routing·공통 build 변화 등 혼입을 확인한다.

### 11.3 계측 유효성은 항목별로 판정한다

적어도 다음을 분리한다.

```text
execution_valid, workload_valid, config_valid, window_valid
lifecycle_valid, task_count_valid, sampling_valid
gpu_activity_valid, cpu_gpu_mapping_valid, tp_rank_mapping_valid
clock_valid_by_pair, producer_consumer_valid
counter_running_valid, counter_scope_valid, artifact_valid
```

A1 비교에는 실제 A1 branch와 지원 shape 표본이 필요하다. A2에는 내부 descriptor 소유권·coverage가 필요하다. B 비용 모델에는 logical expert와 생산 job 연결이 필요하다. 한 지표가 유효하다는 이유로 다른 지표도 통과시키지 않는다.

CPU/GPU clock 오차, ID 누락, graph epoch mismatch를 `max(0)`로 덮지 않는다. 각 지표의 유효 행 mask를 저장하고 같은 비교에서 같은 cohort 정의를 사용한다. ambiguous·invalid 행과 누락된 layer/phase를 별도로 보고한다.

### 11.4 기록기가 계산 대기열을 바꾸지 않게 한다

계측용 stamp task를 상위 FIFO에 추가하지 않는다. hot path에서 파일 출력·문자열 조합·global lock을 추가하지 않는다. 단계마다 `device synchronize`를 호출해 비동기를 직렬화하지 않는다. 현재 검증된 저간섭 기록기를 재사용한다. `[P2 §6; D §15]`

CORR 사용 전 같은 feature 구성의 OFF와 간섭을 확인한다. profiler나 recorder가 변형마다 다르면 직접 서비스 차이를 비교하기 전에 그 변경을 분리한다. 기록 누락과 과부하를 검출하고 계측 무효 값을 성능의 근거로 사용하지 않는다.

### 11.5 time window·NUMA·TP·phase

실제 첫 요청/첫 forward와 마지막 응답/forward 종료를 anchor로 삼는다. `run_bench()`의 클라이언트 로딩 구간을 부하 창으로 계산하지 않는다. PCM 소수초·시간대 파싱, 경계 표본과 완전 포함 표본, perf의 PID/TID 범위·running 비율을 검증한다.

NUMA0/1의 동시 실행 span은 같은 job의 timestamp로 합류·최대 완료·union을 계산한다. 다른 행의 p50을 더해 service 비중을 만들지 않는다. source의 `weight`가 routing weighted reduction이라면 DRAM weight fetch로 재명명하지 않는다.

TP 분석은 `(step/replay, layer, collective position, rank)`를 별도 검증한다. CPU↔GPU mapping 통과가 rank간 mapping까지 보증하지 않는다. Attention/Router/EXTEND/DECODE는 실제 scheduler phase로 나누고 qlen 임계치로 대체하지 않는다.

---

## 12. 정상성·품질·회귀·안정성 게이트

### 12.1 검증 계층

| 계층 | 비교 대상 | 필수 확인 |
|---|---|---|
| 타입·dispatch | 기존 RB/수정 native 대비 A1/A2 | INT4·지원 native 타입, 실제 실행, fallback, 잘못된 constexpr 제거 없음 |
| tile·tensor | 동일 input/weight/scale/descriptor | 유효 출력, error metrics, padding·stride·tail, NaN/Inf |
| expert·Cold job | 동일 expert 집합·순서·routing weights | gate/up·activation·quant·down·reduction, NUMA 합류 |
| layer 합성 | Hot·Cold·잔차·관련 scaling | 누락·중복 합성 없음, scale 위치·횟수, logical map 일치 |
| buffer lifecycle | producer/consumer·slot epoch·generation | 미생산 소비·중복 소비·stale·조기 게시 0 |
| 모델 | 동일 입력 및 fixed-token 진단 | logits·tokens·first divergence, 품질과 독립 보고 |
| 서비스 | 동일 endpoint·생성 옵션·요청 manifest | finish_reason, 오류·절단·timeout, 실제 token counts |
| 지속 실행 | 반복 요청·phase/graph 전환·drain | 메모리 증가, 교착, stale, 오류 재현 입력·이력 |

정상성 실패는 해당 후보의 성능 확인보다 먼저 해결한다. 성능이 높아도 결과가 0이거나 expert 일부를 계산하지 않은 후보는 즉시 기각·복구한다. `[D §14; P2 §13]`

### 12.2 수치 결과에 포함할 값

같은 comparator·동일 입력에서 elementwise 최대 절대 오차, 사전 정의 상대 오차, norm 기반 오차, NaN/Inf 수, 기준값이 0에 가까운 경우의 처리 규칙을 저장한다. 비교식·분모·dtype·반올림 규칙을 명시한다. 서로 다른 오차 지표를 같은 이름으로 바꾸지 않는다.

layer/logit 차이가 생기면 최초 차이의 `(request, token, step, producer_layer, consumer_layer, expert, buffer_generation)`를 가능한 수준에서 연결한다. 기준 자체의 실행 간 변동을 먼저 측정한다. greedy 문자열의 차이를 모두 오류라고 단정하지 않되, 원인을 확인하지 않은 차이를 ‘동등’으로 처리하지 않는다.

### 12.3 exactly-once와 buffer 수명

다음 이벤트의 연결이 필요하다.

```text
input_ready
→ producer_enqueued
→ producer_started
→ NUMA-local work completed
→ producer_completed
→ consumer done published in preserved ordering
→ H2D completed
→ consumer used this generation
→ buffer reusable
```

실제 구현의 사건명·메모리 순서에 맞추되, 완료 전 읽기와 재사용 전 마지막 소비 확인을 유지한다. 디버그용 동기화로만 오류가 사라졌다면 release 성능 경로의 정상성이 입증된 것이 아니다.

A1은 지원하지 않는 타입의 원래 연산을 누락하지 않는지, A2는 descriptor 완료 전 parent가 끝나지 않는지, B는 map 변경 후 이전 generation의 포인터·mask를 참조하지 않는지를 중점 검사한다. empty path가 정상일 때도 초기화·신호·소비 규칙을 확인한다.

### 12.4 C1·LONG 회귀

C1과 작은 decode batch에서는 Cold가 이미 가려지는 경우가 많다. A1/A2의 dispatcher·metadata 비용이 작은 workload를 악화시키지 않는지 확인한다. `[D §7; R075 §5 S26]`

LONG은 전체 처리량뿐 아니라 실제 EXTEND 그룹, 긴 prefill의 TTFT, decode TPOT를 분리한다. 4K와 8K EXTEND 관측의 차이를 근거 없이 chunk 설정 개선으로 전환하지 않는다. 기존 chunk·mixed 정책은 고정한다. 새 최적화가 공용 CPU 경로를 바꾸면 prefill의 fallback·오차도 확인한다. `[D §7; R075 §8b]`

### 12.5 품질 문항과 생성 종료

기존 품질 입력·정답·tokenizer·template·max tokens·stop 설정을 먼저 읽고 보존한다. 같은 입력의 baseline/candidate 출력을 전문·token IDs·finish_reason·정답 채점과 함께 저장한다.

정답률, 텍스트 일치율, tensor/logit 차이, 요청 완료율, 절단·오류는 별도 지표다. 제한된 4문항이나 20문항 통과만으로 모델 전체 무손실을 선언하지 않는다. 필요한 품질 문항·반복에는 횟수 상한을 적용하지 않는다.

### 12.6 안정성 시험의 설계

사전 등록한 안정성 질문을 검증할 만큼 요청·반복·지속 구간을 확보한다. 동일 cache-hot kernel 반복만으로 장시간 서비스 안정성을 대체하지 않는다. 정상 cold/nonempty, graph batch 전환, EXTEND↔DECODE, 반복 drain/재요청, buffer generation 재사용을 포함한다.

정상성이 보장된 입력 범위 안에서 메모리 사용과 오류율·tail을 추적한다. timeout은 무진행·교착의 복구 수단으로 두되 누적 실행 시간을 이유로 전체 검증을 포기하는 상한으로 쓰지 않는다.

오류 발생 시 request 원문 또는 재현 가능한 안전한 참조, token IDs, seed, variant/build/map hash, 직전 job·slot·generation, 전체 stdout/stderr를 저장한다. 원인을 고치지 않은 같은 오류 반복 대신 재현 조건을 좁힌다.

---

## 13. GLM 및 공통 native 경로의 회귀 보호

이번 구현의 주 최적화 대상은 Qwen이다. GLM용 INT4 변환, GLM Hotmap 재설계, 신규 모델 확장은 이번 A1·A2·B 구현에 자동 포함되지 않는다. **공통 C++/Python 경로를 수정하여 GLM native FP8 정상성을 다시 깨뜨리지 않는 회귀 검사**는 포함한다. `[D §13~14]`

첨부는 GLM의 dispatch·scaling 결함을 수정한 D8에서 제한된 정상 출력 게이트 4/4, GPU8 기준 텍스트 일치 3/4를 보고한다. D9의 callback-free 추가 경로는 게이트 4/4지만 텍스트 일치 1/4이므로 별도 변형이다. 단독 FP8 시험의 상대 L1 0.5829%를 모든 tensor·모델의 새 허용 오차로 재사용하지 않는다. `[R075 §8c; D §13]`

공통 코드 변경 시 다음을 확인한다.

| 항목 | 요구 |
|---|---|
| 타입 fallback | A1 지원 타입이 아니어도 수정 native path가 실제로 실행된다. |
| 입력 준비 | FP8/BF16 계열의 gather·quant 등 필요한 연산이 compile 분기로 제거되지 않는다. |
| scaling | GPU 기여·CPU 기여·합성 출력에 원 계약대로 적용하고 이중 적용하지 않는다. |
| 토크나이저 | GLM 시험은 GLM tokenizer·revision·입력 manifest를 사용한다. |
| 기준 경로 | D8 해당 수정의 실제 소스·출력 계약을 확인하고 D9를 몰래 baseline으로 바꾸지 않는다. |
| 시험 범위 | 타입 단독·공통 wrapper 및 필요한 native smoke를 수행한다. 무관한 GLM 장시간 성능 sweep은 하지 않는다. |

공통 코드가 바뀌지 않은 부분의 모든 GLM 실험을 재실행할 필요는 없다. 하지만 영향받는 native 타입·wrapper의 필수 회귀는 생략하지 않는다. GLM 접근 환경이 없다면 실제 차단과 미검증 범위를 기록하고, 해당 공통 변경의 최종 채택 조건에 반영한다.

---

## 14. 전체 실행 순서와 단계별 완료 증거

아래 O0~O10은 **작업 단계**다. 각각 한 세션으로 끝내거나 정해진 부팅 횟수 안에서 수행하라는 의미가 아니다.

| 단계 | 실행 내용 | 다음 단계로 넘길 증거 |
|---|---|---|
| O0 | 실효 소스·수정 바이너리·정상성·새 OFF 기준선 | baseline provenance, 전체 hash, MAIN/C1/LONG manifest, 정책 확인 |
| O1 | 기능 플래그·기존 경로·branch proof 분리 | A1/A2 독립 ON/OFF, H0/HB 입력 계약, 유효성 검사 |
| O2 | 실제 descriptor·job replay·수치 계약 | reference outputs, replay coverage, 사전 수치·성능 기준 |
| O3 | A1 구현·수치·replay·E2E·회귀 | 실제 신규 diff, branch 사용, A1 판정 |
| O4 | A2 적용성 확인·구현·단독/조합 검증 | 내부 비용·tile coverage, A2 또는 비적용 판정 |
| O5 | R_CPU 확정·calibration 분리·비용 재측정 | 새로운 CPU 정책에 대응하는 비용표·독립 데이터 hash |
| O6 | B 후보 교환·메모리·replay·정상성·선별 | HB와 map diff, 한계 비용·HBM·quality proof |
| O7 | 고정 HB와 기능 조합 검증 | 적용 가능한 개별/통합 셀 및 상호작용 결과 |
| O8 | 최종 후보 동결·독립 OFF 확인·C1/LONG | 반복 통계·정밀도·회귀·품질·안정성 |
| O9 | 최종 선택·기각·미적용·한계·rollback 검증 | ACCEPT/REJECT 등 판정, 실제 재현 명령 |
| O10 | 원자료·코드·보고서·접근성 인계 | 파일 검증·hash·게시 상태·실제 전달 |

O3 이후에도 A2·B의 필수 질문이 남으면 필요한 작업을 계속한다. A2가 비적용 판정되면 O5의 R_CPU에서 제외하고 B로 진행한다. 공통 정상성 결함이 발견되면 해당 기준선을 수정·재검증하되, 수정 전후 결과를 같은 모집단으로 섞지 않는다.

### 14.1 작은 질문으로 진척을 관리한다

매 실행이나 추가 반복 전에 다음을 기록한다.

```text
question_id
optimization_id: A1 | A2 | B | INTEGRATION | COMMON
existing_evidence
evidence_gap
why_existing_data_is_insufficient
change_or_repeat_reason
planned_observation
how_the_result_changes_the_decision
required_dependencies
```

동일 조건 반복도 변동·재현성·정밀도 확인이라는 질문이 있으면 유효하다. 결과가 다음 결정을 바꾸지 않는 추가 계측은 진행하지 않는다. 미해결 원인을 핑계로 모델·하드웨어 전체 조사로 확장하지 않는다.

### 14.2 시작 가능한 일과 선행 조건이 있는 일

A1/A2의 source 조사, replay harness, config validator, 데이터 집합 분리, map 검사기는 서로 필요한 범위에서 준비할 수 있다. 실제 성능 부하와 무거운 다른 작업은 동일 호스트에서 병행하지 않는다.

B의 탐색기 구현을 준비하는 것은 가능하지만 **최종 비용표·배치 선택은 R_CPU 비용을 고정한 뒤** 수행한다. 정상성 실패 후보의 본 E2E 실행은 차단하고, 다른 독립 테스트와 결과 보존을 계속한다.

---

## 15. 상태·계속·종료 판정

### 15.1 기능 상태와 실행 상태를 분리한다

아래는 신규 상태 계약이다. 현재 reporter에 같은 의미의 상태가 있으면 매핑을 명시한다.

| 상태 | 의미 | 완료·채택 처리 |
|---|---|---|
| `PLANNED` | 구현·검증 계획만 존재 | 완료 아님 |
| `IMPLEMENTED_UNVALIDATED` | 코드가 있으나 필요한 검증 전 | 활성 채택 금지 |
| `VALIDATED_FUNCTIONAL` | 수치·수명·기능 검증 통과 | 성능 검증을 대신하지 않음 |
| `KERNEL_ONLY_GAIN` | microkernel/replay 이득만 확인 | 서비스 최적화 채택 아님 |
| `ACCEPTED` | 사전 정상성·성능·회귀 조건 통과 | 최종 활성 후보 가능 |
| `REJECTED_REGRESSION` | 정상성 또는 성능·회귀 기준 위반 | 원복, 원인·자료 보존 |
| `NO_MEANINGFUL_GAIN` | 충분한 판정 근거에서 실용 이득 없음 | 기본 OFF, 과장된 성공 금지 |
| `NOT_APPLICABLE_WITH_EVIDENCE` | 특히 A2의 적용할 잔여 비용·구조가 없음 | source·측정 증거가 있어야 종결 가능 |
| `ALREADY_IMPLEMENTED` | 설계의 해당 기능이 현재 기준에 이미 존재 | 신규 구현·신규 이득으로 계수 금지 |
| `INCONCLUSIVE` | 필요한 판정 정밀도·증거 부족 | 질문·설계를 보완한다. 성공 처리 금지 |
| `BLOCKED_ENVIRONMENT` | 실제 권한·환경·원자료 문제 | 영향 범위 명시, 독립 작업은 계속 |
| `BLOCKED_CORRECTNESS` | 수치·수명 문제로 후속 부하 불가 | 원인 수정·복구 전 본 성능 실행 금지 |
| `NOT_EXERCISED_FOR_WORKLOAD` | 기능에 맞는 작업이 해당 실행에 없음 | 다른 필수 replay 또는 적용 범위 확인 필요 |

`BLOCKED_BY_BUDGET`, `SESSION_EXHAUSTED`, ‘추가 횟수 승인 필요’를 새로운 실행 상태로 사용하지 않는다. 과거 기록에는 원문을 남길 수 있으나 현재 진척과 분리한다.

### 15.2 언제 계속하는가

필수 구현·검증 공백이 남고 해결할 방법과 실제 권한·환경이 있으면 계속한다. 한 라운드의 반복이 끝났다고 미해결 기능을 자동 종결하지 않는다. 반대로 이미 부정된 동일 구현을 이득이 나올 때까지 반복하지 않는다.

`INCONCLUSIVE`가 반복되면 기준·후보의 동등성, client·batch 분포, clock/mapping, 소스·비용 정의를 먼저 재검토한다. 필요한 경우 진단을 좁혀 다시 설계하고 그 이력을 저장한다. 데이터가 더 필요하다는 근거가 있으면 추가 수집한다.

### 15.3 전체 작업 완료의 정확한 의미

A1·A2·B 각각에 대해 구현·검증 결과 또는 증거 있는 비적용·기각 판정이 존재하고, 적용 가능한 통합 검증과 최종 선택·원복·원자료 인계가 완료되어야 한다. 필수 공백이 남아 `INCONCLUSIVE` 또는 실제 차단인 경우 전체를 성공 완료로 표시하지 않는다.

**모든 기능의 채택**과 **모든 요청 항목의 평가 완료**는 다르다. A2가 실제 비적용임을 확인했다면 구현을 억지로 추가할 필요가 없지만, 증거 없이 ‘나중에’로 넘긴 상태는 평가 완료가 아니다.

---

## 16. 결과 저장·보고·재현 규격

### 16.1 신규 디렉터리 제안

아래 경로는 신규 계약 예시다. 실제 campaign ID는 기존 ID와 충돌하지 않게 시작 시 정한다. IDE 번호가 비어 있다고 가정하지 않는다.

```text
shadow_assists/features/<NEW_ID>/
  PLAN.md
  PLAN_RESOLVED.md
  SOURCE_INDEX.md
  SOURCE_MANIFEST.json
  CODE_MAP.md
  BASELINE_PROVENANCE.md
  CONFIG_DIFF.md
  ACCEPTANCE_CRITERIA.yaml
  WORK_LOG.md
  PROGRESS_EVENTS.jsonl
  A1_GAP_ANALYSIS.md
  A2_APPLICABILITY.md
  B_COST_MODEL.md
  FULL_REPORT.md
  ANALYSIS_AND_DECISION.md
  A1_DECISION.md
  A2_DECISION.md
  B_DECISION.md
  INTEGRATION_DECISION.md
  COMPLETION_STATUS.md
  ROLLBACK.md
  FULL_RAW_DATA.md
  ARTIFACT_INDEX.csv
  SHA256SUMS.txt
  PUBLISH_RECEIPT.md
  raw_md/part-*.md

optimization/<campaign>/
  source/                  reference/candidate source diffs and build records
  configs/                 a1/a2 policies and effective argv/env
  calibration/             split manifests, expert/job cost data
  placements/H0/           frozen original map and layer budget
  placements/HB_<id>/      candidate map, diff, memory and numerical proof
  variants/<variant>/
    manifest.json
    feature_proofs.json
    correctness/
    replay/
    e2e/<boot>/<workload>/<rep>/
    diagnostics/<boot>/<workload>/<rep>/
    DECISION.md
  state/                   attempts, questions, decisions and resumable state
```

### 16.2 모든 실제 실행의 최소 산출물

```text
requested_config.json
effective_config.json
launch_cmd.sh
bench_cmd.sh
environment.json
build_manifest.json
data_manifest.json
feature_proofs.json
timestamps.json
metrics.json
request_results.jsonl
stdout.log
stderr.log
validation_results.json
```

오류 시 `error.json`, `failing_request.json` 또는 안전한 재현 참조, 관련 buffer/event slice를 추가한다. 상세 로그를 `tail/head`만 저장해 원본을 잃지 않는다. source diff와 `.so`·policy·map hash가 모든 실행에서 연결되어야 한다.

### 16.3 분석기 입력과 재현

분석은 원시 CSV/JSONL과 manifest를 입력으로 다시 생성할 수 있어야 한다. 수치 표를 사람이 편집하여 맞추지 않는다. 분석기 revision, 필터, cohort 정의, 시간 단위, precision, 누락 이유, sampling probability, resampling 단위를 저장한다.

원자료가 없거나 schema가 맞지 않으면 명시적으로 실패한다. 필요한 열을 0으로 채워 실행하지 않는다. `null`, `not_applicable`, `not_measured`, `invalid`를 구분하고, 같은 지표의 분모가 후보별로 달라졌을 때 변화를 보고한다.

### 16.4 원 측정 보고서와 판단 문서를 분리한다

`FULL_REPORT.md`는 환경·실효 설정·모든 반복 원값·유효성·수치·오류·미실행·파일 목록을 기록한다. 예상 speedup, 우수성 주장, 다음 제안은 원값 표에 섞지 않는다.

`ANALYSIS_AND_DECISION.md`에는 A1·A2·B의 가설, 실제 줄인 경로, 반증·혼입·한계, 각 기능의 단독/조합 효과, 채택·기각 이유를 적는다. `INTEGRATION_DECISION.md`에는 최종 활성 조합과 비활성 기능, 전체 이득과 한계 이득을 분리한다. `[H073 §14; D §17]`

### 16.5 진척 보고와 실제 전달

실행 환경이 지원하는 전달 수단으로 준비 시작부터 30분 간격의 진행 기록과 주요 전환·오류를 보고한다는 기존 요구를 계승한다. 파일 저장과 실제 전달을 구분하고, 전달 수단이 없으면 `SAVED_NOT_DELIVERED`로 기록한다. **이 문서 생성은 정기 알림이나 서버 작업을 생성하지 않는다.** `[H073 §15]`

진척에는 완료한 질문, 다음 판단에 필요한 공백, 활성 variant/build, 마지막 유효 결과, 오류·복구, 실제 프로세스·자원 상태를 적는다. `remaining_sessions` 대신 `unresolved_required_questions`를 사용한다.

### 16.6 게시·파일 검증

현재 저장소 권한과 실제 프로젝트의 게시 절차를 확인한 실행자만 승인된 branch에 게시한다. 비밀·credential·개인정보·대형 raw tensor를 무조건 stage하지 않는다. 강제 push나 사용자 수정 삭제는 하지 않는다.

게시 전에 JSON/CSV 파싱, 필수 파일·참조, hash, 원자료 coverage와 결과 재생성을 검사한다. 게시한 경우 data commit·remote SHA·원격 파일 확인을 기록한다. 게시 실패나 접근 불가는 그대로 남기되 로컬 결과 인계는 계속한다. `SAVED`, `PUBLISHED`, `DELIVERED`를 같은 상태로 취급하지 않는다.

---

## 17. 원복·장애 복구·최종 재현

### 17.1 원복 대상

각 변형을 적용하기 전에 다음 복구점을 만든다.

```text
reference source and diff
reference loaded .so and full SHA256
reference Python module paths
H0 hotmap and layer budget with hashes
reference argv/env and their parser results
reference KV/workspace/graph settings
reference worker/TID affinity method
baseline correctness and request manifests
```

A1·A2는 플래그 OFF 경로와 이전 바이너리 복구를 구분한다. 공통 source 자체가 손상됐으면 플래그만 끄는 것으로 충분하지 않다. B는 원본 map 파일을 덮어쓰지 않고 H0를 다시 지정해 새 부팅·capture로 복구한다.

### 17.2 실패 처리 순서

1. 실패한 variant·request·build·map·generation 및 전체 로그를 먼저 보존한다.
2. 더 진행하면 데이터 오염·교착·다른 작업 침해가 생기는 자신의 프로세스만 중단한다.
3. 영향받는 경로를 known-good reference로 복구한다.
4. 짧은 정상성·수명 검사를 통과한 뒤 필요한 비교를 재개한다.
5. 수정 내용·원인 가설·재시도 이유를 기록하고 같은 무수정 오류를 반복하지 않는다.

실험을 빨리 끝내려고 warmup·품질·lifecycle 검사를 생략하거나 global `pkill`로 다른 서버를 종료하지 않는다. 부팅 횟수를 줄이려고 실행 중인 graph에 새로운 pointer/map을 끼워 넣지 않는다.

### 17.3 최종 재현 패키지

최종 선택한 variant는 새로운 안전한 실행 환경/부팅에서 동일 명령으로 재현할 수 있어야 한다. 실제 실행된 `launch_cmd.sh`와 `bench_cmd.sh`를 저장하고, 문서의 의사코드를 성공 명령으로 대체하지 않는다.

다음 사람이 알아야 할 최소 내용은 **어느 source와 바이너리를 사용하고, A1/A2의 어떤 policy를 켜며, 어느 Hotmap을 로드하고, 어떤 입력으로 검증했고, 어떤 기능은 왜 껐는가**다. 원복도 같은 수준으로 구체적이어야 한다.

---

## 18. 최종 완료 체크리스트

### 18.1 공통

- [ ] 현재 source·전체 `.so` hash·import 경로·model·client·manifest가 고정되어 있다.
- [ ] 새 수정 바이너리의 R0 OFF·정상성·C1/LONG 기준이 존재한다.
- [ ] 신규 플래그가 실제 구현되어 있고 OFF/ON·잘못된 값·fallback이 시험됐다.
- [ ] 세션·부팅·재시도·품질의 횟수 guard를 제거했으며 안전 검사 유지가 확인됐다.
- [ ] 수치·성능·회귀·품질·메모리 판정 기준이 후보 확인 전에 등록됐다.
- [ ] 데이터·phase·clock·mapping·lifecycle의 유효성을 별도로 검사했다.

### 18.2 A1

- [ ] 기존 RB와 중복되지 않는 실제 잔여 비용을 확인했다.
- [ ] R=1·2 지원 경로와 미지원 fallback의 실제 분기 증거가 있다.
- [ ] W/scale·양자화·누산·NUMA shard 의미가 보존됐다.
- [ ] microkernel, 실제 expert/job sequence, E2E를 구분해 검증했다.
- [ ] 최종 A1 정책과 채택/기각·이미 구현된 기능의 구분이 있다.

### 18.3 A2

- [ ] 적용성 판단에 내부 호출 구조와 실제 비용 증거가 있다.
- [ ] 상위 FIFO·job 순서·done·producer/consumer 계약이 유지됐다.
- [ ] tile 소유권·누락/중복·stage readiness·NUMA 합류를 시험했다.
- [ ] 같은 job만 묶으며 다음 요청 대기·NUMA stealing·새 pool을 도입하지 않았다.
- [ ] A1과 독립 동작 및 필요한 조합을 확인했다.
- [ ] 미적용이면 실제 코드·측정 근거가 있고 단순 미완료를 비적용으로 바꾸지 않았다.

### 18.4 B

- [ ] R_CPU를 고정한 뒤 비용을 재측정했다.
- [ ] calibration·selection·confirmation과 고정 평가 입력의 역할·hash를 구분했다.
- [ ] 승격과 강등 후보 모두 실제 비용·backing weight·logical map을 확인했다.
- [ ] 총 5,952슬롯과 각 GPU의 실제 HBM·KV·workspace를 검증했다.
- [ ] 비용 모델에서 전체 Cold job·Hot 증가·FIFO 전파를 고려하고 중복 계수하지 않았다.
- [ ] 후보는 부팅 시 정적으로 적용했으며 graph를 정상 재생성했다.
- [ ] INT4↔FP8 경로 및 deferred 집합 변화의 수치·품질·수명을 검증했다.
- [ ] HB와 H0의 실제 E2E 비교 및 B 판정이 있다.

### 18.5 통합·전달

- [ ] 적용 가능한 R0/A1/A2/A12/B/A1B/A2B/A12B의 개별·통합 비교가 있다.
- [ ] 같은 고정 HB 비교와 CPU 정책별 재선정 배치를 구분했다.
- [ ] 탐색 뒤 후보를 동결한 독립 확인·MAIN/C1/LONG·품질·안정성 결과가 있다.
- [ ] 총 개선율과 추가 기능의 한계 이득을 분리하고 퍼센트를 합산하지 않았다.
- [ ] 원값 보고서와 분석·판정 문서가 구분되어 있다.
- [ ] 실패·미적용·무효·미검증 항목을 숨기지 않았다.
- [ ] 최종 활성 기능, 비활성 구현, 미적용 기능, 실제 차단을 명시했다.
- [ ] 원복·최종 재현 명령, 파일 존재·파싱·hash·원자료 접근을 확인했다.
- [ ] 실제 게시·전달 여부를 기록했고 필요한 최종 파일을 제공했다.

---

## 부록 A. 실행자용 최소 데이터 스키마

아래 필드명은 신규 계약이다. 현재 구현의 schema와 다르면 이름 대응표와 버전을 고정한다. 존재하지 않는 값을 빈 문자열·0으로 채워 검증을 통과시키지 않는다.

### A.1 `variant_manifest.json`

```json
{
  "schema_version": "a1-a2-b-v1",
  "campaign_id": "<unique_id>",
  "variant_id": "<R0|A1|A2|A12|B|A1B|A2B|A12B>",
  "status": "PLANNED",
  "execution_policy": "evidence_driven_no_count_cap",
  "plan_sha256": "<full_sha256>",
  "source_commit": "<commit>",
  "source_diff_sha256": "<full_sha256>",
  "binary_sha256": "<full_sha256>",
  "reference_binary_sha256": "<full_sha256>",
  "compiler_manifest_sha256": "<full_sha256>",
  "features": {
    "a1_requested": false,
    "a1_effective": false,
    "a1_policy_sha256": null,
    "a2_requested": false,
    "a2_effective": false,
    "a2_policy_sha256": null
  },
  "placement": {
    "name": "<H0_or_HB_id>",
    "hotmap_sha256": "<full_sha256>",
    "layer_budget_sha256": "<full_sha256>",
    "logical_hot_slots": 5952,
    "cost_basis_cpu_variant": "<cpu_variant>",
    "cost_basis_binary_sha256": "<full_sha256>"
  },
  "model_revision": "003f183a92fbe5b9a8325aaa8b2ae797c91dd90f",
  "cpu_weight_manifest_sha256": "<full_sha256>",
  "tokenizer_manifest_sha256": "<full_sha256>",
  "client_sha256": "<full_sha256>",
  "workload_manifest_sha256": "<full_sha256>",
  "acceptance_criteria_sha256": "<full_sha256>"
}
```

`<...>`은 실행 전 채워야 하는 자리표시자다. 실행기는 미확정 hash·실효 설정 mismatch를 검출해야 한다. JSON이 파싱된다는 사실만으로 실행 준비가 됐다고 판단하지 않는다.

### A.2 `tile_results.csv`

```text
variant_id,baseline_variant_id,build_sha256,policy_sha256,
replay_id,descriptor_id,logical_layer,logical_expert_id,numa_id,
stage,actual_rows,K,N,strides,scale_layout,padding,
kernel_branch,eligible,fallback_reason,cache_mode,
repeat_id,elapsed_ns,max_abs_error,normalized_error,
nan_count,inf_count,numerical_valid,descriptor_valid
```

### A.3 `job_replay_results.csv`

```text
variant_id,build_sha256,hotmap_sha256,replay_manifest_sha256,
sequence_id,job_id,producer_layer,consumer_layer,phase,
actual_batch,capture_rows,slot_epoch,buffer_generation,
expert_set_digest,rows_digest,kernel_policy_id,job_policy_id,
service_span_us,up_gate_us,down_us,prepare_us,
numa0_span_us,numa1_span_us,numa_completion_skew_us,
internal_dispatch_count,descriptor_count,
correctness_valid,lifecycle_valid,coverage_valid
```

단계 시간이 서로 중첩되면 `*_us`의 측정 정의와 `inclusive/exclusive/union`을 별도 사전에 남긴다. 표준 이름만으로 합산 가능하다고 가정하지 않는다.

### A.4 `placement_candidate.jsonl`

```text
candidate_id,parent_map_sha256,candidate_map_sha256,
cpu_cost_basis_variant,cpu_cost_basis_binary_sha256,
calibration_manifest_sha256,selection_manifest_sha256,
promote_logical_experts,demote_logical_experts,
logical_slot_count,layer_budget_sha256,
measured_cost_references,estimated_cost_references,cost_coverage,
screening_result,screening_limitations,
map_valid,memory_valid,numerical_valid,lifecycle_valid,
selection_result,confirmation_result,decision,evidence_paths
```

후보 파일은 원장이므로 실패한 후보도 삭제하지 않는다. confirmation을 실행하지 않았으면 미측정 상태로 남기며 selection 수치를 복사하지 않는다.

### A.5 `e2e_comparison.csv`

```text
campaign_id,comparison_block_id,comparison_role,
variant_id,reference_variant_id,boot_id,session_id,order_index,
binary_sha256,hotmap_sha256,policy_sha256,client_sha256,
workload_id,data_manifest_sha256,mode,
sent,completed,failed,input_tokens,output_tokens,duration_s,
output_tps,ttft_p50_ms,ttft_p95_ms,ttft_p99_ms,
tpot_p50_ms,tpot_p95_ms,tpot_p99_ms,
itl_p95_ms,e2el_p95_ms,hbm_peak_by_rank,
execution_valid,workload_valid,config_valid,window_valid,
quality_reference_id,invalid_reason,raw_metrics_path
```

### A.6 `DECISION.md` 템플릿

```markdown
# <A1|A2|B|INTEGRATION> 판정

## 판정 대상
variant / source / .so / policy / map / workload / client / 기준군

## 실제 변경
기존 기능과 구별되는 source diff, 실제 분기 사용, 미지원 fallback

## 사전 계약
수치·수명·성능·회귀·품질·메모리 기준 파일과 해시

## 검증 결과
정상성 / replay / 원인 관측 / OFF MAIN / C1 / LONG / 독립 확인 / 안정성

## 효과 귀속
직접 비용, 후속 대기, E2E의 연결 및 혼입·불확실성

## 판정
ACCEPTED / REJECTED_REGRESSION / NO_MEANINGFUL_GAIN /
NOT_APPLICABLE_WITH_EVIDENCE / ALREADY_IMPLEMENTED /
INCONCLUSIVE / BLOCKED_...

## 최종 활성 여부와 복구
기본 ON/OFF, 배치 파일, 실제 재현·rollback 명령

## 남은 필수 공백
질문과 영향 범위. 공백이 없으면 없음으로 명시
```

---

## 부록 B. 구현해야 할 도구의 기능 계약

다음 명칭은 **개발해야 할 도구의 역할 예시**다. 현재 파일이 존재하거나 옵션이 동작한다는 뜻이 아니다. 기존 도구가 같은 기능을 제공하면 확장·재사용하고 CLI·입출력 명세를 `TOOL_INTERFACES.md`에 저장한다.

| 도구 역할 | 입력 | 출력·실패 조건 |
|---|---|---|
| baseline/preflight 검사 | source/build/config/data manifest | hash·모듈·GPU/CPU·KV·policy proof; mismatch 실패 |
| descriptor export | 유효 이벤트·실제 입력·model refs | replay manifest·shape·tensor refs; 필요한 자료 누락 표시 |
| tile comparator | 동일 descriptor와 reference/candidate kernel | 출력 오차·시간·branch proof; 정상성 실패 반환 |
| job replay | 실제 연속 job·CPU 정책·Hot 집합 | NUMA·stage·service·수명 결과; queue/graph 변화 분리 |
| A2 plan verifier | descriptor plan·출력 범위·stage DAG | coverage·중복·ready·소유권 검사 |
| placement cost builder | calibration·현재 R_CPU·CPU/GPU 측정 | measured/estimated 구분 비용표·coverage |
| Hot swap generator | H0/HB·비용표·제약 | pair/set 교환·map diff·실현 가능성 |
| Hotmap validator | logical map·budget·rank loader proof | 5,952·bijection·mask·weight/scale·HBM 검사 |
| variant runner | 확정 variant·모드·workload·question | 안전 부팅·smoke·warmup·실행·drain·원장 |
| comparison analyzer | 모든 raw metrics·유효성·블록 | 원값·통계·총/한계 이득·누락·판정 입력 |
| finalizer | 코드·결과·manifest·판정 | 파일 검증·hash·재현·게시·전달 상태 |

도구의 exit code는 실제 실패를 전달해야 한다. shell pipeline을 사용하면 `pipefail` 등으로 실제 실행의 종료 코드를 보존하고 stdout·stderr를 모두 저장한다. 잔여 산출물이 있는 디렉터리를 성공 증거로 재사용하지 않는다.

### B.1 최소 자동화 시험

| 영역 | 시험 사례 | 기대 동작 |
|---|---|---|
| 설정 | A1/A2 미지정·0·1·잘못된 문자열 | 기본 OFF, 명시적 파싱, invalid 입력 실패 |
| 기준 경로 | 신규 빌드에서 두 기능 OFF | 기존 정상 경로와 수치 계약 통과 |
| 타입 | special path 미지원 타입 | 기존 native 계산 실제 실행, 출력 0 회귀 없음 |
| A1 | R=1/2/3, padding/tail, 잘못된 layout | 정확한 지원 분기·fallback·입력 오류 구분 |
| A2 | output tile 중복·누락·조기 stage 진행 | validator 실패, 본 성능 실행 차단 |
| B | 중복 expert, layer 범위 오류, 합계≠5952 | 부팅 전 또는 loader 검증 실패 |
| B 메모리 | KV를 줄인 후보 | 동일 조건 판정 실패 |
| 수명 | 미생산·중복·stale generation | 정확히 검출, 해당 후보 차단 |
| 매핑 | 다른 epoch·rank·phase 연결 | 해당 지표 invalid, 0으로 보정 금지 |
| 데이터 | 다른 tokenizer·token counts·manifest | workload mismatch 검출 |
| 결과 | 단발 반복·실패 요청 | SD/CI 오표기 없음, 실패를 유효 0으로 평균하지 않음 |
| 실행 정책 | 원장 횟수가 옛 상한 초과 | 필요한 유효 작업은 계속 허용 |
| 안전 | 권한 없음·다른 작업 활성·저장 공간 부족 | 해당 실행 차단·복구, 안전 검사 유지 |
| 종료 | 모든 필수 질문이 해결됨 | 새 무관한 실행을 자동 생성하지 않음 |
| 재시도 | 변경·진단 없는 동일 오류 | 무의미 반복 대신 원인·계획 재검토 |

---

## 부록 C. 구현자의 작업 단위 점검표

| 작업 ID | 해야 할 일 | 핵심 변경 또는 자료 | 종료 증거 |
|---|---|---|---|
| C00 | 출처·소스·실효 환경 확인 | 현재 HEAD/diff/.so/import/model/client | provenance 완성 |
| C01 | 횟수 정책·안전 검사 정리 | runner/planner/reporter guard | 합성 원장·안전 시험 |
| C02 | 기준 OFF·수치·입력 계약 | MAIN/C1/LONG·baseline output | R0와 사전 기준 |
| C03 | 플래그·feature proof | A1/A2/B·mode 독립 | 실제 branch·load proof |
| C04 | 실제 shape·job replay | descriptors·tensor/weight refs | reference replay 통과 |
| C05 | A1 잔여 비용 확인 | RB 내부 loop/load/unpack/scale | gap analysis |
| C06 | A1 R=1/2·필요 subvariant | 기존 RB와 구별되는 특수화 | 수치·시간·fallback |
| C07 | A1 서비스 검증 | replay·CORR·OFF MAIN·회귀 | A1 판정 |
| C08 | A2 적용성 확인 | 내부 pool/descriptor/stage 비용 | 적용성 판정 |
| C09 | A2 묶음 executor | 같은 job·local pool·tile ownership | 수치·coverage·service |
| C10 | A2/A12 검증 | A1 독립성·조합·회귀 | A2/CPU 조합 판정 |
| C11 | 비용표 갱신 | R_CPU 기준 CPU/GPU calibration | 비용 provenance |
| C12 | B 생성·validator | 승격/강등·5952·rank HBM | 유효 HB 후보 |
| C13 | B replay·normality·선별 | 전체 Cold 집합·Hot 증가·수명 | 동결 HB |
| C14 | 고정-map 개별·통합 비교 | 적용 가능한 8개 기능 조건 | 상호작용 분리 |
| C15 | 독립 확인·회귀·지속 실행 | OFF MAIN/C1/LONG·quality·stability | 최종 선택 근거 |
| C16 | 원복 시험·보고·전달 | 실제 명령·원값·판정·hash | 재현 가능 인계 |

이 표의 17개 작업은 세션 수가 아니다. 하나의 작업에 필요한 반복은 충분히 수행하고, 이미 답이 있는 작업은 증거를 재사용한다.

---

## 부록 D. 근거 목록·파일 해시·자료 계승 범위

### D.1 작성에 사용한 첨부 파일

아래 SHA256은 이 지시서 작성 환경의 실제 첨부 바이트에서 계산한 값이다.

| ID | 파일 | 주요 사용 절 |
|---|---|---|
| D | `CPU_MoE_병목분석_및_최적화설계.docx` | 9~17장, A1/A2/B, 그림 6~9, 부록 |
| R075 | `FULL_REPORT-6.md` | 1~5장, S29, 7장, 8b·8c, 유효성·클라이언트 기록 |
| V075 | `IDE075_results_review.md` | 대표 근거, 측정 제한, 새 기준선, 착수 판단 |
| P2 | `IDE074_additional_measurement_plan_v2.md` | 0.3, 5~15, 17~18장, 실행 횟수 정책 |
| H073 | `CPU_MoE_optimization_session_handoff-2.md` | 3~5장, 기존 기능, 안전·재현·보고·보존 |

```text
[D]
4b830c8061c8e086c336fa4e21cec40c86dd21c19318d51b2501fd8eb5aaa6af

[R075]
74586edec0b4af703797c64d3e46b41f1f9946c50cbe8f686989b787f657e845

[V075]
7c4f382170a5cbece60f38a85939467ee560058b4a8213423ec2a0dffefb08c1

[P2]
5700e71178491e2171483c08a188e5dc721b62bff16969c4e4dfa3dabb59a394

[H073]
67f1a818f1d87a9ea8fd84f9d527e5ee012b72ae4109542fd4ddd1d0ca4ee15c
```

이 지시서 자신의 해시는 배포 시 외부 checksum/manifest에서 기록한다. 파일 본문 안에 자기 자신의 hash를 확정값으로 넣는 순환 구조를 만들지 않는다.

### D.2 설계 원문과 지시서의 대응

| 설계 원문 | 지시서에 반영한 내용 |
|---|---|
| D 9장 | A1/A2/B를 기존 최적화와 구분, 우선순위와 변경 격리 |
| D 10~10.2장·그림 6 | A1 실제 R1/R2·shape, W와 입력 재사용의 구분, 안전한 fallback |
| D 11장·그림 7 | A2 조건부 적용, 같은 job·NUMA-local descriptor, 상위 FIFO 유지 |
| D 12~12.2장·그림 8 | B 비용 재측정, paired swap, 5,952·HBM, screening과 실제 E2E 구분 |
| D 13~14장 | 공통 FP8 회귀, tensor/logit/quality·buffer lifecycle |
| D 15~16장·그림 9 | 기준선→단일 변경→replay→관측→OFF→독립 회귀·채택 |
| D 17장 | 코드·실제 로드 소스·원자료·원복·횟수 상한 없는 인계 |
| 최신 사용자 요청 | A1·A2·B 모두의 작업·판정과 적용 가능한 통합 검증 포함 |

### D.3 작성 범위와 남겨 둔 미확정 사항

이 지시서는 첨부 설계·측정·검토 자료를 구현 작업으로 구체화했다. 외부 최신 기술 조사, 현재 원격 저장소 HEAD 조회, 원 서버의 source 실행·patch 적용·대형 trace 재집계를 수행한 결과가 아니다.

실제 미확정 사항은 현재 커널의 잔여 코드 비용, A2 내부 묶음 적용 가능성, B의 새 비용표와 최종 map, 새 정밀도·성능 채택 임계값, 최종 활성 조합이다. 이를 이미 알려진 성능이나 구현 완료로 채워 넣지 않았다. 각 항목은 해당 단계의 code inspection·직접 비교·사전 계약·독립 검증으로 확정한다.

---

## 최종 실행 원칙

**기존 RB를 기준으로 A1을 구현하고, A2는 실제 내부 비용에 맞춰 적용하며, 바뀐 CPU 비용으로 B를 설계한다. 세 방안의 단독·통합 효과를 확인하되 정상성과 무계측 서비스 성능으로 채택한다. 필요한 작업은 횟수와 무관하게 끝까지 수행하고, 결과에 영향을 주지 않는 반복·범위 확장은 하지 않는다.**
