# RBC-Attn v0.2 성능 개선 구현 지시서

**작성일:** 2026-09-12  
**대상:** `rbc-dev`에서 RBC-Attn v0.1을 구현·측정한 서버 개발 에이전트  
**기준 자료:** `RESULT.md`의 2026-09-11~12 측정 결과  
**목적:** 기존 구현의 실행계획·GPU 본체·호스트 전달 경로를 수정하여, 동일 입력의 최강 기준선보다 실제 완료시간을 단축한다.

---

## 0. 작업 지시

**자료 조사를 다시 시작하지 말고, 현재 서버에 있는 RBC-Attn 소스를 아래 순서로 수정하라.** 기존 저장소를 유지하고 `rbc-v02-perf` 작업 브랜치에서 진행한다. 이 문서는 구현 명세이며 새로운 ZIP을 받아야 착수할 수 있는 작업이 아니다. 함수명이나 파일 경로가 다르면 동일 기능의 서버 코드에 적용하고 대응표를 남긴다.

이번 변경의 중심은 다음 여섯 가지다.

| 우선순위 | 반드시 구현할 변경 | 제거할 비용 | 적용 위치 |
|---|---|---|---|
| P1 | **최소 task 수 강제 분할을 기본값에서 제거하고, 분할·버킷화가 끝난 최종 계획끼리 선택** | 좋은 계획을 선택한 뒤 다시 분할해 생기는 Q 재적재·partial 증가·잘못된 선택 | CPU planner |
| P2 | **결과 소유자가 하나인 query는 main에서 최종 출력으로 직접 기록** | 불필요한 FP32 partial 쓰기·읽기와 merge 실행 | main·merge kernel |
| P3 | **GPU 계산용 K 타일을 sparse block과 분리하고, V의 생존 구간과 과도한 mask 처리를 축소** | main의 live tensor·register 부담, 불필요한 메타데이터·mask 명령 | main kernel |
| P4 | **미교정 바이트 가중합을 최종 계획의 실측 비용 선택으로 교체** | random·clustered에서 느린 RBC 계획을 선택하는 회귀 | CPU planner·교정 도구 |
| P5 | **고정 용량 pinned/device arena와 단일 descriptor 전송, 재사용 CPU 작업공간을 구현** | 매 호출 할당·pin·배열 변환·다수의 작은 H2D 제출 | runtime·CPU planner |
| P6 | **whole-query 실행을 기본으로 복구하고, 검증된 경우에만 chunk pipeline 사용** | 작은 chunk를 위해 반복하는 계획·왕복 전송·launch·GPU 분할 | stream runtime |

P1~P3은 GPU 실행에 남아 있는 개선 가능성을 직접 확인하는 변경이다. P4는 선택 오류를 줄이고, P5는 개선분을 상쇄하는 호스트 비용을 제거한다. P6은 불필요한 파이프라인을 기본 경로에서 제외한다. **CPU 준비비만 줄인 결과, Q-outer로 복귀한 결과, RBC 고유 분해가 이긴 결과는 서로 다른 성과로 보고한다.**

아래의 수치 목표는 **이번 수정의 채택 기준**이며, 이미 관측된 결과나 보장된 성능이 아니다.

---

## 1. 변경하지 않을 조건과 최종 판정

### 1.1 고정 조건

동일한 Q/K/V, dtype, selected KV block, causal position, attention scale, GQA 매핑을 유지한다. 양자화, KV 삭제, mask 근사, 모델 변경을 추가하지 않는다. CPU로 Q/K/V를 옮겨 계산하는 경로도 추가하지 않는다.

하나의 KV cache identity 안에서만 작업을 합친다. 서로 다른 요청의 동일 block 번호를 같은 데이터로 간주하지 않는다. 실제 모델에서 GQA head별 mask가 다르다면, 현재 API가 같은 mask를 가정한다는 사실을 무시하고 합치지 않는다.

`RESULT.md`에 사용한 H100과 NUMA 배치를 우선 유지한다. GPU 4~7은 사용한 장치 집합이지, 개별 연산을 네 GPU로 분산했다는 뜻으로 해석하지 않는다. 각 run의 실제 GPU UUID, 사용 장치 수, 논리 ordinal을 기록한다. 운영 서비스·드라이버·MIG·전력 설정은 임의로 변경하지 않는다.

### 1.2 정확성 계약

서버 v0.1에서 사전에 고정한 정확성 검사식과 허용오차를 그대로 유지한다. 기록의 BF16 상대오차 기준 `5e-3`, FP16 `1e-3`를 느슨하게 하지 않는다. 상대오차의 분모·절대오차 보조 기준은 **서버의 실제 검사 코드**를 확인해 manifest에 저장한다. 원 배포 ZIP의 다른 허용오차를 대신 적용하지 않는다.

모든 selected edge는 정확히 한 번 반영한다. 빈 행은 `out=0, LSE=-inf`로 처리한다. 비어 있지 않은 유효 행의 NaN/Inf는 실패다. LSE의 로그 밑은 사용한 함수·backend·버전별 계약을 기록한다. RESULT에서 확인한 FlashInfer의 log2 LSE를 다른 버전·API의 보편적 규칙으로 확대하지 않는다.

### 1.3 성과를 세 단계로 구분

| 판정 | 통과 조건 | 허용되는 설명 |
|---|---|---|
| 구현 개선 | 동일 정책의 v0.1보다 실제 시간이 감소 | runtime 또는 kernel 구현 개선 |
| RBC GPU 기여 | 같은 개선 실행기를 쓰는 최강 Q-outer/KV-outer/signature-only보다, **Q-outer가 아닌 RBC 계획**의 GPU 시간이 감소 | RBC 분해 규칙의 추가 이득 |
| CPU–GPU 전체 개선 | 같은 입력 출발점에서 CPU 준비·필요한 전송·GPU·결합을 포함한 시간이 최강 내부 기준선 및 지원되는 native보다 감소 | CPU 참여를 포함한 연산자 가속 |

전체 개선 목표는 **완료시간 10% 이상 감소**다. `gain = 1 - T_new/T_best_baseline`으로 계산한다. 속도비 `T_best_baseline/T_new`와 혼용하지 않는다. 10% 미만의 유의한 개선은 그대로 보고하되 목표 통과로 바꾸지 않는다.

합성 입력만 통과하면 합성 연산자 결과로 한정한다. 전체 모델 tok/s, 실제 서비스 SLO, 논문 신규성 확보를 선언하지 않는다.

---

## 2. 이번 수정에 사용하는 실측 근거

다음은 `RESULT.md`의 관측값이다. [R1]

| 구간 | 기준선 | RBC v0.1 | 수정 방향 |
|---|---:|---:|---|
| Nq512 common/private, run-only | Q-outer 86 μs | 87 μs | 단순 준비비 제거 외에 main 개선이 필요 |
| Nq512 random, run-only | Q-outer 111 μs | 150 μs | 느린 병합·계획 선택 차단 |
| Nq512 clustered, run-only | Q-outer 115 μs | 146 μs | signature-only보다 나쁜 병합 차단 |
| NCU main | Q-outer 109.0 μs | 134.9 μs | main의 분해·자원 사용·접근 경로 수정 |
| NCU merge | Q-outer 7.3 μs | 6.9 μs | merge만으로 역전 불가 |
| NCU main DRAM read | Q-outer 37.4 MB | 91.0 MB | task별 방문량·캐시 상태·Q/K/V 접근 분리 |
| fresh wall, RESULT §4(f) | native 0.744 ms | 1.157 ms | 할당·계획·descriptor 전달 제거 |
| CPU 계획, RESULT §4(f) | — | 0.52 ms | CPU allocator·정렬·후처리 개선 |
| descriptor H2D로 표기된 구간 | — | 0.30 ms | DMA와 host 준비·제출을 분리한 뒤 수정 |
| Nq2048 pipeline | whole-query Q-outer run-only 0.405 ms | 7.515 ms | 작은 chunk pipeline 기본 해제 |

NCU 표의 RBC `main+merge=141.8 μs`, Q-outer `116.3 μs`다. RBC merge를 완전히 없애도 main `134.9 μs`가 남는다. 따라서 **P2만으로 10% 전체 개선을 기대하지 말고 P1·P3·P4를 함께 수행한다.** 정상 재생 86/87 μs와 NCU 116.3/141.8 μs는 서로 다른 측정 조건이다. 한 식에 섞어 예상 개선율을 계산하지 않는다.

### 2.1 RESULT의 해석 중 그대로 코드에 옮기면 안 되는 항목

원 결과의 §4(c)·§7은 KV 비용을 전체 합집합으로 바꾸도록 제안한다. 이 지시서는 이를 채택하지 않는다. **관측된 main DRAM 증가와 그 증가의 원인 해석을 구분**하기 때문이다.

- `B_unique`: 서로 다른 KV 데이터의 작업 집합 크기.
- `B_issued`: task·CTA별로 발행되는 KV 방문 바이트의 합.
- `B_HBM`: 실제 cache 상태와 실행 순서에서 발생한 HBM 읽기.

세 값은 같지 않다. 전체 합집합은 동일 mask의 여러 분해에서 같을 수 있으므로, 이 값 하나로 계획을 선택하면 반복 적재 차이가 사라진다. 원 배포 코드의 KV 방향 이등분은 겹치지 않는 구간을 나누며, 그 분할 자체는 KV 방문 수를 증가시키지 않는다. 다만 Q와 partial은 늘 수 있다. **서버 재구현도 동일하다고 가정하지 말고 분할 전후 descriptor로 확인한다.** [C1]

점유율 약 12%와 register 153개는 관측값이다. 네 정책에서 비슷하므로 RBC만 느린 원인이라고 확정하지 않는다. 이번에는 실제 kernel 시간과 이론 점유율·shared memory·spill·작업 길이를 함께 비교한다. [N1, N2]

---

## 3. 착수: 현재 서버 구현을 기준으로 고정

서버가 명세로부터 재구현했다는 RESULT의 설명을 따른다. 과거 ZIP을 덮어써서 새 기준선을 만들지 않는다.

현재 repository root에서 다음을 수행한다.

```bash
# 기존 소스와 결과를 지우거나 reset하지 않는다.
git status --short
git switch -c rbc-v02-perf
mkdir -p results/v02/{manifest,plans,kernel,calibration,online,pipeline,profiles,report}
git rev-parse HEAD > results/v02/manifest/base_commit.txt
git diff --binary > results/v02/manifest/preexisting_changes.patch
python -m pip freeze > results/v02/manifest/pip_freeze.txt
nvidia-smi -L > results/v02/manifest/gpus.txt
nvidia-smi topo -m > results/v02/manifest/topology.txt
numactl --hardware > results/v02/manifest/numa.txt
```

브랜치가 이미 있으면 이를 확인하고 이어서 사용한다. git을 사용하지 않는 저장소라면 먼저 현재 디렉터리의 소스·설정·결과 해시를 보존한다. untracked 파일은 `git diff`에 들어가지 않으므로 별도 manifest 또는 소스 아카이브에 포함한다.

`results/v02/manifest/source_map.json`에 다음 기능의 실제 위치를 기록한다.

| 기능 | 원 배포본의 위치 — 참고용 | 서버에서 찾을 기능 |
|---|---|---|
| 비용과 병합 | `src/planner.cpp: Cost, row_tasks, join` | 비용 계산·signature 병합 |
| task 분할 | `src/planner.cpp: remap, min_tasks loop` | KV 분할과 query 재매핑 |
| 최종 descriptor | `src/planner.cpp: rbc_plan` | CSR·partial slot 생성 |
| Python 계획 연결 | `rbc/planner.py: make_plan` | ndarray 할당·FFI·복사 |
| GPU 본체 | `rbc/kernels.py: attention_tasks` | QK·softmax·PV 루프 |
| GPU 결합 | `rbc/kernels.py: merge_tasks` | partial reduction |
| runtime | `rbc/runtime.py: DevicePlan` | pinned·device 할당·upload·launch |
| pipeline | `stream_bench.py` | snapshot·D2H·chunk scheduling |

기존 호스트 57건·GPU 37건에 대응하는 서버 테스트를 먼저 실행한다. 테스트 수가 과거 ZIP과 다르다는 이유만으로 결과를 무효 처리하지 않는다. 기능과 허용오차를 기준으로 대응시킨다.

---

## 4. P1 — 계획 선택 뒤 강제 분할을 없애라

### 4.1 바꿀 코드

1. 기본 `min_tasks=2*SM` 또는 상수 `264`를 제거하고 **기본값을 0**으로 변경한다. 과거 동작은 `legacy_min_tasks` 옵션에만 남긴다.
2. Q-outer, signature-only, RBC 병합 계획을 **각각 완성된 후보**로 취급한다.
3. 각 후보의 query remap, KV 분할, task 정렬, kernel bucket, direct/partial 소유권까지 확정한 뒤 비용을 비교한다.
4. 선택한 계획은 immutable 상태로 만들고 이후에 task 수를 맞추기 위해 다시 나누지 않는다.
5. Q-outer fallback은 새 기능으로 추가했다고 기록하지 않는다. 기존에 존재하는 선택을 **최종 실행 기준으로 고치는 작업**이다.

금지할 순서는 다음과 같다.

```text
분할 전 RBC가 싸다고 판단
→ RBC 선택
→ task 수가 부족하므로 264개까지 강제 분할
→ 분할된 계획의 비용은 다시 평가하지 않고 실행
```

변경할 순서는 다음과 같다.

```text
Q-outer / signature-only / bounded-merge 후보 생성
→ 후보별 분할·버킷화·출력 소유권까지 완성
→ 최종 후보를 비교
→ 선택된 descriptor를 그대로 실행
```

### 4.2 분할은 후보로만 생성

초기 교정에서는 다음 세 분할 후보를 비교한다.

- `none`: task를 인위적으로 늘리지 않는다.
- `one_wave`: 실제 SM 수를 참고하여 가장 긴 task를 나누는 제한 후보.
- `two_waves`: 실제 SM 수의 두 배를 참고하는 과거 방식의 대조 후보.

SM 수는 실행 GPU에서 조회한다. 이 수는 목표 task 수 후보를 만드는 기준일 뿐 실제 occupancy나 최적 CTA 수가 아니다. 각 kernel bucket의 thread·register·shared-memory 자원에 따라 resident CTA 수가 달라진다. [N1]

KV split은 **원 KV 목록의 겹치지 않는 구간**으로만 나눈다. 두 자식에 같은 KV 목록 전체를 복제하는 query split과 구분한다. 특정 query가 양쪽에 등장하면 partial 수 증가를 다시 집계한다. 무효 membership은 제거하되 유효 edge는 변경하지 않는다.

### 4.3 반드시 넣을 검사

```text
exact_edges(before) == exact_edges(after)
kv_visits(before) == kv_visits(after)    # 순수한 비중복 KV split인 경우
query_task_incidences(after) = sum(len(task.qids) for task in children)
selected_plan_hash == uploaded_plan_hash
```

두 번째 검사가 실패하면 query split·KV 복제·작업 제거 등 실제로 무엇을 수행했는지 기록한다. 합집합이 같다는 이유로 방문량 증가를 숨기지 않는다.

### 4.4 출력할 계획 기록

`results/v02/plans/<case>/<candidate>.json`에 다음을 저장한다.

```text
mask_hash, q_position_hash, plan_hash, gpu_uuid
policy, max_m, num_warps, num_stages, kernel_k_tile
query_order, split_policy, legacy_min_tasks
ntasks_before, ntasks_after, bucket_launch_count
kv_unique_blocks, kv_visits_before, kv_visits_after
query_load_rows, partial_slots, direct_rows, multi_owner_rows
padded_dot_flops, selected_edge_flops
per_task: head, qids, kids, membership, BM, output_mode
predicted_time_us, prediction_uncertainty_us, fallback_reason
```

mask가 많으면 descriptor 본체는 NPZ 등 별도 파일로 저장하고 JSON에는 경로와 SHA-256을 남긴다. 성능 구간에서 거대한 JSON을 생성하지 않는다.

계측 단위를 구분한다. `query_task_incidences`는 각 task의 query 위치 수의 합이고 GQA head 수를 아직 곱하지 않은 값이다. P2 이후 `partial_slots`는 그중 실제 partial workspace에 쓰는 슬롯 수이며, direct 행은 제외한다. 기존 v0.1의 모든 query-task 슬롯과 같은 이름으로 혼합 집계하지 않는다.

### 4.5 완료 조건

common/private·random·clustered의 기존 세 대표 입력에서 분할 전후 차이를 추적할 수 있어야 한다. 어떤 후보가 선택됐는지, 그 후보가 실행 시점에 바뀌지 않았는지 확인한다. **task 수 감소나 fallback 선택 자체는 성능 성공이 아니다.**

---

## 5. P2 — 단일 소유 query의 partial과 merge를 제거하라

이 변경은 직접 제거하는 메모리 작업이 명확하다. 다만 Q-outer에도 동등하게 적용한다.

### 5.1 출력 소유권 계산

최종 분할이 끝난 뒤 `(cache identity, KV head, query id)`마다 결과를 쓰는 task 수 `degree[row]`를 계산한다.

| degree | 실행 |
|---:|---|
| 0 | `out=0`, `LSE=-inf`를 기록하는 빈 행 경로 |
| 1 | 유일한 task가 main 안에서 정규화한 최종 출력과 LSE를 직접 기록 |
| 2 이상 | 해당 행의 task 결과만 partial workspace에 저장하고 compact merge 실행 |

GQA mapping이 query 위치와 G개 head를 함께 처리하는 현재 표현이면 degree는 query 위치 기준으로 계산하되 출력은 각 GQA head별로 기록한다. future head별 mask 확장 시에는 더 세밀한 소유권이 필요하다.

### 5.2 main epilogue 변경

아래는 구현해야 할 의미론이다. 그대로 컴파일되는 완성 코드가 아니라 현재 Triton/CUDA kernel에 옮길 의사코드다.

```python
# m, l, acc: 해당 task가 실제 반영한 edge의 FP32 상태
if degree[row] == 1:
    if l > 0:
        out[row, gh, :] = acc / l
        lse[row, gh] = m + log(l)
    else:
        out[row, gh, :] = 0
        lse[row, gh] = -inf
else:
    slot = assigned_partial_slot[task_row]
    partial_u[slot, gh, :] = acc
    partial_m[slot, gh] = m
    partial_l[slot, gh] = l
```

실제 GPU 구현은 per-row store mask 또는 direct 전용 kernel variant를 사용한다. 한 task 안에 degree=1과 degree>1 행이 함께 있어도 정확히 한 경로만 쓰게 한다. 서로 다른 CTA가 같은 최종 출력에 동시에 store하는 구현은 금지한다.

### 5.3 merge grid와 workspace 축소

- `degree>=2` 행만 `multi_rows`에 넣는다.
- reduction CSR에는 multi-owner 행의 partial slot만 넣는다.
- merge grid는 전체 `Hkv*Nq`가 아니라 `len(multi_rows)`를 기준으로 생성한다.
- multi-owner 행이 없으면 merge kernel을 호출하지 않는다.
- 단일 소유 행을 위해 `U/M/L`을 할당하지 않는다.
- partial 개수 차이가 큰 경우, reduce 길이를 2/4/8/16/… 버킷으로 나누는 것은 후보로 비교한다. 버킷 증가에 따른 launch 비용도 포함한다.

빈 행 출력은 별도 empty-row 목록 처리나 공통 초기화 경로로 보장한다. 버퍼가 재사용되므로 전 호출 값이 남아도 우연히 맞는 테스트를 만들지 않는다. 빈 행 처리 kernel이 필요한 경우 그 시간은 모든 비교에 포함한다.

### 5.4 줄어드는 논리 바이트

기존 표현의 FP32 `u,m,l` 기준으로, query 위치 하나와 G개의 query head에 대해 제거 가능한 partial write/read는 다음과 같다.

\[
B_{\mathrm{saved}}=2\times4\times G\times(D+2).
\]

최종 output·LSE 쓰기는 없어지지 않는다. 위 식을 전체 HBM 절감으로 간주하지 말고 논리 집계와 카운터를 따로 기록한다.

### 5.5 완료 조건

단일 소유 계획에서 merge launch 수가 0이고 partial workspace의 사용 바이트가 0이어야 한다. 혼합 소유 계획에서는 direct/partial 양쪽이 기존 reference와 동일 허용오차를 충족해야 한다. 출력에 의도적인 sentinel을 채우고 실행하여 모든 행이 이번 호출에서 갱신되는지도 검사한다.

---

## 6. P3 — main의 K 타일과 tensor 생존 구간을 수정하라

### 6.1 sparse block과 kernel K 타일을 분리

입력 mask의 block size `BK=128`은 고정한다. **GPU 내부 처리 타일** `BN`만 `{32,64,128}` 중 선택 가능하게 만든다. mask를 작은 block으로 바꾸거나 선택 KV를 줄이지 않는다.

```python
for block in selected_blocks:
    member = block_membership[block]
    for offset in range(0, BK, BN):
        token_ids = block * BK + offset + arange(BN)
        # 동일 block의 membership, 원래 NK 경계와 causal 조건을 적용
        # QK → online softmax 갱신 → PV
```

BN을 줄이면 score·K/V tile의 작업 집합이 작아지는 대신 루프와 softmax 갱신 횟수가 증가한다. 따라서 `BN=32`를 무조건 채택하지 않는다. FP32 state를 유지하고 tile별 갱신 순서에 따른 수치 차이는 기존 기준으로 검사한다.

### 6.2 V의 load를 PV 직전으로 늦추는 variant 추가

원 배포본은 같은 루프에서 K와 V를 모두 load한 뒤 QK를 계산한다. [C2] 서버 구현도 같은지 먼저 확인한다. 같다면 다음 두 코드를 비교한다.

```text
기존 variant:
  K load + V load → QK → softmax → PV

추가 variant:
  K load → QK → softmax → V load → PV
```

목적은 K·V·score·probability·output accumulator가 동시에 살아 있는 구간을 줄이는 것이다. compiler가 load를 재배치할 수 있으므로 소스 위치만 보고 성공했다고 판단하지 않는다. 생성 코드의 register, shared memory, spill, 실제 지연을 확인한다. 늦은 load로 메모리 중첩이 줄어 느려지면 기존 variant를 유지한다.

### 6.3 full-membership fast path

task의 모든 KV block이 그 task의 모든 실제 query에 연결된 경우 `membership_mode=FULL`로 표시한다. 이 경우 `task_kmask`를 읽고 bit shift하는 연산을 없애는 kernel variant를 만든다.

단, FULL은 **membership 검사만 생략**한다. 아래 조건은 계속 적용한다.

- padding된 query 행 제외
- 마지막 KV block의 `NK` 경계
- causal position
- 그 밖에 현재 API가 지원하는 attention 의미론

동일 signature task는 FULL 후보지만, causal 조건으로 일부 token이 제외될 수 있다. 전체가 유효한 것처럼 causal mask까지 제거하면 안 된다. `FULL/MIXED`별 kernel 분리로 launch가 너무 많아지면 단일 kernel의 uniform task branch와 비교한다.

### 6.4 허용할 타일 후보와 탐색 예산

기존 네 가지 `max_m={16,32,64,128}`를 유지한다. 여기에 BN·warp·stage를 한꺼번에 전수조합하지 않는다.

| 단계 | 비교 | 고정할 것 |
|---|---|---|
| K1 | 기존 max_m 4종, 분할 3종 | v0.1 warp·stage·BN=BK |
| K2 | K1 상위 2개에 BN=32/64/128 | 같은 입력·계획 의미 |
| K3 | K2 상위 2개에 num_warps=4/8, num_stages=1/2/3 | 지원·컴파일 가능한 조합만 |
| K4 | 최종 2개에 V-load late on/off, FULL fast path on/off | 같은 계획과 dtype |

각 정책에 같은 총 후보 수·warmup·반복 예산을 준다. 무효 조합과 컴파일 실패도 기록한다. register 상한을 강제로 낮추는 옵션은 기본으로 넣지 않는다. spill을 만들면서 점유율 수치만 높이는 결과는 채택하지 않는다.

### 6.5 KV 순서에 대한 제한된 변경

각 task의 KV 목록은 실제 page-table 의미를 보존하면서 안정적인 순서를 사용한다. 연속 KV 주소 구간에서는 주소 계산과 metadata load를 줄일 수 있도록 `start/count` 구간 표현을 후보로 추가할 수 있다. 변경 전후 방문 block과 token 경계가 같아야 한다.

서로 다른 CTA의 실행 순서를 CPU가 완전히 제어할 수 있다고 가정하지 않는다. cache locality용 task 정렬은 독립 ablation으로만 비교한다. 여러 CTA가 같은 KV를 참조한다는 이유로 HBM 읽기를 한 번으로 계산하지 않는다.

### 6.6 완료 조건

P1~P3을 적용한 공통 실행기에서 Q-outer/signature-only/RBC를 다시 비교한다. RBC v0.1 대비 개선뿐 아니라 **동일 개선 실행기의 Q-outer 대비 결과**를 남긴다. 목표는 main+필요 merge의 완료시간 감소이며 register 수나 occupancy 자체가 아니다.

---

## 7. P4 — 계획 비용을 ‘실행이 끝나는 시간’ 기준으로 교체하라

### 7.1 삭제할 결정 규칙

교정되지 않은 `byte_weight=1`, `flop_weight=0.015` 등 임의의 가중합은 통계 출력에만 남긴다. 새로운 production 선택에는 사용하지 않는다.

`B_unique`를 `B_issued` 대신 넣는 수정은 금지한다. 또한 CUDA kernel launch는 task 하나당 발생하지 않을 수 있으므로 `task 수 × launch latency`를 넣지 않는다. task 개수, CTA 준비비, 실제 kernel bucket launch 수를 구분한다.

### 7.2 비교할 최종 후보

다음 후보만 유지한다. 새 이름을 늘리거나 별도 알고리즘으로 이탈하지 않는다.

```text
q_outer       : 필수 기준 후보
signature     : 동일 membership만 묶음
bounded_merge : 제한된 cost-aware 병합
```

세 후보 모두 P1의 분할 후보와 P2/P3 실행기를 공유한다. `bounded_merge`는 기본 off로 시작한다. 교정·검증에서 signature보다 좋아지는 영역이 확인된 뒤 해당 영역에만 활성화한다.

### 7.3 작업과 계획의 특징값

입력의 `pattern='random'` 같은 벤치 라벨을 선택기에 넣지 않는다. 실제 descriptor에서 다음을 계산한다.

```text
G, D, dtype, BK, BN, causal
BM·warp·stage별 task 수
KV 길이의 합·p50·p90·max
query load rows, direct rows, multi-owner rows, partial slots
issued KV bytes, unique KV working-set bytes
selected/padded dot FLOPs, full-membership 비율
CTA 수 / SM 수, longest-task 비용 추정
kernel bucket 수, descriptor 실바이트
```

`B_unique`는 cache 작업 집합의 특징이다. `B_issued`는 발행량의 특징이다. 둘 다 사용할 수 있지만 어느 쪽도 HBM 실측이라고 부르지 않는다.

### 7.4 구현할 시간 추정

전체 계획의 일반 GPU 반복 실행시간을 목표값으로 사용한다. 기본 형태는 다음과 같다.

\[
\widehat T_{GPU}(P)=
\sum_b \widehat T_{bucket}(P_b)
+\widehat T_{merge}(P)
+\widehat T_{empty}(P).
\]

각 bucket은 동일 geometry의 task 집합이다. 초기 bucket 특징에는 다음 두 부하·tail 항을 포함한다. 이 값들은 예측 특징이며 실제 실행시간의 보장된 하한이 아니다.

\[
L_{work}=\frac{\sum_t \widehat s_t}{SM\times c_b},\qquad
L_{tail}=\max_t\widehat s_t.
\]

`c_b`는 해당 kernel 자원의 이론적 resident CTA 수, `s_t`는 교정한 task 비용이다. 이 식만으로 실제 cache·메모리 경쟁을 예측했다고 간주하지 않는다. **최종 bucket 시간을 실제 계획 실행으로 교정**하고, global memory·L2 working-set·mask padding의 특징을 보완한다. 단독 CTA에서 측정한 throughput을 모든 CTA에 선형 적용하지 않는다.

복잡한 모델은 필요 없다. 작은 lookup table 또는 규제된 회귀를 사용하되, 지원 범위 밖 입력은 Q-outer로 돌아가게 한다. lookup·특징 계산시간도 `cpu_plan_us`에 포함한다.

### 7.5 교정·평가 분리

- 재현용 seed 42 결과는 기존 결과 확인에만 사용한다.
- 교정용 mask seed: `0,1,2,3,4`.
- 독립 평가용 mask seed: `101,102,103,104,105,106,107,108,109`.
- 실제 capture를 얻으면 요청 또는 trace 단위로 분리한다. 같은 capture를 seed만 바꿔 held-out이라고 하지 않는다.
- G4/G8/G16, Nq256/512/2048 중 평가 전용으로 남긴 형상에서도 선택 오류를 확인한다.
- NCU 실행시간으로 time model을 적합하지 않는다. profiling 없는 CUDA-event 또는 graph timing을 사용한다.

평가 입력마다 모든 후보를 GPU에 실행해 가장 빠른 것을 고르는 방식은 **oracle 진단**에만 허용한다. 실제 온라인 선택은 CPU descriptor와 교정된 표만 사용한다.

### 7.6 병합 연산 자체의 선택 규칙

`bounded_merge`의 내부도 기존 바이트 가중합을 쓰지 않도록 교체한다. 같은 query 그룹의 두 task `a,b`를 합치는 후보마다 다음을 계산한다.

\[
\widehat\Delta(a,b)=
\widehat T_{GPU}(P)
-\widehat T_{GPU}(F(P\setminus\{a,b\}\cup\{a\cup b\}))
-\widehat T_{extra\ planning}.
\]

`F`는 해당 후보의 분할·버킷화·출력 소유권 확정이다. 따라서 병합으로 partial이 줄었더라도 후속 분할에서 다시 늘면 그 이득은 남지 않는다. `a∪b`는 selected edge의 합집합을 정확하게 보존한 masked task이며, query 수가 지원 상한을 넘으면 후보에서 제외한다.

구현은 다음 규칙으로 제한한다.

```text
1. 동일 signature로 만든 초기 계획과 Q-outer 계획의 특징을 계산한다.
2. query를 공유하는 task 쌍만 병합 후보로 둔다.
3. 지원 BM·KV 길이·workspace 조건을 넘는 후보를 제거한다.
4. 후보가 영향을 주는 task/bucket/row의 특징만 갱신해 최종 비용 차이를 계산한다.
5. 오차 여유보다 큰 양의 이득이 있는 쌍 중 최선 하나만 채택한다.
6. 영향받은 후보만 갱신하고 반복한다.
7. 완성된 결과를 최종 Q-outer·signature 계획과 다시 비교한다.
```

query 그룹당 초기 atom이 32개를 넘으면 첫 버전에서는 병합을 건너뛴다. 한 그룹에서 비용을 실제 평가하는 쌍은 누적 64개, 채택하는 병합은 최대 4회로 제한한다. 후보 우선순위는 query 교집합이 큰 순서로 하되, **우선순위는 탐색 순서일 뿐 채택 비용식이 아니다.** 예산을 늘리는 변경은 CPU 계획시간과 전체 성능 개선을 함께 확인한 뒤 적용한다.

최종화를 후보마다 전체 plan 재할당으로 구현하지 않는다. group-local query degree와 bucket별 누적 특징을 저장하고 변경분을 계산한다. 채택된 병합만 descriptor로 materialize한다. 이 제한 탐색은 전역 최적을 보장하지 않으며, 교정된 Q-outer와의 마지막 비교가 항상 남아 있어야 한다.

### 7.7 보수적인 선택 규칙

```python
q = finalized_q_outer
p = best_finalized_non_q_candidate

if unsupported_features or uncertainty_too_large:
    selected = q
elif predicted_gpu_saving(p, q) <= (
        extra_prepare_cost(p, q) + error_margin(p, q)):
    selected = q
else:
    selected = p
```

위 규칙은 성능 하한의 수학적 보장이 아니다. fallback도 비용을 지불하므로 **Q-outer를 고르는 데 CPU 0.5ms를 쓴다면 이미 실패**다. 교정에서 RBC가 이기지 못한 특징 영역은 signature·병합 후보를 만들기 전에 빠르게 우회한다. 빠른 우회에 필요한 input-origin 비용도 포함한다.

초기 진단 기준은 held-out의 GPU 계획 선택 regret 중앙값 3% 이하, 최악 5% 이하다. 여기서

\[
regret=\frac{T_{selected}-T_{best\ measured}}{T_{best\ measured}}.
\]

이는 선택기 품질 기준일 뿐 신규 가속의 증거가 아니다. 거의 항상 Q-outer를 선택해서 통과했다면 `RBC_active_fraction`과 함께 그렇게 보고한다.

---

## 8. P5 — hot path의 할당·pin·작은 전송을 제거하라

### 8.1 runtime을 세 단계로 분리

```text
prepare_capacity(shape_limits)  # 초기화 때만 메모리·event·graph 준비
plan_into(input_mask, slot)     # 새 mask를 기존 메모리에 계획
submit(slot, Q, K, V)           # 사용 구간만 전송하고 GPU 실행
```

`DevicePlan`을 매 호출 새로 만드는 패턴을 제거한다. mask 내용이 바뀌어도 geometry와 capacity가 같으면 arena를 재사용한다. **버퍼 재사용과 plan 재사용은 다르다.** 동적 mask에서는 반드시 새 계획을 만들거나 이번 mask와의 정확한 일치가 확인된 plan만 사용한다.

### 8.2 descriptor를 하나의 연속 버퍼로 작성

다음 요소를 하나의 host pinned slab과 대응하는 device slab 안에 배치한다.

```text
header: version, generation, shape, live counts, array offsets
heads / query offsets / query IDs
KV offsets / KV IDs / membership
bucket task IDs / output ownership / compact reduction metadata
```

C++ planner가 최종 배열을 pinned slab의 예약 구간에 직접 쓰게 한다. Python list→NumPy→pinned Tensor→device Tensor의 연쇄 변환을 없앤다. 기존 입력 mask가 host CSR이면 그 배열을 C++에서 직접 읽는다.

전송은 원칙적으로 **완료된 plan당 하나의 연속 H2D**로 만든다. 불필요한 capacity 전체를 보내지 말고 실제 사용한 descriptor 길이를 보낸다. alignment gap도 실제 전송 바이트에 포함한다. capacity 초과는 setup 확장 또는 명시적 fallback으로 처리하고 silent reallocation을 하지 않는다. [N3]

### 8.3 descriptor 정수 폭 축소

범위 검증 후 가능한 항목에만 int32 offset을 사용한다. 원소 수·byte offset이 32-bit 범위를 넘으면 64-bit 경로로 돌아간다. tensor 주소 계산은 별도로 overflow를 검사한다.

membership은 실제 그룹의 query 위치 수에 맞춰 uint16/uint32/uint64 중 지원한 폭을 선택할 수 있다. FULL task는 membership 배열을 전송하지 않는 형식을 우선 비교한다. 이는 **메타데이터의 무손실 표현 변경**이며 가중치·KV 정밀도 변경이 아니다.

버킷별 TaskIds도 slab 안에 포함한다. Q-position이 바뀌면 그 값을 갱신해야 하며, 동일하다고 확인된 경우에만 upload를 생략한다.

### 8.4 두 개의 slot과 event 의존성

slot마다 다음을 따로 소유한다.

```text
host_input / host_descriptor
GPU descriptor / GPU partial workspace / output ownership
D2H_done / H2D_done / compute_done
generation_id / capacity / live_counts
```

상태 전이는 다음을 지킨다.

```text
FREE → [GPU-origin이면 D2H 진행] → HOST_READY
→ PLANNING → H2D_INFLIGHT → COMPUTE_INFLIGHT → DONE → FREE
```

- CPU가 D2H buffer를 읽기 전에 `D2H_done`을 확인한다.
- CPU가 pinned descriptor를 덮어쓰기 전에 이전 H2D가 끝났는지 확인한다.
- 같은 slot의 device descriptor·workspace를 덮기 전에 이전 compute가 끝났는지 확인한다.
- compute stream은 해당 slot의 H2D event만 기다린다.
- 독립 slot의 전송과 계산을 겹치되, global `cudaDeviceSynchronize()`를 내부 루프에 넣지 않는다.
- 최종 사용자가 결과를 소비하는 완료 의존성은 그대로 보존한다.

일단 compute 완료까지 slot 전체를 재사용하지 않는 보수적 구현으로 시작해도 된다. lifetime을 분리해 더 일찍 host slab만 재사용하는 최적화는 정확성 확인 뒤에 한다.

### 8.5 CUDA Graph 처리

동적 mask마다 graph를 새로 capture하지 않는다. 초기에는 일반 launch의 pool-fresh를 구현하고, 그다음 고정 주소·capacity·kernel variant에 대한 graph를 사용한다.

graph의 grid capacity보다 live task가 적으면 kernel 초입에서 bounds check 후 반환하게 한다. count를 읽기 전에 잘못된 TaskIds를 접근하지 않도록 한다. 무조건 거대한 grid를 capture해 빈 CTA를 과도하게 실행하지 않는다.

plan이 bucket별로 달라지면 미리 정의한 작은 geometry/capacity 집합을 사용한다. graph-cache miss·재capture·capacity 확대는 별도로 계측한다. 이를 매번 발생시키면서 warm replay 시간만 보고하지 않는다. [N5]

### 8.6 CPU 계획기 재사용

원 배포본에는 per-call ndarray 생성·복사와 여러 vector/map 작업이 있다. [C1, C3] 서버 구현을 확인해 다음을 적용한다.

1. worker별 incidence·touched·signature·task buffer를 초기화 때 예약한다.
2. `incidence[nb]` 전체 clear 대신 touched-entry reset 또는 epoch stamp를 비교한다. dense 입력에서 더 느리면 dense-clear 경로를 유지한다.
3. signature는 고정 폭 정수와 정렬·선형 scan으로 생성한다. edge별 tree node 할당을 피한다.
4. 병합 후보마다 Task 복사·map 생성부터 하지 않는다. 집합 크기·비용을 먼저 계산하고 채택된 쌍만 materialize한다.
5. 출력 descriptor와 reduction CSR은 count→prefix sum→fill 순서로 예약된 영역에 쓴다.
6. CPU 비용 측정에서 JSON·edge-validation·full hash 계산을 분리한다. 단, 온라인 cache key나 실제 guard에 사용하는 검사는 hot path 시간에 남긴다.
7. CPU worker는 `1,2,4,8,16`을 비교한다. 작업 크기가 작으면 1개를 유지한다. thread 수 증가를 개선으로 간주하지 않는다.
8. GPU에 가까운 NUMA domain에서 pinned buffer를 준비하고 worker를 배치한다. 운영 중인 AMX worker와 SMT sibling을 공유하지 않는다.

### 8.7 목표와 완료 조건

hot path의 **application 요청 기준** device allocation·host pin 호출 수를 0으로 만든다. allocator가 내부 cache를 재사용했다는 것과 애초에 allocation 요청을 없앴다는 것을 구분해 기록한다.

대표 Nq512 형상에서 `CPU plan+pack`의 초기 engineering 목표는 50 μs 이하로 둔다. 이는 전체 성공 기준이 아니다. GPU에서 얻은 이득이 8 μs이면 추가 준비비가 50 μs인 경로는 여전히 실패다.

최종 손익은 다음 조건과 실제 wall time으로 판정한다.

\[
\Delta T_{GPU} > \Delta T_{exposed\ host+transfer}.
\]

전송 목표는 원인이 분해되기 전 임의의 μs로 고정하지 않는다. 같은 크기 slab의 단일 DMA 대조를 만들고 pack·API submit·DMA·wait를 각각 기록한다. `cudaMemcpyAsync` API 누적시간을 순수 DMA 시간으로 부르지 않는다. [N3, N4]

---

## 9. P6 — whole-query 기본 경로와 조건부 pipeline

### 9.1 기본값

`stream_bench`의 기본 동작은 **현재 준비된 query 전체를 한 번에 계획·전달·실행**하는 것으로 바꾼다. 기존 `chunk=256` pipeline은 대조 옵션으로 보존한다.

가능하다면 서로 독립된 request batch A/B를 번갈아 처리하면서 CPU가 B의 plan을 만들고 GPU가 A를 실행한다. B의 Q와 sparse mask가 실제로 준비됐을 때만 허용한다. 아직 생성되지 않은 다음 token이나 다음 layer의 routing을 가정하지 않는다.

### 9.2 chunk를 다시 허용할 조건

같은 작업량에서 다음 세 값을 모두 비교한다.

```text
whole-query pool-fresh
chunked serial pool-fresh
chunked pipelined pool-fresh
```

run-only whole-query 값도 참고로 남기되 전체 준비비를 포함한 비교와 섞지 않는다. chunk 후보는 `{whole, 1024, 512, 256}`이고 query 수보다 큰 값은 제외한다. chunk마다 P5의 slot을 재사용한다.

pipeline이 chunk-serial만 이기고 whole-query를 못 이기면 채택하지 않는다. 특정 chunk로 CPU 시간을 가린 대신 GPU가 더 작은 작업을 반복했다면 그 비용도 그대로 포함한다.

### 9.3 GPU-origin mask

GPU에서 만들어진 CSR offset·길이·block ID가 필요하면 모두 전달 비용에 포함한다. 길이를 얻기 위한 `.item()`·동기화를 숨기지 않는다. 고정 top-k가 실제 계약인 경우에는 고정 길이를 사용해도 되지만 가변 길이 입력을 고정 길이로 가정하지 않는다.

GPU가 가진 mask를 CPU로 가져와 다시 보내는 경로가 최강 GPU baseline보다 느리면, **CPU-origin 또는 실제 plan 재사용 가능 영역으로 적용 범위를 제한**한다. GPU planner를 별도로 만들어 빠르더라도 그것을 CPU 공동 실행 성과로 기록하지 않는다.

---

## 10. 서버 에이전트가 구현할 도구와 명령

**이 절의 `tools/*`는 이번 작업에서 작성해야 할 실행기다. 이미 제공된 파일이라고 가정하지 않는다.** 현재 소스의 기능을 호출하는 얇은 driver로 만들고 kernel/planner 본체를 별도 구현으로 중복하지 않는다. driver가 생성된 뒤 아래 명령으로 동일 과정을 재현할 수 있어야 한다.

### 10.1 필수 파일

| 파일 | 역할 |
|---|---|
| `configs/rbc_v02.yaml` | 후보 공간·seed·반복·정확성·채택 기준 |
| `tools/run_v02_campaign.py` | 단계별 실행과 선행 산출물 검사 |
| `tools/dump_plan.py` | 분할 전후 descriptor·비용·선택 이유 |
| `tools/calibrate_plan_cost.py` | profiling 없는 실행시간 교정 |
| `tools/compare_v02.py` | 같은 기준의 paired 비교·신뢰구간·적용률 |
| `tests/test_v02_output_ownership.py` | direct/partial/empty 출력 소유권 |
| `tests/test_v02_runtime_lifetime.py` | slot 재사용·generation·동적 mask |
| `tests/test_v02_plan_finalization.py` | 선택 후 계획 불변·분할 보존 |

### 10.2 설정 초안

```yaml
version: rbc-v02
scope: exact_mask_operator

correctness:
  inherit_server_v01_metric: true
  bf16_relative_limit: 0.005
  fp16_relative_limit: 0.001
  reject_nonfinite_nonempty_rows: true

input:
  patterns: [common_private, random, clustered, disjoint, full]
  nq: [256, 512, 2048]
  nk: [16384, 65536, 131072]
  g: [4, 8, 16]
  d: 128
  bk: 128
  selected_blocks: [8, 16, 32]
  reproduce_seed: 42
  calibration_seeds: [0, 1, 2, 3, 4]
  evaluation_seeds: [101, 102, 103, 104, 105, 106, 107, 108, 109]

plan:
  policies: [q_outer, signature, bounded_merge]
  split_candidates: [none, one_wave, two_waves]
  default_forced_min_tasks: 0
  compare_after_finalize: true
  bounded_merge_default: false

kernel:
  max_m: [16, 32, 64, 128]
  bn: [32, 64, 128]
  num_warps: [4, 8]
  num_stages: [1, 2, 3]
  staged_search: true
  direct_output: true
  full_membership_fast_path: true
  force_register_limit: false

runtime:
  slots: 2
  descriptor_h2d: packed_single_slab
  plan_cache: disabled_for_dynamic_mask_tests
  steady_state_allocation: forbidden
  default_execution: whole_query

measurement:
  paired_outer_repeats: 9
  graph_replays_per_sample: 100
  bootstrap_resamples: 10000
  record_raw_samples: true
  cache_modes: [warm_replay, rotating_working_set]
  primary_mask_origin: gpu

acceptance:
  target_total_latency_reduction: 0.10
  diagnostic_gpu_reduction: 0.05
  selector_median_regret_max: 0.03
  selector_worst_regret_max: 0.05
  engineering_representative_plan_pack_us: 50
```

이 YAML의 전체 Cartesian product를 실행하지 않는다. 아래 E0~E5 순서와 상위 후보 제한을 따른다. Nq·Nk·G의 의미와 tensor shape은 서버 v0.1에 맞게 확인한다. `primary_mask_origin`은 **mask의 실제 생성 위치**를 뜻하며, host-origin 실험도 별도 실행한다.

### 10.3 실행 명령 계약

```bash
# 아래 tools는 이번 작업에서 구현한 뒤 실행한다.
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage reproduce
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage plan-audit
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage kernel
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage calibrate
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage online
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage pipeline
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage report
```

`--stage report`는 없는 값을 계산해 채우지 말고 `NOT_RUN`, `UNSUPPORTED`, `FAILED`를 구분한다. 실제 model capture가 아직 없으면 합성 실험 결과만 출력하고 capture 항목은 미수행으로 둔다.

---

## 11. 수행할 대조실험

### E0. 기준선 복구 — 대표 세 입력만

Nq512/Nk65536/G8/D128/BK128에서 common/private·random·clustered를 재현한다. v0.1의 같은 plan/tile/config와 timing 방식을 보존한다. NCU 비교를 재현할 때는 `min_tasks=264` 등 그 profile의 설정을 그대로 기록한다.

일반 실행시간은 profiling 없이 측정한다. 재현값이 과거값과 다르면 먼저 version·clock·plan·cache 상태를 기록하고 새 paired baseline을 고정한다. 과거 86 μs를 강제로 맞추려고 설정을 선택하지 않는다.

### E1. 최종 계획 선택과 분할

같은 세 입력에서 P1 전후를 비교한다. 최적 타일 하나만 먼저 고정한 작은 대조로 시작하고, 이후 같은 탐색 예산으로 정책별 최적 후보를 찾는다.

필수 표:

```text
policy / split / BM / ntasks / KV_visits / partial_slots
main_us / merge_us / GPU_total_us / selected_reason
```

분할 none이 좋아졌다면 task 수뿐 아니라 Q·partial·bucket 변화와 연결해 설명한다. 방문량이 그대로면 KV 재읽기 제거라고 쓰지 않는다.

### E2. GPU 본체 변경 — 공통 실행기와 RBC 효과 분리

다음 순서로 각 변경의 효과를 저장한다.

| 실험 | P1 | direct output | BN·V-load·FULL | 목적 |
|---|---|---|---|---|
| A | 이전 | 이전 | 이전 | v0.1 보존 |
| B | 적용 | 이전 | 이전 | 분할·최종 선택 효과 |
| C | 적용 | 적용 | 이전 | partial/merge 제거 효과 |
| D | 적용 | 적용 | 단계별 적용 | main 실행 효과 |

각 행에서 Q-outer와 RBC를 모두 실행한다. `RBC D`를 `Q-outer A`만으로 비교하지 않는다. 동일 환경의 `Q-outer D`도 기준선이다. signature-only가 더 빠르면 그것을 숨기지 않는다.

### E3. 교정된 선택기와 oracle

교정 seed로 비용모델을 확정하고 evaluation seed에서는 수정하지 않는다. 각 held-out 입력에서 다음을 기록한다.

```text
predicted winner
실제 선택한 candidate
모든 후보를 나중에 측정한 oracle winner
selected vs oracle regret
selected vs best Q-outer 시간
RBC 고유 계획 사용 여부
```

oracle는 결과 해석용이며 온라인 시간에 candidate 전수시험을 넣지 않는다. 선택기가 Q-outer로 돌아가 손실만 막은 경우와, 다른 분해를 선택해 이긴 경우를 나눈다.

### E4. 전체 준비비 포함 비교

측정 모드를 다음처럼 고정한다.

| 모드 | 포함 비용 | 용도 |
|---|---|---|
| `run_only` | 사전 준비된 plan의 GPU·merge | GPU 개선 여지 |
| `cold_init` | capacity 할당·pin·graph 준비 등 | 초기화 비용. JIT는 별도 항목 |
| `pool_fresh_host` | 새 host mask 계획·pack·H2D·GPU·merge·완료 | host-origin 실사용 |
| `pool_fresh_gpu` | GPU mask 준비 완료 이후 필요한 D2H·계획·H2D·GPU·merge·완료 | GPU-origin 주 판정 |
| `exact_plan_cache` | 실제 동일 계획 재사용의 lookup/validation·필요 갱신·GPU | 별도 재사용 시나리오 |

`pool_fresh`에서는 shape capacity만 재사용하고 mask 내용은 바꾼다. mask가 달라졌는데 이전 descriptor를 재생하면 실패다. 같은 mask를 반복하되 cache를 꺼서 측정한 값과, 서로 다른 mask를 매번 계획한 값을 각각 남긴다.

native에도 workspace·plan 수명 관리 등 해당 API가 지원하는 재사용을 허용한다. native의 Python 객체를 매번 새로 만들게 하고 RBC만 preallocated로 실행한 비교를 최종 성과로 쓰지 않는다. native API가 그런 동적 계획 갱신을 지원하지 않는다면 실제 제한과 최소 지원 경로를 기록한다.

### E5. pipeline과 실제 입력

E2에서 의미 있는 non-Q GPU 이득이 있고 E4에서 준비비를 감당할 가능성이 남을 때만 pipeline을 확대한다. whole-query, chunk serial, chunk pipeline을 같은 mask 출발점·동일 작업량·같은 buffer 재사용 조건에서 비교한다.

실제 sparse attention capture가 있으면 모델·layer·dtype·RoPE 적용 상태·cache identity·selected mask·query position을 보존한다. capture가 없다고 dense attention에 임의 sparse mask를 넣어 실제 모델 성과로 쓰지 않는다.

### 입력 생성 오류 수정

`disjoint`가 실패한 이유가 `Nq*selected_blocks > available_blocks`라면 그 형상에서 진짜 disjoint를 생성할 수 없다. 기존 형상은 `UNSUPPORTED_DISJOINT_GEOMETRY`로 남긴다. 별도 host/GPU correctness 입력은 수학적으로 가능한 크기로 추가한다. 겹치는 block을 허용하면서 disjoint라고 표시하지 않는다.

---

## 12. 측정과 profiler 사용 규칙

### 12.1 반복과 시간 경계

CUDA/JIT warmup은 먼저 완료한다. 각 반복에서 같은 입력의 후보 실행 순서를 무작위화한다. 9개의 outer repeat를 수행하고, GPU run-only는 반복당 100회 replay의 시간을 사용한다. mask seed의 차이와 동일 mask 반복 측정의 잡음을 별도로 저장한다.

GPU-origin 전체 시간의 시작점은 **실제 mask 생성이 완료된 시점**이다. indexer 자체를 제외하려면 모든 후보에 같은 경계를 적용한다. CSR 길이·offset이 host에 미리 있다는 가정은 실제로 성립할 때만 허용한다. 최종 output이 다음 GPU 연산에서 소비 가능해진 시점까지를 완료로 본다.

CPU와 GPU 구간의 시간을 단순히 더해 전체 시간으로 쓰지 않는다. 실제 wall time과 Nsight Systems critical path를 사용한다. 비교 직전의 device synchronization은 격리된 단일 호출 측정용으로만 허용하고 pipeline 내부에 반복 삽입하지 않는다.

### 12.2 cache 조건

- `warm_replay`: 같은 Q/K/V·plan을 반복한다. 반복 재생 성능으로 표시한다.
- `rotating_working_set`: 실제 실행을 모사하도록 여러 Q/K/V·mask를 번갈아 사용한다. 선택된 KV 작업 집합과 L2 크기를 함께 기록한다.
- `ncu_cold`: 원인 분리를 위한 profiler cache-control 조건이다. 주 성능값과 분리한다.

세 조건의 결과를 섞어 하나의 speedup으로 만들지 않는다. 여러 input bank를 생성하는 비용은 benchmark input 준비로 따로 기록하되, 실제 path에서 필요한 데이터 전송은 제외하지 않는다.

### 12.3 profiler

먼저 설치 버전에서 metric·option 지원을 조회한다. 특정 metric 이름을 추측해 0으로 채우지 않는다.

```bash
ncu --version > results/v02/manifest/ncu_version.txt
ncu --query-metrics > results/v02/manifest/ncu_metrics.txt
nsys --version > results/v02/manifest/nsys_version.txt
```

대표적인 경우만 프로파일한다. 수집 대상은 main/merge별 시간, DRAM read/write, L2 요청·hit 관련 값, local load/store, register/thread, shared memory/CTA, warp/thread 수, theoretical/achieved occupancy, stall과 launch 수다. 같은 plan hash·kernel variant의 일반 반복 결과와 연결한다.

NCU는 replay·cache control·serialization·clock 처리 때문에 일반 실행과 시간이 달라질 수 있다. warm execution의 cache 관계를 보려면 설치 버전에서 지원하는 application replay와 cache-control을 확인해 사용한다. profiling 중 CUDA event·host wall 시간을 정상 성능값으로 보고하지 않는다. [N2]

NSYS에서는 `cudaMemcpyAsync` host API 시간, GPU copy activity 시간, CPU packing·allocation·wait를 구분한다. 동시에 발생하는 activity를 합해 실제 wall time처럼 보고하지 않는다. [N4]

---

## 13. 단계별 채택·중단 규칙

### 13.1 GPU 단계

P1~P3 적용 후 non-Q RBC 계획이 최강 공통 Q-outer 대비 GPU 시간을 **5% 이상** 줄이는 held-out 영역이 있으면 전체 runtime 평가를 진행한다. 이 5%는 초기 진입 기준이며 최종 10% 전체시간 목표와 다르다.

5% 미만의 유의한 개선이 있으면 기록하고, 측정된 host overhead 차이를 감당할 수 있는지 산술로 확인한 경우에만 진행한다. 개선이 없으면 P5의 일반 runtime 정리는 해도 되지만 대규모 sparse 모델 통합·pipeline 실험으로 확대하지 않는다.

### 13.2 전체 단계

각 workload family에서 입력별 paired latency ratio를 만든다. 9개 독립 평가 seed를 기본 단위로 집계하고 bootstrap 신뢰구간을 보고한다. 같은 mask의 100회 replay를 100개의 독립 workload처럼 사용하지 않는다.

최종 채택은 다음을 모두 충족해야 한다.

1. 동일 정확성 계약 통과.
2. 사전에 정한 적용 영역에서 최강 준비비 포함 기준선 대비 완료시간 감소 10% 이상.
3. 95% 신뢰구간이 개선 방향을 지지.
4. random·clustered 등 비적용 영역의 회귀를 fallback으로 제한. 3% 초과 회귀는 원인을 적고 해당 영역 비활성화.
5. non-Q RBC 실행이 실제 개선에 기여했다는 ablation 존재.

fallback의 회귀 제한은 목표이지 보장 정리가 아니다. 적용 영역은 calibration 입력의 구조적 특징으로 정의하고, evaluation 결과를 본 뒤 유리한 seed만 남기지 않는다.

### 13.3 시간 예산

초기 작업 범위는 P1~P5의 대표 세 입력과 교정·독립 평가로 제한한다. GPU 실측 예산은 우선 누적 8 GPU-hours를 상한으로 둔다. 이는 예상 소요시간이 아니라 재량 확대를 막는 운영 기준이다. correctness 실패 복구, 컴파일, CPU-only 작업 시간은 별도 기록한다.

상한에 도달하면 구현·원본 결과·남은 병목을 반환한다. 새 알고리즘 이름, 새 모델, 새 정밀도로 바꿔 추가 캠페인을 자동 시작하지 않는다.

---

## 14. 반드시 반환할 결과물

다음 디렉터리를 그대로 압축해 제출한다. 실행 코드가 없어서 다음 세션이 다시 구현해야 하는 상태로 반환하지 않는다.

```text
RBC_Attn_v02_result/
  src/ 또는 source.patch + base commit
  configs/rbc_v02.yaml
  tools/                         # 이번에 작성한 실행 도구
  tests/                         # 추가 검사
  manifest/
  plans/                         # 분할 전후·선택·실행 hash
  calibration/                   # 교정 입력과 독립 검증 분리
  raw/                           # 모든 paired 시간 원본
  profiles/                      # NCU·NSYS와 실행 대응표
  FINAL_RESULT.md
  MANIFEST.sha256
```

### 14.1 FINAL_RESULT.md 첫 표

| 입력 | mask 출발점 | 비교 모드 | 최강 기준선/시간 | RBC 선택/시간 | 감소율 | non-Q RBC 비율 | 정확성 | 판정 |
|---|---|---|---:|---:|---:|---:|---|---|
| common/private Nq512 | GPU | pool-fresh | 측정값 | 측정값 | 계산값 | 측정값 | 결과 | 결과 |
| random Nq512 | GPU | pool-fresh | 측정값 | 측정값 | 계산값 | 측정값 | 결과 | 결과 |
| clustered Nq512 | GPU | pool-fresh | 측정값 | 측정값 | 계산값 | 측정값 | 결과 | 결과 |
| Nq2048 | GPU | whole / pipeline | 측정값 | 측정값 | 계산값 | 측정값 | 결과 | 결과 |
| 실제 capture | 실제 위치 | 동일 경계 | 없으면 NOT_RUN | — | — | — | — | — |

다음 표에는 P1~P6 각각의 `단독 효과 / 누적 효과 / 공통 Q-outer도 얻은 효과 / RBC 추가 효과`를 나눠 쓴다. NCU 바이트와 정상 실행시간의 조건도 일치 여부를 표시한다.

### 14.2 원본 JSON 필수 필드

```json
{
  "version": "rbc-v02",
  "source_commit": null,
  "gpu_uuid": null,
  "mask_origin": "gpu",
  "timing_mode": "pool_fresh",
  "input_hash": null,
  "mask_hash": null,
  "q_position_hash": null,
  "selected_policy": null,
  "plan_hash_before_upload": null,
  "plan_hash_executed": null,
  "split_policy": null,
  "actual_tasks": null,
  "kernel_buckets": null,
  "direct_rows": null,
  "multi_owner_rows": null,
  "partial_slots": null,
  "kv_unique_blocks": null,
  "kv_block_visits": null,
  "descriptor_live_bytes": null,
  "descriptor_capacity_bytes": null,
  "host_pin_calls_hotpath": null,
  "device_allocation_requests_hotpath": null,
  "d2h_bytes": null,
  "h2d_bytes": null,
  "h2d_copy_count": null,
  "plan_cache_hit": false,
  "cpu_plan_pack_us": [],
  "dma_activity_us": [],
  "main_us": [],
  "merge_us": [],
  "gpu_total_us": [],
  "wall_us": [],
  "exact_edge_passed": null,
  "numeric_passed": null,
  "notes": []
}
```

위 null은 **산출물 스키마의 미입력 자리**다. 실제 결과에서 측정하지 않은 값은 null과 사유를 유지하며, 0으로 바꾸지 않는다.

---

## 15. 금지 사항

- 전체 KV 합집합 바이트만으로 HBM 비용을 계산하지 않는다.
- Q-outer fallback·pinned pool·일반 kernel fusion을 새 알고리즘의 신규성으로 주장하지 않는다.
- RBC에만 새 kernel·buffer pool·graph를 적용한 뒤 구형 기준선과의 차이를 RBC 분해의 성과로 귀속하지 않는다.
- task 수나 occupancy 목표를 맞추기 위해 최종 선택 계획을 다시 변경하지 않는다.
- register 수를 줄이기 위해 정밀도·FP32 누산을 바꾸지 않는다.
- GPU-origin mask의 길이 획득·D2H·동기화를 계측 밖으로 숨기지 않는다.
- 같은 shape라는 이유로 다른 mask의 plan을 재사용하지 않는다.
- chunk serial만 이긴 pipeline을 whole-query 대비 개선으로 기록하지 않는다.
- 460개와 57/37개라는 테스트 수 차이를 동일 코드의 검증 누락으로 단정하지 않는다.
- 합성 operator latency를 실제 모델 tok/s 또는 논문 SOTA 성과로 바꿔 쓰지 않는다.

---

## 16. 최종 실행 순서

```text
현재 서버 소스·v0.1 결과 보존
→ P1: 강제 분할 제거 + 최종 계획 단위 비교
→ P2: direct output + compact merge
→ P3: K microtile + V 생존 구간 + FULL 경로
→ E1/E2: 같은 개선 실행기에서 Q-outer와 비교
→ P4: 비용 선택 교정 + non-Q 추가 이득 확인
→ P5: pinned/device arena + 단일 전달 + C++ 작업공간 재사용
→ E4: 새 mask와 실제 출발점의 전체시간 비교
→ 유효한 경우에만 P6/E5: whole-query 대비 pipeline·실제 capture
→ 코드·원시 결과·적용 범위 반환
```

**이번 작업의 완료물은 개선 방향에 대한 설명이 아니라, 수정된 소스와 같은 기준선으로 비교한 원본 실행시간이다.** 가속을 미리 약속하지 않는다. 가속이 없으면 위 변경 중 어디서 비용이 남는지 수치와 코드로 반환한다.

---

## 부록 A. 근거와 확인 범위

### 내부 자료

- **[R1] `RESULT.md`**: 서버 v0.1의 수치·구현 경위·검증 범위. 본 지시서의 측정값은 이 문서에서 가져왔다. 서버 원본 소스·JSON·NCU CSV는 이번 작성 환경에 없으므로 직접 감사한 것으로 표현하지 않았다.
- **[R2] `RBC_Attn_v01_실험결과_검토_20260912.md`**: 결과 해석, 합집합 비용·분할·fallback·프로파일 조건에 대한 기존 검토. 원시 측정자료가 아니라 검토 문서다.
- **[D1] `RBC_Attn_알고리즘_구현명세_20260911.md`**: exact-edge 계약, 비용식, Q-outer fallback, partial 결합 명세.
- **[C1] `RBC_Attn_실행패키지_20260911.zip`의 `src/planner.cpp`**: `Cost`, `row_tasks`, `join`, `remap`, `min_tasks` 분할과 descriptor 생성. 직접 열어 확인한 **원 배포본**이며 서버 재구현과 같다는 뜻은 아니다.
- **[C2] 같은 ZIP의 `rbc/kernels.py`**: multi-KV main과 전체 행 merge, K/V load 위치.
- **[C3] 같은 ZIP의 `rbc/planner.py`, `rbc/runtime.py`, `bench.py`, `stream_bench.py`**: per-call 배열·pin·upload·workspace, 기본 min_tasks와 시간 경계.

### 공식 구현·계측 문서

- **[N1] NVIDIA Hopper Tuning Guide**, §1.4.1.1 Occupancy, §1.4.1.2 TMA. register·shared memory·CTA 자원 제한을 확인했다. 본 지시서는 새 TMA kernel이 구현됐다고 주장하지 않는다.  
  https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- **[N2] NVIDIA Nsight Compute Profiling Guide**, Workload Durations, Cache Control, Replay. profiling 시간과 일반 실행의 차이, cache-control을 확인했다.  
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- **[N3] NVIDIA CUDA C++ Best Practices Guide**, Data Transfer Between Host and Device, Pinned Memory, Asynchronous Transfers. 작은 전송 묶기와 pin 비용·lifetime 설계의 근거다.  
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- **[N4] NVIDIA Nsight Systems User Guide**, CUDA trace·GPU activity. API 제출시간과 장치 activity를 구분하기 위한 자료다.  
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- **[N5] NVIDIA CUDA Programming Guide**, §4.2 CUDA Graphs: definition/instantiation/execution, stream-capture 의존성, graph update를 확인했다. 서버의 PyTorch·CUDA 버전에서 가능한 API는 별도 확인한다.  
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html

공식 문서는 하드웨어·API·계측 제약의 근거로만 사용했다. P1~P6의 구체적인 수정과 채택 수치는 **이번 작업을 위해 제안한 구현 지시**다. 서버 성능 개선이나 신규성이 이미 검증됐다는 뜻이 아니다.
