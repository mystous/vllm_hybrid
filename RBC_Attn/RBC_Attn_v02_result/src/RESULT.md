# RBC-Attn v0.1 — 서버 검증 결과

**측정 2026-09-11 ~ 09-12 / H100 80GB HBM3 × 4 (GPU 4~7, NUMA 1) / 전용 컨테이너 `rbc-dev`**
명세: `docs/RBC_Attn_design.md` (설계), `docs/AGENT_TASK.md` (지시서).
원시 자료: `results/*.json`, `results/ncu/*.csv`, `results/nsys_pipeline.nsys-rep`.

> **구현 경위**: 업로드된 것은 명세 두 건뿐이었고 코드 패키지는 제공되지 않았다.
> 따라서 이 저장소의 `src/`, `rbc/`, `bench.py`, `stream_bench.py`, `tests/` 는 명세로부터
> 새로 구현한 것이다. 설계 §3 의 구조 지표 표와 §2.2 의 결합 수식을 재현하는지로 구현 충실도를 확인했다.

---

## 1. 지시서 §7 이 요구한 첫 표

입력별 최강값 (ms). `RBC run-only` 는 CUDA graph 재생, `RBC fresh` 는 새 계획 + 할당 + H2D +
GPU + merge + 동기화. 정책마다 `max_m ∈ {16,32,64,128}` 을 동일하게 탐색해 각자의 최적을 썼다.

| 입력 | 실제/합성 | 최강 native | 최강 내부 대조 | RBC run-only | RBC fresh | RBC pipeline | 정확성 | 판정 |
|---|---|---|---|---|---|---|---|---|
| Nq256 / 64k / G8 | 합성 common_private | 0.229 | **0.053** (Q-outer) | 0.054 | 1.067 | — | 통과 | **실패** |
| Nq512 / 64k / G8 | 합성 common_private | 0.400 | **0.086** (Q-outer) | 0.087 | 1.112 | — | 통과 | **실패** |
| Nq2048 / 64k / G8 | 합성 common_private | 1.328 | 0.302 (Q-outer) | **0.300** | 2.579 | 7.515 | 통과 | **실패** |
| Nq512 / 16k / G8 | 합성 common_private | 0.390 | **0.084** (Q-outer) | 0.085 | 1.171 | — | 통과 | **실패** |
| Nq512 / 128k / G8 | 합성 common_private | 0.409 | **0.088** (Q-outer) | 0.088 | 1.211 | — | 통과 | **동률** |
| Nq512 / 64k / G4 | 합성 common_private | 0.256 | **0.074** (Q-outer) | 0.074 | 1.369 | — | 통과 | **동률** |
| Nq512 / 64k / G16 | 합성 common_private | 미지원 | **0.112** (Q-outer) | 0.113 | 1.290 | — | 통과 | **실패** |
| sel8 / 64k / G8 | 합성 common_private | 0.232 | **0.052** (Q-outer) | 0.053 | 1.026 | — | 통과 | **실패** |
| sel32 / 64k / G8 | 합성 common_private | 0.722 | 0.153 (Q-outer) | **0.152** | 1.481 | — | 통과 | **실패** |
| **random** / 64k / G8 | 합성 반례 | 0.402 | **0.111** (Q-outer) | 0.150 | 1.333 | — | 통과 | **실패 (−35%)** |
| **clustered** / 64k / G8 | 합성 반례 | 0.409 | **0.115** (Q-outer) | 0.146 | 1.314 | — | 통과 | **실패 (−27%)** |

판정 기준은 지시서 §8: *"동일 입력에서 native 및 내부 최강 Q-outer/KV-outer 보다 fresh wall time
또는 실제 pipeline wall time 이 감소해야 한다."* **어떤 입력에서도 감소하지 않았다.**

- 실제 모델 capture 는 측정하지 않았다 (capture 를 제공받지 못했다). 위는 전부 합성이다.
- `disjoint` 패턴은 Nq=512·sel=16 에 블록이 부족해 생성 실패했다 (미측정).
- G=16 은 FlashInfer 가 `Unsupported group_size: 16` 로 거부해 native 대조 불가.

## 2. 정확성

| 항목 | 결과 |
|---|---|
| `tests/test_host.py` + `tests/test_native_metadata.py` | **57 passed** |
| `tests/test_gpu.py` | **37 passed** (CUDA 실행 확인, skip 아님) |
| FP64 참조 대비 출력 상대오차 (BF16 입력) | 2.1e-03 ~ 2.4e-03 |
| LSE 오차 | 5.7e-05 |
| NaN/Inf 오염 | 없음 |
| 빈 행 | out=0, LSE=−inf (정의대로) |

허용오차는 **측정 전에** BF16 상대 5e-3 / FP16 1e-3 으로 고정했고 결과에 맞춰 넓히지 않았다.
네 정책(Q-outer/KV-outer/signature-only/RBC)이 서로 다른 분해를 쓰면서 2e-3 이내로 같은 값에
수렴하며, FlashInfer native 와도 일치한다(출력 최대차 1.8e-04).

**FlashInfer 의 LSE 는 log2 밑이다.** 측정으로 확인했다 (비율 1.442695 = 1/ln2, 자연로그 환산 후
최대차 9.5e-07). 이를 모르면 LSE 비교에서 3.27 만큼 틀린 결론이 나온다.

## 3. 설계 §3 의 구조 지표 — 재현됨

8 query, G=16, D=128, BK=128, 공통 8 + 개인 8:

| 실행 | 설계 예측 (task / partial / KV방문 / padded비) | 측정 |
|---|---|---|
| Q-outer M128 | 1 / 8 / 72 / 4.5배 | **1 / 8 / 72 / 4.50배** |
| KV-outer M128 | 72 / 128 / 72 / 1배 | **72 / 128 / 72 / 1.00배** |
| RBC M128 | 9 / 16 / 72 / 1배 | **9 / 16 / 72 / 1.00배** |

## 4. 왜 이득이 없는가 — 원시 계측에 의한 분류

지시서 §7 이 요구한 분류다. Nsight Compute 로 실제 metric 목록을 조회한 뒤 수집했다
(metric 이름을 추측하지 않았다). Nq=512, Nk=64k, G=8, D=128, min_tasks=264.

| 정책 | 커널 | DRAM 읽기 | DRAM 쓰기 | L2 요청 | 시간 | 레지스터 | 점유율 |
|---|---|---|---|---|---|---|---|
| Q-outer | main | **37.4 MB** | 2.9 MB | 366 MB | **109.0 µs** | 153 | 12.1% |
| | merge | 4.4 MB | 0 | 12.3 MB | 7.3 µs | 32 | 65.9% |
| KV-outer | main | 147.3 MB | 31.3 MB | 508 MB | 165.6 µs | 153 | 12.2% |
| | merge | **34.1 MB** | 0.02 MB | 60.1 MB | 20.1 µs | 32 | 88.7% |
| signature-only | main | 89.9 MB | 5.6 MB | 414 MB | 135.0 µs | 153 | 11.7% |
| | merge | 4.3 MB | 0 | 12.0 MB | 6.9 µs | 32 | 70.5% |
| **RBC** | main | 91.0 MB | 5.8 MB | 416 MB | 134.9 µs | 153 | 11.6% |
| | merge | **4.3 MB** | 0 | 11.9 MB | 6.9 µs | 32 | 71.1% |

### (a) 설계가 주장한 partial 절감은 실제로 일어난다

RBC 의 merge DRAM 읽기가 KV-outer 대비 **−87.5%** (34.1 → 4.3 MB) 다. 설계 §3 이 예측한
partial write/read 감소율 87.5% 와 일치한다. **논리 바이트 감소가 실제 HBM 바이트 감소로 나타났다.**
레지스터 spill 은 없다 (local load/store 명령 0).

### (b) 그러나 merge 는 전체 시간의 5% 뿐이다

RBC 전체 142 µs 중 merge 는 6.9 µs 다. KV-outer 대비 merge 에서 아낀 13 µs 는 의미가 있지만,
비교 대상인 Q-outer 의 merge 도 이미 4.4 MB / 7.3 µs 로 RBC 와 같다. **설계 §3 표는 Q-outer 의
partial 슬롯이 8, RBC 가 16 이라고 했으나, 실제 시간·바이트에서는 Q-outer 가 동등하거나 유리하다.**

### (c) 시간을 지배하는 것은 main 커널의 KV 재읽기다

Q-outer main 의 DRAM 읽기 37.4 MB 대 RBC 91.0 MB — **RBC 가 2.4배 더 읽는다.**
RBC 는 task 를 264개로 쪼개면서 같은 KV block 을 여러 task 가 중복 적재하는 반면,
Q-outer 는 합집합을 한 번만 읽는다.

**이것이 설계 비용 모형의 결함이다.** §2.1 의 `KV logical bytes = 4·k·BK·D` 는 task 별 KV 를
독립 가산해, task 를 합칠 때 생기는 KV 공유 이득을 계산에 넣지 않는다. 실측에서는
partial 절감(−87.5%)보다 KV 중복 읽기 증가(+143%)가 훨씬 커서 순효과가 음수다.

### (d) 비용 인식 병합은 반례에서 해롭다

random 에서 RBC 0.150 대 signature-only 0.117 (−28%), clustered 에서 0.146 대 0.118 (−24%).
**병합을 할수록 나빠진다.** 설계 §2.1 이 "교정되지 않은 결정 휴리스틱" 이라고 명시한 그대로다.
지시서 §4-1 의 판정에 따르면 **추가 cost-aware merge 의 기여는 확인되지 않았고, 오히려 음수다.**

### (e) 점유율이 낮다

main 커널 점유율이 네 정책 모두 11.6~12.2% 다. 설계 §6 이 경고한 "긴 CTA 로 occupancy 하락" 이
관측된다. 레지스터 153개/스레드가 원인이다.

### (f) CPU 계획비가 GPU 이득을 먹는다

| 항목 | Nq512 기준 |
|---|---|
| RBC GPU 본체 (run-only) | 0.086 ms |
| CPU 계획 | 0.52 ms |
| descriptor H2D | 0.30 ms |
| RBC fresh wall | 1.157 ms |
| **native fresh wall** | **0.744 ms** |

`nsys` 로 확인한 파이프라인 구간(Nq=1024, chunk 256): GPU 커널 총 7.47 ms 대비
`cudaMemcpyAsync` 6.82 ms, `cudaHostAlloc` 1.22 ms. **descriptor 전송과 pinned 할당이
GPU 실행과 맞먹는다.**

### (g) 파이프라인도 whole-query baseline 을 이기지 못한다

Nq=2048, chunk 256 × 8, `--routing-source gpu`:

| 항목 | 시간 |
|---|---|
| whole-query 최강 GPU (Q-outer, run-only) | **0.405 ms** |
| 직렬 chunk (계획 → 실행 반복) | 7.972 ms |
| 파이프라인 (CPU 계획 / GPU 실행 중첩) | 7.515 ms |

중첩으로 직렬 대비 5.7% 를 줄였으나 whole-query 최강 대비 **18.6배 느리다**.
지시서 §4-3 이 금지한 "일부러 직렬화한 chunk baseline 만 이기는" 경우이므로 이득으로 쓰지 않는다.

## 5. 판정

**지시서 §8 의 성공 기준을 충족하지 못했다.**

| 지시서 §4 의 세 대조 | 결과 |
|---|---|
| 1. 같은 executor (Q-outer / KV-outer / signature-only / RBC) | **부정.** 구조 있는 입력에서 Q-outer 와 동률(−0.7~+1.9%), 반례에서 −27~−35% |
| 2. native (FlashInfer paged) | **부분.** run-only 는 4.5~4.7배 빠르나 fresh wall 은 1.55배 느리다 |
| 3. 계획비 포함 (run-only / fresh / pipeline) | **부정.** pipeline 이 whole-query 최강 대비 18.6배 |

선택적 진입 목표였던 "fresh/pipeline wall 10% 개선" 은 달성되지 않았다.

**확인된 것**: 설계의 구조 지표(§3 표)는 정확히 재현되고, partial 절감은 실제 HBM 바이트로 −87.5%
나타난다. 정확성 계약(exact edge partition, 결합 수식)도 성립한다.

**확인되지 않은 것**: 그 절감이 시간 이득으로 이어지지 않는다. 비용 모형이 KV 재사용을 누락해
잘못된 분해를 고르고, 그 효과가 partial 절감을 압도한다.

## 6. 이 결과의 범위

- 이 단계 결과는 **operator latency** 이며 모델 tok/s 가 아니다.
- 실제 모델 capture 를 측정하지 않았다. 합성 입력만이다.
- MSA 저자 커널·실제 서비스 커널과 비교하지 않았다. 패키지 KV-outer 는 저자 커널이 아니다.
- 같은 분해를 GPU 에서 만드는 planner 와 비교하지 않았다 (v0.1 범위 밖). 따라서
  **"CPU 배치가 최적" 이라고 쓸 수 없다.**
- 비용 계수는 교정하지 않았다. 교정하면 결과가 달라질 수 있으나, (c) 의 구조적 누락은
  계수 교정으로 해결되지 않는다 (항 자체가 없다).

## 7. 원인이 분류된 만큼의 다음 단계

1. 비용 모형에 **task 간 KV 공유 항**을 넣는다. 현재 `4·k·BK·D` 를 task 집합 전체의
   KV 합집합 기준으로 바꾸면 병합 결정이 뒤집힐 수 있다.
2. descriptor 를 줄인다. 현재 H2D 0.30 ms 는 GPU 본체 0.086 ms 의 3.5배다.
   task_kmask(8B/블록)와 CSR 을 합쳐 한 번에 올리거나, 계획을 GPU 에서 만든다.
3. 점유율 12% 를 올린다. 레지스터 153개를 줄이거나 BLOCK_M 을 낮춘다.
