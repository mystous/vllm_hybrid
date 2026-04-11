# 분석 노트 — 왜 더 빠른 서버에서 stdout 이 더 큰 병목이 되는가

`20260411_121509_analysis_stdout_scaling_on_fast_hardware`

> **형식**: 실험 run 이 아니라 **분석 노트**. 앞선 두 실험
> (`20260411_090942_h100x4_qwen1.5b_capacity_trace_on_500`,
> `20260411_120746_dev_rtx3090_1.5B_silent_stdout_rerun`) 의 결과를 비교하면서
> "왜 성능 좋은 H100 쪽에서 stdout 이 더 큰 병목이었는가?" 라는 질문에 답한다.

---

## 1. 관찰된 역설

| 환경 | 하드웨어 | stdout 이 문제 됐는가? |
|---|---|---|
| **dev** (i9-12900KF + RTX 3090, TP=1) | 느림 | ❌ — per-req/per-call 로그 다 살아 있어도 throughput 영향 0 |
| **H100x4 + Xeon 8480+** (KVM, TP=4) | **빠름** | ✅ — `TRACE=1` 하나 켰더니 **hybrid wall ×7.6, GPU TPOT 22 → 60 ms** |

직관적으로는 "더 빠른 하드웨어 = stdout 쯤 우습지 않나?" 싶지만 실제는 반대. 원인은 **단일 현상이 아니라 세 가지 효과가 동시에 작동** 해서 발생한다.

---

## 2. 구조적 원인 3 가지

### 2.1 Stdout 은 하드웨어가 빨라져도 scale 되지 않는다

하나의 stdout fd 는 커널에서 **파일 offset lock + 단일 write syscall 경로**. CPU 가 6 배 빠르고 GPU 가 8 배 빠르다고 해서 stdout write throughput 이 그만큼 올라가지 않는다. 한 줄 log 의 kernel path 는 두 서버에서 거의 동일한 수 μs.

→ **stdout 은 하드웨어와 독립적인 상수 성능의 단일 channel**.

### 2.2 "초당 발생하는 이벤트 수" 는 하드웨어 성능에 비례해 불어난다

같은 500 req 벤치를 돌릴 때:

| | dev RTX 3090 | H100x4 | H100 ÷ dev |
|---|---:|---:|---:|
| 1.5B GPU-only duration | 8.13 s | 3.68 s | 0.45× (더 짧음) |
| output TP (tok/s) | 7,577 | 16,743 | **2.21×** |
| GPU worker 프로세스 수 (TP) | 1 | **4** | **4×** |
| 같은 stdout fd 에 쓰는 총 프로세스 수 | 3 (APIServer + GPU worker + CPU engine) | 6 (APIServer + 4 GPU workers + CPU engine) | **2×** |

같은 "decode call 마다 한 줄 찍는다" 정책 아래에서, H100 은 **단위 시간당 2 배 이상 token** 을 생성하고 **2 배 더 많은 프로세스**가 동일 fd 에 경쟁적으로 write 한다. 이 두 요인만으로도 **초당 emit 되는 stdout 라인 수가 ~4 배 이상**. stdout capacity 는 그대로 → **상대적 pressure 가 4 배 이상 증가**.

### 2.3 결정적 차이는 env 파일의 TRACE 정책 자체가 달랐다는 것

| run | `VLLM_HYBRID_TRACE_EVERY` | 의미 |
|---|---:|---|
| dev 060712 (previous 기준선) | 500 | decode 500 call 마다 1 줄 |
| H100x4 090942 (capacity, 문제 재현) | **1** | **decode 1 call 마다 1 줄** |
| H100x4 082501 (7B smoke) | 1 | 1 줄/call (NUM_PROMPTS=10 이라 피해 없었음) |
| H100x4 085801 (1.5B thro-adaptive) | 500 (serve.sh default) | 500 call 마다 |

H100 실험 3 은 **dev 보다 500 배 더 aggressive** 한 설정이었다. 거기다 §2.2 의 "4 배 더 많은 emission rate" 를 곱하면 **총 stdout 부담은 ~2000 배**. 반면 stdout 처리 능력은 변함없음.

→ H100x4 085801 (1.5B thro-adaptive) 은 `TRACE_EVERY=500` 유지 + `throughput-adaptive` 덕에 CPU lane 을 거의 안 써서 log 이벤트 수도 낮았음. 그래서 hybrid wall 25.67 s ≈ gpu-only wall 25.70 s 로 **정상 동작**. 이것이 §2.3 의 직접 증거.

---

## 3. 숫자로 확인 — "fast path 에서 숨어 있던 상수 오버헤드가 드러난다"

각 실험에서 대략적인 stdout emission rate 를 추정해보면:

| 실험 | setting | 대략 stdout lines/s | 결과 |
|---|---|---:|---|
| dev 060712 hybrid | `EVERY=500`, TP=1, 3 proc | ~60 | 정상 (34.9 s wall) |
| dev 120549 hybrid (silent) | TRACE off, TP=1, 3 proc | ~1 | 정상 (34.9 s wall, TPOT 동일) |
| H100 085801 thro-adaptive | `EVERY=500`, TP=4, 6 proc | ~150 | **정상** (25.67 s wall ≈ gpu-only) |
| H100 082501 7B smoke | `EVERY=1`, TP=4, 6 proc, NUM=10 | ~500 | 동작 (508 s wall, 7B BW bound 이 지배) |
| **H100 090942 1.5B capacity** | **`EVERY=1`**, TP=4, 6 proc, NUM=500 | **~수천** | **×7.6 slowdown** |

Line count 자체는 수백 단위지만 **per-line 의 kernel syscall + file offset lock + Python logging 비용이 약 10~100 μs 수준**. 초당 수천 줄을 뿌리기 시작하면 수십~수백 ms/sec 가 순수 log 비용으로 소비된다. API server main thread 는 응답 송신/ZMQ dispatch/라우팅/로그 emit 를 모두 담당하므로, log 가 느려지면 **응답 파이프라인 전체가 직렬로 끌림** → GPU TPOT 이 직접 부풀어 오르는 것.

### TPOT 가 결정적 증거

| 실험 | TPOT (before) | TPOT (after) |
|---|---:|---:|
| dev 1.5B GPU-only (stdout 많음 → 줄임) | 27.79 ms | 27.79 ms |
| dev 1.5B Hybrid (stdout 많음 → 줄임) | 30.05 ms | 29.92 ms |
| H100 1.5B GPU-only (TRACE off, 동일 run 비교 불가) | 22.36 ms | — |
| H100 1.5B Hybrid **TRACE=1** | **60.06 ms** (×2.69) | — (아직 재실험 안 함) |

- **dev**: 초당 ~60 lines → kernel 이 여유롭게 소화 → TPOT 변화 zero. stdout 은 병목이 **아니었다**.
- **H100 TRACE=1**: TPOT 가 22 → 60 ms **로 3 배 부풀어** 올라감. GPU executor 는 별도 process 라 실제 compute 는 빨라진 게 아니라, **응답이 메인 thread 로 돌아오는 경로가 느려진** 것. = stdout I/O 가 직접 증거.
- **dev 에서는 stdout 이 병목 아니었고, H100 에서는 병목이었다** 는 판정은 TPOT delta 로 직접 확정 가능.

---

## 4. 일반화 — "fast-path contention"

이 현상은 이번 hybrid 시스템에만 있는 것이 아니다. 고성능 시스템에서 반복적으로 관찰되는 패턴이다:

> **"하드웨어 병목을 제거할수록, 뒤에 숨어 있던 serialization/logging/IPC 같은 상수 오버헤드가 새 bottleneck 으로 드러난다"**

- 느린 하드웨어: 주 병목 = compute. stdout 은 여유 안에 숨어 있음.
- 빠른 하드웨어: 주 병목 = compute 해소. 다음 순위 bottleneck 이 자동 노출 — 여기선 stdout I/O 가 그 역할.

### 비유

시골 1차선 도로 (dev) → 초당 차량 통과량 자체가 적어 톨게이트에서 "매 차량 기록" 정책 돌려도 톨게이트가 안 막힘.

고속도로 6차선 (H100x4) → 초당 통과량 자체가 훨씬 많은데 동일 정책 ("매 차량 기록") 을 돌리면 톨게이트 한 개에 트래픽이 몰려 **도로 전체가 역류**. 톨게이트 자체의 처리 속도는 두 도로에서 같은데 차량 속도만 빨라지니 톨게이트가 포화되는 것.

→ 이 비유에서:
- **도로 = GPU/CPU compute**
- **톨게이트 = stdout write lock**
- **"매 차량 기록" 정책 = `TRACE_EVERY=1`**
- **"500대 마다 기록" = `TRACE_EVERY=500`** (여유 확보)
- **"기록 안 함" = `TRACE=0` + `TRACE_EVERY=0`** (우리의 silent 패치 default)

---

## 5. 이번 패치의 진짜 가치 — dev 에서 재확인

`20260411_120746_dev_rtx3090_1.5B_silent_stdout_rerun` 결과가 이 분석을 뒷받침한다:

- dev 에서 silent 패치 적용 후 1.5B 벤치 재측정 → **TPOT 완전 동일**, wall 동일, throughput 노이즈 범위 (±1%) 내 변화.
- 즉 dev 에서는 **stdout 이 원래부터 병목이 아니었기 때문에 silent 패치의 측정 가능한 benefit 이 없다**.
- 반대로 **H100 에서는 병목이었기 때문에 silent 패치가 7.6 배 wall time 을 되돌려놓을 잠재력** 이 있다.

이 패치의 진짜 target 은 dev 가 아니라 **"가장 빠른 하드웨어에서 가장 큰 숨은 병목을 제거" 한다는 point** 에 있다.

---

## 6. 시사점

1. **Production serving 에서 per-call stdout 은 금지**. TRACE 는 smoke / 첫 부팅 검증 / 재현 시나리오에만 사용, `NUM_PROMPTS` 는 작게.
2. **env 파일 default 는 silent 가 옳다**. `serve.sh` 의 `VLLM_HYBRID_TRACE_EVERY` default 를 50 → 0 으로 바꾼 이번 패치가 바로 그 의도.
3. **TP 증가 시 stdout 부담이 linear 하게 증가** 한다는 점을 고려해 multi-worker 벤치에서는 특히 silent 을 엄격히 유지.
4. **앞으로 성능 분석 시 TPOT 를 fast-path contention 지표로 사용** 가능. TPOT 가 기대보다 크면 GPU 가 아니라 main thread serialization 쪽을 의심해야 함.
5. **다음 H100 실험 전 이 패치 반영 필수**. 특히 `h100x4_qwen1.5b_hybrid_smoke.env` 를 `NUM_PROMPTS=500` 으로 돌린 `090942_capacity_trace_on_500` 재실험 시 wall time 이 106 s → ~14 s 로 수렴해야 본 패치의 H100 benefit 이 확정됨 (예상 ratio: 0.13×).

---

## 7. 관련 실험

| TS | 디렉토리 | 역할 |
|---|---|---|
| 2026-04-11 08:25 | `20260411_082501_h100x4_qwen7b_smoke_cpu_bw_diagnosis` | H100 7B smoke (stdout 영향 없음, BW bound 분석 주 목적) |
| 2026-04-11 08:58 | `20260411_085801_h100x4_qwen1.5b_thro_adaptive_500` | H100 1.5B throughput-adaptive (TRACE_EVERY=500, 정상 동작 대조군) |
| **2026-04-11 09:09** | **`20260411_090942_h100x4_qwen1.5b_capacity_trace_on_500`** | **H100 1.5B capacity + TRACE=1 (×7.6 slowdown 재현 증거)** |
| 2026-04-11 12:07 | `20260411_120746_dev_rtx3090_1.5B_silent_stdout_rerun` | dev 재벤치 — silent 패치의 TPOT 불변성 (병목 아니었음) 확인 |
| **2026-04-11 12:15** | **본 노트** | **분석: 왜 빠른 서버일수록 stdout 이 더 큰 병목인가** |
