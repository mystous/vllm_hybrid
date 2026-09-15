# IDE_069 결과 — 480B TP4 하이브리드 처리량 향상 (HBM 추가 사용 + hot expert 배치)

> 부모 `IDE_068`. 노드 violet-h100-016 (H100 ×8 / Xeon 8480+ ×2 AMX / DDR5 2 TB / **turbo OFF 2.0 GHz — 모든 CPU 수치는 하한**).
> 스택 SGLang 0.5.18 + kt-kernel (sgl-kt: 0.7.0.post1 소스빌드). 벤치 sonnet 512/128 (prefix 100), seed 42, 요청 수 = 4×C.
> 품질 게이트: greedy 4문항 전 셀 + GSM8K 40문항 (deferral 0 대 4). 원본 `eval/results/20260915_*_ide069_*/`.

## 한눈에

| | 구성 (TP4, GPU 4장) | C32 tok/s | C64 tok/s | HBM/장 |
|---|---|---|---|---|
| 출발점 | expert 전량 CPU (IDE_068 b1) | 43.39 (C16) | — | 17 GiB |
| 물리 id 96 | hotmap 없이 앞 96개 GPU (IDE_068 b6) | 51.55 (C16) | — | 76 GiB |
| hotmap 96 | + 빈도 기반 배치 + cuda graph | 470.26 | — | 74.6 GiB |
| + deferral 4 | IDE_030 처방 | 519.81 / 486.90 | 425.15 | 74.6 GiB |
| **+ KV 40,960** | **최종** | 465 ± 6 | **642 ± 7** | 75.6 GiB |

**출발점 대비 14.8× (C64), IDE_030 기록(C32 490.9 / C64 408.2) 대비 C64 +57 %.** GSM 40: deferral 0 = 39/40, deferral 4 = 39/40.

## 1. hotmap 재생성

IDE_030 의 hotmap 은 소실되어 다시 만들었다. SGLang `--expert-distribution-recorder-mode stat` 으로 TP4 하이브리드(expert 0) 에 sonnet 512/128 C16 64req 를 흘려 `logical_count (1000 step × 62층 × 160)` 를 얻고 (총 **71,981,504** expert 호출), 층마다 빈도 내림차순으로 정렬해 `physical_to_logical_map` 을 만들었다 (`eval/ide069/build_hotmap.py`).

| 상위 N | 커버리지 평균 | 최소(층) |
|---|---|---|
| 64 | 94.1 % | 77.1 % |
| 80 | 97.0 % | 85.4 % |
| **96** | **98.6 %** | 91.4 % |
| 112 | 99.5 % | 95.4 % |
| 128 | 99.9 % | 98.0 % |

top-80 = 97.0 % 는 IDE_030 이 8-29 트레이스에서 기록한 값과 일치한다. 원본 `eval/results/20260915_135132_ide069_hotmap/`.

## 2. 손잡이별 효과

### 2.1 hot expert 배치 — 지배 항

| hot 수 | 배치 | C32 tok/s | HBM/장 | 근거 |
|---|---|---|---|---|
| 0 | — | ≈43 (C16) | 17 GiB | IDE_068 b1 |
| 96 | 물리 id 0~95 (빈도 무관) | 51.55 (C16) | 76 GiB | IDE_068 b6 |
| 64 | hotmap | 230.14 | 53 GiB | t6 |
| **96** | **hotmap** | **470 ~ 520** | 74.6 GiB | t1/t2 |
| 100 | hotmap, KV 16k | 356.45 | 76.9 GiB | t12 — KV 압박이 상쇄 |
| 112 | hotmap | **OOM** | — | t3 |

- 같은 96개라도 빈도 기반 배치가 물리 id 배치의 **9배** 다. hotmap 이 이 실험의 핵심이다.
- **TP4 의 HBM 상한은 hot96 이다.** TP4 샤드 기준 expert 1개/층 ≈ 11.8 MB → 62층 = 0.73 GB/expert/GPU. 112개 = 82 GB > 79.2 GB. KV 를 0 으로 해도 안 들어간다 (t3 OOM 으로 확인). hot100 은 들어가지만 KV 가 16k 로 줄어 C32 에서 오히려 손해.

### 2.2 deferral (`--kt-max-deferred-experts-per-token`)

| N | C32 tok/s | GSM 40 |
|---|---|---|
| 0 | 470.26 | 39/40 |
| 2 | 454.27 | — |
| **4** | **519.81** | **39/40** |
| 8 | 500.17 | — |

4 가 최적. 근사 연산이지만 GSM 40 에서 저하 0 — 채택.

### 2.3 KV pool 크기 — 동시성 상한을 정한다

| max-total-tokens (mf) | C32 | C48 | C64 | C80 | C96 |
|---|---|---|---|---|---|
| 24,576 (0.92) | 519.8 / 486.9 | — | 425.2 | — | — |
| **40,960 (0.95)** | 459.3 / 465.2 / 471.9 | 564.6 | **637.2 / 637.5 / 650.8** | 239.8 (붕괴) | — |
| 57,344 (0.96) | — | — | 634.1 | — | 328.7 (붕괴) |

- 24k 는 C32 까지, 40k 는 C64 까지 감당한다. 그 위에서는 KV 부족으로 TPOT 이 3~4배 뛴다.
- KV 를 키우면 C32 는 약 10 % 낮아진다 (mem-fraction 0.92 → 0.95 에서 graph/workspace 여유가 줄어드는 것으로 추정, 원인 미검증).
- 56k 는 64 에서 40k 와 같고 96 은 붕괴 — 40k 가 C64 운전의 적정점.

### 2.4 효과 없음 / 역효과

| 시도 | C32 tok/s | 판정 |
|---|---|---|
| cuda graph ON (expert 0·16, IDE_068 B3) | 43.5 / 46.1 | 효과 없음 — GPU 구간이 스텝의 소수라 launch overhead 가 안 보임 |
| cpuinfer 96 → 112 | 483.24 | 차이 없음 (jitter 범위) |
| EAGLE3 spec (480B 전용 draft) | 293.46 | **역효과** — draft 검증 토큰이 CPU expert 비용을 키움 |
| STANDALONE spec (Qwen3-4B) | 169.77 | **역효과**, TTFT 20 s |
| dispatch static | 엔진 거부 | a2a 백엔드 전제 (`ValueError`) |

### 2.5 jitter

같은 구성 반복: C32 ±1.5 % (459/465/472), C64 ±1.1 % (637/638/651), KV24k C32 ±3.5 % (520/487). 이보다 작은 차이는 주장하지 않는다.

## 3. 최종 구성

```
--tp 4 --attention-backend triton --trust-remote-code --context-length 32768
--kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2
--kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json
--kt-max-deferred-experts-per-token 4
--cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64
--mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic
```
C64 에서 **642 ± 7 tok/s**, TTFT p50 ≈ 3.0 s, TPOT p50 ≈ 77 ms, HBM 75.6 GiB/장, CPU busy 38 %. C32 운전이면 KV 24,576 / mf 0.92 가 520 tok/s 로 더 낫다.

## 4. dual (TP4+오프로딩 ×2) 대 GPU-only TP8 — TSK_053

(진행 중)

### 4.1 기준선 GPU-only TP8+EP8

| C | 출력 tok/s | 벤치 s |
|---|---|---|
| 16 | 701.60 | 11.7 |
| 32 | 1,029.75 | 15.9 |
| 64 | 2,031.11 | 16.1 |

## 5. 한계

- turbo OFF 하한. hot expert 는 FP8 원본에서 올라가며 INT4 GPU expert 는 시도하지 않았다.
- 워크로드 한 종(sonnet 512/128). hotmap 도 같은 워크로드 빈도이므로 다른 워크로드에서는 재측정이 필요하다.
- C32 의 KV 24k 대 40k 차이(≈10 %)의 원인은 확인하지 않았다.
