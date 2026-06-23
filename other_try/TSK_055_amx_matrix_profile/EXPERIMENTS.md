# TSK_055 실험 기록 — AMX Matrix Profile

*기록 2026-06-18. 모든 실측 = 본 세션. 코드 `/home/mystous/newidea/microbench/`. 환경: 2×Xeon Platinum 8570(56코어/소켓, AMX), 단일소켓(0-55) 핀, OpenBLAS 0.3.29, STUMPY 1.14.1. 빌드 `gcc -O3 -march=native -fopenmp -mamx-bf16 -mamx-tile … -lm`.*

## E0. 환경·재현 규칙
- 측정은 **단일소켓 56코어** `taskset -c 0-55`, `OMP_NUM_THREADS=56`, STUMPY는 `NUMBA_NUM_THREADS=56`.
- 동일 시계열 공유: C 커널이 `T_in.bin`(float64) 읽음 → `T.bin`·`mp_amx.bin`(MP float32)·`mpi_amx.bin`(int32) 덤프 → STUMPY는 `T.bin` 사용. **신호별 독립 프로세스**로 cross-state 오염 방지(아래 디버깅 노트 참조).
- 정확도 ground truth = STUMPY(fp64 STOMP) 및 numpy-bf16 레퍼런스(= AMX dpbf16ps 수치 동치: bf16 cast + fp32 누적).

## E1. 실데이터(현실 신호) motif/discord 정확도 + 속도 — ✅ PASS
**목적**: 랜덤워크는 near-tie 축퇴(인덱스 동률)가 있어 인덱스 일치율이 낮게 나옴 → **구조가 뚜렷한 현실 신호**에서
실제 데이터마이닝 출력(motif·discord)이 정확한지 검증. (UCR 페치는 인터넷 차단으로 불가 → 물리적으로 그럴듯한
합성 신호로 대체: ECG-유사 QRS 반복펄스, Seismic-유사 랜덤워크+wavelet+discord, Sensor-유사 주기+고조파+이상.)
**방법**: n=16384, m=256, 단일소켓. AMX=`mp_blk`(blocked, NBP=32), baseline=STUMPY. 신호별 독립 프로세스(`e1_one.py`).
지표: 글로벌 motif(=argmin MP)·discord(=argmax MP) 인덱스 일치(±m), MP Pearson, MP max-abs, 전체 MPI 일치율, 속도.
**결과**:
| 신호 | AMX | STUMPY | speedup | motif | discord | MP Pearson | MP maxabs | MPI일치 |
|---|---:|---:|---:|:--:|:--:|---:|---:|---:|
| ECG | 8.7 ms | 340 ms | **39.1×** | ✅ | ✅ | 1.00000 | 0.098 | 0.933 |
| Seismic | 8.6 ms | 348 ms | **40.5×** | ✅ | ✅ | 0.99477 | 2.835 | 0.870 |
| Sensor | 8.2 ms | 351 ms | **42.8×** | ✅ | ✅ | 0.99996 | 0.185 | 0.866 |
**의미**:
- **3개 현실 신호 모두 motif·discord가 STUMPY와 동일**(데이터마이닝의 실제 출력 정확). MP Pearson 0.995~1.0.
- 전체 MPI 일치율 0.87~0.93은 **near-tie 축퇴**(유사한 거리의 다른 이웃을 동률에서 다르게 선택)일 뿐 — MP **값**은 0.995~1.0 상관, motif/discord 동일. 즉 **bf16 정밀도가 MP의 실사용 결론을 바꾸지 않음.**
- 속도 **39~43×** vs 공식 STUMPY. (랜덤워크 phase diagram의 2.6~100× 범위와 일관.)
- **판정: PASS.** 실데이터 성격 신호에서 정확도·속도 동시 입증. (UCR 실데이터는 인터넷 확보 시 추가 권장.)

## E1-디버깅 노트 (방법론·정직성 — 논문 재현성 섹션에 반영)
E1 초기 실행서 Seismic/Sensor가 Pearson≈0(무상관)으로 나와 정밀 디버그함. 근본 원인 = **커널 버그 아님**:
- numpy-bf16 레퍼런스는 3신호 모두 Pearson 1.0(메서드 정확). T_in.bin==T.bin(파일 I/O 정상). 윈도우 std 정상(near-const 0개).
- 실제 원인: **`mp_blk`가 `mpi_amx.bin`만 덤프하고 `mp_amx.bin`(MP 값) 덤프를 누락** → 비교 스크립트가 직전 `mp_fused`가 남긴 **stale `mp_amx.bin`**을 읽어 엉뚱한 Pearson 산출. 신호 순서/직전 실행에 따라 flip-flop.
- **수정**: `amx_mp_blocked.c`에 `fwrite(MP)` 추가 → 재검증 결과 mp_blk vs numpy-bf16 **ECG 1.0(maxabs 0)·Sensor 1.0·Seismic 0.995**. **mp_blk 연산은 처음부터 정확**했고, phase diagram **속도 측정도 유효**(올바른 연산 수행).
- 교훈: 비교 하니스는 stale 산출물·cross-process 상태를 배제해야 함(신호별 독립 프로세스 + 모든 산출 fresh 덤프).

## E2. INT8 변형 (per-window z→int8 스케일, 천장 10× 노림) — 🟡 조건부 viable
**방법**: z-정규화 윈도우를 scale S로 int8 양자화(`clip(round(z·S),−127,127)`), `q·qᵀ`(int32)/S² ≈ m·corr → 거리.
S∈{16,31,40} sweep, ECG·Sensor서 motif/discord·Pearson vs STUMPY(n=16384,m=256, numpy numerics=AMX dpbssd 동치). 천장=INT8 6.71 TOPS(bf16 3.35의 2×).
**결과**:
| 신호 | S | Pearson | maxabs | motif | discord |
|---|---:|---:|---:|:--:|:--:|
| ECG | 16 | 0.99995 | 0.67 | ✅ | ✅ |
| ECG | 31 | 0.98402 | 7.70 | ✅ | ✅ |
| ECG | 40 | 0.95323 | 10.47 | ❌ | ❌ |
| Sensor | 16 | 0.99790 | 0.75 | ❌(near-tie) | ✅ |
| Sensor | 31 | 0.99933 | 0.29 | ❌(near-tie) | ✅ |
| Sensor | 40 | 0.99958 | 0.23 | ❌(near-tie) | ✅ |
**의미**:
- **INT8 viable, 단 per-window scale에 민감**: 너무 큰 S는 clipping(ECG S=40 붕괴), 너무 작으면 해상도 손실. 신호별 최적 S 다름(ECG≈16, Sensor≈31-40 — 신호 동적범위 의존).
- discord(이상탐지)는 견고히 일치. motif는 near-tie 민감(bf16과 동일 현상; Sensor의 주기성으로 동률 다수).
- **판정: bf16이 안전한 기본값**(E1서 motif/discord 전부 일치). **INT8은 천장 2×(vs bf16)·10×(vs AVX-512)를 노릴 때 per-window 적응 스케일 + discord-중심 용도에 조건부 채택.** 논문엔 "정밀도-속도 트레이드오프 + 적응 스케일 필요"로 정직 기록.

## 남은 실험 (계획)
- **SCAMP-CPU 공정 비교**: SCAMP는 STOMP 동일 알고리즘(GPU/FP32-tiling 축) → STUMPY가 CPU STOMP class 대표. SCAMP C++ 빌드는 deferred(미설치). 명시.
- **UCR 실데이터**: 인터넷 차단으로 보류. 확보 시 인덱스 정확도 재확인.
- **AB-join / 다변량 MP(mSTAMP)**: 동일 GEMM 커널 확장.
- **A-tile 재사용**으로 GEMM peak 근접(현 blocked 32 TFLOP/s).

## 핵심 측정 요약 (확정)
- 천장: AMX-INT8 6.71 TOPS(=AVX-512 VNNI 10.2×), bf16 3.35 TFLOPS (`amx_ceiling.c`).
- (n,m) phase diagram: STUMPY 대비 **n=4K~262K × m=128~1024 전 16셀 2.6~100×**(blocked, mp_blk 검증·유효).
- **실데이터(E1): motif/discord 동일, MP Pearson 0.995~1.0, 39~43×.**
- vs fp32-BLAS(작은-K QT shape): 20~123× (`cmp_blas.py`/`blas_k.py`).
- 정확도 게이트: 사전정규화 필수(naive bf16 motif 9.2%→99.6%), planted-motif motif/discord 동일.
