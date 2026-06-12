# SUB_232 [D19] — MEASUREMENTS (확정판, 2026-06-12)

> **판정 요약 (negative + 원인 확정)**: LD_PRELOAD memcpy 디스패처는 **무효**
> (M1 0.972 / M2 0.961 — 게이트 +3% 미달, noise 내 음수). **원인 실측**: steady
> state 의 libc memcpy 는 13.6M 호출 / 평균 **34 B** (Python 객체 churn) 이고
> NT-급 (≥256 KB) 은 전 런에서 **25회** 뿐 — SUB_201 의 "host path 80%
> memcpy-bound" 는 **torch 내부 copy 커널** (libc 미경유) 을 가리킴.
> **교훈: 개입 지점은 libc 가 아니라 vLLM/torch 코드 내부여야 한다.**

## 1. 측정

70B suffix canonical, 3셀 × 7 corpus, 셀별 fresh boot, aggressor 없음 (순수 serving A/B).
`libtunedmemcpy.so`: <θ_NT 는 `rep movsb`(FSRM), ≥θ_NT 는 AVX-512 NT-store.

| corpus | M0 glibc | M1 θ=8MB | M2 θ=256KB | M1/M0 | M2/M0 |
|---|---:|---:|---:|---:|---:|
| sharegpt | 4,701 | 4,526 | 4,622 | 0.963 | 0.983 |
| swebench | 5,413 | 5,203 | 5,562 | 0.961 | 1.028 |
| humaneval | 4,381 | 4,793 | 4,399 | 1.094 | 1.004 |
| mbpp | 2,903 | 2,693 | 2,665 | 0.928 | 0.918 |
| wildchat | 4,953 | 5,009 | 5,040 | 1.011 | 1.018 |
| lmsys | 4,222 | 3,873 | 3,835 | 0.917 | 0.908 |
| mix | 7,509 | 7,040 | 6,609 | 0.938 | 0.880 |
| **기하평균** | | | | **0.972** | **0.961** |

## 2. 메커니즘 증거 (0.5B offline + TUNED_MEMCPY_STATS)

```
small(rep movsb): n=13,621,164  bytes=469,074,817   # 평균 34 B/호출
nt(≥256KB)      : n=25          bytes=43,503,884    # 사실상 부팅/초기화 분
```
- libc memcpy 트래픽은 합계 469 MB (수 분 런) — host path 의 GB/s 급 데이터 이동은
  libc 를 안 탄다 (torch contiguous copy·cudaMemcpyAsync staging·detok 의 PyUnicode
  내부 등). LD_PRELOAD 의 도달 범위 밖.
- M1/M2 의 −3~−4% 는 noise 하단 (일중 변동 ±4~6%) — 유해하지도 않지만 무익.

## 3. 후속 (serving-성능 트랙 재조준)

1. **개입 지점 재탐색이 선행** — py-spy 프로파일 (SUB_161 방법 재사용) 로 70B suffix
   host 시간의 실제 분포 (torch copy? detok? sampler? tree walk?) 를 본 호스트에서
   재실측 → 다음 SUB 선택을 데이터로.
2. 후보: SUB_233 (suffix tree walk — arctic_inference C++ 포크 필요), torch copy
   호출부 직접 개입 (신규 SUB 후보), SUB_225 (GIL/프로세스).
3. 디스패처 코드 (.c) 는 보존 — D14 partition-aware 커널·RELAY-Q 대형 경로에 재사용.
