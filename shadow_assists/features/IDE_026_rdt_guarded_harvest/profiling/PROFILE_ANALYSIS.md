# 워커 프로파일 분석 — pad 전후 host 병목 (2026-06-13)

> py-spy `--native` 120s, Llama-3.1-70B TP=8 의 EngineCore 자식 워커(TP0), conc=32 실 trace.
> 1차 = suffix K=32 (`worker0_profile.speedscope.json`), 2차 = K6+pad (`worker0_pad_profile.speedscope.json`).
> 누적 %는 스택-포함 (호출 깊이) 기준.

## 카테고리별 누적시간

| 카테고리 | 1차 suffix K=32 | 2차 K6+pad | 변화 |
|---|---:|---:|---|
| **torch.ops 디스패치** (pybind11→ops→createStackForSchema) | **83.3%** | **28.8%** | **−54.5%p** |
| attention (FlashInfer/trtllm/tvm_ffi) | 55.3% | 13.3% | −42.0%p |
| **suffix (arctic draft 생성)** | 6.6% | **32.3%** | **+25.7%p** |
| sampler | ~0 | 7.3% | 표면화 |
| 런타임 (malloc/lock) | 9.9% | 10.6% | ≈ |
| CUDA 런타임 | 12.6% | 9.3% | ≈ |

## 판정

1. **SUB_213 pad 효과 = 프로파일로 증명**: torch.ops 디스패치 체인 83→29%, FlashInfer
   attention 55→13% (둘 다 FULL cudagraph 안으로 흡수). +38% tps 의 물리적 정체 확인.
2. **pad 이후 새 1위 병목 = suffix draft 생성 (6.6→32.3%, 5×)**: arctic suffix C
   확장의 tree 탐색 (`suffix_decoding.py:411`, `_C.cpython…suffix` 다수 프레임).
   디스패치가 사라지자 상대적으로 드러남.
3. **다음 serving 후보 우선순위 (데이터 기반)**:
   - 🥇 **SUB_233 (D20)** suffix tree walk SW prefetch — 새 1위(32%) 직격. arctic
     의존 load 체인에 prefetch 파이프라이닝 → draft 생성 가속 = 직접 tps. **기대 최대.**
   - 🥈 SUB_225 (D12) GIL/런타임 — malloc/lock 10% + IPC 대기 잔존.
   - 🥉 SUB_231 (D18) false sharing — 잔여 디스패치/큐 경합.
   - sampler 7.3% (SUB_161 의 "sampler 44%" 가 본 빌드/조건에선 축소되었으나 잔존).

## 산출물
`worker0_profile.speedscope.json` (1차) / `worker0_pad_profile.speedscope.json` (2차)
— speedscope.app 에서 flame graph 로 열람 가능.
