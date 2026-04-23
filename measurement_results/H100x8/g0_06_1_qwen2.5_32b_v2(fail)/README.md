# g0_06_1_qwen2.5_32b_v2(fail) — §06-1 v2 측정 (regression 확인)

## 정체성 및 결과 요약

§06-1 의 2nd revision (VNNI `vpdpbusd` intrinsic + s8s8 compensation + prefetch). **scope 구간 (seqs 2/4/8) 에서 v1 대비 7–13% outTP 저하**가 측정돼 실패로 판정. **v1 으로 롤백 후 §06-1 최종 선택은 v1**.

본 디렉토리는 데이터 보존용. 이후 revision 에서 같은 함정을 피하기 위한 근거 자료로 유지.

- 측정일: 2026-04-20
- Branch 시점: `ninja-gap/06-1-m-aware-mlp-kernel` (commit `33361eadc` 기반)
- env: `g0_h100x8_qwen32b_06.env` (§06 on)
- 환경: H100x8 + Qwen2.5-32B + TP=8 + 500 req × 128/128

## 4-way outTP 비교

| seqs | base (§06 off) | §06 on (v0) | v1 | **v2** | v2 vs v1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 908.9 | 1,069.7 | **1,196.3** | 1,074.3 | **−10.2%** |
| 2 | 895.9 | 654.6 | **794.0** | 702.9 | **−11.5%** |
| 4 | 595.3 | 370.0 | **496.2** | 431.5 | **−13.0%** |
| 8 | 575.2 | 211.2 | **272.3** | 254.1 | **−6.7%** |
| 16 | 637.8 | 118.2 | 122.1 | 122.8 | +0.6% |
| 32 | 423.1 | 63.7 | 61.4 | 61.2 | −0.3% |
| 64 | 339.7 | 32.2 | 32.0 | 31.8 | −0.7% |

## 왜 실패했는가 — 추정 원인

### 1. Half-tile waste (가장 유력)

`_mm512_dpbusd_epi32` 는 16 lanes × 4 bytes = **64 INT8 병렬** instruction. Q8_0 block 은 32 INT8 뿐 → lane 8–15 zero-padded → peak throughput 의 **절반만 사용**. v2 commit 시점에 "v3 에서 two-block packing 으로 회수" 로 self-tag 했던 지점. 이 손실이 생각보다 컸다.

### 2. s8s8 compensation overhead

- Block 당 32 scalar add (comp 계산)
- M inner loop 당: `_mm256_add_epi8` (u8 shift), `_mm512_inserti32x8` × 2, `128 * comp` 곱셈
- v1 의 `cvtepi8_epi16 × 2` + `madd_epi16` 보다 총 μop 수 **증가**
- VNNI dot 의 이득이 compensation overhead 로 상쇄

### 3. 컴파일러 최적화 상쇄

GCC 12 가 v1 의 `madd + reduce` 를 pipeline 에 잘 맞춰 scheduling. VNNI 단일 instruction 이 이론적으로 우위여도 실제 throughput 은 비슷하거나 열세일 수 있음.

## seqs=1 (GEMV 경로) 에서도 v1 대비 −10% — 해석

M=1 는 v2 kernel 변경과 무관한 `q8_0_gemv_vnni_impl` 경로. 그럼에도 −10% 차이는:
- **측정 노이즈 (single-run variance)** 가능성 가장 큼
- 또는 전체 binary 의 optimization 차이 (pipeline/cache layout)

## 결론

v2 의 단순 VNNI 전환으로는 의미 있는 이득 없음. **v1 상태로 롤백 (옵션 A)**. 남은 선택지:
- **v3 (two-block packing)**: lane 완전 사용. 성공 시 v2 의 half-tile 문제 해소. 구현 복잡.
- **`vpdpbssd` (s8×s8 직접)**: compensation overhead 제거. AVX-VNNI-INT8 feature 및 gcc 13+ 필요.
- **Tier 1 후보 또는 §24 (W8A8)**: §06-1 scope 를 벗어나 다른 축에서 이득 확보.

현재 기준 결론은 **v1 유지 + Tier 1 후보 재평가**다. 이 README 작성 당시 남아 있던 `§11/§25` 전환 문구는 이제 stale 이며, `§11`은 2026-04-20 Phase 1 기각 상태다.

## 구조

```
g0_06_1_qwen2.5_32b_v2(fail)/
├── README.md                     # 본 파일
├── analysis_g0.ipynb             # nbconvert 통과, 에러 0
├── analysis_{bench,cpu_heatmap,gpu_power_mem,util_timeseries}.png
├── g0_h100x8_qwen32b_06.env      # env snapshot
├── gpu_only_baseline/            # TP=8 gpu_only (v2 측정 시점)
└── seqs{1,2,4,8,16,32,64}/       # hybrid §06+§06-1 v2 sweep
```

## 관련

- **v1 (확정)**: `../g0_06_1_qwen2.5_32b_v1/` — §06-1 최종 선택
- **§06 on (v0)**: `../g0_06_qwen2.5_32b/` — kernel 변경 전 §06
- **baseline**: `../g0_00_qwen2.5_32b_base/` — §06 off
- 기법 문서: `NinjaGap_Todo/06-1_m_aware_mlp_kernel.md`
- 이 실패 데이터가 유용한 시점: v3 (two-block packing) 또는 `vpdpbssd` 로 재시도할 때 lane waste + compensation overhead 의 ball-park 확인 자료
