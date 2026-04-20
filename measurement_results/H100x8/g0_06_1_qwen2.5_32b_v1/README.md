# g0_06_1_qwen2.5_32b_v1 — §06-1 Phase 1 (weight reuse only, 중간 버전)

## 정체성

§06-1 "M-aware MLP Q8_0 Kernel" 의 **첫 구현본** 측정 결과. `quant_q8_0.cpp::q8_0_linear_impl` 에 M 분기 추가 + `q8_0_gemm_vnni_impl` (weight 를 M 개 row 에 걸쳐 재사용하는 loop) 신설. kernel 내부는 여전히 `_mm512_madd_epi16` + `_mm512_reduce_add_epi32` 기반 (VNNI intrinsic 미사용).

- 측정일: 2026-04-20
- Branch: `ninja-gap/06-1-m-aware-mlp-kernel`
- Git: `0c066f0e7` (신규 kernel) + `749e8f6b2` (문서) 기반
- env: `g0_h100x8_qwen32b_06.env` (§06 on, `HYBRID_VNNI_HOT_PATH=1`)
- 환경: H100x8 + Qwen2.5-32B + TP=8 + 500 req × 128/128

## 버전 의미

"v1" = §06-1 Phase 1 의 **첫 revision**. 추후 개선 revision 예정:
- **v1 (본 디렉토리)**: weight block load 를 inner m loop 로 공유 (현재)
- v2 (예정): software prefetch 추가
- v3 (예정): VNNI intrinsic (`_mm512_dpbusd_epi32`) 전환
- (조건부) v4: M 축 병렬 처리로 reduce 횟수 감소

각 revision 은 같은 g0_06_1 scope 안에서 측정 iteration 으로 누적. v1 은 "baseline for Phase 1 iteration" 역할.

## 측정 요약 (outTP 기준)

3-way 비교 — base (§06 off) / g0_06 (§06 on) / 본 측정 (§06 + §06-1 v1):

| seqs | base | §06 on | §06-1 v1 | vs §06 on | vs base |
|---:|---:|---:|---:|---:|---:|
| 1 | 908.9 | 1,069.7 | **1,196.3** | +11.8% | +31.6% |
| 2 | 895.9 | 654.6 | 794.0 | **+21.3%** | −11.4% |
| 4 | 595.3 | 370.0 | 496.2 | **+34.1%** | −16.7% |
| 8 | 575.2 | 211.2 | 272.3 | **+28.9%** | −52.7% |
| 16 | 637.8 | 118.2 | 122.1 | +3.2% | −80.9% |
| 32 | 423.1 | 63.7 | 61.4 | −3.6% | −85.5% |
| 64 | 339.7 | 32.2 | 32.0 | −0.8% | −90.6% |

## 판정 결과

**1차 (§06 on 대비 회복, scope seqs 2/4/8)**: ✅ 통과
- seqs 2: +21.3%, seqs 4: +34.1%, seqs 8: +28.9%
- §06-1 이 §06 의 batch 영역 역효과를 의도대로 완화

**2차 (baseline 대비 역효과 해소)**: 🔶 부분 성공
- seqs 2: −11.4% (거의 회복)
- seqs 4/8: 여전히 baseline 에 열세 (−17%/−53%)
- VNNI INT8 compute 한계 + kernel 최적화 여지가 AMX BF16 을 완전히 커버 못함
- 추가 이득은 v2/v3 의 개선 또는 §24 병합 재설계 영역

**Scope 밖 (seqs ≥ 16)**: ✅ 의도대로 동작
- §06 on 과 거의 동일 (Δ ±3%) — kernel 의 `if (M < 16)` 분기 정확히 작동

## 구조

```
g0_06_1_qwen2.5_32b_v1/
├── analysis_g0.ipynb            # g0_06 노트북 재사용, ROOT 자동 감지
├── analysis_bench.png
├── analysis_cpu_heatmap.png
├── analysis_gpu_power_mem.png
├── analysis_util_timeseries.png
├── g0_h100x8_qwen32b_06.env     # 측정 env snapshot
├── gpu_only_baseline/           # TP=8 gpu_only 대조군 (2026-04-20 새 측정)
├── seqs{1,2,4,8,16,32,64}/      # hybrid §06+§06-1 sweep
└── _initial_failed_retry/       # seqs=2 초기 실패분 (참고용)
```

## 관련 디렉토리

- **비교 대상 baseline**: `../g0_00_qwen2.5_32b_base/` (TP=8 §06 off)
- **비교 대상 §06 on**: `../g0_06_qwen2.5_32b/` (§06 on, kernel batch 결함 상태)
- **과거 snapshot**: `../g0_00_qwen2.5_32b_tp4/` (TP=4 시기)

## 관련 문서

- §06-1 기법 문서: `NinjaGap_Todo/06-1_m_aware_mlp_kernel.md`
- 판정 기준 원본: 같은 문서 "성공 조건" / "실패 시 디버깅 우선순위" 섹션
- 관련 branch: `ninja-gap/06-1-m-aware-mlp-kernel`
