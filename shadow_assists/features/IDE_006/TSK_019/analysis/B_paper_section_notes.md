# Phase B — NEO 논문 detailed reading notes

> 분석 시각: KST 2026-05-15 ~
> 자료 출처: arXiv 2411.01142 (MLSys 2025) — 직접 PDF 추출 불가 (binary), 기 분석 doc + WebSearch 인용
> 우선순위: 사전 분석 자료 (`NEO_code_deepdive.md`, `NEO_redesign.md`) 가 이미 paper section 발췌 보유 → 그 인용 + WebSearch 결과 종합

---

## B.1 paper metadata

| 항목 | 값 |
|---|---|
| 제목 | NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference |
| 저자 | Liu et al. |
| Venue | MLSys 2025 |
| arXiv | 2411.01142 (2024-11 첫 공개) |
| PDF | https://yangzhou1997.github.io/paper/neo_mlsys25.pdf |
| HTML | https://arxiv.org/html/2411.01142 |
| Code | https://github.com/NEO-MLSys25/NEO |
| License | Apache-2.0 |

---

## B.2 핵심 메커니즘 (인용)

`NEO_redesign.md` 가 paper 의 핵심 인용 보유:

> "The KV cache system of NEO is divided into two separate components: the 'GPU-cache' located in the GPU's HBM, and the 'CPU-cache' located in the CPU main memory. For any request that has already been prefilled in the system, its KV cache will either reside entirely in the GPU-cache—designated as a 'GPU-request'—or entirely in the CPU-cache."

> "Asymmetric pipelining runs two asymmetric sub-batches concurrently: one offloads the decoding attention computation and KV cache of a subset of requests into the CPU, and another one runs the rest in the GPU."

> "To achieve full GPU-CPU overlapping, Neo integrates the prefilling stage computation into the GPU decoding sub-batch, so that the prefilling stage computation (in GPU) also happens in parallel with the CPU attention computation."

→ **핵심**: request 단위 KV exclusive ownership (mirror 가 아니라 exclusive), 두 sub-batch asymmetric pipeline, prefill 도 CPU attention 과 overlap.

---

## B.3 정량 성능 결과 (WebSearch 인용)

> "NEO achieves 36%, 26%, and 14% higher throughput compared to GPU-only approach on T4, A10G, and H100 GPUs, respectively, while maintaining the same latency; with more powerful CPUs, Neo achieves up to 79.3% throughput gain on A10G GPU."

| GPU | NEO gain (paper claim) |
|---|---:|
| T4 (16 GB) | **+36%** |
| A10G (24 GB) | +26% (strong CPU: **+79.3%**) |
| **H100 (80 GB)** | **+14%** ← prod target 이지만 가장 낮음 |

→ NEO 의 이론 상한 (H100) 이 우리 측정 wall 의 sweet spot. H100×8 환경에서 14% 이상 기대는 무리.

---

## B.4 CPU kernel 구현 (sec 7 — NEO_code_deepdive.md §7 발췌)

`NEO_code_deepdive.md:657-700` 가 NEO paper + repo 의 CPU kernel 영역 정리:

| 항목 | NEO `pacpu` |
|---|---|
| SIMD 폭 | **AVX2 (256-bit)** (논문 시점 ISPC 기본) |
| Multi-thread | OpenMP team (core.h 의 omp parallel + 2 barrier) |
| dtype | **FP16 (storage) + FP32 (accumulator)** (`dtype.h`) |
| AMX 지원 | **없음** |
| GQA | inline broadcast (loop 안 QH_PER_KVH 반복) |
| NUMA | 미적용 |

> `NEO_code_deepdive.md:683` 인용:
> "NEO 는 ISPC 의 SIMD intrinsic 으로 *AVX2 가속* 만, IDE_006 은 AVX-512 + AMX 까지 가속. prod target (Xeon SPR + AMX) 에서 IDE_006 의 AVX-512/AMX kernel 이 우위."

다만 우리 cherry-pick 의 `CMakeLists.txt` 가 `avx512spr-x16` active 로 변경되어 있음 (`A_neo_upstream_audit.md` §A.7 참조) — NEO 원본 commented out 이지만 우리는 AVX-512 활성.

---

## B.5 CPU 평가 환경 (paper) — 미상

PDF 직접 추출 실패로 본 plan 의 verification gate 의 "CPU 모델 명시" 부분 미충족. 후속 측정 필요:

- WebSearch 결과 에 paper의 CPU 모델 (Xeon SPR vs Ice Lake vs custom) 명확치 않음
- A10G "strong CPU" 시 79.3% 이득 — strong CPU 가 SPR 또는 Genoa 추정
- H100 14% 이득 측정 시 CPU 모델 미상 — 그러나 paper 일반적 패턴은 GPU 와 같은 host CPU 사용 (PCIe 짝)

후속 작업: paper PDF 직접 다운로드 후 `pdftotext` (poppler-utils 설치 필요) 로 §5 Evaluation 의 hardware 영역 인용 확보.

---

## B.6 ISPC 선택 이유 / AMX 미사용 이유 — paper 명시 여부

기 분석 doc 에서 paper 의 ISPC 선택 이유 / AMX 미사용 이유 직접 quote 찾지 못함. **추정 근거**:

1. **ISPC 선택**: paper 시점 (2024-11) ISPC 가 SPR `avx512spr-x16` target 보유 (ISPC 1.19+, 2023). portable + auto-vectorize. C++ intrinsic 보다 코드 작고 PMD model 명확.

2. **AMX 미사용**: paper 시점에 ISPC 가 AMX backend 미보유. ISPC 1.30 (2025) 즈음 `avx512gnr-*` (Granite Rapids) target + `<amx.isph>` header 가 도입됨 (WebSearch 결과). NEO upstream 은 그 후 update 없음 (release 0, AMX issue 0).

3. **dtype 선택**: storage FP16 + compute FP32 — BF16 native 가 SPR AMX 의 sweet spot 인데 NEO 는 FP16 고수. paper 의 정확도 검증 가능성 (BF16 native conversion 의 precision drop 영역) 추정. v1.1 측정 (SUB_006 v42, BF16 manual kernel) 이 **token loss 2.84→3.70%, throughput -3.16%** 회귀로 reject 된 이유와 정합 (`Performance_analaysis_v1.5.md` 의 시도 history).

---

## B.7 paper Future Work / Limitations — 미확인

PDF 추출 실패로 future work 섹션 직접 인용 불가. 추정 영역:
- AMX integration (ISPC 가 지원하기 시작한 후의 가능성)
- Multi-NUMA + multi-CPU host 영역
- Granite Rapids / 이후 SPR 후속 CPU 영역

후속 plan E 의 followup_questions 에 "paper future work 정확 인용" 항목 등재.

---

## B.8 결론 — Phase B fact summary

- NEO paper 의 CPU kernel = **ISPC + AVX2/AVX-512 자동 lower, AMX 부재**
- dtype = **FP16 storage + FP32 accumulator**
- 정량 우위: T4 36% / A10G 26% (strong CPU 79.3%) / **H100 14%**
- 본 plan 의 prod target (H100×8) = NEO 효과의 **최소 영역 (14% 이론 상한)**
- AMX 미사용 = paper 시점 ISPC 미지원 + dtype FP16 고수 (BF16 변환 precision drop 우려)

→ Phase D 에서 우리 측정 (cdec_wait 8.75 ms/layer) 과 14% 이론 상한의 정합성 검토.
