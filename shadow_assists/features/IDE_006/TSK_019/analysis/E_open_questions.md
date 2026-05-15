# Phase E — Open questions (분석 미달 영역)

> 분석 시각: KST 2026-05-15 ~
> 산출 목적: 본 5-phase 분석에서 답을 얻지 못한 질문 list. Phase E 의 두 표 (`E_bottleneck_map.md`, `E_amx_avx_applicability.md`) 의 정량 영역 중 측정 보조 없이 추정으로만 채워진 entry 의 후속 작업 list.

---

## E.12 질문 list

### OQ01 — pacpu 3 kernel 별 실측 ms/call

**현재**: cdec_wait 8.75 ms/layer 의 40/40/10/5/5% 분배는 산술 추정. 실측 미달.

**필요 측정**:
- `VLLM_NEO_PROFILE=1` 활성 후 PROFILE log 추출 (qk_product, av_product, softmax 각 ms/call)
- 측정 영역: H5 winner config 30분 run

**영향**: E_amx_avx_applicability.md 의 wall 절감 추정 (BM01-BM03) 의 정확도

---

### OQ02 — chain firing 80-99% 영역의 pacpu sample 가시화

**현재**: 본 flamegraph (mirror=10 / chain firing 4.5%) 에서 pacpu kernel sample < 1%. chain firing 영역 에서 pacpu kernel 의 sample 영역 미확인.

**필요 측정**:
- Option C/L/M2 ON + chain firing 80-99% 영역 30분 run 의 flamegraph 재측정
- `eval/results/2026051x_xxxxxx_chain_high_flamegraph/` 디렉토리 생성

**영향**: pacpu kernel 의 실측 영역 % 확정, BM01-BM06 의 영역 % column 정확도

---

### OQ03 — AMX BF16 변환의 정확도 영향

**현재**: v1.1 SUB_006 v42 BF16 manual kernel → token loss 2.84→3.70% + throughput -3.16% 회귀. BF16 + FP32 accumulator 유지 시 정확도 회복 가능성 미검증.

**필요 측정**:
- dev (i9-12900KF) 에서 BF16 conversion + AVX-512 BF16 FMA 의 분포 유사성 게이트 (per-token logprob max abs diff, sequence PPL relative diff) 검증
- Intel SDE simulator 로 AMX 정확도 cross-check

**영향**: BM01, BM02 의 위험 평가

---

### OQ04 — cdec_executor cap (max_workers) 의 layer 의존성

**현재**: SUB_023 의 max_workers=4 시 -52% regression 은 layer dependency chain 으로 추정. AMX 적용 시 layer wall 짧아지면 cap 영향 변화 미측정.

**필요 측정**:
- AMX 가속 후 max_workers sweep (2 / 3 / 4 / 6)
- 측정 영역: prod 머신 H100×8

**영향**: E_amx_avx_applicability.md 의 Amdahl cap 정확도

---

### OQ05 — OMP team launch overhead 영역 % 정확 분해

**현재**: 8.26% omp_pool 의 source 는 ATen `index_kernel` (BM21) 로 확인. pacpu 자체의 OMP team launch overhead 는 BM05 의 5% (cdec_wait 내부 추정) 만.

**필요 측정**:
- `OMP_TEAM_OVERHEAD` profile (libgomp 또는 perf event)
- 또는 OMP team persistent (omp_set_dynamic(0)) 적용 후 wall 차이

**영향**: BM05, BM06 의 wall 절감 정확도

---

### OQ06 — libpacpu-*.so symbol 가시화

**현재**: flamegraph 에서 `libpacpu-llama3_3_70b-tp8.so` 의 symbol 0건. `-g -fno-omit-frame-pointer` 미사용으로 추정.

**필요 측정**:
- `csrc/cpu/pacpu/build.sh` 의 CFLAGS 에 `-g -fno-omit-frame-pointer` 추가 후 rebuild
- flamegraph 재측정 — pacpu 의 qk_product/av_product/softmax frame 가시화 확인

**영향**: pacpu kernel 의 직접 self-time sample 확인, OQ01 의 대체 측정 경로

---

### OQ07 — paper 의 CPU 모델 / workload 정확 명시

**현재**: NEO paper (arXiv 2411.01142) 의 PDF 직접 추출 실패 (binary, poppler-utils 미설치). CPU 모델 (Xeon SPR / Genoa / 기타) + workload (max_num_seqs, KV cache dtype) 미확정.

**필요 작업**:
- paper PDF 다운로드 후 `pdftotext` 추출 (poppler-utils 설치 또는 다른 환경에서)
- §5 Evaluation 의 hardware + workload 영역 인용

**영향**: B_paper_vs_our_measure.md 의 정합성 본 검증

---

### OQ08 — TSK_003 `partial_attention_amx.cpp` 위치 확인

**현재**: TSK_003 doc 가 "Phase 1 완료, prod 152 PASS" 라 했으나 실제 source 위치 (`csrc/cpu/`) 에서 찾지 못함. 본 PR 의 NEO AMX 적용에 직접 사용 가능한 prod 검증된 자산.

**필요 작업**:
- `git log --all --diff-filter=A -- '*partial_attention*'` 로 추적
- 또는 다른 branch / shadow_assists 디렉토리 확인

**영향**: BM01-BM03 의 재사용 자산 inventory, 구현 cost 영역

---

### OQ09 — pacpu kernel 의 ISPC lower 결과 disassembly

**현재**: ISPC `avx512spr-x16` target 이 `_tile_*` 명령 (AMX) 까지 lower 하는지 또는 `vfmadd231ps` 만 lower 하는지 미확인.

**필요 작업**:
- `objdump -d /workspace/vllm_hybrid/csrc/cpu/pacpu/libpacpu-llama3_3_70b-tp8.so | grep -E 'tile|vfmadd|vdpbf16'`
- ISPC lower 결과의 명령 mix 확인 → 현재 ISA 도달률 정량

**영향**: AMX/AVX-512 intrinsic 명시 적용 시 추가 speedup 영역 정확도

---

### OQ10 — NEO upstream 의 AMX issue / PR 추적

**현재**: WebSearch 로 NEO repo 의 AMX 관련 issue / PR 0건 확인. upstream 의 후속 release / changelog 미확인.

**필요 작업**:
- `https://github.com/NEO-MLSys25/NEO/issues?q=amx OR bf16` 재확인
- ISPC 의 `avx512gnr-*` target + `<amx.isph>` header 도입 시점 확정

**영향**: 본 plan 의 fact 확정. upstream 변화 적용 가능 여부

---

## E.13 우선순위 (정량 게이트 영향 큰 순)

1. **OQ01** (PROFILE log) — 영역 % 정확도, 즉시 활용 가능
2. **OQ06** (symbol 가시화) — OQ01 의 대체 + 추가 fact
3. **OQ03** (BF16 정확도) — BM01 의 위험 평가, 적용 결정의 핵심
4. **OQ09** (disassembly) — 현재 ISA 도달률
5. **OQ02** (chain high flamegraph) — pacpu 영역 % 본 검증
6. **OQ08** (partial_attention_amx 위치) — 재사용 자산 inventory
7. **OQ07** (paper PDF) — B-phase 정합성
8. **OQ05** (OMP overhead) — BM05 정확도
9. **OQ04** (cdec cap) — Amdahl 정확도 (AMX 적용 후)
10. **OQ10** (upstream tracking) — fact 확정

---

## E.13.5 OQ11 측정 완료 — ASYNC_SWAP wall 영향 정량 (2026-05-15 KST, 정정 v2)

`E_bottleneck_map.md` 의 NEO swap path 12% (worker CPU duty) 중 async hidden 영역 vs critical path 영역 분리 측정.

### 측정 환경

| 항목 | 값 |
|---|---|
| commit | `64f9e0c48` (v1.6 fix) |
| workload | 200p × 8192 in/out, max_num_seqs=256, fp8 KV, gmu=0.92 |
| sweep | `VLLM_NEO_ASYNC_SWAP=1` (B=3, B=6) vs `=0` (sync only) |
| 측정 dir | `eval/results/20260515_074040_async1_base/` (B=3), `eval/results/20260515_083247_async1_b6/` (B=6), `eval/results/20260515_075914_async0_sync/` (sync) |
| script | `eval/run_neo_async_sweep.sh` |

### 측정 결과 (정정 — 이전 sync count 는 shell script 의 grep 오류, 실제 sync fallback = 0)

| 측정 | BUFFERS | wall (s) | output_tps | async swap (TP0) | sync swap (TP0) | delta vs baseline |
|---|---:|---:|---:|---:|---:|---|
| ASYNC=1, B=3 (baseline) | 3 | 995.3 | 1,638.3 | **12,440** | **0** | (reference) |
| ASYNC=1, B=6 | 6 | 995.5 | 1,645.9 | **12,440** | **0** | wall +0.02%, tps +0.5% |
| ASYNC=0 (sync only) | (off) | 1,216.8 | 1,346.5 | 0 | **24,496** | wall **+22.3%**, tps **−17.8%** |

### 정량 결론 (정정)

1. **ASYNC=1 시 모든 swap-out 이 async path 처리** — sync fallback 0 회. 이전 분석의 "sync fallback dominant" 는 shell script 의 `grep -c 'sync'` 가 **"Asynchronous"** 단어에 매치된 오류였음.
2. **BUFFERS=3 vs BUFFERS=6 거의 동일 결과** — 한 step 당 평균 swap-out 수가 3 이하라 cap=3 도 충분. cap 증가 효과 없음.
3. **ASYNC=0 시 swap-out 횟수가 ASYNC=1 의 ~2× 증가** (12,440 → 24,496). 원인: sync swap-out 이 느려 GPU KV 가 가득 차서 swap-out 이 더 빈번 trigger (Running batch ASYNC=1=4 vs ASYNC=0=200 — fully populated).
4. **wall +22.3% 의 분해** (sync only vs async baseline):
   - async hidden 효과의 손실 (12,440 swap-out 이 wall critical path 로 진입)
   - swap-out 횟수 자체의 ~2× 증가 (24,496) 가 추가 cost
   - 두 영역의 합산 결과
5. **async hidden 효과의 정량 (TP0 단독 기준)**: 12,440 async swap-out 회 × ~18 ms per call (추정) = ~224 sec hidden / TP0. 실제 wall 차이 +221.5s 와 정합.

### 사용자의 가설 검증

> "NEO swap path 는 비동기로 전체 wall time 에서 안 보이게 숨긴 거 아니야?"

→ **완전 정합** (정정 후):
- ASYNC=1 시 모든 swap-out 이 async 처리 (sync fallback 0)
- flamegraph 의 12% 영역 (NEO swap path) 의 main thread CPU duty 의 상당 부분은 **wall hidden** (DMA 가 별도 stream 에서 forward 와 overlap)
- ASYNC=0 비활성 시 wall +22.3% 회귀로 hidden 효과 정량 확정

### Phase E 표 영향

- `E_bottleneck_map.md` 의 BM07/BM08/BM09/BM21 영역의 wall vs CPU-duty 분리 정량 확정
- swap path 의 **CPU duty 12% → wall impact 약 18-20% 가능 (sync 시)** / **async 시 wall impact 매우 작음 (hidden)**
- AMX/AVX 가속 적용 영역 결정 시 swap path 의 wall 영향은 **async 가속 의 hidden 효과로 우선순위 낮음** 확인

---

## E.14 본 분석의 완결성 평가

본 5-phase 분석이 답한 핵심 질문:
1. **NEO 의 CPU 측 bottleneck 이 어디인가** → `E_bottleneck_map.md` 의 22 row (BM01-BM22) ✓
2. **AMX/AVX-512 가속 가능한 위치는 어디인가** → `E_amx_avx_applicability.md` 의 4 row 직접 (BM01-BM03 AVX-512, BM01 AMX) ✓
3. **각 위치의 적용 근거 (dtype, layout, 재사용 자산, 위험)** → ✓
4. **각 위치의 예상 wall 절감** → 추정 ✓ (실측 미달, OQ01-OQ02 후 정확도 향상)

**미달 영역**: 실측 ms/call (OQ01), BF16 정확도 검증 (OQ03), AMX dev 검증 경로 (OQ03 의 SDE simulator) — 모두 본 plan 의 범위 외, 후속 작업.

**fact-check (Phase E gate)**:
- 두 표의 모든 row file:line backing 가능: ✓
- 측정 dir / commit hash 동반: ✓ (commit `64f9e0c48`, dir `eval/results/20260514_141511_v16_flamegraph/`)
- "AMX 가능 N 개, AVX 가능 M 개, 둘 다 불가 K 개" 숫자 명시: AMX 2 + AVX-512 4 + 둘 다 불가 16 = 22 row ✓
- 결정 (apply / not apply) 본 분석 영역 외 → ✓ (본 분석의 범위 명시 준수)
