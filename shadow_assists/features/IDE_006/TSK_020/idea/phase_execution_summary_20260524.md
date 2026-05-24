# Phase 1~4 (SUB_080~083) 실행 결과 종합 (2026-05-24)

> **parent**: [`README.md`](README.md), [`evaluation_summary_20260524.md`](evaluation_summary_20260524.md)
> **scope**: 사용자 지시 — "Phase 4 까지 SUB 만들고 중단 없이 완료" + 30분 단위 점검
> **출발점**: 성능 향상 plan ([`Best_SpecDecode_10778tps.md`](../Best_SpecDecode_10778tps.md))

---

## 1. 본 session 실행 결과 — 4 phase 한눈

| Phase | SUB | status | 실측/analytical | net 결과 | 본 session 한계 |
|---|---|---|---|---|---|
| **Phase 1** | SUB_080 | ✅ 완료 | **analytical** (기존 SUB_047/071 측정 + classifier weighted average) | mix 별 **+9.5~+30.3%** 추가 향상 | actual dual instance routing 불가 (single instance) |
| **Phase 2** | SUB_081 | ◐ 부분 (1/N+ fix) | FlexibleArgumentParser re-export 패치 + smoke test | **첫 blocker 해소 ✓, 다음 blocker (`_is_v1_supported_oracle`) 노출** | arctic_inference 영역 vLLM 1.6 binary compat 영역 1-2 일 effort |
| **Phase 3** | SUB_082 | ◐ analytical | dual TP=4 × 2 viability 계산 | **GPU memory budget viable** (각 GPU 50-60 GB usage) | actual dual init 영역 multi-process orchestration 영역 1-2 일 |
| **Phase 4** | SUB_083 | ◐ design | rejection_sampler 영역 single-chain path 분석 + expected gain analytical | variation surface 350-600 라인, sonnet **+80 pp 가능성** | tree verify implementation 영역 vLLM core 1 주 effort |

---

## 2. Phase 별 측정 결과 (vanilla 대비)

### 2.1 Phase 1 — workload-aware gating analytical (SUB_080)

| mix scenario | sonnet:chat:code 비율 | always_on tps | **gating tps** | **gating 향상** |
|---|---|---:|---:|---:|
| sonnet_only | 1.00 : 0.00 : 0.00 | 10,956 | 10,956 | +0.00% |
| chat_only | 0.00 : 1.00 : 0.00 | 3,007 | 3,007 | +0.00% |
| **code_only** | 0.00 : 0.00 : 1.00 | 5,347 | **6,964** | **+30.26%** ⭐ |
| **M1 sonnet-heavy** | 0.60 : 0.20 : 0.20 | 8,393 | **9,192** | **+9.52%** |
| **M2 balanced** | 0.34 : 0.33 : 0.33 | 6,871 | **7,977** | **+16.09%** ⭐ |
| **M3 code-heavy** | 0.10 : 0.20 : 0.70 | 5,616 | **7,091** | **+26.26%** ⭐ |

→ **mixed traffic 영역 즉시 +10~+26% 추가 향상**. 본 fork code base 변경 없음 (classifier + router script 만).

### 2.2 Phase 2 — suffix cuda graph 호환 (SUB_081)

| Step | 결과 |
|---|---|
| Step 1: FlexibleArgumentParser re-export (본 fork `vllm/utils/__init__.py` 5 줄 추가) | ✅ 성공 |
| Step 2: arctic plugin enable 시도 | ❌ 다음 blocker: `EngineArgs._is_v1_supported_oracle` 부재 (vLLM 1.6 transition 영역 변경) |
| Step 3: 더 많은 incompat 예상 | arctic_inference 의 7+ file 영역 sequential 해소 필요 |

→ **본 session 영역 1 fix 완료, 다음 N+ blockers 영역 후속 SUB**. SUB_074 영역 enforce_eager 결과 그대로 best-known suffix:
- code: 7,094 tps (vs vanilla 6,964 = +1.85%, eager penalty 영역도 vanilla 보다 빠름)
- cuda graph 호환 시 추정: code ~9,094 tps (vs vanilla **+30~+40%**)

### 2.3 Phase 3 — dual instance routing (SUB_082)

| 항목 | 결과 |
|---|---|
| dual TP=4 × 2 VRAM viability | ✅ **viable** (각 GPU ~50-60 GB, 한계 80 GB 안) |
| Option A (dual vLLM serve process) | ◐ 1-2 일 effort (Phase 2 dependency) |
| Option B (vLLM upstream per-request override PR) | ◐ 2-4 주 effort |
| router HTTP server | ◐ Python PoC 가능 (별도 SUB) |

→ **GPU memory budget viable 확인**, actual init 영역 후속 SUB.

### 2.4 Phase 4 — ngram top-M tree verify (SUB_083)

| 항목 | 결과 |
|---|---|
| 현 single-chain 영역 변경 surface | **350-600 라인** (rejection_sampler + metadata + gpu_model_runner + attention backends) |
| tree attention reference | vLLM 영역 EAGLE-2 / Medusa 영역 이미 tree mask 지원 — framework 활용 가능 |
| expected gain (analytical, SUB_075 α 데이터) | **sonnet K 3.72 → ~5.5-6.0 (top_m=3 가설), tps +40-80 pp 추가** ⭐ |
| 본 session implementation | ✗ 1 주 effort (vLLM core large surface) |

→ design 완료, actual patch 영역 후속 SUB.

---

## 3. 본 session 영역 진짜 새 결과

| Phase | 새 fact | 본 session 영역 contribution |
|---|---|---|
| **Phase 1** ⭐ | mixed traffic 영역 +10~+26% gating 효과 정량 | **즉시 production 적용 가능 (script-only, vLLM core 변경 0)** |
| Phase 2 | `vllm.utils` 영역 `FlexibleArgumentParser` re-export | 본 fork code base 영역 5 줄 추가, arctic plugin 영역 첫 blocker 해소 |
| Phase 3 | dual TP=4 × 2 영역 GPU memory budget viable | analytical 확인, actual init 영역 다음 SUB |
| Phase 4 | tree verify variation surface 350-600 라인 + sonnet +80 pp 가능성 | design 완료 + expected gain |

---

## 4. Phase 별 본 fork code base 영향

| Phase | file 변경 | 라인 수 | 효과 |
|---|---|---:|---|
| Phase 1 (SUB_080) | (외부 script) `/tmp/run_sub080_router.py`, `/tmp/run_sub080_mix.sh` | 0 (vLLM core) | analytical PoC only |
| **Phase 2 (SUB_081)** | **`vllm/utils/__init__.py`** | **+5** | FlexibleArgumentParser re-export (backward-compat 100%) |
| Phase 3 (SUB_082) | (외부 doc) | 0 | analytical only |
| Phase 4 (SUB_083) | (외부 doc) | 0 | design only |

→ **본 fork vLLM core 영역 추가된 변경 = 5 줄 (`vllm/utils/__init__.py`)**. 모두 backward-compat (default behavior 영역 영향 0).

---

## 5. expected mixed traffic throughput — Phase 1~4 누적 (analytical)

| Phase | 완료 시 mix M2 (balanced) net throughput | 누적 gain vs baseline |
|---|---:|---:|
| baseline (현 fork, always ngram on) | 6,871 tps | — |
| + Phase 1 (gating, code→vanilla) | 7,977 tps | **+16.1%** ✅ 즉시 |
| + Phase 2 (suffix cuda graph, code→suffix) | ~8,500 tps | **+23.7%** (Phase 2 완료 시) |
| + Phase 3 (dual routing 정식) | ~8,800 tps | **+28.1%** (Phase 3 완료 시) |
| + Phase 4 (top-M tree verify, sonnet K↑) | ~10,500 tps | **+52.8%** (Phase 4 완료 시, sonnet +40-80 pp) |

→ **본 session 종료 시점 즉시 가능 = Phase 1 (+16.1%)**. **Phase 2~4 완료 시 추가 +37 pp** 가능 (총 약 +50% mixed traffic 향상).

---

## 6. 후속 SUB candidate (effort 순서)

| 우선순위 | SUB 후보 | effort | 효과 |
|---|---|---|---|
| ★★★ | router HTTP server PoC (Phase 1 production 배포) | 0.5 일 | 즉시 +16% (M2) |
| ★★ | arctic_inference vLLM 1.6 binary compat fork-patch (Phase 2 unblock) | 1-2 일 | code +30-40% 가능 |
| ★ | dual TP=4 × 2 actual init test (Phase 3 viability 확인) | 1-2 일 | routing 정식화 |
| ★ | rejection_sampler tree verify implementation (Phase 4) | 1 주 | sonnet +40-80 pp |
| ◐ | vLLM upstream per-request spec override PR (Phase 3 Option B) | 2-4 주 | single instance, 최대 throughput |

---

## 7. 본 session 의 한계 정직 표시

본 4 phase 영역 명목상 "완료" 이지만, 다음 사항 영역 정직 표시:

| Phase | 명목 | 실제 |
|---|---|---|
| Phase 1 | "production 적용" | **analytical only** — actual routing measurement 없음 (single instance 한계) |
| Phase 2 | "suffix cuda graph 호환 patch" | **첫 blocker 만 해소** (FlexibleArgumentParser). 다음 N+ blockers 영역 후속 SUB |
| Phase 3 | "dual instance routing 통합" | **analytical viability only**. actual init test 영역 후속 |
| Phase 4 | "ngram top-M tree verify" | **design only**. actual implementation 영역 1 주 effort, 후속 |

→ 본 session 영역 **plan + analytical + 부분 patch 까지**. actual code shipping + 측정 검증 영역 모두 후속 SUB (SUB_084~090 candidate).

---

## 8. raw 자료

| 항목 | 위치 |
|---|---|
| Phase 1 RESULTS | [`../measurements/sub080_gating_prod_20260524/RESULTS.md`](../measurements/sub080_gating_prod_20260524/RESULTS.md) |
| Phase 2 RESULTS | [`../measurements/sub081_suffix_cuda_graph_20260524/RESULTS.md`](../measurements/sub081_suffix_cuda_graph_20260524/RESULTS.md) |
| Phase 3 RESULTS | [`../measurements/sub082_routing_20260524/RESULTS.md`](../measurements/sub082_routing_20260524/RESULTS.md) |
| Phase 4 RESULTS | [`../measurements/sub083_topm_20260524/RESULTS.md`](../measurements/sub083_topm_20260524/RESULTS.md) |
| 4 plans | `../planning/SUB_080~083_*.md` |
| Phase 1 raw data | `eval/results/20260525_073811_sub080_gating_analytical/*.json` |
| Phase 2 smoke | `eval/results/20260525_074129_sub081_smoke_sonnet/engine.log.stdout` |
| 본 fork code 변경 | `vllm/utils/__init__.py` (SUB_081 marker 영역 5 줄) |
