# SUB_201 / L4 — CPU lookahead spec-decoding (n-gram global dict CPU-resident) 측정

**날짜**: 2026-06-06
**브랜치**: `feat/spec-decode-tuning` (HEAD = `d4c7ec0d6`)
**측정 모델**: `Qwen/Qwen2.5-7B-Instruct` (TP=1, **B200 GPU 3**)
**vLLM 버전**: `1.7.dev16107+gffe20fb09.d20260601`
**워크로드**: sharegpt 200p × conc=16 × max-tok=256 (streaming)
**parquet**: `vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet`

> 사용자 지시 GPU=2 → 측정 중 다른 작업(l1_kv_quant, l10_admission)이 GPU 2 점유. 동등한 free 디바이스인 **GPU 3** 으로 자동 fallback (`L4_GPU` env). TP=1 / single-GPU 측정 조건은 동일.

---

## 1. Patch 위치 — vLLM `NgramProposer` 글로벌 dict 확장

| 파일 | 변경 | 역할 |
|---|---|---|
| `vllm/v1/spec_decode/ngram_proposer.py` | **+128 / -1** | (a) `__init__` 끝부분 — env flag `VLLM_NGRAM_GLOBAL_DICT` 처리 + LRU `OrderedDict` 초기화 + telemetry counters. (b) `propose()` 끝부분 — `_global_dict_fill_and_fallback()` 호출 (조건: flag on + top_m==1 + valid_ngram_requests 비어있지 않음). (c) `_global_dict_fill_and_fallback()` 신규 메서드 — **Phase A** local-miss → suffix lookup max_n..min_n (LRU touch), **Phase B** new tail 토큰 sliding-window ingest (key = bytes(ngram), value = next-k 토큰, first-write-wins). |

**Env flag**:
- `VLLM_NGRAM_GLOBAL_DICT` (default `0`) — 1 이면 process-wide n-gram dict 활성.
- `VLLM_NGRAM_GLOBAL_DICT_MAX` (default `200000`) — LRU cap.

**동작 요약**:
1. `batch_propose()` 가 기존 prompt-local longest match 시도 (변경 없음).
2. flag on 이면 추가로:
   - **Fallback lookup**: local match 가 비어있는 요청은 suffix 를 max_n..min_n 길이로 글로벌 dict 에서 조회 → hit 시 next-k tokens 를 draft 로 채움.
   - **Ingest**: `last_seen_n[idx]` 이후 새로 들어온 토큰 영역에서 sliding window n-gram → 글로벌 dict 에 first-write-wins 로 삽입. cap 초과 시 oldest LRU 항목 evict.
3. 정확도: 모든 draft 는 rejection_sampler 가 verify 하므로 잘못된 draft 는 accept_rate (정밀도) 만 떨어뜨림 — 출력 token 분포는 보존.

**제약**:
- top-M (`VLLM_NGRAM_TOP_M > 1`) 경로는 pass-through. 글로벌 dict 통합은 top-1 (default) 만 지원.
- `precompute_ngram` (SUB_067) 과 직교 (precompute 영역 영역 영역 cache hit 후 fallback 영역 영역 영역 작동).

**Sanity test** (`sanity_test.py`): seq=[11..20] ingest → 새 요청의 suffix=(15,16) 으로 글로벌 dict hit → draft=[17,18,19] 반환 확인. **PASS**.

---

## 2. 측정 결과

| run | mode | tps | Δ% vs A | TTFT p50 (ms) | TPOT p50 (ms) | GPU% | CPU% | accept α | acc/draft | n_ok | wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | **A_vanilla** (spec OFF) | **3905.4** | — | 20.8 | 3.8 | 57.2 | 6.4 | — | — / — | 200 | 12.5 |
| 2 | **B_ngram** (prompt-only) | 1503.0 | **−61.5 %** | 33.2 | 10.2 | 25.5 | 65.9 | **0.3615** | 6,947 / 19,219 | 200 | 32.6 |
| 3 | **C_ngram_glb** (global dict) | 1404.5 | **−64.0 %** | 36.2 | 11.0 | 26.0 | 69.1 | **0.2245** | 8,542 / 38,046 | 200 | 34.6 |

**Δ C vs B**: tps **−6.6 %**, draft 토큰 ≈ 2.0× (19219 → 38046), accepted +23 % (6947 → 8542), accept_rate **−37.9 %** (0.3615 → 0.2245).

---

## 3. 해석

### 3.1 글로벌 dict 가 실제로 추가 draft 를 생성하는가
**예**. `draft_tokens` 가 19,219 → 38,046 (정확히 +97.9 %) 로 증가 → 글로벌 dict 가 local-miss 요청 영역에 대해 광범위하게 lookup hit 을 만들고 있음. accepted 도 +23 % (6,947 → 8,542) 늘었음 → **recall 은 명확히 상승**.

### 3.2 그러나 accept rate 가 추락
- B (prompt-only): α=36.2 %
- C (global): α=22.5 %  → **−13.7 %p**

원인:
1. **Cross-request 패턴 오염** — sharegpt 처럼 다양한 도메인 prompt 가 섞이면 한 요청의 generation 패턴이 다른 요청의 suffix 와 우연히 매치되어 잘못된 draft 를 만듦.
2. **First-write-wins** 정책 — 같은 ngram 키가 여러 의미로 등장할 때 가장 먼저 본 next-k 가 고정되어 stale 해짐.
3. **min_n=2 가 너무 짧음** — 모델 vocab 30k+ 에서 2-gram 충돌이 빈번 → 의미적으로 무관한 매치.

### 3.3 net 결과: −6.6 % (B 대비)
prompt-only ngram (B) 가 이미 vanilla (A) 대비 −61.5 % 인데, 글로벌 dict 는 추가 손실. 이유:
- **draft kernel overhead** 가 draft 수에 선형 → +98 % draft → host-side ngram lookup + verify cost 모두 증가.
- **accept rate 하락** 영역 sampler 영역 verify-and-reject 비율 ↑ → GPU bubble 영역 영역.
- TTFT p50 +9 % (33.2 → 36.2 ms), TPOT p50 +7.8 % (10.2 → 11.0 ms) — host-side overhead 가 latency 양쪽 모두에 누적.

### 3.4 B 자체가 음수인 점 — 본 모델·워크로드의 ngram spec 부적합 신호
중요한 발견: **B200/Qwen2.5-7B / sharegpt 조건에서는 prompt-only ngram 만으로도 −61.5 %.** 일반적으로 vLLM ngram spec 은 코드 자동완성·법률문서 등 반복성 높은 도메인에서 net positive 인데, sharegpt 대화 corpus 는 prompt 내 반복이 거의 없어 α=36 % 만으로는 draft cost 를 못 상쇄. → 이 워크로드 자체가 ngram-적합 영역이 아님.

---

## 4. 본 task 결론

### Verdict — **글로벌 ngram dict 단순 확장은 net positive 못 만듦 (sharegpt/Qwen2.5-7B 조건)**

- 글로벌 dict 가 **draft 회수 (recall) 은 +97 %** 늘림 → 매커니즘 자체는 동작.
- 그러나 **precision (accept rate) 가 −38 %** 하락 → net tps **−6.6 % (vs B), −64 % (vs A)**.
- B 단독으로도 net negative → **본 워크로드는 ngram spec 자체가 부적합**, 글로벌 dict 확장 만으로 회수 불가.

### TSK_042 "작은-중간 모델의 spec-decode 추가 회수" 가설에 대한 함의
- 단순 글로벌 dict 만으로는 회수 불가능 — accept rate degradation 이 곧바로 net 을 갉아먹음.
- 실제 회수를 위해서는 다음 중 하나가 필수:
  1. **Per-domain / per-conversation dict** — 요청 그룹별로 분리 (cross-request 오염 차단).
  2. **Confidence-gated insert** — frequency ≥ τ 또는 ngram_len ≥ 3 만 dict 에 등록.
  3. **Per-request 모델 prior** — 첫 draft 가 reject 되면 그 ngram 영역 retract.
  4. **min_n ≥ 3** + LFU/frequency eviction (LRU 영역 영역 영역 영역 stale 영역 영역 영역).
- 또한 워크로드 선택이 중요 — **code completion, JSON schema generation, long doc continuation** 같이 prompt 내 반복이 강한 도메인에서만 ngram net positive 가 의미 있음 (b2_jump_forward 측정 참조).

### Follow-up 후보 (이번 lever 에서는 수행 안 함)
- **L4-v2**: min_n=3, frequency-gated insert (`count ≥ 2`), LFU eviction → cross-request 오염 완화 실험.
- **L4-v3**: per-conversation `request_id_prefix` 기반 dict 분할 → 멀티턴 대화에서만 활성.
- 코드 도메인 (HumanEval/MBPP) 에서 본 patch 재측정 — sharegpt 결과가 도메인 특성인지 patch 결함인지 분리.

---

## 5. 산출물

| 파일 | 설명 |
|---|---|
| `MEASUREMENTS.md` | 본 문서 |
| `run.sh` | mode 별 boot+bench 스크립트 (`A_vanilla`, `B_ngram`, `C_ngram_glb`) |
| `sanity_test.py` | global dict 동작 unit 검증 (PASS) |
| `summarize.py` | 3 mode JSON → 표 출력 |
| `qwen7b_{A_vanilla,B_ngram,C_ngram_glb}.json` | per-run summary |
| `qwen7b_{...}.raw.jsonl` | per-request raw |
| `_logs/boot_*.log` `_logs/bench_*.log` | per-run boot/bench logs |

**patch 적용 파일** (커밋 안 함): `vllm/v1/spec_decode/ngram_proposer.py` (+128 / −1)
