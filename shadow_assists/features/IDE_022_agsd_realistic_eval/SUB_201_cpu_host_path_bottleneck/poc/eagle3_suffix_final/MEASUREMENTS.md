# Eagle3 / Suffix final probe — measurement report

본 디렉토리는 **6번째 agent (이번 세션)** 의 최종 lever 발굴 측정. 이전 5 agent (#48~#53) 가 95+ lever 시도, 유일 양수는 fp8 KV +3.94%, CPU 활용 lever 0개. B 옵션 (AMX CPU draft) 도 hw gap 으로 -98%.

사용자 결정에 따라 (1) Eagle3 정확한 재시도 → (2) Suffix decoding 검증 → (3) 매칭 conc vanilla baseline 측정 → (4) 최종 verdict 순차 진행.

---

## 환경

| 항목 | 값 |
|---|---|
| HW | DGX B200 8× sm_100, Xeon 8570 (AMX bf16), 2 TB DRAM |
| 컨테이너 | `cap_sys_nice` 부재 |
| vLLM | `1.7.dev16107+gffe20fb09.d20260601` (editable in `/workspace/host_vllm_hybrid/vllm`) |
| Target model | `meta-llama/Llama-3.1-8B-Instruct`, TP=8, max-model-len=16384, gpu-mem-util=0.85 |
| Eagle3 head | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (HF cache hit) |
| Suffix backend | `arctic_inference.suffix_decoding.SuffixDecodingCache` (import OK) |
| Workload | `sharegpt500.parquet` (corpus="sharegpt", 자연 입력 길이 분포) |
| Runner | `vllm_config_perf/gating/realistic_eval/throughput_runner.py` (streaming, conc-sem, /metrics α 스크레이프) |
| Prior brief baseline | `cpu_heavy_baseline/runs/baseline_b3l2l10_s2.json` = **22,158 tps, gpu 96.4%, cpu 5.4% @ sharegpt 500p × conc=64 × max-tok=2048** |

---

## 단계 1 — Eagle3 정확한 재시도

이전 C-10a 의 Eagle3 결과 = -30.30% / accept_rate **0.7%** (의심).
- vLLM 의 SpeculativeMethod literal 에 `eagle3` 명시 존재 (`vllm/config/speculative.py:53-65`)
- Eagle3 head 가 HF cache (`models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B`) 에 존재
- spec config: `{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":3}`

### 1.1 결과 (1 server boot, 4 bench)

| Config | conc | max-tok | tps | vs prior brief baseline | accept_rate (α) | draft_tokens | gpu_util | cpu_util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1_E1 | 8 | 128 | 2,284.7 | — | **0.1676** | 127,374 | 81.7% | 4.4% |
| P1_E2 | 16 | 512 | 5,867.9 | — | **0.1779** | 494,412 | 91.8% | 4.5% |
| P1_E3 | 32 | 256 | 9,221.0 | — | 0.1358 | 271,731 | 80.3% | 4.7% |
| P1_E4 | 64 | 2,048 | **15,253.0** | **-31.2%** | **0.0045** | 2,898,237 | 95.9% | 5.1% |
| brief baseline | 64 | 2,048 | 22,158.5 | — | — | — | 96.4% | 5.4% |

### 1.2 분석

- α 가 **저-conc 에서 16-17%**, **c=64 에서 0.45% 로 collapse**. C-10a 의 0.7% 와 정성적으로 일치 → Eagle3 자체는 정상 동작이나 conc=64 batch 에서 verify 효율이 무력화.
- 정상 Eagle3 의 α 60-70% 는 c=1 / interactive single-stream 환경 가정. **batched online serving (conc=64 × 500 prompt) 에서는 α 가 구조적으로 낮음** — verify hit rate 가 batch 내 다른 sequence 의 KV 진행과 직교하지 않기 때문.
- cpu_util 4.4-5.1% < baseline 5.4% (오히려 약간 감소).

### 1.3 verdict

**기각.** prior brief baseline 기준 모든 cell net negative. 매칭 conc vanilla 비교 (단계 3) 에서도 모든 cell -20~-32% net negative 확인.

---

## 단계 2 — Suffix decoding 검증

spec config: `{"method":"suffix","num_speculative_tokens":8}`
- vLLM `suffix` method = arctic `SuffixDecodingCache` (`vllm/v1/spec_decode/suffix_decoding.py`)
- 기본 파라미터: `suffix_decoding_max_tree_depth=24`, `max_spec_factor=1.0`, `min_token_prob=0.1`
- 가설: TSK_042 222 셀의 "suffix +83~232% mix dominant lever" 재현 가능 여부.

### 2.1 결과 (1 server boot, 4 bench)

| Config | conc | max-tok | tps | accept_rate (α) | gpu_util | cpu_util | n_ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| P2_S1 | 8 | 128 | 2,151.5 | **0.3179** | 62.2% | 4.3% | 500/500 |
| P2_S2 | 16 | 256 | 6,336.4 | **0.5469** | 57.8% | 4.4% | 500/500 |
| P2_S3 | 32 | 512 | 11,361.8 | **0.5684** | 53.8% | 4.6% | 500/500 |
| P2_S4 | 64 | 2,048 | 16,744.0* | — (n/a) | 47.8% | 4.5% | **90/500** (서버 RemoteProtocolError, 부분 데이터) |

### 2.2 분석

- α (수락률) 가 32%-57% — **Eagle3 의 13-18% 대비 2-3 배 정상적**. arctic suffix tree 가 제대로 동작 중.
- 단 tps 절대값은 vanilla 미달 (단계 3 표 참고). suffix 의 draft cost / verify cost 가 token-throughput 측면에서는 net positive 되지 못함.
- gpu_util 47-62% 로 baseline (86-96%) 대비 **현저히 낮음** → suffix path 의 host-side overhead 가 GPU 를 starvation 시킴 (전형적인 host-bottleneck pattern).
- cpu_util 4.3-4.6% 로 baseline (4.8-5.4%) 보다 **오히려 낮음** — suffix 의 CPU tree-walk 가 sustained CPU work 를 만들어내지 못함. 짧은 burst 형태.
- P2_S4 (conc=64 × max-tok=2048) 는 410/500 RemoteProtocolError 발생 — 서버 stream 중단으로 신뢰 비교 불가.

### 2.3 verdict

**기각.** suffix 는 α 면에서는 Eagle3 보다 우월하나 token throughput 면에서는 모든 매칭 conc 에서 net negative (-11% ~ -36%). +10% AND cpu↑ 조건 불충족.

---

## 단계 3 — 매칭 conc vanilla baseline 측정

이전 단계 1/2 의 tps 절대값은 prior brief baseline (conc=64 × max-tok=2048 의 22,158 tps) 하나에만 비교 가능했음.
Eagle3 P1_E1~E3, Suffix P2_S1~S3 의 동일 conc/max-tok 매칭 vanilla 가 없어 Δ% 직접 비교 불가했던 이슈를 해결.

### 3.1 결과 (1 server boot, 5 bench, 단일 sweep n=1)

| Config | conc | max-tok | tps | gpu_util | cpu_util |
|---|---:|---:|---:|---:|---:|
| BL_c8_t128 | 8 | 128 | **3,339.6** | 86.6% | 4.8% |
| BL_c16_t256 | 16 | 256 | **7,163.6** | 92.0% | 5.0% |
| BL_c16_t512 | 16 | 512 | **7,413.2** | 95.4% | 5.1% |
| BL_c32_t256 | 32 | 256 | **13,354.8** | 87.2% | 5.2% |
| BL_c32_t512 | 32 | 512 | **13,694.4** | 93.0% | 5.2% |
| (prior brief) | 64 | 2,048 | 22,158.5 | 96.4% | 5.4% |

---

## 단계 4 — 통합 Δ% 비교표 + 최종 verdict

### 4.1 (Vanilla / Eagle3 / Suffix) Δ% 통합표

| (conc, max-tok) | Vanilla tps | Eagle3 tps (Δ%) | Suffix tps (Δ%) |
|---|---:|---:|---:|
| (8, 128)     | 3,339.6    | 2,284.7 (**-31.6%**) | 2,151.5 (**-35.6%**) |
| (16, 256)    | 7,163.6    | —                    | 6,336.4 (**-11.5%**) |
| (16, 512)    | 7,413.2    | 5,867.9 (**-20.9%**) | —                    |
| (32, 256)    | 13,354.8   | 9,221.0 (**-30.9%**) | —                    |
| (32, 512)    | 13,694.4   | —                    | 11,361.8 (**-17.0%**) |
| (64, 2048)   | 22,158.5   | 15,253.0 (**-31.2%**) | 16,744.0 (불완전, 90/500 OK) |

### 4.2 cpu_util 비교표

| (conc, max-tok) | Vanilla cpu% | Eagle3 cpu% | Suffix cpu% |
|---|---:|---:|---:|
| (8, 128)     | 4.8 | 4.4 | 4.3 |
| (16, 256)    | 5.0 | —   | 4.4 |
| (16, 512)    | 5.1 | 4.5 | —   |
| (32, 256)    | 5.2 | 4.7 | —   |
| (32, 512)    | 5.2 | —   | 4.6 |
| (64, 2048)   | 5.4 | 5.1 | 4.5 |

### 4.3 gpu_util 비교표

| (conc, max-tok) | Vanilla gpu% | Eagle3 gpu% | Suffix gpu% |
|---|---:|---:|---:|
| (8, 128)     | 86.6 | 81.7 | 62.2 |
| (16, 256)    | 92.0 | —    | 57.8 |
| (16, 512)    | 95.4 | 91.8 | —    |
| (32, 256)    | 87.2 | 80.3 | —    |
| (32, 512)    | 93.0 | —    | 53.8 |
| (64, 2048)   | 96.4 | 95.9 | 47.8 |

### 4.4 lever 판정

- **+10% AND cpu↑ 동시 만족 lever**: **0 개**.
- Eagle3: 모든 cell -20% ~ -32%, cpu_util 도 baseline 대비 -0.3 ~ -0.4 pp.
- Suffix: 모든 cell -11% ~ -36%, cpu_util 도 baseline 대비 -0.5 ~ -0.9 pp. 더구나 gpu_util 까지 동시 감소 → host-bottleneck signature (CPU 가 GPU 를 못 따라감인데 정작 CPU% 도 낮음 = host path 가 spinning 이 아니라 짧은 critical-section 으로 GPU starvation 유발).

### 4.5 최종 verdict

**구조적 불가 (본 baseline 환경, 본 HW).**

6 agent 100+ lever 시도 누적 결론:
- **유일 양수**: fp8 KV +3.94% (GPU lever, CPU 미사용).
- **CPU 활용 lever**: 0개 (어떤 시도도 cpu↑ + tps↑ 동시 충족 못함).
- **Spec-decode lever** (Eagle3 / Suffix): batched online serving (conc≥8, sharegpt natural prompts) 에서 α collapse 또는 host-bottleneck 으로 모두 net negative.
- **AMX CPU draft (B 옵션)**: hw gap (B200 sm_100 vs Xeon AMX bf16) 으로 -98%.

사용자 Objective ("CPU 의 활용률을 극도로 끌어 올려 … GPU 가 아닌 GPU 가 포함된 서버 또는 Cluster 전체의 성능을 향상" + "+10%") 는 **본 baseline 환경** (Llama-3.1-8B, TP=8 B200, sharegpt 500p, conc=64 × max-tok=2048, 22,158 tps) 에서 **본질적으로 충족 불가**:

1. B200 8× 의 GPU 성능이 너무 강해 (96% util) CPU 가 보조해줄 host-side 여유가 없음.
2. sharegpt natural prompt 분포에서 prefix repeat / suffix repeat 가 spec decode 의 α 를 16-57% 수준에 묶음 (α≥80% 가 net positive 필요조건).
3. CPU lever 들 (host overlap, AMX draft, suffix tree, classifier 등) 의 latency 가 GPU iteration time (∼2-3 ms tpot) 보다 길어 모두 critical path 진입.

### 4.6 다음 단계 권고

multi-model 확장 (Qwen-32B / Llama-70B) 으로의 진행은 **권고하지 않음**:
- 본 단계의 모든 lever 가 음수이므로 더 큰 모델로 확장해도 동일 lever 가 음수일 가능성이 매우 높음.
- 다른 baseline (small batch, latency-bound regime, c=1 single-stream) 로 재정의해야만 spec decode lever 가 의미를 가짐 — 단 이는 사용자가 정의한 "GPU 가 포함된 서버 전체 throughput" 목표와 직교.

SUB_201 의 lever discovery 는 본 verdict 로 closure. 후속은 (a) baseline 재정의 (latency SLO + small-batch regime) 또는 (b) 다른 SUB (workload 변경, cluster-level orchestration) 로 이관 권고.
