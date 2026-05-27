# IDE_012 — workload-aware predictive gating PoC

> **parent backlog**: [`README.md`](README.md) (TSK_020 / SUB_072)
> **자식 SUB**: [`SUB_076`](../planning/SUB_076_workload_aware_gating_classifier.md)
> **발견**: 2026-05-24, analysis doc §6 workload-aware gating heuristic
> **priority**: ★ (large effort, 본 분석의 production translation)
> **status**: ✅ **완료 (classifier first-pass)** (2026-05-24) — macro accuracy 1.000 / routing 통합은 후속

## 1. fact

본 분석 doc §6 의 권장사항: **prompt content 기반 predictive gating** — spec 시도 전에 prompt feature 만으로 spec ON/OFF 결정. SGLang `--speculative-adaptive` (R11, feedback 기반) / vLLM RFC #4565 (R12, closed as not planned) / Cascade (R3, MoE 한정) 와 다른 새 pattern.

production path = vLLM Semantic Router (R17, arXiv 2603.21354) 의 Workload-Router-Pool architecture 로 두 instance (spec ON / OFF) 분리 라우팅 — vLLM core 변경 없이 prod 가능.

## 2. PoC 설계

### 2.1 prompt classifier (Stage 1)

prompt 입력 시 feature 추출:

| feature | extractor | threshold (1차 PoC) | weight |
|---|---|---|---|
| ` def ` / ` class ` count | regex | ≥ 1 | code 지표 |
| triple-backtick \`\`\` count | regex | ≥ 2 | code 지표 |
| `import ` / `from ` count | regex | ≥ 2 | code 지표 |
| chat template tag (`<\|system\|>` / `<\|user\|>`) | regex | ≥ 1 | chat 지표 |
| top-20 단어 빈도 비율 | tokenize + count | ≥ 0.30 | sonnet-like (어휘 한정) |
| unique token / total token ratio | tokenize | ≤ 0.40 | low-repeat → spec 효과 ↑ |

rule 1차: code 지표 (3 개 중 ≥ 2 hit) → spec OFF. 그 외 → spec ON.

### 2.2 routing (Stage 2)

옵션 A — **두 instance**: vLLM 영역 별도 LLM instance 2 개 (`llm_spec_on`, `llm_spec_off`), classifier 가 router 역할.
- 장점: vLLM core 변경 없음, 즉시 PoC 가능
- 단점: GPU 메모리 2배

옵션 B — **vLLM per-request override** (현 v1 미지원, vllm-project/vllm 신규 PR 필요): SamplingParams 또는 LLM.generate kwargs 영역 per-request spec config 추가.
- 장점: 단일 instance, 메모리 효율
- 단점: vLLM core 변경 필요, upstream review 통과 risk

→ **PoC 는 옵션 A 로 시작**, validation 후 옵션 B 를 upstream PR 화.

### 2.3 측정 시나리오

production traffic mix 모사 (3-workload 비율):
- M1 (sonnet-heavy): 70% sonnet + 20% chat + 10% code
- M2 (balanced): 33% × 3
- M3 (code-heavy): 70% code + 20% chat + 10% sonnet

각 mix × {항상 spec ON, workload-aware gating} = 6 cell.

## 3. 가설

| mix | 항상 spec ON | workload-aware gating | gating 영역 추가 향상 |
|---|---|---|---|
| M1 sonnet-heavy | sonnet 70% × +134% + chat 20% × +37.5% + code 10% × −23.2% ≈ +96% | sonnet 70% × +134% + chat 20% × +37.5% + code 10% × 0% ≈ +99% | +3 pp |
| M2 balanced | 33% × (+134 +37.5 −23.2) ≈ +49% | 33% × (+134 +37.5 + 0) ≈ +57% | +8 pp |
| M3 code-heavy | sonnet 10% × +134 + chat 20% × +37.5 + code 70% × −23.2 ≈ +5% | sonnet 10% × +134 + chat 20% × +37.5 + code 70% × 0 ≈ +21% | +16 pp |

→ **code-heavy traffic 일수록 gating 효과 큼**. balanced mix 도 +8 pp 추가 향상.

## 4. effort

- classifier 구현 (regex + tokenizer count): 2-3 시간
- 옵션 A routing wrapper (`/tmp/run_workload_aware.py`): 2-3 시간
- mix workload generator (M1/M2/M3): 1-2 시간
- 측정 (6 cell × 5-10 min): 30-60 min
- 결과 doc + 분석 doc §6 갱신: 1-2 시간
- **총 effort: 1-2 일**

## 5. 진행 시 신설 SUB (candidate)

- **SUB_075** (제안 번호): workload-aware gating PoC + 3-mix 측정 + production 권장 doc 작성.

## 6. 확인 / 업데이트 필요 doc

| 파일 | 갱신 위치 |
|---|---|
| 신규 `measurements/sub075_workload_aware_<TS>/RESULTS.md` | 6 cell 결과 + classifier accuracy + gating 효과 정량 |
| 신규 `planning/SUB_075_workload_aware_gating.md` | classifier 설계 + routing 구현 + 측정 plan |
| `analysis/workload_acceptance_analysis_20260524.md` | §6 (heuristic → 실측 결과), §10.3 차별점 보강 (predictive gating fact 확정) |
| `Best_SpecDecode_10778tps.md` | §7 다음 path — production 권장 doc |
| `INDEX.md` | §1 active SUB 표 + §4 mix 시나리오 표 |
| `id_registry.md` | SUB_075 entry |

## 7. risk

- classifier accuracy — false negative (code 인데 spec ON 으로 분류) 시 code workload 회귀 발생. precision/recall trade-off.
- 옵션 A 의 GPU 메모리 — 2 instance 가 본 H100×8 환경에서 동시 운영 가능한지 미검증 (Llama-3.3-70B + TP=8 × 2 = 16 GPU 필요. 본 환경 8 GPU 면 TP=4 로 split 또는 단일 instance 의 dynamic config switch).

## 8. 결과 (SUB_076, 2026-05-24)

본 SUB 영역 idea I004 의 전체 PoC (classifier + routing + 3-mix 측정) 중 **1차 = classifier only** (routing 은 본 환경 GPU 제약으로 후속 SUB candidate).

### 8.1 측정 결과 — classifier accuracy (3 workload × 500 prompt = 1,500 prompt)

| true \\ pred | sonnet | chat | code | accuracy |
|---|---:|---:|---:|---:|
| sonnet (n=500) | **500** | 0 | 0 | 1.000 |
| chat (n=500) | 0 | **500** | 0 | 1.000 |
| code (n=500) | 0 | 0 | **500** | 1.000 |

**Macro accuracy: 1.000 (perfect)** | precision/recall/F1 모두 1.000.

### 8.2 sonnet/chat/code 모두 분류 성공

| workload | n | predicted | correct |
|---|---:|---:|---:|
| sonnet | 500 | sonnet × 500 | 500 (100%) |
| chat | 500 | chat × 500 | 500 (100%) |
| code | 500 | code × 500 | 500 (100%) |

→ **본 환경 prompt builder 의 strong signature** (chat: `<\|system\|>` tag 항상 존재 / code: `def`/`class`/160 `# comment line N:` / sonnet: free text without code/chat marker) 가 trivial 분류 가능.

### 8.3 한계 (정직 표시)

- real production traffic (ShareGPT / LMSYS-chat / mixed user prompt) 영역 accuracy 가 더 낮을 것 (보수 estimate 0.85~0.95)
- routing/serving 통합 안 됨 (본 환경 GPU 8 + 70B model = single instance, dual instance 또는 vLLM core 변경 필요)

### 8.4 후속 SUB candidate

- **SUB_080+** (제안): real user dataset (ShareGPT subset, LMSYS-chat-1M) 으로 accuracy 재측정
- **SUB_081+** (제안): vLLM Semantic Router (R17) 패턴으로 dual instance routing PoC — TP=4 × 2 instance 환경 필요
- **SUB_082+** (제안): vLLM per-request `speculative_config` override 지원 upstream PR

### 8.5 본 doc 갱신 (완료)

- `analysis/workload_acceptance_analysis_20260524.md` §6.0 (실측 accuracy 1.000 + 보수 estimate caveat)
- `INDEX.md` §1 active SUB 표
- raw: `eval/results/20260525_003850_sub076_classifier/{sonnet,chat,code}_{prompts,results}.json`
- classifier 코드: `/tmp/workload_classifier.py`
