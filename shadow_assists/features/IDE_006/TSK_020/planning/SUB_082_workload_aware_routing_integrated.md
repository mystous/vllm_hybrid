# SUB_082 — Phase 3: workload-aware routing 통합 (dual instance)

> **parent**: TSK_020 (성능 향상 plan Phase 3)
> **status**: 활성 (2026-05-24 신설)
> **effort**: 3-5 일
> **based on**: SUB_080 (Phase 1 단순 heuristic), vLLM Semantic Router (R17 arXiv 2603.21354)

## 1. 목표

Phase 1 의 단순 heuristic 을 dual vLLM instance (ngram + suffix) 영역 정식화. per-request workload classification → optimal spec method 선택.

## 2. 진행 절차

### Step 1 — dual instance viability 확인

본 H100×8 환경 영역 70B + TP=4 × 2 instance 가능한지:
- 70B + TP=4 메모리 사용 (kv cache + weights) 측정
- 2 instance 동시 운영 시 OOM viability
- alternative: instance 분리 (별도 GPU group)

### Step 2 — router service 구현

- HTTP server (FastAPI 또는 vLLM 영역 router) 가 prompt 받음
- classifier (IDE_012) 영역 분류 → 적절한 instance 영역 forward
- ngram instance: cap=8 + spec=7 (sonnet/chat용)
- suffix instance: spec=32 (code용, Phase 2 완료 후 cuda graph 모드)

### Step 3 — 3 mix scenario 측정

SUB_080 의 same scenario 영역 dual instance routing 측정. expected: M2 (balanced) 영역 SUB_080 대비 +6-8 pp 추가 향상.

## 3. risk

- TP=4 × 2 instance 영역 GPU memory infeasibility — 본 환경 GPU 8 영역 70B FP16/BF16 영역 TP=4 영역 cache 영역 GPU memory 영역 limit
- alternative: vLLM upstream per-request `speculative_config` override 지원 PR (single instance) — large upstream PR, review cycle 길음

## 4. 본 환경 한계 — likely outcome

본 H100×8 환경 영역 70B model 영역 TP=8 single instance 영역 max memory utilization 영역 dual instance 영역 viable 어려움. likely outcome:
- (a) viability 측정 결과 infeasible 확인 → upstream PR path 영역 분기 추천
- (b) 더 작은 model (Qwen 72B 또는 32B) 영역 TP=4 × 2 영역 PoC

## 5. 산출물

- viability 측정 결과 doc
- router service 구현 (또는 PoC level)
- `measurements/sub082_routing_<TS>/RESULTS.md`
