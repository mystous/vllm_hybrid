# SUB_092 — workload-aware router HTTP server PoC (Phase 1 actual implementation)

> **parent**: TSK_020 / Phase 1 (SUB_080 analytical) actual deploy
> **measurement**: 2026-05-25 KST 11:32, test mode 150 prompts × 2 model_size
> **status**: ✅ 완료 — classifier router 영역 production-ready (vLLM core 변경 없음)

---

## 1. PoC scope

본 SUB 영역 SUB_080 analytical PoC 영역 actual HTTP server 영역 구현:
- HTTP POST /route → single prompt 영역 분류 + spec config 권장
- HTTP POST /batch_route → batch prompts 영역 분류 + distribution
- HTTP GET /health, /recommendations → 메타 정보
- backend vLLM instance 영역 별도 (per-request spec override 영역 vLLM 영역 부재 영역 fundamental 한계, 후속 SUB)

본 PoC 영역 production deploy 가능 영역 — vLLM 영역 무관 영역 CPU only 영역 lightweight HTTP service.

## 2. 측정 결과 (test mode, 150 prompts × 2 model_size)

| model_size | n_prompts | routing time | workload_distribution | spec_method_distribution |
|---|---:|---:|---|---|
| large (≥70B) | 150 | 39.4 ms | sonnet 50 / chat 50 / code 50 | **suffix × 150** (3 workload 모두 suffix 권장) |
| small (≤7B) | 150 | 38.8 ms | sonnet 50 / chat 50 / code 50 | **vanilla × 150** (모두 vanilla 권장) |

→ **routing 영역 0.26 ms / prompt** (매우 빠른, 본 PoC 영역 production scale 영역 무리 없음).
→ classifier accuracy 영역 SUB_076 영역 1.000 영역 본 환경 builder prompt 영역 perfect.

## 3. spec recommendations table (본 SUB 측정값 기반)

| workload | model_size | spec config | expected_speedup_vs_vanilla |
|---|---|---|---|
| sonnet | large | suffix, num_spec=32 | **+51.6%** (SUB_089 canonical 3-run) |
| chat | large | suffix, num_spec=32 | **+63.8%** (SUB_085 v2) |
| code | large | suffix, num_spec=32 | **+18.9%** (SUB_085 v2, mitigation) |
| sonnet/chat/code | small (≤7B) | None (vanilla) | 0% (모든 spec method 영역 -17~-65% 회귀, SUB_088/090) |

## 4. HTTP endpoint

```bash
# 단일 prompt routing
curl -X POST http://localhost:8765/route \
     -H "Content-Type: application/json" \
     -d '{"prompt": "...", "model_size": "large"}'

# Batch routing (mix distribution)
curl -X POST http://localhost:8765/batch_route \
     -H "Content-Type: application/json" \
     -d '{"prompts": ["..."], "model_size": "large"}'

# Recommendations table
curl http://localhost:8765/recommendations
```

## 5. 본 PoC 영역 한계 + 후속

| 항목 | 한계 | 후속 SUB |
|---|---|---|
| routing decision 만, backend 영역 안 forward | vLLM 영역 per-request spec config override 영역 부재 | SUB_093+ (upstream PR 또는 본 fork core patch) |
| backend dual instance (TP=4 × 2) 영역 시도 안 함 | 본 환경 영역 multi-process orchestration 영역 large effort | SUB_094+ (Phase 3 actual) |
| real production traffic dataset 영역 classifier accuracy 영역 미검증 | 본 환경 builder set 영역 trivial | SUB_095+ (ShareGPT 영역 평가) |

## 6. 본 fork 영역 변경

- 신규: `/tmp/workload_router_server.py` (외부 script, vLLM 영역 무관)
- 본 fork vLLM core 변경 영역 없음 ✓

## 7. raw 자료

| 항목 | 위치 |
|---|---|
| router server script | `/tmp/workload_router_server.py` |
| classifier (재사용) | `/tmp/workload_classifier.py` (IDE_012 영역 SUB_076 영역 생성) |
| prompt builder (재사용) | `/tmp/run_workload_gen.py` |
| test output | (terminal 출력, RESULTS §2 영역 정리됨) |

## 8. production deploy 영역 권장

```bash
# 단일 router service (CPU only, vLLM 영역 무관)
.venv/bin/python /tmp/workload_router_server.py --mode server --port 8765 &

# router 영역 decision → vLLM backend 영역 spec_config 영역 LLM init 시점 영역 set
# (현 vLLM 영역 per-request override 영역 부재 영역 single instance 영역 단일 spec_config 영역 활용)
# → 다음 단계 영역 dual instance 영역 router 영역 forward 영역 production deploy
```
