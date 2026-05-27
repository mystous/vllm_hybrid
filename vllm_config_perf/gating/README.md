# AGSD Gating Subsystem — workload-aware spec decoding 라우터

> **요약**: vLLM 의 per-request `speculative_config` override 가 아직 upstream 에 없으므로, 본 라우터가 **2 개의 vLLM 인스턴스를 띄우고 (vanilla / Trident) workload 분류 결과에 따라 forward**.
>
> **출처 측정**: SUB_076 (classifier accuracy 1.000) → SUB_080 (analytical gating PoC) → SUB_092 (HTTP router PoC) → SUB_094 (end-to-end 3 mix 모두 net positive).
>
> **상위 README**: [`../README.md`](../README.md)

---

## 0. 한 줄 결론

3 mix scenario × 200 prompt × Qwen 7B × 2 GPU 환경에서:

| Mix | vanilla-only | trident-only | **AGSD-gated** | vs vanilla | vs trident |
|---|---:|---:|---:|---:|---:|
| balanced (34:33:33) | 3,753 | 4,238 | **6,073** | **+61.8%** ⭐ | **+43.3%** ⭐ |
| sonnet-heavy (60:20:20) | 3,865 | 5,234 | **6,025** | **+55.9%** ⭐ | **+15.1%** |
| code-heavy (10:20:70) | 3,966 | 7,512 | **8,825** | **+122.5%** ⭐ | **+17.5%** |

→ **모든 mix scenario 에서 단일 backend 보다 우월**. classifier 정확도 100%, overhead 0.16% (1.22 ms / 759 ms 평균 forward time).

---

## 1. 아키텍처

```
                    client (mixed traffic, concurrency=32)
                              │
                              ▼
   ┌──────────────────────────────────────────────────┐
   │  CPU router (agsd_router.py)                     │
   │  - FastAPI + uvicorn + uvloop (asyncio)          │  ← high concurrency I/O
   │  - ProcessPoolExecutor (16 workers, regex)       │  ← parallel classifier
   │  - httpx AsyncClient (1024 conn pool)            │  ← forwarder
   │  - 1.22 ms classify avg                          │
   └──────────────┬───────────────────────────────────┘
                  │ classify → decision → forward
         ┌────────┴────────┐
         ▼                 ▼
   ┌─────────────┐    ┌─────────────────────┐
   │ vLLM serve  │    │ vLLM serve          │
   │ vanilla     │    │ suffix+PIECEWISE    │
   │ GPU 1       │    │ GPU 2               │
   │ port 8001   │    │ port 8002           │
   └─────────────┘    └─────────────────────┘
```

---

## 2. routing 정책 (workload → backend)

| workload | backend | 이유 |
|---|---|---|
| **chat** | vanilla | Qwen 7B chat 환경에서 vanilla 와 trident 차이 작음 (+7.1%) — parallel GPU 활용을 위해 vanilla 로 분배 |
| **sonnet** | trident | suffix decoder sweet spot (+52~+79%) |
| **code** | trident | suffix mitigation (+18.8% vs ngram -20.2%) |

→ env `AGSD_CHAT_BACKEND=trident` 로 chat 도 trident 로 보낼 수 있음 (large model 환경에서 권장).

상세 결정 매트릭스: [`recommendations.py`](recommendations.py).

---

## 3. 파일 구성

| 파일 | 역할 | 의존 |
|---|---|---|
| [`workload_classifier.py`](workload_classifier.py) | regex 기반 prompt → workload 분류 (3 카테고리) | stdlib only |
| [`recommendations.py`](recommendations.py) | (workload, model_size) → spec config 권장값 매트릭스 | stdlib only |
| [`agsd_router.py`](agsd_router.py) | FastAPI + uvloop + ProcessPool + httpx async 라우터 | `fastapi`, `httpx`, `uvloop`, `uvicorn` |
| [`launcher.sh`](launcher.sh) | 2 vLLM backend + router 동시 기동/종료 | bash, `vllm serve` |
| [`benchmark_mixed.py`](benchmark_mixed.py) | 3 mix × 3 scenario benchmark client | `httpx` |
| [`README.md`](README.md) | (본 문서) |

---

## 4. 빠른 시작 (Quick Start)

### 4.1 pip install

```bash
.venv/bin/pip install fastapi uvicorn[standard] uvloop httpx
```

### 4.2 backend + router 기동

```bash
# AGSD_MODEL 환경변수로 모델 변경 가능 (default Qwen/Qwen2.5-7B-Instruct)
# 본 launcher 는 GPU 1, 2 사용. 다른 GPU 면 CUDA_VISIBLE_DEVICES 직접 수정.
bash vllm_config_perf/gating/launcher.sh up

# 종료
bash vllm_config_perf/gating/launcher.sh down
```

### 4.3 단일 prompt routing 결정 query

```bash
curl -X POST http://127.0.0.1:8000/route \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Shall I compare thee to a summer'\''s day?"}'
# → {"workload":"sonnet","backend":"trident","backend_url":"...","classify_ms":1.2}
```

### 4.4 batch routing 결정

```bash
curl -X POST http://127.0.0.1:8000/batch_route \
     -H "Content-Type: application/json" \
     -d '{"prompts": ["...", "...", "..."]}'
# → {"workloads":[...],"backends":[...],"distribution":{"vanilla":..., "trident":...}}
```

### 4.5 OpenAI-compat forward (실제 generation)

```bash
curl http://127.0.0.1:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"model","prompt":"...","max_tokens":256,"temperature":0}'
```

### 4.6 mixed traffic benchmark

```bash
# vanilla-only baseline
.venv/bin/python vllm_config_perf/gating/benchmark_mixed.py \
    --scenario vanilla --mix balanced --out /tmp/v_bal.json

# trident-only baseline
.venv/bin/python vllm_config_perf/gating/benchmark_mixed.py \
    --scenario trident --mix balanced --out /tmp/t_bal.json

# AGSD-gated
.venv/bin/python vllm_config_perf/gating/benchmark_mixed.py \
    --scenario AGSD --mix balanced --out /tmp/agsd_bal.json
```

3 mix × 3 scenario = 9 회차 측정 → SUB_094 결과와 같은 표 생성.

---

## 5. classifier 알고리즘

```python
def classify(prompt: str) -> str:
    if n_chat_tag >= 1:           # <|system|> / <|user|> / <|assistant|>
        return "chat"
    code_hits = sum([
        n_import >= 2,            # import / from ... import
        n_comment_line >= 10,     # ^\s*#
        n_py_kw >= 3,             # def/class/return/for/while/...
    ])
    if code_hits >= 2:
        return "code"
    return "sonnet"  # default
```

- **measurement**: SUB_076 — self-classification accuracy **1.000** (SUB_044/047/071 builder dataset)
- **production 추정**: 0.85 ~ 0.95 (실제 ShareGPT / LMSYS-chat 등 mixed/ambiguous prompt 에서는 약간 떨어짐)
- **overhead**: 단일 prompt classify ~1.2 ms (CPU regex). ProcessPool 16 workers 환경에서 batch 200 prompts → ~16 ms wall.

---

## 6. 한계 + 후속 개선

| 한계 | 회피 / 후속 |
|---|---|
| 본 라우터는 *2 instance* 운영 — GPU 2 장 필요 | vLLM 의 per-request `speculative_config` override 가 upstream 에 들어오면 *1 instance* 로 통합 가능 |
| classifier 가 regex 기반 — ambiguous prompt 에서 정확도 떨어짐 | LLM-as-classifier (small 1B 모델) — latency trade-off |
| forward streaming 시 backpressure 미세 처리 | uvloop + httpx async 충분 (1024 conn pool); 더 큰 scale 은 connection pool tuning |
| chat 워크로드 routing 정책 | 모델 크기에 따라 다름 (Llama 70B chat → trident, Qwen 7B chat → vanilla). `AGSD_CHAT_BACKEND` 환경변수로 조정 |
| sustained QPS 의 router stability | SUB_094 는 burst (200 prompt × 32 concurrency). 장시간 부하 측정은 후속 SUB |

---

## 7. 측정 출처 (참고 문서)

- [`measurements/sub076_classifier_20260524/RESULTS.md`](../measurements/sub076_classifier_20260524/RESULTS.md) — classifier accuracy 1.000
- [`measurements/sub080_gating_prod_20260524/RESULTS.md`](../measurements/sub080_gating_prod_20260524/RESULTS.md) — analytical gating decision
- [`measurements/sub092_router_poc_20260525/RESULTS.md`](../measurements/sub092_router_poc_20260525/RESULTS.md) — HTTP router PoC (decision-only)
- [`measurements/sub094_agsd_e2e_20260525/RESULTS.md`](../measurements/sub094_agsd_e2e_20260525/RESULTS.md) — end-to-end 3 mix × 3 scenario (★ 본 deliverable 의 핵심)
- [`measurements/sub095_agsd_e2e_multi_model_20260525/RESULTS.md`](../measurements/sub095_agsd_e2e_multi_model_20260525/RESULTS.md) — Qwen 0.5B/1.5B/7B/32B 확장
- raw bench JSON: [`measurements/sub094_agsd_e2e_20260525/benchmark_*.json`](../measurements/sub094_agsd_e2e_20260525/)
- raw bench JSON (multi-model): [`measurements/sub095_agsd_e2e_multi_model_20260525/qwen*/benchmark_*.json`](../measurements/sub095_agsd_e2e_multi_model_20260525/)
- production guide: [`docs/spec_decoding/README.md`](../docs/spec_decoding/README.md)
- idea: [`docs/idea/IDE_012_workload_aware_gating_poc.md`](../docs/idea/IDE_012_workload_aware_gating_poc.md)

---

## 8. Change Log

| 일자 | 변경 |
|---|---|
| 2026-05-27 | 신규 작성. SUB_094 의 ephemeral `/tmp/sub094_*.py` 스크립트를 architecture 명세 기준 production-ready 형태로 재구성. classifier (SUB_076), recommendations table (SUB_080), router (SUB_092/094), launcher, benchmark client 포함. |
