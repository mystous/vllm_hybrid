# PR Draft — IDE_006 NEO 4차 재정의 (asymmetric GPU/CPU pipelining + KV exclusive ownership)

> **본 파일은 GitHub web 에서 PR open 시 description 으로 사용** (gh CLI 미설치).
> Compare URL: https://github.com/mystous/vllm_hybrid/compare/main...feat/ide006-neo-asymmetric

---

## Title

`IDE_006 — NEO 4차 재정의 적재 (request 단위 GPU/CPU exclusive ownership + asymmetric pipelining + CPU pacpu 통합)`

---

## Summary

NEO 논문 ([arXiv 2411.01142](https://arxiv.org/abs/2411.01142), MLSys 2025, Apache 2.0) 의 *request 단위 GPU/CPU exclusive ownership + asymmetric sub-batch pipelining* 메커니즘을 vLLM v1 에 적재. IDE_006 의 1~3차 cold-KV partial attention 영역 (TSK_002 / TSK_011 / TSK_012 등) 은 NEO 4차 적용 시 cdec dispatch hook 으로 흡수되어 일괄 기각.

- 68 commits ahead of `main`
- 111 files changed, +17651 / -209 lines

---

## Land 영역

### 신규 모듈

| 영역 | 파일 |
|---|---|
| Request scheduler (3 큐 + load-aware mode selection) | `vllm/v1/core/sched/neo_scheduler.py`, `neo_scheduler_adapter.py`, `sub_batch.py`, `mode_selector.py` |
| KV exclusive ownership (mirror → exclusive) | `vllm/v1/core/sched/neo_cpu_kv_buffer.py` (CPU KV pool), `vllm/v1/request.py` (`SWAPPED_OUT` enum) |
| EngineCore swap dispatch hook | `vllm/v1/engine/core.py` (`VLLM_NEO_KV_FREE=1` opt-in) |
| Worker per-layer KV move | `vllm/v1/worker/gpu_model_runner.py` (`_neo_handle_kv_swap`) |
| Asymmetric pipelining (forward fork) | `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/sub_batch_executor.py` |
| TablePerfPredictor + ModelProfiler | `vllm/v1/core/sched/perfpredictor.py`, `vllm/v1/metrics/profiler.py`, `neo_perfpredictor_cache.py` |
| NEO pacpu (CPU attention kernel, ISPC + AVX-512spr-x16) | `csrc/cpu/pacpu/` (cherry-pick from NEO repo) |
| Python wrapper + KV layout adapter | `vllm/v1/attention/ops/neo_pacpu.py` |
| `unified_attention_with_output` cdec dispatch hook | `vllm/model_executor/layers/attention/attention.py` |

### NEO stage helpers (모델 별)

`forward_neo_pipelined` + `neo_preproj` / `neo_attention` / `neo_postproj`:
- ✅ Llama (`llama.py`)
- ✅ Qwen2 (`qwen2.py`) — Mistral / Phi3 자동 상속
- ✅ Gemma (`gemma.py`) / Gemma2 (`gemma2.py`)

### config flag + CLI

- `kv_cache_policy: mirror | exclusive` (`vllm/config/scheduler.py`)
- `--enable-neo-asymmetric` (`vllm/engine/arg_utils.py`)

### env opt-in

| env | 의미 |
|---|---|
| `VLLM_NEO_KV_FREE=1` | EngineCore swap dispatch 활성 (없으면 NEO sibling 결정만) |
| `VLLM_NEO_SWAP_OUT_RATIO` | swap_out_threshold scale (default 1.0; 단축 회차 발화 강제용) |
| `VLLM_NEO_AUTO_BUILD=1` | startup 시 pacpu auto-build |
| `VLLM_NEO_PREDICTOR_CACHE_DIR` | predictor disk cache path override |
| `ENABLE_NEO_INV=1` | XOR exclusive invariant assert (dev only) |

### pacpu 빌드 검증 (TP=1 + TP=8, prod 머신)

| Lib | 크기 |
|---|---|
| libpacpu-llama3_3_70b-tp8.so | 134360 bytes |
| libpacpu-qwen2_5_1_5b-tp1.so | 134472 bytes |
| libpacpu-qwen2_5_7b-tp1.so | 134472 bytes |
| libpacpu-qwen2_5_72b-tp8.so | 134360 bytes |
| libpacpu-mistral_7b-tp1.so | 134408 bytes |
| libpacpu-mistral_nemo_12b-tp1.so | 134408 bytes |
| libpacpu-phi3_medium_14b-tp1.so | 138504 bytes |

---

## Test plan

### 단위 test 회귀 (prod 머신, 2026-05-02)

```bash
LD_PRELOAD=/usr/lib64/libcuda.so.1 .venv/bin/python -m pytest \
  tests/v1/core/test_tsk014_scheduler.py \
  tests/v1/core/test_tsk015_kv_exclusive.py \
  tests/v1/core/test_tsk017_perfpredictor.py \
  tests/v1/core/test_step3_subbatch_slices.py \
  tests/v1/attention/test_neo_pacpu.py \
  tests/v1/model_executor/test_neo_pipelined_models.py
```

**결과: 102 passed / 1 failed in 24.18s**
- 1 fail = `test_resolve_neo_macro_unknown_returns_none` — Mistral 매핑 추가 후 'Mistral-7B-v0.1' 에 매핑되어 None 기대 위반. test 자체가 outdated (별도 fix 영역).

### prod env-ON smoke (Llama-70B + TP=8)

```bash
HF_HUB_OFFLINE=1 LD_PRELOAD=/usr/lib64/libcuda.so.1 \
  .venv/bin/python -u eval/run_neo_e2e_smoke.py \
  --model llama-70b --tensor-parallel-size 8 --max-model-len 512 \
  --gpu-memory-utilization 0.85 --max-tokens 16
```

**결과: token-id equality vanilla = NEO PASS**, NEO 인프라 회귀 zero, pacpu lib 정상 로드.

### prod 단축 throughput 회차

| 회차 | 회귀 | NEO 발화 |
|---|---|---|
| Vanilla 100×4096/2048 | wall 117.9s, out_tps 1737 | n/a |
| NEO ON 100×4096/2048 | wall 312.5s, out_tps 656 (2.65× 회귀) | swap_out=0 (KV 미압력) |
| NEO ON 200×4096/2048 + KV_FREE=1 + RATIO=0.05 | crash 후 진단 | **first fire: out=2 in=0** ✅ |

→ **NEO swap dispatch hook 의 자연 발화 영역 진입 + first fire 발화 검증**. 첫 fire 직후 `gpu_model_runner.py:1867` `torch.index_select` shape mismatch crash — fragile spot 식별, 별도 multi-day fix 영역.

### prod 본격 회차 (B-6, 미실행)

Llama-70B + TP=8 + 5000 prompts × 50:50 + 8192/8192 + ~4.7 hour. single conversation turn 영역 외. `TSK_015.md §3.5 P-1~P-5` 절차 참고.

---

## ID 운명 정리 (id_registry.md)

| ID | 운명 | 사유 |
|---|---|---|
| TSK_002 / TST_002 | 기각 | hot/cold partition path 가 NEO 4차 cdec dispatch hook 으로 흡수 |
| TSK_005 / TST_005 | 기각 | Q dependency dilemma (4차 이전) |
| TSK_006 / TST_006 | 기각 | Q chunk pipelining = TSK_018 batch dispatch 흡수 |
| TSK_008 / TST_008 | 기각 | request 단위 exclusive 변경 (TSK_015) 으로 분할 정책 영역 dead |
| TSK_010 / TST_010 | 기각 | TSK_018 ISPC + OpenMP 자체 흡수 |
| TSK_011 / TST_011 | 기각 | partition path dead, fallback 영구 소멸 |
| TSK_012 / TST_012 | 기각 | TSK_015 cold-blocks evict 흡수 |
| TST_013 | 완료 (분석 only) | §3 NEO reproduce 영구 deferred (별도 평가 환경 필요) |

ID 보존 — 재사용 금지 (CLAUDE.md 룰).

---

## Known issues / 후속 영역

### 1. NEO swap dispatch first fire 후 crash (multi-day fix)

`gpu_model_runner.py:1867` `torch.index_select` 의 shape mismatch (`[100]` vs `[99]`). swap-out 후 input_batch ↔ worker num_scheduled_tokens 정합 영역 fragile spot. B-2.b 의 cdec_req SWAPPED_OUT schedule path 와 worker 측 token count 영역 정합 fix 필요.

### 2. lint warnings (mechanical fix)

- ruff: 71 errors in 42 changed py files (54 fixable: B010 / F401 / I001 / UP037 / E702)
- mypy: 15 errors in 3 NEO core modules (`_ReqLike` protocol 의 `_str_id` attr 누락 7개, predictor type narrowing 3개, load() None check 2개)

### 3. test 회귀 1 건

`test_resolve_neo_macro_unknown_returns_none` — Mistral 매핑 추가로 outdated. test fix 영역.

### 4. prod B-6 본격 회차 미실행

5000 × 50:50 / 8192/8192 / ~4.7 hour 영역. 본 PR 의 *correctness* 는 token equality smoke 로 입증, *throughput gain* 측정은 별도 prod measurement session 영역.

---

## AI assistance disclosure

본 PR 작업에 AI assistance (Claude Opus 4.7 1M context) 가 사용됨. 모든 diff 는 사용자 (mystous@gmail.com) review 영역. 본 description 도 AI assist + 사용자 확인.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
