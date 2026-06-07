# SUB_201 L2 — CPU prefetch + tokenization overlap

## 가설
TSK_043 의 host-bound (launch 38%) 분해에 따르면 vLLM API 서버는 request 도착 → BPE
tokenize → schedule 의 sync chain 에서 token id 가 ready 되기 전 GPU forward 를
시작할 수 없습니다. 다음 request 의 prompt tokenize 를 **백그라운드 worker** 에서
미리 (= 현재 in-flight GPU forward 와 overlap 되어) 수행하면 첫 prefill 의 critical
path 에서 BPE 비용을 줄일 수 있습니다.

## vLLM 내부 분석 (call site)

- `vllm/v1/engine/async_llm.py:AsyncLLM.add_request` 는 이미 비동기.
- 실제 BPE tokenize 는 `vllm/renderers/base.py:_tokenize_prompt_async` 에서 호출되고,
  `vllm/utils/async_utils.py:AsyncMicrobatchTokenizer.encode` 가 `loop.run_in_executor`
  를 거쳐 `ThreadPoolExecutor` 로 위임함.
- 기본 executor 는 **max_workers=1** 의 single-thread pool (`renderer/base.py` 의
  `ThreadPoolExecutor(max_workers=pool_workers)` 와 공유, 그리고 renderer pool 의
  default `renderer_num_workers = 1`).
- 따라서 동시에 여러 request 가 도착해도 tokenize 가 **단일 worker 에서 직렬화**
  되며, `batch_wait_timeout_s = 2 ms` 의 head-of-line wait 가 첫 request 의 TTFT 에
  더해짐.

## Patch (env-gated)

`VLLM_PREFETCH_TOKENIZE=1` 일 때만 활성화. ABI 무변경.

1. `vllm/envs.py` — 새 env 변수 등록:
   - `VLLM_PREFETCH_TOKENIZE` (bool, default 0)
   - `VLLM_PREFETCH_TOKENIZE_WORKERS` (int, default 2)
2. `vllm/utils/async_utils.py:AsyncMicrobatchTokenizer.__init__` — env 가 set 되면:
   - 공유 single-thread executor 대신 **dedicated multi-thread executor** 사용.
   - `batch_wait_timeout_s` 를 `2 ms → 0.5 ms` 로 단축 (HOL wait 감소).

## 실험

| 항목 | 값 |
|---|---|
| GPU | NVIDIA B200 × 1 (GPU index 2) |
| 모델 | `Qwen/Qwen2.5-7B-Instruct` (BF16, eager) |
| TP | 1 |
| 데이터셋 | random (sharegpt 대용; sharegpt json 부재) |
| input_len / output_len | 512 / 256 |
| num_prompts | 200 |
| max_concurrency | 16 |
| metric | `vllm bench serve` (openai backend, /v1/completions) |

상세 결과는 `MEASUREMENTS.md`, 원본 결과는 `runs/bench_*.json`.

## 산출물

- `run_bench.sh` — baseline / prefetch_on 두 모드의 server launch + bench
- `summarize.py` — 두 json 비교 표 출력
- `runs/bench_baseline.json|.stdout` — baseline 측정 결과
- `runs/bench_prefetch_on.json|.stdout` — prefetch_on 측정 결과
- `runs/server_*.log` — 서버 stdout/stderr
- `MEASUREMENTS.md` — 비교 표 + Δ% + 결론
