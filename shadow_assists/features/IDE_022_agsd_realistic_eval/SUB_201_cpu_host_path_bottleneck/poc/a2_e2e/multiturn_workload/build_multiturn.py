"""Phase B11 — multi-turn workload synth (옵션 A).

목적:
  A2 KV tiering lever 의 fetch path(prefix-cache hit) 를 본격 발화시키려면
  공유 prefix(4096+ token) 가 conversation 내 turn 마다 반복돼야 한다.
  본 스크립트는 200 conversation × 평균 5+ turn 을 합성 후 throughput_runner
  가 읽는 parquet 포맷(raw_text/corpus/...) 으로 저장한다.

구성:
  * 공유 system prompt = ~4096 token 의 기술 배경 텍스트
  * 각 conversation 의 모든 turn 이 동일 system prompt prefix 로 시작
  * turn N = system + (user_1\nassistant_1\n)... user_N
  * 각 conversation 5 turn → 200 conv × 5 turn = 1000 rows
  * throughput_runner --limit 1000 으로 모두 발사 (conc=64)
  * 같은 conversation 의 turn 들이 다른 conversation 도착 사이에 도달하므로
    프리픽스 캐시 히트가 자연스럽게 발생 (prefix-cache eviction 와 race).
"""
from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SYSTEM_SEED = (
    "You are an expert software engineer with deep knowledge of distributed "
    "systems, Linux kernel internals, CUDA, KV-cache management, vLLM, and "
    "modern transformer inference. "
)

SYSTEM_PARA = (
    "When you answer, follow these rules: (1) be precise and concise, "
    "(2) include code only when needed, (3) reference benchmarks if relevant, "
    "(4) think step by step. The following is internal architectural "
    "background you should use across all turns in this conversation:\n\n"
    "vLLM uses a paged KV cache. Each token contributes one slot to each "
    "layer. With GQA Qwen2.5-7B has 28 layers, 2 KV heads, 128 head_dim, "
    "bf16 elements. The total cache footprint per token is therefore "
    "28 * 2 * 128 * 2 (bytes) * 2 (K+V) = 28672 bytes per token. Block "
    "size defaults to 16 tokens. Scheduler tracks logical block table; "
    "physical block table lives on GPU HBM. The DRAM tiering lever evicts "
    "cold blocks D2H to a pinned host pool and fetches on hit. Prefix-cache "
    "hit accelerates the prefill of repeated system prompts: instead of "
    "recomputing the same KV vectors, vLLM looks up the block hash and "
    "reuses the existing block. This is the lever we want to exercise.\n\n"
)

# 대략 16 para ≈ 4100 token (Qwen2.5 tokenizer 기준; build_multiturn 호출 전 측정)
NUM_PARAS = 17

USER_TURNS = [
    "Explain how the DRAM tiering eviction stream overlaps with the forward pass when async mode is enabled. What synchronization primitives are required, and where should the lazy event_sync be placed?",
    "Compare the block-level hash policy in vLLM with the prefix-hash policy in SGLang. Which is more memory-efficient under high concurrency, and why?",
    "Sketch pseudocode for a worker-side fetch path that pulls evicted blocks from pinned host memory back into the GPU block pool, including coalescing of contiguous block ranges.",
    "Suppose we want to add an LRU2 admission policy on top of the existing FIFO eviction. How would this interact with the prefix-cache hit accounting? Walk me through the critical sections.",
    "What instrumentation would you add to the KVDramTier telemetry to track fetch-block latency p50/p99 separately from eviction latency? Provide concrete counter names and where to increment them.",
    "If we observe n_fetch=0 but n_evict>0 in production, list the top three probable root causes and the diagnostic queries / commands you would run to confirm each.",
    "Draft a benchmark plan to validate that multi-turn chat workloads achieve net positive throughput with the A2 lever, including the corpus, concurrency, and the statistical gate.",
    "Explain how cudaMemcpyAsync ordering interacts with CUDA graphs when capture mode is FULL_AND_PIECEWISE. Where might we hit illegal-memory-access bugs?",
    "Describe a strategy to dynamically size the pinned host pool based on observed prefix-hit rate. What feedback loop frequency would you target and why?",
    "Outline a minimal patch to surface KVDramTier metrics through the existing /metrics endpoint so that they show up as Prometheus counters next to vllm:prefix_cache_hits_total.",
]


def make_system_prompt() -> str:
    return SYSTEM_SEED + (SYSTEM_PARA * NUM_PARAS)


def make_conversation_rows(conv_idx: int, sys_prompt: str, n_turns: int):
    """단일 conversation 의 turn 별 누적 prompt 를 row 로 반환.

    turn N 의 raw_text = sys_prompt
                       + "\n\nUser: ..." + (이전 turn user/assistant)
                       + 이번 turn user
                       + "\n\nAssistant:"
    """
    rows = []
    history: list[str] = []
    rng = random.Random(conv_idx)
    # 5 turn 을 위해 user turn 후보에서 랜덤 선택 (중복 허용 — multi-turn 자연성)
    turns = [rng.choice(USER_TURNS) for _ in range(n_turns)]
    fake_assistant = (
        "Sure. The key observation is that under async eviction the host stream "
        "owns the D2H copy while the compute stream advances forward decode. We "
        "tie correctness to a lazy event_sync at block allocation and free. "
        "Concretely:\n"
        " - on evict: cudaMemcpyAsync(host, gpu, bytes, stream=tier_stream); record event.\n"
        " - on alloc: if block.last_evict_event is not None: cudaStreamWaitEvent(compute, evt).\n"
        " - on fetch: cudaMemcpyAsync(gpu, host, bytes, stream=tier_stream); record event.\n"
        "This pattern preserves D2H/H2D ordering while letting forward overlap."
    )
    for i, ut in enumerate(turns):
        history.append(f"\n\nUser: {ut}")
        body = sys_prompt + "".join(history) + "\n\nAssistant:"
        rows.append({
            "prompt_id": f"mt-{conv_idx:04d}-t{i+1}",
            "prompt_hash": hashlib.sha1(body.encode("utf-8")).hexdigest(),
            "corpus": "multiturn",
            "lang": "en",
            "n_input_tok": 0,  # not used by runner
            "raw_text": body,
        })
        # 다음 turn 의 prefix 에 가짜 assistant 답을 누적
        history.append(f" {fake_assistant}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-conv", type=int, default=200)
    ap.add_argument("--n-turns", type=int, default=5)
    ap.add_argument("--print-stats", action="store_true")
    args = ap.parse_args()

    sys_prompt = make_system_prompt()

    all_rows: list[dict] = []
    for c in range(args.n_conv):
        all_rows.extend(make_conversation_rows(c, sys_prompt, args.n_turns))

    # interleave: 같은 conv 의 turn 들이 시간상 떨어져 도달하도록 round-robin
    # → conc=64 인 상황에서 turn-1 들이 먼저 prefix 캐싱, turn-2..N 에서 hit.
    interleaved: list[dict] = []
    for t in range(args.n_turns):
        for c in range(args.n_conv):
            interleaved.append(all_rows[c * args.n_turns + t])

    tbl = pa.Table.from_pylist(interleaved)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, args.out)
    print(f"[mt-build] wrote {len(interleaved)} rows → {args.out}")

    if args.print_stats:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            n_sys = len(tok(sys_prompt, add_special_tokens=False)["input_ids"])
            print(f"[mt-build] system prefix tokens: {n_sys}")
            sample_lens = [
                len(tok(r["raw_text"], add_special_tokens=False)["input_ids"])
                for r in interleaved[::200]
            ]
            print(f"[mt-build] sample lens (every-200th): {sample_lens}")
        except Exception as e:  # noqa: BLE001
            print(f"[mt-build] tok-stats skipped: {e!r}")


if __name__ == "__main__":
    main()
