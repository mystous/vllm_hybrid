"""A1 correctness gate — vanilla vs cpu_amx_draft 분포 유사성 검증.

CLAUDE.md §Constraint 운영 해석:
  - per-token logprob max abs diff < 0.1
  - sequence PPL relative diff < 5%

두 vllm OpenAI 호환 서버를 (이미 부팅된 상태) 차례로 호출, 동일 100 prompt 의
greedy completion 을 비교한다. 호출 시점에는 모드 하나만 켜져 있으므로
PORT_A / PORT_B 인자로 vanilla / cpu_amx_draft 두 run 의 결과 파일을 별도로
저장한 다음, 이 스크립트로 합쳐서 metric 을 산정한다.

phase 1) `--collect PORT MODEL OUT.jsonl` — 100 prompt × max_tokens=64 × t=0.0
         × top_logprobs=5 결과를 JSONL 로 dump. PPL 계산용으로 매 토큰 logprob
         포함.
phase 2) `--compare RUN_A.jsonl RUN_B.jsonl OUT.json` — token-level diff + PPL
         diff 산정.

NOTE: vLLM 의 echo+logprobs 는 prompt logprob 만 echo (생성 token 의 logprob 은
별도 필드). 우리는 *생성 토큰만* 비교한다. (prompt 는 동일하므로 prompt logprob
은 자명히 일치.)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx

PROMPTS = [
    "The capital of France is",
    "Briefly: photosynthesis is",
    "Translate to French: 'Good morning, how are you today?' →",
    "List 3 prime numbers under 20:",
    "Why does the sky look blue?",
    "Summarize the plot of Hamlet in two sentences.",
    "Explain Bayes' theorem in one sentence.",
    "What is the boiling point of water at sea level in Celsius?",
    "Write a haiku about autumn.",
    "Who wrote 'Pride and Prejudice'?",
    "In Python, how do you reverse a list in place?",
    "Define entropy in physics.",
    "Define entropy in information theory.",
    "State the Pythagorean theorem.",
    "List the planets in order from the Sun:",
    "What is the speed of light in vacuum (m/s)?",
    "Give an example of a renewable energy source:",
    "Name a function that computes the factorial of n in Python:",
    "What does HTTP stand for?",
    "Convert 100 Fahrenheit to Celsius:",
    "What is the chemical symbol for gold?",
    "Name three rivers in Europe:",
    "What language family does Mandarin belong to?",
    "What is mitochondria's main function?",
    "Give the formula for the area of a circle:",
    "What is the largest desert in the world?",
    "Who painted the Mona Lisa?",
    "Define recursion in computer science:",
    "What does CPU stand for?",
    "Name a famous Renaissance artist:",
    "What is the smallest prime number?",
    "Explain what a black hole is.",
    "What is the largest ocean on Earth?",
    "Who developed the theory of general relativity?",
    "List two operating systems based on Linux:",
    "Briefly explain Newton's third law:",
    "What is the SI unit of electric current?",
    "Translate to Spanish: 'Where is the library?'",
    "What does DNA stand for?",
    "List 3 noble gases:",
    "What is the freezing point of water in Fahrenheit?",
    "Name a sorting algorithm with average O(n log n):",
    "What planet is known as the Red Planet?",
    "Briefly explain the greenhouse effect:",
    "What is 7 factorial?",
    "Write the chemical equation for combustion of methane:",
    "Name an algorithm for shortest path in a graph:",
    "What is the difference between TCP and UDP in one line?",
    "Give one example of an autoimmune disease:",
    "What does API stand for?",
    "What is the tallest mountain on Earth?",
    "Name the inventor of the World Wide Web:",
    "Define a vector in mathematics:",
    "What is the chemical formula for table salt?",
    "Briefly explain Moore's law:",
    "List three primary colors of light:",
    "Define a palindrome with an example:",
    "What is the volume of a cube with side 3?",
    "What is the time complexity of binary search?",
    "Name a JVM language other than Java:",
    "What is the population of Tokyo (approx)?",
    "What does GPU stand for?",
    "Convert 256 to binary:",
    "What is the largest prime under 30?",
    "Briefly state the fundamental theorem of calculus:",
    "What language did the ancient Romans speak?",
    "What is the boiling point of nitrogen at 1 atm (Celsius)?",
    "Name a country in South America:",
    "What is the chemical symbol for iron?",
    "Briefly explain pH:",
    "Give the formula for kinetic energy:",
    "Who proposed the heliocentric model?",
    "What is the half-life of carbon-14?",
    "Briefly: what is overfitting in ML?",
    "Briefly: what is gradient descent?",
    "What is sigmoid function?",
    "Define dropout in neural networks:",
    "What is the role of an optimizer in training?",
    "What is one advantage of attention over RNN?",
    "Briefly: what is a transformer model?",
    "Name a popular benchmark for LLMs:",
    "Briefly: what is a hash function?",
    "Briefly: what is a JIT compiler?",
    "Define context-free grammar:",
    "What is the airspeed velocity of an unladen swallow?",
    "Briefly explain BFS vs DFS:",
    "Define isomorphism in math:",
    "Briefly explain finite state machine:",
    "What does ACID stand for in databases?",
    "Name a NoSQL database:",
    "Define normalization in databases:",
    "Briefly: what is a deadlock?",
    "Define mutex vs semaphore:",
    "Briefly: what is virtual memory?",
    "Name a popular Linux distribution:",
    "Define a Turing machine briefly:",
    "What is P vs NP?",
    "Briefly state Halting problem:",
    "Define lambda calculus briefly:",
    "Name a functional programming language:",
    "Briefly: what is a monad?",
    "Define currying in FP:",
]
PROMPTS = PROMPTS[:100]
assert len(PROMPTS) == 100, f"expected 100 prompts, got {len(PROMPTS)}"


async def _one_collect(client, port, model, idx, prompt, max_tokens=64):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": 5,
        "echo": False,
    }
    r = await client.post(
        f"http://127.0.0.1:{port}/v1/completions", json=payload,
        timeout=httpx.Timeout(300.0, connect=10.0))
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    return {
        "i": idx,
        "prompt": prompt,
        "text": choice["text"],
        "tokens": choice.get("logprobs", {}).get("tokens", []),
        "token_logprobs": choice.get("logprobs", {}).get(
            "token_logprobs", []),
        "finish_reason": choice["finish_reason"],
    }


async def _collect_async(port: int, model: str, out: Path,
                         conc: int = 16, max_tokens: int = 64,
                         n_prompts: int = 100):
    out.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(conc)
    t0 = time.time()
    done = 0
    prompts = PROMPTS[:n_prompts]

    async def _bounded(idx, prompt, cl):
        nonlocal done
        async with sem:
            row = await _one_collect(cl, port, model, idx, prompt, max_tokens)
            done += 1
            if done % max(1, len(prompts) // 5) == 0:
                print(f"[collect] {done}/{len(prompts)} "
                      f"elapsed={time.time()-t0:.1f}s",
                      flush=True)
            return row
    async with httpx.AsyncClient(limits=httpx.Limits(
            max_connections=conc*2,
            max_keepalive_connections=conc*2)) as cl:
        rows = await asyncio.gather(*(
            _bounded(i, p, cl) for i, p in enumerate(prompts)))
    # preserve original prompt order (idx)
    rows.sort(key=lambda r: r["i"])
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[collect] done in {time.time()-t0:.1f}s → {out}", flush=True)


def collect(port: int, model: str, out: Path, max_tokens: int = 64,
            n_prompts: int = 100, conc: int = 16) -> None:
    """Phase 1: collect run output (greedy, with per-token logprobs)."""
    asyncio.run(_collect_async(port, model, out, conc=conc,
                                max_tokens=max_tokens, n_prompts=n_prompts))


def compare(run_a: Path, run_b: Path, out: Path) -> None:
    """Phase 2: token-level + sequence PPL comparison."""
    a_rows = [json.loads(l) for l in run_a.read_text().splitlines() if l]
    b_rows = [json.loads(l) for l in run_b.read_text().splitlines() if l]
    assert len(a_rows) == len(b_rows), \
        f"row count mismatch: {len(a_rows)} vs {len(b_rows)}"

    per_prompt = []
    max_abs_diff = 0.0
    n_pairs = 0
    sum_logp_a = 0.0
    sum_logp_b = 0.0
    n_tok_total_a = 0
    n_tok_total_b = 0
    seq_ppl_rel_diffs = []
    token_match_count = 0
    token_total_compared = 0
    for ra, rb in zip(a_rows, b_rows):
        assert ra["prompt"] == rb["prompt"], "prompt order mismatch"
        a_tok = ra["tokens"]
        b_tok = rb["tokens"]
        a_lp = [x for x in ra["token_logprobs"] if x is not None]
        b_lp = [x for x in rb["token_logprobs"] if x is not None]
        n_a, n_b = len(a_lp), len(b_lp)
        n_min = min(n_a, n_b)
        # token-level diff: compare up to n_min (positions present in both)
        prompt_max_diff = 0.0
        prompt_match = 0
        for i in range(n_min):
            if a_tok[i] == b_tok[i]:
                prompt_match += 1
            diff = abs(a_lp[i] - b_lp[i])
            if diff > prompt_max_diff:
                prompt_max_diff = diff
        if prompt_max_diff > max_abs_diff:
            max_abs_diff = prompt_max_diff
        n_pairs += n_min
        token_match_count += prompt_match
        token_total_compared += n_min
        # sequence PPL on the full generated sequence of each run separately
        if n_a > 0 and n_b > 0:
            ppl_a = math.exp(-sum(a_lp) / n_a)
            ppl_b = math.exp(-sum(b_lp) / n_b)
            rel_diff = abs(ppl_a - ppl_b) / max(ppl_a, ppl_b)
            seq_ppl_rel_diffs.append(rel_diff)
        sum_logp_a += sum(a_lp)
        sum_logp_b += sum(b_lp)
        n_tok_total_a += n_a
        n_tok_total_b += n_b
        per_prompt.append({
            "i": ra["i"],
            "prompt": ra["prompt"][:60],
            "n_a": n_a, "n_b": n_b,
            "token_match_n": prompt_match,
            "token_match_frac": (prompt_match / n_min) if n_min else 0.0,
            "max_lp_diff": prompt_max_diff,
            "ppl_a": math.exp(-sum(a_lp)/n_a) if n_a > 0 else None,
            "ppl_b": math.exp(-sum(b_lp)/n_b) if n_b > 0 else None,
        })

    agg_ppl_a = math.exp(-sum_logp_a / n_tok_total_a) if n_tok_total_a else None
    agg_ppl_b = math.exp(-sum_logp_b / n_tok_total_b) if n_tok_total_b else None
    agg_ppl_rel = (abs(agg_ppl_a - agg_ppl_b) / max(agg_ppl_a, agg_ppl_b)
                   if agg_ppl_a and agg_ppl_b else None)
    mean_ppl_rel = (sum(seq_ppl_rel_diffs) / len(seq_ppl_rel_diffs)
                    if seq_ppl_rel_diffs else None)
    summary = {
        "n_prompts": len(a_rows),
        "n_tokens_a_total": n_tok_total_a,
        "n_tokens_b_total": n_tok_total_b,
        "n_pairs_compared": n_pairs,
        "logprob_max_abs_diff": max_abs_diff,
        "token_match_n": token_match_count,
        "token_total_compared": token_total_compared,
        "token_match_frac": (token_match_count / token_total_compared
                             if token_total_compared else 0.0),
        "agg_ppl_a": agg_ppl_a,
        "agg_ppl_b": agg_ppl_b,
        "agg_ppl_rel_diff": agg_ppl_rel,
        "mean_seq_ppl_rel_diff": mean_ppl_rel,
        "gates": {
            "logprob_max_abs_diff_lt_0.1": max_abs_diff < 0.1,
            "agg_ppl_rel_lt_0.05": (agg_ppl_rel is not None
                                    and agg_ppl_rel < 0.05),
            "mean_seq_ppl_rel_lt_0.05": (mean_ppl_rel is not None
                                          and mean_ppl_rel < 0.05),
        },
    }
    out.write_text(json.dumps(
        {"summary": summary, "per_prompt": per_prompt},
        indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--port", type=int, required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-tokens", type=int, default=64)
    c.add_argument("--n-prompts", type=int, default=100)
    c.add_argument("--conc", type=int, default=16)
    p = sub.add_parser("compare")
    p.add_argument("--run-a", required=True)
    p.add_argument("--run-b", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args.port, args.model, Path(args.out),
                max_tokens=args.max_tokens, n_prompts=args.n_prompts,
                conc=args.conc)
    else:
        compare(Path(args.run_a), Path(args.run_b), Path(args.out))


if __name__ == "__main__":
    main()
