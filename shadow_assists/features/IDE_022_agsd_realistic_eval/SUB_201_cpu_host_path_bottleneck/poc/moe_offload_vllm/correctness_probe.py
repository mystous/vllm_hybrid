#!/usr/bin/env python3
"""Correctness probe: query 10 prompts × max_tokens=128 and dump logprobs.

Usage: python correctness_probe.py <port> <out.json>
"""
import json
import sys
import time

import requests

PROMPTS = [
    "Explain the difference between supervised and unsupervised learning in one paragraph.",
    "Write a Python function that computes the nth Fibonacci number iteratively.",
    "What is the largest planet in our solar system, and how is its mass compared to Earth?",
    "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
    "Summarize what an LLM is in two sentences.",
    "Describe how a transformer attention block works at a high level.",
    "List three causes of inflation in modern economies.",
    "Give a one-paragraph plot summary of Hamlet.",
    "What does the acronym GPU stand for, and what is its primary purpose?",
    "Write a haiku about machine learning.",
]


def main(port: int, out: str) -> None:
    url = f"http://localhost:{port}/v1/completions"
    results = []
    for i, p in enumerate(PROMPTS):
        body = {
            "model": "moe-offload-test",
            "prompt": p,
            "max_tokens": 128,
            "temperature": 0.0,  # greedy
            "top_p": 1.0,
            "logprobs": 1,
            "seed": 0,
        }
        t0 = time.time()
        r = requests.post(url, json=body, timeout=300)
        r.raise_for_status()
        j = r.json()
        elapsed = time.time() - t0
        choice = j["choices"][0]
        rec = {
            "i": i,
            "prompt": p,
            "text": choice.get("text", ""),
            "finish_reason": choice.get("finish_reason"),
            "logprobs": choice.get("logprobs"),
            "elapsed_s": elapsed,
        }
        results.append(rec)
        print(f"[{i}] {elapsed:.2f}s  finish={rec['finish_reason']}  "
              f"text_len={len(rec['text'])}")
    with open(out, "w") as f:
        json.dump({"prompts": PROMPTS, "results": results}, f, indent=2)
    print(f"saved → {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
