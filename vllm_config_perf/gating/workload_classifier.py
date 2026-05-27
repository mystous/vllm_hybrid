"""AGSD workload classifier — regex-based, lightweight CPU-only.

분류 카테고리: sonnet / chat / code

본 분류 알고리즘은 SUB_076 (2026-05-24) 측정에서 self-classification accuracy 1.000
달성 (SUB_044/047/071 builder dataset). 실제 production traffic (ShareGPT/LMSYS-chat 등)
에서는 더 낮을 것 (보수 estimate 0.85~0.95).

알고리즘:
1. n_chat_tag ≥ 1   → "chat"     (<|system|> / <|user|> / <|assistant|>)
2. code_hits ≥ 2     → "code"    (n_import ≥ 2, n_comment_line ≥ 10, n_py_kw ≥ 3)
3. default          → "sonnet"

본 분류기는 단독 import 가능하며 vLLM 의존성 없음.
"""

from __future__ import annotations

import re
from typing import Literal

WorkloadType = Literal["sonnet", "chat", "code"]

# ---- regex patterns (precompiled, thread-safe) ----
_RE_CHAT_TAG = re.compile(r"<\|(system|user|assistant)\|>")
_RE_IMPORT = re.compile(r"^\s*(?:import |from \w+ import )", re.MULTILINE)
_RE_COMMENT_LINE = re.compile(r"^\s*#", re.MULTILINE)
_RE_PY_KW = re.compile(
    r"\b(return|else|elif|except|raise|yield|lambda|for|while|def|class|if|try)\b"
)
_RE_TRIPLE_TICK = re.compile(r"```")


def classify(prompt: str) -> WorkloadType:
    """단일 prompt 의 workload 카테고리를 결정.

    Args:
        prompt: classification 대상 텍스트

    Returns:
        "sonnet" | "chat" | "code"
    """
    n_chat_tag = len(_RE_CHAT_TAG.findall(prompt))
    if n_chat_tag >= 1:
        return "chat"

    n_import = len(_RE_IMPORT.findall(prompt))
    n_comment_line = len(_RE_COMMENT_LINE.findall(prompt))
    n_py_kw = len(_RE_PY_KW.findall(prompt))

    code_hits = sum(
        [
            n_import >= 2,
            n_comment_line >= 10,
            n_py_kw >= 3,
        ]
    )
    if code_hits >= 2:
        return "code"

    return "sonnet"


def classify_batch(prompts: list[str]) -> list[WorkloadType]:
    """multi-prompt batch classify. ProcessPoolExecutor 와 사용 가능."""
    return [classify(p) for p in prompts]


# ---- self-test ----
if __name__ == "__main__":
    samples = [
        ("<|system|>\nYou are helpful.<|user|>\nHello<|assistant|>\n", "chat"),
        ("Shall I compare thee to a summer's day?", "sonnet"),
        (
            "import os\nimport sys\n# main entry\n# logic line 1\n"
            "# logic line 2\n# logic line 3\n# logic line 4\n# logic line 5\n"
            "# logic line 6\n# logic line 7\n# logic line 8\n# logic line 9\n"
            "# logic line 10\ndef main():\n    for i in range(10):\n        return i\n",
            "code",
        ),
    ]
    for prompt, expected in samples:
        actual = classify(prompt)
        marker = "OK" if actual == expected else "FAIL"
        print(f"[{marker}] {expected!r:>8} == {actual!r:<8}  preview={prompt[:40]!r}")
