"""TSK_042 corpus loader — 실 trace 데이터셋에서 prompt 추출 (streaming).

검증된 corpus (2026-06-02, mystous HF 계정):
  open : RyokoAI/ShareGPT52K, princeton-nlp/SWE-bench_Lite,
         openai_humaneval, mbpp  (code 는 LiveCodeBench 대체)
  gated: allenai/WildChat-1M, lmsys/lmsys-chat-1m  (동의 완료)

각 corpus 에서 "사용자 입력 텍스트"(마지막 user turn / 문제 설명)만 추출.
full-cache 하지 않고 streaming 으로 필요한 만큼만 (prompt_sampler 가 limit).

iter_corpus(name, limit) -> yield {"corpus","raw_text","native_lang"}
"""

from __future__ import annotations

from itertools import islice
from typing import Iterator

from datasets import load_dataset


def _last_turn(conv, role_key, content_key, user_vals):
    """대화 리스트에서 마지막 user turn 의 content 추출."""
    last = None
    for turn in conv:
        if turn.get(role_key) in user_vals:
            last = turn.get(content_key)
    return last


def _sharegpt(row):
    # {"conversations":[ '{"from":"human","value":...}', ... ]}  (각 turn 이 JSON 문자열)
    import json
    conv = []
    for t in (row.get("conversations") or []):
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except Exception:  # noqa: BLE001
                continue
        if isinstance(t, dict):
            conv.append(t)
    return _last_turn(conv, "from", "value", {"human", "user"})


def _wildchat(row):
    # {"conversation":[{"role":"user"/"assistant","content":...}], ...}
    return _last_turn(row.get("conversation") or [], "role", "content", {"user"})


def _lmsys(row):
    return _last_turn(row.get("conversation") or [], "role", "content", {"user"})


def _swebench(row):
    return row.get("problem_statement")


def _humaneval(row):
    return row.get("prompt")


def _mbpp(row):
    return row.get("text") or row.get("prompt")


# name -> (hf_id, config, split, extractor, native_lang_key)
CORPORA: dict[str, tuple] = {
    "sharegpt":  ("RyokoAI/ShareGPT52K",        None,        "train", _sharegpt,  None),
    "swebench":  ("princeton-nlp/SWE-bench_Lite", None,      "test",  _swebench,  None),
    "humaneval": ("openai_humaneval",            None,        "test",  _humaneval, None),
    "mbpp":      ("mbpp",                        "full",      "train", _mbpp,      None),
    "wildchat":  ("allenai/WildChat-1M",         None,        "train", _wildchat,  "language"),
    "lmsys":     ("lmsys/lmsys-chat-1m",         None,        "train", _lmsys,     "language"),
}

OPEN = ["sharegpt", "swebench", "humaneval", "mbpp"]
GATED = ["wildchat", "lmsys"]
ALL = OPEN + GATED


def iter_corpus(name: str, limit: int | None = None) -> Iterator[dict]:
    """corpus 에서 (최대 limit×oversample) 행을 streaming 으로 읽어
    {"corpus","raw_text","native_lang"} yield. 비거나 짧은 건 caller 가 filter."""
    hf_id, config, split, extractor, lang_key = CORPORA[name]
    kwargs = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    ds = load_dataset(hf_id, **kwargs)
    # length filter 후 limit 채우려면 여유분 필요 → caller 가 limit 줄 때 oversample
    take = None if limit is None else limit * 8
    for row in islice(ds, take):
        text = extractor(row)
        if not text or not isinstance(text, str):
            continue
        yield {
            "corpus": name,
            "raw_text": text,
            "native_lang": (row.get(lang_key) if lang_key else None),
        }


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or OPEN
    for nm in names:
        try:
            sample = list(iter_corpus(nm, limit=2))
            ex = sample[0]["raw_text"][:60].replace("\n", " ") if sample else "(empty)"
            print(f"  ✅ {nm:10s} got {len(sample)} rows | ex: {ex!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {nm:10s} {type(e).__name__}: {str(e).splitlines()[0][:80]}")
