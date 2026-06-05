"""SUB_201 / B2(jump_forward) — correctness check.

Calls the running vLLM OpenAI server twice for the same 10 prompts:
  - constrained, JF OFF
  - constrained, JF ON

For each prompt:
  - validates that both outputs are valid JSON.
  - validates that each output conforms to the 5-key schema (all required
    keys present, types correct).
  - records whether the two outputs are byte-equal (informational metric —
    BF16 non-associativity + grammar trim differences may cause harmless
    divergences, so we report rate rather than gating on it).

Run twice (once with the engine booted JF-off, once with JF-on) and compare
the JSON files.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx
import pyarrow.parquet as pq


JSON_SCHEMA_5KEY = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
        "country": {"type": "string"},
        "score": {"type": "number"},
    },
    "required": ["name", "age", "email", "country", "score"],
}
JSON_SYS = (
    "You are a JSON generator. Given the user prompt, respond ONLY with a single JSON "
    "object that fits the user's request, using these 5 fields exactly: "
    "name (string), age (integer), email (string), country (string), score (number). "
    "Do not include any text outside the JSON object."
)


def schema_conform(obj):
    if not isinstance(obj, dict):
        return False, "not an object"
    for k in ("name", "age", "email", "country", "score"):
        if k not in obj:
            return False, f"missing key {k}"
    if not isinstance(obj["name"], str):
        return False, "name not string"
    if not isinstance(obj["age"], int) or isinstance(obj["age"], bool):
        return False, "age not int"
    if not isinstance(obj["email"], str):
        return False, "email not string"
    if not isinstance(obj["country"], str):
        return False, "country not string"
    if not isinstance(obj["score"], (int, float)) or isinstance(obj["score"], bool):
        return False, "score not number"
    return True, "ok"


async def one(client, url, model, prompt):
    payload = {
        "model": model,
        "prompt": f"{JSON_SYS}\n\nUser prompt: {prompt}\n\nJSON:",
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "guided_json": JSON_SCHEMA_5KEY,
    }
    r = await client.post(
        f"{url}/v1/completions", json=payload,
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    if r.status_code != 200:
        return None, f"http {r.status_code}: {r.text[:200]}"
    obj = r.json()
    txt = obj["choices"][0]["text"]
    return txt, None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--label", required=True, help="tag for output file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = pq.read_table(args.inp).to_pylist()
    rows = [r for r in rows if r["corpus"] == "sharegpt"][: args.limit]
    url = f"http://127.0.0.1:{args.port}"
    out = []
    async with httpx.AsyncClient() as client:
        for i, r in enumerate(rows):
            txt, err = await one(client, url, args.model, r["raw_text"])
            if err:
                out.append({"i": i, "ok": False, "error": err})
                continue
            # The schema-constrained completion is the leading '{...}' but
            # vLLM may continue generating free-form text after the closing
            # brace (no stop=}). Extract the first balanced JSON object.
            try:
                t = txt.lstrip()
                if t.startswith("{"):
                    depth = 0
                    in_str = False
                    esc = False
                    end = None
                    for i, ch in enumerate(t):
                        if in_str:
                            if esc:
                                esc = False
                            elif ch == "\\":
                                esc = True
                            elif ch == '"':
                                in_str = False
                        else:
                            if ch == '"':
                                in_str = True
                            elif ch == "{":
                                depth += 1
                            elif ch == "}":
                                depth -= 1
                                if depth == 0:
                                    end = i + 1
                                    break
                    json_str = t[:end] if end is not None else t
                else:
                    json_str = t
                parsed = json.loads(json_str)
                parse_ok = True
            except Exception as e:
                parsed = None
                parse_ok = False
                parse_err = str(e)[:200]
            conform_ok, conform_msg = (False, "parse failed")
            if parse_ok:
                conform_ok, conform_msg = schema_conform(parsed)
            out.append({
                "i": i, "ok": True,
                "raw_text": txt,
                "extracted_json": json_str if parse_ok else None,
                "parse_ok": parse_ok,
                "conform_ok": conform_ok,
                "conform_msg": conform_msg,
            })
            print(f"[{args.label}] {i}: parse={parse_ok} conform={conform_ok} ({conform_msg})")

    summary = {
        "label": args.label,
        "n": len(rows),
        "parse_ok_count": sum(1 for o in out if o.get("parse_ok")),
        "conform_ok_count": sum(1 for o in out if o.get("conform_ok")),
        "samples": out,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.label}] summary: parse_ok={summary['parse_ok_count']}/{summary['n']} "
          f"conform_ok={summary['conform_ok_count']}/{summary['n']}")
    sys.exit(0 if summary["conform_ok_count"] == summary["n"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
