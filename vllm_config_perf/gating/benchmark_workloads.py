"""AGSD per-workload benchmark client (32B / 500p / long-context).

benchmark_mixed.py 의 확장판:
  - 6 workload: pure sonnet / chat / code  +  mix balanced / sonnet-heavy / code-heavy
  - 500 prompts / workload
  - 8192-token input (tokenizer 정밀 패딩, per-prompt 랜덤화로 prefix-cache collapse 방지)
  - max_tokens 가변 (default 8192)
  - 3 scenario: vanilla(8001) / trident(8002) / AGSD(router 8000)

분류 신호 보존 (workload_classifier 와 정합):
  - chat   : 본문 맨 앞에 <|system|>/<|user|>/<|assistant|> 태그
  - code   : import ≥2 + 주석라인 ≥10 + python 키워드 ≥3 (맨 앞)
  - sonnet : 위 신호 없음 → default

사용:
    python benchmark_workloads.py --scenario AGSD --workload balanced \
        --num-prompts 500 --target-input-len 8192 --max-tokens 8192 \
        --concurrency 32 --out result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx

# ---- workload 분포 (500 prompt) ----
WORKLOAD_DEFINITIONS: dict[str, dict[str, float]] = {
    # pure
    "sonnet": {"sonnet": 1.0},
    "chat": {"chat": 1.0},
    "code": {"code": 1.0},
    # mix (SUB_093/097 정의)
    "balanced": {"sonnet": 0.34, "chat": 0.33, "code": 0.33},      # M2 34:33:33
    "sonnet-heavy": {"sonnet": 0.60, "chat": 0.20, "code": 0.20},  # M1 60:20:20
    "code-heavy": {"sonnet": 0.10, "chat": 0.20, "code": 0.70},    # M3 10:20:70
}

BACKEND_URL: dict[str, str] = {
    "vanilla": "http://127.0.0.1:8001/v1",
    "trident": "http://127.0.0.1:8002/v1",
    "AGSD": "http://127.0.0.1:8000/v1",
}

_SONNET_LINES: list[str] = []
_TOKENIZER = None


def _load_resources(sonnet_path: Path, model: str) -> None:
    global _SONNET_LINES, _TOKENIZER
    _SONNET_LINES = [
        ln.strip() for ln in sonnet_path.read_text().splitlines() if ln.strip()
    ]
    if not _SONNET_LINES:
        raise RuntimeError(f"no usable lines in {sonnet_path}")
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(model)


def _ntok(text: str) -> int:
    return len(_TOKENIZER(text, add_special_tokens=False)["input_ids"])


def _sample_block(rng: random.Random, n_lines: int) -> str:
    return "\n".join(rng.choice(_SONNET_LINES) for _ in range(n_lines))


def _grow_to_target(head: str, rng: random.Random, target: int,
                    line_fmt=lambda s: s) -> str:
    """head 로 시작해서 sonnet 라인 filler 를 붙여 target 토큰까지 키운다.
    토큰 측정은 chunk 단위(64 line)로 해서 비용을 줄인다."""
    text = head
    cur = _ntok(text)
    while cur < target:
        block = "\n".join(line_fmt(rng.choice(_SONNET_LINES)) for _ in range(64))
        text += "\n" + block
        cur = _ntok(text)
    return text


def _build_sonnet(rng: random.Random, target: int, nonce: str) -> str:
    head = (
        f"[doc {nonce}] Read the following Shakespearean passage carefully, then "
        f"continue it for at least 200 lines maintaining iambic pentameter.\n\n"
        + _sample_block(rng, 4)
    )
    return _grow_to_target(head, rng, target)


def _build_chat(rng: random.Random, target: int, nonce: str) -> str:
    head = (
        "<|system|>\nYou are a helpful assistant specialized in literature and "
        f"engineering.\n<|user|>\n(session {nonce}) Please analyze the passage below "
        "in great depth, then answer follow-up questions about meter, theme, and "
        "imagery.\n\n" + _sample_block(rng, 4)
    )
    body = _grow_to_target(head, rng, target - 8)
    return body + "\n<|assistant|>\n"


def _build_code_python(rng: random.Random, target: int, nonce: str) -> str:
    # import ≥2, 주석 ≥10, py keyword ≥3 신호를 맨 앞에 확정 배치
    head = (
        "import os\nimport sys\nimport json\nimport time\nimport re\n"
        f"# module {nonce}\n"
        "# logic line 1\n# logic line 2\n# logic line 3\n# logic line 4\n"
        "# logic line 5\n# logic line 6\n# logic line 7\n# logic line 8\n"
        "# logic line 9\n# logic line 10\n# logic line 11\n# logic line 12\n"
        "def process(items, batch_size=32):\n"
        "    for i in range(0, len(items), batch_size):\n"
        "        batch = items[i:i+batch_size]\n"
        "        if batch:\n"
        "            try:\n"
        "                yield batch\n"
        "            except Exception:\n"
        "                raise\n"
        "class Worker:\n"
        "    def __init__(self, cfg):\n"
        "        self.cfg = cfg\n"
        "    def run(self):\n"
        "        return [x for x in self.cfg if x]\n"
    )
    # code filler: sampled sonnet 을 주석으로 (코드 신호 유지)
    return _grow_to_target(head, rng, target, line_fmt=lambda s: f"# {s}")


def _build_code_rust(rng: random.Random, target: int, nonce: str) -> str:
    """Rust source code variant — use/crate/fn/let mut signal block at head."""
    head = (
        "use std::collections::HashMap;\n"
        "use std::sync::Arc;\n"
        "use std::time::Instant;\n"
        "use serde::{Serialize, Deserialize};\n"
        "use tokio::sync::Mutex;\n"
        f"// module {nonce}\n"
        "// rust crate logic line 1\n// rust crate logic line 2\n"
        "// rust crate logic line 3\n// rust crate logic line 4\n"
        "// rust crate logic line 5\n// rust crate logic line 6\n"
        "// rust crate logic line 7\n// rust crate logic line 8\n"
        "// rust crate logic line 9\n// rust crate logic line 10\n"
        "// rust crate logic line 11\n// rust crate logic line 12\n"
        "pub struct Worker {\n"
        "    pub cfg: Arc<Mutex<HashMap<String, String>>>,\n"
        "}\n"
        "impl Worker {\n"
        "    pub fn new() -> Self {\n"
        "        let mut cfg = HashMap::new();\n"
        "        cfg.insert(String::from(\"role\"), String::from(\"main\"));\n"
        "        Self { cfg: Arc::new(Mutex::new(cfg)) }\n"
        "    }\n"
        "    pub async fn run(&self) -> Result<Vec<u32>, String> {\n"
        "        let mut out = Vec::new();\n"
        "        for i in 0..32 {\n"
        "            out.push(i as u32);\n"
        "        }\n"
        "        Ok(out)\n"
        "    }\n"
        "}\n"
    )
    return _grow_to_target(head, rng, target, line_fmt=lambda s: f"// {s}")


def _build_code_json(rng: random.Random, target: int, nonce: str) -> str:
    """JSON-like dataset variant — repeated key:value structures."""
    head = (
        "{\n"
        f"  \"id\": \"{nonce}\",\n"
        "  \"schema\": \"v1\",\n"
        "  \"created\": \"2026-01-01T00:00:00Z\",\n"
        "  \"updated\": \"2026-06-01T00:00:00Z\",\n"
        "  \"author\": {\"name\": \"system\", \"role\": \"automation\"},\n"
        "  \"tags\": [\"alpha\", \"beta\", \"gamma\", \"delta\"],\n"
        "  \"counts\": {\"prompts\": 500, \"requests\": 256, \"errors\": 0},\n"
        "  \"flags\": {\"enabled\": true, \"debug\": false, \"trace\": true},\n"
        "  \"metrics\": {\"p50\": 11.5, \"p99\": 41.2, \"throughput\": 4.2},\n"
        "  \"notes\": [\n"
    )
    # Treat each filler sonnet line as a JSON string array entry.
    body_iter = []
    def _line_fmt(s: str) -> str:
        # escape any double quotes inside, and trailing comma
        return f"    \"{s.replace(chr(92), chr(92)+chr(92)).replace(chr(34), chr(92)+chr(34))}\","
    body = _grow_to_target(head, rng, target, line_fmt=_line_fmt)
    return body + "\n    \"end\"\n  ]\n}\n"


_CODE_VARIANT_MAP = {
    "python": _build_code_python,
    "rust":   _build_code_rust,
    "json":   _build_code_json,
}


def _build_code(rng: random.Random, target: int, nonce: str) -> str:
    """Code prompt builder with variant selection via env.

    Selects between Python / Rust / JSON corpora to allow the LHC code-workload
    finding (+10.67% on Python in Phase 4 conc256) to be checked for
    generalization across different prefix shapes / token distributions.

    Env: WORKLOAD_CODE_VARIANT in {python (default), rust, json}.
    """
    variant = os.environ.get("WORKLOAD_CODE_VARIANT", "python").lower()
    builder = _CODE_VARIANT_MAP.get(variant, _build_code_python)
    return builder(rng, target, nonce)


_BUILDERS = {"sonnet": _build_sonnet, "chat": _build_chat, "code": _build_code}


def build_prompts(workload: str, num_prompts: int, target_input_len: int,
                  seed: int = 0) -> list[tuple[str, str]]:
    spec = WORKLOAD_DEFINITIONS[workload]
    counts: dict[str, int] = {}
    assigned = 0
    types = list(spec.keys())
    for t in types[:-1]:
        counts[t] = round(spec[t] * num_prompts)
        assigned += counts[t]
    counts[types[-1]] = num_prompts - assigned  # remainder

    items: list[tuple[str, str]] = []
    gidx = 0
    for t, n in counts.items():
        for _ in range(n):
            rng = random.Random(seed * 1_000_000 + gidx)
            nonce = f"{workload}-{gidx:05d}-{rng.randint(0, 1<<30)}"
            prompt = _BUILDERS[t](rng, target_input_len, nonce)
            items.append((t, prompt))
            gidx += 1
    random.Random(seed).shuffle(items)
    return items


async def _request_one(client, url, workload, prompt, max_tokens, sem, model):
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{url}/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": False,
                },
                timeout=httpx.Timeout(3600.0, connect=10.0),
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            ok = resp.status_code == 200
            ctok = ptok = 0
            if ok:
                usage = resp.json().get("usage", {})
                ctok = usage.get("completion_tokens", 0)
                ptok = usage.get("prompt_tokens", 0)
            else:
                return {"workload": workload, "ok": False, "wall_ms": wall_ms,
                        "tokens": 0, "prompt_tokens": 0,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
            return {"workload": workload, "ok": ok, "wall_ms": wall_ms,
                    "tokens": ctok, "prompt_tokens": ptok}
        except Exception as e:  # noqa: BLE001
            return {"workload": workload, "ok": False,
                    "wall_ms": (time.perf_counter() - t0) * 1000.0,
                    "tokens": 0, "prompt_tokens": 0, "error": repr(e)}


async def run(scenario, workload, num_prompts, target_input_len, max_tokens,
              concurrency, seed, out_path, model):
    print(f"[build] workload={workload} n={num_prompts} target_in={target_input_len} ...",
          flush=True)
    items = build_prompts(workload, num_prompts, target_input_len, seed)
    avg_in = sum(_ntok(p) for _, p in items[:20]) / 20
    print(f"[build] done. sample avg input tokens (first 20) = {avg_in:.0f}", flush=True)

    url = BACKEND_URL[scenario]
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=concurrency * 2,
                            max_keepalive_connections=concurrency * 2)
    ) as client:
        t_start = time.perf_counter()
        results = await asyncio.gather(*(
            _request_one(client, url, w, p, max_tokens, sem, model) for w, p in items
        ))
        wall_total_s = time.perf_counter() - t_start

    ok = [r for r in results if r["ok"]]
    n_ok = len(ok)
    total_out = sum(r["tokens"] for r in ok)
    total_in = sum(r["prompt_tokens"] for r in ok)
    out_tps = total_out / wall_total_s if wall_total_s > 0 else 0.0
    walls = sorted(r["wall_ms"] for r in ok)
    p50 = walls[len(walls)//2] if walls else 0.0
    p99 = walls[min(len(walls)-1, int(len(walls)*0.99))] if walls else 0.0

    summary = {
        "scenario": scenario, "workload": workload,
        "num_prompts": len(items), "n_ok": n_ok,
        "target_input_len": target_input_len, "max_tokens": max_tokens,
        "concurrency": concurrency,
        "wall_total_s": wall_total_s,
        "total_input_tokens": total_in, "total_output_tokens": total_out,
        "output_tps": out_tps,
        "req_per_s": n_ok / wall_total_s if wall_total_s > 0 else 0.0,
        "p50_ms": p50, "p99_ms": p99,
        "by_workload_ok": {w: sum(1 for r in ok if r["workload"] == w)
                           for w in ("sonnet", "chat", "code")},
        "n_err": len(results) - n_ok,
        "first_errors": [r.get("error") for r in results if not r["ok"]][:3],
    }
    Path(out_path).write_text(json.dumps({"summary": summary}, indent=2))
    print(f"=== {scenario} × {workload} ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=list(BACKEND_URL.keys()), required=True)
    p.add_argument("--workload", choices=list(WORKLOAD_DEFINITIONS.keys()), required=True)
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--target-input-len", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--sonnet", type=Path,
                   default=Path("benchmarks/sonnet.txt"))
    p.add_argument("--out", default="result.json")
    args = p.parse_args()
    _load_resources(args.sonnet, args.model)
    asyncio.run(run(args.scenario, args.workload, args.num_prompts,
                    args.target_input_len, args.max_tokens, args.concurrency,
                    args.seed, args.out, args.model))


if __name__ == "__main__":
    main()
