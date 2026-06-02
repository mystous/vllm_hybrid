"""TSK_042 prompt sampler — corpus_loader → 샘플 prompt 셋 (CPU only).

각 corpus 에서 dedup(sha1) + length filter(32≤tok≤4096) + langdetect 후
corpus당 N 개 샘플 → sampled_prompts.parquet (+ corpus_stats 출력).

사용:
  python prompt_sampler.py --corpora open --n 50 --out /tmp/sampled.parquet
  python prompt_sampler.py --corpora all  --n 500 --out runs/.../sampled_prompts.parquet
"""

from __future__ import annotations

import argparse
import hashlib

import pyarrow as pa
import pyarrow.parquet as pq

from vllm_config_perf.gating.realistic_eval import corpus_loader as cl

_TOK = None


def _ntok(text: str) -> int:
    return len(_TOK(text, add_special_tokens=False)["input_ids"])


def _norm_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode("utf-8", "ignore")).hexdigest()


def _lang(text: str, native):
    if native:
        return str(native)
    try:
        from langdetect import detect
        return detect(text[:2000])
    except Exception:  # noqa: BLE001
        return "unknown"


def sample_corpus(name, n, min_tok, max_tok):
    rows, seen = [], set()
    for rec in cl.iter_corpus(name, limit=n):
        text = rec["raw_text"]
        h = _norm_hash(text)
        if h in seen:
            continue
        nt = _ntok(text)
        if nt < min_tok or nt > max_tok:
            continue
        seen.add(h)
        rows.append({
            "prompt_id": f"{name}-{len(rows):05d}",
            "prompt_hash": h,
            "corpus": name,
            "lang": _lang(text, rec.get("native_lang")),
            "n_input_tok": nt,
            "raw_text": text,
        })
        if len(rows) >= n:
            break
    return rows


def main():
    global _TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="open", help="open / all / 콤마구분 목록")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-tok", type=int, default=32)
    ap.add_argument("--max-tok", type=int, default=4096)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--out", default="/tmp/sampled_prompts.parquet")
    args = ap.parse_args()

    if args.corpora == "open":
        names = cl.OPEN
    elif args.corpora == "all":
        names = cl.ALL
    else:
        names = args.corpora.split(",")

    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(args.model)

    all_rows = []
    print(f"# corpus stats (n={args.n}, tok {args.min_tok}-{args.max_tok})")
    print("| corpus | sampled | tok p50 | tok max | langs(top3) |")
    print("|---|---:|---:|---:|---|")
    for nm in names:
        try:
            rows = sample_corpus(nm, args.n, args.min_tok, args.max_tok)
        except Exception as e:  # noqa: BLE001
            print(f"| {nm} | ERR | — | — | {type(e).__name__}: {str(e)[:40]} |")
            continue
        all_rows.extend(rows)
        toks = sorted(r["n_input_tok"] for r in rows)
        from collections import Counter
        langs = Counter(r["lang"] for r in rows).most_common(3)
        p50 = toks[len(toks) // 2] if toks else 0
        print(f"| {nm} | {len(rows)} | {p50} | {toks[-1] if toks else 0} | "
              f"{', '.join(f'{l}:{c}' for l, c in langs)} |")

    if all_rows:
        table = pa.Table.from_pylist(all_rows)
        pq.write_table(table, args.out)
        print(f"\n[sampler] wrote {len(all_rows)} prompts → {args.out}")
    else:
        print("\n[sampler] no rows sampled")


if __name__ == "__main__":
    main()
