#!/usr/bin/env python3
"""분포 수준 동등성 (CLAUDE.md binding 지표): GSM8K 앞 N문항의 gold 풀이를 teacher-forcing 으로 채점해 per-token logprob 저장.
usage: lp_compare.py <out.json> <N>     (서버 127.0.0.1:30000, echo+logprobs)
비교: lp_compare.py --cmp a.json b.json  → max abs diff (답 구간), PPL relative diff, 토큰 일치율(argmax 은 미수집이라 생략)"""
import json, sys, math, urllib.request, concurrent.futures as cf
if sys.argv[1] == "--cmp":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    worst = 0.0; ppl_a = []; ppl_b = []; n_tok = 0
    for ra, rb in zip(a["rows"], b["rows"]):
        la, lb = ra["lp"], rb["lp"]
        assert len(la) == len(lb), (len(la), len(lb))
        s = ra["ans_start"]
        for x, y in zip(la[s:], lb[s:]):
            if x is None or y is None: continue
            worst = max(worst, abs(x - y)); n_tok += 1
        ppl_a.append(math.exp(-sum(v for v in la[s:] if v is not None) / max(1, len(la) - s)))
        ppl_b.append(math.exp(-sum(v for v in lb[s:] if v is not None) / max(1, len(lb) - s)))
    rel = max(abs(x - y) / x for x, y in zip(ppl_a, ppl_b))
    mean_rel = sum(abs(x - y) / x for x, y in zip(ppl_a, ppl_b)) / len(ppl_a)
    print(f"LPCMP prompts={len(a['rows'])} answer_tokens={n_tok} max_abs_logprob_diff={worst:.4f} worst_ppl_rel_diff={rel:.4f} mean_ppl_rel_diff={mean_rel:.4f} mean_ppl_a={sum(ppl_a)/len(ppl_a):.4f} mean_ppl_b={sum(ppl_b)/len(ppl_b):.4f}")
    raise SystemExit
out_f, N = sys.argv[1], int(sys.argv[2])
import pyarrow.parquet as pq
t = pq.read_table("/home/mystous/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet")
rows = t.slice(0, N).to_pylist()
def score(r):
    q = f"Question: {r['question']}\nAnswer:"; full = q + " " + r["answer"]
    req = json.dumps({"model": "q480", "prompt": full, "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1}).encode()
    for _ in range(3):
        try:
            resp = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30000/v1/completions", req, {"Content-Type": "application/json"}), timeout=300)
            d = json.loads(resp.read()); break
        except Exception as e:
            err = e
    else:
        raise err
    lp = d["choices"][0]["logprobs"]; toks = lp["tokens"]; tlp = lp["token_logprobs"]
    # 답 구간 시작 = prompt 텍스트 길이에 해당하는 토큰 인덱스 (text_offset 사용)
    offs = lp.get("text_offset")
    ans_start = next((i for i, o in enumerate(offs) if o >= len(q)), 0) if offs else 0
    n_prompt = len(toks) - 1  # 마지막 1개는 생성 토큰 → 제외
    return {"lp": tlp[:n_prompt], "ans_start": ans_start, "n": n_prompt}
with cf.ThreadPoolExecutor(2) as ex: res = list(ex.map(score, rows))
json.dump({"n": N, "rows": res}, open(out_f, "w"))
print("saved", out_f, "tokens", sum(r["n"] for r in res))
