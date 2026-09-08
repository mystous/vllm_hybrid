import json, sys, re, urllib.request, concurrent.futures as cf
# usage: gsm_eval.py <out.json> <n>
out_f, N = sys.argv[1], int(sys.argv[2])
try:
    import pandas as pd
    df = pd.read_parquet("/home/mystous/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet")
    rows = df.head(N).to_dict("records")
except Exception:
    import pyarrow.parquet as pq
    t = pq.read_table("/home/mystous/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet")
    rows = t.slice(0, N).to_pylist()
def gold(ans): return ans.split("####")[-1].strip().replace(",", "")
def ask(q):
    msgs = [{"role": "user", "content": f"{q}\n\nSolve step by step, then give the final numeric answer on the last line as: #### <number>"}]
    req = json.dumps({"model": "q480", "messages": msgs, "max_tokens": 768, "temperature": 0}).encode()
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30000/v1/chat/completions", req, {"Content-Type": "application/json"}), timeout=1200)
    return json.loads(r.read())["choices"][0]["message"]["content"]
def extract(t):
    m = re.findall(r"####\s*(-?[\d,]+\.?\d*)", t)
    if m: return m[-1].replace(",", "").rstrip(".")
    m = re.findall(r"-?[\d,]+\.?\d*", t.replace(",", ""))
    return m[-1].rstrip(".") if m else None
res = []
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(ask, r["question"]): r for r in rows}
    for f in cf.as_completed(futs):
        r = futs[f]
        try: txt = f.result()
        except Exception as e: txt = f"ERR {e}"
        g = gold(r["answer"]); pred = extract(txt)
        res.append({"gold": g, "pred": pred, "ok": pred == g, "text_tail": txt[-120:]})
acc = sum(1 for x in res if x["ok"]) / len(res)
json.dump({"n": len(res), "acc": acc, "results": res}, open(out_f, "w"))
print(f"ACC {acc:.3f} ({sum(1 for x in res if x['ok'])}/{len(res)})")
