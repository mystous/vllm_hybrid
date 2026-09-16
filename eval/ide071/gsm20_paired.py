#!/usr/bin/env python3
"""IDE_072 — GSM20_PAIRED: GSM8K test 앞 40문항(원본 GSM40 = gsm_eval.py df.head(40), ID = 행 index 0..39) 을 정렬한 뒤 앞 20개 (index 0..19).
greedy (temperature 0), chat template (gsm_eval.py 와 동일 user prompt), max_tokens 1024, 동시성 4, 문항별 반복 없음.
저장: 문항 index, 질문 sha256, 정답, 전체 생성 텍스트, 추출 답, 정오, finish_reason, usage(completion_tokens), truncation(finish_reason=='length'), 오류 원문.
사용: gsm20_paired.py <out.json> <model>"""
import json, sys, re, urllib.request, hashlib, time, concurrent.futures as cf
out_f, MODEL = sys.argv[1], sys.argv[2]
PQ = "/home/mystous/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet"
import pyarrow.parquet as pq
rows = pq.read_table(PQ).slice(0, 40).to_pylist(); rows = [dict(r, idx=i) for i, r in enumerate(rows)]
sel = sorted(rows, key=lambda r: r["idx"])[:20]
PROMPT = "{q}\n\nSolve step by step, then give the final numeric answer on the last line as: #### <number>"
def gold(ans): return ans.split("####")[-1].strip().replace(",", "")
def extract(t):
    m = re.findall(r"####\s*(-?[\d,]+\.?\d*)", t)
    if m: return m[-1].replace(",", "").rstrip(".")
    m = re.findall(r"-?[\d,]+\.?\d*", t.replace(",", ""))
    return m[-1].rstrip(".") if m else None
def ask(r):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(q=r["question"])}], "max_tokens": 1024, "temperature": 0}
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
        j = json.loads(resp.read()); ch = j["choices"][0]
        return {"text": ch["message"]["content"], "finish_reason": ch.get("finish_reason"), "usage": j.get("usage"), "error": None, "seconds": time.time() - t0}
    except Exception as e:
        return {"text": None, "finish_reason": None, "usage": None, "error": str(e)[:500], "seconds": time.time() - t0}
res = []
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(ask, r): r for r in sel}
    for f in cf.as_completed(futs):
        r = futs[f]; a = f.result(); g = gold(r["answer"]); pred = extract(a["text"]) if a["text"] else None
        res.append({"idx": r["idx"], "question_sha256": hashlib.sha256(r["question"].encode()).hexdigest(), "gold": g, "pred": pred, "correct": (pred == g) if pred is not None else None,
                    "text": a["text"], "finish_reason": a["finish_reason"], "completion_tokens": (a["usage"] or {}).get("completion_tokens"), "truncated": a["finish_reason"] == "length", "error": a["error"], "seconds": a["seconds"], "token_ids": None})
res.sort(key=lambda x: x["idx"])
summary = {"model": MODEL, "n": len(res), "correct": sum(1 for x in res if x["correct"]), "incorrect": sum(1 for x in res if x["correct"] is False), "unscored": sum(1 for x in res if x["correct"] is None),
           "truncated": sum(1 for x in res if x["truncated"]), "errors": sum(1 for x in res if x["error"]), "max_tokens": 1024, "concurrency": 4, "temperature": 0,
           "selection": "GSM8K test rows 0..39 (gsm_eval.py head(40)) sorted by index → first 20", "prompt_template_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
           "harness_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "dataset": PQ, "token_ids": "null (OpenAI chat API 는 token id 미제공)", "results": res}
json.dump(summary, open(out_f, "w"), indent=1, ensure_ascii=False)
print(f"GSM20 {summary['correct']}/{summary['n']} correct, {summary['unscored']} unscored, {summary['truncated']} truncated, {summary['errors']} errors")
