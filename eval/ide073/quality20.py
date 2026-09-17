#!/usr/bin/env python3
"""IDE_073 — 동일 20문항 출력 비교 (§9). 입력 quality_requests.jsonl (GSM8K rows 0..19), chat API, temperature 0, top_p 1, max_tokens 1024, 동시성 4.
GLM: chat_template_kwargs enable_thinking=false (세 비교군 동일). 문항별 전문·추출답·정오·finish_reason·usage·오류 저장.
사용: quality20.py <questions.jsonl> <model_name> <out.json> [--thinking-off]"""
import json, sys, re, urllib.request, hashlib, time, concurrent.futures as cf
qf, MODEL, out_f = sys.argv[1], sys.argv[2], sys.argv[3]; thinking_off = "--thinking-off" in sys.argv
rows = [json.loads(l) for l in open(qf)]
PROMPT = "{q}\n\nSolve step by step, then give the final numeric answer on the last line as: #### <number>"
def extract(t):
    m = re.findall(r"####\s*(-?[\d,]+\.?\d*)", t)
    if m: return m[-1].replace(",", "").rstrip(".")
    m = re.findall(r"-?[\d,]+\.?\d*", t.replace(",", ""))
    return m[-1].rstrip(".") if m else None
def ask(r):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(q=r["question"])}], "max_tokens": 1024, "temperature": 0, "top_p": 1.0}
    if thinking_off: body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900)
        j = json.loads(resp.read()); ch = j["choices"][0]; msg = ch["message"]
        return {"text": msg.get("content"), "reasoning": msg.get("reasoning_content"), "finish_reason": ch.get("finish_reason"), "usage": j.get("usage"), "error": None, "seconds": time.time() - t0, "http": 200}
    except urllib.error.HTTPError as e:
        return {"text": None, "reasoning": None, "finish_reason": None, "usage": None, "error": f"HTTP {e.code}: {e.read()[:300]}", "seconds": time.time() - t0, "http": e.code}
    except Exception as e:
        return {"text": None, "reasoning": None, "finish_reason": None, "usage": None, "error": str(e)[:300], "seconds": time.time() - t0, "http": None}
res = []
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(ask, r): r for r in rows}
    for f in cf.as_completed(futs):
        r = futs[f]; a = f.result(); pred = extract(a["text"]) if a["text"] else None
        res.append({"qid": r["qid"], "question_hash": r["question_hash"], "gold": r["gold"], "pred": pred, "exact_match": (pred == r["gold"]) if pred is not None else None, "text": a["text"], "reasoning_content": a["reasoning"],
                    "finish_reason": a["finish_reason"], "completion_tokens": (a["usage"] or {}).get("completion_tokens"), "prompt_tokens": (a["usage"] or {}).get("prompt_tokens"), "truncated": a["finish_reason"] == "length", "error": a["error"], "http": a["http"], "seconds": a["seconds"], "token_ids": None})
res.sort(key=lambda x: x["qid"])
s = {"model": MODEL, "n": len(res), "correct": sum(1 for x in res if x["exact_match"]), "incorrect": sum(1 for x in res if x["exact_match"] is False), "unscored": sum(1 for x in res if x["exact_match"] is None), "truncated": sum(1 for x in res if x["truncated"]), "errors": sum(1 for x in res if x["error"]),
     "generation": {"temperature": 0, "top_p": 1.0, "max_tokens": 1024, "concurrency": 4, "endpoint": "/v1/chat/completions", "chat_template_kwargs": {"enable_thinking": False} if thinking_off else None, "ignore_eos": False},
     "prompt_template_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), "questions_sha256": hashlib.sha256(open(qf, "rb").read()).hexdigest(), "harness_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "token_ids": "null (chat API 미제공)", "results": res}
json.dump(s, open(out_f, "w"), indent=1, ensure_ascii=False)
print(f"Q20 {s['correct']}/{s['n']} exact, unscored {s['unscored']}, truncated {s['truncated']}, errors {s['errors']}")
