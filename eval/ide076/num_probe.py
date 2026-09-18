#!/usr/bin/env python3
"""IDE_076 §8.10 B 수치 검증 프로브 (분포 수준): 살아 있는 서버에 고정 텍스트를 teacher-forcing(echo=true, max_tokens=1, logprobs=1) 으로 보내
프롬프트 토큰 logprob 을 받아 PPL 을 계산하고, 짧은 고정 프롬프트의 greedy 생성 토큰(32) 을 기록한다. 부팅 간 비교는 num_compare().
사용: num_probe.py --port 30000 --model qwen480 --out <json> [--texts <txt 파일; 빈 줄로 구분>]
비교: num_probe.py compare <ref.json> <cand.json>  → PPL 상대차, 토큰별 logprob max/mean abs diff, greedy 토큰 일치율
호스트 동기화·성능 경로와 무관(세션 전에만 실행)."""
import sys, json, argparse, math, time, urllib.request, os
DEFAULT_TEXTS = os.path.expanduser("~/projects/vllm_hybrid/eval/ide076/num_probe_texts.txt")
GREEDY_PROMPTS = ["The capital of France is", "def fibonacci(n):\n    ", "In 1969, humans first landed on the Moon. The mission was called",
                  "Water boils at", "The quick brown fox", "SELECT name FROM users WHERE", "Once upon a time, in a small village,", "The derivative of x^2 is"]


def post(port, payload, timeout=120):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())


def probe(port, model, texts):
    out = {"t": time.time(), "port": port, "model": model, "texts": [], "greedy": []}
    for i, t in enumerate(texts):
        r = post(port, {"model": model, "prompt": t, "max_tokens": 1, "echo": True, "logprobs": 1, "temperature": 0})
        lp = r["choices"][0]["logprobs"]; tl = lp.get("token_logprobs") or []; toks = lp.get("tokens") or []
        vals = [v for v in tl if v is not None]
        out["texts"].append({"i": i, "n_tokens": len(vals), "sum_logprob": sum(vals), "ppl": math.exp(-sum(vals) / max(len(vals), 1)), "token_logprobs": tl, "tokens": toks})
    for i, p in enumerate(GREEDY_PROMPTS):
        r = post(port, {"model": model, "prompt": p, "max_tokens": 32, "temperature": 0, "logprobs": 1})
        lp = r["choices"][0].get("logprobs") or {}
        out["greedy"].append({"i": i, "text": r["choices"][0]["text"], "tokens": lp.get("tokens"), "token_logprobs": lp.get("token_logprobs")})
    return out


def compare(ref, cand):
    res = {"texts": [], "greedy": []}
    for a, b in zip(ref["texts"], cand["texts"]):
        # echo + max_tokens=1 이므로 마지막 토큰은 '생성' 토큰(부팅마다 다를 수 있음) → 프롬프트 토큰만 비교 (teacher-forcing)
        ta, tb = a["token_logprobs"][:-1], b["token_logprobs"][:-1]
        same_len = len(ta) == len(tb) and (a.get("tokens") or [])[:-1] == (b.get("tokens") or [])[:-1]
        va = [v for v in ta if v is not None]; vb = [v for v in tb if v is not None]
        ppl_a = math.exp(-sum(va) / max(len(va), 1)); ppl_b = math.exp(-sum(vb) / max(len(vb), 1))
        diffs = [abs(x - y) for x, y in zip(ta, tb) if x is not None and y is not None] if same_len else []
        res["texts"].append({"i": a["i"], "n_tokens": len(va), "ppl_ref": ppl_a, "ppl_cand": ppl_b, "ppl_rel_diff": ppl_b / ppl_a - 1 if ppl_a else None,
                             "same_tokenization": same_len, "logprob_abs_diff_max": max(diffs) if diffs else None, "logprob_abs_diff_mean": (sum(diffs) / len(diffs)) if diffs else None})
    for a, b in zip(ref["greedy"], cand["greedy"]):
        ta, tb = a.get("tokens") or [], b.get("tokens") or []
        n = min(len(ta), len(tb)); first_div = next((k for k in range(n) if ta[k] != tb[k]), None)
        res["greedy"].append({"i": a["i"], "identical_text": a["text"] == b["text"], "first_divergence_pos": first_div, "n_ref": len(ta), "n_cand": len(tb)})
    rel = [x["ppl_rel_diff"] for x in res["texts"] if x["ppl_rel_diff"] is not None]; mx = [x["logprob_abs_diff_max"] for x in res["texts"] if x["logprob_abs_diff_max"] is not None]; mn = [x["logprob_abs_diff_mean"] for x in res["texts"] if x["logprob_abs_diff_mean"] is not None]
    res["summary"] = {"ppl_rel_diff_mean": sum(rel) / len(rel) if rel else None, "ppl_rel_diff_max_abs": max(abs(v) for v in rel) if rel else None, "logprob_abs_diff_max": max(mx) if mx else None, "logprob_abs_diff_mean": sum(mn) / len(mn) if mn else None,
                      "greedy_identical": sum(1 for g in res["greedy"] if g["identical_text"]), "greedy_n": len(res["greedy"]), "same_tokenization_all": all(x["same_tokenization"] for x in res["texts"])}
    return res


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        r = compare(json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))); print(json.dumps(r["summary"], indent=1))
        if len(sys.argv) > 4: json.dump(r, open(sys.argv[4], "w"), indent=1)
        return
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=30000); ap.add_argument("--model", default="qwen480"); ap.add_argument("--out", required=True); ap.add_argument("--texts", default=DEFAULT_TEXTS)
    a = ap.parse_args(); texts = [t.strip() for t in open(a.texts).read().split("\n\n") if t.strip()]
    out = probe(a.port, a.model, texts); json.dump(out, open(a.out, "w")); print("num_probe:", len(out["texts"]), "texts,", len(out["greedy"]), "greedy prompts →", a.out)


if __name__ == "__main__": main()
