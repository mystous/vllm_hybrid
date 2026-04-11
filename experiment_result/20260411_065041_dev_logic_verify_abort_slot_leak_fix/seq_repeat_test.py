#!/usr/bin/env python3
"""
50+ req 순차 반복: 1 req 씩 완료 후 다음 req 송출. CPU slot cycle 누수/영구점유 검증.

- max_tokens 를 작게 (16) 해서 CPU 1 req 가 너무 오래 걸리지 않게.
- 매 req 마다 routing 결정과 completion 결과를 기록.
- router stats log 는 서버 측에서 자동 남음.
"""
import json
import time
import requests
import sys

URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N = 60
MAX_TOKENS = 16  # small to keep CPU req latency manageable

results = []
t0 = time.time()
for i in range(N):
    req_t0 = time.time()
    payload = {
        "model": MODEL,
        "prompt": f"Count from 1 to 5 in English:",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        r = requests.post(URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        out = data["choices"][0]["text"]
        finish = data["choices"][0].get("finish_reason")
        usage = data.get("usage", {})
        elapsed = time.time() - req_t0
        results.append({
            "i": i,
            "ok": True,
            "elapsed_s": round(elapsed, 3),
            "finish_reason": finish,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "out": out[:60],
        })
        print(f"[{i:3d}] ok elapsed={elapsed:6.2f}s finish={finish} comp={usage.get('completion_tokens')}")
    except Exception as e:
        results.append({"i": i, "ok": False, "error": str(e)})
        print(f"[{i:3d}] ERR {e}")
        break
    # small gap between requests so router stats can emit in between
    time.sleep(0.05)

total = time.time() - t0
print(f"\nTotal: {total:.2f}s for {len(results)} reqs, avg={total/max(1,len(results)):.2f}s/req")
print(f"Success: {sum(1 for r in results if r.get('ok'))}/{len(results)}")

with open("/tmp/seq_results.json", "w") as f:
    json.dump({"n": N, "total_s": round(total, 3), "results": results}, f, indent=2)
print("saved /tmp/seq_results.json")
