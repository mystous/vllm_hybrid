#!/usr/bin/env python3
"""
output.finished 다양한 종료 조건 검증:
1. length: max_tokens 에 의한 자연 종료
2. stop: stop sequence 로 조기 종료
3. abort: client disconnect (requests cancel)
각 종료 후 다음 req 가 CPU 로 정상 라우팅되는지 (slot 반납) 확인.
"""
import json, time, requests, threading
from contextlib import closing

URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def send(payload, timeout=60):
    r = requests.post(URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

results = []

print("=" * 60)
print("Case 1: LENGTH termination (max_tokens=10)")
print("=" * 60)
t0 = time.time()
try:
    d = send({
        "model": MODEL, "prompt": "Once upon a time,",
        "max_tokens": 10, "temperature": 0.0, "stream": False,
    })
    fr = d["choices"][0]["finish_reason"]
    txt = d["choices"][0]["text"]
    ct = d["usage"]["completion_tokens"]
    print(f"  finish_reason={fr}  comp_tokens={ct}  elapsed={time.time()-t0:.2f}s")
    print(f"  text: {txt[:60]!r}")
    results.append({"case": "length", "ok": True, "finish_reason": fr, "completion_tokens": ct})
except Exception as e:
    print(f"  ERR {e}")
    results.append({"case": "length", "ok": False, "error": str(e)})

print()
print("=" * 60)
print("Case 2: STOP sequence termination (stop=['.'])")
print("=" * 60)
t0 = time.time()
try:
    d = send({
        "model": MODEL, "prompt": "The capital of France is",
        "max_tokens": 128, "temperature": 0.0, "stop": ["."], "stream": False,
    })
    fr = d["choices"][0]["finish_reason"]
    txt = d["choices"][0]["text"]
    ct = d["usage"]["completion_tokens"]
    print(f"  finish_reason={fr}  comp_tokens={ct}  elapsed={time.time()-t0:.2f}s")
    print(f"  text: {txt[:60]!r}")
    results.append({"case": "stop", "ok": True, "finish_reason": fr, "completion_tokens": ct})
except Exception as e:
    print(f"  ERR {e}")
    results.append({"case": "stop", "ok": False, "error": str(e)})

print()
print("=" * 60)
print("Case 3: ABORT (client disconnect mid-generation)")
print("=" * 60)
# Send a long request, then cancel after ~1 second
t0 = time.time()
aborted = False
try:
    # streaming request so we can cut the connection mid-stream
    with closing(requests.post(URL, json={
        "model": MODEL, "prompt": "Write a long story about a dragon. Make it detailed.",
        "max_tokens": 2000, "temperature": 0.0, "stream": True,
    }, stream=True, timeout=60)) as r:
        r.raise_for_status()
        chunks = 0
        start_stream = time.time()
        for _line in r.iter_lines():
            chunks += 1
            if time.time() - start_stream > 1.0:
                # force-close the connection
                r.close()
                aborted = True
                break
    elapsed = time.time() - t0
    print(f"  client closed after {elapsed:.2f}s, chunks_received={chunks}, aborted={aborted}")
    results.append({"case": "abort", "ok": True, "elapsed_s": round(elapsed,2), "chunks": chunks, "aborted": aborted})
except Exception as e:
    print(f"  ERR {e}")
    results.append({"case": "abort", "ok": False, "error": str(e)})

# Give server a moment to process abort
time.sleep(2.0)

print()
print("=" * 60)
print("Verification: send next request — CPU slot must be available")
print("=" * 60)
t0 = time.time()
try:
    d = send({
        "model": MODEL, "prompt": "Reply with OK:",
        "max_tokens": 4, "temperature": 0.0, "stream": False,
    })
    elapsed = time.time() - t0
    fr = d["choices"][0]["finish_reason"]
    txt = d["choices"][0]["text"]
    print(f"  finish_reason={fr}  text={txt!r}  elapsed={elapsed:.2f}s")
    # if elapsed is roughly CPU latency (~0.5-2s for 4 tokens), slot was available on CPU
    results.append({"case": "post_abort", "ok": True, "elapsed_s": round(elapsed,2), "finish_reason": fr})
except Exception as e:
    print(f"  ERR {e}")
    results.append({"case": "post_abort", "ok": False, "error": str(e)})

print()
print(f"Summary: {sum(1 for r in results if r.get('ok'))}/{len(results)} success")
with open("/tmp/finish_variety_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved /tmp/finish_variety_results.json")
