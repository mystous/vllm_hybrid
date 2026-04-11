#!/usr/bin/env python3
"""
CPU slot 점유 중인 request 를 client disconnect 로 abort 시키고 slot 반납 확인.
"""
import json, time, requests, threading
from contextlib import closing

URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def send_blocking(payload, timeout=60):
    r = requests.post(URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

print("Step 1: Wait for CPU slot free, then send long CPU req (streaming + cancel)...")
# drain any pending
time.sleep(1.0)

# send a long request, expect cpu-first to send to CPU
t_start = time.time()
aborted = False
chunks = 0
try:
    with closing(requests.post(URL, json={
        "model": MODEL, "prompt": "Write a long poem about the ocean, with many verses.",
        "max_tokens": 500, "temperature": 0.0, "stream": True,
    }, stream=True, timeout=120)) as r:
        r.raise_for_status()
        stream_start = time.time()
        for line in r.iter_lines():
            chunks += 1
            # cancel after 2 seconds of streaming (should be mid-generation)
            if time.time() - stream_start > 2.0:
                r.close()
                aborted = True
                break
    elapsed = time.time() - t_start
    print(f"  long req: chunks={chunks} aborted={aborted} elapsed={elapsed:.2f}s")
except Exception as e:
    print(f"  ERR {e}")

print()
print("Step 2: Probe requests to check CPU slot release timing...")
# After abort, repeatedly probe whether CPU becomes available
probes = []
for i in range(10):
    pt = time.time()
    try:
        d = send_blocking({
            "model": MODEL, "prompt": "Say OK.",
            "max_tokens": 4, "temperature": 0.0,
        }, timeout=30)
        el = time.time() - pt
        fr = d["choices"][0]["finish_reason"]
        # rough heuristic: elapsed ~0.4-1.0s → CPU; ~0.05s → GPU
        backend = "CPU?" if el > 0.3 else "GPU?"
        print(f"  probe {i}: elapsed={el:.3f}s finish={fr} -> {backend}")
        probes.append({"i": i, "elapsed_s": round(el, 3), "finish_reason": fr, "backend_guess": backend})
    except Exception as e:
        print(f"  probe {i}: ERR {e}")
        probes.append({"i": i, "error": str(e)})
    time.sleep(0.2)

with open("/tmp/cpu_abort_results.json", "w") as f:
    json.dump({"long_aborted": aborted, "long_chunks": chunks, "probes": probes}, f, indent=2)
print("\nsaved /tmp/cpu_abort_results.json")
