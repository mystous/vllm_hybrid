#!/usr/bin/env python3
"""IDE_075 §7.1 — barrier 클라이언트 (컨테이너 vllm-h100 에서 실행). vllm bench serve 와 같은 요청 의미:
/v1/completions, raw prompt(custom jsonl 의 prompt), max_tokens=<out>, temperature 0, top_p 1, ignore_eos, stream, max_concurrency C, 요청 순서 = 파일 순서(seed 고정: vllm bench 의 custom 은 순서 유지).
상태 기계: CLIENT_INIT → (입력·연결 준비) → CLIENT_READY(파일 <ctl>.ready 생성) → START_SIGNAL(<ctl>.start 파일 대기) → 전송 → LAST_RESPONSE → 결과 저장.
요청별 timestamp(ns, 컨테이너 host 시계 = 같은 호스트): t_send, t_first_token, t_end, 토큰 수(usage 없으면 chunk 수). 요약은 vllm bench 와 같은 정의(completed, duration, output_throughput, TTFT/TPOT/ITL/E2EL p50/p95/p99).
사용: probe_client.py <jsonl> <n> <C> <out_tokens> <model> <ctl_prefix> <result_json>"""
import asyncio, json, sys, time, os, aiohttp, statistics as st
jsonl, n, C, out_tok, model, ctl, res_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
URL = "http://127.0.0.1:30000/v1/completions"


def pct(v, q): v = sorted(v); return v[int(round(q * (len(v) - 1)))] if v else None


async def one(session, sem, i, prompt, ev):
    async with sem:
        body = {"model": model, "prompt": prompt, "max_tokens": out_tok, "temperature": 0, "top_p": 1.0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
        r = {"request_id": i, "t_send_ns": time.time_ns(), "t_first_token_ns": None, "t_end_ns": None, "chunks": 0, "output_tokens": None, "error": None, "itl_ns": []}
        prev = None
        try:
            async with session.post(URL, json=body) as resp:
                async for line in resp.content:
                    if not line.startswith(b"data:"): continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]": break
                    now = time.time_ns()
                    try: j = json.loads(payload)
                    except Exception: continue
                    if j.get("usage"): r["output_tokens"] = j["usage"].get("completion_tokens")
                    ch = j.get("choices") or []
                    if ch and ch[0].get("text"):
                        r["chunks"] += 1
                        if r["t_first_token_ns"] is None: r["t_first_token_ns"] = now
                        elif prev: r["itl_ns"].append(now - prev)
                        prev = now
                r["t_end_ns"] = time.time_ns(); r["http"] = resp.status
        except Exception as e: r["error"] = str(e); r["t_end_ns"] = time.time_ns()
        ev.append(r)


async def main():
    prompts = [json.loads(l)["prompt"] for l in open(jsonl)][:n]
    conn = aiohttp.TCPConnector(limit=C); timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        async with session.get("http://127.0.0.1:30000/health") as h: assert h.status == 200   # 준비 확인 (본 요청 아님, 별도 기록)
        t_ready = time.time_ns(); open(ctl + ".ready", "w").write(str(t_ready))
        while not os.path.exists(ctl + ".start"): await asyncio.sleep(0.005)
        t_start_sig = time.time_ns(); sem = asyncio.Semaphore(C); ev = []
        t0 = time.time_ns(); await asyncio.gather(*[one(session, sem, i, p, ev) for i, p in enumerate(prompts)]); t1 = time.time_ns()
    ev.sort(key=lambda r: r["request_id"]); ok = [r for r in ev if not r["error"] and r["t_first_token_ns"]]
    dur = (t1 - t0) / 1e9; outs = sum((r["output_tokens"] or r["chunks"]) for r in ok)
    ttft = [(r["t_first_token_ns"] - r["t_send_ns"]) / 1e6 for r in ok]; e2e = [(r["t_end_ns"] - r["t_send_ns"]) / 1e6 for r in ok]
    tpot = [((r["t_end_ns"] - r["t_first_token_ns"]) / 1e6) / max(1, (r["output_tokens"] or r["chunks"]) - 1) for r in ok]; itl = [x / 1e6 for r in ok for x in r["itl_ns"]]
    summary = {"completed": len(ok), "failed": len(ev) - len(ok), "duration": dur, "total_output_tokens": outs, "output_throughput": outs / dur if dur else None, "median_ttft_ms": pct(ttft, .5), "p95_ttft_ms": pct(ttft, .95), "p99_ttft_ms": pct(ttft, .99), "median_tpot_ms": pct(tpot, .5), "p95_tpot_ms": pct(tpot, .95), "p99_tpot_ms": pct(tpot, .99), "median_itl_ms": pct(itl, .5), "p95_itl_ms": pct(itl, .95), "median_e2el_ms": pct(e2e, .5), "p95_e2el_ms": pct(e2e, .95),
               "t_client_ready_ns": t_ready, "t_start_signal_seen_ns": t_start_sig, "t_first_request_sent_ns": min(r["t_send_ns"] for r in ev), "t_last_response_ns": max(r["t_end_ns"] for r in ev), "client": "probe_client.py (aiohttp, stream)", "C": C, "n": n, "out_tokens": out_tok, "clock": "CLOCK_REALTIME ns (컨테이너, 같은 호스트)"}
    json.dump({"summary": summary, "requests": [{k: v for k, v in r.items() if k != "itl_ns"} for r in ev]}, open(res_path, "w")); print(json.dumps(summary))


asyncio.run(main())
