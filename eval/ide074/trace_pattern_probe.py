#!/usr/bin/env python3
"""IDE_074 M3 사전 조사 — 기존 D2 retry2 TP-0 트레이스에서 graph id / graph node id / memcpy 패턴으로
층 단위 분절이 가능한지 확인한다 (추정으로 이벤트를 만들지 않음; 패턴 존재 여부와 계수만 기록)."""
import gzip, json, collections, sys, os
tf = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/huggingface/kt/ide073/profiles/qwen/D2_1789610161-TP-0.trace.json.gz")   # IDE_075: 인자 지원
j = json.load(gzip.open(tf)); base = j.get("baseTimeNanoseconds", 0) / 1e3
ev = [e for e in j["traceEvents"] if e.get("ph") == "X"]
ops = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
ann = [e for e in ev if e.get("cat") == "gpu_user_annotation"]
print("ops", len(ops), "annotations", len(ann))
# graph id 별 op 수·memcpy 수
by_graph = collections.defaultdict(lambda: collections.Counter())
for e in ops:
    g = e.get("args", {}).get("graph id"); by_graph[g][e["cat"]] += 1
    if e["cat"] == "gpu_memcpy": by_graph[g][e["name"][:22]] += 1
print("graph ids:", len(by_graph))
for g, c in sorted(by_graph.items(), key=lambda kv: -sum(kv[1].values()))[:8]: print(" graph", g, dict(c))
# graph node id 범위 (graph id 별) → 한 replay 의 노드 수
nodes = collections.defaultdict(set)
for e in ops:
    a = e.get("args", {})
    if a.get("graph id", 0): nodes[a["graph id"]].add(a.get("graph node id"))
for g, s in sorted(nodes.items())[:8]: print(" graph", g, "distinct node ids", len(s), "min/max", min(s), max(s))
# step annotation 이름 분포
print("annotation names:", collections.Counter(e["name"][:30] for e in ann).most_common(12))
# replay 분절: graph id 별로 node id 가 감소하는 지점 = 새 replay 시작
ops.sort(key=lambda e: e["ts"])
replays = collections.Counter(); last_node = {}
for e in ops:
    a = e.get("args", {}); g = a.get("graph id", 0)
    if not g: continue
    n = a.get("graph node id", 0)
    if g in last_node and n <= last_node[g]: replays[g] += 1
    last_node[g] = n
print("replay 경계 수 (graph id 별, node id 감소 기준):", dict(replays))
# 한 replay 안에서 HtoD memcpy 수 (가장 큰 graph 1개, 첫 replay)
g0 = max(by_graph, key=lambda k: sum(by_graph[k].values()) if k else 0)
seq = [e for e in ops if e.get("args", {}).get("graph id") == g0]
first = []; prev = -1
for e in seq:
    n = e["args"].get("graph node id", 0)
    if n <= prev and first: break
    first.append(e); prev = n
h2d = [e for e in first if e["cat"] == "gpu_memcpy" and "HtoD" in e["name"]]; d2h = [e for e in first if e["cat"] == "gpu_memcpy" and "DtoH" in e["name"]]
moe = [e for e in first if "fused_moe" in e["name"]]
print(f"graph {g0} 첫 replay: ops {len(first)} HtoD {len(h2d)} DtoH {len(d2h)} fused_moe {len(moe)}")
print(" HtoD bytes:", collections.Counter(e["args"].get("bytes") for e in h2d).most_common(4), " DtoH bytes:", collections.Counter(e["args"].get("bytes") for e in d2h).most_common(5))
# 층 패턴: HtoD 를 기준으로 그 직전 커널과의 간격 (GPU 유휴 = cold 대기 노출 후보) 분포 — 첫 replay
gaps = []
for i, e in enumerate(first):
    if e["cat"] == "gpu_memcpy" and "HtoD" in e["name"] and i > 0:
        p = first[i - 1]; gaps.append((e["ts"] - (p["ts"] + p["dur"]), p["name"][:40]))
print(" HtoD 직전 op 와의 gap µs (첫 replay, 앞 10개):", [(round(g, 1), n) for g, n in gaps[:10]])
print(" 직전 op 이름 분포:", collections.Counter(n for g, n in gaps).most_common(5))
# HtoD 직후 op
after = collections.Counter()
for i, e in enumerate(first[:-1]):
    if e["cat"] == "gpu_memcpy" and "HtoD" in e["name"]: after[first[i + 1]["name"][:50]] += 1
print(" HtoD 직후 op 이름:", after.most_common(4))
