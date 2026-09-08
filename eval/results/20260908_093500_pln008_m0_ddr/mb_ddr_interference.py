#!/usr/bin/env python3
"""M0 파라미터: 소켓별 DDR 읽기 대역폭과 2-스트림(expert 스트리밍 + KV 읽기 모사) 동시 사용 간섭.

- 스트림 A: numa 노드 n 의 메모리를 t_a 스레드로 순차 읽기 (expert 가중치 스트리밍 모사)
- 스트림 B: 같은/다른 numa 노드 메모리를 t_b 스레드로 순차 읽기 (HiCache KV 읽기 모사)
각각 단독·동시 실행의 GB/s 를 재서 간섭 계수 = 동시합 / 단독합 을 낸다.
usage: mb_ddr_interference.py <out_json>   (컨테이너 내, numactl 필요 없음 — torch 텐서를 numa 노드에 바인딩)
"""
import json, os, sys, time, threading, ctypes
import numpy as np

OUT = sys.argv[1]
GB = 1 << 30
SIZE = 4 * GB  # 스트림당 4GB (L3 105MB×2 를 압도)

libnuma = ctypes.CDLL("libnuma.so.1")
libnuma.numa_alloc_onnode.restype = ctypes.c_void_p
libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
libnuma.numa_run_on_node.argtypes = [ctypes.c_int]

def alloc_on_node(nbytes, node):
    p = libnuma.numa_alloc_onnode(nbytes, node)
    assert p, "numa_alloc failed"
    buf = (ctypes.c_uint8 * nbytes).from_address(p)
    arr = np.frombuffer(buf, dtype=np.float32)
    arr[:] = 1.0  # first-touch on node
    return arr

def read_stream(arr, threads, node, reps, result, key):
    """threads 개의 스레드가 arr 를 조각내어 합산 (읽기 전용). 각 스레드는 node 에 고정."""
    n = arr.size; chunk = n // threads
    def worker(i):
        libnuma.numa_run_on_node(node)
        s = 0.0
        for _ in range(reps):
            s += float(arr[i*chunk:(i+1)*chunk].sum())
        result[key + f"_t{i}"] = s
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    dt = time.perf_counter() - t0
    return arr.nbytes * reps / dt / GB

res = {"size_gb": SIZE / GB, "cases": []}
bufs = {0: alloc_on_node(SIZE, 0), 1: alloc_on_node(SIZE, 1)}
print("allocated", flush=True)

def case(name, a_node, a_thr, b_node=None, b_thr=0, reps=3):
    r = {}
    # 단독
    ga = read_stream(bufs[a_node], a_thr, a_node, reps, r, "a")
    gb = read_stream(bufs[b_node], b_thr, b_node, reps, r, "b") if b_thr else 0.0
    # 동시
    out = {}
    def run_a(): out["a"] = read_stream(bufs[a_node], a_thr, a_node, reps, r, "ca")
    def run_b(): out["b"] = read_stream(bufs[b_node], b_thr, b_node, reps, r, "cb")
    ta = threading.Thread(target=run_a); tb = threading.Thread(target=run_b) if b_thr else None
    ta.start(); tb and tb.start(); ta.join(); tb and tb.join()
    rec = dict(name=name, a_node=a_node, a_thr=a_thr, b_node=b_node, b_thr=b_thr,
               solo_a=round(ga,1), solo_b=round(gb,1), conc_a=round(out["a"],1), conc_b=round(out.get("b",0.0),1))
    rec["interference"] = round((rec["conc_a"] + rec["conc_b"]) / max(1e-9, rec["solo_a"] + rec["solo_b"]), 3)
    print(rec, flush=True); res["cases"].append(rec)

# 단독 대역폭 (소켓별, 스레드 수)
for node in (0, 1):
    for thr in (12, 24, 48):
        case(f"solo_n{node}_t{thr}", node, thr)
# 간섭: expert 스트리밍(48스레드, 소켓 n) + KV 읽기(8/16스레드, 같은 소켓 / 다른 소켓)
for a_node in (0, 1):
    for b_node in (0, 1):
        for b_thr in (8, 16):
            case(f"int_a{a_node}_b{b_node}_bt{b_thr}", a_node, 48, b_node, b_thr)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
