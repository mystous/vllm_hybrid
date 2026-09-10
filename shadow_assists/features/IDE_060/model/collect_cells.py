#!/usr/bin/env python3
"""측정된 모든 bench 셀을 하나의 CSV 로 모은다 (EXP-E11 비용모델 calibration 입력).
각 로그의 Namespace(...) 줄에서 워크로드 파라미터를, 요약부에서 지표를 뽑는다.
구성 (KV dtype / graph set / chunk / mixed / cpuinfer) 은 체인 스크립트로부터의 수동 매핑."""
import re, os, csv, glob, sys

RES = os.path.expanduser("~/projects/vllm_hybrid/eval/results")

# cell 이름 -> 구성. (kv, graph_max, bucket_list, chunk, mixed, cpuinfer, maxtot)
CFG = {
    # 20260910_002544_expE00_E01_E02
    "A00":     ("bf16", 64,  "auto",            4096, 0, 96, None),
    "A10":     ("fp8",  64,  "auto",            4096, 0, 96, None),
    "A01":     ("bf16", 64,  "32,64",           4096, 0, 96, None),
    "A11m64":  ("fp8",  64,  "32,64",           4096, 0, 96, None),
    "BREPRO":  ("fp8",  224, "32..224/32",      4096, 0, 96, 143360),
    "C1":      ("fp8",  224, "32..224/32",      4096, 0, 96, 143360),
    "C2":      ("fp8",  224, "32..224/32",      4096, 0, 96, 63654),
    # 20260910_055709_expE06_E07
    "TUNED":   ("bf16", 96,  "32,64,96",        4096, 0, 96, None),
    "EPOCHB":  ("fp8",  224, "32..224/32",      4096, 0, 96, 143360),
    # 20260910_070926 (ops = mixed on)
    "EPBOPS":  ("fp8",  224, "32..224/32",      4096, 1, 96, 143360),
    "TUNOPS":  ("bf16", 96,  "32,64,96",        4096, 1, 96, None),
    "EPBPROF": ("fp8",  224, "32..224/32",      4096, 0, 96, 143360),
    "TUNPROF": ("bf16", 96,  "32,64,96",        4096, 0, 96, None),
    # 20260910_121248
    "F00":     ("bf16", 96,  "32,64,96",        4096, 0, 96, None),
    "F10":     ("fp8",  224, "32..224/32",      4096, 0, 96, 143360),
    "CPU48":   ("fp8",  224, "32..224/32",      4096, 0, 48, 143360),
    "CPU72":   ("fp8",  224, "32..224/32",      4096, 0, 72, 143360),
    "OPSC":    ("fp8",  224, "32..224/32",      4096, 1, 96, 143360),
    "G_EPO":   ("fp8",  224, "32..224/32",      4096, 0, 96, 143360),
    "G_TUN":   ("bf16", 96,  "32,64,96",        4096, 0, 96, None),
    # 20260910_*_e11_stage1_chunkcal (chunk / eager 벌점 보정, 후보 공간 밖 graph96)
    "CH4096":  ("fp8",  96,  "32,64,96",        4096, 0, 96, None),
    "CH8192":  ("fp8",  96,  "32,64,96",        8192, 0, 96, None),
    # 20260910_155436 (E12)
    "EPCORE":  ("fp8",  224, "32..224/32",      4096, 0, 96, 143360),
    "EPOPS":   ("fp8",  224, "32..224/32",      4096, 1, 96, 143360),
}

NUM = r"([0-9]+\.?[0-9]*)"
PATS = {
    "tok_s":    re.compile(r"Output token throughput \(tok/s\):\s*"+NUM),
    "req_s":    re.compile(r"Request throughput \(req/s\):\s*"+NUM),
    "dur":      re.compile(r"Benchmark duration \(s\):\s*"+NUM),
    "tpot":     re.compile(r"Mean TPOT \(ms\):\s*"+NUM),
    "tpot_p99": re.compile(r"P99 TPOT \(ms\):\s*"+NUM),
    "ttft_p50": re.compile(r"Median TTFT \(ms\):\s*"+NUM),
    "ttft_p99": re.compile(r"P99 TTFT \(ms\):\s*"+NUM),
    "ok":       re.compile(r"Successful requests:\s*([0-9]+)"),
    "tot_tok_s":re.compile(r"Total Token throughput \(tok/s\):\s*"+NUM),
}
ARGS = {
    "np":   re.compile(r"num_prompts=([0-9]+)"),
    "inlen":re.compile(r"sonnet_input_len=([0-9]+)"),
    "outlen":re.compile(r"sonnet_output_len=([0-9]+)"),
    "C":    re.compile(r"max_concurrency=([0-9a-zA-Z]+)"),
    "rate": re.compile(r"request_rate=([0-9a-zA-Z.]+)"),
    "seed": re.compile(r"seed=([0-9]+)"),
}

def cell_of(fname):
    b = os.path.basename(fname)[:-4]
    for pre in ("p_", "e06_", "e07_", "e07ops_", "e03_", "rep_", "soak_", "c1_", "h", "o_"):
        if b.startswith(pre):
            b = b[len(pre):]; break
    # 뒤쪽 _C.. / _r.. / _s.. / _512x128 제거
    b = re.sub(r"_(C[0-9]+|r[0-9.]+|s[0-9]+|[0-9]+x[0-9]+)$", "", b)
    b = re.sub(r"_(C[0-9]+|r[0-9.]+|s[0-9]+|[0-9]+x[0-9]+)$", "", b)
    b = re.sub(r"_(P[123]_\w+)$", "", b)
    return b

rows = []
for d in sorted(glob.glob(os.path.join(RES, "20260910_*")) + glob.glob(os.path.join(RES, "20260909_1[789]*")) + glob.glob(os.path.join(RES, "20260909_2*"))):
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        if os.path.basename(f) in ("RUN.log",) or "server_" in f: continue
        try: txt = open(f, errors="ignore").read()
        except Exception: continue
        if "Output token throughput" not in txt: continue
        r = {"dir": os.path.basename(d), "file": os.path.basename(f)}
        for k, p in {**PATS, **ARGS}.items():
            m = p.search(txt); r[k] = m.group(1) if m else ""
        cell = cell_of(f); r["cell"] = cell
        cfg = CFG.get(cell)
        if cfg:
            r["kv"], r["graph_max"], r["buckets"], r["chunk"], r["mixed"], r["cpuinfer"], r["maxtot"] = cfg
        else:
            for k in ("kv","graph_max","buckets","chunk","mixed","cpuinfer","maxtot"): r[k] = ""
        rows.append(r)

cols = ["dir","file","cell","kv","graph_max","buckets","chunk","mixed","cpuinfer","maxtot",
        "inlen","outlen","C","rate","np","seed","ok","tok_s","tot_tok_s","req_s","dur","tpot","tpot_p99","ttft_p50","ttft_p99"]
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cells.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow(r)
print(f"{len(rows)} cells -> {out}")
unk = sorted({r["cell"] for r in rows if not r["kv"]})
if unk: print("구성 미매핑 cell:", unk)
