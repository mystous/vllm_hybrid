#!/usr/bin/env python3
"""지시서 §14 의 제출 디렉터리를 구성하고 MANIFEST.sha256 을 만든다."""
import hashlib, os, shutil, subprocess, sys

ROOT = "/work"
OUT = os.path.join(ROOT, "RBC_Attn_v02_result")
RES = os.path.join(ROOT, "results", "v02")

def copytree(src, dst, **kw):
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True, **kw)

os.makedirs(OUT, exist_ok=True)
# src / configs / tools / tests
copytree(os.path.join(ROOT, "rbc"), os.path.join(OUT, "src", "rbc"))
copytree(os.path.join(ROOT, "src"), os.path.join(OUT, "src", "cpp"))
for f in ("build.sh", "bench.py", "stream_bench.py", "profile_one.py",
          "run_matrix.sh", "run_host_tests.sh", "run_gpu.sh", "RESULT.md"):
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        shutil.copy2(p, os.path.join(OUT, "src", f))
copytree(os.path.join(ROOT, "configs"), os.path.join(OUT, "configs"))
copytree(os.path.join(ROOT, "tools"), os.path.join(OUT, "tools"))
copytree(os.path.join(ROOT, "tests"), os.path.join(OUT, "tests"))
# manifest / plans / calibration / raw / profiles
copytree(os.path.join(RES, "manifest"), os.path.join(OUT, "manifest"))
copytree(os.path.join(RES, "plan"), os.path.join(OUT, "plans"))
copytree(os.path.join(RES, "select"), os.path.join(OUT, "calibration"))
os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
for sub in ("kernel", "runtime"):
    copytree(os.path.join(RES, sub), os.path.join(OUT, "raw", sub))
os.makedirs(os.path.join(OUT, "profiles"), exist_ok=True)
prof = os.path.join(RES, "profiles")
if os.path.exists(prof):
    copytree(prof, os.path.join(OUT, "profiles"))
else:
    with open(os.path.join(OUT, "profiles", "README.md"), "w") as f:
        f.write("# profiles\n\nNCU·NSYS 프로파일은 이번 v0.2 작업에서 수집하지 않았다 "
                "(NOT_RUN).\nv0.1 의 NCU 결과는 `manifest/ncu_metrics.txt` 의 메트릭 "
                "목록과 함께 상위 RESULT.md 에 있다.\n지시서 §7.5 에 따라 시간 모델은 "
                "profiling 없는 CUDA graph timing 으로만 교정했다.\n")

# MANIFEST.sha256
lines = []
for dirpath, dirnames, filenames in os.walk(OUT):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in sorted(filenames):
        if fn == "MANIFEST.sha256" or fn.endswith(".pyc"):
            continue
        full = os.path.join(dirpath, fn)
        h = hashlib.sha256()
        with open(full, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {os.path.relpath(full, OUT)}")
lines.sort(key=lambda x: x.split("  ", 1)[1])
with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f:
    f.write("\n".join(lines) + "\n")
n = sum(os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(OUT) for f in fs)
print(f"파일 {len(lines)}개, 합계 {n/1e6:.1f} MB -> {OUT}")
