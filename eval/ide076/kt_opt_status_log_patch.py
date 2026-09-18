#!/usr/bin/env python3
"""IDE_076 C03 feature proof: 서버 프로세스 안에서 kt_opt_status() 를 한 번 stderr 로 남겨 부팅별 실효 플래그(a1/a2/proof)를 server.full.log 에 기록한다.
성능 경로 무관(첫 wrapper 초기화 시 1 회 출력). 컨테이너에서 apply | revert | status. 대상 experts_base.py (백업 experts_base.py.pre_optlog)."""
import sys, os, shutil, hashlib
PY = "/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"; BK = "/sgl-workspace/ide076_backup/experts_base.py.pre_optlog"
ANCHOR = "        self.layer_idx = layer_idx\n"
NEW = ANCHOR + """        try:   # IDE_076 kt-opt status log (feature proof, 1 회)
            if not getattr(type(self), "_kt_opt_logged", False) and hasattr(kt_kernel_ext, "kt_opt_status"):
                type(self)._kt_opt_logged = True; import sys as _s_ol
                print(f"[kt-opt] status {kt_kernel_ext.kt_opt_status()}", file=_s_ol.stderr, flush=True)
        except Exception as _e_ol:
            pass
"""


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def apply():
    if not os.path.exists(BK): shutil.copy(PY, BK)
    s = open(PY).read()
    if "IDE_076 kt-opt status log" in s: print("already", sha(PY)); return
    assert s.count(ANCHOR) == 1, s.count(ANCHOR)
    s = s.replace(ANCHOR, NEW); import ast; ast.parse(s); open(PY, "w").write(s); print("applied", sha(PY))


def revert():
    if os.path.exists(BK): shutil.copy(BK, PY); print("reverted", sha(PY))


def status(): print("experts_base.py", sha(PY), "optlog" if "IDE_076 kt-opt status log" in open(PY).read() else "no-optlog")


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
