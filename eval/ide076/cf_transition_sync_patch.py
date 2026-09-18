#!/usr/bin/env python3
"""IDE_076 공통 정상성 수정 후보 (사고 00:08~00:59): callback-free 경로에서 eager(비-capture) 스텝의 첫 층(layer_idx==0) 에 진입할 때
CPU FIFO 를 비운다(cpu_infer.sync()). 디코드 웨이브 종료 → 빈 배치 → prefill(eager) 전환에서 이전 스텝의 deferred 작업이 남은 채
슬롯/pinned 버퍼가 재사용되는 경쟁을 막는 완화책. 그래프(디코드) 경로는 건드리지 않음. 컨테이너에서 apply | revert | status.
대상: /usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py (submit_forward 의 D2H 복사 직전)."""
import sys, os, shutil, hashlib
P = "/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"; BK = "/sgl-workspace/ide076_backup/experts_base.py.pre_cfsync"
ANCHOR = "        input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)"
NEW = """        if self.layer_idx == 0 and _os_cf0.environ.get("KT_CALLBACK_FREE") and not torch.cuda.is_current_stream_capturing() and not _os_cf0.environ.get("KT_CF_NO_STEP_SYNC"):   # IDE_076 CF-fix: eager 스텝 시작 시 이전 스텝 deferred 완료 대기
            self._active_cpu_infer().sync()
""" + ANCHOR
IMPORT_ANCHOR = "import torch"


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def apply():
    os.makedirs(os.path.dirname(BK), exist_ok=True)
    if not os.path.exists(BK): shutil.copy(P, BK)
    s = open(P).read()
    if "IDE_076 CF-fix" in s: print("already", sha(P)); return
    assert s.count(ANCHOR) == 1, s.count(ANCHOR)
    s = s.replace(ANCHOR, NEW)
    i = s.index(IMPORT_ANCHOR); s = s[:i] + "import os as _os_cf0   # IDE_076 CF-fix\n" + s[i:]
    open(P, "w").write(s); import ast; ast.parse(s); print("applied", sha(P))


def revert():
    if os.path.exists(BK): shutil.copy(BK, P); print("reverted", sha(P))


def status(): print("experts_base.py", sha(P), "cf-fix" if "IDE_076 CF-fix" in open(P).read() else "unpatched")


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
