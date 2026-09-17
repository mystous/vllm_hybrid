#!/usr/bin/env python3
"""IDE_075 G00 — GLM routed_scaling_factor 불일치 최소 수정 (컨테이너 sgl-kt, 로컬 sglang tree). 근거: G00_code_review.md H1/H2.
현재: GPU 러너가 hot 기여에 ×rsf 를 적용하고(fused_moe.py), CPU(kt) 기여는 미적용, 디코드(dual_stream) 경로는 KT 일 때 합계에 다시 ×rsf → prefill = rsf·g + c, decode = rsf²·g + rsf·c.
수정: (1) kt_ep_wrapper.apply 에서 cpu_output 에 layer.moe_runner_config.routed_scaling_factor 를 곱함 (None/1.0 이면 그대로 → Qwen 영향 없음),
      (2) glm4_moe.py dual_stream 의 KT 분기(합계 ×rsf) 제거. 결과: prefill = decode = rsf·(g + c).
사용: glm_rsf_patch.py apply | revert | status. 백업 /sgl-workspace/ide074_backup/*.rsf_orig"""
import os, sys, shutil, hashlib
S = "/sgl-workspace/sglang/python/sglang/srt"; F_KT = f"{S}/layers/moe/kt_ep_wrapper.py"; F_GLM = f"{S}/models/glm4_moe.py"; BK = "/sgl-workspace/ide074_backup"; MARK = "IDE_075_RSF"


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def apply():
    os.makedirs(BK, exist_ok=True)
    for f in (F_KT, F_GLM):
        b = f"{BK}/{os.path.basename(f)}.rsf_orig"
        if not os.path.exists(b): shutil.copy(f, b)
        assert MARK not in open(f).read(), f
    s = open(F_KT).read()
    old = "            output = output + cpu_output\n\n        return StandardCombineInput(hidden_states=output)"
    new = "            _rsf = getattr(getattr(layer, \"moe_runner_config\", None), \"routed_scaling_factor\", None)   # IDE_075_RSF: GPU 러너가 hot 기여에 곱하는 routed_scaling_factor 를 CPU 기여에도 적용 (None/1.0 → 무변경)\n            output = output + (cpu_output * _rsf if (_rsf is not None and _rsf != 1.0) else cpu_output)\n\n        return StandardCombineInput(hidden_states=output)"
    assert s.count(old) == 1; s = s.replace(old, new)
    # fast path (num_gpu_experts == 0): return StandardCombineInput(hidden_states=cpu_output) 형태를 찾아 rsf 적용
    import re as _re
    old_fp = "            if self.tp_rank == 0:\n                output = self.sync(x)\n            else:\n                output = torch.zeros_like(x)\n            return StandardCombineInput(hidden_states=output)"
    new_fp = "            if self.tp_rank == 0:\n                output = self.sync(x)\n                _rsf0 = getattr(getattr(layer, \"moe_runner_config\", None), \"routed_scaling_factor\", None)   # IDE_075_RSF fast path (num_gpu_experts==0)\n                if _rsf0 is not None and _rsf0 != 1.0: output = output * _rsf0\n            else:\n                output = torch.zeros_like(x)\n            return StandardCombineInput(hidden_states=output)"
    if s.count(old_fp) == 1: s = s.replace(old_fp, new_fp); print("fast path patched")
    else: print("fast path pattern not found (수동 확인 필요)", s.count(old_fp))
    open(F_KT, "w").write(s)
    s = open(F_GLM).read()
    old = "            if not _is_cuda or isinstance(self.experts.quant_method, KTEPWrapperMethod):\n                final_hidden_states *= self.routed_scaling_factor"
    new = "            if not _is_cuda:   # IDE_075_RSF: KT 분기 제거 (러너가 hot 에, kt_ep_wrapper 가 cold 에 이미 ×rsf)\n                final_hidden_states *= self.routed_scaling_factor"
    assert s.count(old) == 1; open(F_GLM, "w").write(s.replace(old, new))
    for f in (F_KT, F_GLM): print("patched", os.path.basename(f), sha(f)[:12])


def revert():
    for f in (F_KT, F_GLM):
        b = f"{BK}/{os.path.basename(f)}.rsf_orig"
        if os.path.exists(b): shutil.copy(b, f); print("reverted", f)


def status():
    for f in (F_KT, F_GLM): print(("RSF " if MARK in open(f).read() else "orig"), sha(f)[:12], f)


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
