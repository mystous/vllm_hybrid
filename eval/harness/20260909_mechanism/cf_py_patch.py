#!/usr/bin/env python3
"""IDE_033 Python patch (kt_kernel/experts_base.py): KT_CALLBACK_FREE=1 이면 submit/sync 의 host 콜백 노드를
arm_packet(캡처 시 새 슬롯 / eager 시 층별 고정 슬롯 재무장) + go_on_stream(memop) / wait_done_on_stream(memop) 으로 대체.
env 미설정 시 기존 경로 그대로."""
P = "/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"
s = open(P).read()

# (1) submit_forward: immediate submit 지점 — 기존 (IDE_030 채널/신호 패치가 들어간) 블록을 찾아 CF 분기 삽입
old_imm = """        else:
            _ci_s = self._active_cpu_infer()
            _ci_s.submit_with_cuda_stream(cuda_stream, immediate_task)"""
new_imm = """        else:
            _ci_s = self._active_cpu_infer()
            import os as _os_cf
            if _os_cf.environ.get("KT_CALLBACK_FREE"):
                # IDE_033: host 노드 대신 패킷 등록 + GPU memop 트리거. deferred 는 아래에서 패킷에 합류.
                self._cf_pending_imm = immediate_task
                self._cf_pending_ci = _ci_s
                self._cf_pending_stream = cuda_stream
            else:
                _ci_s.submit_with_cuda_stream(cuda_stream, immediate_task)"""
assert s.count(old_imm) == 1, s.count(old_imm)
s = s.replace(old_imm, new_imm)

# (2) deferred submit 지점 (submit_forward 안, sync_submit 아닌 경로) → CF 면 패킷 완성 + go
old_def = """            if sync_submit:
                self.cpu_infer.submit(deferred_task)
            else:
                self._active_cpu_infer().submit_with_cuda_stream(cuda_stream, deferred_task)
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True"""
new_def = """            if sync_submit:
                self.cpu_infer.submit(deferred_task)
            elif getattr(self, "_cf_pending_imm", None) is not None:
                self._cf_pending_def = deferred_task
            else:
                self._active_cpu_infer().submit_with_cuda_stream(cuda_stream, deferred_task)
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True
        # IDE_033: 패킷 확정 (imm [+def]) → 슬롯 결정 → go memop
        if getattr(self, "_cf_pending_imm", None) is not None:
            _ci = self._cf_pending_ci; _imm = self._cf_pending_imm
            _def = getattr(self, "_cf_pending_def", None); _has = _def is not None
            _dummy = (0, 0)
            if torch.cuda.is_current_stream_capturing():
                _slot = _ci.arm_packet(_imm, _def if _has else _dummy, _has)      # graph 별 고정 슬롯 (재생 시 재사용)
            else:
                if not hasattr(self, "_cf_eager_slot"):
                    self._cf_eager_slot = _ci.arm_packet(_imm, _def if _has else _dummy, _has)
                else:
                    _ci.rearm_packet(self._cf_eager_slot, _imm, _def if _has else _dummy, _has)
                _slot = self._cf_eager_slot
            _ci.go_on_stream(self._cf_pending_stream, _slot)
            self._cf_last_slot = _slot
            self._cf_pending_imm = None; self._cf_pending_def = None"""
assert s.count(old_def) == 1, s.count(old_def)
s = s.replace(old_def, new_def)

# (3) sync_forward: CF 면 done memop 대기 (기존 SGL_INTERLEAVE 분기 앞에)
old_sync = """        if _os.environ.get("SGL_INTERLEAVE"):
            _ci = self._active_cpu_infer() if hasattr(self, "_active_cpu_infer") else self.cpu_infer"""
new_sync = """        if _os.environ.get("KT_CALLBACK_FREE") and getattr(self, "_cf_last_slot", None) is not None:
            self._active_cpu_infer().wait_done_on_stream(cuda_stream, self._cf_last_slot)
        elif _os.environ.get("SGL_INTERLEAVE"):
            _ci = self._active_cpu_infer() if hasattr(self, "_active_cpu_infer") else self.cpu_infer"""
assert s.count(old_sync) == 1, s.count(old_sync)
s = s.replace(old_sync, new_sync)
open(P, "w").write(s)
import ast; ast.parse(s)
print("IDE_033 python patch applied")
