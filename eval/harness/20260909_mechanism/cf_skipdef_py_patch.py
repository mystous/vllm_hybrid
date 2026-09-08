#!/usr/bin/env python3
"""IDE_036 python: arm/rearm 호출에 (mask_ptr, n_experts, out_row_bytes) 전달. 마스크 = self.gpu_experts_mask (pinned bool, 논리 id 기준 — hotmap 재배치 후 physical 이 논리 순서와 다르므로 C++ should_skip_expert 와 동일한 마스크를 그대로 사용)."""
P="/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"; s=open(P).read()
if "_cf_mask_args" in s: print("already"); raise SystemExit
old="""            if torch.cuda.is_current_stream_capturing():
                _slot = _ci.arm_packet(_imm, _def if _has else _dummy, _has)      # graph 별 고정 슬롯 (재생 시 재사용)
            else:
                if not hasattr(self, "_cf_eager_slot"):
                    self._cf_eager_slot = _ci.arm_packet(_imm, _def if _has else _dummy, _has)
                else:
                    _ci.rearm_packet(self._cf_eager_slot, _imm, _def if _has else _dummy, _has)"""
new="""            _cf_mask_args = (int(self.gpu_experts_mask.data_ptr()), int(self.gpu_experts_mask.numel()), int(output_cpu[next_slot].shape[-1]) * 2)
            if torch.cuda.is_current_stream_capturing():
                _slot = _ci.arm_packet(_imm, _def if _has else _dummy, _has, *_cf_mask_args)      # graph 별 고정 슬롯 (재생 시 재사용)
            else:
                if not hasattr(self, "_cf_eager_slot"):
                    self._cf_eager_slot = _ci.arm_packet(_imm, _def if _has else _dummy, _has, *_cf_mask_args)
                else:
                    _ci.rearm_packet(self._cf_eager_slot, _imm, _def if _has else _dummy, _has, *_cf_mask_args)"""
assert s.count(old)==1, s.count(old); s=s.replace(old,new)
import ast; ast.parse(s); open(P,"w").write(s); print("IDE_036 python patch applied")
