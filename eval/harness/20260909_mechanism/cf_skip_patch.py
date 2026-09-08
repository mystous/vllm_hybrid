#!/usr/bin/env python3
"""IDE_033-b: 빈 immediate 작업 생략 (KT_CF_SKIP_EMPTY_IMM=1, CF 경로 전용).
N ≥ k (cold 전량 deferral) 이면 immediate ids 는 구조적으로 전부 -1 이고 incremental=True 층에서 immediate 작업은
스레드풀 fan-out + 빈 merge 만 수행 (고정비 t0). 이를 생략하면 패킷 = [done 신호, deferred] 가 되어
sync(L) 은 FIFO 상 def(L-1) 완료 직후 풀린다. incremental=False 층 (layer 0) 은 출력 초기화가 필요하므로 유지."""
P="/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"; s=open(P).read()
old="""        incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        immediate_task = self.moe.forward_task(
            bsz_slot_tensor.data_ptr(),
            immediate_experts_ids_cpu[current_slot].size(-1),
            immediate_experts_ids_cpu[current_slot].data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output_cpu[current_slot].data_ptr(),
            incremental,
        )
        if sync_submit:
            # Keep subscribe_ascend_stream"""
new="""        incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        import os as _os_sk
        _cf_skip_imm = (bool(_os_sk.environ.get("KT_CALLBACK_FREE")) and bool(_os_sk.environ.get("KT_CF_SKIP_EMPTY_IMM"))
                        and incremental and not sync_submit
                        and self.max_deferred_experts_per_token >= self.num_experts_per_tok)
        immediate_task = (0, 0) if _cf_skip_imm else self.moe.forward_task(
            bsz_slot_tensor.data_ptr(),
            immediate_experts_ids_cpu[current_slot].size(-1),
            immediate_experts_ids_cpu[current_slot].data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output_cpu[current_slot].data_ptr(),
            incremental,
        )
        if sync_submit:
            # Keep subscribe_ascend_stream"""
assert s.count(old)==1, s.count(old); s=s.replace(old,new)
import ast; ast.parse(s); open(P,"w").write(s); print("cf skip patch applied")
