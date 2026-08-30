#!/usr/bin/env bash
# lmsysorg/sglang:latest 신규 컨테이너에 kt-kernel + 호환 패치 4건 적용. usage: setup_kt_container.sh <container>
set -e
CN=$1
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
docker exec $CN bash -c '
set -e
pip install -q --no-deps kt-kernel==0.7.0.post2 2>&1 | grep -v WARNING | tail -1 || true
mkdir -p /usr/local/lib/python3.12/dist-packages/scripts
curl -sfL https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/kt-kernel/scripts/convert_cpu_weights.py -o /usr/local/lib/python3.12/dist-packages/scripts/convert_cpu_weights.py
python3 - <<EOF
p = "/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"
s = open(p).read()
if "gpu_experts_mask=None" not in s:
    s = s.replace("                moe_intermediate_size=intermediate_size_full,",
                  "                moe_intermediate_size=intermediate_size_full,\n                gpu_experts_mask=None,")
    open(p, "w").write(s)
p = "/usr/local/lib/python3.12/dist-packages/sgl_kernel/moe.py"
s = open(p).read()
if "ignore_invalid_expert" not in s:
    s = s.replace("""    pad_sorted_token_ids=False,
):
    torch.ops.sgl_kernel.moe_align_block_size.default(""", """    pad_sorted_token_ids=False,
    ignore_invalid_expert=False,
):
    torch.ops.sgl_kernel.moe_align_block_size.default(""")
    s = s.replace("""        cumsum_buffer,
        pad_sorted_token_ids,
    )""", """        cumsum_buffer,
        pad_sorted_token_ids,
        ignore_invalid_expert,
    )""", 1)
    open(p, "w").write(s)
print("patches ok")
EOF
apt-get install -y -qq numactl 2>&1 | tail -1 || true

# --- PLN_007 패치 3건 (2026-08-30) ---
python3 - <<PEOF
# 1) qwen3_moe.forward_normal: expert 배치 정보 전달
p="/sgl-workspace/sglang/python/sglang/srt/models/qwen3_moe.py"; s=open(p).read()
if "expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(\n                layer_id=self.layer_id,\n            ),\n        )\n        final_hidden_states = self.experts(hidden_states, topk_output)" not in s:
    s=s.replace("""        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)""",
"""        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(
            hidden_states,
            router_logits,
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=self.layer_id,
            ),
        )
        final_hidden_states = self.experts(hidden_states, topk_output)""")
    open(p,"w").write(s)
# 2) kt_ep_wrapper: 진짜 gpu_experts_mask 전달 + 3) x=0 fast path
p="/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"; s=open(p).read()
if "_gpu_mask" not in s:
    s=s.replace("""                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=None,""",
"""                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=_gpu_mask,""")
    s=s.replace("""            self.wrapper = KTMoEWrapper(""",
"""            _gpu_mask = None
            if self.num_gpu_experts > 0:
                _gpu_mask = torch.zeros(num_experts, dtype=torch.bool)
                _gpu_mask[: self.num_gpu_experts] = True
            self.wrapper = KTMoEWrapper(""")
if "Fast path: no GPU experts" not in s:
    s=s.replace("""        # Step 2: Prepare GPU computation by masking CPU expert IDs""",
"""        if self.num_gpu_experts == 0:
            # Fast path: no GPU experts -> skip the GPU fused-MoE pipeline entirely
            if self.tp_rank == 0:
                output = self.sync(x)
            else:
                output = torch.zeros_like(x)
            return StandardCombineInput(hidden_states=output)

        # Step 2: Prepare GPU computation by masking CPU expert IDs""")
open(p,"w").write(s)
print("pln007 patches ok")
PEOF

python3 -c "import kt_kernel" && echo "kt ok"
'
