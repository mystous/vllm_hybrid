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
python3 -c "import kt_kernel" && echo "kt ok"
'
