#!/usr/bin/env bash
# IDE_070 / TSK_056 — sglang kt_ep_wrapper 에 layer 별 GPU expert 수 주입 (컨테이너 로컬 패치).
# KT_GPU_EXPERTS_PER_LAYER=<json 파일 경로> 가 설정되면 KTConfig.num_gpu_experts 를 그 파일의
# per_layer[layer_idx] 로 바꾼다. 미설정이면 기존과 동일 (--kt-num-gpu-experts 균일).
# 물리 id 0..N_l-1 이 GPU 라는 규약은 그대로이므로 hotmap(층별 빈도순) 과 함께 써야 한다.
# 사용: patch_per_layer_experts.sh <container> [--revert]
set -euo pipefail
export PATH=$HOME/bin:$PATH
CN=$1; MODE=${2:-apply}
F=/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py
docker exec -i "$CN" python3 - "$MODE" <<'PY'   # -i 필수: 없으면 heredoc 이 전달되지 않아 아무것도 하지 않고 종료 (9-15 20:30 nu5952 무효 원인)
import sys, re
mode = sys.argv[1]
p = "/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"
s = open(p).read()
MARK = "# [IDE_070] per-layer gpu experts"
old = "        num_gpu_experts=server_args.kt_num_gpu_experts,"
new = """        num_gpu_experts=_ide070_per_layer_experts(server_args.kt_num_gpu_experts, layer_idx),"""
helper = f'''
{MARK}
def _ide070_per_layer_experts(default_n, layer_idx):
    """KT_GPU_EXPERTS_PER_LAYER (json: {{"per_layer": [...]}}) 가 있으면 층별 값을 쓴다."""
    import json as _j, os as _o
    path = _o.environ.get("KT_GPU_EXPERTS_PER_LAYER", "")
    if not path:
        return default_n
    try:
        per = _j.load(open(path))["per_layer"]
        n = int(per[layer_idx])
        _kt_logger.info("IDE_070 per-layer gpu experts: layer %d -> %d (default %d)", layer_idx, n, default_n)
        return n
    except Exception as e:  # 실패는 숨기지 않는다
        _kt_logger.error("IDE_070 per-layer gpu experts load failed: %s", e)
        raise
'''
if mode == "--revert":
    if MARK not in s:
        print("not patched"); sys.exit(0)
    s = s.replace(new, old)
    i = s.index("\n" + MARK); j = s.index("        raise\n", i) + len("        raise\n")
    s = s[:i] + s[j:]
    open(p, "w").write(s); print("reverted"); sys.exit(0)
if MARK in s:
    print("already patched"); sys.exit(0)
assert old in s, "anchor not found"
s = s.replace(old, new, 1)
# helper 를 KTConfig 클래스 정의 앞에 삽입
anchor = "\n@dataclass\nclass KTConfig:"
assert anchor in s
s = s.replace(anchor, helper + anchor, 1)
open(p, "w").write(s); print("patched")
PY
docker exec "$CN" python3 -c "import ast,sys; ast.parse(open('$F').read()); print('syntax ok')"
