#!/usr/bin/env python3
"""IDE_066: KTransformers MoE 합산 지점에 NaN 계측을 넣는다.

SGLang 의 비-투기 경로에는 NaN 검사가 샘플러 (layers/sampler.py:95) 한 곳뿐이라
"어느 단계에서 NaN 이 처음 생겼는지" 를 알 수 없다. kt_ep_wrapper.apply() 의
  output = gpu_combine_input.hidden_states
  cpu_output = self.sync(x)          <- CPU expert (INT4/AMX) 결과
  output = output + cpu_output
지점에서 CPU 쪽과 GPU 쪽을 따로 세면 원인 구간이 갈린다.

동기화 비용을 피하려고 장치 상의 카운터에 누적하고, KT_NAN_PROBE_EVERY 호출마다
한 번만 .item() 으로 읽는다. 환경변수 KT_NAN_PROBE=1 일 때만 동작한다.

usage: docker exec -i sgl-kt python3 - < kt_nan_probe_patch.py
"""
import re, shutil, sys

P = "/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"
src = open(P).read()

if "KT_NAN_PROBE" in src:
    print("이미 적용됨")
    sys.exit(0)

shutil.copy(P, P + ".ide066.bak")

HEAD = '''
import logging as _kt_logging
import os as _kt_os

_kt_logger = _kt_logging.getLogger(__name__)
_KT_NAN_PROBE = _kt_os.environ.get("KT_NAN_PROBE", "0") == "1"
_KT_PROBE_EVERY = int(_kt_os.environ.get("KT_NAN_PROBE_EVERY", "4000"))


def _kt_probe_accum(obj, cpu_out, gpu_out, layer_idx):
    """CPU/GPU expert 출력의 NaN 을 장치 카운터에 누적. 주기적으로만 호스트로 읽는다."""
    st = obj.__dict__.get("_kt_nan_stats")
    if st is None:
        dev = cpu_out.device
        st = obj.__dict__["_kt_nan_stats"] = {
            "n": 0,
            "cpu": torch.zeros((), dtype=torch.long, device=dev),
            "gpu": torch.zeros((), dtype=torch.long, device=dev),
            "reported_cpu": 0,
            "reported_gpu": 0,
        }
    st["cpu"] += torch.isnan(cpu_out).any().long()
    st["gpu"] += torch.isnan(gpu_out).any().long()
    st["n"] += 1
    if st["n"] % _KT_PROBE_EVERY == 0:
        c = int(st["cpu"].item())
        g = int(st["gpu"].item())
        if c != st["reported_cpu"] or g != st["reported_gpu"]:
            _kt_logger.error(
                "KT_NAN_PROBE layer=%s calls=%d cpu_expert_nan=%d gpu_expert_nan=%d",
                layer_idx, st["n"], c, g,
            )
            st["reported_cpu"], st["reported_gpu"] = c, g

'''

# 모듈 상단 (KTRANSFORMERS_AVAILABLE 블록 뒤) 에 삽입
anchor = "except ImportError:\n    KTRANSFORMERS_AVAILABLE = False\n"
assert anchor in src, "삽입 지점을 찾지 못했다"
src = src.replace(anchor, anchor + HEAD, 1)

# 합산 지점에 계측 호출 삽입
old = """        output = gpu_combine_input.hidden_states
        if self.tp_rank == 0:
            cpu_output = self.sync(x)
            output = output + cpu_output"""
new = """        output = gpu_combine_input.hidden_states
        if self.tp_rank == 0:
            cpu_output = self.sync(x)
            if _KT_NAN_PROBE:
                _kt_probe_accum(
                    self, cpu_output, output,
                    getattr(getattr(self, "kt_config", None), "layer_idx", "?"),
                )
            output = output + cpu_output"""
assert old in src, "합산 지점을 찾지 못했다"
src = src.replace(old, new, 1)

open(P, "w").write(src)
import py_compile
py_compile.compile(P, doraise=True)
print("적용 완료:", P)
print("복구:", P + ".ide066.bak")
