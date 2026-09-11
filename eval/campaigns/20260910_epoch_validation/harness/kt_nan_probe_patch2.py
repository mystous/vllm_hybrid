#!/usr/bin/env python3
"""IDE_066 계측 2판 — 1판의 두 가지 한계를 고친다.

1판 결과: 샘플러에서 NaN 치환이 4건 있었는데 MoE 합산 지점 계측은 0건이었다.
그런데 1판은 "NaN 이 0이면 아무것도 기록하지 않는" 설계라 **계측이 살아 있었는지 자체를
확인할 수 없다**. 또 `tp_rank == 0` 안에만 있어 나머지 세 랭크의 GPU expert 출력을 보지 못한다.

2판에서 고치는 것:
  (a) 심장박동 기록: 점검 시점마다 누적 호출 수와 카운터 값을 무조건 남긴다 (NaN 0이어도).
      -> "0건" 이 "없었다" 인지 "확인 못 했다" 인지 구분된다.
  (b) 전 랭크 GPU 출력 점검: rank 0 이 아닌 랭크도 GPU expert 출력을 본다.
      (CPU expert 출력은 rank 0 에만 존재하므로 그대로)
  (c) 점검 주기 2000 -> 500.

CUDA 그래프 재생 중에도 장치 카운터 누적 (isnan 커널) 은 일어나고, 보고를 트리거하는
Python 카운터는 그래프 밖 단계 (prefill·mixed) 에서만 증가한다. 보고 값에는 재생분이 포함된다.

usage: docker exec -i sgl-kt python3 - < kt_nan_probe_patch2.py
"""
import shutil, sys, py_compile

P = "/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"
src = open(P).read()

if "_KT_PROBE_V2" in src:
    print("2판 이미 적용됨")
    sys.exit(0)
if "KT_NAN_PROBE" not in src:
    print("1판이 적용돼 있지 않다 — 먼저 kt_nan_probe_patch.py 를 적용하라")
    sys.exit(1)

shutil.copy(P, P + ".v1.bak")

OLD_FN_START = "def _kt_probe_accum(obj, cpu_out, gpu_out, layer_idx):"
i = src.index(OLD_FN_START)
j = src.index("\n\n", src.index("st[\"reported_cpu\"], st[\"reported_gpu\"] = c, g"))
NEW_FN = '''_KT_PROBE_V2 = True


def _kt_probe_accum(obj, cpu_out, gpu_out, layer_idx, rank):
    """CPU/GPU expert 출력의 NaN 을 장치 카운터에 누적. 주기적으로 무조건 기록한다.

    cpu_out 가 None 이면 (rank != 0) GPU 쪽만 센다.
    """
    st = obj.__dict__.get("_kt_nan_stats")
    if st is None:
        dev = gpu_out.device
        st = obj.__dict__["_kt_nan_stats"] = {
            "n": 0,
            "cpu": torch.zeros((), dtype=torch.long, device=dev),
            "gpu": torch.zeros((), dtype=torch.long, device=dev),
        }
    if cpu_out is not None:
        st["cpu"] += torch.isnan(cpu_out).any().long()
    st["gpu"] += torch.isnan(gpu_out).any().long()
    st["n"] += 1
    if st["n"] % _KT_PROBE_EVERY == 0:
        c = int(st["cpu"].item())
        g = int(st["gpu"].item())
        _kt_logger.error(
            "KT_NAN_PROBE rank=%s layer=%s py_calls=%d cpu_expert_nan=%d gpu_expert_nan=%d",
            rank, layer_idx, st["n"], c, g,
        )
'''
src = src[:i] + NEW_FN + src[j:]

OLD_CALL = """            if _KT_NAN_PROBE:
                _kt_probe_accum(
                    self, cpu_output, output,
                    getattr(getattr(self, "kt_config", None), "layer_idx", "?"),
                )
            output = output + cpu_output"""
NEW_CALL = """            if _KT_NAN_PROBE:
                _kt_probe_accum(
                    self, cpu_output, output,
                    getattr(getattr(self, "kt_config", None), "layer_idx", "?"),
                    self.tp_rank,
                )
            output = output + cpu_output"""
assert OLD_CALL in src
src = src.replace(OLD_CALL, NEW_CALL, 1)

# rank != 0 경로에도 GPU 출력 점검을 넣는다 (합산 직전, output 이 GPU 결과인 상태)
OLD_HEAD = """        output = gpu_combine_input.hidden_states
        if self.tp_rank == 0:"""
NEW_HEAD = """        output = gpu_combine_input.hidden_states
        if _KT_NAN_PROBE and self.tp_rank != 0:
            _kt_probe_accum(
                self, None, output,
                getattr(getattr(self, "kt_config", None), "layer_idx", "?"),
                self.tp_rank,
            )
        if self.tp_rank == 0:"""
assert OLD_HEAD in src
src = src.replace(OLD_HEAD, NEW_HEAD, 1)

src = src.replace('_KT_PROBE_EVERY = int(_kt_os.environ.get("KT_NAN_PROBE_EVERY", "4000"))',
                  '_KT_PROBE_EVERY = int(_kt_os.environ.get("KT_NAN_PROBE_EVERY", "500"))')

open(P, "w").write(src)
py_compile.compile(P, doraise=True)
print("2판 적용 완료:", P)
