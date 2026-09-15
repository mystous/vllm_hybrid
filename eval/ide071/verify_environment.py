#!/usr/bin/env python3
"""IDE_071 P0 — 환경 동결 (지시서 §3.1). evidence/software_manifest.json + environment 텍스트 저장. 비밀정보는 저장하지 않는다."""
import json, os, re, subprocess, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def run(c, t=120):
    try: return sh(c, timeout=t).stdout.strip()
    except Exception as e: return f"ERR {e}"
def dx(c, cn=CN, t=120):
    try: return dexec(cn, c, timeout=t).stdout.strip()
    except Exception as e: return f"ERR {e}"
def fsha(p):
    try: return sha256(p)
    except Exception as e: return f"ERR {e}"

def main():
    out = f"{FEAT}/evidence"; os.makedirs(out, exist_ok=True)
    m = {"taken": now(), "host": {}, "cpu": {}, "memory": {}, "gpu": {}, "containers": {}, "software": {}, "git": {}, "files": {}}
    h = m["host"]; h["hostname"] = run("hostname"); h["kernel"] = run("uname -r"); h["os"] = run("grep PRETTY /etc/os-release")
    h["uptime"] = run("uptime"); h["tz"] = run("date +%Z"); h["other_gpu_procs"] = run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
    c = m["cpu"]; c["lscpu"] = run("lscpu"); c["no_turbo"] = run("cat /sys/devices/system/cpu/intel_pstate/no_turbo"); c["governor"] = run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    c["max_freq_khz"] = run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"); c["turbostat_idle_5s"] = run("sudo turbostat --interval 3 --num_iterations 1 --quiet --show Avg_MHz,Busy%,Bzy_MHz,PkgWatt 2>&1 | head -3", 30)
    c["rapl_power_limit"] = run("sudo cat /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw /sys/class/powercap/intel-rapl:1/constraint_0_power_limit_uw 2>/dev/null")
    c["cgroup_root"] = run("cat /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null")
    c["thp"] = run("cat /sys/kernel/mm/transparent_hugepage/enabled"); c["numa_hw"] = run("numactl --hardware | head -12")
    mm = m["memory"]; mm["dimm_speed"] = run("sudo dmidecode -t memory | grep -E 'Configured Memory Speed|^\\s+Size:' | sort | uniq -c"); mm["free"] = run("free -g"); mm["numastat_m"] = run("numastat -m | head -8")
    g = m["gpu"]; g["nvidia_smi"] = run("nvidia-smi --query-gpu=index,uuid,pci.bus_id,driver_version,memory.total,power.limit,power.max_limit,clocks.max.sm,clocks.max.mem,clocks.applications.graphics,compute_mode,ecc.mode.current --format=csv")
    g["topo"] = run("nvidia-smi topo -m"); g["xid_recent"] = run("sudo dmesg 2>/dev/null | grep -i xid | tail -5")
    ct = m["containers"]
    for cn in (CN, BENCH_CN, "sgl-kt5", "sgl-kt2"):
        ct[cn] = {"inspect": run(f"{DOCKER} inspect {cn} --format '{{{{.Id}}}} {{{{.Image}}}} {{{{.State.Status}}}} cpuset={{{{.HostConfig.CpusetCpus}}}} mems={{{{.HostConfig.CpusetMems}}}} shm={{{{.HostConfig.ShmSize}}}} pids={{{{.HostConfig.PidsLimit}}}}'"),
                  "image_digest": run(f"{DOCKER} inspect {cn} --format '{{{{index .RepoDigests 0}}}}' 2>/dev/null || {DOCKER} image inspect $({DOCKER} inspect {cn} --format '{{{{.Image}}}}') --format '{{{{index .RepoDigests 0}}}}' 2>/dev/null")}
    s = m["software"]
    s["sgl_kt_versions"] = dx("python3 -c \"import torch,sglang,flashinfer,triton,kt_kernel;print('torch',torch.__version__);print('sglang',sglang.__version__);print('flashinfer',flashinfer.__version__);print('triton',triton.__version__);print('cuda',torch.version.cuda);print('nccl',torch.cuda.nccl.version() if hasattr(torch.cuda,'nccl') else None)\" 2>/dev/null")
    s["kt_kernel_pkg"] = dx("pip show kt-kernel 2>/dev/null | head -3; ls -la /usr/local/lib/python3.12/dist-packages/kt_kernel/*.so")
    s["kt_kernel_ext_sha256"] = dx("sha256sum /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so")
    s["sglang_git"] = dx("cd /sgl-workspace/sglang && git rev-parse HEAD && git status --short | head -20 && git diff --stat | tail -1")
    s["ktransformers_git"] = dx("cd /sgl-workspace/ktransformers && git rev-parse HEAD && git status --short | head -20 && git diff --stat | tail -1")
    s["deep_gemm"] = dx("ls -d /sgl-workspace/sglang/python/sglang/srt/layers/quantization/deep_gemm* /usr/local/lib/python3.12/dist-packages/deep_gemm* 2>/dev/null; python3 -c 'import deep_gemm' 2>&1 | tail -1")
    s["ompthreads_env"] = dx("env | grep -E 'OMP|MKL|OPENBLAS|KT_|SGL|NCCL|CUDA' | sort")
    s["torch_threads"] = dx("python3 -c \"import torch;print('intra',torch.get_num_threads(),'inter',torch.get_num_interop_threads())\" 2>/dev/null")
    s["nccl"] = dx("python3 -c \"import torch;print(torch.cuda.nccl.version())\" 2>/dev/null")
    s["bench_cn"] = dx("python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null; wc -l /tmp/sonnet.txt", BENCH_CN)
    s["container_diff_files"] = dx("cd /sgl-workspace/sglang && git diff --name-only; cd /sgl-workspace/ktransformers && git diff --name-only")
    # 소스 패치 diff 저장
    d1 = dx("cd /sgl-workspace/sglang && git diff", t=120); d2 = dx("cd /sgl-workspace/ktransformers && git diff", t=120)
    open(f"{out}/source_changes.patch", "w").write("# sglang @ /sgl-workspace/sglang\n" + d1 + "\n\n# ktransformers @ /sgl-workspace/ktransformers\n" + d2 + "\n")
    gt = m["git"]; gt["branch"] = run(f"git -C {REPO} branch --show-current"); gt["head"] = run(f"git -C {REPO} rev-parse HEAD"); gt["status"] = run(f"git -C {REPO} status --short | head -20")
    gt["remote"] = re.sub(r"https://[^@]*@", "https://", run(f"git -C {REPO} remote get-url origin"))
    f = m["files"]
    for k, p in {"hotmap_v1": f"{HOME}/.cache/huggingface/kt/ide069/hotmap.json", "hotmap_v2": f"{HOME}/.cache/huggingface/kt/ide070/hotmap_v2.json",
                 "layer_budget_5952": f"{HOME}/.cache/huggingface/kt/ide070/layer_budget_5952.json", "hotmap_mixed_0.25": f"{HOME}/.cache/huggingface/kt/ide070/hotmap_mixed_0.25.json",
                 "model_config": f"{TOK_HOST}/config.json", "ide070_result": f"{REPO}/shadow_assists/features/IDE_070/RESULT.md", "ide069_full_report": f"{REPO}/shadow_assists/features/IDE_069/FULL_REPORT.md",
                 "instruction": f"{FEAT}/cpu_offload_no_02_실험계획_지시서.md"}.items():
        f[k] = {"path": p, "sha256": fsha(p), "bytes": os.path.getsize(p) if os.path.exists(p) else None}
    f["kt_weights"] = run(f"ls -la {HOME}/.cache/huggingface/kt/qwen3-480b-int4 | head -5; du -sh {HOME}/.cache/huggingface/kt/qwen3-480b-int4")
    f["model_snapshot"] = run(f"ls {TOK_HOST} | head -80 | tr '\\n' ' '; du -sh {TOK_HOST}/")
    f["disk"] = run("df -h /data /home | tail -2")
    jdump(m, f"{out}/software_manifest.json")
    # 텍스트 요약
    txt = [f"# environment (redacted) {m['taken']['wall_kst']}", "", "## host", h["hostname"], h["kernel"], h["os"], "", "## cpu", f"no_turbo={c['no_turbo']} governor={c['governor']} max_freq_khz={c['max_freq_khz']}", c["turbostat_idle_5s"], c["cgroup_root"], "", "## memory", mm["dimm_speed"], mm["free"], "", "## gpu", g["nvidia_smi"], "", "## containers"]
    for cn, v in ct.items(): txt.append(f"{cn}: {v['inspect']}  digest={v['image_digest']}")
    txt += ["", "## software", s["sgl_kt_versions"], s["kt_kernel_ext_sha256"], "sglang git: " + s["sglang_git"].replace("\n", " | "), "kt git: " + s["ktransformers_git"].replace("\n", " | "), s["torch_threads"], "", "## git", f"{gt['branch']} {gt['head']} {gt['remote']}"]
    open(f"{out}/environment.before.txt", "w").write("\n".join(txt) + "\n")
    print(f"{out}/software_manifest.json"); print("\n".join(txt[:40]))

if __name__ == "__main__":
    main()
