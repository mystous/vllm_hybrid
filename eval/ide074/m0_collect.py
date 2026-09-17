#!/usr/bin/env python3
"""IDE_074 M0 — 기준 환경·원자료·Qwen OPT4 고정 구성 수집 (읽기 전용).
산출: evidence/environment.json, SOURCE_INDEX.md, evidence/effective_config_reference.json (IDE_073 Q-OPT4-b1/a1 실효값 사본), evidence/hotmap_budget.json"""
import json, os, subprocess, hashlib, datetime, glob
REPO = os.path.expanduser("~/projects/vllm_hybrid"); FEAT = f"{REPO}/shadow_assists/features/IDE_074"; C73 = f"{REPO}/eval/results/IDE_073_20260917"
DOCKER = os.path.expanduser("~/bin/docker"); CN = "sgl-kt"
os.makedirs(f"{FEAT}/evidence", exist_ok=True)


def sh(cmd, timeout=120):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout); return {"cmd": cmd, "rc": r.returncode, "out": r.stdout.strip()[-20000:], "err": r.stderr.strip()[-2000:]}


def dx(cmd, timeout=120): return sh(f"{DOCKER} exec {CN} sh -c {json.dumps(cmd)}", timeout)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    env = {"collected_kst": now, "base_commit": "67d24dd84ed5681e97851c018af01d3a2d917543"}
    env["git"] = {"head": sh(f"git -C {REPO} rev-parse HEAD")["out"], "branch": sh(f"git -C {REPO} branch --show-current")["out"], "status_short": sh(f"git -C {REPO} status --short")["out"], "base_is_ancestor": sh(f"git -C {REPO} merge-base --is-ancestor 67d24dd84ed5681e97851c018af01d3a2d917543 HEAD")["rc"] == 0}
    env["host"] = {"hostname": sh("hostname")["out"], "kernel": sh("uname -r")["out"], "lscpu": sh("lscpu")["out"], "smt": sh("cat /sys/devices/system/cpu/smt/active")["out"], "no_turbo": sh("cat /sys/devices/system/cpu/intel_pstate/no_turbo")["out"], "governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")["out"], "numactl_H": sh("numactl -H")["out"], "meminfo_head": sh("head -5 /proc/meminfo")["out"], "df": sh("df -h / /models ~/.cache 2>/dev/null")["out"], "load": sh("cat /proc/loadavg")["out"], "top_procs": sh("ps -eo pid,pcpu,pmem,comm --sort=-pcpu | head -12")["out"]}
    env["gpu"] = {"query": sh("nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.used,memory.total,clocks.sm,clocks.mem,power.draw,power.limit,persistence_mode --format=csv")["out"], "topo": sh("nvidia-smi topo -m")["out"], "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")["out"], "app_clocks": sh("nvidia-smi -q -d CLOCK | head -60")["out"]}
    env["container"] = {"inspect_image": sh(f"{DOCKER} inspect {CN} --format '{{{{.Image}}}} {{{{.State.Status}}}} {{{{.Config.Image}}}}'")["out"], "image_digest": sh(f"{DOCKER} images --digests --format '{{{{.Repository}}}}:{{{{.Tag}}}} {{{{.Digest}}}}' | head -20")["out"],
                        "python": dx("python3 -c 'import sys,torch,sglang,kt_kernel,os;print(sys.version);print(torch.__version__, torch.version.cuda);print(sglang.__file__);print(kt_kernel.__file__);print(sglang.__version__)'")["out"],
                        "sglang_git": dx("cd /sgl-workspace/sglang && git rev-parse HEAD && git status --short | head -40 && git diff --stat | tail -3")["out"],
                        "kt_so": dx("python3 - <<'EOF'\nimport kt_kernel,os,hashlib,glob\nd=os.path.dirname(kt_kernel.__file__)\nfor p in sorted(glob.glob(d+'/**/*.so',recursive=True)):\n    print(hashlib.sha256(open(p,'rb').read()).hexdigest(), os.path.getsize(p), p)\nEOF")["out"],
                        "kt_version": dx("pip show kt-kernel 2>/dev/null | head -3; pip show sglang 2>/dev/null | head -3")["out"],
                        "cupti_profiler_smoke": "IDE_073 세션에서 torch.profiler CUDA activity 단독 시험 kernel 5건 정상 기록 (2026-09-17 10:52)"}
    # 고정 구성 파일 해시·층별 값 (컨테이너 경로)
    hb = {}
    for p in ("/models/kt/ide070/hotmap_v2.json", "/models/kt/ide070/layer_budget_5952.json"):
        r = dx(f"sha256sum {p}; wc -c {p}"); hb[p] = r["out"]
    bud = dx("cat /models/kt/ide070/layer_budget_5952.json")["out"]
    try: bj = json.loads(bud); hb["layer_budget_values"] = bj; hb["layer_budget_sum"] = sum(bj.values()) if isinstance(bj, dict) else sum(bj)
    except Exception as e: hb["layer_budget_parse"] = str(e)
    hm = dx("python3 -c \"import json;j=json.load(open('/models/kt/ide070/hotmap_v2.json'));print(type(j).__name__, (list(j.keys())[:5] if isinstance(j,dict) else len(j)))\"")["out"]; hb["hotmap_shape"] = hm
    hb["kt_weight_dir"] = dx("ls /models/kt/qwen3-480b-int4 | head; du -sh /models/kt/qwen3-480b-int4")["out"]
    hb["model_snapshot"] = dx("ls /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/")["out"]
    json.dump(hb, open(f"{FEAT}/evidence/hotmap_budget.json", "w"), indent=1, ensure_ascii=False)
    # 입력·워크로드 고정
    wl = json.load(open(f"{REPO}/eval/ide071/configs/workloads.json")); env["workloads_M073"] = {k: v for k, v in wl.items() if k.startswith("M073_QWEN")}
    env["input_files"] = {}
    for p in sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/kt/ide073/inputs/qwen/*"))): env["input_files"][p] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
    # IDE_073 참조 실효 구성·증빙 사본
    ref = f"{C73}/Q-OPT4-b1/a1"
    for fn in ("effective_config.json", "runtime_proofs.json", "thread_affinity.json", "launch_cmd.sh", "requested_config.json"):
        if os.path.exists(f"{ref}/{fn}"): env.setdefault("reference_files", {})[fn] = {"path": f"{ref}/{fn}", "sha256": sha(f"{ref}/{fn}")}
    if os.path.exists(f"{ref}/effective_config.json"): json.dump(json.load(open(f"{ref}/effective_config.json")), open(f"{FEAT}/evidence/effective_config_reference.json", "w"), indent=1, ensure_ascii=False)
    if os.path.exists(f"{ref}/launch_cmd.sh"): env["reference_launch_cmd"] = open(f"{ref}/launch_cmd.sh").read().strip().splitlines()[-1]
    json.dump(env, open(f"{FEAT}/evidence/environment.json", "w"), indent=1, ensure_ascii=False)
    # SOURCE_INDEX.md
    L = [f"# SOURCE_INDEX — IDE_074 ({now})", "", "지시서 §3.1 원자료·코드 존재 확인 (읽기 전용).", "", "| 구분 | 경로 | 존재 | 비고 |", "|---|---|---|---|"]
    items = [("결과·상태", f"{REPO}/shadow_assists/features/IDE_073/FULL_REPORT.md"), ("결과·상태", f"{REPO}/shadow_assists/features/IDE_073/COMPLETION_STATUS.md"), ("결과·상태", f"{REPO}/shadow_assists/features/IDE_073/CONFIG_DIFF.md"), ("결과·상태", f"{REPO}/shadow_assists/features/IDE_073/BASELINE_PROVENANCE.md"), ("결과·상태", f"{REPO}/shadow_assists/features/IDE_073/PLAN_RESOLVED.md"),
             ("측정기", f"{REPO}/eval/ide073/probes.py"), ("분석기", f"{REPO}/eval/ide073/analyze_probes.py"), ("하네스", f"{REPO}/eval/ide073/run_campaign.py"), ("하네스", f"{REPO}/eval/ide071/common.py"), ("하네스", f"{REPO}/eval/ide071/run_cell.py"), ("입력 정의", f"{REPO}/eval/ide071/configs/workloads.json"),
             ("원장", f"{C73}/state/RUN_MANIFEST.json"), ("원장", f"{C73}/state/RUN_STATE.json"), ("원장", f"{C73}/state/execution_events.jsonl"), ("유효 GPU 프로브", f"{C73}/qwen-PROBES/bootA_retry2/D2/metrics.json"),
             ("CPU 단계", f"{C73}/qwen-PROBES/bootB/D1/cpu_jobs.csv"), ("무계측·CPU 프로브", f"{C73}/qwen-PROBES/bootA_retry/D0/metrics.json"), ("무계측·CPU 프로브", f"{C73}/qwen-PROBES/bootA_retry/D3/metrics.json"), ("실제 적용 증빙", f"{REPO}/shadow_assists/features/IDE_073/config/qwen/OPT4")]
    for k, p in items: L.append(f"| {k} | `{os.path.relpath(p, REPO)}` | {'있음' if os.path.exists(p) else '없음'} | {('%d B' % os.path.getsize(p)) if os.path.isfile(p) else ''} |")
    tr = json.load(open(f"{C73}/qwen-PROBES/bootA_retry2/D2/metrics.json"))["trace_files"]
    for n, b in tr:
        hp = os.path.expanduser(f"~/.cache/huggingface/kt/ide073/profiles/qwen/{n}"); L.append(f"| D2 trace (서버 보존) | `{hp}` (컨테이너 /models/kt/ide073/profiles/qwen/{n}) | {'있음' if os.path.exists(hp) else '없음'} | {b} B |")
    pd = f"{C73}/qwen-PROBES/bootA_retry/D3/perf.data"; L.append(f"| D3 perf.data (서버 보존) | `{os.path.relpath(pd, REPO)}` | {'있음' if os.path.exists(pd) else '없음'} | {os.path.getsize(pd) if os.path.exists(pd) else ''} B |")
    L += ["", "첨부 인계서 H/R/O 는 이번 세션에 업로드되지 않았고 지시서(PLAN.md) 만 있음. 지시서 SHA256: " + sha(f"{FEAT}/PLAN.md"), "", "환경 수집: evidence/environment.json, evidence/hotmap_budget.json, evidence/effective_config_reference.json"]
    open(f"{FEAT}/SOURCE_INDEX.md", "w").write("\n".join(L) + "\n"); print("M0 done")


if __name__ == "__main__": main()
