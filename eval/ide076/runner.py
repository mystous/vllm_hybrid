#!/usr/bin/env python3
"""IDE_076 (A1·A2·B) variant runner — PLAN.md §3.5/§4/§16.
IDE_075 하네스(eval/ide075/harness.py)를 importlib 로 로드해 캠페인 경로·원장·실행 정책만 바꾼다.
실행 정책: evidence_driven_no_count_cap — 세션/부팅/재시도/품질 횟수 guard 없음 (EXT=True 로 IDE_075 의 상한 분기를 명시적으로 비활성화; 안전 검사(HEALTH_OK·server_died·drain)는 유지).
사용: python3 runner.py <PLAN> [--variant <id>] [--env K=V ...] [--hotmap <cn path>] [--budget <cn path>] [--tag <t>]
  PLAN: R0_REF (MAIN×3, C1, LONG; OFF, vllm) | R0_PROBE (PROBE128 CORR, 원인 분석용) | 이름은 PLANS 참조
변형 정보는 variants/<variant>/manifest.json 에 기록 (build sha, env, hotmap/budget sha, mode)."""
import importlib.util, os, sys, json, time, subprocess, hashlib, argparse
HOME = os.path.expanduser("~"); REPO = f"{HOME}/projects/vllm_hybrid"
os.environ["IDE075_EXTENDED"] = "1"   # 상한 분기 비활성화 (PLAN.md §3.5: 명시적 무제한 정책)
os.environ.setdefault("IDE071_FEAT", f"{REPO}/shadow_assists/features/IDE_076")
spec = importlib.util.spec_from_file_location("h75", f"{REPO}/eval/ide075/harness.py"); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
CAMP = f"{REPO}/eval/results/IDE_076_20260917"; ST = f"{CAMP}/state"; os.makedirs(ST, exist_ok=True)
H.CAMP = CAMP; H.ST = ST; H.LEDGER = f"{ST}/execution_events.jsonl"; H.SP = f"{ST}/RUN_STATE.json"; H.L74 = "/nonexistent"   # IDE_074 원장 합산 안 함
H.KT_HOST = f"{HOME}/.cache/huggingface/kt/ide076"; H.KT_CN = "/models/kt/ide076"; H.EXT = True
subprocess.run(f"{H.DOCKER} exec {H.CN} sh -c 'mkdir -p {H.KT_CN} && chmod 777 {H.KT_CN}'", shell=True)   # 호스트 경로는 컨테이너(root) 소유 → 컨테이너에서 생성
POLICY = {"execution_policy": "evidence_driven_no_count_cap", "count_guards": "disabled (H.EXT=True; BUDGET_TOTAL 은 참조만, 종료 분기 없음)", "safety_guards_kept": ["HEALTH_OK boot verdict", "server_died → plan abort", "drain confirm", "own server only stop"]}

PLANS = {
    "R0_REF": [("OFF", "M1_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"), ("OFF", "M2_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"), ("OFF", "M3_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"),
               ("OFF", "C1_OFF", "M073_QWEN_LOW_CONCURRENCY", 1, 16, "vllm"), ("OFF", "LONG_OFF", "M073_QWEN_LONGER_PREFILL", 8, 32, "vllm")],
    "R0_PROBE": [("CORR", "P1_PROBE_CORR", "M073_QWEN_PROBE128", 64, 128, "vllmw"), ("OFF", "P2_PROBE_OFF", "M073_QWEN_PROBE128", 64, 128, "vllm")],
    "MAIN3": [("OFF", "M1_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"), ("OFF", "M2_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"), ("OFF", "M3_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm")],
    "MAIN3_C1_LONG": [("OFF", "M1_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"), ("OFF", "M2_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"), ("OFF", "M3_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"),
                      ("OFF", "C1_OFF", "M073_QWEN_LOW_CONCURRENCY", 1, 16, "vllm"), ("OFF", "LONG_OFF", "M073_QWEN_LONGER_PREFILL", 8, 32, "vllm")],
    "CORR1": [("CORR", "P1_PROBE_CORR", "M073_QWEN_PROBE128", 64, 128, "vllmw")],
    # B calibration (PLAN.md §8.2): 고정 MAIN/PROBE 와 분리된 sonnet 프롬프트(다른 seed). 서버는 recorder(stat) 로 부팅, 세션 전후 start/dump/stop.
    "CALIB": [("CORR", "CAL1_CORR", "M076_QWEN_CALIB", 64, 128, "vllmw"), ("CORR", "CAL2_CORR", "M076_QWEN_CALIB_B", 64, 128, "vllmw")],
    "SELECT": [("OFF", "SEL1_OFF", "M076_QWEN_SELECT", 64, 128, "vllm"), ("OFF", "SEL2_OFF", "M076_QWEN_SELECT", 64, 128, "vllm")],
    # B CONFIRMATION (PLAN §8.2): 고정 입력(MAIN×3/C1/LONG) + 새 seed 독립 입력 2 세션; 수치 프로브(§8.10) 는 warmup 뒤 IDE076_NUMPROBE=1 로 실행
    "CONFIRM": [("OFF", "M1_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"), ("OFF", "M2_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"), ("OFF", "M3_MAIN_OFF", "M073_QWEN_MAIN_SHORT", 64, 256, "vllm"),
                ("OFF", "C1_OFF", "M073_QWEN_LOW_CONCURRENCY", 1, 16, "vllm"), ("OFF", "LONG_OFF", "M073_QWEN_LONGER_PREFILL", 8, 32, "vllm"),
                ("OFF", "IND1_OFF", "M076_QWEN_CONFIRM", 64, 128, "vllm"), ("OFF", "IND2_OFF", "M076_QWEN_CONFIRM", 64, 128, "vllm")],
}
H.WORKLOADS["M076_QWEN_CALIB"] = {"dataset": "sonnet", "in": 512, "out": 128, "prefix": 0, "seed": 20261001, "ignore_eos": True, "cache_policy": "engine_cache_flush_before_each_rep", "note": "IDE_076 CALIBRATION: sonnet 행 조합 seed 20261001 (MAIN_SHORT 는 20260916 고정 manifest) — 고정 평가 입력과 분리"}
H.WORKLOADS["M076_QWEN_CALIB_B"] = {"dataset": "sonnet", "in": 512, "out": 128, "prefix": 0, "seed": 20261002, "ignore_eos": True, "cache_policy": "engine_cache_flush_before_each_rep", "note": "IDE_076 CALIBRATION 2차 seed"}
H.WORKLOADS["M076_QWEN_CONFIRM"] = {"dataset": "sonnet", "in": 512, "out": 128, "prefix": 0, "seed": 20261021, "ignore_eos": True, "cache_policy": "engine_cache_flush_before_each_rep", "note": "IDE_076 CONFIRMATION: 새 seed 독립 입력 (calibration·selection·MAIN 과 분리)"}
H.WORKLOADS["M076_QWEN_SELECT"] = {"dataset": "sonnet", "in": 512, "out": 128, "prefix": 0, "seed": 20261011, "ignore_eos": True, "cache_policy": "engine_cache_flush_before_each_rep", "note": "IDE_076 SELECTION: 후보 배치 선별용 (calibration·MAIN 과 분리)"}


def sha_cn(p):
    r = subprocess.run(f"{H.DOCKER} exec {H.CN} sha256sum {p}", shell=True, capture_output=True, text=True); return r.stdout.split()[0] if r.returncode == 0 and r.stdout else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("plan"); ap.add_argument("--variant", default="R0_REF"); ap.add_argument("--env", nargs="*", default=[]); ap.add_argument("--hotmap"); ap.add_argument("--budget"); ap.add_argument("--tag", default=""); ap.add_argument("--boot-id"); ap.add_argument("--unset", nargs="*", default=[], help="진단용: cfg env 에서 제거할 키 (예: KT_CALLBACK_FREE KT_CF_SKIP_EMPTY_IMM)"); ap.add_argument("--sarg", nargs="*", default=[], help="진단용: 서버 인자 추가/치환 K=V (V=true 면 bare flag)")
    a = ap.parse_args(); sessions = PLANS[a.plan]
    extra_env = dict(kv.split("=", 1) for kv in a.env)
    # 변형별 env 주입: IDE_075 boot() 는 rc73.cfg 의 env + MODE_ENV 를 쓰므로, 추가 env 는 cfg 를 감싸서 넣는다
    cfg0 = H.rc73.cfg
    def cfg_patched(model, profile):
        sa, env, gpus, pin, patch = cfg0(model, profile); sa = dict(sa); env = dict(env); env.update(extra_env)
        for k in a.unset: env.pop(k, None)
        for kv in a.sarg:
            k, v = kv.split("=", 1); sa[k] = True if v == "true" else v
        if a.hotmap: sa["init-expert-location"] = a.hotmap
        if a.budget: env["KT_GPU_EXPERTS_PER_LAYER"] = a.budget
        return sa, env, gpus, pin, patch
    H.rc73.cfg = cfg_patched
    boot_id = a.boot_id or f"{a.variant}_{a.plan}_{time.strftime('%H%M%S')}"
    if a.plan == "CALIB":   # recorder 부팅 인자·dump 디렉터리
        cfg1 = cfg_patched
        def cfg_patched(model, profile):
            sa, env, gpus, pin, patch = cfg1(model, profile); sa = dict(sa); env = dict(env); sa["expert-distribution-recorder-mode"] = "stat"; env["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = f"{H.KT_CN}/{boot_id}/rec"; return sa, env, gpus, pin, patch
        H.rc73.cfg = cfg_patched; subprocess.run(f"{H.DOCKER} exec {H.CN} sh -c 'mkdir -p {H.KT_CN}/{boot_id}/rec && chmod -R 777 {H.KT_CN}/{boot_id}'", shell=True)
    vdir = f"{CAMP}/variants/{a.variant}"; os.makedirs(vdir, exist_ok=True)
    so = sha_cn("/usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so")
    sa, env, gpus, pin, patch = cfg_patched("qwen", "OPT4")
    man = {"schema_version": "a1-a2-b-v1", "campaign_id": "IDE_076_20260917", "variant_id": a.variant, "plan": a.plan, "tag": a.tag, "boot_id": boot_id, "execution_policy": POLICY,
           "binary_sha256": so, "wrapper_experts_base_sha256": sha_cn("/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"),
           "cf_fix": (lambda o: "rearm" if "CF-rearm" in o else ("sync_only" if "CF-fix" in o else "none"))(subprocess.run(f"{H.DOCKER} exec {H.CN} grep -o 'IDE_076 CF-[a-z]*' /usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py", shell=True, capture_output=True, text=True).stdout),
           "features": {"a1_requested": extra_env.get("KT_OPT_A1_ENABLE", "unset"), "a2_requested": extra_env.get("KT_OPT_A2_ENABLE", "unset"), "a1_effective": None, "a2_effective": None},
           "placement": {"hotmap": sa.get("init-expert-location"), "hotmap_sha256": sha_cn(sa.get("init-expert-location")), "layer_budget": env.get("KT_GPU_EXPERTS_PER_LAYER"), "layer_budget_sha256": sha_cn(env.get("KT_GPU_EXPERTS_PER_LAYER")) if env.get("KT_GPU_EXPERTS_PER_LAYER") else None, "logical_hot_slots": 5952},
           "env": env, "server_args": sa, "model_revision": "003f183a92fbe5b9a8325aaa8b2ae797c91dd90f", "client": "vllm bench serve / vllmw wrapper (eval/ide075/vllm_bench_wrapper.py)", "t_start": H.now()}
    H.jdump(man, f"{vdir}/manifest_{boot_id}.json")
    H.ledger({"event": "variant_start", "variant": a.variant, "boot_id": boot_id, "plan": a.plan, "policy": POLICY["execution_policy"]})
    mode_plan = "OFF" if sessions[0][0] == "OFF" else "CORE"
    b, pids, kt, out = H.boot(mode_plan, boot_id)
    if b["verdict"] != "HEALTH_OK": print("BOOT FAILED", b); H.ledger({"event": "plan_abort", "plan": a.plan, "boot_id": boot_id, "verdict": b["verdict"]}); return 2
    H.jdump(man, f"{out}/variant_manifest.json")
    if os.environ.get("IDE076_NUMPROBE"):   # §8.10 수치 프로브 (세션 전, 성능 경로 무관; 실패해도 플랜은 계속)
        r_ = subprocess.run(f"python3 {REPO}/eval/ide076/num_probe.py --port {H.PORT} --model qwen480 --out {out}/numprobe.json", shell=True, capture_output=True, text=True, timeout=900)
        H.ledger({"event": "numprobe", "boot_id": boot_id, "rc": r_.returncode, "tail": (r_.stdout + r_.stderr)[-300:]}); print("numprobe rc", r_.returncode, flush=True)
    for spec_ in sessions:
        mode, name, wl, C, n, client = spec_
        if a.plan == "CALIB": H.sh(f"curl -s -X POST http://127.0.0.1:{H.PORT}/start_expert_distribution_record", timeout=60)
        m = H.session(mode, name, wl, C, n, boot_id, out, pids, kt, client=client); print(name, m["rc"], (m["summary"] or {}).get("output_throughput"), flush=True)
        if a.plan == "CALIB":
            H.sh(f"curl -s -X POST http://127.0.0.1:{H.PORT}/dump_expert_distribution_record", timeout=120); H.sh(f"curl -s -X POST http://127.0.0.1:{H.PORT}/stop_expert_distribution_record", timeout=60); time.sleep(5)
            import glob as _g, shutil as _sh; fs = sorted(_g.glob(f"{H.KT_HOST}/{boot_id}/rec/*.pt"), key=os.path.getmtime); os.makedirs(f"{out}/{name}/recorder", exist_ok=True)
            for f_ in fs[-4:]: _sh.copy(f_, f"{out}/{name}/recorder/")
            H.ledger({"event": "calibration_dump", "boot_id": boot_id, "session": name, "files": [os.path.basename(f_) for f_ in fs[-4:]]})
        if not m["healthy_after"]: H.ledger({"event": "server_died", "boot_id": boot_id, "after": name}); break
    H.save_server_log(out); H.stop_server(); H.sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {H.CN} --revert", timeout=120)
    H.ledger({"event": "variant_end", "variant": a.variant, "boot_id": boot_id}); print("plan done", a.plan, a.variant, boot_id); return 0


if __name__ == "__main__": sys.exit(main())
