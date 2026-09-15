#!/usr/bin/env python3
"""IDE_071 P6 — CALIBRATION 트레이스 수집 (DIAGNOSTIC 셀: expert-distribution recorder ON).
앵커 구성 + `--expert-distribution-recorder-mode stat` 으로 부팅, CALIBRATION 입력 (sonnet seed 20260930, prefix 0, 512/128, C64, 512 요청) 을 흘리고 dump.
→ recorder .pt 를 eval/results/<camp>/CAL00_recorder/ 에 저장. 처리량은 DIAGNOSTIC 으로만 기록 (일반 성능표 제외).
사용: run_calibration.py <campaign_dir> <anchor_spec.json>
"""
import json, os, sys, time, glob, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from run_cell import build_args, R0, WORKLOADS

def main():
    camp, spec_p = sys.argv[1], sys.argv[2]; spec = json.load(open(spec_p))
    out = f"{camp}/CAL00_recorder/a1"; os.makedirs(out, exist_ok=True)
    rec_host = f"{HOME}/.cache/huggingface/kt/ide071/rec_cal"; os.makedirs(rec_host, exist_ok=True); rec_cn = "/models/kt/ide071/rec_cal"
    sa = dict(R0["server_args"]); sa.update(spec.get("server_args", {})); sa["expert-distribution-recorder-mode"] = "stat"
    env = dict(spec.get("env", {})); env["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = rec_cn
    st = {"cell_id": "CAL00_recorder", "kind": "DIAGNOSTIC", "t": now(), "anchor": spec["cell_id"]}
    stop_server()
    b = boot_server(build_args(sa), env, spec.get("gpus", "0,1,2,3"), out, timeout=1500); st["boot"] = b
    if b["verdict"] != "HEALTH_OK": st["exit_status"] = "FAILED_BOOT"; jdump(st, f"{out}/status.json"); stop_server(); return
    jdump(server_info(), f"{out}/server_info.json")
    model = sa["served-model-name"]
    run_bench(dict(WORKLOADS["SHORT_COLD"]), 16, 32, model, f"{out}/warmup", 1, timeout=900)
    flush_cache(); time.sleep(3)
    sh(f"curl -s -X POST http://127.0.0.1:{PORT}/start_expert_distribution_record")
    wl = dict(WORKLOADS["SHORT_COLD"]); wl["seed"] = 20260930   # CALIBRATION 입력 (HOLDOUT seed 20260916 과 분리)
    r = run_bench(wl, 64, 512, model, f"{out}/CAL_C64", 20260930, timeout=1800); st["bench"] = r["summary"]
    sh(f"curl -s -X POST http://127.0.0.1:{PORT}/dump_expert_distribution_record"); sh(f"curl -s -X POST http://127.0.0.1:{PORT}/stop_expert_distribution_record")
    time.sleep(5)
    files = sorted(glob.glob(f"{rec_host}/*.pt"), key=os.path.getmtime)[-4:]
    for f in files: shutil.copy(f, out)
    st["recorder_files"] = [os.path.basename(f) for f in files]; st["exit_status"] = "COMPLETED" if files else "FAILED_RUNTIME"
    save_server_log(out); stop_server(); jdump(st, f"{out}/status.json"); print(st["exit_status"], files)

if __name__ == "__main__":
    main()
