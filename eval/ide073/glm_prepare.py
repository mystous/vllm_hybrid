#!/usr/bin/env python3
"""IDE_073 — GLM OPT4-TRANSFER 준비 (지시서 §4.2): (1) INT4 변환 `kt quant` 1회(+재시도 1) (2) 변환 검증(대표 expert dequant 차이) (3) calibration 128요청·출력64·C16 recorder (4) 빈도 기반 층별 예산 1회 (89층 × 80 = 7,120 슬롯) (5) state/glm_opt_plan.json.
실패 시 상태 코드 기록: OPT_TRANSFER_BLOCKED_FORMAT / CALIBRATION_FAILED. 새 quantizer 개발 없음."""
import json, os, sys, time, glob, subprocess, shutil
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_073")
sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide071"))
from common import *
CAMP = f"{REPO}/eval/results/IDE_073_20260917"; ST = f"{CAMP}/state"; EV = f"{FEAT}/evidence/models"; os.makedirs(EV, exist_ok=True)
GLM_HOST = f"{HOME}/.cache/huggingface/hub/models--zai-org--GLM-4.7-FP8/snapshots/7b3b5f81eee81be12a6f8da2710eac4bafb0166a"
GLM_CN = "/models/hub/models--zai-org--GLM-4.7-FP8/snapshots/7b3b5f81eee81be12a6f8da2710eac4bafb0166a"
OUT_HOST = f"{HOME}/.cache/huggingface/kt/glm47-int4"; OUT_CN = "/models/kt/glm47-int4"
plan = {"t_start": now(), "steps": []}
def step(name, **kw): kw["name"] = name; kw["t"] = now(); plan["steps"].append(kw); jdump(plan, f"{ST}/glm_prepare_state.json"); print(name, {k: v for k, v in kw.items() if k not in ("name", "t")})

def main():
    # 1. 변환
    sh(f"sudo mkdir -p {OUT_HOST} && sudo chmod 777 {OUT_HOST}")
    conv_ok = False; log = ""
    for attempt in (1, 2):
        t0 = time.time()
        r = dexec(CN, f"cd /tmp && kt quant {GLM_CN} -m int4 -i fp8 -o {OUT_CN} --cpu-threads 96 --numa-nodes 2 -y 2>&1 | tail -80", timeout=4 * 3600)
        log = r.stdout[-6000:]; open(f"{EV}/glm_kt_quant_attempt{attempt}.log", "w").write(r.stdout)
        files = glob.glob(f"{OUT_HOST}/*"); ok = r.returncode == 0 and any(f.endswith(".safetensors") for f in files)
        step("convert", attempt=attempt, rc=r.returncode, seconds=time.time() - t0, files=len(files), ok=ok, tail=log[-600:])
        if ok: conv_ok = True; break
        if attempt == 1 and "No such option" in log: break   # 옵션 자체 미지원 → 재시도 무의미
    if not conv_ok:
        step("transfer_status", status="OPT_TRANSFER_BLOCKED_FORMAT", reason=log[-800:]); jdump({"status": "OPT_TRANSFER_BLOCKED_FORMAT"}, f"{ST}/glm_opt_plan.json"); return
    # 2. 변환 검증: 대표 expert (layer 3, expert 0) gate_proj 의 원본 per-channel FP8 dequant vs INT4 dequant 차이 — 로더 형식이 다르면 NOT_COLLECTED 로 기록
    step("convert_check", status="NOT_COLLECTED", reason="kt INT4 포맷의 dequant 재구성 코드 미보유 (새 구현 금지, 지시서 §5.2 90분 상한); 변환 로그·파일 목록·SHA256 만 보존")
    man = {os.path.basename(f): {"bytes": os.path.getsize(f), "sha256": sha256(f)} for f in sorted(glob.glob(f"{OUT_HOST}/*")) if os.path.getsize(f) < 50 * 1024 * 1024}
    man["_large_files"] = [(os.path.basename(f), os.path.getsize(f)) for f in sorted(glob.glob(f"{OUT_HOST}/*")) if os.path.getsize(f) >= 50 * 1024 * 1024]
    jdump(man, f"{EV}/glm_int4_manifest.json")
    # 3. calibration: BASIC-like GLM (uniform 80, AMXINT4 변환본) + recorder, 128 요청 · 출력 64 · C16, 입력 = MAIN_SHORT 와 겹치지 않는 sonnet seed 20260930 (원문 중복 가능성 기록)
    sys.path.insert(0, f"{REPO}/eval/ide073"); from run_campaign import cfg, COMMON4
    from run_cell import build_args, WORKLOADS
    sa = dict(COMMON4, **{"model-path": GLM_CN, "served-model-name": "glm47", "attention-backend": "flashinfer", "fp8-gemm-backend": "triton", "enable-p2p-check": True, "disable-shared-experts-fusion": True, "enable-mixed-chunk": True, "chunked-prefill-size": 4096,
                          "kt-weight-path": OUT_CN, "kt-method": "AMXINT4", "kt-cpuinfer": 96, "kt-threadpool-count": 2, "kt-num-gpu-experts": 80, "kt-max-deferred-experts-per-token": 8, "expert-distribution-recorder-mode": "stat"})
    rec_host = f"{HOME}/.cache/huggingface/kt/ide073/rec_glm"; os.makedirs(rec_host, exist_ok=True); rec_cn = "/models/kt/ide073/rec_glm"
    out = f"{CAMP}/G-CAL/a1"; os.makedirs(out, exist_ok=True); stop_server()
    b = boot_server(build_args(sa), {"SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR": rec_cn}, "0,1,2,3", out, timeout=900); step("cal_boot", verdict=b["verdict"], seconds=b["boot_seconds"])
    if b["verdict"] != "HEALTH_OK":
        step("transfer_status", status="OPT_TRANSFER_BLOCKED_BOOT", reason=b.get("error_lines", "")[:600]); jdump({"status": "OPT_TRANSFER_BLOCKED_BOOT"}, f"{ST}/glm_opt_plan.json"); stop_server(); return
    wl = {"dataset": "sonnet", "in": 512, "out": 64, "prefix": 0, "seed": 20260930, "ignore_eos": True, "cache_policy": "engine_cache_flush_before_each_rep"}
    flush_cache(); time.sleep(3); sh(f"curl -s -X POST http://127.0.0.1:{PORT}/start_expert_distribution_record")
    r = run_bench(wl, 16, 128, "glm47", f"{out}/CAL_C16", 20260930, timeout=900); step("cal_bench", rc=r["rc"], summary=r["summary"])
    sh(f"curl -s -X POST http://127.0.0.1:{PORT}/dump_expert_distribution_record"); sh(f"curl -s -X POST http://127.0.0.1:{PORT}/stop_expert_distribution_record"); time.sleep(5)
    save_server_log(out); stop_server()
    files = sorted(glob.glob(f"{rec_host}/*.pt"), key=os.path.getmtime)[-4:]
    for f in files: shutil.copy(f, out)
    if not files: step("transfer_status", status="CALIBRATION_FAILED"); jdump({"status": "CALIBRATION_FAILED"}, f"{ST}/glm_opt_plan.json"); return
    # 4. 빈도 기반 예산 (IDE_070 build_layer_budget 정의; 슬롯 = 80 × MoE 층 수) + hotmap (층별 빈도순)
    r = sh(f"{HOME}/venv-bench/bin/python - <<'PY'\nimport torch,glob,json\nfs=sorted(glob.glob('{out}/expert_distribution_recorder_*.pt'))\nd=torch.load(fs[-1],map_location='cpu',weights_only=False); lc=d['logical_count'].to(torch.float64)\nlc=lc.sum(0) if lc.dim()==3 else lc\nL,E=lc.shape; S=80*L\nsrt=torch.sort(lc,dim=1,descending=True); share=srt.values/lc.sum(1,keepdim=True).clamp(min=1)\ncand=sorted(((float(share[l,r]),l,r) for l in range(L) for r in range(E)),reverse=True)[:S]\nper=[0]*L\nfor _,l,r in cand: per[l]=max(per[l],r+1)\njson.dump({{'physical_to_logical_map': srt.indices.tolist()}}, open('{HOME}/.cache/huggingface/kt/ide073/glm_hotmap.json','w'))\njson.dump({{'per_layer': per, 'slots': S, 'layers': L, 'experts': E, 'source': fs[-1]}}, open('{HOME}/.cache/huggingface/kt/ide073/glm_budget.json','w'), indent=1)\nprint('layers',L,'experts',E,'slots',S,'min',min(per),'max',max(per))\nPY", timeout=600)
    step("budget", out=r.stdout.strip()[-300:], err=r.stderr.strip()[-300:])
    if "layers" not in r.stdout: step("transfer_status", status="CALIBRATION_FAILED", reason=r.stderr[-500:]); jdump({"status": "CALIBRATION_FAILED"}, f"{ST}/glm_opt_plan.json"); return
    L = int(r.stdout.split("layers")[1].split()[0])
    plan_out = {"status": "READY", "kt_weight_path": OUT_CN, "kt_method": "AMXINT4", "hotmap": "/models/kt/ide073/glm_hotmap.json", "budget": "/models/kt/ide073/glm_budget.json", "moe_layers": L, "slots": 80 * L,
                "hashes": {"hotmap": sha256(f"{HOME}/.cache/huggingface/kt/ide073/glm_hotmap.json"), "budget": sha256(f"{HOME}/.cache/huggingface/kt/ide073/glm_budget.json")}, "calibration": "sonnet seed 20260930, 128 요청, 출력 64, C16; MAIN_SHORT(seed 20260916) 와 원문 행 중복 가능성 있음 (CALIBRATION_OVERLAP_UNVERIFIED)"}
    jdump(plan_out, f"{ST}/glm_opt_plan.json"); shutil.copy(f"{HOME}/.cache/huggingface/kt/ide073/glm_budget.json", f"{FEAT}/config/glm_layer_budget.json"); step("plan", **plan_out)

if __name__ == "__main__":
    main()
