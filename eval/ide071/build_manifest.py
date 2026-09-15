#!/usr/bin/env python3
"""IDE_071 — 셀 manifest 생성 (manifests/cells.jsonl). 단계별 셀 정의. 앵커(선택 구성) 가 필요한 단계는 --anchor <cell_json> 로 받는다.
사용: build_manifest.py <campaign_dir> <phase> [--anchor path] [--out cells_<phase>.jsonl]
"""
import copy, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

EPOCH_ENV = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "KT_AVX_RB": "1", "KT_AVX_PF": "2", "KT_FUSE_QIN": "1", "KT_AMX_MIN_QLEN": "1000000", "KT_AMX_MIN_ROWS": "3",
             "KT_CALLBACK_FREE": "1", "KT_COLD_DEFER": "1", "KT_COLD_TAU": "0.25"}
EPOCH_ARGS = {"init-expert-location": "/models/kt/ide070/hotmap_mixed_0.25.json", "kt-max-deferred-experts-per-token": 8, "mem-fraction-static": 0.94,
              "chunked-prefill-size": 4096, "kv-cache-dtype": "fp8_e5m2", "cuda-graph-max-bs": 224, "cuda-graph-bs": [32, 64, 96, 128, 160, 192, 224], "max-total-tokens": 143360}
HOTMAP_V1 = "/models/kt/ide069/hotmap.json"; HOTMAP_V2 = "/models/kt/ide070/hotmap_v2.json"; BUDGET_5952 = "/models/kt/ide070/layer_budget_5952.json"

def cell(cid, phase, parent=None, kind="CONFIG", sem="SEMANTICS_UNVERIFIED", prec="GPU_FP8_CPU_INT4_KV_BF16", workloads=None, env=None, sa=None, **kw):
    c = {"cell_id": cid, "phase_id": phase, "parent_cell_id": parent, "change_kind": kind, "semantics_family": sem, "precision_family": prec,
         "workloads": workloads or [{"workload_id": "SHORT_COLD", "C": 64, "reps": 3}], "env": env or {}, "server_args": sa or {}, "gpus": "0,1,2,3", "status": "PENDING"}
    c.update(kw); return c

def p1():
    L = [{"workload_id": "LEGACY_070", "C": 64, "reps": 3}]; S = [{"workload_id": "SHORT_COLD", "C": 64, "reps": 3}]
    cf = {"KT_CALLBACK_FREE": "1"}; cfs = {"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1"}
    return [
        cell("R00_r0_def4", "P1", None, workloads=L + S, note="R0 재현. LEGACY_070 3회 + SHORT_COLD 3회"),
        cell("R01_r0_def0", "P1", "R00_r0_def4", sem="SEMANTICS_PRESERVING", workloads=S, sa={"kt-max-deferred-experts-per-token": 0}, note="deferral 0, KT_COLD_DEFER/TAU 미설정. 정확성 기준 출력 (accuracy 단계에서 별도)"),
        cell("R02_d1_cf", "P1", "R00_r0_def4", workloads=L, env=cf, note="IDE_070 d1_cf 재현"),
        cell("R03_d3_cf_skip_def8_pin", "P1", "R00_r0_def4", workloads=L + S, env=cfs, sa={"kt-max-deferred-experts-per-token": 8}, pin_nonkt=True, note="IDE_070 d3 재현"),
        cell("R04_d4_epoch", "P1", "R00_r0_def4", prec="GPU_FP8_CPU_INT4_KV_FP8E5M2", workloads=L + S, env=dict(EPOCH_ENV), sa=dict(EPOCH_ARGS), note="IDE_070 d4 재현 (KT_CF_SKIP_EMPTY_IMM·pin 없음 — 지시서 §1.1)"),
        cell("R05_v2_u96", "P1", "R00_r0_def4", workloads=S, sa={"init-expert-location": HOTMAP_V2}, note="hotmap v2 uniform 96"),
        cell("R06_v2_nu5952", "P1", "R05_v2_u96", kind="PATCH", workloads=S, sa={"init-expert-location": HOTMAP_V2}, env={"KT_GPU_EXPERTS_PER_LAYER": BUDGET_5952}, patch="per_layer", expected_slots=5952, note="층별 예산 5,952 + 패치. 서버 로그의 층별 62개 값 합계로 적용 확인"),
        cell("R07_gpu_only_tp8_ep8", "P1", None, prec="GPU_FP8_ALL_GPU_KV_BF16", workloads=[{"workload_id": "SHORT_COLD", "C": 32, "reps": 3}, {"workload_id": "SHORT_COLD", "C": 64, "reps": 3}],
             gpus="0,1,2,3,4,5,6,7", replace_server_args=True,
             sa={"model-path": MODEL_FP8_CN, "served-model-name": "q480", "host": "127.0.0.1", "port": 30000, "tp": 8, "ep-size": 8, "attention-backend": "triton", "trust-remote-code": True,
                 "context-length": 32768, "cuda-graph-max-bs": 64, "cuda-graph-backend-prefill": "disabled", "mem-fraction-static": 0.90, "max-total-tokens": 131072},
             note="GPU-only 참조군 (IDE_069 TSK_053 과 동일 인자: tp8 ep8 mem 0.90 KV 131,072) (GPU 8, CPU expert 없음). GPU 4~7 미사용 확인 후 실행"),
    ]

def p2():
    out = []
    for cfv, skip, pin, d in itertools.product([0, 1], [0, 1], [0, 1], [0, 4, 8]):
        if skip and not cfv: continue
        env = {}
        if cfv: env["KT_CALLBACK_FREE"] = "1"
        if skip: env["KT_CF_SKIP_EMPTY_IMM"] = "1"
        cid = f"P2_cf{cfv}_skip{skip}_pin{pin}_def{d}"
        c = cell(cid, "P2", "R00_r0_def4", env=env, sa={"kt-max-deferred-experts-per-token": d}, pin_nonkt=bool(pin),
                 sem="SEMANTICS_PRESERVING" if d == 0 else "SEMANTICS_UNVERIFIED",
                 required_runtime_proofs=["callback_free_path_used", "skip_empty_effective_state", "all_rank_tid_affinity", "per_layer_deferred_limit"],
                 note="skip-empty 는 소스상 deferred ≥ top-k(8) 이고 callback-free 일 때만 활성 (experts_base.py:853) — def0/4 에서는 NO_EFFECTIVE_PATH 로 표기")
        if skip and d < 8: c["expected_effect"] = "NO_EFFECTIVE_PATH_skip_empty"
        out.append(c)
    # D2 재현 (= P2_cf1_skip1_pin0_def8) 은 행렬에 포함. overlap scheduler 교차 2셀
    out.append(cell("P2X_cf0_overlap_off", "P2", "P2_cf0_skip0_pin0_def4", sa={"disable-overlap-schedule": True}, note="overlap scheduler OFF × callback-free OFF"))
    out.append(cell("P2X_cf1_overlap_off", "P2", "P2_cf1_skip0_pin0_def4", env={"KT_CALLBACK_FREE": "1"}, sa={"disable-overlap-schedule": True}, note="overlap scheduler OFF × callback-free ON"))
    return out

def p3(anchor):
    """CPU 커널·스레드 (지시서 §7.1, §7.3). anchor = P2 에서 선택된 셀 spec."""
    out = []; base_env = dict(anchor.get("env", {})); base_sa = dict(anchor.get("server_args", {})); pin = anchor.get("pin_nonkt", False)
    def mk(cid, env_add=None, sa_add=None, note=""):
        e = dict(base_env); s = dict(base_sa)
        for k, v in (env_add or {}).items():
            if v is None: e.pop(k, None)
            else: e[k] = v
        s.update(sa_add or {})
        return cell(cid, "P3", anchor["cell_id"], env=e, sa=s, pin_nonkt=pin, note=note)
    for v in ["0", "1"]: out.append(mk(f"P3_avx_rb_{v}", {"KT_AVX_RB": v}, note="KT_AVX_RB: getenv!=NULL 이면 ON (값 무관, amx_kernels.hpp:1769) — 0 도 ON"))
    for v in ["0", "1", "2", "4"]: out.append(mk(f"P3_avx_pf_{v}", {"KT_AVX_PF": v}, note="KT_AVX_PF: atoi, prefetch distance (amx_kernels.hpp:1773)"))
    for v in ["0", "1"]: out.append(mk(f"P3_fuse_qin_{v}", {"KT_FUSE_QIN": v}, note="KT_FUSE_QIN: getenv!=NULL 이면 ON (moe_base.hpp:511)"))
    for v in ["1", "2", "3", "4", "8", "16"]: out.append(mk(f"P3_amx_min_rows_{v}", {"KT_AMX_MIN_ROWS": v}, note="AMX if qlen>MIN_QLEN or m>=MIN_ROWS (moe.hpp:240-252); 기본 1<<30"))
    for v in ["32", "64", "128", "256", "512", "1024", "1000000"]: out.append(mk(f"P3_amx_min_qlen_{v}", {"KT_AMX_MIN_QLEN": v}, note="기본 4*expert_num/topk = 80"))
    for w in [80, 88, 104, 108, 112]: out.append(mk(f"P3_cpuinfer_{w}", None, {"kt-cpuinfer": w}, note="워커 수"))
    for lib in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        for v in ["1", "4"]: out.append(mk(f"P3_{lib.lower()}_{v}", {lib: v}, note="§7.3 병렬 라이브러리 스레드"))
    for v in [1, 2, 4]: out.append(mk(f"P3_tokenizer_workers_{v}", None, {"tokenizer-worker-num": v, "detokenizer-worker-num": v}, note="§9.5 tokenizer/detokenizer worker"))
    return out

def p4(anchor):
    out = []; base_env = dict(anchor.get("env", {})); base_sa = dict(anchor.get("server_args", {})); pin = anchor.get("pin_nonkt", False)
    def mk(cid, sa_add=None, prec=None, note="", wl=None):
        s = dict(base_sa); s.update(sa_add or {})
        return cell(cid, "P4", anchor["cell_id"], env=dict(base_env), sa=s, pin_nonkt=pin, prec=prec or anchor.get("precision_family"), note=note, workloads=wl)
    for dt in ["fp8_e5m2", "fp8_e4m3"]: out.append(mk(f"P4_kv_{dt}_tok40960", {"kv-cache-dtype": dt}, prec="GPU_FP8_CPU_INT4_KV_" + dt.upper(), note="token budget 고정"))
    for dt in ["fp8_e5m2"]: out.append(mk(f"P4_kv_{dt}_tok81920", {"kv-cache-dtype": dt, "max-total-tokens": 81920}, prec="GPU_FP8_CPU_INT4_KV_" + dt.upper(), note="HBM budget 고정 (bf16 40,960 tokens ≈ fp8 81,920 tokens)"))
    for t in [24576, 57344]: out.append(mk(f"P4_kvtok_{t}", {"max-total-tokens": t}, note="KV token pool"))
    for mf in [0.90, 0.92, 0.94, 0.96]: out.append(mk(f"P4_memfrac_{int(mf*100)}", {"mem-fraction-static": mf}, note="mem-fraction (실제 할당 비교)"))
    for mb in [96, 128, 224]: out.append(mk(f"P4_graph_max_{mb}", {"cuda-graph-max-bs": mb}, note="decode graph max batch"))
    out.append(mk("P4_graph_G1", {"cuda-graph-bs": [1, 2, 4, 8, 16, 24, 32, 48, 64]}, note="G1 목록"))
    out.append(mk("P4_graph_G2", {"cuda-graph-bs": [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128], "cuda-graph-max-bs": 128}, note="G2 목록"))
    out.append(mk("P4_graph_G3", {"cuda-graph-bs": [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256], "cuda-graph-max-bs": 256}, note="G3 목록"))
    out.append(mk("P4_graph_nopad", {"disable-cuda-graph-padding": True}, note="graph padding 비활성"))
    out.append(mk("P4_graph_off", {"disable-cuda-graph": True}, note="graph OFF (eager)"))
    # 경계 실험: C63/64/65/80 × graph max 64 vs 128 (KV 81920 fp8 로 충분한 KV)
    for gm in [64, 128]:
        out.append(mk(f"P4B_graphmax{gm}_kv81920", {"cuda-graph-max-bs": gm, "kv-cache-dtype": "fp8_e5m2", "max-total-tokens": 81920}, prec="GPU_FP8_CPU_INT4_KV_FP8E5M2",
                      wl=[{"workload_id": "SHORT_COLD", "C": c, "reps": 2} for c in (63, 64, 65, 80)], note="§8.4 경계 실험"))
    return out

def p5(anchor):
    out = []; base_env = dict(anchor.get("env", {})); base_sa = dict(anchor.get("server_args", {})); pin = anchor.get("pin_nonkt", False)
    def mk(cid, sa_add=None, note="", wl=None, env_add=None):
        s = dict(base_sa); s.update(sa_add or {}); e = dict(base_env); e.update(env_add or {})
        return cell(cid, "P5", anchor["cell_id"], env=e, sa=s, pin_nonkt=pin, prec=anchor.get("precision_family"), note=note, workloads=wl)
    PH = [{"workload_id": "SHORT_COLD", "C": 64, "reps": 3}, {"workload_id": "PREFILL_HEAVY", "C": 16, "reps": 2}]
    for v in [512, 1024, 2048, 4096, 8192]: out.append(mk(f"P5_chunk_{v}", {"chunked-prefill-size": v}, note="chunked prefill", wl=PH))
    for v in [1, 4, 8, 16]: out.append(mk(f"P5_prefill_max_req_{v}", {"prefill-max-requests": v}, note="prefill batch 요청 상한"))
    for v in [4096, 8192, 16384]: out.append(mk(f"P5_max_prefill_tokens_{v}", {"max-prefill-tokens": v}, note="max prefill tokens"))
    out.append(mk("P5_mixed_chunk_on", {"enable-mixed-chunk": True}, note="mixed chunk"))
    for v in [32, 48, 64]: out.append(mk(f"P5_max_running_{v}", {"max-running-requests": v}, note="max running requests (client C64 유지)"))
    for v in [0.6, 0.8, 1.2, 1.4]: out.append(mk(f"P5_conserv_{int(v*10)}", {"schedule-conservativeness": v}, note="schedule conservativeness"))
    for v in ["fcfs", "lpm"]: out.append(mk(f"P5_policy_{v}", {"schedule-policy": v}, note="schedule policy"))
    for v in [2, 4, 8]: out.append(mk(f"P5_decode_steps_{v}", {"num-continuous-decode-steps": v}, note="continuous decode steps"))
    for v in [2, 4]: out.append(mk(f"P5_recv_interval_{v}", {"scheduler-recv-interval": v}, note="scheduler receive interval"))
    for v in [1, 2, 4, 8, 16]: out.append(mk(f"P5_triton_kvsplit_{v}", {"triton-attention-num-kv-splits": v}, note="Triton KV splits"))
    for v in [16, 32, 64]: out.append(mk(f"P5_page_{v}", {"page-size": v}, note="KV page size"))
    for v in ["fa3", "flashinfer"]: out.append(mk(f"P5_attn_{v}", {"attention-backend": v}, note="attention backend (최소 실행 검증 포함)"))
    out.append(mk("P5_attn_prefill_fa3_decode_flashinfer", {"prefill-attention-backend": "fa3", "decode-attention-backend": "flashinfer"}, note="단계별 backend"))
    for v in ["triton", "flashinfer_trtllm", "deep_gemm"]: out.append(mk(f"P5_moe_runner_{v}", {"moe-runner-backend": v}, note="GPU hot expert MoE runner (kt wrapper 경유 여부 확인)"))
    out.append(mk("P5_no_custom_allreduce", {"disable-custom-all-reduce": True}, note="NCCL all-reduce 경로"))
    out.append(mk("P5_flashinfer_allreduce_fusion", {"enable-flashinfer-allreduce-fusion": True}, note="all-reduce+RMSNorm fusion"))
    for v in [1, 2, 4]: out.append(mk(f"P5_tok_workers_{v}", {"tokenizer-worker-num": v, "detokenizer-worker-num": v}, note="tokenizer/detokenizer worker"))
    out.append(mk("P5_tok_batch_encode", {"enable-tokenizer-batch-encode": True}, note="batch encode"))
    return out

def p6(anchor):
    out = []; base_env = dict(anchor.get("env", {})); base_sa = dict(anchor.get("server_args", {})); pin = anchor.get("pin_nonkt", False)
    def mk(cid, sa_add=None, env_add=None, note="", **kw):
        s = dict(base_sa); s.update(sa_add or {}); e = dict(base_env); e.update(env_add or {})
        return cell(cid, "P6", anchor["cell_id"], env=e, sa=s, pin_nonkt=pin, prec=anchor.get("precision_family"), note=note, **kw)
    out.append(mk("PL00_v1_u96", {"init-expert-location": HOTMAP_V1, "kt-num-gpu-experts": 96}, note="hotmap v1 uniform 96"))
    out.append(mk("PL01_v2_u96", {"init-expert-location": HOTMAP_V2, "kt-num-gpu-experts": 96}, note="hotmap v2 uniform 96"))
    out.append(mk("PL02_v2_nu5952", {"init-expert-location": HOTMAP_V2, "kt-num-gpu-experts": 96}, {"KT_GPU_EXPERTS_PER_LAYER": BUDGET_5952}, note="v2 + 기존 층별 예산", patch="per_layer", expected_slots=5952, kind="PATCH"))
    for tag in ["PL03_cal_freq", "PL04_cal_coldfrac", "PL05_cal_waitreduce"]:
        out.append(mk(tag, {"init-expert-location": f"/models/kt/ide071/{tag}_hotmap.json", "kt-num-gpu-experts": 96}, {"KT_GPU_EXPERTS_PER_LAYER": f"/models/kt/ide071/{tag}_budget.json"},
                      note="새 CALIBRATION 트레이스 기반 (build_hotmap.py 생성)", patch="per_layer", expected_slots=5952, kind="PATCH", requires_files=[f"{HOME}/.cache/huggingface/kt/ide071/{tag}_hotmap.json"]))
    for a in [0.25, 0.5, 0.75]:
        out.append(mk(f"PL06_static_alpha{int(a*100)}", {"init-expert-location": f"/models/kt/ide071/hotmap_alpha{int(a*100)}.json", "kt-num-gpu-experts": 96}, note="prefill/decode 가중 정적 map (새 트레이스)", requires_files=[f"{HOME}/.cache/huggingface/kt/ide071/hotmap_alpha{int(a*100)}.json"]))
    for u in [88, 92, 98, 100]:
        out.append(mk(f"PL07_u{u}_kvfixed", {"kt-num-gpu-experts": u}, note=f"uniform {u} (슬롯 {u*62}), KV 고정 40,960 — HBM 초과 시 FAILED_BOOT 기록"))
    return out

def p7(anchor):
    out = []; base_env = dict(anchor.get("env", {})); base_sa = dict(anchor.get("server_args", {})); pin = anchor.get("pin_nonkt", False)
    def mk(cid, sa_add=None, env_add=None, note="", **kw):
        s = dict(base_sa); s.update(sa_add or {}); e = dict(base_env); e.update(env_add or {})
        return cell(cid, "P7", anchor["cell_id"], env=e, sa=s, pin_nonkt=pin, prec=anchor.get("precision_family"), note=note, **kw)
    out.append(mk("P7_numa_interleave", note="numactl --interleave=all", launch_prefix="numactl --interleave=all"))
    out.append(mk("P7_numa_membind0", note="numactl --membind=0 (대조군)", launch_prefix="numactl --membind=0"))
    out.append(mk("P7_pool1", {"kt-threadpool-count": 1}, note="threadpool 1 (대조군)"))
    out.append(mk("P7_hugepage", None, {"KT_HUGEPAGE": "1"}, note="kt madvise hugepage (moe_base.hpp:16)"))
    out.append(mk("P7_gpu0145", note="GPU 0,1,4,5 대조군 1회", gpus="0,1,4,5"))
    return out


AWQ_CN = "/models/hub/models--QuantTrio--Qwen3-Coder-480B-A35B-Instruct-AWQ/snapshots/9ce3eaa67fe88609afec235117e97eb03d9b3cda"
EAGLE3_CN = "/models/hub/models--lmsys--SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI/snapshots/b2a87592e335783be6c617c3be93ceeab99334a2"
Q4B_CN = "/models/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

def p8(anchor):
    """W4 (AWQ) 단계별 분리 (§12.1) + TP/EP 구성 (§12.3). 작은 요청 셀은 SHORT_COLD C1/2/4/8/16 을 각 4 요청씩."""
    out = []
    small = [{"workload_id": "SHORT_COLD", "C": c, "n": max(4, c), "reps": 1} for c in (1, 2, 4, 8, 16)]
    gpu_only = {"model-path": AWQ_CN, "served-model-name": "q480awq", "host": "127.0.0.1", "port": 30000, "tp": 4, "attention-backend": "triton", "trust-remote-code": True,
                "context-length": 32768, "mem-fraction-static": 0.90, "max-total-tokens": 16384}
    out.append(cell("W01_awq_gpuonly_tp4_graphoff", "P8", None, kind="FORMAT", prec="GPU_AWQ4_ALL_GPU_KV_BF16", workloads=small, replace_server_args=True, sa=dict(gpu_only, **{"disable-cuda-graph": True}), note="AWQ GPU-only TP4, 작은 KV, graph OFF, kt 없음"))
    out.append(cell("W02_awq_gpuonly_tp4_graphoff_half", "P8", "W01_awq_gpuonly_tp4_graphoff", kind="FORMAT", prec="GPU_AWQ4_ALL_GPU_KV_FP16", workloads=small, replace_server_args=True, sa=dict(gpu_only, **{"disable-cuda-graph": True, "dtype": "half"}), note="dtype half"))
    out.append(cell("W02b_awq_gpuonly_tp4_graphoff_awq_marlin", "P8", "W01_awq_gpuonly_tp4_graphoff", kind="FORMAT", prec="GPU_AWQ4_ALL_GPU_KV_BF16", workloads=small, replace_server_args=True, sa=dict(gpu_only, **{"disable-cuda-graph": True, "quantization": "awq_marlin"}), note="runner 명시 awq_marlin"))
    out.append(cell("W03_awq_gpuonly_tp4_graphon", "P8", "W01_awq_gpuonly_tp4_graphoff", kind="FORMAT", prec="GPU_AWQ4_ALL_GPU_KV_BF16", workloads=small + [{"workload_id": "SHORT_COLD", "C": 64, "reps": 2}], replace_server_args=True, sa=dict(gpu_only, **{"cuda-graph-max-bs": 64, "cuda-graph-backend-prefill": "disabled"}), note="GPU-only graph ON"))
    hyb = {"model-path": AWQ_CN, "served-model-name": "q480awq", "host": "127.0.0.1", "port": 30000, "tp": 4, "attention-backend": "triton", "trust-remote-code": True, "context-length": 32768,
           "kt-weight-path": "/models/kt/qwen3-480b-int4", "kt-method": "AMXINT4", "kt-cpuinfer": 96, "kt-threadpool-count": 2, "kt-num-gpu-experts": 96, "init-expert-location": HOTMAP_V1,
           "kt-max-deferred-experts-per-token": 4, "ep-dispatch-algorithm": "dynamic", "mem-fraction-static": 0.95, "max-total-tokens": 40960}
    out.append(cell("W04_awq_hybrid_h96_graphoff", "P8", "W01_awq_gpuonly_tp4_graphoff", kind="FORMAT", prec="GPU_AWQ4_CPU_INT4_KV_BF16", workloads=small, replace_server_args=True, sa=dict(hyb, **{"disable-cuda-graph": True}), note="MIXED_WEIGHT_PROVENANCE (GPU AWQ / CPU INT4 from FP8)"))
    out.append(cell("W05_awq_hybrid_h96_graphon", "P8", "W04_awq_hybrid_h96_graphoff", kind="FORMAT", prec="GPU_AWQ4_CPU_INT4_KV_BF16", workloads=small + [{"workload_id": "SHORT_COLD", "C": 64, "reps": 3}], replace_server_args=True, sa=dict(hyb, **{"cuda-graph-max-bs": 64, "cuda-graph-backend-prefill": "disabled"}), note=""))
    for h in (112, 128, 144):
        out.append(cell(f"W06_awq_hybrid_h{h}_graphon", "P8", "W05_awq_hybrid_h96_graphon", kind="FORMAT", prec="GPU_AWQ4_CPU_INT4_KV_BF16", replace_server_args=True, sa=dict(hyb, **{"kt-num-gpu-experts": h, "cuda-graph-max-bs": 64, "cuda-graph-backend-prefill": "disabled"}), note=f"hot {h}"))
    out.append(cell("W07_awq_gpuonly_tp4_all160", "P8", "W03_awq_gpuonly_tp4_graphon", kind="FORMAT", prec="GPU_AWQ4_ALL_GPU_KV_BF16", replace_server_args=True, sa=dict(gpu_only, **{"cuda-graph-max-bs": 64, "cuda-graph-backend-prefill": "disabled", "max-total-tokens": 40960}), note="GPU-only all160 W4, KV 40,960"))
    # TP/EP (§12.3) — 앵커 구성 기준
    base_env = dict(anchor.get("env", {})); base_sa = dict(anchor.get("server_args", {})); pin = anchor.get("pin_nonkt", False)
    def mk(cid, sa_add=None, note="", **kw):
        s = dict(base_sa); s.update(sa_add or {}); return cell(cid, "P8", anchor["cell_id"], env=dict(base_env), sa=s, pin_nonkt=pin, prec=anchor.get("precision_family"), note=note, **kw)
    out.append(mk("T01_tp4_ep4_hybrid", {"ep-size": 4}, note="TP4+EP4 하이브리드 (kt wrapper 지원 여부는 부팅으로 판정)"))
    out.append(mk("T02_tp4_dp_attention", {"enable-dp-attention": True, "dp-size": 4}, note="DP attention (지원 여부)"))
    out.append(mk("T03_tp2_hybrid", {"tp": 2}, note="TP2 하이브리드 단일 (HBM 성립 여부)", gpus="0,1"))
    return out

def p9(anchor):
    out = []; base_env = dict(anchor.get("env", {})); base_sa = dict(anchor.get("server_args", {})); pin = anchor.get("pin_nonkt", False)
    W = [{"workload_id": "SHORT_COLD", "C": c, "reps": 2} for c in (1, 4, 16)] + [{"workload_id": "SHORT_COLD", "C": 64, "reps": 2}]
    def mk(cid, sa_add, note):
        s = dict(base_sa); s.update(sa_add); return cell(cid, "P9", anchor["cell_id"], env=dict(base_env), sa=s, pin_nonkt=pin, prec=anchor.get("precision_family"), workloads=W, note=note, sem="SEMANTICS_UNVERIFIED")
    for steps, dt in ((1, 2), (2, 3), (3, 4)):
        out.append(mk(f"S01_eagle3_steps{steps}_draft{dt}", {"speculative-algorithm": "EAGLE3", "speculative-draft-model-path": EAGLE3_CN, "speculative-num-steps": steps, "speculative-eagle-topk": 1, "speculative-num-draft-tokens": dt}, "EAGLE3 (rejection sampling 기본, threshold 미변경)"))
    out.append(mk("S02_standalone4b_steps3_draft4", {"speculative-algorithm": "STANDALONE", "speculative-draft-model-path": Q4B_CN, "speculative-num-steps": 3, "speculative-eagle-topk": 1, "speculative-num-draft-tokens": 4}, "STANDALONE Qwen3-4B 대조군 1개"))
    out.append(mk("S03_ngram_steps3_draft4", {"speculative-algorithm": "NGRAM", "speculative-num-steps": 3, "speculative-num-draft-tokens": 4}, "n-gram (지원 여부)"))
    return out

if __name__ == "__main__":
    camp = sys.argv[1]; phase = sys.argv[2]; anchor = None; outp = None
    a = sys.argv[3:]
    if "--anchor" in a: anchor = json.load(open(a[a.index("--anchor") + 1]))
    if "--out" in a: outp = a[a.index("--out") + 1]
    gen = {"P1": lambda: p1(), "P2": lambda: p2(), "P3": lambda: p3(anchor), "P4": lambda: p4(anchor), "P5": lambda: p5(anchor), "P6": lambda: p6(anchor), "P7": lambda: p7(anchor), "P8": lambda: p8(anchor), "P9": lambda: p9(anchor)}[phase]
    cells = gen(); os.makedirs(f"{camp}/manifests", exist_ok=True)
    outp = outp or f"{camp}/manifests/cells_{phase}.jsonl"
    with open(outp, "w") as f:
        for c in cells:
            c["config_hash"] = config_hash({"env": c["env"], "sa": c["server_args"], "gpus": c["gpus"], "pin": c.get("pin_nonkt", False), "patch": c.get("patch"), "prefix": c.get("launch_prefix")})
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(outp, len(cells), "cells")
