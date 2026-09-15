#!/usr/bin/env python3
"""IDE_071 P0 — 옵션 등록부 (지시서 §3.2). 설치본 --help 존재 여부 + 소스 위치 + 기본값 (server_args.py) + KT_* 환경변수 소스 조건.
→ evidence/option_registry.json, OPTION_REGISTRY.md, evidence/source_locations.md
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

HELP = open(f"{FEAT}/evidence/sglang_launch_server_help.txt").read()
def cli_supported(name): return bool(re.search(rf"^\s+--{re.escape(name)}(\s|,|$)", HELP, re.M))
def help_text(name):
    m = re.search(rf"^\s+--{re.escape(name)}[^\n]*\n((?:\s{10,}[^\n]*\n)*)", HELP, re.M)
    return (m.group(0).strip()[:400] if m else None)
def src_default(argname):
    """server_args.py 의 dataclass 기본값."""
    r = dexec(CN, f"grep -nE '^\\s+{argname}: ' /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -2")
    return r.stdout.strip()[:200] or None

CLI = [  # (name, category, note)
 ("cuda-graph-max-bs", "graph", ""), ("cuda-graph-bs", "graph", ""), ("cuda-graph-max-bs-decode", "graph", "별칭/단계별"), ("cuda-graph-bs-decode", "graph", ""), ("cuda-graph-backend-prefill", "graph", ""),
 ("disable-cuda-graph", "graph", ""), ("disable-cuda-graph-padding", "graph", ""), ("enable-breakable-cuda-graph", "graph", ""), ("disable-piecewise-cuda-graph", "graph", ""), ("piecewise-cuda-graph-tokens", "graph", ""),
 ("max-total-tokens", "kv", ""), ("mem-fraction-static", "kv", ""), ("kv-cache-dtype", "kv", ""), ("page-size", "kv", ""),
 ("chunked-prefill-size", "sched", ""), ("prefill-max-requests", "sched", ""), ("max-prefill-tokens", "sched", ""), ("enable-mixed-chunk", "sched", ""), ("max-running-requests", "sched", ""),
 ("schedule-conservativeness", "sched", ""), ("schedule-policy", "sched", ""), ("num-continuous-decode-steps", "sched", ""), ("scheduler-recv-interval", "sched", ""), ("disable-overlap-schedule", "sched", ""),
 ("attention-backend", "attn", ""), ("prefill-attention-backend", "attn", ""), ("decode-attention-backend", "attn", ""), ("triton-attention-num-kv-splits", "attn", ""),
 ("moe-runner-backend", "moe", ""), ("disable-custom-all-reduce", "comm", ""), ("enable-flashinfer-allreduce-fusion", "comm", ""),
 ("tokenizer-worker-num", "api", ""), ("detokenizer-worker-num", "api", ""), ("enable-tokenizer-batch-encode", "api", ""),
 ("kt-weight-path", "kt", ""), ("kt-method", "kt", ""), ("kt-cpuinfer", "kt", ""), ("kt-threadpool-count", "kt", ""), ("kt-num-gpu-experts", "kt", ""), ("kt-max-deferred-experts-per-token", "kt", ""),
 ("init-expert-location", "eplb", ""), ("ep-dispatch-algorithm", "eplb", ""), ("ep-size", "parallel", ""), ("enable-dp-attention", "parallel", ""), ("dp-size", "parallel", ""),
 ("speculative-algorithm", "spec", ""), ("speculative-draft-model-path", "spec", ""), ("speculative-num-steps", "spec", ""), ("speculative-eagle-topk", "spec", ""), ("speculative-num-draft-tokens", "spec", ""),
 ("enable-hierarchical-cache", "hicache", ""), ("hicache-size", "hicache", ""),
]
ENV = [  # (name, source_file:line, semantics)
 ("KT_CALLBACK_FREE", "kt_kernel/experts_base.py:880,996", "set(비어있지 않은 문자열) 이면 패킷 등록 + GPU memop 트리거 경로; unset 이면 cudaLaunchHostFunc 경로. '0' 도 set 으로 취급 (bool(str))"),
 ("KT_CF_SKIP_EMPTY_IMM", "kt_kernel/experts_base.py:853", "callback-free AND max_deferred >= top-k(8) AND 이전 층 deferred 보류 있음 일 때 immediate 작업 생략. def0/4 에서는 효과 없음"),
 ("KT_COLD_DEFER", "kt_kernel/experts_base.py:553", "set 이면 deferral 기준을 'cold(GPU 미상주) AND score < KT_COLD_TAU' 로 바꿈 (hot 은 항상 immediate). 미설정 시 protected_k(=topk-max_deferred) topk 보호 방식"),
 ("KT_COLD_TAU", "kt_kernel/experts_base.py:555", "float, 기본 1.0. KT_COLD_DEFER 와 함께만 의미"),
 ("KT_AVX_RB", "kt-kernel/operators/amx/la/amx_kernels.hpp:1769", "getenv!=NULL 이면 레지스터 블로킹 AVX-512 커널 (값 무관; '0' 도 ON). 정수 누산이므로 bit 동일 (주석)"),
 ("KT_AVX_PF", "kt-kernel/operators/amx/la/amx_kernels.hpp:1773", "atoi → prefetch distance, 기본 0"),
 ("KT_FUSE_QIN", "kt-kernel/operators/amx/moe_base.hpp:511", "getenv!=NULL 이면 입력 quantization 결합 (값 무관)"),
 ("KT_QA_PAR", "kt-kernel/operators/amx/moe_base.hpp:515", "getenv!=NULL 이면 quantize-A 병렬"),
 ("KT_AMX_MIN_ROWS", "kt-kernel/operators/amx/moe.hpp:240", "atoi, 기본 1<<30. GEMM 선택: qlen > MIN_QLEN OR m >= MIN_ROWS → amx::mat_mul, 아니면 amx::vec_mul(AVX-512 VNNI)"),
 ("KT_AMX_MIN_QLEN", "kt-kernel/operators/amx/moe.hpp:244", "atoi, 기본 4*expert_num/topk = 80. 1000000 이면 qlen 조건이 사실상 거짓 → MIN_ROWS 만으로 AMX 선택"),
 ("KT_HUGEPAGE", "kt-kernel/operators/amx/moe_base.hpp:16", "getenv!=NULL 이면 madvise hugepage"),
 ("KT_AMX_BCACHE", "amx_kernels.hpp:1893", "getenv!=NULL"), ("KT_AMX_A_LOADD", "amx_kernels.hpp:1897", "getenv!=NULL"),
 ("KT_GPU_EXPERTS_PER_LAYER", "sglang kt_ep_wrapper.py (IDE_070 패치, 컨테이너 로컬)", "json per_layer[layer_idx] → KTConfig.num_gpu_experts. 패치 적용 시에만"),
 ("KT_FORCE_SYNC_SUBMIT", "kt_kernel/experts_base.py:75", "'1' 이면 동기 submit/sync 경로"),
 ("SGL_INTERLEAVE", "kt_kernel/experts_base.py:998", "signal 슬롯 대기 경로 (IDE_030)"),
 ("KT_CF_SKIP_EMPTY_DEF", "kt-kernel/cpu_backend/cpuinfer.h:139", "빈 deferred 생략"),
 ("KT_PHASE_PROF", "cpu_backend", "phase 계측 (진단용)"), ("KT_TQ_TIMING", "cpu_backend", "TaskQueue 계측 (진단용)"), ("KT_MOE_PHASE_TIMING", "cpu_backend", "MoE phase 계측"),
]

def main():
    reg = []
    for name, cat, note in CLI:
        sup = cli_supported(name); arg = name.replace("-", "_")
        reg.append({"name": "--" + name, "category": cat, "source_kind": "LOCAL_CODE", "installed_cli_supported": sup, "help_excerpt": help_text(name),
                    "source_file": "/sgl-workspace/sglang/python/sglang/srt/server_args.py", "resolved_default_src": src_default(arg) if sup else None,
                    "runtime_application_probe": "/get_server_info 의 " + arg, "requires_patch": False, "check_timestamp": now()["wall_kst"], "note": note,
                    "evidence_file": "evidence/sglang_launch_server_help.txt"})
    for name, loc, sem in ENV:
        reg.append({"name": name, "category": "env", "source_kind": "LOCAL_CODE", "installed_cli_supported": None, "source_file": loc, "semantics_from_source": sem,
                    "runtime_application_probe": "서버 로그 [kt-cf] / 패치 로그 / 진단 카운터", "requires_patch": "패치" in loc, "check_timestamp": now()["wall_kst"], "evidence_file": "evidence/source_locations.md"})
    for name, kind, note in [("polling interval / bounded spin", "NEW_PATCH_PROPOSAL", "§7.4 로컬 옵션 없음 (cpuinfer.h poll_loop_ 는 고정 spin)"), ("exact grouped expert execution", "NEW_PATCH_PROPOSAL", "§7.4"),
                             ("dynamic GPU expert cache/prefetch", "NEW_PATCH_PROPOSAL", "§10.4"), ("per-layer non-uniform GPU experts", "LOCAL_CODE", "IDE_070 패치 (patch_per_layer_experts.sh)"),
                             ("hot expert replication", "NEW_PATCH_PROPOSAL", "§10.5"), ("routing-aware request scheduling", "NEW_PATCH_PROPOSAL", "§10.5")]:
        reg.append({"name": name, "category": "patch", "source_kind": kind, "installed_cli_supported": False, "requires_patch": True, "note": note, "check_timestamp": now()["wall_kst"]})
    jdump(reg, f"{FEAT}/evidence/option_registry.json")
    md = ["# OPTION_REGISTRY — IDE_071 (P0)", "", f"생성 {now()['wall_kst']}. 설치본: SGLang 0.5.18 @71de97b (+로컬 수정), kt-kernel 0.7.0.post1 (컨테이너 sgl-kt). `installed_cli_supported` 는 `python3 -m sglang.launch_server --help` (evidence/sglang_launch_server_help.txt) 기준. 실제 적용 여부는 셀별 `runtime_proofs.json`/`effective_config.json`.", "",
          "| name | category | source_kind | installed_cli_supported | source / default | semantics (source) |", "|---|---|---|---|---|---|"]
    for r in reg:
        md.append(f"| `{r['name']}` | {r['category']} | {r['source_kind']} | {r.get('installed_cli_supported')} | {(r.get('resolved_default_src') or r.get('source_file') or '')[:90]} | {(r.get('semantics_from_source') or r.get('note') or '')[:160]} |")
    open(f"{FEAT}/OPTION_REGISTRY.md", "w").write("\n".join(md) + "\n")
    sl = ["# source_locations — IDE_071", "", "| symbol | location | note |", "|---|---|---|"]
    for name, loc, sem in ENV: sl.append(f"| {name} | {loc} | {sem} |")
    sl += ["| KTEPWrapperMethod.apply / submit / sync | /sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py:300-430 | CPU expert 계산은 tp_rank==0 만 수행 (`if self.tp_rank != 0: return`) |",
           "| create_kt_config_from_server_args | kt_ep_wrapper.py:96-131 | num_gpu_experts 균일 (패치 지점) |",
           "| callback-free handoff (IDE_033) | /sgl-workspace/ktransformers/kt-kernel/cpu_backend/cpuinfer.h:129-290 | mapped go/done flags + poller thread |",
           "| KExpertsCPUBuffer (pinned ring) | kt_kernel/experts_base.py:216-270 | pin_memory=True, slot = layer_idx % buffer_depth |",
           "| do_gate_up_gemm / do_down_gemm | kt-kernel/operators/amx/moe.hpp:252-270 | AMX vs AVX(vec_mul) 선택 조건 |",
           "| hotmap_mixed α | eval/harness/20260909_mechanism/build_hotmap_mixed.py | score(e)=α·p_prefill(e)+(1−α)·p_decode(e), prefill 패스 = >64 tok, decode = ≤64 tok |"]
    open(f"{FEAT}/evidence/source_locations.md", "w").write("\n".join(sl) + "\n")
    print(len(reg), "options;", sum(1 for r in reg if r.get("installed_cli_supported") is True), "cli supported;", [r["name"] for r in reg if r.get("installed_cli_supported") is False and r["category"] != "patch"])

if __name__ == "__main__":
    main()
