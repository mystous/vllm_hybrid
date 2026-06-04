import json, sys, os, re
OLD="/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602"
if not os.path.isdir(OLD):
    OLD="/home/mystous/vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602"
def baseline(tag, cond, m):
    f=f"{OLD}/summ_{tag}_{m}_{cond}.json"
    try: return json.load(open(f))['output_tps']
    except Exception: return None
for f in sys.argv[1:]:
    d = json.load(open(f))
    a=d.get('accept_rate'); rv=d.get('route_vanilla_n'); rs=d.get('route_suffix_n')
    rvf=d.get('route_vanilla_frac'); rsf=d.get('route_suffix_frac')
    route = f"vanilla {rv}({rvf}) / suffix {rs}({rsf})" if rv is not None else "n/a"
    tag=d['model']; cond=d['condition']; lt=d['output_tps']
    print(f"### {tag} x {d.get('method','llm-d')} x {cond}")
    print(f"- throughput : {lt:.1f} tps  ({d['n_ok']}/{d['n']} ok, {d.get('n_err',0)} err, wall {d.get('wall_total_s')}s)")
    vt=baseline(tag,cond,'vanilla'); st=baseline(tag,cond,'suffix')
    if vt or st:
        cmp=[]
        if vt: cmp.append(f"vanilla {vt:.0f} ({(lt/vt-1)*100:+.0f}%)")
        if st: cmp.append(f"suffix {st:.0f} ({(lt/st-1)*100:+.0f}%)")
        print(f"- baseline 대비: " + " | ".join(cmp))
    print(f"- TTFT p50/p99: {d['ttft_ms_p50']}/{d['ttft_ms_p99']} ms | TPOT p50/p99: {d['tpot_ms_p50']}/{d['tpot_ms_p99']} ms")
    print(f"- accept a : {a} (acc {d.get('accept_tokens')}/draft {d.get('draft_tokens')}) | 라우팅: {route}")
    print(f"- gpu {d.get('gpu_util')}% cpu {d.get('cpu_util')}% mem {d.get('gpu_mem_mib')}MiB | conc={d['concurrency']} max={d['max_tokens']}")
