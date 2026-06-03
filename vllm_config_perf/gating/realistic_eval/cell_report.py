import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f))
    a=d.get('accept_rate'); rv=d.get('route_vanilla_n'); rs=d.get('route_suffix_n')
    rvf=d.get('route_vanilla_frac'); rsf=d.get('route_suffix_frac')
    route = f"vanilla {rv}({rvf}) / suffix {rs}({rsf})" if rv is not None else "n/a"
    print(f"### {d['model']} x llm-d x {d['condition']}")
    print(f"- throughput : {d['output_tps']:.1f} tps  ({d['n_ok']}/{d['n']} ok, {d.get('n_err',0)} err, wall {d.get('wall_total_s')}s)")
    print(f"- TTFT p50/p99: {d['ttft_ms_p50']}/{d['ttft_ms_p99']} ms | TPOT p50/p99: {d['tpot_ms_p50']}/{d['tpot_ms_p99']} ms")
    print(f"- accept a : {a} (acc {d.get('accept_tokens')}/draft {d.get('draft_tokens')}) | 라우팅: {route}")
    print(f"- gpu {d.get('gpu_util')}% cpu {d.get('cpu_util')}% mem {d.get('gpu_mem_mib')}MiB | conc={d['concurrency']} max={d['max_tokens']}")
