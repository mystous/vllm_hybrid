import sys, re, json
def parse(text):
    a=d=r=0.0
    for l in text.splitlines():
        if l.startswith('#'): continue
        m=re.match(r'(vllm:\S+?)(\{[^}]*\})?\s+([0-9.eE+-]+)', l)
        if not m: continue
        n,v=m.group(1),float(m.group(3))
        if n=='vllm:spec_decode_num_accepted_tokens_total': a+=v
        elif n=='vllm:spec_decode_num_draft_tokens_total': d+=v
        elif n=='vllm:request_success_total': r+=v
    return a,d,r
if len(sys.argv)>1 and sys.argv[1]=='--patch':
    f=sys.argv[2]
    nums=list(map(float,sys.argv[3:15]))   # sa sd sr va vd vr sa2 sd2 sr2 va2 vd2 vr2
    sa,sd,sr,va,vd,vr,sa2,sd2,sr2,va2,vd2,vr2=nums
    acc=sa2-sa; draft=sd2-sd; sreq=sr2-sr; vreq=vr2-vr; tot=sreq+vreq
    o=json.load(open(f))
    o['accept_tokens']=acc; o['draft_tokens']=draft
    o['accept_rate']=round(acc/draft,4) if draft>0 else None
    o['route_suffix_n']=int(round(sreq)); o['route_vanilla_n']=int(round(vreq))
    o['route_suffix_frac']=round(sreq/tot,4) if tot>0 else None
    o['route_vanilla_frac']=round(vreq/tot,4) if tot>0 else None
    json.dump(o,open(f,'w'),indent=1)
    print(f"    patched alpha={o['accept_rate']} route v/s={o['route_vanilla_n']}/{o['route_suffix_n']}")
else:
    a,d,r=parse(sys.stdin.read()); print(a,d,r)
