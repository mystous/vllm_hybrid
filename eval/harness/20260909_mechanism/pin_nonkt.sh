#!/usr/bin/env bash
# 컨테이너 내부 실행 (python 일괄 처리판): KTransformers 워커(numa_*_t_*) 를 제외한 sglang 프로세스 스레드를 빈 코어로 이동.
FREE=${FREE:-48-55,104-111,160-167,216-223}
python3 - "$FREE" <<'PY'
import os,sys,re
def parse(spec):
    s=set()
    for part in spec.split(","):
        a,b=(part.split("-")+[None])[:2]; s.update(range(int(a), int(b if b else a)+1))
    return s
free=parse(sys.argv[1]); n=0; procs=0
for pid in os.listdir("/proc"):
    if not pid.isdigit(): continue
    try:
        comm=open(f"/proc/{pid}/comm").read().strip()
        st=open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0]
    except Exception: continue
    if st=="Z" or not (comm.startswith("sglang::") or comm=="python3"): continue
    try: tids=os.listdir(f"/proc/{pid}/task")
    except Exception: continue
    if len(tids)<50: continue
    procs+=1
    for t in tids:
        try:
            tc=open(f"/proc/{pid}/task/{t}/comm").read().strip()
            if tc.startswith("numa_"): continue
            os.sched_setaffinity(int(t), free); n+=1
        except Exception: pass
print(f"pinned {n} non-kt threads in {procs} procs to {sys.argv[1]}")
PY
