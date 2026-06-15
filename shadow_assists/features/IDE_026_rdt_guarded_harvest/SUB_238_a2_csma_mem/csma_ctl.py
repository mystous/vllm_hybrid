import sys,os,time,signal,glob
# CSMA-MEM: mbm_total 센싱 → ceiling 초과면 harvest SIGSTOP, 미만이면 SIGCONT.
# argv: harvest_pid ceiling_MBps secs epoch_ms
hpid=int(sys.argv[1]); ceil=float(sys.argv[2])*1e6; secs=float(sys.argv[3]); ep=float(sys.argv[4])/1000.0
mons=glob.glob("/sys/fs/resctrl/mon_data/mon_L3_*/mbm_total_bytes")
def total():
    s=0
    for m in mons:
        try: s+=int(open(m).read())
        except: pass
    return s
prev=total(); t0=time.perf_counter(); stopped=False; nstop=0; ncont=0
try:
    while time.perf_counter()-t0<secs:
        time.sleep(ep)
        cur=total(); bw=(cur-prev)/ep; prev=cur
        if bw>ceil and not stopped:
            try: os.kill(hpid,signal.SIGSTOP); stopped=True; nstop+=1
            except: pass
        elif bw<=ceil and stopped:
            try: os.kill(hpid,signal.SIGCONT); stopped=False; ncont+=1
            except: pass
finally:
    try: os.kill(hpid,signal.SIGCONT)
    except: pass
    print(f"CSMA,stops={nstop},conts={ncont}")
