import sys,os,time,glob,math
# MERCATO: 혼잡가격으로 harvest MBA% 동적조정. argv: group target_MBps secs gamma
g=sys.argv[1]; target=float(sys.argv[2])*1e6; secs=float(sys.argv[3]); gamma=float(sys.argv[4])
mons=glob.glob("/sys/fs/resctrl/mon_data/mon_L3_*/mbm_total_bytes")
def total(): return sum(int(open(m).read()) for m in mons)
def set_mba(pct):
    try: open(g+"/schemata","w").write(f"MB:0={pct};1={pct}\n")
    except: pass
p=1.0; prev=total(); t0=time.perf_counter(); ep=0.1; hist=[]
while time.perf_counter()-t0<secs:
    time.sleep(ep); cur=total(); bw=(cur-prev)/ep; prev=cur
    err=(bw-target)/target
    p=max(1.0, min(10.0, p*math.exp(gamma*err)))
    mba=max(10,min(100,int(round(100/p/10)*10)))
    set_mba(mba); hist.append((round(bw/1e6),mba))
print("MERCATO,price=%.2f,last_mba=%d,bw_mba=%s"%(p,mba,hist[-6:]))
