import sys, os, time, signal, subprocess
# SIGSTOP/SIGCONT duty controller. argv: duty_pct period_ms agg_cpus secs
duty=float(sys.argv[1])/100.0; period=float(sys.argv[2])/1000.0
agg=sys.argv[3]; secs=float(sys.argv[4])
BIN=os.path.join(os.path.dirname(os.path.dirname(__file__)),"src","victim_aggressor")
# 실제 경로 보정
BIN="shadow_assists/features/IDE_026_rdt_guarded_harvest/src/victim_aggressor"
p=subprocess.Popen(["taskset","-c",agg,BIN,"--role","aggressor","--cpus",agg,
                    "--array-mb","32","--aggr-mode","basic","--secs","100000"],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
on=period*duty; off=period*(1-duty)
t0=time.perf_counter()
try:
    if duty>=0.999:
        time.sleep(secs)
    elif duty<=0.001:
        p.send_signal(signal.SIGSTOP); time.sleep(secs)
    else:
        while time.perf_counter()-t0 < secs:
            p.send_signal(signal.SIGCONT); time.sleep(on)
            p.send_signal(signal.SIGSTOP); time.sleep(off)
finally:
    try:
        pg=os.getpgid(p.pid); os.killpg(pg,signal.SIGKILL)
    except Exception: pass
    p.kill()
