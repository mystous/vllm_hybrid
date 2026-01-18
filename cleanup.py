import os
import signal
import subprocess
import glob

def cleanup():
    # Kill processes
    try:
        pids = subprocess.check_output(["pgrep", "-f", "vllm"]).decode().split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except subprocess.CalledProcessError:
        pass
        
    try:
        pids = subprocess.check_output(["pgrep", "-f", "python3"]).decode().split()
        my_pid = os.getpid()
        for pid in pids:
            if int(pid) != my_pid: 
                 # Filter based on cmdline to be safe? 
                 # Assuming dedicated env for now, but safer to target api_server
                 try:
                     cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode().replace('\0', ' ')
                     if "api_server" in cmd or "test_client" in cmd:
                         os.kill(int(pid), signal.SIGKILL)
                 except:
                     pass
    except:
        pass

    # Clean SHM
    for f in glob.glob("/dev/shm/*vllm*"):
        try:
            os.remove(f)
        except:
            pass
            
if __name__ == "__main__":
    cleanup()
