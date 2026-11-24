#!/usr/bin/env python3
"""peak-memory.py NAME COMMAND...: runs COMMAND and prints the peak memory of the server whose program is NAME.

It samples twice a second while COMMAND runs and exits with COMMAND's status. GB10 has one memory pool, so the peak
is the largest sum in one sample of the GPU memory nvidia-smi gives the job's compute processes and the server's
RssAnon + RssShmem. RssFile stays out: a mapped weight file's pages are page cache the kernel can drop, and VmHWM
(printed beside the peak) counts them. The fall in MemAvailable since the last sample before the server counts every
byte once, with whatever else the job held meanwhile.
"""
import glob, os, signal, subprocess, sys, time

name, command = sys.argv[1], sys.argv[2:]


def server() -> str | None:
    for path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            program = open(path, "rb").read().split(b"\0")[0].decode()
        except OSError:
            continue
        if os.path.basename(program) == name:
            return path.split("/")[2]
    return None


def kib(path: str) -> dict[str, int]:
    """The kB fields of a /proc status or meminfo file."""
    fields = (line.split(":", 1) for line in open(path))
    return {k: int(v.split()[0]) for k, v in fields if v.endswith("kB\n")}


def gpu_kib() -> int:
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.split()
    return 1024 * sum(int(mib) for mib in rows)


job = subprocess.Popen(command, start_new_session=True)


def stop(_signum, _frame):
    os.killpg(job.pid, signal.SIGTERM)
    job.wait()
    sys.exit(124)


signal.signal(signal.SIGTERM, stop)
before = kib("/proc/meminfo")["MemAvailable"]
peak, hwm, fall, seen = (0, 0, 0), 0, 0, False
while job.poll() is None:
    pid = server()
    try:
        status = kib(f"/proc/{pid}/status") if pid else None
    except OSError:
        status = None
    available = kib("/proc/meminfo")["MemAvailable"]
    if status:
        seen = True
        gpu, resident = gpu_kib(), status.get("RssAnon", 0) + status.get("RssShmem", 0)
        peak = max(peak, (gpu + resident, gpu, resident))
        hwm, fall = max(hwm, status.get("VmHWM", 0)), max(fall, before - available)
    elif not seen:
        before = available
    time.sleep(0.5)
gib = lambda k: f"{k / 2**20:.2f} GiB"
if seen:
    print(f"RESULT {name} peak memory: {gib(peak[0])} (GPU {gib(peak[1])} + resident anon and shmem {gib(peak[2])}), "
          f"VmHWM {gib(hwm)}, MemAvailable fell {gib(fall)}", flush=True)
elif job.returncode == 0:
    sys.exit(f"{name} never ran under {' '.join(command)}: no peak memory to report")
sys.exit(job.returncode)
