"""Report-ready run metadata for every processing / training run.

Appends one JSON line per run to logs/run_log.jsonl with: script, arguments, start / end
time (local, ISO), wall-clock seconds, hardware (CPU, cores, RAM, disk of the data drive),
OS, Python and package versions, plus anything the script adds (counts, throughput,
output size, MD5s, epochs, ...).

Usage in a script:
    from run_log import RunLog
    with RunLog("make_resized_512.py", notes="...") as log:
        ...work...
        log.add(images_in=..., images_out=..., output_bytes=...)
    # on exit: wall time and throughput (if 'images_out' given) are computed and the line is written

Standalone (prints the environment block):
    python -I run_log.py
"""
import datetime
import importlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs", "run_log.jsonl")
PACKAGES = ["numpy", "scipy", "pandas", "PIL", "skimage", "sklearn", "torch", "torchvision"]


def _powershell(cmd):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception as e:  # not on Windows, or PowerShell unavailable
        return f"unavailable ({type(e).__name__})"


def environment():
    env = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": _powershell("(Get-CimInstance Win32_Processor).Name"),
        "cpu_cores": _powershell("(Get-CimInstance Win32_Processor).NumberOfCores"),
        "cpu_logical": os.cpu_count(),
        "ram_gb": _powershell("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"),
        "data_disk": _powershell("$d=Get-Partition -DriveLetter %s | Get-Disk; \"$($d.FriendlyName) ($($d.BusType))\"" % ROOT[0]),
        "packages": {},
    }
    for p in PACKAGES:
        try:
            m = importlib.import_module(p)
            env["packages"][p] = getattr(m, "__version__", "?")
        except Exception:
            pass
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
    except Exception:
        env["gpu"] = "unknown"
    return env


def folder_size(path, ext=None):
    """(number of files, total bytes) under path, optionally only files ending in ext."""
    n = b = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            if ext is None or f.lower().endswith(ext):
                n += 1
                b += os.path.getsize(os.path.join(dirpath, f))
    return n, b


class RunLog:
    def __init__(self, script, notes=""):
        self.rec = {"script": script, "argv": sys.argv[1:], "notes": notes}

    def add(self, **kw):
        self.rec.update(kw)

    def __enter__(self):
        self.rec["start"] = datetime.datetime.now().isoformat(timespec="seconds")
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        wall = time.perf_counter() - self._t0
        self.rec["end"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.rec["wall_seconds"] = round(wall, 1)
        self.rec["status"] = "ok" if exc_type is None else f"failed: {exc_type.__name__}: {exc}"
        if "images_out" in self.rec and wall > 0:
            self.rec["images_per_second"] = round(self.rec["images_out"] / wall, 2)
        self.rec["environment"] = environment()
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.rec, ensure_ascii=False) + "\n")
        print(f"[run_log] {self.rec['script']}: {self.rec['wall_seconds']} s, status {self.rec['status']} -> {os.path.relpath(LOG, ROOT)}")
        return False


if __name__ == "__main__":
    print(json.dumps(environment(), indent=2))
