"""Record Linux wall time, CPU time and child peak RSS for a command."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

output = Path(sys.argv[1])
start = time.monotonic()
process = subprocess.run(sys.argv[2:], check=False)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
output.write_text(json.dumps({
    "wall_seconds": time.monotonic() - start,
    "user_seconds": usage.ru_utime,
    "system_seconds": usage.ru_stime,
    "peak_rss_kib": usage.ru_maxrss,
    "returncode": process.returncode,
}, indent=2) + "\n")
sys.exit(process.returncode)
