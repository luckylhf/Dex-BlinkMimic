"""Record GPU memory and the owned training process RSS until it exits."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
with Path(a.out).open('a',buffering=1) as f:
    while True:
        proc=Path('/proc')/str(a.pid)
        try:
            if proc.joinpath('stat').read_text().split(') ',1)[1][0]=='Z':break
            rss=next(x for x in proc.joinpath('status').read_text().splitlines() if x.startswith('VmRSS:'))
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
            f.write(json.dumps(dict(epoch=time.time(),train_pid=a.pid,rss=rss,gpu=gpu))+'\n')
        except (FileNotFoundError,ProcessLookupError):break
        except Exception as error:f.write(json.dumps(dict(epoch=time.time(),error=repr(error)))+'\n')
        time.sleep(5)
