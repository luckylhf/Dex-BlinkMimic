"""Pull only completed, verified half-hour evaluations into the local workspace."""
from pathlib import Path
import datetime
import json
import shutil
import subprocess
import sys
import time

remote, output = sys.argv[1:3]
out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=True)
assert str(Path(remote).parent) == '/home/longhongfeng/Desktop/Dex-BlinkMimic/evaluations'
assert Path(remote).name.startswith(('progress3h_','progress1h_'))
assert all(c.isalnum() or c in '/_-' for c in remote)
deadline = time.time()+24*3600
seen = set()
env_status = out/'local_sync_status.json'

def run(cmd, timeout=1800):
    return subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=timeout)

while True:
    try:
        if shutil.disk_usage(out).free < 10*1024**3:
            raise RuntimeError('Local free space below 10 GiB reserve')
        raw=run(['ssh','-o','ControlMaster=no','-o','ControlPath=none','-o','BatchMode=yes','-o','ConnectTimeout=10','luckylhf',f'cat {remote}/status.json'],30).stdout
        status=json.loads(raw)
        for name in ('config.json','schedule.json','status.json','comparisons.json','progress.csv','README.md'):
            run(['rsync','-az','-e','ssh -o ControlMaster=no -o ControlPath=none','--timeout=90',f'luckylhf:{remote}/{name}',str(out)+'/'])
        for folder in status.get('ready_milestones',[]):
            assert folder.startswith('minute_') and folder[7:].isdigit()
            if folder not in seen:
                destination=out/folder;destination.mkdir(exist_ok=True)
                run(['rsync','-az','-e','ssh -o ControlMaster=no -o ControlPath=none','--timeout=90','--exclude','development/','--exclude','*.npz',
                     '--exclude','*.pt','--exclude','*.onnx','--exclude','*.mjb',
                     f'luckylhf:{remote}/{folder}/',str(destination)+'/'])
                ready=json.loads((destination/'ready.json').read_text())
                import hashlib
                for video in ready['videos']:
                    path=out/video['path'];h=hashlib.sha256(path.read_bytes()).hexdigest()
                    assert h==video['sha256'],path
                seen.add(folder)
        env_status.write_text(json.dumps(dict(updated_at=datetime.datetime.now().astimezone().isoformat(),
            remote_state=status['state'],verified_local_milestones=sorted(seen)),indent=2)+'\n')
        print(datetime.datetime.now().isoformat(),status['state'],sorted(seen),flush=True)
        if status['state'] in ('completed','completed_with_errors','controller_failed','failed_training_exited_early'):
            break
    except Exception as error:
        env_status.write_text(json.dumps(dict(updated_at=datetime.datetime.now().astimezone().isoformat(),error=repr(error),
            verified_local_milestones=sorted(seen)),indent=2)+'\n')
        print('SYNC_RETRY',repr(error),flush=True)
    if time.time()>deadline:
        print('SYNC_TIMEOUT',flush=True);break
    time.sleep(60)
