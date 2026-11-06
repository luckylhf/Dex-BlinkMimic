"""Independent wall-clock deadline for an authorized finite continuation."""
from pathlib import Path
import datetime
import json
import sys
import time

study = Path(sys.argv[1]).resolve()
config = json.loads((study / 'config.json').read_text())
while time.time() < config['deadline_epoch']:
    time.sleep(min(30., max(.1, config['deadline_epoch'] - time.time())))
stop = Path(config['run_dir']) / 'STOP'
stop.write_text('Authorized continuation wall-clock deadline reached\n')
(study / 'deadline_reached.json').write_text(json.dumps({
    'time': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'stop_file': str(stop), 'action': 'request checkpoint and graceful stop at the next iteration'
}, indent=2) + '\n')
