"""Gate, execute exactly 3500 fresh updates, then evaluate fixed references."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from dex_mimic.common import atomic,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--run',required=True);ap.add_argument('--evaluation',required=True)
    a=ap.parse_args();root=Path(__file__).resolve().parents[1];run=Path(a.run).resolve();dataset=Path(a.dataset).resolve()
    for receipt in [root/'runs/recipe_graph_smoke_20261107_v5/smoke_receipt.json',
                    root/'runs/recipe_full_graph_smoke_20261107_recompute/smoke_receipt.json',
                    root/'runs/recipe_final_preflight_20261107.json',
                    root/'runs/recipe_recompute_parity_20261107.json',
                    root/'runs/recipe_full_graph_smoke_20261107_recompute/parameter_update_receipt.json']:
        assert json.loads(receipt.read_text())['passed'],receipt
    run.mkdir(parents=True,exist_ok=False)
    config=json.loads((root/'configs/dex_blink_recipe.json').read_text())
    assert config['iterations']==3500 and config['wm_mirror']==.5
    cmd=[sys.executable,'-u','-m','dex_mimic.cli','train','--config','configs/dex_blink_recipe.json',
         '--compact-reference',str(dataset/'compact.pt'),'--log_dir',str(run),'--stop-file',str(run/'STOP')]
    source_files=['src/blinkmimic/train.py','src/blinkmimic/mdp.py','src/blinkmimic/motion.py','src/blinkmimic/env.py',
                  'src/blinkmimic/capture_compat.py','src/dex_mimic/data/compact_motion.py','configs/dex_blink_recipe.json']
    source_files+=sorted(str(p.relative_to(root)) for p in (root/'src/blinkmimic').rglob('*.py') if str(p.relative_to(root)) not in source_files)
    source_files+=['deployment/run_recipe_fresh.py','deployment/evaluate_recipe_fresh.py','deployment/check_recipe_environment.py','assets/dex_evt/model.json']
    atomic(run/'provenance.json',dict(from_scratch=True,parent_checkpoint=None,command=cmd,
        dataset_manifest_sha256=sha(dataset/'manifest.json'),compact_receipt=json.loads((dataset/'compact_receipt.json').read_text()),
        code_sha256={f:sha(root/f) for f in source_files},started_at=datetime.now().astimezone().isoformat()))
    env={**os.environ,'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'4','TORCHINDUCTOR_COMPILE_THREADS':'8'}
    env.pop('DEX_LIVE_REFERENCE_INDEX',None)
    with (run/'stdout.log').open('w') as log:
        train=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic(run/'status.json',dict(phase='training',train_pid=train.pid,from_scratch=True,target_iterations=3500))
        with (run/'monitor.log').open('w') as ml:
            monitor=subprocess.Popen([sys.executable,str(root/'deployment/monitor_training_resources.py'),'--pid',str(train.pid),'--out',str(run/'gpu_monitor.jsonl')],cwd=root,stdout=ml,stderr=subprocess.STDOUT)
            code=train.wait()
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:monitor.terminate()
    if code:
        atomic(run/'status.json',dict(phase='training_failed',exit_code=code));raise SystemExit(code)
    ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    assert ck['iteration']==3500 and ck['args']['resume'] is None
    for field in ['wm','pi','pi_live','critic']:
        assert all(torch.isfinite(v).all() for v in ck[field].values() if torch.is_tensor(v)),field
    atomic(run/'training_receipt.json',dict(passed=True,iteration=3500,from_scratch=True,
        checkpoint_sha256=sha(run/'last.pt'),finished_at=datetime.now().astimezone().isoformat()))
    del ck
    atomic(run/'status.json',dict(phase='evaluating',iteration=3500))
    with (run/'evaluation.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(root/'deployment/evaluate_recipe_fresh.py'),'--run',str(run),
                        '--dataset',str(dataset),'--out',str(Path(a.evaluation).resolve())],cwd=root,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    atomic(run/'status.json',dict(phase='complete',iteration=3500,evaluation=str(Path(a.evaluation).resolve())))


if __name__=='__main__':main()
