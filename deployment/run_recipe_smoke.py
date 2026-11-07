"""Isolated graph/mirror/Muon smoke; never resumes a checkpoint."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--out',required=True);ap.add_argument('--full',action='store_true')
    a=ap.parse_args();root=Path(__file__).resolve().parents[1];out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,'-u','-m','dex_mimic.cli','train','--config','configs/dex_blink_recipe.json',
         '--compact-reference',str(Path(a.dataset).resolve()/'compact.pt'),'--log_dir',str(out),
         '--eprobe_start','0','--eprobe_every','3' if a.full else '1','--iterations','6','--ckpt_every','1']
    if not a.full:cmd+=['--num_envs','64','--wm_batch','64','--pi_batch','64','--buffer_steps','128','--eprobe_max_worlds','8']
    (out/'command.json').write_text(json.dumps(cmd,indent=2))
    env={**os.environ,'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'8'}
    with (out/'stdout.log').open('w') as log:
        result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    receipt=dict(passed=result.returncode==0,exit_code=result.returncode,full_configuration=a.full,checked_at=datetime.now().astimezone().isoformat())
    if result.returncode==0:
        import torch
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert ck['iteration']==6 and ck['args']['resume'] is None
        assert ck['args']['wm_mirror']==.5 and ck['args']['pi_opt']==ck['args']['wm_opt']=='muon'
        assert ck['args']['cuda_graph'] and ck['args']['collect_graph'] and ck['args']['eprobe_graph']
        assert ck['ctrl']['ep_cursor']>ck['args']['eprobe_max_worlds']
        for key in ['wm','pi','pi_live','critic']:
            assert all(torch.isfinite(t).all() for t in ck[key].values() if torch.is_tensor(t))
        lines=(out/'stdout.log').read_text()
        for token in ['captured wm_update','captured policy_update','--collect_graph: captured','--eprobe_graph: captured']:
            assert token in lines,token
        receipt.update(iteration=ck['iteration'],probe_cursor=ck['ctrl']['ep_cursor'],all_graph_paths_observed=True)
    (out/'smoke_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
