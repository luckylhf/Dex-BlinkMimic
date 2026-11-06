"""Version the evaluation study without touching training or the deadline guard."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def bootstrap(study):
    cfg = json.loads((study/'config.json').read_text())
    project = Path(cfg['run_dir']).parents[1]
    env = dict(os.environ, DEX_MIMIC_ROOT=str(project), DEX_ASSETS=str(project/'assets/dex_evt'),
               PYTHONPATH=str(project/'src'), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2', MUJOCO_GL='egl')
    try:
        subprocess.run([sys.executable, '-u', '-m', 'dex_mimic.evaluation.evaluate',
                        '--checkpoint', cfg['historical_seed_checkpoint'], '--suite', cfg['suite'],
                        '--out', str(study/'seed_001899'), '--history', str(study/'history'), '--actor', 'ema'],
                       cwd=project, env=env, check=True)
        atomic(study/'history_bootstrap.json', dict(ready=True, iteration=1899, profile=cfg['evaluation_profile'], epoch=time.time()))
    except Exception as error:
        atomic(study/'history_bootstrap.json', dict(ready=False, error=repr(error), epoch=time.time()))
        raise


def migrate(old, new, old_pid):
    assert not new.exists(), 'Use a fresh directory; never overwrite evaluation evidence'
    cfg = json.loads((old/'config.json').read_text())
    status = json.loads((old/'status.json').read_text())
    assert status['active_minute'] is None and time.time()-Path(old/'status.json').stat().st_mtime < 30
    cmdline = (Path('/proc')/str(old_pid)/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert 'progress_controller.py' in cmdline and str(old) in cmdline
    assert 'progress_deadline_guard' not in cmdline
    project = Path(cfg['run_dir']).parents[1]
    source_seed = project/'evaluations/latest_test_20261106_1430/checkpoint/ckpt_001899.pt'
    assert hashlib.sha256(source_seed.read_bytes()).hexdigest() == '65e98b8b06e21f671b41830e405ac1d5f5bf96bf0184371384839ad644bb7d52'
    new.mkdir()
    for name in ('checkpoints', 'references'):
        shutil.copytree(old/name, new/name, copy_function=os.link)
    shutil.copy2(old/'schedule.json', new/'schedule.json')
    seed = new/'checkpoints/seed_001899.pt'
    os.link(source_seed, seed)
    suite = json.loads((old/'suite.json').read_text())
    suite['evaluation_profile'] = 'jump_tracking_v2'
    for row in suite['references'].values():
        row['reference'] = str(new/'references'/Path(row['reference']).name)
    atomic(new/'suite.json', suite)
    cfg.update(study_dir=str(new), suite=str(new/'suite.json'), evaluation_profile='jump_tracking_v2',
               historical_seed_checkpoint=str(seed), requires_history_bootstrap=True,
               supersedes_study=str(old), training_termination_changed=False)
    atomic(new/'config.json', cfg)
    jobs = json.loads((old/'jobs.json').read_text())
    for job in jobs.values():
        job['checkpoint'] = str(new/'checkpoints'/Path(job['checkpoint']).name)
        assert hashlib.sha256(Path(job['checkpoint']).read_bytes()).hexdigest() == job['checkpoint_sha256']
        job['attempts'] = 0
        job.pop('done', None)
        job.pop('failed', None)
        stage = new/f'minute_{job["minute"]:03d}'
        stage.mkdir()
        atomic(stage/'job.json', job)
    atomic(new/'jobs.json', jobs)
    atomic(new/'comparisons.json', [])
    atomic(new/'history_bootstrap.json', dict(ready=False, iteration=1899))
    # Only the idle evaluator is replaced. Neither training nor its guard receives a signal.
    os.kill(old_pid, signal.SIGTERM)
    status.update(state='superseded_by_jump_tracking_v2', superseded_by=str(new))
    atomic(old/'status.json', status)
    with (old/'README.md').open('a') as f:
        f.write('\n此目录保留 strict_v1 历史证据；后续评测已迁移至 '+str(new)+'。禁止跨规则直接比较终止时长。\n')
    pids = {}
    for name, command in (
        ('bootstrap', ['nice', '-n', '10', sys.executable, '-u', str(Path(__file__).resolve()), '--bootstrap', str(new)]),
        ('controller', [sys.executable, '-u', str(project/'deployment/progress_controller.py'), str(new)]),
    ):
        with (new/(name+'.log')).open('a') as log:
            process = subprocess.Popen(command, cwd=project, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pids[name] = process.pid
    receipt = dict(old_study=str(old), new_study=str(new), profile='jump_tracking_v2',
                   migrated_at=datetime.datetime.now().astimezone().isoformat(),
                   training_pid=cfg['train_pid'], training_termination_changed=False,
                   deadline_at=cfg['deadline_at'], pids=pids, legacy_controller_pid=old_pid)
    atomic(new/'migration_receipt.json', receipt)
    (project/'deployment/current_progress_study.txt').write_text(str(new.relative_to(project))+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=Path)
    parser.add_argument('--old', type=Path)
    parser.add_argument('--new', type=Path)
    parser.add_argument('--old-controller-pid', type=int)
    args = parser.parse_args()
    if args.bootstrap:
        bootstrap(args.bootstrap.resolve())
    else:
        migrate(args.old.resolve(), args.new.resolve(), args.old_controller_pid)
