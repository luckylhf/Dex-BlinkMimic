"""Continue the final full state, grow accepted BONES, and stop at an absolute time."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from dex_mimic.common import atomic, sha
from dex_mimic.evaluation.three_tests import default_suite, load_release, validate_references


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deadline', required=True, help='ISO datetime with explicit timezone')
    p.add_argument('--reuse-live', type=Path, help='reuse a verified append-only catalog and its running updater')
    a = p.parse_args()
    deadline = datetime.fromisoformat(a.deadline)
    if deadline.tzinfo is None:
        raise ValueError('Deadline needs an explicit timezone')
    deadline_epoch = deadline.timestamp()
    if deadline_epoch <= time.time():
        raise ValueError('Deadline is already past')
    root = Path.cwd()
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = proc.read_bytes().split(b'\0')
        except (OSError, ProcessLookupError):
            continue
        if b'dex_mimic.cli' in command and b'train' in command:
            raise RuntimeError('An existing training process is still running; preserve it')
    load_release(root)
    seed = root / 'data/mixed/lafan77_bones080_bounded_20261106'
    old_run = root / 'runs/lafan_bones_mix1h_bounded_20261106_195934'
    ck = torch.load(old_run/'last.pt', map_location='cpu', weights_only=False)
    assert ck['iteration'] == 12659 and ck['args']['num_envs'] == 8192
    assert ck['reference_rotation']['dataset_sha256'] == sha(seed/'manifest.json')
    zone = timezone(timedelta(hours=8))
    tag = datetime.now(zone).strftime('%Y%m%d_%H%M%S')
    run = root / 'runs' / ('lafan_bones_live_untilnoon_'+tag)
    study = root / 'evaluations' / ('progress1h_live_untilnoon_'+tag)
    live = a.reuse_live.resolve() if a.reuse_live else root / 'data/live' / ('lafan_bones_untilnoon_'+tag)
    for directory in (run, study):
        directory.mkdir(parents=True, exist_ok=False)
    for directory in ('checkpoints', 'references'):
        (study/directory).mkdir()
    atomic(study/'previous_pointers.json', {n:(root/'deployment'/n).read_text() for n in
        ('current_run.txt', 'current_progress_study.txt', 'current_bones_mix_release.txt')})
    source_manifest = Path(json.loads((seed/'manifest.json').read_text())['source_dataset'])/'manifest.json'
    index = json.loads((seed/'features/index.json').read_text())
    for row in index['records']:
        for key in ('pack', 'manifest'):
            row[key] = str((seed/row[key]).resolve())
    if not a.reuse_live:
        live.mkdir(parents=True, exist_ok=False)
        atomic(live/'index.json', dict(schema=1, revision=0, seed_dataset_sha256=sha(seed/'manifest.json'),
            seed_source_manifest=str(source_manifest), frame_capacity=index['frame_capacity'],
            records=index['records'], published_epoch=time.time()))
        atomic(live/'status.json', dict(state='starting', revision=0, shards=len(index['records']),
            included_bones=4371, included_lafan=77, interval_seconds=300))
    else:
        from dex_mimic.training.live_rotation import LiveRotation
        LiveRotation(seed, live_index=live/'index.json')
    suite = dict(evaluation_profile='jump_tracking_v2', reference_preprocessing='supermimic_12341_three_tests_v1', references={})
    standard = default_suite(root)
    for name, row in json.loads(standard.read_text())['references'].items():
        source = standard.parent/row['reference']
        dst = study/'references'/(name+'.npz')
        shutil.copy2(source, dst)
        assert sha(dst) == row['reference_sha256']
        suite['references'][name] = dict(reference=str(dst), reference_sha256=sha(dst), training_member=False)
    previous = root/'evaluations/progress1h_bones_bounded_20261106_195934'
    for name in ('walk1_subject5', 'run1_subject5'):
        dst = study/'references'/(name+'.npz')
        shutil.copy2(previous/'references'/(name+'.npz'), dst)
        suite['references'][name] = dict(reference=str(dst), reference_sha256=sha(dst), training_member=True)
    validate_references({n:r['reference'] for n,r in suite['references'].items()})
    atomic(study/'suite.json', suite)
    baseline = study/'checkpoints/baseline_012659.pt'
    shutil.copy2(old_run/'ckpt_012659.pt', baseline)
    assert sha(baseline) == '6fc8137e0b80b5ba227433fe2242a3a68f435084fda3136be5a3eff1e6e881d1'
    resume = run/'resume_012659.pt'
    shutil.copy2(old_run/'last.pt', resume)
    assert sha(resume) == sha(old_run/'last.pt')
    atomic(run/'resume_receipt.json', dict(iteration=12659, source=str(old_run/'last.pt'),
        sha256=sha(resume), full_state_byte_identical=True, optimizer_and_models_preserved=True,
        dataset_sampler_and_difficulty_preserved=True, seed_release_sha256=sha(seed/'manifest.json')))
    wall = ck['wall_time']
    del ck
    started = time.time()
    iso = lambda t: datetime.fromtimestamp(t, zone).isoformat()
    final_minute = math.ceil((deadline_epoch-started)/60)
    milestones = sorted(set(list(range(0, int((deadline_epoch-started)//3600)*60+1, 60))+[final_minute]))
    env = dict(os.environ, DEX_MIMIC_ROOT=str(root), DEX_ASSETS=str(root/'assets/dex_evt'),
        PYTHONPATH=str(root/'src'), DEX_LIVE_REFERENCE_INDEX=str(live/'index.json'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    command = [str(root/'.venv/bin/python'), '-u', '-m', 'dex_mimic.cli', 'train',
        '--config', 'configs/dex_blink_safe.json', '--data-release', str(seed),
        '--switch-seconds', '150', '--switch-warmup-steps', '64', '--skip-evaluation',
        '--log_dir', str(run), '--resume', str(resume), '--iterations', '1000000',
        '--max_hours', str((wall+deadline_epoch-started)/3600), '--stop-file', str(run/'STOP')]
    atomic(run/'launch_command.json', dict(argv=command))
    shutil.copy2(root/'configs/dex_blink_safe.json', run/'package_config.json')
    with (run/'stdout.log').open('w') as log:
        train = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    cfg = dict(study_dir=str(study), run_dir=str(run), train_pid=train.pid, started_epoch=started,
        deadline_epoch=deadline_epoch, started_at=iso(started), deadline_at=iso(deadline_epoch),
        resume_iteration=12659, resume_checkpoint=str(resume), baseline_checkpoint=str(baseline),
        num_envs=8192, milestone_minutes=milestones, evaluation_interval_minutes=60,
        suite=str(study/'suite.json'), evaluation_profile='jump_tracking_v2',
        render_motions=[], live_data_dir=str(live),
        training_data_description='全部 77 段 LAFAN + 持续接入的已验收 BONES（每 5 分钟扫描）',
        study_title='LAFAN + BONES 动态增长续训：每小时进展',
        history_description='统一旧 SuperMimic 三动作输入 + 固定 LAFAN 行走/跑步；从 12659 新建历史，旧无过渡结果不混入。')
    atomic(study/'config.json', cfg)
    atomic(study/'schedule.json', [dict(minute=m, target_at=iso(deadline_epoch if m==final_minute else started+m*60)) for m in milestones])
    pids = dict(training=train.pid)
    reused_updater = None
    if a.reuse_live:
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try: command_line = proc.read_bytes().split(b'\0')
            except OSError: continue
            if str(root/'deployment/update_live_bones.py').encode() in command_line and str(live).encode() in command_line:
                reused_updater = int(proc.parent.name)
                pids['data_updater'] = reused_updater
    for name, argv, logfile, child_env in [
        ('data_updater', [str(root/'deployment/update_live_bones.py'), '--live', str(live),
            '--batches', '/home/longhongfeng/Desktop/Data2/BONES-SEED/02_Dex转换数据',
            '--deadline', str(deadline_epoch)], live/'updater.log', dict(env, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2')),
        ('deadline_guard', [str(root/'deployment/progress_deadline_guard.py'), str(study)], study/'deadline_guard.log', env),
        ('controller', [str(root/'deployment/progress_controller.py'), str(study)], study/'controller.log', env),
        ('resource_monitor', [str(root/'deployment/monitor_training_resources.py'), '--pid', str(train.pid),
            '--out', str(run/'gpu_monitor.jsonl')], run/'gpu_monitor.log', env),
    ]:
        if name == 'data_updater' and reused_updater:
            continue
        with logfile.open('w') as log:
            child = subprocess.Popen([sys.executable, '-u', *argv], cwd=root, env=child_env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pids[name] = child.pid
    receipt = dict(pids=pids, started_at=iso(started), deadline_at=iso(deadline_epoch),
        run=str(run), study=str(study), live_data=str(live), resume_iteration=12659,
        num_envs=8192, evaluation_interval_minutes=60, data_scan_interval_seconds=300,
        resident_reference_frames=600000, base_sampling_mass=dict(lafan=.5, bones=.5),
        final_checkpoint_and_graceful_stop=True)
    atomic(study/'launch_receipt.json', receipt)
    (run/'train.pid').write_text(str(train.pid)+'\n')
    (root/'deployment/current_run.txt').write_text(str(run.relative_to(root))+'\n')
    (root/'deployment/current_progress_study.txt').write_text(str(study.relative_to(root))+'\n')
    (root/'deployment/current_live_bones.txt').write_text(str(live.relative_to(root))+'\n')
    print('LIVE_CONTINUATION_STARTED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
