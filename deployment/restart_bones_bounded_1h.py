"""Restart the full hour using existing bounded rotation and a verified reused baseline."""
from datetime import datetime,timezone,timedelta
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from dex_mimic.common import atomic,sha
from dex_mimic.evaluation.three_tests import validate_references


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--baseline-study',required=True);a=p.parse_args()
    root=Path.cwd();release=Path(a.release).resolve();old=Path(a.baseline_study).resolve()
    # Reusing a completed baseline is valid only with the current frozen inputs.
    old_suite=json.loads((old/'suite.json').read_text())
    validate_references({n:Path(r['reference']) if Path(r['reference']).is_absolute()
                         else old/Path(r['reference']) for n,r in old_suite['references'].items()})
    rc=json.loads((release/'resume_transition.json').read_text());rm=json.loads((release/'manifest.json').read_text())
    assert rc['learning_states_exactly_verified'] and rm['complete'] and rm['train_clips']==4448
    previous=json.loads((old/'config.json').read_text());baseline_ready=json.loads((old/'minute_000/ready.json').read_text())
    assert baseline_ready['iteration']==10819 and len(baseline_ready['metrics'])==7
    assert all(v['full_decode_passed'] for v in baseline_ready['videos'])
    original_contract=json.loads((old/'minute_000/evaluation/development/bundle/contract.json').read_text())
    assert original_contract['policy_sha256']=='9a05919fa6f23bac77868d483d555a7458c1d3a615eac14a4580d1be5c501c64'
    zone=timezone(timedelta(hours=8));tag=datetime.now(zone).strftime('%Y%m%d_%H%M%S')
    run=root/'runs'/('lafan_bones_mix1h_bounded_'+tag)
    study=root/'evaluations'/('progress1h_bones_bounded_'+tag)
    run.mkdir();study.mkdir();(study/'checkpoints').mkdir()
    def copy_frozen(source,target):
        if Path(source).suffix in ('.mp4','.png','.npz','.pt','.onnx','.mjb'):
            os.link(source,target);return target
        return shutil.copy2(source,target)
    for name in ('references','reference_inputs'):
        shutil.copytree(old/name,study/name,copy_function=copy_frozen)
    shutil.copytree(old/'minute_000',study/'minute_000',copy_function=copy_frozen)
    (study/'history').mkdir()
    for name in ('identity.json','best.json'):shutil.copy2(old/'history'/name,study/'history'/name)
    suite=json.loads((old/'suite.json').read_text())
    for row in suite['references'].values():
        row['reference']=str(study/'references'/Path(row['reference']).name)
        assert sha(row['reference'])==row['reference_sha256']
    identity=json.loads((study/'history/identity.json').read_text())
    assert identity['references']=={name:row['reference_sha256'] for name,row in suite['references'].items()}
    atomic(study/'suite.json',suite)
    for name in ('bones_evaluation_selection.json','training_data_receipt.json'):shutil.copy2(old/name,study/name)
    shutil.copy2(old/'training_data_receipt.json',run/'data_manifest.json')
    shutil.copy2(release/'manifest.json',study/'bounded_release.json')
    shutil.copy2(release/'resume_transition.json',study/'resume_transition.json')
    baseline=study/'checkpoints/baseline_010819.pt';os.link(Path(previous['baseline_checkpoint']),baseline)
    resume=release/'resume_bounded_010819.pt';assert sha(resume)==rc['adapted_sha256']
    started=time.time();deadline=started+3600;iso=lambda t:datetime.fromtimestamp(t,zone).isoformat()
    env=dict(os.environ,DEX_MIMIC_ROOT=str(root),DEX_ASSETS=str(root/'assets/dex_evt'),PYTHONPATH=str(root/'src'),
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    command=[str(root/'.venv/bin/python'),'-u','-m','dex_mimic.cli','train','--config','configs/dex_blink_safe.json',
        '--data-release',str(release),'--switch-seconds','150','--switch-warmup-steps','64','--skip-evaluation',
        '--log_dir',str(run),'--resume',str(resume),'--iterations','1000000',
        '--max_hours',str(rc['wall_time']/3600+1),'--stop-file',str(run/'STOP')]
    atomic(run/'launch_command.json',dict(argv=command));shutil.copy2(root/'configs/dex_blink_safe.json',run/'package_config.json')
    with (run/'stdout.log').open('w') as log:
        train=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    cfg=dict(previous,study_dir=str(study),run_dir=str(run),resume_checkpoint=str(resume),baseline_checkpoint=str(baseline),
        suite=str(study/'suite.json'),started_epoch=started,deadline_epoch=deadline,started_at=iso(started),deadline_at=iso(deadline),
        train_pid=train.pid,data_release=str(release),resident_frame_capacity=rm['frame_capacity'],shards=rm['shards'],
        study_title='LAFAN + BONES 一小时混合续训（有界分片）',
        failed_full_resident_attempt=str(old),baseline_reused_from=str(old),
        history_description='第 10819 次基线复用此前相同 7 项完整评测及已验证视频，策略、模型和参考哈希一致；本轮独立建立后续最佳记录。')
    atomic(study/'config.json',cfg)
    atomic(study/'schedule.json',[dict(minute=m,target_at=iso(started+m*60)) for m in (0,30,60)])
    job=dict(minute=0,iteration=10819,checkpoint=str(baseline),checkpoint_sha256=sha(baseline),
        target_at=iso(started),captured_at=iso(started),capture_delay_seconds=0.,attempts=0,done=True)
    ready=dict(baseline_ready,target_at=iso(started),captured_at=iso(started),capture_delay_seconds=0.,
               baseline_evaluation_reused_from=str(old),baseline_weights_precede_bones_training=True)
    atomic(study/'minute_000/job.json',job);atomic(study/'minute_000/ready.json',ready)
    atomic(study/'jobs.json',{'0':job});atomic(study/'comparisons.json',[ready])
    pids={'training':train.pid}
    for key,args,logfile in (
        ('deadline_guard',[str(root/'deployment/progress_deadline_guard.py'),str(study)],study/'deadline_guard.log'),
        ('controller',[str(root/'deployment/progress_controller.py'),str(study)],study/'controller.log'),
        ('resource_monitor',[str(root/'deployment/monitor_training_resources.py'),'--pid',str(train.pid),'--out',str(run/'gpu_monitor.jsonl')],run/'resource_monitor.log')):
        with logfile.open('w') as log:
            child=subprocess.Popen([sys.executable,'-u',*args],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pids[key]=child.pid
    receipt=dict(pids=pids,started_at=iso(started),deadline_at=iso(deadline),run=str(run),study=str(study),
        release=str(release),all_selected_clips=4448,resident_frame_capacity=rm['frame_capacity'],shards=rm['shards'],
        prior_attempt_failure='full-resident CUDA allocation failure before first logged iteration',
        restart_counts_a_full_new_hour=True,baseline_reused_from=str(old))
    atomic(study/'launch_receipt.json',receipt)
    old_guard=json.loads((old/'launch_receipt.json').read_text())['pids']['deadline_guard']
    proc=Path('/proc')/str(old_guard)/'cmdline'
    if proc.exists():
        cmd=proc.read_bytes().replace(b'\0',b' ').decode()
        if 'progress_deadline_guard.py' in cmd and str(old) in cmd:os.kill(old_guard,signal.SIGTERM)
    (run/'train.pid').write_text(str(train.pid)+'\n')
    (root/'deployment/current_run.txt').write_text(str(run.relative_to(root))+'\n')
    (root/'deployment/current_progress_study.txt').write_text(str(study.relative_to(root))+'\n')
    (root/'deployment/current_bones_mix_release.txt').write_text(str(release.relative_to(root))+'\n')
    print('BOUNDED_MIX_STARTED',json.dumps(receipt),flush=True)


if __name__=='__main__':main()
