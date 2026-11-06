"""Launch the authorized one-hour mixed-data continuation with durable comparisons."""
import argparse
from datetime import datetime,timezone,timedelta
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from blinkmimic.dex.dex_constants import load_dex_npz
from dex_mimic.common import atomic,sha
from dex_mimic.deployment.build_bundle import prepare_motion
from dex_mimic.evaluation.three_tests import default_suite, validate_references, PREPROCESSING


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
    root=Path.cwd();dataset=Path(a.dataset).resolve()
    data=json.loads((dataset/'manifest.json').read_text());transition=json.loads((dataset/'resume_transition.json').read_text())
    assert data['lafan_included']==77 and data['bones_included']>0 and transition['learning_states_exactly_verified']
    zone=timezone(timedelta(hours=8));tag=datetime.now(zone).strftime('%Y%m%d_%H%M%S')
    run=root/'runs'/('lafan_bones_mix1h_'+tag);study=root/'evaluations'/('progress1h_bones_'+tag)
    run.mkdir();study.mkdir()
    for name in ('checkpoints','references','reference_inputs'): (study/name).mkdir()
    atomic(study/'previous_pointers.json',{n:(root/'deployment'/n).read_text() for n in ('current_run.txt','current_progress_study.txt')})
    for target in (study/'training_data_receipt.json',run/'data_manifest.json'):shutil.copy2(dataset/'manifest.json',target)
    shutil.copy2(dataset/'resume_transition.json',study/'resume_transition.json')
    pair=root/'evaluations/lafan_pair_ckpt010819_20261106'
    suite=dict(evaluation_profile='jump_tracking_v2',reference_preprocessing=PREPROCESSING,references={})
    for suite_path,names in ((default_suite(root),('gbtc','dance_boy','dance_girl')),(pair/'suite.json',('walk1_subject5','run1_subject5'))):
        original=json.loads(suite_path.read_text())
        for name in names:
            source=Path(original['references'][name]['reference'])
            if not source.is_absolute():source=suite_path.parent/source
            dst=study/'references'/source.name
            shutil.copy2(source,dst);assert sha(dst)==original['references'][name]['reference_sha256']
            suite['references'][name]=dict(reference=str(dst),reference_sha256=sha(dst),training_member=name.startswith(('walk','run')))
    validate_references({n:r['reference'] for n,r in suite['references'].items()})
    # Select by reference geometry/speed, before evaluating or training the actor on this snapshot.
    candidates=[]
    for row in sorted((r for r in data['motions'] if r['corpus']=='bones' and r['frames']>=250),key=lambda r:-r['frames']):
        label=Path(row['source_bvh']).stem.lower()
        if any(word in label for word in ('chair','sitting','sit_','lying','floor','crawl','kneel','squat','getup','get_up')):continue
        pos,quat,joints,_=load_dex_npz(dataset/'motions'/row['training_file'])
        upright=1-2*(quat[:,1]**2+quat[:,2]**2)
        if float(pos[:,2].min())<.78 or float(upright.min())<.8:continue
        candidates.append(row)
    assert len(candidates)>=2,'Need two non-object upright BONES controls'
    static=min(candidates,key=lambda r:(r['horizontal_speed_p95_m_s'],-r['frames']))
    dynamic=max((r for r in candidates if r['name']!=static['name']),key=lambda r:(r['horizontal_speed_p95_m_s'],r['frames']))
    selected={}
    for name,row in (('bones_static',static),('bones_dynamic',dynamic)):
        source=dataset/'motions'/row['training_file'];copied=study/'reference_inputs'/(name+'.npz')
        shutil.copy2(source,copied);assert sha(copied)==row['training_sha256']
        dst=study/'references'/(name+'.npz');prepare_motion(str(copied),str(dst),10)
        suite['references'][name]=dict(reference=str(dst),reference_sha256=sha(dst),training_member=True)
        selected[name]=dict(row,selection='upright accepted motion; lower/higher root speed among eligible snapshot controls')
    atomic(study/'bones_evaluation_selection.json',dict(selected_before_training=True,motions=selected))
    atomic(study/'suite.json',suite)
    source=root/'runs/lafan_all77_continue3h_20261106_153921/ckpt_010819.pt'
    baseline=study/'checkpoints/baseline_010819.pt';shutil.copy2(source,baseline)
    assert sha(baseline)=='390d740ffd40debc866cb4e6c5ba358dcbbd461418acb134e6df923851899822'
    resume=dataset/'resume_mixed_010819.pt';assert sha(resume)==transition['adapted_sha256']
    started=time.time();deadline=started+3600
    iso=lambda t:datetime.fromtimestamp(t,zone).isoformat()
    env=dict(os.environ,DEX_MIMIC_ROOT=str(root),DEX_ASSETS=str(root/'assets/dex_evt'),PYTHONPATH=str(root/'src'),
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    command=[str(root/'.venv/bin/python'),'-u','-m','dex_mimic.cli','train','--config','configs/dex_blink_safe.json',
             '--motion_dir',str(dataset/'motions'),'--log_dir',str(run),'--resume',str(resume),
             '--iterations','1000000','--max_hours',str(transition['wall_time']/3600+1),
             '--stop-file',str(run/'STOP')]
    atomic(run/'launch_command.json',dict(argv=command,environment={k:env[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    shutil.copy2(root/'configs/dex_blink_safe.json',run/'package_config.json')
    with (run/'stdout.log').open('w') as log:
        process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    cfg=dict(study_dir=str(study),run_dir=str(run),parent_run=str(source.parent),resume_checkpoint=str(resume),
        baseline_checkpoint=str(baseline),resume_iteration=10819,started_epoch=started,deadline_epoch=deadline,
        started_at=iso(started),deadline_at=iso(deadline),duration_seconds=3600,interval_seconds=1800,
        milestone_minutes=[0,30,60],suite=str(study/'suite.json'),evaluation_profile='jump_tracking_v2',
        evaluation_actor='ema',num_envs=8192,training_clip_count=data['clips'],train_pid=process.pid,
        render_motions=['gbtc','dance_boy','dance_girl'],study_title='LAFAN + BONES 一小时混合续训',
        training_data_description=f'LAFAN 77 段 + BONES {data["bones_included"]} 段已验收快照；基础采样权重各 50%，自适应采样保留',
        history_description='以第 10819 次基线重新建立本轮最佳记录；7 项数值评测，3 条训练外动作生成比对视频。',
        dataset=str(dataset),data_manifest_sha256=sha(dataset/'manifest.json'),training_termination_changed=False)
    atomic(study/'config.json',cfg)
    atomic(study/'schedule.json',[dict(minute=m,target_at=iso(started+m*60)) for m in cfg['milestone_minutes']])
    pids={'training':process.pid}
    for key,args,logfile in (
        ('deadline_guard',[str(root/'deployment/progress_deadline_guard.py'),str(study)],study/'deadline_guard.log'),
        ('controller',[str(root/'deployment/progress_controller.py'),str(study)],study/'controller.log'),
        ('resource_monitor',[str(root/'deployment/monitor_training_resources.py'),'--pid',str(process.pid),'--out',str(run/'resources.jsonl')],run/'resource_monitor.log')):
        with logfile.open('w') as log:
            child=subprocess.Popen([sys.executable,'-u',*args],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pids[key]=child.pid
    atomic(study/'launch_receipt.json',dict(pids=pids,started_at=cfg['started_at'],deadline_at=cfg['deadline_at'],dataset=str(dataset),run=str(run),study=str(study)))
    (run/'train.pid').write_text(str(process.pid)+'\n')
    (root/'deployment/current_run.txt').write_text(str(run.relative_to(root))+'\n')
    (root/'deployment/current_progress_study.txt').write_text(str(study.relative_to(root))+'\n')
    (root/'deployment/current_bones_mix_dataset.txt').write_text(str(dataset.relative_to(root))+'\n')
    print('MIX_TRAINING_STARTED',json.dumps(dict(config=cfg,pids=pids),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
