"""Durable half-hour checkpoint capture, serialized evaluation and progress reporting."""
from pathlib import Path
from collections import deque
import csv
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

study = Path(sys.argv[1]).resolve()
lock = (study/'controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
cfg = json.loads((study/'config.json').read_text())
final_minute = max(cfg['milestone_minutes'])
run = Path(cfg['run_dir'])
project = run.parents[1]
zone = datetime.timezone(datetime.timedelta(hours=8))

def iso(epoch):
    return datetime.datetime.fromtimestamp(epoch, zone).isoformat()

def atomic(path, data):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n'); temp.replace(path)

def read(path, default):
    return json.loads(path.read_text()) if path.exists() else default

def alive(pid):
    try:
        stat = Path('/proc')/str(pid)/'stat'
        return stat.read_text().split(') ',1)[1][0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False

def capture(minute):
    target = cfg['deadline_epoch'] if minute == final_minute else cfg['started_epoch'] + minute*60
    if minute == 0:
        source, iteration = Path(cfg['baseline_checkpoint']), cfg['resume_iteration']
    else:
        candidates = [(int(m.group(1)),p) for p in run.glob('ckpt_*.pt') if (m:=re.fullmatch(r'ckpt_(\d+)\.pt',p.name))]
        if not candidates:
            return None
        iteration, source = max(candidates)
    stage = study/f'minute_{minute:03d}'; stage.mkdir(exist_ok=True)
    data_path=source.with_suffix('.dataset.json')
    if minute>0 and cfg.get('live_data_dir') and not data_path.exists():return None
    frozen = study/'checkpoints'/f'minute_{minute:03d}_ckpt_{iteration:06d}.pt'
    shutil.copy2(source, frozen)
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    now = time.time()
    job = dict(minute=minute, iteration=iteration, checkpoint=str(frozen), checkpoint_sha256=digest,
               target_at=iso(target), captured_at=iso(now), capture_delay_seconds=max(0., now-target), attempts=0)
    if data_path.exists():
        data=json.loads(data_path.read_text());assert data['iteration']==iteration
        atomic(stage/'data_at_checkpoint.json',data);job['data_at_checkpoint']=data
    atomic(stage/'job.json', job)
    return job

jobs = read(study/'jobs.json', {})
records = read(study/'comparisons.json', [])
active = None
last_status = 0

def publish(state):
    global last_status
    ready = [f'minute_{r["minute"]:03d}' for r in records]
    status = dict(state=state, updated_at=iso(time.time()), training_alive=alive(cfg['train_pid']),
                  train_pid=cfg['train_pid'], started_at=cfg['started_at'], deadline_at=cfg['deadline_at'],
                  ready_milestones=ready, captured_minutes=sorted(int(m) for m in jobs),
                  failed_minutes=[int(m) for m,j in jobs.items() if j.get('failed')],
                  evaluation_profile=cfg['evaluation_profile'],
                  data_growth=read(Path(cfg['live_data_dir'])/'status.json',{}) if cfg.get('live_data_dir') else None,
                  history_bootstrap=read(study/'history_bootstrap.json', {}),
                  active_minute=active['minute'] if active else None,
                  active_phase=read(study/f'minute_{active["minute"]:03d}'/'phase.json', {}) if active else None)
    atomic(study/'status.json', status)
    atomic(study/'comparisons.json', sorted(records,key=lambda r:r['minute']))
    lines = ['# '+cfg.get('study_title','三小时续训：每半小时进展'), '',
             f'训练开始：{cfg["started_at"]}；计划截止：{cfg["deadline_at"]}。从第 {cfg["resume_iteration"]} 次完整状态恢复。', '',
             f'{cfg.get("num_envs",8192)} 实例；'+cfg.get('training_data_description','全部 77 段训练数据')+'；网络、奖励、优化器等沿用原包参数，截止后保存并退出。', '',
             f'每 {cfg.get("evaluation_interval_minutes",30)} 分钟冻结最新保存点，以固定参考集、{cfg["evaluation_profile"]}、原生 CPU MuJoCo + ONNX CPU 协议评测 EMA。'+cfg.get('history_description','已评测最佳以同规则重评的第 1899 次作为历史种子，随后纳入基线 3500 和本次采样。'), '',
             ('本轮每小时输出五动作指标与 checkpoint；训练期间不自动渲染视频，避免抢占显存。五动作 12659 留底视频另行保留。'
              if cfg.get('render_motions') == [] else '视频三栏：参考运动学 / 已评测最佳 EMA / 本次 EMA。高度超差只记为跟踪偏差，不截断；倾斜超差仍终止，但不等同于摔倒认定。无支撑和中途重置。'
              if cfg['evaluation_profile'] != 'strict_v1' else '视频三栏：参考运动学 / 已评测最佳 EMA / 本次 EMA。高度或倾斜超差即终止，终止后定格；无支撑和中途重置。'), '',
             '| 续训分钟 | 迭代 | GBTC 闭环秒 | dance-boy 闭环秒 | dance-girl 闭环秒 | 视频 |',
             '|---:|---:|---:|---:|---:|---|']
    for record in sorted(records,key=lambda r:r['minute']):
        metrics = record['metrics']; minute=record['minute']
        links=' / '.join(f'[{v["motion"]}]({v["path"]})' for v in record['videos'])
        if not links:links=f'[指标](minute_{minute:03d}/ready.json)'
        lines.append(f'| {minute} | {record["iteration"]} | {metrics["gbtc"]["elapsed_seconds"]:.2f} | {metrics["dance_boy"]["elapsed_seconds"]:.2f} | {metrics["dance_girl"]["elapsed_seconds"]:.2f} | {links} |')
    lines += ['', '完成时长、平均全身位置误差、最大根部偏差和平均关节误差见 comparisons.json 与 progress.csv。播完不代表跟踪精度已验收。', '',
              '定时任务取到的 checkpoint 可能早于时点约 3 分钟（沿用原包保存频率）；实际迭代、捕获时间、哈希均有记录。最后一个时点等待最终保存完成后取最终 checkpoint。', '',
              'status.json 显示执行状态；单次错误在 minute_XXX/error.json。视频仅在完整解码与哈希验证通过后进入 ready_milestones，供本地同步。', '']
    (study/'README.md').write_text('\n'.join(lines))
    with (study/'progress.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['minute','iteration','motion','elapsed_seconds','reference_seconds','completed','mean_body_global_rmse_m','max_root_error_m','mean_joint_rmse_rad','elapsed_change_from_start_seconds','body_error_change_from_start_m','evaluation_profile','outcome','tracking_criteria_passed','height_violation_seconds'])
        baseline=next((r for r in records if r['minute']==0),None)
        for r in sorted(records,key=lambda r:r['minute']):
            for name,m in r['metrics'].items():
                b=baseline['metrics'][name] if baseline else m
                writer.writerow([r['minute'],r['iteration'],name,m['elapsed_seconds'],m['reference_seconds'],m['completed'],m['mean_body_global_rmse_m'],m['max_root_error_m'],m['mean_joint_rmse_rad'],m['elapsed_seconds']-b['elapsed_seconds'],m['mean_body_global_rmse_m']-b['mean_body_global_rmse_m'],r['evaluation_profile'],m.get('outcome'),m['tracking_criteria_passed'],m.get('height_violation_seconds')])
    last_status=time.time()

try:
    publish('starting')
    while True:
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
        now=time.time();training_alive=alive(cfg['train_pid'])
        if now>=cfg['deadline_epoch'] and not (run/'STOP').exists():
            (run/'STOP').write_text('Authorized continuation deadline reached\n')
        if not training_alive and now < cfg['deadline_epoch']-5:
            atomic(study/'training_error.json', dict(time=iso(now),error='Training exited before the requested deadline',log=str(run/'stdout.log')))
            publish('failed_training_exited_early')
            break
        for minute in cfg['milestone_minutes']:
            key=str(minute)
            target = cfg['deadline_epoch'] if minute == final_minute else cfg['started_epoch']+minute*60
            if key in jobs or now<target:
                continue
            if minute==final_minute and training_alive:
                continue
            job=capture(minute)
            if job:
                jobs[key]=job;atomic(study/'jobs.json',jobs)
        if active is not None:
            stage=study/f'minute_{active["minute"]:03d}'
            ready=read(stage/'ready.json',None)
            if ready:
                if not any(r['minute']==ready['minute'] for r in records):records.append(ready)
                jobs[str(active['minute'])]['done']=True
                active=None;atomic(study/'jobs.json',jobs);publish('waiting')
            elif not alive(active['pid']):
                entry=jobs[str(active['minute'])]
                if entry['attempts']>=2:entry['failed']=True
                active=None;atomic(study/'jobs.json',jobs);publish('worker_error')
        bootstrap = read(study/'history_bootstrap.json', {})
        if bootstrap.get('error'):
            raise RuntimeError('History baseline re-evaluation failed: '+bootstrap['error'])
        if active is None and (not cfg.get('requires_history_bootstrap') or bootstrap.get('ready')):
            pending=[j for j in jobs.values() if not j.get('done') and not j.get('failed')]
            if pending:
                job=min(pending,key=lambda j:j['minute']);stage=study/f'minute_{job["minute"]:03d}'
                existing=read(stage/'worker_pid.json',{})
                if existing and alive(existing['pid']):
                    pid=existing['pid']
                else:
                    job['attempts']+=1
                    with (stage/'worker.log').open('a') as log:
                        process=subprocess.Popen(['nice','-n','10',sys.executable,str(project/'deployment/progress_eval_worker.py'),str(study),str(stage/'job.json')],cwd=project,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                    pid=process.pid;atomic(stage/'worker_pid.json',{'pid':pid})
                active=dict(minute=job['minute'],pid=pid);atomic(study/'jobs.json',jobs);publish('evaluating')
        if not training_alive and len(jobs)==len(cfg['milestone_minutes']) and all(j.get('done') or j.get('failed') for j in jobs.values()):
            publish('completed_with_errors' if any(j.get('failed') for j in jobs.values()) else 'completed')
            break
        if time.time()-last_status>=10:publish('evaluating' if active else 'waiting')
        time.sleep(2)
except Exception as error:
    atomic(study/'controller_error.json',dict(error=repr(error),time=iso(time.time())))
    publish('controller_failed')
    raise
