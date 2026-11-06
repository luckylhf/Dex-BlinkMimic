"""One isolated CPU evaluation and verified video comparison for a frozen checkpoint."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

study, job_path = map(lambda x: Path(x).resolve(), sys.argv[1:3])
config = json.loads((study/'config.json').read_text())
profile = config['evaluation_profile']
job = json.loads(job_path.read_text())
stage = job_path.parent
project = Path(config['run_dir']).parents[1]
env = dict(os.environ, DEX_MIMIC_ROOT=str(project), DEX_ASSETS=str(project/'assets/dex_evt'),
           PYTHONPATH=str(project/'src'), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl')

def phase(value):
    atomic(stage/'phase.json', dict(phase=value, updated_epoch=time.time()))

def run(command, logfile):
    with logfile.open('w') as log:
        subprocess.run(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

try:
    assert sha(job['checkpoint']) == job['checkpoint_sha256']
    out = stage/'evaluation'
    if not (out/'summary.json').exists():
        if out.exists():
            out.rename(stage/('incomplete_evaluation_'+str(int(time.time()))))
        phase('cpu_evaluation')
        run([sys.executable, '-m', 'dex_mimic.evaluation.evaluate', '--checkpoint', job['checkpoint'],
             '--suite', config['suite'], '--out', str(out), '--history', str(study/'history'), '--actor', 'ema'],
            stage/'evaluation.log')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['iteration'] == job['iteration'] and summary['evaluation_profile'] == profile
    comparison = json.loads((out/'comparison.json').read_text())
    if config.get('render_motions') is not None:
        comparison['motions'] = {name:comparison['motions'][name] for name in config['render_motions']}
    comparison.update(latest_label=f'+{job["minute"]:03d}min · EMA', best_column_label='已评测最佳 EMA',
                      latest_column_label='本轮 EMA', render_full_reference=True,
                      evaluation_note=('高度超差仅记录；倾斜超差仍终止；播完不等于跟踪合格'
                                       if profile != 'strict_v1' else '高度/倾斜超差即终止；播完不等于跟踪合格'))
    atomic(out/'comparison.json', comparison)
    if comparison['motions']:
        phase('rendering')
        run([sys.executable, '-m', 'dex_mimic.evaluation.render', '--config', str(out/'comparison.json')], stage/'render.log')
    phase('media_validation')
    videos, metrics = [], {}
    for name, row in summary['external'].items():
        report = json.loads((out/(name+'_native.json')).read_text())
        assert report['evaluation_profile'] == profile and report['physical_support'] is False
        metrics[name] = dict(row, mean_joint_rmse_rad=sum(x['joint_rmse_rad'] for x in report['metrics'])/len(report['metrics']),
                             reference_seconds=(report['reference']['num_frames']-1)*.02,
                             first_height_violation_seconds=report['first_height_violation_seconds'])
        if name not in comparison['motions']:
            continue
        path = out/'videos'/(name+'_best_vs_latest.mp4')
        evidence = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == evidence['video_sha256']
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(path)], text=True))
        stream = probe['streams'][0]
        expected = len(range(0, report['reference']['num_frames'], 2)) + 25
        assert stream['width'] == 1920 and stream['height'] == 1080 and stream['r_frame_rate'] == '25/1'
        assert int(stream['nb_frames']) == expected
        subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(path),'-f','null','-'], check=True)
        videos.append(dict(motion=name, path=str(path.relative_to(study)), sha256=sha(path),
                           frames=expected, video_seconds=float(probe['format']['duration']), full_decode_passed=True))
    best = json.loads((study/'history/best.json').read_text())
    ready = dict(minute=job['minute'], iteration=job['iteration'], target_at=job['target_at'],
                 captured_at=job['captured_at'], checkpoint_sha256=job['checkpoint_sha256'],
                 capture_delay_seconds=job['capture_delay_seconds'], metrics=metrics,
                 evaluated_best={k:dict(iteration=v['iteration'],metrics=v['metrics']) for k,v in best.items()},
                 videos=videos, completed_epoch=time.time(), evaluation_profile=profile, actor='ema')
    atomic(stage/'ready.json', ready)
    phase('ready')
    print('PROGRESS_EVALUATION_READY', job['minute'], job['iteration'], flush=True)
except Exception as error:
    atomic(stage/'error.json', dict(error=repr(error), time=time.time()))
    raise
