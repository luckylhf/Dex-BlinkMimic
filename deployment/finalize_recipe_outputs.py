"""Finish saved-trajectory comparisons and verify media; never starts training."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import shutil

from dex_mimic.common import atomic,sha


def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',required=True);ap.add_argument('--run',required=True);ap.add_argument('--wait-pid',type=int)
    a=ap.parse_args();ev=Path(a.evaluation).resolve();run=Path(a.run).resolve();root=Path(__file__).resolve().parents[1]
    if a.wait_pid:
        while alive(a.wait_pid):time.sleep(10)
    assert json.loads((ev/'status.json').read_text())['phase']=='complete'
    config=json.loads((ev/'comparison.json').read_text())
    config.update(out=str(ev/'videos_previous'),best_column_label='上一轮最终',latest_column_label='本轮3500',
                  comparison_title='上一轮最终与本轮3500',selection_note='数据分布、观测与训练预算不同；此处只比较实际效果',source_prefix='上一轮')
    config['motions']={name:dict(row,best_folder=str(ev/'previous34435'),best_iteration=34435,
                                best_label='上一轮最终 EMA 34435') for name,row in config['motions'].items() if name in ['gbtc','dance_boy','dance_girl']}
    atomic(ev/'comparison_previous.json',config)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','MUJOCO_GL':'egl','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','PYTHONPATH':str(root/'src')}
    atomic(ev/'finalize_status.json',dict(phase='rendering_previous',updated_at=datetime.now().astimezone().isoformat()))
    with (ev/'render_previous.log').open('w') as log:
        subprocess.run([sys.executable,'-m','dex_mimic.evaluation.render','--config',str(ev/'comparison_previous.json')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    import imageio_ffmpeg
    from PIL import Image
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();videos=[];pngs=[]
    atomic(ev/'finalize_status.json',dict(phase='verifying_media',updated_at=datetime.now().astimezone().isoformat()))
    for directory in ['videos','videos_previous']:
        for p in sorted((ev/directory).glob('*.mp4')):
            result=subprocess.run([ffmpeg,'-v','error','-i',str(p),'-f','null','-'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
            assert result.returncode==0 and not result.stderr.strip(),(p,result.stderr)
            row=dict(file=str(p.relative_to(ev)),bytes=p.stat().st_size,sha256=sha(p),full_decode_passed=True)
            ffprobe=shutil.which('ffprobe')
            if ffprobe:
                info=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_entries','format=duration:stream=width,height,codec_name','-of','json',str(p)],text=True))
                row.update(duration_seconds=float(info['format']['duration']),streams=info['streams'])
            videos.append(row)
        for p in sorted((ev/directory).glob('*.png')):
            with Image.open(p) as im:im.verify()
            pngs.append(str(p.relative_to(ev)))
    assert len(videos)==8,len(videos)
    atomic(ev/'media_receipt.json',dict(passed=True,videos=videos,pngs_verified=pngs,metrics_sha256=sha(ev/'metrics.json'),
                                      training_checkpoint_sha256=json.loads((run/'training_receipt.json').read_text())['checkpoint_sha256']))
    atomic(ev/'finalize_status.json',dict(phase='complete',videos=len(videos),updated_at=datetime.now().astimezone().isoformat()))
    atomic(run/'status.json',dict(phase='complete',iteration=3500,evaluation=str(ev),
                                postprocess_recovery='CSV header handled; original physical evaluations retained',videos_verified=len(videos)))
    print('ALL_OUTPUTS_VERIFIED',len(videos),flush=True)


if __name__=='__main__':main()
