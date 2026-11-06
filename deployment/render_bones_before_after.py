"""Fixed pre-BONES vs half-hour comparison from existing recorded physics only."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from dex_mimic.common import atomic,sha
from dex_mimic.evaluation.evaluate import POLICY_FIELDS


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--minute',type=int,default=30);a=p.parse_args()
    study=Path(a.study).resolve();stage=study/f'minute_{a.minute:03d}';out=stage/'before_after'
    out.mkdir(exist_ok=False)
    baseline=study/'minute_000/evaluation';latest=stage/'evaluation'
    before=json.loads((baseline/'summary.json').read_text());after=json.loads((latest/'summary.json').read_text())
    assert before['iteration']==10819 and after['evaluation_profile']==before['evaluation_profile']=='jump_tracking_v2'
    contracts=[json.loads((folder/'development/bundle/contract.json').read_text()) for folder in (baseline,latest)]
    assert {k:v for k,v in contracts[0].items() if k not in POLICY_FIELDS}=={k:v for k,v in contracts[1].items() if k not in POLICY_FIELDS}
    for folder,contract in zip((baseline,latest),contracts):
        assert sha(folder/'development/bundle/policy.onnx')==contract['policy_sha256']
    suite=json.loads((study/'suite.json').read_text());common={}
    for name in before['external']:
        x=json.loads((baseline/f'{name}_native.json').read_text());y=json.loads((latest/f'{name}_native.json').read_text())
        assert x['model_sha256']==y['model_sha256'] and x['reference']['source_sha256']==y['reference']['source_sha256']
        assert x['physical_support'] is y['physical_support'] is False
        n=min(len(x['metrics']),len(y['metrics']));xs,ys=x['metrics'][:n],y['metrics'][:n]
        assert all(r['time']==s['time'] for r,s in zip(xs,ys))
        keys=('body_global_rmse_m','joint_rmse_rad','height_error_m')
        avg=lambda rows:{k:sum(z[k] for z in rows)/n for k in keys}
        common[name]=dict(common_steps=n,common_seconds=n*.02,before=avg(xs),after=avg(ys),
            before_max_root_error_m=max(z['root_error_m'] for z in xs),after_max_root_error_m=max(z['root_error_m'] for z in ys),
            before_seconds=x['elapsed_seconds'],after_seconds=y['elapsed_seconds'],
            before_completed=x['completed'],after_completed=y['completed'],
            before_full_metrics=before['external'][name],after_full_metrics=after['external'][name])
    atomic(out/'metrics.json',dict(before_iteration=before['iteration'],after_iteration=after['iteration'],
        evaluation_profile='jump_tracking_v2',metrics=common))
    config=dict(latest_folder=str(latest),latest_iteration=after['iteration'],latest_label='BONES +30min',
        evaluation_profile='jump_tracking_v2',transport='native',out=str(out/'videos'),
        best_column_label='未加 BONES · EMA',latest_column_label='加入 BONES 半小时 · EMA',
        comparison_title='加入前后固定对照',render_full_reference=True,
        selection_note=f'固定对照：基线 {before["iteration"]:,} 次 vs 半小时 {after["iteration"]:,} 次；同一参考、同一规则',
        source_prefix='基线',latest_source_prefix='加入后',motions={})
    for name in ('gbtc','dance_boy','dance_girl'):
        config['motions'][name]=dict(reference=suite['references'][name]['reference'],best_folder=str(baseline),
            best_iteration=before['iteration'],best_label=f'未加 BONES #{before["iteration"]}',same_model=False)
    atomic(out/'comparison.json',config)
    atomic(out/'status.json',dict(state='rendering',updated_epoch=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',MUJOCO_GL='egl')
    with (out/'render.log').open('w') as log:
        subprocess.run([sys.executable,'-u','-m','dex_mimic.evaluation.render','--config',str(out/'comparison.json')],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic(out/'status.json',dict(state='media_validation',updated_epoch=time.time()))
    videos=[]
    for name in config['motions']:
        path=out/'videos'/(name+'_best_vs_latest.mp4');evidence=json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==evidence['video_sha256']
        assert [r['iteration'] for r in evidence['runs']]==[before['iteration'],after['iteration']]
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(path)],text=True))
        stream=probe['streams'][0]
        with np.load(suite['references'][name]['reference'],allow_pickle=False) as z:expected=len(range(0,len(z['q']),2))+25
        assert stream['width']==1920 and stream['height']==1080 and stream['r_frame_rate']=='25/1' and int(stream['nb_frames'])==expected
        subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(path),'-f','null','-'],check=True)
        videos.append(dict(motion=name,path=str(path.relative_to(out)),sha256=sha(path),full_decode_passed=True,frames=expected))
    job=json.loads((stage/'job.json').read_text())
    receipt=dict(before_iteration=before['iteration'],after_iteration=after['iteration'],minute=a.minute,
        checkpoint_captured_at=job['captured_at'],before_checkpoint_sha256=contracts[0]['checkpoint_sha256'],
        after_checkpoint_sha256=contracts[1]['checkpoint_sha256'],before_policy_sha256=before['policy_sha256'],
        after_policy_sha256=after['policy_sha256'],evaluation_profile='jump_tracking_v2',videos=videos,
        completed_at=datetime.now().astimezone().isoformat())
    lines=['# BONES 混合续训半小时：固定加入前后对照','',
        f'加入前：第 {before["iteration"]} 次 EMA；加入后半小时采样：第 {after["iteration"]} 次 EMA。视频三栏固定为参考 / 未加 BONES / 加入 BONES 半小时，不使用逐动作最佳替换基线。','',
        '参考文件、物理模型、控制参数和 jump_tracking_v2 规则相同。没有重跑策略、重置、支撑或修改轨迹，只渲染此前记录的完整物理回放。高度超差只记录，倾斜超差终止。','',
        '| 视频 | 加入前闭环/秒 | 加入后闭环/秒 | 完整参考/秒 |','|---|---:|---:|---:|']
    for name in config['motions']:
        r=common[name];v=next(v for v in videos if v['motion']==name)
        lines.append(f'| [{name}]({v["path"]}) | {r["before_seconds"]:.2f} | {r["after_seconds"]:.2f} | {r["after_full_metrics"].get("reference_seconds", json.loads((latest/f"{name}_native.json").read_text())["reference"]["num_frames"]*.02-.02):.2f} |')
    lines+=['','## 相同有效时间窗的位置误差','',
        '舞蹈两次回放时长不同，以下仅比较各动作共同有效前缀，不能把不同长度的整段平均误差直接当作同区间改善。','',
        '| 动作 | 共同时间窗/秒 | 加入前全身平均误差/厘米 | 加入后全身平均误差/厘米 | 变化 |',
        '|---|---:|---:|---:|---:|']
    for name,r in common.items():
        x=r['before']['body_global_rmse_m'];y=r['after']['body_global_rmse_m']
        lines.append(f'| {name} | {r["common_seconds"]:.2f} | {x*100:.2f} | {y*100:.2f} | {(y/x-1)*100:+.1f}% |')
    lines+=['','外部舞蹈的时长和全身位置误差改善；原 LAFAN 走跑仍完整完成，但位置精度回退。BONES 待机和转身起跑控制的误差下降。dance-boy 播完仍有 0.28 秒高度超差记录；dance-girl 未完成完整 241.56 秒。不能把完成等同于跟踪验收。','',
        '这是加入 BONES 并额外续训半小时后的前后比较。没有同等时长、同等分片条件的纯 LAFAN 续训对照，不能将全部变化单独归因为 BONES 数据。','',
        '视频 1920×1080、25 fps、1 倍速，参考持续到完整末尾；策略终止后定格并标注。相机各自跟随骨盆，下方轨迹保持同一世界坐标和等比例尺度。视频已完整解码并核验哈希。数值见 metrics.json，保存点和媒体身份见 ready.json。','']
    (out/'README.md').write_text('\n'.join(lines));atomic(out/'ready.json',receipt)
    atomic(out/'status.json',dict(state='completed',updated_epoch=time.time()))
    print('BEFORE_AFTER_READY',json.dumps(receipt),flush=True)


if __name__=='__main__':main()
