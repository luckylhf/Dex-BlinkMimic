"""Evaluate the final actor on preselected clean walking/running training clips."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from dex_mimic.common import atomic, sha

NAMES = ('walk1_subject5', 'run1_subject5')
CHECKPOINT_SHA = '390d740ffd40debc866cb4e6c5ba358dcbbd461418acb134e6df923851899822'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    project=Path.cwd();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,DEX_MIMIC_ROOT=str(project),DEX_ASSETS=str(project/'assets/dex_evt'),
             PYTHONPATH=str(project/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',MUJOCO_GL='egl')
    os.environ.update(env)

    def phase(state, **extra):
        atomic(out/'status.json',dict(state=state,updated_at=datetime.now().astimezone().isoformat(),**extra))

    def run(command, logfile):
        with Path(logfile).open('w') as f:
            subprocess.run(command,cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

    try:
        import numpy as np
        import torch
        from dex_mimic.deployment.build_bundle import prepare_motion
        from blinkmimic.rotations import quat_to_mat
        torch.set_num_threads(2)
        phase('preparing_references')
        for name in ('inputs','references','checkpoint','logs'): (out/name).mkdir()
        manifest=json.loads((project/'deployment/all77_data_receipt.json').read_text())
        rows={x['name']:x for x in manifest['motions']}
        selected={}
        suite=dict(evaluation_profile='jump_tracking_v2',references={})
        for name in NAMES:
            row=rows[name];assert row['original_quality_status']=='accepted' and not row['original_quality_flags']
            source=project/'data/lafan1/motions_all77_exploratory'/row['training_file']
            assert sha(source)==row['training_sha256']
            copied=out/'inputs'/source.name;shutil.copy2(source,copied);assert sha(copied)==row['training_sha256']
            reference=out/'references'/source.name
            prepare_motion(str(copied),str(reference),10)
            with np.load(copied,allow_pickle=False) as a,np.load(reference,allow_pickle=False) as b:
                assert float(a['fps'])==50. and str(a['quat_order'])=='wxyz'
                assert np.array_equal(a['body_pos_w'][:,0],b['root_p'])
                assert np.array_equal(a['joint_pos'],b['q'])
                expected=quat_to_mat(torch.tensor(a['body_quat_w'][:,0],dtype=torch.float32)).numpy()
                assert np.array_equal(expected,b['root_R'])
                meta=json.loads(str(b['metadata']));n=len(a['joint_pos'])
                speed=np.linalg.norm(np.diff(a['body_pos_w'][:,0,:2],axis=0)*50,axis=1)
                stats=dict(seconds=(n-1)*.02,root_horizontal_speed_median_m_s=float(np.median(speed)),
                           root_horizontal_speed_p95_m_s=float(np.quantile(speed,.95)))
            selected[name]=dict(title=name+(' · 行走' if name.startswith('walk') else ' · 跑步'),
                               source=str(source),input_copy=str(copied),reference=str(reference),
                               reference_sha256=sha(reference),training_sha256=row['training_sha256'],
                               source_bvh_sha256=row['source_sha256'],original_quality_status='accepted',
                               original_quality_flags=[],mode=row['mode'],frames=n,training_member=True,
                               selection_reason='较低动态的基础行走对照' if name.startswith('walk') else '较高速度的跑步动态对照',
                               root_joint_rotation_arrays_unchanged=True,model_sha256=meta['model_sha256'],**stats)
            suite['references'][name]=dict(reference=str(reference),reference_sha256=sha(reference),training_member=True)
        # Record the selection before seeing either rollout result.
        atomic(out/'selection_receipt.json',dict(selected_before_rollout=True,motions=selected,
               scope='Two accepted training members: locomotion contrast, not coverage of all 77 clips or a stand-only test'))
        atomic(out/'suite.json',suite)
        run_dir=project/(project/'deployment/current_run.txt').read_text().strip()
        source=run_dir/'ckpt_010819.pt';assert sha(source)==CHECKPOINT_SHA
        checkpoint=out/'checkpoint'/source.name;shutil.copy2(source,checkpoint);assert sha(checkpoint)==CHECKPOINT_SHA
        phase('native_rollouts',iteration=10819)
        evaluation=out/'evaluation'
        run([sys.executable,'-u','-m','dex_mimic.evaluation.evaluate','--checkpoint',str(checkpoint),
             '--suite',str(out/'suite.json'),'--out',str(evaluation),'--history',str(out/'history'),'--actor','ema'],out/'logs/evaluation.log')
        summary=json.loads((evaluation/'summary.json').read_text())
        assert summary['iteration']==10819 and summary['evaluation_profile']=='jump_tracking_v2'
        config=dict(out=str(out/'videos'),iteration=10819,evaluation_profile='jump_tracking_v2',
                    evaluation_folder=str(evaluation),motions=selected)
        atomic(out/'render_config.json',config)
        atomic(out/'rollout_summary.json',summary)
        phase('rendering',iteration=10819,motions=list(NAMES))
        def render_one(name):
            run([sys.executable,'-u',str(project/'deployment/render_reference_policy_pair.py'),
                 '--config',str(out/'render_config.json'),'--motion',name],out/'logs'/f'render_{name}.log')
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(render_one,NAMES))
        phase('media_validation',iteration=10819)
        videos=[];metrics={}
        for name in NAMES:
            path=out/'videos'/f'{name}_checkpoint_010819_reference_policy.mp4'
            evidence=json.loads(path.with_suffix('.json').read_text())
            assert sha(path)==evidence['video_sha256'] and evidence['checkpoint_sha256']==CHECKPOINT_SHA
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(path)],text=True))
            stream=probe['streams'][0]
            assert stream['width']==1280 and stream['height']==960 and stream['r_frame_rate']=='25/1'
            assert int(stream['nb_frames'])==evidence['frames']
            subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(path),'-f','null','-'],check=True)
            videos.append(dict(motion=name,path=str(path.relative_to(out)),sha256=sha(path),
                               full_decode_passed=True,frames=evidence['frames'],seconds=float(probe['format']['duration'])))
            report=json.loads((evaluation/f'{name}_native.json').read_text())
            assert report['physical_support'] is False
            early=[x for x in report['metrics'] if x['time']<=5.+1e-8]
            metrics[name]=dict(summary['external'][name],reference_seconds=selected[name]['seconds'],
                mean_joint_rmse_rad=sum(x['joint_rmse_rad'] for x in report['metrics'])/len(report['metrics']),
                first_height_violation_seconds=report['first_height_violation_seconds'],
                initial_window_seconds=early[-1]['time'],
                initial_mean_body_global_rmse_m=sum(x['body_global_rmse_m'] for x in early)/len(early),
                initial_max_root_error_m=max(x['root_error_m'] for x in early))
        receipt=dict(iteration=10819,actor='ema',evaluation_profile='jump_tracking_v2',checkpoint_sha256=CHECKPOINT_SHA,
                     selection=selected,metrics=metrics,videos=videos,completed_at=datetime.now().astimezone().isoformat())
        lines=['# 最终第 10819 次模型：LAFAN 行走与跑步对照','',
               '两段均从本次实际训练的 77 段中预先选出，原转换质量均为 accepted、无质量 flags。选择依据是较低动态的行走与较高速度的跑步，不依据模型跑出的效果挑选。两段均参与训练，属于训练内跟踪测试，不代表训练外泛化或 77 段全面验收。','',
               '原始 50 Hz 训练输入逐帧保留位置、关节和旋转；没有裁剪、变速或重新重定向。每段只从首帧初始化一次，无支撑、无中途重置。参考为运动学，策略为真实记录的 MuJoCo 闭环轨迹。','',
               '| 比对视频 | 完整参考/秒 | 实际闭环/秒 | 结果 | 平均全身误差/米 | 最大根部偏差/米 |',
               '|---|---:|---:|---|---:|---:|']
        for name,m in metrics.items():
            video=next(v for v in videos if v['motion']==name)
            outcome='完整播放' if m['completed'] else '倾斜超差终止'
            lines.append(f'| [{name}]({video["path"]}) | {m["reference_seconds"]:.2f} | {m["elapsed_seconds"]:.2f} | {outcome} | {m["mean_body_global_rmse_m"]:.3f} | {m["max_root_error_m"]:.3f} |')
        lines+=['','jump_tracking_v2：高度误差 >0.25 米只记录，倾斜矩阵竖直分量差 >0.8 仍终止。完成不等于跟踪精度合格，倾斜终止也不是额外的接触式摔倒分类。终止后画面定格并标注，不补造后续轨迹。', '',
                '两段开头也含动作变化，不能替代纯静站能力测试。最初最多 5 秒的误差另存于 ready.json，便于观察起始阶段。','',
                '视频 1280×960、25 fps、1 倍速，完整参考持续展示，结束加 1 秒定格。两栏相机使用相同角度和距离，分别跟随自身骨盆；下方轨迹使用同一世界坐标及等比例坐标轴。','',
                f'固定 checkpoint SHA-256：`{CHECKPOINT_SHA}`。选材与数组一致性见 selection_receipt.json；完整数值和媒体哈希见 ready.json。两个成片均通过完整解码与帧数检查。','']
        (out/'README.md').write_text('\n'.join(lines))
        atomic(out/'ready.json',receipt);phase('completed',iteration=10819)
        print('LAFAN_PAIR_READY',json.dumps(metrics,ensure_ascii=False),flush=True)
    except Exception as error:
        phase('failed',error=repr(error));raise


if __name__=='__main__':main()
