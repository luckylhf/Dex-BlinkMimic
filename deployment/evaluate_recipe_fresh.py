"""Fixed-input physical evaluation after fresh training; render only after GPU training exits."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='2'
os.environ['MUJOCO_GL']='egl'

from dex_mimic.common import atomic,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--postprocess-only',action='store_true')
    a=ap.parse_args();root=Path(__file__).resolve().parents[1];run=Path(a.run).resolve();out=Path(a.out).resolve()
    if a.postprocess_only:
        suite=json.loads((out/'suite.json').read_text())
        summaries=json.loads((out/'metrics.json').read_text())
    else:
        out.mkdir(parents=True,exist_ok=False);refs=out/'references';refs.mkdir()
        source=root/'evaluations/latest_three_materials_ckpt028765_20261107_070806/references'
        suite=dict(evaluation_profile='jump_tracking_v2',references={})
        for name in ['gbtc','dance_boy','dance_girl']:
            target=refs/(name+'.npz');shutil.copy2(source/(name+'.npz'),target)
            suite['references'][name]=dict(reference=str(target),reference_sha256=sha(target),training_member=False)
        from dex_mimic.deployment.build_bundle import prepare_motion
        for name in ['run1_subject5','walk1_subject5']:
            target=refs/(name+'.npz')
            prepare_motion(str(Path(a.dataset)/'motions'/(name+'.npz')),str(target),10)
            suite['references'][name]=dict(reference=str(target),reference_sha256=sha(target),training_member=True)
        atomic(out/'suite.json',suite)
        summaries={}
        checkpoints={'zero':run/'ckpt_000000.pt','fresh3500':run/'last.pt',
                     'previous34435':root/'runs/lafan_bones_live_untilnoon_20261106_220416/last.pt'}
        for name,checkpoint in checkpoints.items():
            atomic(out/'status.json',dict(phase='evaluating',actor=name,updated_at=datetime.now().astimezone().isoformat()))
            cmd=[sys.executable,'-m','dex_mimic.evaluation.evaluate','--checkpoint',str(checkpoint),'--suite',str(out/'suite.json'),
                 '--out',str(out/name),'--history',str(out/('history_'+name)),'--actor','ema','--allow-legacy-three-tests']
            with (out/(name+'.log')).open('w') as log:
                subprocess.run(cmd,cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
            summaries[name]=json.loads((out/name/'summary.json').read_text())
        atomic(out/'metrics.json',summaries)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    first=(run/'train_log.csv').open().readline().strip()
    values=np.atleast_2d(np.loadtxt(run/'train_log.csv',delimiter=',',skiprows=1 if first.startswith('iteration,') else 0))
    curves=out/'curves';curves.mkdir(exist_ok=True)
    panels=[('世界模型多步平均损失',3),('世界模型第8步误差',4),('真实仿真步奖励',10),
            ('归一化回合回报',12),('平均训练回合长度 / 步',11),('结束回合中终止所占比例',13),
            ('策略目标损失（不直接代表效果）',5),('迭代速度 / 次每秒',14)]
    fig,axes=plt.subplots(4,2,figsize=(16,14),layout='constrained')
    for ax,(title,col) in zip(axes.flat,panels):
        ax.plot(values[:,0],values[:,col],alpha=.3,lw=.8,color='#287f9e')
        if len(values)>=5:
            ax.plot(values[4:,0],np.convolve(values[:,col],np.ones(5)/5,mode='valid'),color='#005c7f')
        ax.axvline(1000,color='#c17a2e',ls='--',alpha=.5)
        ax.set_title(title);ax.set_xlabel('从头训练迭代');ax.grid(alpha=.2)
    fig.suptitle('固定去重 LAFAN + BONES · Blink 配方从头训练3500次\n4096环境；mirror=0.5；虚线为定向探测启用点；深线为5点后向均值',fontsize=16)
    fig.savefig(curves/'training_curves.png',dpi=150);plt.close(fig)
    progress=dict(first_iteration=int(values[0,0]),last_iteration=int(values[-1,0]),
        first500_medians={name:float(np.median(values[values[:,0]<=500,col])) for name,col in panels},
        last500_medians={name:float(np.median(values[values[:,0]>3000,col])) for name,col in panels})
    atomic(out/'training_summary.json',progress)

    comparison=dict(latest_folder=str(out/'fresh3500'),latest_iteration=3500,latest_label='从零训练 · Blink 配方 3500',
        evaluation_profile='jump_tracking_v2',transport='native',out=str(out/'videos'),motions={},
        best_column_label='零更新',latest_column_label='从头训练',comparison_title='零更新与3500次',
        render_full_reference=False,selection_note='固定输入；相同机器人与评测；零更新也含参考关节前馈',
        source_prefix='起点',latest_source_prefix='本轮')
    for name,row in suite['references'].items():
        comparison['motions'][name]=dict(reference=row['reference'],best_folder=str(out/'zero'),best_iteration=0,
            best_label='零更新 live 权重及归一化；未优化参数',same_model=False)
    atomic(out/'comparison.json',comparison)
    atomic(out/'status.json',dict(phase='rendering',updated_at=datetime.now().astimezone().isoformat()))
    with (out/'render.log').open('w') as log:
        subprocess.run([sys.executable,'-m','dex_mimic.evaluation.render','--config',str(out/'comparison.json')],cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
    atomic(out/'status.json',dict(phase='complete',updated_at=datetime.now().astimezone().isoformat(),videos=[p.name for p in (out/'videos').glob('*.mp4')]))
    print('EVALUATION_COMPLETE',out,flush=True)


if __name__=='__main__':main()
