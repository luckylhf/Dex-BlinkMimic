"""Chronological search for the first retained EMA checkpoints completing both clips."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dex_mimic.common import atomic, sha

NAMES = ('walk1_subject5', 'run1_subject5')
RUNS = ('lafan_all77_package8192_20261106_134056', 'lafan_all77_continue3h_20261106_153921')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    root=Path.cwd();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,DEX_MIMIC_ROOT=str(root),DEX_ASSETS=str(root/'assets/dex_evt'),
             PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',MUJOCO_GL='egl')
    os.environ.update(env)
    pair=root/'evaluations/lafan_pair_ckpt010819_20261106'
    suite=json.loads((pair/'suite.json').read_text())
    assert suite['evaluation_profile']=='jump_tracking_v2' and set(suite['references'])==set(NAMES)
    for row in suite['references'].values():assert sha(row['reference'])==row['reference_sha256']
    atomic(out/'suite.json',suite)
    inventory=[]
    for run in RUNS:
        for path in (root/'runs'/run).glob('ckpt_*.pt'):
            inventory.append(dict(iteration=int(path.stem.split('_')[-1]),checkpoint=str(path),
                                  run=run,bytes=path.stat().st_size))
    inventory.sort(key=lambda r:r['iteration'])
    assert len({r['iteration'] for r in inventory})==len(inventory)
    atomic(out/'inventory.json',inventory)
    provenance={str(p.relative_to(root)):sha(p) for p in (
        root/'src/dex_mimic/deployment/native_mujoco.py',root/'src/dex_mimic/deployment/runtime.py',
        root/'src/dex_mimic/deployment/build_bundle.py',root/'src/dex_mimic/evaluation/criteria.py')}
    atomic(out/'protocol.json',dict(actor='ema',evaluation_profile='jump_tracking_v2',transport='native',
           reference_suite=suite,source_hashes=provenance,max_steps=0,one_start=True,mid_rollout_resets=False,
           search='Every retained checkpoint in increasing iteration order; no monotonicity assumption',
           scope='The 77-clip 8192-env training lineage; independent earlier 4096-env trials are excluded'))
    records=[];first={};joint=None

    def publish(state, **extra):
        atomic(out/'status.json',dict(state=state,updated_at=datetime.now().astimezone().isoformat(),
               tested_count=len(records),available_checkpoints=len(inventory),first_completed=first,
               first_both_completed=joint,**extra))
        atomic(out/'records.json',records)
        with (out/'progress.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['iteration','training_wall_minutes','saved_at','motion','elapsed_seconds',
                'completed','tracking_criteria_passed','mean_body_global_rmse_m','max_root_error_m','termination_reason'])
            for r in records:
                for name,m in r['metrics'].items():w.writerow([r['iteration'],r['training_wall_minutes'],r['saved_at'],name,
                    m['elapsed_seconds'],m['completed'],m['tracking_criteria_passed'],m['mean_body_global_rmse_m'],
                    m['max_root_error_m'],m['termination_reason']])

    try:
        import torch
        torch.set_num_threads(2)
        (out/'evaluations').mkdir();(out/'logs').mkdir()
        publish('searching')
        for candidate in inventory:
            checkpoint=Path(candidate['checkpoint']);digest=sha(checkpoint)
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert ck['iteration']==candidate['iteration'] and ck['args']['num_envs']==8192
            meta=dict(iteration=ck['iteration'],training_wall_minutes=ck['wall_time']/60,
                      saved_at=ck['saved_at'],checkpoint_sha256=digest,run=candidate['run'])
            del ck
            publish('evaluating',current_iteration=meta['iteration'])
            folder=out/'evaluations'/f'ckpt_{meta["iteration"]:06d}'
            if meta['iteration']==10819:
                folder=pair/'evaluation'
                assert json.loads((folder/'development/bundle/contract.json').read_text())['checkpoint_sha256']==digest
            else:
                with (out/'logs'/f'ckpt_{meta["iteration"]:06d}.log').open('w') as log:
                    subprocess.run([sys.executable,'-u','-m','dex_mimic.evaluation.evaluate',
                        '--checkpoint',str(checkpoint),'--suite',str(out/'suite.json'),'--out',str(folder),
                        '--history',str(out/'history'),'--actor','ema'],cwd=root,env=env,
                        stdout=log,stderr=subprocess.STDOUT,check=True)
            summary=json.loads((folder/'summary.json').read_text())
            assert summary['iteration']==meta['iteration'] and summary['evaluation_profile']=='jump_tracking_v2'
            result=dict(meta,evaluation_folder=str(folder),metrics=summary['external'])
            records.append(result)
            for name in NAMES:
                if result['metrics'][name]['completed'] and name not in first:
                    first[name]=dict(meta,metrics=result['metrics'][name],
                        immediately_preceding_tested_checkpoint=(dict(iteration=records[-2]['iteration'],
                            metrics=records[-2]['metrics'][name]) if len(records)>1 else None))
            if all(result['metrics'][name]['completed'] for name in NAMES):joint=dict(meta,metrics=result['metrics'])
            publish('searching')
            print('CHECKPOINT_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
            if joint is not None:break
        assert joint is not None, 'No retained checkpoint completed both clips'
        receipt=dict(first_completed=first,first_both_completed=joint,tested_count=len(records),
                     first_retained_iteration=inventory[0]['iteration'],searched_iterations=[r['iteration'] for r in records],
                     all_earlier_retained_checkpoints_tested=True,actor='ema',evaluation_profile='jump_tracking_v2',
                     exact_learning_iteration_between_saves_unknown=True,records=records)
        atomic(out/'result.json',receipt)
        lines=['# LAFAN 行走与跑步：最早完整跟踪保存点','',
               '范围为正式 77 段 / 8192 实例训练链（初始训练及其三小时续训）。此前独立 4096 实例试跑不混入本训练链。', '',
               '逐个按迭代顺序检查保存点，不假设能力随训练单调提升；固定 EMA、jump_tracking_v2、原生 CPU MuJoCo + ONNX CPU、相同两条完整参考、从首帧初始化一次、无支撑或中途重置。高度超差只记录，倾斜超差终止。', '',
               '| 动作 | 首次完成迭代 | 累计训练分钟 | 保存时间 | 完整闭环秒 | 全身平均误差/米 | 最大根部偏差/米 |',
               '|---|---:|---:|---|---:|---:|---:|']
        for name,r in first.items():
            m=r['metrics'];lines.append(f'| {name} | {r["iteration"]} | {r["training_wall_minutes"]:.2f} | {r["saved_at"]} | {m["elapsed_seconds"]:.2f} | {m["mean_body_global_rmse_m"]:.3f} | {m["max_root_error_m"]:.3f} |')
        lines+=['',f'首次两段同时完成：第 {joint["iteration"]} 次保存点，累计训练 {joint["training_wall_minutes"]:.2f} 分钟。', '',
                f'共检查 {len(records)} 个保存点；这些首次完成点之前的所有保留保存点都已实测。最早保留点是第 {inventory[0]["iteration"]} 次；未保存的迭代无法恢复，因此结论是“最早可核验保存点”，不是精确到某一次更新的首次学会时刻。', '',
                '完成表示该固定协议下单次连续播到末尾，不等于任意初态、随机扰动、后续所有保存点或真机均稳定；精度需结合位置误差判断。', '',
                '| 迭代 | 累计训练分钟 | 行走闭环秒 | 行走完成 | 跑步闭环秒 | 跑步完成 |', '|---:|---:|---:|---|---:|---|']
        for r in records:
            w,run=(r['metrics'][n] for n in NAMES)
            lines.append(f'| {r["iteration"]} | {r["training_wall_minutes"]:.2f} | {w["elapsed_seconds"]:.2f} | {w["completed"]} | {run["elapsed_seconds"]:.2f} | {run["completed"]} |')
        lines+=['','逐保存点证据见 evaluations/、records.json、progress.csv；完整结论见 result.json；输入及实现身份见 suite.json 和 protocol.json。','']
        (out/'README.md').write_text('\n'.join(lines));publish('completed')
        print('FIRST_COMPLETION_FOUND',json.dumps(dict(first=first,joint=joint),ensure_ascii=False),flush=True)
    except Exception as error:
        publish('failed',error=repr(error));raise


if __name__=='__main__':main()
