"""Describe pose tracking and global drift from existing physical trajectories."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import mujoco
from blinkmimic.robot import get_robot
from blinkmimic.kinematics import Kinematics
from blinkmimic.rotations import quat_to_mat,yaw_mat,mv3
from dex_mimic.common import atomic


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',required=True);a=ap.parse_args()
    os.nice(10);torch.set_num_threads(2)
    folder=Path(a.evaluation);suite=json.loads((folder/'suite.json').read_text())
    robot=get_robot('dex');kin=Kinematics(robot,'cpu');results={};series={}
    for actor in ['zero','fresh3500','previous34435']:
        root=folder/actor;model=mujoco.MjModel.from_binary_path(str(root/'development/bundle/model.mjb'))
        qadr=[int(model.jnt_qposadr[model.joint('robot/'+name).id]) for name in robot.joint_names]
        free=int(np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0]);rq=int(model.jnt_qposadr[free])
        results[actor]={};series[actor]={}
        for name,row in suite['references'].items():
            report=json.loads((root/(name+'_native.json')).read_text());N=len(report['metrics'])
            with np.load(root/(name+'_native.trajectory.npz')) as z:pose=z['qpos'][1:N+1].copy()
            with np.load(row['reference']) as z:
                refp=torch.from_numpy(z['root_p'][1:N+1].copy());refR=torch.from_numpy(z['root_R'][1:N+1].copy())
                refbody=torch.from_numpy(z['body_p'][1:N+1].copy());duration=(len(z['q'])-1)*.02
            p=torch.tensor(pose[:,rq:rq+3],dtype=torch.float32);R=quat_to_mat(torch.tensor(pose[:,rq+3:rq+7],dtype=torch.float32))
            q=torch.tensor(pose[:,qadr],dtype=torch.float32);zero=torch.zeros_like(p)
            bp=kin.forward(p,R,zero,zero,q,torch.zeros_like(q))[0]
            dR=yaw_mat(R@refR.transpose(-1,-2));anchor=torch.cat((p[:,:2],refp[:,2:3]),-1)
            aligned=anchor[:,None]+mv3(dR[:,None],refbody-refp[:,None])
            anchored=(aligned-bp).square().sum(-1).mean(-1).sqrt().numpy()
            glob=(refbody-bp).square().sum(-1).mean(-1).sqrt().numpy()
            recorded=np.array([r['body_global_rmse_m'] for r in report['metrics']])
            parity=float(abs(glob-recorded).max());assert parity<5e-5,(actor,name,parity)
            joints=np.array([r['joint_rmse_rad'] for r in report['metrics']])
            series[actor][name]=dict(global_body=recorded,anchored_body=anchored,joints=joints)
            results[actor][name]=dict(completed=report['completed'],seconds=report['elapsed_seconds'],reference_seconds=duration,
                termination_reason=report['termination_reason'],global_body_mean_m=float(recorded.mean()),
                pelvis_xy_yaw_aligned_body_mean_m=float(anchored.mean()),joint_rmse_mean_deg=float(joints.mean()*180/math.pi),
                max_root_error_m=max(r['root_error_m'] for r in report['metrics']),fk_recorded_metric_parity_max_m=parity)
    prefixes={}
    for other in ['zero','previous34435']:
        prefixes[other]={}
        for name in suite['references']:
            n=min(len(series[other][name]['global_body']),len(series['fresh3500'][name]['global_body']))
            prefixes[other][name]=dict(seconds=n*.02,**{
                actor:{key:float(values[:n].mean()) for key,values in series[actor][name].items()}
                for actor in [other,'fresh3500']})
    atomic(folder/'tracking_diagnostics.json',dict(results=results,common_prefix=prefixes,
        alignment='Same XY/yaw reference re-anchoring as training reward; reference height retained',
        limits='Three development tests are not a blind benchmark; previous34435 has different training budget/data/root observations'))
    print(json.dumps(results,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
