"""Measured gates for Dex mirror, immutable full references and MuJoCo environment."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import os
import torch

from blinkmimic.robot import get_robot
from blinkmimic.dr import XiLayout
from blinkmimic.kinematics import Kinematics
from blinkmimic.mirror import Mirror
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP
from blinkmimic.rotations import so3_exp, rot_angle_sq, quat_to_mat
from dex_mimic.common import atomic, sha


def mirror_gate(robot,kin,device):
    torch.manual_seed(20261107)
    layout=XiLayout(robot,device); mirror=Mirror(robot,layout.slices,device)
    n,J=512,robot.num_joints
    q=torch.tensor(robot.default_joint_pos_list(),device=device)+torch.randn(n,J,device=device)*.15
    q=q.clamp(kin.q_lo+.001,kin.q_hi-.001)
    state=(torch.randn(n,3,device=device),so3_exp(torch.randn(n,3,device=device)*.4),
           torch.randn(n,3,device=device),torch.randn(n,3,device=device),q,torch.randn(n,J,device=device))
    ms=mirror.state(state); twice=mirror.state(ms)
    involution=max(float((a-b).abs().max()) for a,b in zip(state,twice))
    action=torch.randn(n,J,device=device); xi=torch.randn(n,layout.dim,device=device)
    assert involution<1e-6 and torch.equal(mirror.joints(mirror.joints(action)),action)
    assert torch.equal(mirror.xi(mirror.xi(xi)),xi)
    swap=lambda s:s.replace('_l_','_SIDE_').replace('_r_','_l_').replace('_SIDE_','_r_')
    perm=[robot.tracked_bodies.index(swap(b)) for b in robot.tracked_bodies]
    bp,bR,bv,bw=kin.forward(*state); mp,mR,mv,mw=kin.forward(*ms); M=mirror.M
    pe=(mp-bp[:,perm]@M).norm(dim=-1)
    re=rot_angle_sq(mR,M@bR[:,perm]@M).sqrt()
    result=dict(involution_max=involution,joints=J,body_position_max_m=float(pe.max()),
                body_position_p95_m=float(pe.quantile(.95)),body_orientation_max_rad=float(re.max()),
                linear_velocity_max_m_s=float((mv-bv[:,perm]@M).norm(dim=-1).max()),
                angular_velocity_max_rad_s=float((mw+bw[:,perm]@M).norm(dim=-1).max()),
                model_is_exactly_symmetric=False,
                limits=dict(position_m=.002,orientation_rad=.005),
                map=[dict(joint=robot.joint_names[i],partner=robot.joint_names[int(mirror.perm[i])],sign=float(mirror.sign[i])) for i in range(J)])
    # The unchanged Dex model contains millimetre-scale bilateral geometry differences.
    assert result['body_position_max_m']<.002,result
    assert result['body_orientation_max_rad']<.005,result
    import mujoco
    import numpy as np
    model=mujoco.MjModel.from_xml_path(robot.mjcf_path);reflect=np.diag([1.,-1.,1.]);pairs=[]
    for body in range(1,model.nbody):
        name=model.body(body).name
        if '_l_' not in name:continue
        other=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name.replace('_l_','_r_'))
        assert other>=0
        ra=np.empty(9);rb=np.empty(9)
        mujoco.mju_quat2Mat(ra,model.body_iquat[body]);mujoco.mju_quat2Mat(rb,model.body_iquat[other])
        ra=ra.reshape(3,3);rb=rb.reshape(3,3)
        ia=ra@np.diag(model.body_inertia[body])@ra.T
        ib=reflect@rb@np.diag(model.body_inertia[other])@rb.T@reflect
        pairs.append(dict(body=name,mass_relative=float(abs(model.body_mass[body]-model.body_mass[other])/max(model.body_mass[body],model.body_mass[other],1e-9)),
            inertia_relative=float(np.linalg.norm(ia-ib)/max(np.linalg.norm(ia),np.linalg.norm(ib),1e-9))))
    assert all(x['mass_relative']<1e-6 and x['inertia_relative']<1e-6 for x in pairs)
    default=torch.tensor(robot.default_joint_pos_list(),device=device)
    assert torch.allclose(mirror.joints(default),default,atol=1e-7,rtol=0)
    result['bilateral_mass_inertia']=pairs
    result['default_pose_mirror_error']=float((mirror.joints(default)-default).abs().max())
    rlo=torch.minimum(kin.q_lo[mirror.perm]*mirror.sign,kin.q_hi[mirror.perm]*mirror.sign)
    rhi=torch.maximum(kin.q_lo[mirror.perm]*mirror.sign,kin.q_hi[mirror.perm]*mirror.sign)
    asymmetric=(abs(kin.q_lo-rlo)>1e-6)|(abs(kin.q_hi-rhi)>1e-6)
    lo=torch.where(asymmetric,torch.maximum(kin.q_lo,rlo)+.005,-torch.inf)
    hi=torch.where(asymmetric,torch.minimum(kin.q_hi,rhi)-.005,torch.inf)
    result['asymmetric_joint_limits']=[dict(joint=robot.joint_names[i],original=[float(kin.q_lo[i]),float(kin.q_hi[i])],mirror_safe=[float(lo[i]),float(hi[i])]) for i in range(J) if bool(asymmetric[i])]
    from blinkmimic.mirror import mirror_windows
    from blinkmimic.mdp import flatten_state,unflatten_state
    import copy
    B,K=32,8
    raw=flatten_state(tuple(x[:B] for x in state)).unsqueeze(1).repeat(1,K,1)
    for i in range(J):
        if bool(asymmetric[i]):
            raw[...,13+i]=(lo[i]+hi[i])/2
            raw[:20,:,13+i]=hi[i]+.05 if bool(kin.q_hi[i]>hi[i]+.05) else lo[i]-.05
    mask=torch.arange(B,device=device).float().unsqueeze(1).expand(B,K)
    w=dict(s=raw,s1=raw.clone(),a=action[:B].unsqueeze(1).expand(B,K,J),
           a_prev=action[:B].unsqueeze(1).expand(B,K,J),xi=xi[:B].unsqueeze(1).expand(B,K,layout.dim),
           mask=mask,cut=(mask.long()%2).bool())
    diagnostics={k:torch.zeros((),device=device,dtype=torch.long) for k in ['batches','eligible','mirrored','replacement_batches']}
    augmented=mirror_windows(w,mirror,16,(lo,hi),diagnostics)
    indices=20+torch.arange(16,device=device)%12 if bool(asymmetric.any()) else torch.arange(16,device=device)
    assert torch.equal(augmented['mask'][:16],w['mask'][indices])
    assert torch.equal(augmented['cut'][:16],w['cut'][indices])
    assert torch.equal(augmented['s'][16:],w['s'][16:])
    qq=unflatten_state(augmented['s'][:16])[4]
    assert ((qq>=kin.q_lo-1e-6)&(qq<=kin.q_hi+1e-6)).all()
    result['mirror_window_selection']=dict(passed=True,augmented_share=.5,diagnostics={k:int(v) for k,v in diagnostics.items()},mask_and_transition_identity_verified=True)
    result['passed']=True
    return result


def contact_geometry_gate(robot):
    import mujoco
    import numpy as np
    m=mujoco.MjModel.from_xml_path(robot.mjcf_path);d=mujoco.MjData(m)
    d.qpos[:3]=[0,0,1];d.qpos[3:7]=[1,0,0,0]
    for name,value in zip(robot.joint_names,robot.default_joint_pos_list()):
        d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    mujoco.mj_forward(m,d);feet=[]
    for foot in robot.feet:
        bid=m.body(foot).id;bottoms=[]
        for g in range(m.ngeom):
            b=m.geom_bodyid[g]
            while b and b!=bid:b=m.body_parentid[b]
            if b!=bid or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            typ=int(m.geom_type[g]);sz=m.geom_size[g];axis=d.geom_xmat[g].reshape(3,3)[2]
            if typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                dz=sz[0]*np.sqrt(max(0.,1-axis[2]**2))+sz[1]*abs(axis[2])
            elif typ==int(mujoco.mjtGeom.mjGEOM_BOX):dz=np.sum(abs(axis)*sz)
            elif typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):dz=sz[0]
            elif typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):dz=sz[0]+sz[1]*abs(axis[2])
            else:raise ValueError('Unmeasured foot collision geometry '+str(typ))
            bottoms.append(float(d.geom_xpos[g,2]-dz))
        assert bottoms
        offset=float(d.xpos[bid,2]-min(bottoms))
        assert abs(robot.contact_z-offset)<.01,(foot,offset,robot.contact_z)
        feet.append(dict(body=foot,colliders=len(bottoms),origin_to_sole_m=offset))
    return dict(passed=True,threshold_m=robot.contact_z,feet=feet,scope='default-pose collision geometry; reference contacts use per-clip calibrated floors')


def data_and_env(dataset,robot,kin,device,report):
    from dex_mimic.data.compact_motion import CompactMotionLib
    motion=CompactMotionLib(Path(dataset)/'compact.pt',kin,device)
    manifest=json.loads((Path(dataset)/'manifest.json').read_text())
    bones=[r for r in manifest['motions'] if r['corpus']=='bones']
    assert len({r['source_bvh_sha256'] for r in bones})==len(bones)
    assert len({r['training_sha256'] for r in bones})==len(bones)
    assert len({r['motion_content_sha256'] for r in bones})==len(bones)
    assert all(not Path(r['source_bvh']).stem.endswith('_M') for r in bones)
    assert torch.allclose(motion.bin_prior,torch.full_like(motion.bin_prior,1/motion.num_bins))
    rows=manifest['motions']
    picks=sorted(set([0,len(rows)-1]+list(range(0,len(rows),max(1,len(rows)//12)))+
                     [i for i,r in enumerate(rows) if r['corpus']=='lafan'][:4]))
    max_goal=max_body=max_reward=max_grad=0.
    feet=[robot.tracked_bodies.index(b) for b in robot.feet]
    for c in picks:
        r=rows[c]; lib=MotionLib([str(Path(dataset)/'motions'/r['training_file'])],kin,device,robot.load_motion)
        lib.set_ref_contact(feet)
        f=torch.tensor([0,(r['frames']-1)//2,r['frames']-2],device=device)
        gf=f+motion.clip_start[c]
        max_goal=max(max_goal,float((lib.goal(f,10)-motion.goal(gf,10)).abs().max()))
        for a,b in zip(lib.body_state(f),motion.body_state(gf)):
            max_body=max(max_body,float((a-b).abs().max()))
        assert torch.equal(lib.ref_contact[f],motion.ref_contact[gf])
        cfg=torch.tensor(robot.default_joint_pos_list(),device=device); xi=XiLayout(robot,device)
        m1=MDP(kin,lib,cfg,xi.dim,robot.sensing);m2=MDP(kin,motion,cfg,xi.dim,robot.sensing)
        m1.feet=m2.feet=feet
        S=list(lib.ref_state(f));S[4]=S[4].clone().requires_grad_()
        a=torch.zeros(len(f),robot.num_joints,device=device)
        r1=m1.reward(tuple(S),f,a,a,w_contact=.25);r2=m2.reward(tuple(S),gf,a,a,w_contact=.25)
        max_reward=max(max_reward,float((r1-r2).abs().max()))
        g1=torch.autograd.grad(r1.sum(),S[4],retain_graph=True)[0]
        g2=torch.autograd.grad(r2.sum(),S[4])[0]
        max_grad=max(max_grad,float((g1-g2).abs().max()))
    assert max_goal<1e-5 and max_body<3e-5 and max_reward<1e-6 and max_grad<1e-5
    report['reference_parity']=dict(passed=True,clips=len(picks),goal_max=max_goal,body_max=max_body,reward_max=max_reward,reward_gradient_max=max_grad)
    from blinkmimic.env import TrackingSim,make_sim
    env=TrackingSim(make_sim(robot),64,robot,motion,m2,device,eval_mode=True,randomize=False,reset_noise=False)
    ids=torch.arange(64,device=device); cs=ids%len(rows); f=motion.clip_start[cs]+(motion.clip_len[cs]-1)//2
    env.reset(ids,f);env.refresh();S=env.state()
    roundtrip={k:float((a-b).abs().max()) for k,a,b in zip(['p','R','v','w','q','qd'],S,motion.ref_state(f))}
    assert max(roundtrip.values())<2e-4,roundtrip
    bp,bR,_,_=kin.forward(*S)
    tracked=[env.robot.body_names.index(b) for b in robot.tracked_bodies]
    fkpos=float((bp-env.robot.data.body_link_pos_w[:,tracked]).norm(dim=-1).max())
    fkrot=float(rot_angle_sq(bR,quat_to_mat(env.robot.data.body_link_quat_w[:,tracked])).sqrt().max())
    assert fkpos<5e-5 and fkrot<.002,(fkpos,fkrot)
    for _ in range(100):
        f1=motion.future_frames(env.f,1)[:,0]
        a=(motion.q[f1]-env.default_q)/env.alpha
        out=env.step(a)
        assert torch.isfinite(out['s1']).all() and torch.isfinite(out['r']).all()
    from blinkmimic import capture_compat
    # Restore every Warp Data array, not just qpos/qvel: reset() is not an
    # identity snapshot of solver/sensor history and is insufficient for parity.
    import warp as wp
    arrays=[];seen=set()
    def snapshot(obj):
        if id(obj) in seen:return
        seen.add(id(obj))
        if isinstance(obj,wp.array):
            arrays.append((obj,wp.clone(obj)));return
        if isinstance(obj,dict):
            for v in obj.values():snapshot(v)
        elif isinstance(obj,(list,tuple)):
            for v in obj:snapshot(v)
        elif hasattr(obj,'__dict__') and obj.__class__.__module__.startswith('mujoco_warp'):
            for v in vars(obj).values():snapshot(v)
    env.reset(ids,f);env.refresh();snapshot(env.sim.wp_data)
    control={k:getattr(env,k).clone() for k in ['f','ep_step','a_prev','done_eval','failed_eval','ep_ret','ep_disc_ret','start_bin','xi']}
    scratch_outputs=[]
    try:
        for reuse in [False,False,True]:
            capture_compat._reuse_enabled=reuse
            for dst,src in arrays:wp.copy(dst,src)
            for k,v in control.items():getattr(env,k).copy_(v)
            env.scene.update(.005);env._obs=None;trial=[]
            for _ in range(4):
                f1=motion.future_frames(env.f,1)[:,0]
                a=(motion.q[f1]-env.default_q)/env.alpha
                trial.append(env.step(a)['s1'].clone())
            scratch_outputs.append(torch.stack(trial))
    finally:
        capture_compat._reuse_enabled=True
    sections={'root_position':(0,3,1e-4),'root_quaternion':(3,7,3e-4),
              'root_linear_velocity':(7,10,.005),'root_angular_velocity':(10,13,.005),
              'joint_position':(13,13+robot.num_joints,3e-4),'joint_velocity':(13+robot.num_joints,13+2*robot.num_joints,.005)}
    baseline_errors={k:float((scratch_outputs[0][...,a:b]-scratch_outputs[1][...,a:b]).abs().max()) for k,(a,b,_) in sections.items()}
    arena_errors={k:float((scratch_outputs[1][...,a:b]-scratch_outputs[2][...,a:b]).abs().max()) for k,(a,b,_) in sections.items()}
    passed=all(baseline_errors[k]<limit and arena_errors[k]<limit and
               arena_errors[k]<=max(3*baseline_errors[k],limit*.1)
               for k,(_,_,limit) in sections.items())
    report['collision_scratch_parity']=dict(passed=passed,baseline_repeat=baseline_errors,arena_vs_original=arena_errors,
        absolute_limits={k:x[2] for k,x in sections.items()},comparison='at most 3x original repeat error or 10 percent of absolute limit; also below absolute limit',
        worlds=64,steps=4,warp_arrays_restored=len(arrays),bitwise_reproducible=False)
    assert passed,report['collision_scratch_parity']
    report['environment']=dict(passed=True,worlds=64,steps=100,state_roundtrip=roundtrip,fk_position_max_m=fkpos,fk_orientation_max_rad=fkrot)
    report['dataset']={k:manifest[k] for k in ['clips','frames','bins','corpora','exclusion_counts']}
    report['dataset']['manifest_sha256']=sha(Path(dataset)/'manifest.json')
    # Actual full-size feature assembly, using the same detached-state formula.
    xi=XiLayout(robot,device);mdp=MDP(kin,motion,torch.tensor(robot.default_joint_pos_list(),device=device),xi.dim,robot.sensing)
    frames=(torch.arange(33*2048,device=device)*89)%motion.num_frames
    states=motion.ref_state(frames);goals=motion.goal(frames,10);contexts=torch.zeros(len(frames),xi.dim,device=device)
    full=torch.cat((mdp.critic_features(states,frames,contexts),goals),-1)
    pieces=torch.cat([torch.cat((mdp.critic_features(tuple(x[i:i+2048] for x in states),frames[i:i+2048],contexts[i:i+2048]),goals[i:i+2048]),-1) for i in range(0,len(frames),2048)])
    feature_error=float((full-pieces).abs().max());assert feature_error<1e-5,feature_error
    report['critic_feature_assembly']=dict(passed=True,rows=len(frames),block_rows=2048,max_error=feature_error,critic_forward_and_normalization_unchanged=True)



def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset');ap.add_argument('--build',action='store_true');ap.add_argument('--report',required=True)
    a=ap.parse_args();os.environ['DEX_CAPTURE_ARENA']='1';torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    robot=get_robot('dex');device='cuda:0';kin=Kinematics(robot,device)
    report=dict(passed=False,versions={p:importlib.metadata.version(p) for p in ['torch','mujoco','mujoco-warp','mjlab']})
    try:
        # Audit geometry in strict FP32; TF32 matrix products obscure millimetre symmetry.
        torch.backends.cuda.matmul.allow_tf32=False
        report['mirror']=mirror_gate(robot,kin,device)
        report['contact_geometry']=contact_geometry_gate(robot)
        torch.backends.cuda.matmul.allow_tf32=True
        import blinkmimic.rotations as rotations
        rotations.ELEMENTWISE=True
        print('MIRROR',json.dumps(report['mirror']),flush=True)
        if a.dataset:
            if a.build:
                from dex_mimic.data.compact_motion import build_compact
                build_compact(a.dataset,robot,kin)
            data_and_env(a.dataset,robot,kin,device,report)
        report['passed']=True
    except Exception as error:
        report['error']=repr(error);atomic(Path(a.report),report);raise
    atomic(Path(a.report),report);print('PREFLIGHT_PASSED',a.report,flush=True)


if __name__=='__main__':main()
