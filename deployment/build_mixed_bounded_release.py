"""Cache every frozen mixed-data clip in bounded packs without altering quality labels."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path

import torch
import blinkmimic.motion
import blinkmimic.kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.robot import get_robot
from blinkmimic.kinematics import Kinematics
from dex_mimic.common import atomic,sha
from dex_mimic.data.bounded_motion import pack_motion,BoundedMotionLib
from dex_mimic.paths import ASSETS


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    p.add_argument('--capacity',type=int,default=600000);a=p.parse_args()
    dataset=Path(a.dataset).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'features').mkdir();source=json.loads((dataset/'manifest.json').read_text());rows=source['motions']
    assert len(rows)==4448 and all(33<=r['frames']<=a.capacity for r in rows)
    count=max(1,math.ceil(sum(r['frames'] for r in rows)/a.capacity))
    while True:
        shards=[[] for _ in range(count)];sizes=[0]*count;groups=[Counter() for _ in range(count)]
        for row in sorted(rows,key=lambda r:(r['corpus'],-r['frames'],r['name'])):
            eligible=[i for i in range(count) if sizes[i]+row['frames']<=a.capacity]
            if not eligible:count+=1;break
            i=min(eligible,key=lambda i:(groups[i][row['corpus']],sizes[i],i))
            shards[i].append(row);sizes[i]+=row['frames'];groups[i][row['corpus']]+=row['frames']
        else:break
    manifest=dict(complete=False,snapshot=False,source_frozen_snapshot=True,source_library_complete=False,
        completion_scope='All 4448 explicitly frozen experimental training clips; not the full BONES library',
        training_policy='Preserve authorized LAFAN accepted/review labels; BONES accepted only',
        source_dataset=str(dataset),source_manifest_sha256=sha(dataset/'manifest.json'),train_clips=len(rows),
        train_frames=source['frames'],shards=count,frame_capacity=a.capacity,base_sampling_mass=dict(lafan=.5,bones=.5),
        original_quality_counts=dict(Counter(r['original_quality_status'] for r in rows)),
        created_at=datetime.now().astimezone().isoformat())
    atomic(out/'manifest.json',manifest)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    robot=get_robot('dex');kin=Kinematics(robot,'cuda:0');index=[]
    for i,part in enumerate(shards):
        part.sort(key=lambda r:r['name']);totals={g:sum(r['bins'] for r in part if r['corpus']==g) for g in ('lafan','bones')}
        assert all(totals.values())
        weights={r['name']:.5*r['bins']/totals[r['corpus']] for r in part}
        paths=[str(dataset/'motions'/r['training_file']) for r in part]
        for r,path in zip(part,paths):assert sha(path)==r['training_sha256']
        lib=MotionLib(paths,kin,'cuda:0',robot.load_motion)
        assert lib.names==[r['name'] for r in part] and lib.num_frames==sum(r['frames'] for r in part)
        clip_weights=torch.tensor([weights[n] for n in lib.names],device='cuda:0')
        bins=torch.bincount(lib.bin_clip,minlength=len(part))
        lib.bin_prior.copy_(clip_weights[lib.bin_clip]/bins[lib.bin_clip]);lib.bin_prior.div_(lib.bin_prior.sum())
        pack=pack_motion(lib);pack['source_hashes']={r['name']:r['training_sha256'] for r in part}
        bounded=BoundedMotionLib(pack,'cuda:0',a.capacity)
        f=torch.cat((lib.clip_start,lib.clip_end,(lib.clip_start+lib.clip_end)//2))
        assert torch.equal(lib.goal(f,10),bounded.goal(f,10)),'Goal parity failed'
        shard_manifest=out/f'shard_{i:03d}.json';atomic(shard_manifest,dict(records=part,weights=weights,capacity=a.capacity))
        target=out/'features'/f'shard_{i:03d}.pt';torch.save(pack,target)
        index.append(dict(pack=str(target.relative_to(out)),manifest=shard_manifest.name,pack_sha256=sha(target),
            manifest_sha256=sha(shard_manifest),frames=lib.num_frames,clips=len(part),goal_boundary_midpoint_parity=True,
            corpora={g:dict(clips=sum(r['corpus']==g for r in part),frames=sum(r['frames'] for r in part if r['corpus']==g)) for g in ('lafan','bones')}))
        print('SHARD_READY',i,index[-1]['frames'],index[-1]['clips'],flush=True)
        del bounded,pack,lib;torch.cuda.empty_cache()
    assert sorted(r['name'] for part in shards for r in part)==sorted(r['name'] for r in rows)
    manifest['complete']=True;atomic(out/'manifest.json',manifest)
    atomic(out/'features/index.json',dict(records=index,frame_capacity=a.capacity,dataset_manifest_sha256=sha(out/'manifest.json'),
        robot_xml_sha256=sha(robot.mjcf_path),model_contract_sha256=sha(ASSETS/'model.json'),
        motion_code_sha256=sha(blinkmimic.motion.__file__),kinematics_code_sha256=sha(blinkmimic.kinematics.__file__)))
    # Derive a sampler-sized resume; all model and optimizer tensors still come from the immutable original.
    old=torch.load(dataset/'resume_original_010819.pt',map_location='cpu',weights_only=False)
    ck=dict(old);ck['ctrl']=dict(old['ctrl'])
    pack=torch.load(out/index[0]['pack'],map_location='cpu',weights_only=True)
    capacity=a.capacity//25+1;prior=torch.zeros(capacity);prior[:pack['num_bins']]=pack['tensors']['bin_prior']
    ck['ctrl']['env_bins']=dict(bin_G=torch.zeros_like(prior),bin_fail=torch.zeros_like(prior),bin_w_avg=prior)
    ck['ctrl'].update(ep_fail=None,focus_q=None,focus_share=0.)
    ck['reference_rotation']=dict(dataset_sha256=sha(out/'manifest.json'),shard=0,cycles=0,interval_seconds=150,
        generation=1,seconds_until_switch=150,difficulty_memory={})
    target=out/'resume_bounded_010819.pt';torch.save(ck,target)
    readback=torch.load(target,map_location='cpu',weights_only=False)
    def equal(x,y):
        if torch.is_tensor(x):return torch.equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(a,b) for a,b in zip(x,y))
        return x==y
    for key in ('wm','pi','pi_live','critic','opt_wm','opt_pi','opt_v','pi_avg'):assert equal(old[key],readback[key]),key
    receipt=dict(iteration=old['iteration'],wall_time=old['wall_time'],adapted_sha256=sha(target),
        original_sha256=sha(dataset/'resume_original_010819.pt'),learning_states_exactly_verified=True,
        dataset_release_sha256=sha(out/'manifest.json'),source_manifest_sha256=sha(dataset/'manifest.json'),
        sampler_capacity=capacity,resident_frame_capacity=a.capacity,shards=count,training_code_changed=False)
    atomic(out/'resume_transition.json',receipt)
    print('BOUNDED_RELEASE_READY',json.dumps(receipt),flush=True)


if __name__=='__main__':main()
