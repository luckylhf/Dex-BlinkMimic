"""Freeze only completed accepted BONES outputs, retain all LAFAN, and adapt sampler state."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from blinkmimic.dex.dex_constants import MANIFEST, NAMES, load_dex_npz
from dex_mimic.common import atomic, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    root=Path.cwd();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    motions=out/'motions';motions.mkdir();evidence=out/'evidence';evidence.mkdir()
    batch=Path('/home/longhongfeng/Desktop/Data2/BONES-SEED/02_Dex转换数据/Retarget_080_full_20261106')
    raw=(batch/'accepted.json').read_bytes();(out/'accepted_snapshot.json').write_bytes(raw)
    accepted=json.loads(raw)['motions'];rows=[];excluded=[]
    for i,entry in enumerate(accepted):
        try:
            folder=batch/entry['output'];run=json.loads((folder/'run.json').read_text())
            meta=json.loads((folder/'motion.json').read_text());quality=json.loads((folder/'quality.json').read_text())
            assert run['status']=='complete' and run['quality']==quality['status']=='accepted'
            assert not quality['flags'] and meta['model_sha256']==MANIFEST['model_sha256']
            assert tuple(meta['joint_names'])==NAMES and meta['input_sha256']==entry['input_sha256']
            for name in ('training.npz','motion.npz','motion.json','quality.json','identity.json'):
                assert sha(folder/name)==run['files'][name],name
            source=batch/entry['training_npz'];assert sha(source)==entry['training_sha256']
            pos,quat,joints,fps=load_dex_npz(source)
            with np.load(folder/'motion.npz',allow_pickle=False) as z:
                assert np.array_equal(pos,z['root_pos']) and np.array_equal(joints,z['dof_pos'])
                assert np.array_equal(quat,z['root_rot'][:,[3,0,1,2]])
            assert len(joints)==run['frames']==quality['frames']==meta['output_frames']
            assert abs(np.linalg.norm(quat,axis=1)-1).max()<1e-3
            name=entry['case'];dst=motions/(name+'.npz');assert not dst.exists()
            shutil.copy2(source,dst);assert sha(dst)==entry['training_sha256']
            ev=evidence/name;ev.mkdir()
            for f in ('run.json','motion.json','quality.json','identity.json'):shutil.copy2(folder/f,ev/f)
            speed=np.linalg.norm(np.diff(pos[:,:2],axis=0)*50,axis=1)
            rows.append(dict(name=name,corpus='bones',training_file=dst.name,training_sha256=sha(dst),
                source=str(source),source_bvh=entry['input'],source_bvh_sha256=entry['input_sha256'],
                frames=len(joints),bins=math.ceil((len(joints)-1)/50),fps=fps,original_quality_status='accepted',
                original_quality_flags=[],mode=entry['mode'],fingerprint=entry['fingerprint'],arrays_unchanged=True,
                horizontal_speed_p95_m_s=float(np.quantile(speed,.95))))
        except Exception as error:
            excluded.append(dict(case=entry['case'],error=repr(error)))
        if (i+1)%250==0:print('BONES_VALIDATED',i+1,'/',len(accepted),'included',len(rows),flush=True)
    assert rows,'No accepted compatible BONES motions'
    bones_count=len(rows)
    lf=json.loads((root/'deployment/all77_data_receipt.json').read_text())
    for entry in lf['motions']:
        source=root/'data/lafan1/motions_all77_exploratory'/entry['training_file']
        assert sha(source)==entry['training_sha256']
        pos,quat,joints,fps=load_dex_npz(source);dst=motions/source.name;assert not dst.exists()
        shutil.copy2(source,dst);assert sha(dst)==entry['training_sha256']
        rows.append(dict(name=dst.stem,corpus='lafan',training_file=dst.name,training_sha256=sha(dst),
            source=str(source),frames=len(joints),bins=math.ceil((len(joints)-1)/50),fps=fps,
            original_quality_status=entry['original_quality_status'],original_quality_flags=entry['original_quality_flags'],arrays_unchanged=True))
    rows.sort(key=lambda r:r['name']);assert len({r['name'] for r in rows})==len(rows)
    totals={corpus:dict(clips=sum(r['corpus']==corpus for r in rows),
        frames=sum(r['frames'] for r in rows if r['corpus']==corpus),
        bins=sum(r['bins'] for r in rows if r['corpus']==corpus)) for corpus in ('lafan','bones')}
    weights={r['name']:.5*r['bins']/totals[r['corpus']]['bins'] for r in rows}
    assert abs(sum(weights.values())-1)<1e-9
    atomic(motions/'sampling_weights.json',weights)
    frames=sum(r['frames'] for r in rows)
    estimated_peak=14313+(frames-827691)*1356/1024**2
    summary=dict(frozen_at=datetime.now().astimezone().isoformat(),source_batch=str(batch),
        accepted_index_sha256=sha(out/'accepted_snapshot.json'),accepted_index_entries=len(accepted),
        bones_included=bones_count,lafan_included=77,excluded=excluded,corpora=totals,frames=frames,
        clips=len(rows),fps=50.,model_sha256=MANIFEST['model_sha256'],reader_parity_verified=True,
        retargeting_rerun=False,source_batch_modified=False,live_growth_enabled=False,
        base_sampling_mass=dict(lafan=.5,bones=.5),adaptive_sampler_unchanged=True,
        estimated_training_peak_MiB=estimated_peak,motions=rows)
    atomic(out/'manifest.json',summary)
    assert estimated_peak<15800,'Full resident reference estimate exceeds conservative VRAM margin; use bounded shards'
    # The new dataset has different bin indices. Keep all learning state; rebuild only data-dependent sampler arrays.
    source=root/'runs/lafan_all77_continue3h_20261106_153921/last.pt'
    original=out/'resume_original_010819.pt';shutil.copy2(source,original);assert sha(source)==sha(original)
    torch.set_num_threads(2);ck=torch.load(original,map_location='cpu',weights_only=False)
    assert ck['iteration']==10819 and 'opt_pi' in ck and ck['args']['num_envs']==8192
    prior=torch.tensor([weights[r['name']]/r['bins'] for r in rows for _ in range(r['bins'])],dtype=torch.float32)
    prior/=prior.sum();old_bins=len(ck['ctrl']['env_bins']['bin_G'])
    ck['ctrl']['env_bins']=dict(bin_G=torch.zeros_like(prior),bin_fail=torch.zeros_like(prior),bin_w_avg=prior.clone())
    ck['ctrl']['ep_fail']=None;ck['ctrl']['focus_q']=None;ck['ctrl']['focus_share']=0.
    transition=dict(original_checkpoint=str(source),original_sha256=sha(original),iteration=10819,
        dataset_manifest_sha256=sha(out/'manifest.json'),old_bins=old_bins,new_bins=len(prior),
        reset_fields=['ctrl.env_bins','ctrl.ep_fail','ctrl.focus_q','ctrl.focus_share'],
        learning_state_preserved=['wm','pi','pi_live','critic','opt_wm','opt_pi','opt_v','pi_avg'],
        training_code_changed=False,wall_time=ck['wall_time'],sim_steps=ck['sim_steps'])
    ck['dataset_transition']=transition
    target=out/'resume_mixed_010819.pt';torch.save(ck,target)
    check=torch.load(target,map_location='cpu',weights_only=False);old=torch.load(original,map_location='cpu',weights_only=False)
    def equal(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    for key in transition['learning_state_preserved']:assert equal(old[key],check[key]),key
    transition.update(adapted_sha256=sha(target),learning_states_exactly_verified=True)
    atomic(out/'resume_transition.json',transition)
    print('MIX_READY',json.dumps({k:v for k,v in summary.items() if k!='motions'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
