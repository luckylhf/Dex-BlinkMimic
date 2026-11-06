"""Adapt user-provided 30 Hz robot PKLs to the existing 50 Hz evaluation reader."""
from pathlib import Path
import hashlib
import json
import pickle
import sys

import numpy as np
import torch
from blinkmimic.dex.dex_constants import MANIFEST, NAMES, load_dex_npz
from blinkmimic.rotations import quat_slerp
from dex_mimic.deployment.build_bundle import prepare_motion

torch.set_num_threads(2)
root = Path(sys.argv[1]).resolve()
(root / 'motions').mkdir(exist_ok=True)
(root / 'references').mkdir(exist_ok=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

rows = []
for source_name, name in [('gbtc.pkl', 'gbtc'), ('dance_boy.pkl', 'dance_boy'), ('dance-girl.pkl', 'dance_girl')]:
    p = root / 'inputs' / source_name
    data = pickle.loads(p.read_bytes())
    assert data['model_sha256'] == MANIFEST['model_sha256']
    assert tuple(data['joint_names']) == NAMES and data['quat_order'] == 'xyzw'
    src_fps = float(data['fps'])
    pos, quat, joints = (torch.tensor(data[k], dtype=torch.float32) for k in ('root_pos', 'root_rot', 'dof_pos'))
    quat = quat[:, [3, 0, 1, 2]]
    assert all(torch.isfinite(x).all() for x in (pos, quat, joints))
    assert torch.max(torch.abs(torch.linalg.vector_norm(quat, dim=-1) - 1)) < 1e-3
    n = int(np.floor(((len(pos) - 1) / src_fps) / .02)) + 1
    t = torch.arange(n, dtype=torch.float32) * .02 * src_fps
    i = torch.clamp(t.floor().long(), max=len(pos) - 2)
    a = (t - i.float()).clamp(0, 1)
    res_p = pos[i] * (1 - a[:, None]) + pos[i + 1] * a[:, None]
    res_j = joints[i] * (1 - a[:, None]) + joints[i + 1] * a[:, None]
    res_q = quat_slerp(quat[i], quat[i + 1], a)
    dst = root / 'motions' / (name + '.npz')
    np.savez_compressed(dst, joint_pos=res_j.numpy(), joint_names=np.asarray(NAMES),
                        body_names=np.array(['pelvis']), body_pos_w=res_p[:, None].numpy(),
                        body_quat_w=res_q[:, None].numpy(), fps=np.array(50.),
                        quat_order=np.array('wxyz'), model_sha256=np.array(MANIFEST['model_sha256']),
                        original_pkl_sha256=np.array(sha(p)), original_fps=np.array(src_fps))
    loaded = load_dex_npz(dst)
    assert np.array_equal(loaded[0], res_p.numpy()) and np.array_equal(loaded[2], res_j.numpy())
    assert np.array_equal(loaded[1], res_q.numpy())
    ref = root / 'references' / (name + '.npz')
    prepare_motion(str(dst), str(ref), 10)
    rows.append(dict(name=name, source=str(p), source_sha256=sha(p), source_frames=len(pos),
                     source_fps=src_fps, frames=n, fps=50., seconds=(n-1)*.02,
                     adaptation='linear root/joints and quaternion SLERP, xyzw to wxyz; no retargeting',
                     motion=str(dst), motion_sha256=sha(dst), reference=str(ref), reference_sha256=sha(ref)))
receipt = dict(motions=rows, source_model_sha256=MANIFEST['model_sha256'],
               evaluation_profile='strict_v1', evaluation_transport='native CPU MuJoCo + ONNX CPU',
               historical_best_evaluated=False)
(root / 'input_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
suite = dict(evaluation_profile='strict_v1', references={r['name']:dict(reference=r['reference'], reference_sha256=r['reference_sha256'], training_member=False) for r in rows})
(root / 'suite.json').write_text(json.dumps(suite, indent=2) + '\n')
print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)
