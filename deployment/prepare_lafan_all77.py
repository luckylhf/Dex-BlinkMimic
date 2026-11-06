"""Stage all 77 clips for explicitly authorized exploratory training.

Keep acceptance records unchanged. Only reshape root/joint arrays and reorder
quaternions into the training reader's format; never rerun retargeting here.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil

import numpy as np
from blinkmimic.dex.dex_constants import MANIFEST, NAMES, load_dex_npz

root = Path(__file__).resolve().parents[1]
batch = root / 'data/lafan1/Retarget_061_全量77_20261106_095755'
out = root / 'data/lafan1/motions_all77_exploratory'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

records = read(batch / 'batch.json')['records']
coverage = read(batch / 'coverage.json')
assert len(records) == 77 and coverage['full_run_complete']
assert not out.exists(), 'Use a fresh output directory to preserve prior evidence'
out.mkdir()
rows = []
names = set()
for record in records:
    complete = [a for a in record['attempts'] if a['status'] == 'complete']
    assert complete, record['case']
    selected = (next(a for a in complete if a['mode'] == record['selected_mode'])
                if record['status'] == 'accepted' else complete[-1])
    folder = batch / selected['output']
    run = read(folder / 'run.json')
    meta = read(folder / 'motion.json')
    quality = read(folder / 'quality.json')
    identity = read(folder / 'identity.json')
    assert run['status'] == 'complete'
    assert quality['status'] == record['status'] == run['quality']
    assert sorted(quality['flags']) == sorted(selected['flags']) == sorted(run['flags'])
    assert meta['input_sha256'].startswith(record['case'].split('_', 1)[1])
    assert meta['model_sha256'] == MANIFEST['model_sha256']
    assert identity['assets']['mjcf/dex_29.xml'] == MANIFEST['model_sha256']
    for filename in ('motion.npz', 'motion.json', 'quality.json', 'identity.json'):
        assert sha(folder / filename) == run['files'][filename], (folder, filename)
    name = Path(meta['input']).stem
    assert name not in names
    names.add(name)
    with np.load(folder / 'motion.npz', allow_pickle=False) as z:
        p, q, j = (z[k].copy() for k in ('root_pos', 'root_rot', 'dof_pos'))
        assert tuple(z['joint_names'].tolist()) == NAMES
        assert str(z['quat_order']) == 'xyzw' and float(z['fps']) == 50.
    assert len(j) == run['frames'] == meta['output_frames'] == quality['frames']
    assert len(j) >= 33 and p.shape == (len(j), 3) and q.shape == (len(j), 4)
    assert j.shape == (len(j), 29)
    assert all(np.isfinite(x).all() for x in (p, q, j))
    assert np.max(np.abs(np.linalg.norm(q, axis=1) - 1)) < 1e-3
    dst = out / (name + '.npz')
    if record['status'] == 'accepted':
        src = batch / record['training_npz']
        assert sha(src) == record['training_sha256']
        shutil.copy2(src, dst)
    else:
        np.savez_compressed(
            dst, joint_pos=j, joint_names=np.asarray(NAMES),
            body_names=np.array(['pelvis']), body_pos_w=p[:, None],
            body_quat_w=q[:, None, [3, 0, 1, 2]], fps=np.array(50.),
            quat_order=np.array('wxyz'), model_sha256=np.array(meta['model_sha256']),
            converter=np.array('dex_frax'), converter_fingerprint=np.array(run['fingerprint']),
            storage_format=np.array('root_joint_lossless_v1'),
            training_scope=np.array('all77_exploratory_user_authorized'),
            source_quality_status=np.array(quality['status']),
            source_quality_flags=np.asarray(quality['flags'], dtype=str))
    loaded = load_dex_npz(dst)
    assert np.array_equal(loaded[0], p)
    assert np.array_equal(loaded[1], q[:, [3, 0, 1, 2]])
    assert np.array_equal(loaded[2], j)
    evidence = out / 'evidence' / name
    evidence.mkdir(parents=True)
    for filename in ('quality.json', 'run.json', 'motion.check.json'):
        if (folder / filename).exists():
            shutil.copy2(folder / filename, evidence / filename)
    row = dict(name=name, case=record['case'], source=meta['input'],
               source_sha256=meta['input_sha256'], mode=selected['mode'],
               output=selected['output'], source_motion_sha256=sha(folder / 'motion.npz'),
               original_quality_status=quality['status'], original_quality_flags=quality['flags'],
               frames=len(j), fps=50., training_file=dst.name, training_sha256=sha(dst),
               arrays_unchanged=True, quaternion_change='xyzw_to_wxyz_only',
               all_attempts=record['attempts'])
    rows.append(row)
    print(name, len(j), selected['mode'], quality['status'], flush=True)

assert len(list(out.glob('*.npz'))) == len(rows) == 77
assert sum(r['frames'] for r in rows) == coverage['source_output_frames'] == 827691
summary = dict(scope='all77_exploratory_user_authorized', initialization='fresh',
               selection='accepted: selected output; review: last complete attempt',
               source_batch=str(batch), output=str(out), clips=len(rows),
               frames=sum(r['frames'] for r in rows), fps=50.,
               original_quality_counts=dict(Counter(r['original_quality_status'] for r in rows)),
               mode_counts=dict(Counter(r['mode'] for r in rows)),
               original_acceptance_unchanged=True, retargeting_rerun=False,
               reader_parity_verified=True, motions=rows)
(out / 'manifest.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
(root / 'deployment/all77_data_receipt.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'motions'}, indent=2, ensure_ascii=False))
