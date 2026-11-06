"""Validate completed BONES clips and append immutable CPU-built mixed shards."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import blinkmimic.motion
from blinkmimic.robot import get_robot
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.dex.dex_constants import MANIFEST, NAMES, load_dex_npz
from dex_mimic.common import atomic, sha
from dex_mimic.data.bounded_motion import pack_motion, BoundedMotionLib


def read_catalog(index):
    known = {}
    for item in index['records']:
        for row in json.loads(Path(item['manifest']).read_text())['records']:
            if row['name'] in known and known[row['name']]['training_sha256'] != row['training_sha256']:
                raise ValueError('Source identity changed across immutable shards')
            known[row['name']] = row
    return known


def validate_clip(batch, entry, live):
    folder = batch / entry['output']
    run = json.loads((folder / 'run.json').read_text())
    meta = json.loads((folder / 'motion.json').read_text())
    quality = json.loads((folder / 'quality.json').read_text())
    assert run['status'] == 'complete' and run['quality'] == quality['status'] == 'accepted'
    assert not quality['flags'] and meta['model_sha256'] == MANIFEST['model_sha256']
    assert tuple(meta['joint_names']) == NAMES and meta['input_sha256'] == entry['input_sha256']
    for name in ('training.npz', 'motion.npz', 'motion.json', 'quality.json', 'identity.json'):
        assert sha(folder / name) == run['files'][name], name
    source = batch / entry['training_npz']
    assert sha(source) == entry['training_sha256']
    pos, quat, joints, fps = load_dex_npz(source)
    with np.load(folder / 'motion.npz', allow_pickle=False) as z:
        assert np.array_equal(pos, z['root_pos']) and np.array_equal(joints, z['dof_pos'])
        assert np.array_equal(quat, z['root_rot'][:, [3, 0, 1, 2]])
    assert len(joints) == run['frames'] == quality['frames'] == meta['output_frames']
    assert 33 <= len(joints) <= 400000
    assert abs(np.linalg.norm(quat, axis=1)-1).max() < 1e-3
    name = entry['case']
    assert Path(name).name == name and name not in ('.', '..')
    dst = live / 'sources' / (name+'.npz')
    if not dst.exists():
        temp = dst.with_suffix('.tmp')
        shutil.copy2(source, temp)
        assert sha(temp) == entry['training_sha256']
        temp.replace(dst)
    assert sha(dst) == entry['training_sha256']
    evidence = live / 'evidence' / name
    evidence.mkdir(exist_ok=True)
    for f in ('run.json', 'motion.json', 'quality.json', 'identity.json'):
        shutil.copy2(folder / f, evidence / f)
    return dict(name=name, corpus='bones', training_file=dst.name, output=str(dst),
                training_sha256=entry['training_sha256'], source=str(source), source_batch=str(batch),
                source_bvh=entry['input'], source_bvh_sha256=entry['input_sha256'],
                frames=len(joints), bins=math.ceil((len(joints)-1)/50), fps=fps,
                original_quality_status='accepted', original_quality_flags=[],
                mode=entry['mode'], fingerprint=entry['fingerprint'], arrays_unchanged=True)


def make_lafan_groups(rows):
    groups = [[]]
    frames = 0
    for row in sorted(rows, key=lambda r: r['name']):
        if frames + row['frames'] > 180000 and groups[-1]:
            groups.append([])
            frames = 0
        groups[-1].append(row)
        frames += row['frames']
    assert all(sum(r['frames'] for r in group) <= 200000 for group in groups)
    return groups


def append_shard(live, index, bones, anchors, robot, kin):
    revision = index['revision'] + 1
    target = live / 'shards' / f'addition_{revision:05d}_{time.time_ns()}'
    target.mkdir()
    rows = sorted(bones+anchors, key=lambda r: r['name'])
    assert len({r['name'] for r in rows}) == len(rows)
    assert sum(r['frames'] for r in rows) <= index['frame_capacity']
    paths = [r['output'] for r in rows]
    for row, path in zip(rows, paths):
        assert sha(path) == row['training_sha256']
    lib = MotionLib(paths, kin, 'cpu', robot.load_motion)
    assert lib.names == [r['name'] for r in rows]
    totals = {g: sum(r['bins'] for r in rows if r['corpus'] == g) for g in ('lafan', 'bones')}
    weights = {r['name']: .5*r['bins']/totals[r['corpus']] for r in rows}
    bins = torch.bincount(lib.bin_clip, minlength=len(rows))
    w = torch.tensor([weights[n] for n in lib.names])
    lib.bin_prior.copy_(w[lib.bin_clip]/bins[lib.bin_clip])
    lib.bin_prior.div_(lib.bin_prior.sum())
    pack = pack_motion(lib)
    pack['source_hashes'] = {r['name']: r['training_sha256'] for r in rows}
    bounded = BoundedMotionLib(pack, 'cpu', index['frame_capacity'])
    frames = torch.cat((lib.clip_start, lib.clip_end, (lib.clip_start+lib.clip_end)//2))
    assert torch.equal(lib.goal(frames, 10), bounded.goal(frames, 10))
    del bounded
    manifest = target / 'manifest.json'
    atomic(manifest, dict(records=rows, weights=weights, cpu_features=True,
                          goal_start_mid_end_parity=True, base_sampling_mass=dict(lafan=.5, bones=.5)))
    path = target / 'features.pt'
    torch.save(pack, path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)
    record = dict(pack=str(path), manifest=str(manifest), pack_sha256=sha(path),
                  manifest_sha256=sha(manifest), frames=lib.num_frames, clips=len(rows),
                  goal_boundary_midpoint_parity=True, added_bones=len(bones),
                  corpora={g: dict(clips=sum(r['corpus'] == g for r in rows),
                                   frames=sum(r['frames'] for r in rows if r['corpus'] == g)) for g in totals})
    index = dict(index, records=index['records']+[record], revision=revision, published_epoch=time.time())
    atomic(live / 'index.json', index)  # Publish only fully written, verified packs.
    print('LIVE_SHARD_PUBLISHED', revision, len(bones), lib.num_frames, flush=True)
    return index


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--live', type=Path, required=True)
    p.add_argument('--batches', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--interval', type=float, default=300)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    os.nice(10)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    live = a.live.resolve()
    for name in ('sources', 'evidence', 'shards'):
        (live / name).mkdir(exist_ok=True)
    index = json.loads((live / 'index.json').read_text())
    known = read_catalog(index)
    seed_manifest = json.loads(Path(index['seed_source_manifest']).read_text())
    anchors = []
    for row in seed_manifest['motions']:
        if row['corpus'] != 'lafan':
            continue
        anchors.append(dict(row, output=str(Path(index['seed_source_manifest']).parent/'motions'/row['training_file'])))
    groups = make_lafan_groups(anchors)
    robot = get_robot('dex')
    kin = Kinematics(robot, 'cpu')
    ignored = {}
    while time.time() < a.deadline:
        try:
            pending, rejected, counts, fingerprints = [], [], {}, {}
            pending_names = set()
            for batch in sorted(a.batches.glob('Retarget_*')):
                accepted = batch / 'accepted.json'
                if not accepted.exists():
                    continue
                raw = accepted.read_bytes()
                import hashlib
                fingerprints[batch.name] = hashlib.sha256(raw).hexdigest()
                entries = json.loads(raw)['motions']
                counts[batch.name] = len(entries)
                for entry in entries:
                    name = entry['case']
                    if name in known:
                        assert known[name]['training_sha256'] == entry['training_sha256'], 'Existing accepted identity changed: '+name
                        continue
                    token = (name, entry['training_sha256'])
                    if name in pending_names or token in ignored:
                        continue
                    try:
                        pending.append(validate_clip(batch, entry, live))
                        pending_names.add(name)
                    except Exception as error:
                        rejected.append(dict(batch=batch.name, case=name, error=repr(error)))
                        # Retry on the next scan: a converter may still be publishing metadata.
                    if time.time() >= a.deadline:
                        break
            parts, frames = [[]], 0
            for row in pending:
                if frames + row['frames'] > 400000 and parts[-1]:
                    parts.append([])
                    frames = 0
                parts[-1].append(row)
                frames += row['frames']
            for part in parts:
                if not part or time.time() >= a.deadline:
                    continue
                index = append_shard(live, index, part, groups[index['revision'] % len(groups)], robot, kin)
                known.update({r['name']: r for r in part})
            atomic(live / 'status.json', dict(state='watching' if time.time() < a.deadline else 'deadline',
                updated_epoch=time.time(), revision=index['revision'], shards=len(index['records']),
                included_bones=sum(r['corpus'] == 'bones' for r in known.values()), included_lafan=77,
                accepted_entries_by_batch=counts, accepted_snapshot_hashes=fingerprints,
                rejected_this_scan=rejected, interval_seconds=a.interval, source_data_modified=False))
        except Exception as error:
            with (live / 'errors.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(), error=repr(error)))+'\n')
            print('LIVE_SCAN_RETRY', repr(error), flush=True)
        if a.once:
            break
        time.sleep(min(a.interval, max(0, a.deadline-time.time())))
    print('LIVE_UPDATER_EXIT', flush=True)


if __name__ == '__main__':
    main()
