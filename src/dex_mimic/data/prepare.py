"""Import accepted Dex-Retarget outputs and build deterministic training shards."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

from dex_mimic.common import atomic, sha, stamp
from dex_mimic.paths import ASSETS
from .retarget_import import CONVERTER, RETARGET_URL, EVIDENCE_FILES, read_result, validate_record

TARGETS = {'dance': .40, 'locomotion': .25, 'stationary_upper': .25, 'dynamic': .10}


def row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def import_dataset(args):
    manifest = Path(args.manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty import manifest')
    allowed = {'id', 'source', 'actor', 'sampling_group', 'dataset', 'anchor', 'evaluation_only'}
    seen = set()
    for row in rows:
        if not {'id', 'source', 'actor', 'sampling_group'} <= row.keys() or set(row) - allowed:
            raise ValueError('Manifest requires id/source/actor/sampling_group; optional dataset/anchor/evaluation_only')
        name = row['id']
        if not isinstance(name, str) or not name.replace('_', '').replace('-', '').isalnum() or name in seen:
            raise ValueError('Duplicate or invalid motion ID')
        seen.add(name)
        if row['sampling_group'] not in TARGETS or not isinstance(row['actor'], str) or not row['actor']:
            raise ValueError('Unknown sampling_group or empty actor')
        for key in ('anchor', 'evaluation_only'):
            row.setdefault(key, False)
            if not isinstance(row[key], bool):
                raise ValueError(key + ' must be a JSON boolean')
        source = Path(row['source']).expanduser()
        row['source'] = str((manifest.parent / source).resolve())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ('records', 'motions', 'evidence'):
        (out / name).mkdir()
    catalog = dict(status='importing', converter=CONVERTER, retarget_repository=RETARGET_URL,
                   source_manifest_sha256=sha(manifest), records={}, created_at=stamp())
    atomic(out / 'catalog.json', catalog)
    for row in rows:
        evidence, hashes = read_result(row['source'])
        run = evidence['run.json']
        target = out / 'motions' / (row['id'] + '.npz')
        shutil.copyfile(Path(row['source']) / 'training.npz', target)
        receipt = {}
        folder = out / 'evidence' / row['id']
        folder.mkdir()
        for name in EVIDENCE_FILES:
            path = folder / name
            shutil.copyfile(Path(row['source']) / name, path)
            if sha(path) != hashes[name]:
                raise ValueError('Evidence changed during copy: ' + name)
            receipt[name] = dict(path=str(path), sha256=hashes[name])
        record = dict(row, source_manifest_row=dict(row), source_row_sha256=row_sha(row),
                      converter=CONVERTER, retarget_repository=RETARGET_URL,
                      converter_fingerprint=run['fingerprint'], converter_version=evidence['identity.json'].get('version'),
                      solver_mode=run['mode'], status='imported', grade='A_kinematic_candidate',
                      quality=evidence['quality.json'], frames=run['frames'], output=str(target),
                      output_sha256=sha(target), evidence=receipt, arrays_unchanged=True,
                      train_eligible=not row['evaluation_only'],
                      split='evaluation' if row['evaluation_only'] else 'train',
                      dynamic_status='not_tested_by_controller', imported_at=stamp())
        validate_record(record)
        atomic(out / 'records' / (row['id'] + '.json'), record)
        catalog['records'][row['id']] = record['source_row_sha256']
    catalog['status'] = 'complete'
    atomic(out / 'catalog.json', catalog)
    print(json.dumps(dict(imported=len(rows), output=str(out), retarget_repository=RETARGET_URL), indent=2))


def load_records(directory):
    directory = Path(directory)
    catalog = json.loads((directory.parent / 'catalog.json').read_text())
    if (catalog.get('status') != 'complete' or catalog.get('converter') != CONVERTER
            or catalog.get('retarget_repository') != RETARGET_URL):
        raise ValueError('Build requires a completed Dex-Retarget import catalog')
    result = []
    for name, fingerprint in catalog['records'].items():
        if not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Invalid catalog ID')
        record = json.loads((directory / (name + '.json')).read_text())
        row = record['source_manifest_row']
        if (record['id'] != name or row_sha(row) != fingerprint
                or record.get('source_row_sha256') != fingerprint
                or any(record.get(k) != v for k, v in row.items())):
            raise ValueError('Import record disagrees with its catalog: ' + name)
        result.append(record)
    return result


def training_rows(records):
    rows = []
    for record in records:
        validate_record(record)
        if record.get('evaluation_only'):
            raise ValueError('External evaluation motion marked as training: ' + record['id'])
        rows.append(dict(record, split='train'))
    return rows


def sampling_weights(rows):
    groups = defaultdict(list)
    for row in rows: groups[row['sampling_group']].append(row)
    total = sum(TARGETS[g] for g in groups)
    result = {}
    for category, clips in groups.items():
        old = [x for x in clips if x.get('anchor', False)]
        new = [x for x in clips if not x.get('anchor', False)]
        for subset, share in ((old, .25 if new else 1), (new, .75 if old else 1)):
            actors = defaultdict(list)
            for row in subset: actors[row['actor']].append(row)
            for actor_clips in actors.values():
                for row in actor_clips:
                    result[row['id']] = TARGETS[category]/total*share/len(actors)/len(actor_clips)
    assert abs(sum(result.values())-1) < 1e-8
    return result

def build_release(args):
    records = load_records(args.records)
    rows = training_rows(records)
    if not rows: raise ValueError('No A-grade motions')
    import torch
    from blinkmimic.robot import get_robot
    from blinkmimic.kinematics import Kinematics
    import blinkmimic.motion, blinkmimic.kinematics
    from .bounded_motion import build_pack, BoundedMotionLib
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('Duplicate motion IDs')
    for r in rows:
        if r['frames'] < 33 or r['frames'] > args.capacity: raise ValueError('Whole clip outside capacity: '+r['id'])
        if r.get('evaluation_only'): raise ValueError('External evaluation motion marked as training')
    anchors = sorted([r for r in rows if r.get('anchor')], key=lambda r:r['id'])
    others = [r for r in rows if not r.get('anchor')]
    base = sum(r['frames'] for r in anchors)
    room = args.capacity-base
    if room <= 0 or any(r['frames'] > room for r in others): raise ValueError('Increase capacity: anchors plus whole clip do not fit')
    n = max(1, math.ceil(sum(r['frames'] for r in others)/room))
    while True:
        shards = [anchors.copy() for _ in range(n)]
        sizes, groups = [base]*n, [Counter() for _ in range(n)]
        for r in sorted(others, key=lambda r:(r['sampling_group'], -r['frames'], r['id'])):
            available = [i for i in range(n) if sizes[i]+r['frames'] <= args.capacity]
            if not available: n += 1; break
            i = min(available, key=lambda i:(groups[i][r['sampling_group']], sizes[i], i))
            shards[i].append(r); sizes[i] += r['frames']; groups[i][r['sampling_group']] += r['frames']
        else: break
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'features').mkdir()
    manifest = dict(complete=False, snapshot=False, train_clips=len(rows), train_frames=sum(r['frames'] for r in rows),
                    anchors=len(anchors), shards=n, frame_capacity=args.capacity,
                    grades=dict(Counter(r.get('grade') for r in records)), training_policy='all_qualified',converter=CONVERTER,retarget_repository=RETARGET_URL,
                    created_at=stamp(), internal_probes_are_heldout=False)
    atomic(out/'manifest.json', manifest)
    robot = get_robot('dex'); kin = Kinematics(robot,args.device); index = []
    for i, shard in enumerate(shards):
        shard.sort(key=lambda r:r['id'])
        data = dict(records=shard, weights=sampling_weights(shard), capacity=args.capacity)
        path = out/f'shard_{i:03d}.json'; atomic(path,data)
        pack, lib = build_pack(data, kin, robot, args.device)
        bounded = BoundedMotionLib(pack,args.device,args.capacity)
        frames = torch.cat([lib.clip_start, lib.clip_end])
        if not torch.equal(lib.goal(frames,10), bounded.goal(frames,10)): raise AssertionError('Goal boundary parity')
        target = out/'features'/f'shard_{i:03d}.pt'
        torch.save(pack,target)
        index.append(dict(pack=str(target.relative_to(out)), manifest=path.name, pack_sha256=sha(target),
                          manifest_sha256=sha(path), frames=lib.num_frames, clips=len(lib.names), goal_boundary_parity=True))
        del pack, lib, bounded
    manifest['complete']=True; atomic(out/'manifest.json',manifest)
    atomic(out/'features/index.json',dict(records=index,frame_capacity=args.capacity,
        dataset_manifest_sha256=sha(out/'manifest.json'), robot_xml_sha256=sha(robot.mjcf_path),
        model_contract_sha256=sha(ASSETS/'model.json'), motion_code_sha256=sha(blinkmimic.motion.__file__),
        kinematics_code_sha256=sha(blinkmimic.kinematics.__file__)))
    print(json.dumps(manifest,indent=2))

def main():
    p = argparse.ArgumentParser(description=__doc__, epilog='Human motion conversion: ' + RETARGET_URL)
    s = p.add_subparsers(dest='command', required=True)
    c = s.add_parser('import', help='Copy and verify accepted Dex-Retarget training exports')
    c.add_argument('--manifest', required=True)
    c.add_argument('--out', required=True)
    c.set_defaults(fn=import_dataset)
    b = s.add_parser('build', help='Build feature shards from imported records')
    b.add_argument('--records', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--capacity', type=int, default=600000)
    b.add_argument('--device', default='cpu')
    b.add_argument('--threads', type=int, default=4)
    b.set_defaults(fn=build_release)
    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
