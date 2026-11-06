"""Read accepted Dex-Retarget exports without importing or running a solver."""
import json
from pathlib import Path

import numpy as np

from dex_mimic.common import sha
from dex_mimic.paths import ASSETS

RETARGET_URL = 'https://github.com/luckylhf/Dex-Retarget'
CONVERTER = 'dex_frax'
EVIDENCE_FILES = ('run.json', 'identity.json', 'motion.json', 'quality.json')


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_evidence(evidence):
    run, identity, meta, quality = (evidence[n] for n in EVIDENCE_FILES)
    model = read_json(ASSETS / 'model.json')
    if sha(ASSETS / 'mjcf/dex_29.xml') != model['model_sha256']:
        raise ValueError('Training model XML differs from its model manifest')
    if (run.get('status') != 'complete' or run.get('quality') != 'accepted'
            or quality.get('status') != 'accepted' or run.get('flags') != []
            or quality.get('flags') != []):
        raise ValueError('Only completed, accepted Dex-Retarget results with no flags may be imported')
    if identity.get('converter') != CONVERTER:
        raise ValueError('Convert human motion with ' + RETARGET_URL)
    fingerprint = identity.get('fingerprint')
    if not fingerprint or fingerprint != run.get('fingerprint'):
        raise ValueError('Dex-Retarget conversion fingerprint mismatch')
    if (identity.get('assets', {}).get('mjcf/dex_29.xml') != model['model_sha256']
            or meta.get('model_sha256') != model['model_sha256']
            or meta.get('joint_names') != model['joint_names']):
        raise ValueError('Dex-Retarget model or joint order differs from the training model')
    if (run.get('fps') != 50. or meta.get('output_fps') != 50.
            or identity.get('fps') != 50. or meta.get('quat_order') != 'xyzw'):
        raise ValueError('Expected Dex-Retarget 50 Hz output with explicit XYZW motion convention')
    if (run.get('frames') != quality.get('frames')
            or run.get('frames') != meta.get('output_frames')
            or not isinstance(run.get('frames'), int) or run['frames'] < 33):
        raise ValueError('Inconsistent frame counts or fewer than 33 training frames')
    if (identity.get('mode') != run.get('mode')
            or identity.get('input_sha256') != meta.get('input_sha256')):
        raise ValueError('Dex-Retarget input or solver mode identity mismatch')
    return model


def validate_training(path, model, frames, fingerprint):
    with np.load(path, allow_pickle=False) as z:
        if (str(z['converter']) != CONVERTER or str(z['converter_fingerprint']) != fingerprint
                or str(z['model_sha256']) != model['model_sha256']
                or z['joint_names'].tolist() != model['joint_names']
                or float(z['fps']) != 50. or str(z['quat_order']) != 'wxyz'):
            raise ValueError('Training NPZ converter/model/joints/frame convention mismatch')
        root = z['body_names'].tolist().index('pelvis')
        p, q, j = z['body_pos_w'][:, root], z['body_quat_w'][:, root], z['joint_pos']
        if (p.shape != (frames, 3) or q.shape != (frames, 4) or j.shape != (frames, 29)
                or not all(np.isfinite(x).all() for x in (p, q, j))
                or not np.allclose(np.linalg.norm(q, axis=1), 1., atol=1e-3, rtol=0)):
            raise ValueError('Invalid training NPZ shapes, values or quaternion norms')
        return p.copy(), q.copy(), j.copy()


def read_result(folder):
    folder = Path(folder).resolve()
    evidence = {n: read_json(folder / n) for n in EVIDENCE_FILES}
    model = validate_evidence(evidence)
    run = evidence['run.json']
    names = ('identity.json', 'motion.json', 'quality.json', 'motion.npz', 'training.npz')
    for name in names:
        if sha(folder / name) != run.get('files', {}).get(name):
            raise ValueError('Dex-Retarget output hash mismatch: ' + name)
    if run.get('training_npz') != 'training.npz':
        raise ValueError('Missing accepted training export')
    p, q, j = validate_training(folder / 'training.npz', model, run['frames'], run['fingerprint'])
    with np.load(folder / 'motion.npz', allow_pickle=False) as motion:
        if (str(motion['quat_order']) != 'xyzw' or float(motion['fps']) != 50.
                or motion['joint_names'].tolist() != model['joint_names']
                or not np.array_equal(p, motion['root_pos'])
                or not np.array_equal(j, motion['dof_pos'])
                or not np.array_equal(q, motion['root_rot'][:, [3, 0, 1, 2]])):
            raise ValueError('Training NPZ differs from Dex-Retarget motion arrays')
    # Catch a producer changing files during the read, including its completion receipt.
    hashes = {n: sha(folder / n) for n in EVIDENCE_FILES}
    if any(read_json(folder / n) != evidence[n] for n in EVIDENCE_FILES):
        raise ValueError('Dex-Retarget evidence changed while importing')
    for name in names:
        if sha(folder / name) != run['files'][name]:
            raise ValueError('Dex-Retarget output changed while importing: ' + name)
    return evidence, hashes


def validate_record(record):
    if record.get('converter') != CONVERTER or record.get('retarget_repository') != RETARGET_URL:
        raise ValueError('Expected an imported Dex-Retarget record')
    evidence = {}
    for name, item in record['evidence'].items():
        if name not in EVIDENCE_FILES or sha(item['path']) != item['sha256']:
            raise ValueError('Imported evidence changed: ' + name)
        evidence[name] = read_json(item['path'])
    if set(evidence) != set(EVIDENCE_FILES):
        raise ValueError('Incomplete imported Dex-Retarget evidence')
    model = validate_evidence(evidence)
    run = evidence['run.json']
    for name in EVIDENCE_FILES[1:]:
        if record['evidence'][name]['sha256'] != run['files'].get(name):
            raise ValueError('Imported evidence disagrees with the producer receipt: ' + name)
    if (sha(record['output']) != record['output_sha256']
            or record['output_sha256'] != run['files'].get('training.npz')
            or record['frames'] != run['frames']
            or record['converter_fingerprint'] != run['fingerprint']):
        raise ValueError('Imported training data disagrees with Dex-Retarget receipt')
    if record.get('grade') != 'A_kinematic_candidate' or record.get('quality') != evidence['quality.json']:
        raise ValueError('Imported quality must preserve Dex-Retarget acceptance')
    validate_training(record['output'], model, run['frames'], run['fingerprint'])
