"""Freeze accepted, non-mirrored, deduplicated BONES and the authorized 77 LAFAN clips.

Source directories are read only. The release owns copies, so source deletion does
not change a running experiment. Completed quality decisions are never upgraded.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from blinkmimic.dex.dex_constants import MANIFEST, load_dex_npz
from dex_mimic.common import atomic, sha


def content_hash(p, quat, joints):
    # q and -q represent the same orientation. Keep all translations and timing.
    quat = quat.copy()
    dominant = np.take_along_axis(quat, np.abs(quat).argmax(-1)[:, None], -1)
    quat *= np.where(dominant < 0, -1, 1)
    h = hashlib.sha256()
    for arr in (p, quat, joints):
        x = np.ascontiguousarray(arr, dtype='<f4')
        h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', required=True)
    ap.add_argument('--lafan', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    motions = out/'motions'; motions.mkdir()
    snapshots = out/'catalog_snapshots'; snapshots.mkdir()
    candidates, excluded, rows = [], [], []
    for catalog in sorted(Path(a.batches).glob('Retarget_*/accepted.json')):
        snapshot = snapshots/(catalog.parent.name+'.json')
        shutil.copy2(catalog, snapshot)
        for entry in json.loads(snapshot.read_text())['motions']:
            if Path(entry['input']).stem.endswith('_M'):
                excluded.append(dict(batch=catalog.parent.name, case=entry['case'], reason='artificial_mirror_source'))
                continue
            candidates.append((catalog.parent, entry))
    # Prefer newer completed converter versions for the same source, deterministically.
    candidates.sort(key=lambda x: (x[0].name, x[1]['case']), reverse=True)
    seen_source, seen_file, seen_content = {}, {}, {}
    for i, (batch, e) in enumerate(candidates):
        source_id = e['input_sha256']; src = batch/e['training_npz']
        reason = None; duplicate_of = None
        if source_id in seen_source:
            reason, duplicate_of = 'duplicate_source_sha256', seen_source[source_id]
        elif e['training_sha256'] in seen_file:
            reason, duplicate_of = 'duplicate_training_sha256', seen_file[e['training_sha256']]
        if reason:
            excluded.append(dict(batch=batch.name, case=e['case'], reason=reason, duplicate_of=duplicate_of)); continue
        folder = batch/e['output']
        try:
            run = json.loads((folder/'run.json').read_text())
            meta = json.loads((folder/'motion.json').read_text())
            quality = json.loads((folder/'quality.json').read_text())
            assert run['status'] == 'complete' and run['quality'] == quality['status'] == 'accepted'
            assert not quality['flags'] and meta['input_sha256'] == source_id
            assert meta['model_sha256'] == MANIFEST['model_sha256']
            assert sha(src) == e['training_sha256']
            p, q, j, fps = load_dex_npz(src)
            ch = content_hash(p, q, j)
        except (AssertionError, OSError, ValueError, KeyError) as error:
            excluded.append(dict(batch=batch.name, case=e['case'], reason='validation_failed', error=repr(error))); continue
        if ch in seen_content:
            excluded.append(dict(batch=batch.name, case=e['case'], reason='duplicate_motion_arrays', duplicate_of=seen_content[ch])); continue
        name = e['case']; dst = motions/(name+'.npz')
        shutil.copy2(src, dst); assert sha(dst) == e['training_sha256']
        row = dict(name=name, corpus='bones', training_file=dst.name, training_sha256=e['training_sha256'],
                   source=str(src), source_bvh=e['input'], source_bvh_sha256=source_id,
                   motion_content_sha256=ch, frames=len(j), bins=(len(j)-2)//50+1, fps=fps,
                   original_quality_status='accepted', original_quality_flags=[],
                   converter_release=meta.get('converter_release'), artificial_mirror=False)
        rows.append(row)
        seen_source[source_id] = seen_file[e['training_sha256']] = seen_content[ch] = name
        if len(rows)%500 == 0:
            print('FROZEN_BONES',len(rows),'candidate',i+1,'/',len(candidates),flush=True)
    lafan = Path(a.lafan)
    legacy=Path(__file__).resolve().parents[1]/'data/mixed/lafan77_bones080_20261106/manifest.json'
    labels={r['name']:r for r in json.loads(legacy.read_text())['motions'] if r['corpus']=='lafan'}
    for src in sorted(lafan.glob('*.npz')):
        p,q,j,fps = load_dex_npz(src); dst = motions/src.name
        assert not dst.exists(); shutil.copy2(src,dst)
        # Preserve the original exploratory quality record from the prior frozen manifest.
        rows.append(dict(name=src.stem,corpus='lafan',training_file=dst.name,training_sha256=sha(dst),
                         source=str(src),frames=len(j),bins=(len(j)-2)//50+1,fps=fps,
                         original_quality_status=labels[src.stem]['original_quality_status'],
                         original_quality_flags=labels[src.stem]['original_quality_flags']))
    assert sum(r['corpus']=='lafan' for r in rows)==77
    rows.sort(key=lambda x:x['name'])
    assert len({r['name'] for r in rows})==len(rows)
    report=dict(schema=1,complete=True,frozen_at=datetime.now().astimezone().isoformat(),
                model_sha256=MANIFEST['model_sha256'],training_from_scratch=True,
                offline_mirror_sources=False,wm_online_mirror_share=0.5,
                sampling='upstream uniform time-bin prior plus adaptive returns/failures; no corpus balancing',
                clips=len(rows),frames=sum(r['frames'] for r in rows),bins=sum(r['bins'] for r in rows),
                corpora={c:dict(clips=sum(r['corpus']==c for r in rows),frames=sum(r['frames'] for r in rows if r['corpus']==c)) for c in ('lafan','bones')},
                exclusion_counts=dict(Counter(r['reason'] for r in excluded)),
                snapshots={p.name:sha(p) for p in snapshots.iterdir()},motions=rows)
    atomic(out/'exclusions.json',dict(records=excluded))
    atomic(out/'manifest.json',report)
    print('DATASET_READY',json.dumps({k:v for k,v in report.items() if k not in ['motions','snapshots']},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
