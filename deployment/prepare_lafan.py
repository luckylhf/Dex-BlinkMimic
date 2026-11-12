"""Verify the copied conversion batch and stage only its accepted training files."""
from pathlib import Path
import collections
import hashlib
import json
import os
import shutil

root = Path(__file__).resolve().parents[1]
original = Path(os.environ.get('DEX_LAFAN_BATCH','/path/to/LAFAN1/02_Dex转换数据/Retarget_061_全量77'))
copied = root / 'data/lafan1' / original.name
out = root / 'data/lafan1/motions_accepted'
out.mkdir(exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

files = []
for p in sorted(original.rglob('*')):
    if p.is_file():
        rel = p.relative_to(original)
        digest = sha(p)
        assert sha(copied / rel) == digest, rel
        files.append({'path': str(rel), 'bytes': p.stat().st_size, 'sha256': digest})
assert sum(p.is_file() for p in copied.rglob('*')) == len(files)
accepted = json.loads((copied / 'accepted.json').read_text())['motions']
coverage = json.loads((copied / 'coverage.json').read_text())
assert coverage['full_run_complete'] and len(accepted) == 2
motions = []
for row in accepted:
    src = copied / row['training_npz']
    assert sha(src) == row['training_sha256']
    name = Path(row['input']).stem
    dst = out / (name + '.npz')
    shutil.copy2(src, dst)
    assert sha(dst) == row['training_sha256']
    motions.append(dict(row, name=name, staged=str(dst.relative_to(root))))
report = {'source': str(original), 'copy': str(copied), 'copy_file_count': len(files),
          'copy_bytes': sum(x['bytes'] for x in files), 'copy_hashes_verified': True,
          'coverage': coverage, 'training_scope': 'accepted_only', 'motions': motions,
          'review_clips_excluded': 75, 'files': files}
(root / 'deployment/data_receipt.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2, ensure_ascii=False))
