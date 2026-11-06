"""Materialize the exact old SuperMimic inputs for the three fixed test PKLs."""
import argparse
import json
from pathlib import Path
import shutil

from dex_mimic.common import sha
from dex_mimic.paths import ROOT
from dex_mimic.evaluation.three_tests import PREPROCESSING, load_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='new study directory; existing references/results are refused')
    parser.add_argument('--inputs', type=Path, help='PKL directory; default: root/inputs or project/test/videos')
    args = parser.parse_args()
    root = args.root.resolve()
    release, manifest = load_release()
    inputs = args.inputs or (root / 'inputs' if (root / 'inputs').is_dir() else ROOT / 'test/videos')
    if any((root / x).exists() for x in ('motions', 'references', 'suite.json', 'history', 'input_receipt.json')):
        raise FileExistsError('Use a new study directory; historical references and results are immutable')
    # Changed PKLs need a new protocol version; never substitute another input's references.
    for row in manifest['motions']:
        if sha(inputs / row['input_file']) != row['input_pkl_sha256']:
            raise ValueError('Input PKL changed: ' + row['input_file'] + '; do not substitute frozen references')
    root.mkdir(parents=True, exist_ok=True)
    for folder in ('inputs', 'motions', 'references'):
        (root / folder).mkdir(exist_ok=True)
    rows = []
    for original in manifest['motions']:
        row = dict(original)
        source = inputs / row['input_file']
        target = root / 'inputs' / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        for kind in ('motion', 'reference'):
            target_asset = root / row[kind]
            shutil.copy2(release / row[kind], target_asset)
            if sha(target_asset) != row[kind + '_sha256']:
                raise ValueError('Copied reference asset differs')
            row[kind] = str(target_asset)
        row.update(source=str(target), source_sha256=row['input_pkl_sha256'],
                   adaptation='exact frozen legacy preprocessing; PKL identity checked; no retargeting')
        rows.append(row)
    receipt = dict(reference_preprocessing=PREPROCESSING, evaluation_profile='jump_tracking_v2',
                   frozen_release=str(release), frozen_manifest_sha256=sha(release / 'manifest.json'),
                   source_model_sha256=manifest['model_sha256'], motions=rows,
                   evaluation_transport='native CPU MuJoCo + ONNX CPU', historical_best_evaluated=False)
    suite = dict(reference_preprocessing=PREPROCESSING, evaluation_profile='jump_tracking_v2',
                 references={r['name']: dict(reference=r['reference'], reference_sha256=r['reference_sha256'],
                                             training_member=False) for r in rows})
    (root / 'input_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    (root / 'suite.json').write_text(json.dumps(suite, indent=2)+'\n')
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
