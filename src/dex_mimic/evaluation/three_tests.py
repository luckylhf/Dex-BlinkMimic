"""Frozen SuperMimic 12341 input contract for the three named development tests."""
import json
from pathlib import Path

from dex_mimic.common import sha
from dex_mimic.paths import ROOT

PREPROCESSING = 'supermimic_12341_three_tests_v1'
RELEASE = Path('data/evaluation/supermimic_12341_v1')
REFERENCE_HASHES = {
    'gbtc': 'dae551fb1746a878035623ad2a580d2013e01884dae15c136085009ef590e229',
    'dance_boy': '70a03385f05be0b4c2d4d6ff9ea8dd0450a19b701729f4b8a48ee45a43a9247f',
    'dance_girl': 'cade6a483f4f999e1893ad186561dab8a7e5aa5a20afd53955cd9e033d25e98f',
}


def default_suite(project=ROOT):
    return Path(project) / RELEASE / 'suite.json'


def validate_references(references, allow_legacy=False):
    """Gate the named tests, including copies embedded in larger mixed suites."""
    found = False
    for name, path in references.items():
        if name not in REFERENCE_HASHES:
            continue
        found = True
        if sha(path) != REFERENCE_HASHES[name]:
            if not allow_legacy:
                raise ValueError(
                    f'{name}: nonstandard three-test reference. Use {default_suite()} '
                    'and a new evaluation history. Historical reproduction only: '
                    '--allow-legacy-three-tests.')
            return 'legacy_or_custom_three_tests'
    return PREPROCESSING if found else None


def load_release(project=ROOT):
    root = Path(project) / RELEASE
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['reference_preprocessing'] != PREPROCESSING:
        raise ValueError('Unexpected three-test preprocessing version')
    if {r['name'] for r in manifest['motions']} != set(REFERENCE_HASHES):
        raise ValueError('Incomplete three-test release')
    for row in manifest['motions']:
        if row['reference_sha256'] != REFERENCE_HASHES[row['name']]:
            raise ValueError('Unexpected frozen reference identity')
        for kind in ('motion', 'reference'):
            if sha(root / row[kind]) != row[kind + '_sha256']:
                raise ValueError('Frozen three-test asset hash mismatch: ' + row[kind])
    return root, manifest
