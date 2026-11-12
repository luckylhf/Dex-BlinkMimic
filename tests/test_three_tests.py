"""Prevent mixed inputs and accidental replacement of historical evaluations."""
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from dex_mimic.common import sha
from dex_mimic.evaluation import three_tests

from dex_mimic.evaluation.three_tests import (
    PREPROCESSING, default_suite, load_release, validate_references,
)
from dex_mimic.paths import ROOT


class ThreeTestProtocolTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.project = Path(directory.name)
        self.release = self.project / three_tests.RELEASE
        self.release.mkdir(parents=True)
        self.original_hashes = three_tests.REFERENCE_HASHES.copy()
        self.original_manifest = json.loads((ROOT / three_tests.RELEASE / 'manifest.json').read_text())
        manifest = json.loads(json.dumps(self.original_manifest))
        suite = json.loads(default_suite().read_text())
        hashes = {}
        for row in manifest['motions']:
            for kind in ('motion', 'reference'):
                path = self.release / row[kind]
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(('synthetic fixture ' + kind + ' ' + row['name']).encode())
                row[kind + '_sha256'] = sha(path)
            hashes[row['name']] = row['reference_sha256']
            suite['references'][row['name']]['reference_sha256'] = row['reference_sha256']
        (self.release / 'manifest.json').write_text(json.dumps(manifest))
        (self.release / 'suite.json').write_text(json.dumps(suite))
        hash_patch = patch.object(three_tests, 'REFERENCE_HASHES', hashes)
        hash_patch.start()
        self.addCleanup(hash_patch.stop)
        release_patch = patch.object(three_tests, 'load_release', lambda: load_release(self.project))
        release_patch.start()
        self.addCleanup(release_patch.stop)

    def test_frozen_assets_and_per_motion_policy(self):
        for row in self.original_manifest['motions']:
            self.assertEqual(row['reference_sha256'], self.original_hashes[row['name']])
        root, manifest = load_release(self.project)
        suite = json.loads((root / 'suite.json').read_text())
        refs = {name: root / row['reference'] for name, row in suite['references'].items()}
        self.assertEqual(validate_references(refs), PREPROCESSING)
        rows = {r['name']: r for r in manifest['motions']}
        self.assertEqual([rows[n]['head_skip_frames_50hz'] for n in refs], [27, 5, 0])
        self.assertEqual([rows[n]['frames'] for n in refs], [947, 4011, 12079])
        self.assertEqual(rows['dance_girl']['standing_transition_seconds'], 0)

    def test_missing_local_assets_explain_required_preparation(self):
        (self.release / 'references/gbtc.npz').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing local evaluation assets'):
            load_release(self.project)

    def test_corrupt_local_reference_is_rejected(self):
        (self.release / 'references/gbtc.npz').write_bytes(b'changed fixture')
        with self.assertRaisesRegex(ValueError, 'asset hash mismatch'):
            load_release(self.project)

    def test_evaluator_checks_local_reference_before_creating_outputs(self):
        from dex_mimic.evaluation.evaluate import evaluate
        (self.release / 'references/gbtc.npz').unlink()
        output = self.project / 'new_evaluation'
        with self.assertRaisesRegex(FileNotFoundError, 'Missing local evaluation reference'):
            evaluate(SimpleNamespace(suite=str(self.release / 'suite.json'), out=str(output)))
        self.assertFalse(output.exists())

    def test_nonstandard_reference_requires_explicit_historical_mode(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / 'raw.npz'
            raw.write_bytes(b'old raw reference')
            with self.assertRaisesRegex(ValueError, 'nonstandard three-test reference'):
                validate_references({'gbtc': raw})
            self.assertEqual(validate_references({'gbtc': raw}, allow_legacy=True),
                             'legacy_or_custom_three_tests')
        self.assertIsNone(validate_references({}))

    def test_materializer_refuses_to_overwrite_a_study(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'references').mkdir()
            marker = root / 'references/keep'
            marker.write_text('preserve')
            with patch.object(sys, 'argv', ['prepare', str(root)]):
                with self.assertRaises(FileExistsError):
                    runpy.run_path(str(ROOT / 'deployment/prepare_three_test_pkls.py'), run_name='__main__')
            self.assertEqual(marker.read_text(), 'preserve')

    def test_changed_pkl_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = Path(d) / 'inputs'
            inputs.mkdir()
            (inputs / 'gbtc.pkl').write_bytes(b'changed input')
            out = Path(d) / 'new_study'
            with patch.object(sys, 'argv', ['prepare', str(out), '--inputs', str(inputs)]):
                with self.assertRaisesRegex(ValueError, 'Input PKL changed'):
                    runpy.run_path(str(ROOT / 'deployment/prepare_three_test_pkls.py'), run_name='__main__')
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
