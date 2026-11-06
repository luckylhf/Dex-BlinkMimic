"""Prevent mixed inputs and accidental replacement of historical evaluations."""
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from dex_mimic.evaluation.three_tests import (
    PREPROCESSING, default_suite, load_release, validate_references,
)
from dex_mimic.paths import ROOT


class ThreeTestProtocolTests(unittest.TestCase):
    def test_frozen_assets_and_per_motion_policy(self):
        root, manifest = load_release()
        suite = json.loads(default_suite().read_text())
        refs = {name: root / row['reference'] for name, row in suite['references'].items()}
        self.assertEqual(validate_references(refs), PREPROCESSING)
        rows = {r['name']: r for r in manifest['motions']}
        self.assertEqual([rows[n]['head_skip_frames_50hz'] for n in refs], [27, 5, 0])
        self.assertEqual([rows[n]['frames'] for n in refs], [947, 4011, 12079])
        self.assertEqual(rows['dance_girl']['standing_transition_seconds'], 0)

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
