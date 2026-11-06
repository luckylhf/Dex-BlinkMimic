"""External conversion boundary: acceptance, data parity and frozen evidence."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from dex_mimic.common import sha
from dex_mimic.data.prepare import import_dataset, load_records, training_rows
from dex_mimic.data.retarget_import import read_result
from dex_mimic.paths import ASSETS


class RetargetImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'retarget' / 'frax'
        self.source.mkdir(parents=True)
        self.model = json.loads((ASSETS / 'model.json').read_text())
        self.write('identity.json', dict(converter='dex_frax', version='0.8.1', fingerprint='fixture',
                   assets={'mjcf/dex_29.xml': self.model['model_sha256']}, mode='frax', fps=50., input_sha256='input'))
        self.write('motion.json', dict(model_sha256=self.model['model_sha256'], joint_names=self.model['joint_names'],
                   output_fps=50., output_frames=40, quat_order='xyzw', input_sha256='input'))
        self.write('quality.json', dict(status='accepted', flags=[], frames=40))
        p = np.arange(120, dtype=np.float32).reshape(40, 3) / 1000
        q = np.tile(np.array([0., 0., 0., 1.], dtype=np.float32), (40, 1))
        j = np.zeros((40, 29), dtype=np.float32)
        np.savez(self.source / 'motion.npz', root_pos=p, root_rot=q, dof_pos=j,
                 joint_names=self.model['joint_names'], quat_order='xyzw', fps=50.)
        np.savez(self.source / 'training.npz', body_pos_w=p[:, None], body_quat_w=q[:, None, [3, 0, 1, 2]],
                 joint_pos=j, body_names=['pelvis'], joint_names=self.model['joint_names'],
                 quat_order='wxyz', fps=50., model_sha256=self.model['model_sha256'],
                 converter='dex_frax', converter_fingerprint='fixture')
        self.run = dict(status='complete', quality='accepted', flags=[], fps=50., frames=40,
                        mode='frax', fingerprint='fixture', training_npz='training.npz')
        self.receipt()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, value):
        (self.source / name).write_text(json.dumps(value))

    def receipt(self):
        self.run['files'] = {p.name: sha(p) for p in self.source.iterdir() if p.name != 'run.json'}
        self.write('run.json', self.run)

    def import_motion(self, **metadata):
        row = dict(id='walk', source='retarget/frax', actor='subject1', sampling_group='locomotion', **metadata)
        manifest = self.root / 'sources.jsonl'
        manifest.write_text(json.dumps(row) + '\n')
        with contextlib.redirect_stdout(io.StringIO()):
            import_dataset(argparse.Namespace(manifest=manifest, out=self.root / 'imported'))
        return load_records(self.root / 'imported' / 'records')

    def test_import_preserves_bytes_and_build_validation_needs_no_converter_source(self):
        expected = sha(self.source / 'training.npz')
        rows = self.import_motion()
        shutil.rmtree(self.root / 'retarget')
        selected = training_rows(rows)
        self.assertEqual(sha(selected[0]['output']), expected)
        self.assertEqual(selected[0]['quality'], dict(status='accepted', flags=[], frames=40))

    def test_review_and_unfinished_outputs_are_rejected(self):
        for status, quality, flags in [('complete', 'review', ['arm_pose']), ('running', 'accepted', [])]:
            with self.subTest(status=status, quality=quality):
                self.run.update(status=status, quality=quality, flags=flags)
                self.write('quality.json', dict(status=quality, flags=flags, frames=40))
                self.receipt()
                with self.assertRaisesRegex(ValueError, 'completed, accepted'):
                    read_result(self.source)

    def test_corrupt_export_hash_is_rejected(self):
        with (self.source / 'training.npz').open('ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            read_result(self.source)

    def test_motion_array_mismatch_is_rejected_even_with_matching_file_hashes(self):
        path = self.source / 'motion.npz'
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['root_pos'][0, 0] += .01
        np.savez(path, **arrays)
        self.receipt()
        with self.assertRaisesRegex(ValueError, 'differs from Dex-Retarget motion'):
            read_result(self.source)

    def test_wrong_model_or_joint_order_is_rejected(self):
        meta = json.loads((self.source / 'motion.json').read_text())
        meta['joint_names'] = list(reversed(meta['joint_names']))
        self.write('motion.json', meta)
        self.receipt()
        with self.assertRaisesRegex(ValueError, 'model or joint order'):
            read_result(self.source)

    def test_evaluation_input_cannot_enter_training_release(self):
        rows = self.import_motion(evaluation_only=True)
        with self.assertRaisesRegex(ValueError, 'evaluation motion'):
            training_rows(rows)

    def test_mutated_imported_evidence_or_training_is_rejected(self):
        rows = self.import_motion()
        evidence = Path(rows[0]['evidence']['quality.json']['path'])
        original = evidence.read_bytes()
        evidence.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            training_rows(rows)
        evidence.write_bytes(original)
        with Path(rows[0]['output']).open('ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'training data disagrees'):
            training_rows(rows)

    def test_incomplete_import_catalog_cannot_build(self):
        self.import_motion()
        catalog = self.root / 'imported' / 'catalog.json'
        value = json.loads(catalog.read_text())
        value['status'] = 'importing'
        catalog.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'completed Dex-Retarget import'):
            load_records(catalog.parent / 'records')


if __name__ == '__main__':
    unittest.main()
