"""Quality-label preflight needs no historical library or GPU training stack."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dex_mimic.common import sha


path = Path(__file__).resolve().parents[1] / 'deployment/freeze_recipe_dataset.py'
spec = importlib.util.spec_from_file_location('freeze_recipe_dataset', path)
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


class FreezeRecipeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.motions = self.root / 'motions'
        self.motions.mkdir()
        self.labels = self.root / 'quality.json'
        self.rows = []
        for i in range(77):
            path = self.motions / f'clip_{i:02d}.npz'
            np.savez(path, joint_pos=np.zeros((40, 29), dtype=np.float32))
            self.rows.append(dict(name=path.stem, training_sha256=sha(path),
                                  original_quality_status='accepted' if i < 2 else 'review',
                                  original_quality_flags=[] if i < 2 else ['original_review_flag']))
        self.save()

    def save(self):
        self.labels.write_text(json.dumps(dict(motions=self.rows)))

    def test_labels_and_flags_are_preserved_and_receipt_is_hashed(self):
        original = copy.deepcopy(self.rows)
        files, labels, digest = freeze.lafan_inputs(self.motions, self.labels)
        self.assertEqual(len(files), 77)
        self.assertEqual(labels, {r['name']: r for r in original})
        self.assertEqual(digest, sha(self.labels))

    def test_missing_receipt_fails_before_output_is_created(self):
        out = self.root / 'out'
        with self.assertRaises(FileNotFoundError):
            freeze.main(['--batches', str(self.root), '--lafan', str(self.motions),
                         '--lafan-labels', str(self.root / 'missing.json'), '--out', str(out)])
        self.assertFalse(out.exists())

    def test_missing_label_is_rejected(self):
        self.rows.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, 'missing labels'):
            freeze.lafan_inputs(self.motions, self.labels)

    def test_duplicate_or_incomplete_quality_is_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.save()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            freeze.lafan_inputs(self.motions, self.labels)
        self.rows.pop()
        del self.rows[0]['original_quality_flags']
        self.save()
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            freeze.lafan_inputs(self.motions, self.labels)

    def test_changed_motion_is_rejected_before_copy(self):
        with (self.motions / 'clip_00.npz').open('ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs from its quality receipt'):
            freeze.lafan_inputs(self.motions, self.labels)

    def test_missing_motion_is_rejected(self):
        (self.motions / 'clip_00.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'exactly 77'):
            freeze.lafan_inputs(self.motions, self.labels)


if __name__ == '__main__':
    unittest.main()
